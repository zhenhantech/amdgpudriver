# AI Kernel Submission 关键代码参考

**文档类型**: 代码参考  
**创建时间**: 2025-01-XX  
**目的**: 提供AI kernel submission流程中关键代码的详细位置和实现细节

---

## 目录

1. [ROCm Runtime层代码](#rocm-runtime层代码)
2. [驱动层代码](#驱动层代码)
3. [MES调度器代码](#mes调度器代码)
4. [HSA Runtime代码](#hsa-runtime代码)
5. [FlashInfer代码](#flashinfer代码)

---

## ROCm Runtime层代码

### 1. Doorbell写入代码

**文件路径**: `ROCR-Runtime/.../amd_blit_kernel.cpp:1301`

**关键函数**: Kernel提交和doorbell更新

**代码位置**:
```cpp
// ROCm Runtime源码位置
// 文件: ROCR-Runtime/src/core/runtime/amd_blit_kernel.cpp
// 行号: 1301

void SubmitKernelToQueue(AQLQueue* queue, AQLPacket* packet) {
    // 1. 写入AQL packet到AQL Queue
    uint32_t wptr = queue->write_index;
    queue->packets[wptr % queue->size] = *packet;
    
    // 2. 更新wptr
    wptr++;
    queue->write_index = wptr;
    
    // 3. 写入doorbell寄存器
    *((volatile uint32_t*)queue->doorbell) = wptr;
    
    // 4. 记录日志 (LOG_LEVEL=5)
    if (log_level >= 5) {
        printf("[***rocr***] HWq=%p, id=%d, Dispatch Header = 0x%x, "
               "rptr=%d, wptr=%d\n",
               queue->base_address, queue->id, 
               packet->header, queue->read_index, wptr);
    }
}
```

**关键发现**:
- ✅ 日志标记为`[***rocr***]`（ROCm Runtime）
- ✅ 包含`HWq`（Hardware Queue地址）
- ✅ 包含`rptr`和`wptr`（读写指针）
- ✅ 包含`Dispatch Header`（AQL packet的header）

---

## 驱动层代码

### 1. Entity创建和Ring绑定

**文件路径**: `/usr/src/amdgpu-6.12.12-2194681.el8/amd/amdgpu/amdgpu_ctx.c`

**关键函数**: `amdgpu_ctx_get_entity()`

**代码实现**:
```c
int amdgpu_ctx_get_entity(struct amdgpu_ctx *ctx, u32 hw_ip, u32 instance,
                          u32 ring, struct drm_sched_entity **entity)
{
    int r;
    struct drm_sched_entity *ctx_entity;

    if (hw_ip >= AMDGPU_HW_IP_NUM) {
        DRM_ERROR("unknown HW IP type: %d\n", hw_ip);
        return -EINVAL;
    }

    /* Right now all IPs have only one instance - multiple rings. */
    if (instance != 0) {
        DRM_DEBUG("invalid ip instance: %d\n", instance);
        return -EINVAL;
    }

    if (ring >= amdgpu_ctx_num_entities[hw_ip]) {
        DRM_DEBUG("invalid ring: %d %d\n", hw_ip, ring);
        return -EINVAL;
    }

    if (ctx->entities[hw_ip][ring] == NULL) {
        r = amdgpu_ctx_init_entity(ctx, hw_ip, ring);
        if (r)
            return r;
    }

    ctx_entity = &ctx->entities[hw_ip][ring]->entity;
    *entity = ctx_entity;
    return 0;
}
```

**关键参数**:
- `hw_ip`: 硬件IP类型（`AMDGPU_HW_IP_COMPUTE`或`AMDGPU_HW_IP_DMA`）
- `ring`: Ring索引（0-3 for COMPUTE, 0-1 for DMA）
- `entity`: 返回的Entity指针

**Entity数量配置**:
```c
const unsigned int amdgpu_ctx_num_entities[AMDGPU_HW_IP_NUM] = {
    [AMDGPU_HW_IP_GFX]     = 1,
    [AMDGPU_HW_IP_COMPUTE] = 4,    // Compute Ring: 4个Entity
    [AMDGPU_HW_IP_DMA]     = 2,    // SDMA Ring: 2个Entity
    // ...
};
```

### 2. Entity初始化

**文件路径**: `/usr/src/amdgpu-6.12.12-2194681.el8/amd/amdgpu/amdgpu_ctx.c`

**关键函数**: `amdgpu_ctx_init_entity()`

**代码实现**:
```c
static int amdgpu_ctx_init_entity(struct amdgpu_ctx *ctx, u32 hw_ip,
                                  const u32 ring)
{
    struct drm_gpu_scheduler **scheds = NULL, *sched = NULL;
    struct amdgpu_device *adev = ctx->mgr->adev;
    struct amdgpu_ctx_entity *entity;
    enum drm_sched_priority drm_prio;
    unsigned int hw_prio, num_scheds;
    int32_t ctx_prio;
    int r;

    entity->hw_ip = hw_ip;  // 保存hw_ip类型
    
    hw_ip = array_index_nospec(hw_ip, AMDGPU_HW_IP_NUM);

    if (!(adev)->xcp_mgr) {
        scheds = adev->gpu_sched[hw_ip][hw_prio].sched;
        num_scheds = adev->gpu_sched[hw_ip][hw_prio].num_scheds;
    } else {
        struct amdgpu_fpriv *fpriv;
        fpriv = container_of(ctx->ctx_mgr, struct amdgpu_fpriv, ctx_mgr);
        r = amdgpu_xcp_select_scheds(adev, hw_ip, hw_prio, fpriv,
                        &num_scheds, &scheds);
        if (r)
            goto cleanup_entity;
    }

    r = drm_sched_entity_init(&entity->entity, drm_prio, scheds, num_scheds,
                  &ctx->guilty);
    // ...
}
```

**关键操作**:
- ✅ Entity的Ring类型由`hw_ip`参数决定
- ✅ `hw_ip`用于索引`adev->gpu_sched[hw_ip][hw_prio]`，获取对应的scheduler列表
- ✅ 如果有XCP管理器，通过`amdgpu_xcp_select_scheds`选择scheduler
- ✅ 最终Entity绑定到选定的scheduler（Ring）

### 3. Ring类型定义

**文件路径**: `/usr/src/amdgpu-6.12.12-2194681.el8/amd/amdgpu/amdgpu_ring.h`

**关键定义**:
```c
enum amdgpu_ring_type {
    AMDGPU_RING_TYPE_GFX      = AMDGPU_HW_IP_GFX,
    AMDGPU_RING_TYPE_COMPUTE = AMDGPU_HW_IP_COMPUTE,
    AMDGPU_RING_TYPE_SDMA    = AMDGPU_HW_IP_DMA,
    AMDGPU_RING_TYPE_UVD    = AMDGPU_HW_IP_UVD,
    AMDGPU_RING_TYPE_VCE    = AMDGPU_HW_IP_VCE,
    AMDGPU_RING_TYPE_UVD_ENC = AMDGPU_HW_IP_UVD_ENC,
    AMDGPU_RING_TYPE_VCN_DEC = AMDGPU_HW_IP_VCN_DEC,
    AMDGPU_RING_TYPE_VCN_ENC = AMDGPU_HW_IP_VCN_ENC,
    AMDGPU_RING_TYPE_VCN_JPEG = AMDGPU_HW_IP_VCN_JPEG,
    AMDGPU_RING_TYPE_VPE    = AMDGPU_HW_IP_VPE,
    AMDGPU_RING_TYPE_KIQ,
    AMDGPU_RING_TYPE_MES,
    AMDGPU_RING_TYPE_UMSCH_MM,
    AMDGPU_RING_TYPE_CPER,
};
```

---

## MES调度器代码

### 1. MES add_hw_queue实现

**文件路径**: `/mnt/md0/zhehan/code/coderampup/amdgpu/drm/amd/amdgpu/mes_v12_0.c`

**关键函数**: `mes_v12_0_add_hw_queue()`

**代码实现**:
```c
static int mes_v12_0_add_hw_queue(struct amdgpu_mes *mes,
                                  struct mes_add_queue_input *input)
{
    struct amdgpu_device *adev = mes->adev;
    union MESAPI__ADD_QUEUE mes_add_queue_pkt;
    
    mes_add_queue_pkt.queue_type =
        convert_to_mes_queue_type(input->queue_type);
    
    return mes_v12_0_submit_pkt_and_poll_completion(mes,
            AMDGPU_MES_SCHED_PIPE,
            &mes_add_queue_pkt, sizeof(mes_add_queue_pkt),
            offsetof(union MESAPI__ADD_QUEUE, api_status));
}
```

**关键操作**:
- ✅ MES通过`mes_v12_0_submit_pkt_and_poll_completion`提交packet到MES调度器
- ✅ Packet类型是`MESAPI__ADD_QUEUE`
- ✅ Queue类型通过`convert_to_mes_queue_type`转换

### 2. MES packet提交

**文件路径**: `/mnt/md0/zhehan/code/coderampup/amdgpu/drm/amd/amdgpu/mes_v12_0.c`

**关键函数**: `mes_v12_0_submit_pkt_and_poll_completion()`

**代码实现**:
```c
static int mes_v12_0_submit_pkt_and_poll_completion(struct amdgpu_mes *mes,
                                                    int pipe, void *pkt, int size,
                                                    int api_status_off)
{
    struct amdgpu_ring *ring = &mes->ring[pipe];
    // ...
    amdgpu_ring_write_multiple(ring, pkt, size / 4);
    amdgpu_ring_commit(ring);
    // ...
}
```

**关键发现**:
- ✅ MES使用`mes->ring[pipe]`提交packet
- ✅ `ring->type = AMDGPU_RING_TYPE_MES`
- ✅ MES Ring用于提交MES管理命令（如ADD_QUEUE）

### 3. MES Ring类型

**文件路径**: `/mnt/md0/zhehan/code/coderampup/amdgpu/drm/amd/amdgpu/mes_v12_0.c`

**关键定义**:
```c
static const struct amdgpu_ring_funcs mes_v12_0_ring_funcs = {
    .type = AMDGPU_RING_TYPE_MES,
    // ...
};
```

**关键发现**:
- ✅ MES Ring的类型是`AMDGPU_RING_TYPE_MES`
- ✅ 这不是Compute Ring或SDMA Ring
- ✅ MES Ring用于MES调度器的管理操作

---

## HSA Runtime代码

### 1. Queue创建

**文件路径**: `/mnt/md0/zhehan/code/rocm6.4.3/ROCR-Runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp`

**关键函数**: `GpuAgent::QueueCreate()`

**代码实现**:
```cpp
hsa_status_t GpuAgent::QueueCreate(size_t size, hsa_queue_type32_t queue_type,
                                   core::HsaEventCallback event_callback,
                                   void* data, uint32_t private_segment_size,
                                   uint32_t group_segment_size,
                                   core::Queue** queue) {
  // Handle GWS queues.
  if (queue_type == HSA_QUEUE_TYPE_COOPERATIVE) {
    // ...
  }

  // AQL queues must be a power of two in length.
  if (!IsPowerOfTwo(size)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // Enforce max size
  if (size > maxAqlSize_) {
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  // Enforce min size
  if (size < minAqlSize_) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // Allocate scratch memory
  ScratchInfo scratch = {0};
  // ...

  // Ensure utility queue has been created.
  queues_[QueueUtility].touch();

  // Create an HW AQL queue
  auto aql_queue =
      new AqlQueue(this, size, node_id(), scratch, event_callback, data, is_kv_device_);
  *queue = aql_queue;
  aql_queues_.push_back(aql_queue);
  
  return HSA_STATUS_SUCCESS;
}
```

**关键操作**:
- ✅ 创建AQL Queue（Architected Queuing Language）
- ✅ Queue大小必须是2的幂
- ✅ 分配scratch内存
- ✅ 注册到HSA Runtime

### 2. KFD接口调用

**文件路径**: `/mnt/md0/zhehan/code/rocm6.4.3/ROCR-Runtime/libhsakmt/src/queues.c`

**关键函数**: `hsaKmtCreateQueueExt()`

**代码实现**:
```c
kmt_status_t hsaKmtCreateQueueExt(uint32_t node_id, 
                                   hsa_queue_type32_t type,
                                   uint32_t queue_size,
                                   uint32_t queue_priority,
                                   uint64_t queue_properties,
                                   void* ring_base_address,
                                   uint64_t* queue_id,
                                   uint64_t* doorbell_offset) {
    struct kfd_ioctl_create_queue_args args = {0};
    
    switch (Type) {
    case HSA_QUEUE_COMPUTE_AQL:
        args.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
        break;
    // ...
    }
    
    err = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_CREATE_QUEUE, &args);
    // ...
}
```

**关键操作**:
- ✅ 将`HSA_QUEUE_COMPUTE_AQL`映射到`KFD_IOC_QUEUE_TYPE_COMPUTE_AQL`
- ✅ 调用`AMDKFD_IOC_CREATE_QUEUE` ioctl
- ✅ 返回queue_id和doorbell_offset

---

## FlashInfer代码

### 1. Python API层

**文件路径**: `flashinfer/flashinfer/prefill.py:661-857`

**关键函数**: `single_prefill_with_kv_cache()`

**关键代码**:
```python
def single_prefill_with_kv_cache(q, k, v, causal=True, kv_layout='NHD'):
    # 获取JIT模块
    module_getter = get_single_prefill_module(backend)
    # 调用JIT模块的run方法
    return module_getter(...).run(q, k, v, tmp, out, ...)
```

### 2. JIT模块层

**文件路径**: `flashinfer/flashinfer/jit/attention/pytorch.py:826-839`

**关键函数**: `get_single_prefill_module()`

**关键代码**:
```python
def get_single_prefill_module(backend):
    def backend_module(*args):
        # ...
        if args not in modules_dict:
            uri = get_single_prefill_uri(backend, *args)
            if has_prebuilt_ops and uri in prebuilt_ops_uri:
                # 路径1: Prebuilt Ops (预编译的kernel)
                if backend == "fa2":
                    _kernels = torch.ops.flashinfer_hip_kernels
                    run_func = _kernels.single_prefill_with_kv_cache.default
                else:
                    _kernels_sm90 = torch.ops.flashinfer_kernels_sm90
                    run_func = _kernels_sm90.single_prefill_with_kv_cache_sm90.default
            else:
                # 路径2: JIT编译 (运行时编译)
                module = gen_single_prefill_module(backend, *args).build_and_load()
                run_func = module.run.default
            
            # 注册自定义op
            @register_custom_op(f"flashinfer::{uri}_run", ...)
            def run_single_prefill(...):
                run_func(...)  # 调用实际的kernel函数
```

### 3. PyTorch Extension加载

**文件路径**: `flashinfer/flashinfer/jit/core.py:119-129`

**关键函数**: `JitSpec.build_and_load()`

**关键代码**:
```python
def build_and_load(self):
    # 编译并加载.so文件
    torch.ops.load_library(so_path)  # 加载动态库
    # 调用C++/HIP kernel函数
    return torch.ops.<module_name>.run(...)
```

---

## 关键数据结构

### 1. AQL Queue结构

```c
struct AQLQueue {
    uint64_t base_address;      // Queue基地址
    uint32_t doorbell;           // Doorbell寄存器地址
    uint32_t size;               // Queue大小（必须是2的幂）
    volatile uint32_t write_index;  // 写指针 (wptr)
    volatile uint32_t read_index;   // 读指针 (rptr)
    AQLPacket packets[];         // AQL packet数组
};
```

### 2. AQL Packet结构

```c
struct AQLPacket {
    uint16_t header;             // Packet header (type=2表示kernel dispatch)
    uint16_t setup;               // Setup字段
    uint32_t workgroup_size_x;    // Workgroup大小X
    uint32_t workgroup_size_y;    // Workgroup大小Y
    uint32_t workgroup_size_z;    // Workgroup大小Z
    uint16_t grid_size_x;        // Grid大小X
    uint16_t grid_size_y;        // Grid大小Y
    uint16_t grid_size_z;        // Grid大小Z
    uint16_t private_segment_size;  // 私有段大小
    uint16_t group_segment_size;    // 组段大小
    uint64_t kernel_object;      // Kernel对象地址
    uint64_t kernarg_address;    // Kernel参数地址
    uint64_t completion_signal;  // 完成信号地址
    // ... 其他字段
};
```

### 3. MES Queue Input结构

```c
struct mes_add_queue_input {
    uint32_t queue_type;          // Queue类型 (MES_QUEUE_TYPE_COMPUTE等)
    uint64_t queue_address;       // Queue地址
    uint64_t doorbell_offset;     // Doorbell偏移
    uint32_t queue_size;          // Queue大小
    // ... 其他字段
};
```

---

## 关键系统调用

### 1. KFD_IOC_CREATE_QUEUE

**ioctl号**: `AMDKFD_IOC_CREATE_QUEUE`

**参数结构**:
```c
struct kfd_ioctl_create_queue_args {
    uint32_t queue_id;
    uint32_t queue_type;         // KFD_IOC_QUEUE_TYPE_COMPUTE_AQL等
    uint32_t queue_priority;
    uint64_t ring_base_address;
    uint64_t ring_size;
    uint64_t queue_address;
    uint64_t queue_size;
    uint64_t doorbell_offset;
    // ... 其他字段
};
```

**用途**: 创建AQL Queue并注册到KFD驱动

### 2. KFD_IOC_DESTROY_QUEUE

**ioctl号**: `AMDKFD_IOC_DESTROY_QUEUE`

**用途**: 销毁AQL Queue

---

## 相关文档

- `AI_KERNEL_SUBMISSION_FLOW.md` - AI Kernel Submission流程详解
- `DRIVER_47_MES_KERNEL_SUBMISSION_ANALYSIS.md` - MES Kernel提交机制分析
- `DRIVER_46_MES_ADD_HW_QUEUE_ANALYSIS.md` - MES add_hw_queue实现分析
- `DRIVER_42_ROCM_CODE_ANALYSIS.md` - ROCm 6.4.3代码分析

---

## 更新日志

- **2025-01-XX**: 创建AI Kernel Submission关键代码参考文档

