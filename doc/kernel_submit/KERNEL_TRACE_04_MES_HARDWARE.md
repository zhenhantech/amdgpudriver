# Kernel提交流程追踪 (4/5) - MES调度器与硬件层

**范围**: MES调度器的实现和硬件交互  
**代码路径**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/`  
**关键操作**: MES add_hw_queue、MES Ring、硬件Doorbell检测

---

## ⚠️ 硬件要求说明

### MES 支持的 GPU 架构

**MES (Micro-Engine Scheduler)** 是硬件调度器，**仅在以下 GPU 上可用**：

| GPU 系列 | 代表型号 | GC IP 版本 | MES 支持 | 备注 |
|---------|---------|-----------|---------|------|
| **CDNA3** | MI300A/X | IP_VERSION(12, 0, x) | ✅ 支持 | 2023+ |
| **CDNA2** | MI250X, MI210 | IP_VERSION(9, 4, 1) | ✅ 支持 | 2021+ |
| **CDNA2** | **MI308X (Aqua Vanjaram)** | **IP_VERSION(9, 4, 2/3)** | **❌ 不支持** | **使用 CPSCH** |
| **CDNA1** | MI100 | IP_VERSION(9, 4, 0) | ❌ 不支持 | 使用 CPSCH |
| **Vega 20** | MI50, MI60 | IP_VERSION(9, 0, x) | ❌ 不支持 | 使用 CPSCH |
| **RDNA3** | RX 7900 XT/XTX | IP_VERSION(11, 0, x) | ✅ 支持 | 2022+ |
| **RDNA2** | RX 6000 系列 | IP_VERSION(10, 3, x) | ❌ 不支持 | 使用 CPSCH |

### 检查您的 GPU 是否支持 MES

```bash
# 方法1: 检查 enable_mes 参数
cat /sys/module/amdgpu/parameters/mes
# 输出: 1 = MES 启用, 0 = CPSCH 模式

# 方法2: 查看 dmesg 日志
dmesg | grep -i mes
# 如果看到 "MES enabled" 说明使用 MES
# 如果看到 "CPSCH mode" 或没有 MES 相关日志，说明使用 CPSCH

# 方法3: 查看 GPU 信息
rocminfo | grep -i "Name"
# 根据 GPU 型号判断
```

### 代码中的 MES 启用条件

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/amdgpu_discovery.c`

```c
static int amdgpu_discovery_set_mes_ip_blocks(struct amdgpu_device *adev)
{
    uint32_t gc_ip_version = amdgpu_ip_version(adev, GC_HWIP, 0);
    
    switch (gc_ip_version) {
    // RDNA3 系列
    case IP_VERSION(11, 0, 0):
    case IP_VERSION(11, 0, 1):
    case IP_VERSION(11, 0, 2):
    case IP_VERSION(11, 0, 3):
    case IP_VERSION(11, 0, 4):
    case IP_VERSION(11, 5, 0):
    case IP_VERSION(11, 5, 1):
    case IP_VERSION(11, 5, 2):
    // CDNA3 (MI300A/X)
    case IP_VERSION(12, 0, 0):
    case IP_VERSION(12, 0, 1):
        adev->enable_mes = true;  // ✅ 支持 MES
        break;
        
    default:
        // IP_VERSION(9, 4, x) - CDNA1/CDNA2 大部分型号
        // IP_VERSION(10, 3, x) - RDNA2
        adev->enable_mes = false; // ❌ 不支持 MES，使用 CPSCH
        break;
    }
}
```

> ⚠️ **重要发现**：MI308X (Aqua Vanjaram) 虽然名称类似 MI300 系列，但实际使用 **ALDEBARAN 架构**，GC IP 版本为 `IP_VERSION(9, 4, 2/3)`，**不支持 MES**，使用 **CPSCH 调度器**。这是基于实际硬件验证的结果。

### 本文档适用范围

- ✅ **本文档描述 MES 调度器的工作原理**，适用于支持 MES 的 GPU
- ⚠️ **如果您的 GPU 不支持 MES**（如 MI308X、MI100、Vega），系统将使用 **CPSCH 调度器**，流程会有所不同
- 📖 CPSCH 模式下，kernel 提交可能需要经过驱动层 Ring，而不是直接通过 doorbell

---

## 📋 本层概述

MES (Micro-Engine Scheduler) 是AMD GPU的硬件调度器，负责：
1. 管理GPU的硬件Queue
2. 检测Doorbell更新
3. 从AQL Queue读取packet并调度执行
4. 管理多个Queue的调度

本文档将深入MES的软件接口实现。

---

## 1️⃣ MES初始化

### 1.1 MES结构体定义

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/amdgpu_mes.h`

```c
struct amdgpu_mes {
    struct amdgpu_device *adev;              // 设备对象
    
    // MES固件
    const struct firmware *fw[AMDGPU_MAX_MES_PIPES];
    
    // MES Ring（用于提交MES命令）
    struct amdgpu_ring ring[AMDGPU_MAX_MES_PIPES];
    
    // MES函数指针表
    const struct amdgpu_mes_funcs *funcs;
    
    // MES调度管道
    uint32_t sched_pipe_mask;
    uint32_t compute_pipe_mask;
    uint32_t gfx_pipe_mask;
    uint32_t sdma_pipe_mask;
    
    // MES上下文
    struct amdgpu_bo *mes_ctx_bo;            // MES context buffer
    uint64_t mes_ctx_gpu_addr;
    void *mes_ctx_cpu_ptr;
    
    // Queue管理
    struct ida doorbell_ida;                 // Doorbell ID分配器
    struct mutex mutex_hidden;               // 互斥锁
    
    // 统计信息
    uint32_t total_max_queue;
    uint32_t num_mes_queues;
    
    // ... 其他字段
};
```

### 1.2 MES函数指针表

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/amdgpu_mes.h`

```c
struct amdgpu_mes_funcs {
    // Queue管理
    int (*add_hw_queue)(struct amdgpu_mes *mes,
                       struct mes_add_queue_input *input);
    int (*remove_hw_queue)(struct amdgpu_mes *mes,
                          struct mes_remove_queue_input *input);
    int (*suspend_gang)(struct amdgpu_mes *mes,
                       struct mes_suspend_gang_input *input);
    int (*resume_gang)(struct amdgpu_mes *mes,
                      struct mes_resume_gang_input *input);
    
    // MES控制
    int (*set_hw_resources)(struct amdgpu_mes *mes);
    int (*query_sched_status)(struct amdgpu_mes *mes);
    
    // 其他MES操作
    // ...
};
```

### 1.3 MES初始化（以v12.0为例）

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/mes_v12_0.c`

```c
static int mes_v12_0_init(struct amdgpu_device *adev)
{
    struct amdgpu_mes *mes = &adev->mes;
    int r;
    
    // 1. 设置函数指针
    mes->funcs = &mes_v12_0_funcs;
    
    // 2. 初始化MES Ring
    r = mes_v12_0_init_microcode(adev);
    if (r) {
        dev_err(adev->dev, "Failed to init MES microcode\n");
        return r;
    }
    
    // 3. 分配MES context buffer
    r = amdgpu_bo_create_kernel(adev,
                                AMDGPU_MES_CTX_SIZE,
                                PAGE_SIZE,
                                AMDGPU_GEM_DOMAIN_GTT,
                                &mes->mes_ctx_bo,
                                &mes->mes_ctx_gpu_addr,
                                &mes->mes_ctx_cpu_ptr);
    if (r) {
        dev_err(adev->dev, "Failed to allocate MES context\n");
        return r;
    }
    
    // 4. 初始化doorbell分配器
    ida_init(&mes->doorbell_ida);
    
    // 5. 初始化互斥锁
    mutex_init(&mes->mutex_hidden);
    
    return 0;
}
```

---

## 2️⃣ MES add_hw_queue 实现

### 2.1 add_hw_queue 入口

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/mes_v12_0.c`

```c
static int mes_v12_0_add_hw_queue(struct amdgpu_mes *mes,
                                  struct mes_add_queue_input *input)
{
    union MESAPI__ADD_QUEUE mes_add_queue_pkt;
    int pipe, queue_type, r;
    
    // 1. 确定queue类型
    queue_type = convert_to_mes_queue_type(input->queue_type);
    
    // 2. 确定使用哪个MES pipe
    pipe = mes_v12_0_select_pipe(mes, queue_type);
    if (pipe < 0) {
        dev_err(mes->adev->dev, "No available MES pipe\n");
        return -EINVAL;
    }
    
    // 3. 清零MES packet
    memset(&mes_add_queue_pkt, 0, sizeof(mes_add_queue_pkt));
    
    // 4. 填充MES ADD_QUEUE packet
    mes_v12_0_fill_add_queue_packet(input, &mes_add_queue_pkt);
    
    // 5. 提交packet到MES并等待完成
    // 这是关键步骤！
    r = mes_v12_0_submit_pkt_and_poll_completion(mes,
                                                 pipe,
                                                 &mes_add_queue_pkt,
                                                 sizeof(mes_add_queue_pkt),
                                                 offsetof(union MESAPI__ADD_QUEUE, 
                                                         api_status));
    
    if (r) {
        dev_err(mes->adev->dev, "Failed to add queue to MES: %d\n", r);
        return r;
    }
    
    // 6. 检查MES返回的状态
    if (mes_add_queue_pkt.api_status.api_completion_fence_value !=
        AMDGPU_MES_STATUS_SUCCESS) {
        dev_err(mes->adev->dev, 
                "MES add queue failed with status: 0x%x\n",
                mes_add_queue_pkt.api_status.api_completion_fence_value);
        return -EINVAL;
    }
    
    return 0;
}
```

### 2.2 填充ADD_QUEUE Packet

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/mes_v12_0.c`

```c
static void mes_v12_0_fill_add_queue_packet(
    struct mes_add_queue_input *input,
    union MESAPI__ADD_QUEUE *pkt)
{
    // Packet header
    pkt->header.type = MES_API_TYPE_SCHEDULER;
    pkt->header.opcode = MES_SCH_API_ADD_QUEUE;
    pkt->header.dwsize = sizeof(*pkt) / 4;
    
    // Process信息
    pkt->process_id = input->process_id;
    pkt->page_table_base_addr = input->page_table_base_addr;
    pkt->process_va_start = input->process_va_start;
    pkt->process_va_end = input->process_va_end;
    pkt->process_quantum = input->process_quantum;
    pkt->process_context_addr = input->process_context_addr;
    
    // Gang调度信息（MI300等新架构）
    pkt->gang_context_addr = input->gang_context_addr;
    pkt->inprocess_gang_priority = input->inprocess_gang_priority;
    pkt->gang_global_priority_level = input->gang_global_priority_level;
    
    // Queue信息
    pkt->queue_type = convert_to_mes_queue_type(input->queue_type);
    pkt->mqd_addr = input->mqd_addr;
    pkt->wptr_addr = input->wptr_addr;
    pkt->queue_size = input->queue_size;
    pkt->doorbell_offset = input->doorbell_offset;
    
    // GDS (Global Data Share)
    pkt->gds_base = input->gds_base;
    pkt->gds_size = input->gds_size;
    pkt->gws_base = input->gws_base;
    pkt->gws_size = input->gws_size;
    pkt->oa_mask = input->oa_mask;
    
    // 调试和trace
    pkt->trap_handler_addr = input->tba_addr;
    pkt->tma_addr = input->tma_addr;
    
    // 其他标志
    pkt->is_kfd_process = input->is_kfd_process;
    pkt->is_aql_queue = (input->queue_type == MES_QUEUE_TYPE_COMPUTE_AQL);
}
```

### 2.3 MES Packet结构

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/mes_v12_0.c`

```c
// MES ADD_QUEUE API packet结构
union MESAPI__ADD_QUEUE {
    struct {
        // Header (4 DWords)
        union MES_API_HEADER header;
        
        // Process信息
        uint32_t process_id;
        uint64_t page_table_base_addr;
        uint64_t process_va_start;
        uint64_t process_va_end;
        uint64_t process_quantum;
        uint64_t process_context_addr;
        
        // Gang信息
        uint64_t gang_context_addr;
        uint32_t inprocess_gang_priority;
        uint32_t gang_global_priority_level;
        
        // Queue信息
        uint32_t queue_type;              // Compute/SDMA等
        uint64_t mqd_addr;                // MQD地址
        uint64_t wptr_addr;               // 写指针地址
        uint32_t queue_size;              // Queue大小
        uint64_t doorbell_offset;         // Doorbell偏移
        
        // GDS信息
        uint32_t gds_base;
        uint32_t gds_size;
        uint32_t gws_base;
        uint32_t gws_size;
        uint32_t oa_mask;
        
        // Trap handler
        uint64_t trap_handler_addr;
        uint64_t tma_addr;
        
        // 标志
        uint32_t is_kfd_process;
        uint32_t is_aql_queue;
        uint32_t is_tmz_queue;
        
        // Reserved
        uint32_t reserved[10];
        
        // API status (MES填充)
        struct MES_API_STATUS api_status;
    };
    
    uint32_t max_dwords[API_FRAME_SIZE_IN_DWORDS];
};
```

---

## 3️⃣ MES Ring和Packet提交

### 3.1 MES Ring初始化

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/mes_v12_0.c`

```c
static int mes_v12_0_ring_init(struct amdgpu_device *adev)
{
    struct amdgpu_mes *mes = &adev->mes;
    struct amdgpu_ring *ring;
    int i, r;
    
    // 为每个MES pipe初始化一个ring
    for (i = 0; i < AMDGPU_MAX_MES_PIPES; i++) {
        ring = &mes->ring[i];
        
        // 设置ring类型
        ring->ring_obj = NULL;
        ring->use_doorbell = true;
        ring->doorbell_index = (adev->doorbell_index.mes_ring0 << 1) + i;
        
        // 设置ring函数指针
        ring->funcs = &mes_v12_0_ring_funcs;
        
        // 初始化ring
        r = amdgpu_ring_init(adev, ring, 1024,
                           &mes->mes_irq, 0);
        if (r) {
            dev_err(adev->dev, "Failed to init MES ring %d\n", i);
            return r;
        }
    }
    
    return 0;
}
```

### 3.2 提交Packet到MES Ring

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/mes_v12_0.c`

```c
static int mes_v12_0_submit_pkt_and_poll_completion(
    struct amdgpu_mes *mes,
    int pipe,
    void *pkt,
    int size,
    int api_status_off)
{
    struct amdgpu_ring *ring = &mes->ring[pipe];
    struct amdgpu_device *adev = mes->adev;
    union MESAPI__ADD_QUEUE *x_pkt = pkt;
    signed long timeout = 3000000;  // 3秒超时
    int r;
    
    // 1. 锁定ring
    r = amdgpu_ring_lock(ring, (size + 7) / 4);
    if (r) {
        dev_err(adev->dev, "Failed to lock MES ring\n");
        return r;
    }
    
    // 2. 设置fence（用于同步）
    x_pkt->api_status.api_completion_fence_addr = mes->mes_ctx_gpu_addr +
        offsetof(struct amdgpu_mes_ctx, api_completion_fence);
    x_pkt->api_status.api_completion_fence_value = ++mes->api_fence_value;
    
    // 3. 写入packet到ring
    // 使用ring的write_multiple函数
    amdgpu_ring_write_multiple(ring, pkt, size / 4);
    
    // 4. 提交ring（写入doorbell，通知MES）
    amdgpu_ring_commit(ring);
    
    // 5. 解锁ring
    amdgpu_ring_unlock(ring);
    
    // 6. 轮询等待MES完成
    // MES完成后会更新api_completion_fence
    r = mes_v12_0_poll_api_status(mes,
                                  x_pkt->api_status.api_completion_fence_value,
                                  timeout);
    if (r) {
        dev_err(adev->dev, "MES API timeout\n");
        return r;
    }
    
    return 0;
}
```

### 3.3 轮询MES完成状态

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/mes_v12_0.c`

```c
static int mes_v12_0_poll_api_status(struct amdgpu_mes *mes,
                                     uint64_t fence_value,
                                     signed long timeout)
{
    volatile uint64_t *fence_ptr = 
        (volatile uint64_t *)(mes->mes_ctx_cpu_ptr +
                             offsetof(struct amdgpu_mes_ctx,
                                     api_completion_fence));
    
    signed long wait_time = timeout;
    
    // 轮询等待fence值更新
    while (*fence_ptr != fence_value && wait_time > 0) {
        usleep_range(10, 100);  // 休眠10-100微秒
        wait_time -= 10;
    }
    
    if (*fence_ptr != fence_value) {
        dev_err(mes->adev->dev,
                "MES API timeout: expected 0x%llx, got 0x%llx\n",
                fence_value, *fence_ptr);
        return -ETIMEDOUT;
    }
    
    return 0;
}
```

### 3.4 amdgpu_ring_commit - 提交到硬件

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/amdgpu_ring.c`

```c
void amdgpu_ring_commit(struct amdgpu_ring *ring)
{
    uint32_t count;
    
    // 1. 计算写入的命令数
    count = ring->wptr & ring->buf_mask;
    
    // 2. CPU内存屏障（确保命令可见）
    mb();
    
    // 3. 写入ring的wptr
    amdgpu_ring_set_wptr(ring);
    
    // 4. 如果使用doorbell，写入doorbell寄存器
    // 这会通知硬件有新命令
    if (ring->use_doorbell) {
        // 计算doorbell地址
        uint32_t *doorbell = (uint32_t *)(ring->adev->doorbell.ptr + 
                                         ring->doorbell_index);
        
        // 写入doorbell（触发硬件）
        WRITE_ONCE(*doorbell, ring->wptr);
    }
}
```

---

## 4️⃣ MES Ring类型和函数

### 4.1 MES Ring函数指针表

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/mes_v12_0.c`

```c
static const struct amdgpu_ring_funcs mes_v12_0_ring_funcs = {
    .type = AMDGPU_RING_TYPE_MES,          // Ring类型：MES
    .align_mask = 1,
    .nop = 0,                               // NOP命令
    .support_64bit_ptrs = true,
    .get_rptr = mes_v12_0_ring_get_rptr,
    .get_wptr = mes_v12_0_ring_get_wptr,
    .set_wptr = mes_v12_0_ring_set_wptr,
    .emit_frame_size = 8,                   // 每个命令的大小
    .emit_ib_size = 7,
    .emit_ib = mes_v12_0_ring_emit_ib,
    .emit_fence = mes_v12_0_ring_emit_fence,
    .test_ring = mes_v12_0_ring_test_ring,
    .test_ib = mes_v12_0_ring_test_ib,
    .insert_nop = mes_v12_0_ring_insert_nop,
    .pad_ib = amdgpu_ring_generic_pad_ib,
    .emit_wreg = mes_v12_0_ring_emit_wreg,
    .emit_reg_wait = mes_v12_0_ring_emit_reg_wait,
    .emit_reg_write_reg_wait = mes_v12_0_ring_emit_reg_write_reg_wait,
};
```

**关键发现**:
- ✅ MES Ring的类型是 `AMDGPU_RING_TYPE_MES`
- ✅ 不是Compute Ring或SDMA Ring
- ✅ 专门用于MES管理命令

---

## 5️⃣ Doorbell机制深入

### 5.1 Doorbell映射

在KFD创建queue时，会分配doorbell偏移：

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_doorbell.c`

```c
uint64_t kfd_get_doorbell_dw_offset_in_bar(struct kfd_node *dev,
                                           struct kfd_process_device *pdd,
                                           struct queue *q)
{
    // 计算doorbell在BAR空间中的偏移
    uint64_t doorbell_id = q->doorbell_id;
    uint64_t offset;
    
    // doorbell_id * doorbell_size
    offset = doorbell_id * dev->device_info.doorbell_size;
    
    // 加上进程的doorbell基地址
    offset += pdd->doorbell_index * dev->device_info.doorbell_size;
    
    return offset;
}
```

### 5.2 用户空间mmap Doorbell

在HSA Runtime中（前面已经看到）：

```cpp
// 映射doorbell到用户空间
void* doorbell_ptr = mmap(
    NULL,
    sizeof(uint64_t),
    PROT_READ | PROT_WRITE,
    MAP_SHARED,
    kfd_fd,
    doorbell_offset  // 这是KFD返回的偏移
);

// 用户空间可以直接写入！
*doorbell_ptr = write_index;
```

### 5.3 硬件检测Doorbell

```
用户空间写入doorbell
    ↓
写入到映射的内存地址
    ↓
通过PCIe BAR映射到GPU的doorbell寄存器
    ↓
GPU硬件实时监控doorbell寄存器
    ↓
检测到更新，触发MES硬件调度器
    ↓
MES从AQL Queue读取packet
    ↓
解析packet，调度kernel执行
```

---

## 6️⃣ MES硬件调度流程

### 6.1 MES检测Doorbell更新

```
硬件层面（无软件代码，纯硬件逻辑）:

MES硬件调度器持续监控:
  ↓
检测到Doorbell寄存器更新
  ↓
根据Doorbell ID定位到对应的Queue
  ↓
读取Queue的MQD（Memory Queue Descriptor）
  ↓
从MQD获取:
  - Queue基地址
  - 当前read_ptr
  - Queue大小
  ↓
计算packet地址:
  packet_addr = queue_base + (read_ptr % queue_size) * 64
  ↓
从GPU内存读取AQL packet（64字节）
  ↓
解析packet header:
  - type = 2: Kernel Dispatch
  - type = 1: Barrier
  - type = 3: Agent Dispatch
  ↓
如果是Kernel Dispatch:
  提取kernel信息:
    - grid大小
    - workgroup大小
    - kernel代码地址
    - kernel参数地址
  ↓
  分配GPU资源:
    - 选择Compute Unit (CU)
    - 分配LDS (Local Data Share)
    - 分配VGPR/SGPR
  ↓
  调度kernel到CU执行
  ↓
  更新read_ptr
  ↓
  继续检查是否有更多packet
```

### 6.2 从软件角度看MES

```
软件视角:

1. Queue创建阶段:
   KFD Driver → MES Ring → ADD_QUEUE命令
   MES硬件记录Queue信息（从MQD）

2. Kernel提交阶段:
   用户空间 → 写AQL packet → 写doorbell
   （无需软件参与）
   
3. Kernel执行阶段:
   MES硬件自动:
     - 检测doorbell
     - 读取packet
     - 调度执行
     - 更新completion signal

软件无感知，完全由硬件处理！
```

---

## 7️⃣ 关键数据结构

### 7.1 mes_add_queue_input

```c
// KFD传递给MES的queue信息
struct mes_add_queue_input {
    uint32_t process_id;                  // 进程ID (PASID)
    uint64_t page_table_base_addr;        // 页表基地址
    uint64_t process_va_start;            // 进程虚拟地址起始
    uint64_t process_va_end;              // 进程虚拟地址结束
    uint64_t process_quantum;             // 进程时间片
    uint64_t process_context_addr;        // 进程context地址
    uint64_t gang_context_addr;           // Gang context地址
    
    uint32_t queue_type;                  // Queue类型
    uint64_t mqd_addr;                    // MQD地址
    uint64_t wptr_addr;                   // 写指针地址
    uint32_t queue_size;                  // Queue大小
    uint64_t doorbell_offset;             // Doorbell偏移
    
    uint32_t gds_base;                    // GDS基地址
    uint32_t gds_size;                    // GDS大小
    // ... 其他字段
};
```

### 7.2 MES Queue类型

```c
enum mes_queue_type {
    MES_QUEUE_TYPE_GFX,
    MES_QUEUE_TYPE_COMPUTE,
    MES_QUEUE_TYPE_COMPUTE_AQL,           // 我们使用的类型
    MES_QUEUE_TYPE_SDMA,
    MES_QUEUE_TYPE_SDMA_XGMI,
};
```

---

## 8️⃣ 流程图

```
KFD Driver: create_queue_mes()
  │
  │ 准备 mes_add_queue_input
  ↓
调用 mes->funcs->add_hw_queue()
  ↓
────────────────────────────────────────────────
AMDGPU Driver: mes_v12_0_add_hw_queue()
  │
  │ 1. 转换queue类型
  │ 2. 选择MES pipe
  │ 3. 填充ADD_QUEUE packet
  ↓
mes_v12_0_submit_pkt_and_poll_completion()
  │
  │ 1. 锁定MES ring
  │ 2. 设置completion fence
  │ 3. 写入packet到ring
  ↓
amdgpu_ring_commit()
  │
  │ 1. 内存屏障
  │ 2. 更新ring->wptr
  │ 3. 写入doorbell寄存器  ← 触发硬件！
  ↓
────────────────────────────────────────────────
硬件层: MES硬件调度器

检测到MES Ring的doorbell更新
  ↓
从MES Ring读取ADD_QUEUE packet
  ↓
解析packet，提取queue信息
  ↓
从MQD读取queue详细配置
  ↓
注册queue到MES调度表
  ↓
设置completion fence（通知软件）
  ↓
────────────────────────────────────────────────
返回路径:

mes_v12_0_poll_api_status()
  │ 轮询completion fence
  ↓
Queue创建完成
  ↓
返回KFD Driver
```

---

## 9️⃣ 关键代码位置总结

| 功能 | 文件路径 | 关键函数 |
|------|---------|---------|
| MES初始化 | `amdgpu/mes_v12_0.c` | `mes_v12_0_init()` |
| add_hw_queue入口 | `amdgpu/mes_v12_0.c` | `mes_v12_0_add_hw_queue()` |
| 填充MES packet | `amdgpu/mes_v12_0.c` | `mes_v12_0_fill_add_queue_packet()` |
| 提交packet | `amdgpu/mes_v12_0.c` | `mes_v12_0_submit_pkt_and_poll_completion()` |
| Ring commit | `amdgpu/amdgpu_ring.c` | `amdgpu_ring_commit()` |
| 轮询完成 | `amdgpu/mes_v12_0.c` | `mes_v12_0_poll_api_status()` |
| Doorbell管理 | `amdkfd/kfd_doorbell.c` | `kfd_get_doorbell_dw_offset_in_bar()` |
| MES Ring函数 | `amdgpu/mes_v12_0.c` | `mes_v12_0_ring_funcs` |

---

## 🔟 关键发现

### 10.1 两种Doorbell用途

**MES Ring的Doorbell** (Queue创建时使用):
```
KFD → 准备ADD_QUEUE命令
    ↓
写入MES Ring
    ↓
写入MES Ring的doorbell
    ↓
MES硬件检测，处理ADD_QUEUE命令
    ↓
注册Queue到MES
```

**AQL Queue的Doorbell** (Kernel提交时使用):
```
用户空间 → 写AQL packet
         ↓
         写入AQL Queue的doorbell
         ↓
         MES硬件检测
         ↓
         从AQL Queue读取packet
         ↓
         调度kernel执行
```

### 10.2 MES Ring vs AQL Queue

| 特性 | MES Ring | AQL Queue |
|------|---------|----------|
| 用途 | MES管理命令 | Kernel提交 |
| 命令类型 | ADD_QUEUE, REMOVE_QUEUE等 | Kernel Dispatch |
| 访问者 | KFD驱动（内核空间） | 用户空间（HSA Runtime） |
| 频率 | 低（Queue创建/销毁时） | 高（每次kernel启动） |
| Ring类型 | AMDGPU_RING_TYPE_MES | 不是Ring，是Queue |

### 10.3 为什么Doorbell机制高效？

1. **零系统调用**:
   - Kernel提交时无需进入内核
   - 直接写入映射的doorbell

2. **硬件直接处理**:
   - MES硬件监控doorbell
   - 无需软件中介

3. **并行处理**:
   - 多个Queue可以同时提交
   - MES硬件并行调度

---

## 1️⃣1️⃣ 下一步

在最后一章，我们将详细介绍：
- AQL Packet的完整格式
- 关键数据结构的详细定义
- Context、Entity等概念的深入理解

继续阅读: [KERNEL_TRACE_05_DATA_STRUCTURES.md](./KERNEL_TRACE_05_DATA_STRUCTURES.md)


