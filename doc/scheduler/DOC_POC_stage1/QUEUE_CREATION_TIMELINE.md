# KFD_IOC_CREATE_QUEUE 调用时机详解

**日期**: 2026-02-04  
**目的**: 说明从用户态创建AQL Queue到调用KFD_IOC_CREATE_QUEUE的完整时序

---

## 📌 核心答案

**Q: `KFD_IOC_CREATE_QUEUE` 在 AQL_Queue_A 创建时什么时候会调用？**

**A: 在用户态Runtime分配好ring-buffer等资源后，立即调用此ioctl通知内核创建对应的MQD和管理结构**

```
时序关系:
  T1: 用户态分配ring-buffer、read/write指针（mmap内存）
  T2: ioctl(KFD_IOC_CREATE_QUEUE) ← ⭐ 在这个时刻调用
  T3: 内核创建MQD、分配doorbell、配置CWSR
  T4: 返回用户态：queue_id + doorbell_offset
  T5: 用户态mmap doorbell，开始使用队列
```

**关键**: AQL Queue的数据结构（ring-buffer）由**用户态**分配，但MQD等内核管理结构由**ioctl触发内核创建**。

---

## 🔄 完整调用链（从上到下）

### Level 1: 用户应用层（HIP）

```cpp
// 用户代码
hipStream_t stream;
hipStreamCreate(&stream);  // ← 入口点
```

**作用**: HIP封装接口，用户调用。

---

### Level 2: HIP Runtime层

```cpp
// HIP Runtime内部（闭源，推测流程）
hipError_t hipStreamCreate(hipStream_t* stream) {
    // 1. 调用HSA Runtime创建队列
    hsa_queue_t* hsa_queue;
    hsa_queue_create(agent,        // GPU设备
                     queue_size,   // 队列大小（如64KB）
                     HSA_QUEUE_TYPE_MULTI,
                     callback,
                     &hsa_queue);  // ← 关键：调用HSA API
    
    // 2. 封装为hipStream_t
    *stream = wrap_hsa_queue(hsa_queue);
    return hipSuccess;
}
```

**作用**: HIP调用底层HSA Runtime API。

---

### Level 3: HSA Runtime层（ROCR-Runtime）

**代码位置**: ROCm/ROCR-Runtime (或 ROCm/rocm-systems)  
**关键函数**: `hsa_queue_create()`

```cpp
// HSA Runtime实现（开源，简化版）
hsa_status_t hsa_queue_create(
    hsa_agent_t agent,
    uint32_t size,
    hsa_queue_type_t type,
    void (*callback)(hsa_status_t, hsa_queue_t*, void*),
    hsa_queue_t** queue)
{
    // ===== 步骤1: 用户态资源分配 ⭐ =====
    // 1.1 分配队列结构
    hsa_queue_t* q = malloc(sizeof(hsa_queue_t));
    
    // 1.2 分配ring-buffer（通过mmap共享内存）
    size_t ring_size = size * sizeof(hsa_packet_t);  // 如 256 * 64B = 16KB
    void* ring_buffer = mmap(NULL, ring_size,
                             PROT_READ | PROT_WRITE,
                             MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    q->base_address = (uint64_t)ring_buffer;
    
    // 1.3 分配read/write指针（用户态可见）
    uint64_t* read_ptr = mmap(...);   // GPU更新这里
    uint64_t* write_ptr = mmap(...);  // 用户更新这里
    q->read_dispatch_id = (uint64_t)read_ptr;
    q->write_dispatch_id = (uint64_t)write_ptr;
    
    // ===== 步骤2: 调用libhsakmt创建内核队列 ⭐⭐⭐ =====
    // 2.1 准备ioctl参数
    struct kfd_ioctl_create_queue_args args = {
        .gpu_id = get_gpu_id(agent),
        .queue_type = HSA_QUEUE_COMPUTE_AQL,  // AQL格式
        .queue_percentage = 100,              // 队列百分比
        .queue_priority = 15,                 // 优先级（0-15）
        
        // ⭐ 关键：传递用户态分配的地址
        .ring_base_address = (uint64_t)ring_buffer,
        .ring_size = ring_size,
        .read_pointer_address = (uint64_t)read_ptr,
        .write_pointer_address = (uint64_t)write_ptr,
    };
    
    // 2.2 打开KFD设备（如果未打开）
    int kfd_fd = open("/dev/kfd", O_RDWR);
    
    // 2.3 ⭐⭐⭐ 调用ioctl - 在这个时刻！⭐⭐⭐
    int ret = ioctl(kfd_fd, AMDKFD_IOC_CREATE_QUEUE, &args);
    //             ↑↑↑ 这里就是 KFD_IOC_CREATE_QUEUE 被调用的时刻！
    
    if (ret != 0) {
        // 清理用户态资源
        munmap(ring_buffer, ring_size);
        free(q);
        return HSA_STATUS_ERROR;
    }
    
    // ===== 步骤3: 接收内核返回的信息 =====
    // 3.1 保存queue_id（用于后续操作）
    q->queue_id = args.queue_id;
    
    // 3.2 获取doorbell地址
    uint32_t doorbell_offset = args.doorbell_offset;
    
    // ===== 步骤4: mmap doorbell（用户态可写） ⭐ =====
    // 4.1 通过mmap映射doorbell寄存器
    void* doorbell_ptr = mmap(NULL, 8,  // 8字节doorbell
                              PROT_WRITE,
                              MAP_SHARED,
                              kfd_fd,
                              doorbell_offset);  // 使用内核返回的offset
    q->doorbell_ptr = (uint64_t*)doorbell_ptr;
    
    // ===== 步骤5: 返回队列给调用者 =====
    *queue = q;
    return HSA_STATUS_SUCCESS;
}
```

**关键时刻**:
```
T1: 用户态mmap ring-buffer, read_ptr, write_ptr
T2: ioctl(AMDKFD_IOC_CREATE_QUEUE, &args)  ← ⭐⭐⭐ 就在这里！
T3: 内核处理（见Level 4）
T4: 返回 queue_id + doorbell_offset
T5: 用户态mmap doorbell
```

---

### Level 4: 内核KFD层（处理ioctl）

**代码位置**: `/usr/src/amdgpu-6.12.12-2194681.el8_preempt/amd/amdkfd/kfd_chardev.c:311`

```c
static int kfd_ioctl_create_queue(struct file *filep, 
                                  struct kfd_process *p,
                                  void *data)
{
    struct kfd_ioctl_create_queue_args *args = data;  // ← 用户态传入的参数
    struct queue_properties q_properties;
    uint32_t doorbell_offset_in_process = 0;
    
    // ===== 步骤1: 验证并转换用户参数 ⭐ =====
    err = set_queue_properties_from_user(&q_properties, args);
    // 从 args 提取：
    //   - ring_base_address  → q_properties.queue_address
    //   - ring_size          → q_properties.queue_size
    //   - read_pointer_addr  → q_properties.read_ptr
    //   - write_pointer_addr → q_properties.write_ptr
    //   - queue_priority     → q_properties.priority
    //   - queue_type         → q_properties.type
    
    // ===== 步骤2: 查找GPU设备 =====
    pdd = kfd_process_device_data_by_id(p, args->gpu_id);
    dev = pdd->dev;
    
    // ===== 步骤3: 分配doorbell ⭐⭐ =====
    if (!pdd->qpd.proc_doorbells) {
        err = kfd_alloc_process_doorbells(dev->kfd, pdd);
        // 为进程分配doorbell页面（2个4KB页）
    }
    
    // ===== 步骤4: 获取和引用用户BO（Buffer Objects）=====
    err = kfd_queue_acquire_buffers(pdd, &q_properties);
    // 引用用户态的ring_buffer, read_ptr, write_ptr BO
    
    // ===== 步骤5: 创建队列（核心调用）⭐⭐⭐ =====
    err = pqm_create_queue(&p->pqm,        // Process Queue Manager
                          dev, 
                          &q_properties,   // 队列属性
                          &queue_id,       // 输出：分配的queue_id
                          NULL, NULL, NULL,
                          &doorbell_offset_in_process);
    // 这个函数会：
    //   1. 创建 kfd_queue 结构
    //   2. 分配 MQD（Memory Queue Descriptor）⭐⭐⭐
    //   3. 分配 doorbell ID
    //   4. 配置 CWSR 上下文保存区
    //   5. 初始化 MQD 内容（包括ring地址、doorbell等）
    //   6. 如果 is_active=true，加入 runlist
    
    // ===== 步骤6: 返回信息给用户态 ⭐ =====
    args->queue_id = queue_id;  // 队列ID（用于后续操作）
    
    // 构造doorbell offset（用于mmap）
    args->doorbell_offset = KFD_MMAP_TYPE_DOORBELL;
    args->doorbell_offset |= KFD_MMAP_GPU_ID(args->gpu_id);
    args->doorbell_offset |= doorbell_offset_in_process;
    
    return 0;  // 成功
}
```

---

### Level 5: KFD队列管理层

**代码位置**: `/usr/src/amdgpu-6.12.12-2194681.el8_preempt/amd/amdkfd/kfd_process_queue_manager.c`

```c
int pqm_create_queue(struct process_queue_manager *pqm,
                    struct kfd_node *dev,
                    struct queue_properties *properties,
                    unsigned int *qid,
                    ...)
{
    // ===== 步骤1: 分配队列结构 =====
    struct queue *q = kzalloc(sizeof(*q), GFP_KERNEL);
    
    // ===== 步骤2: 分配MQD内存 ⭐⭐⭐ =====
    // 2.1 获取MQD管理器
    mqd_mgr = dev->dqm->mqd_mgrs[properties->type];
    
    // 2.2 分配MQD BO（GPU可访问的内存）
    err = mqd_mgr->allocate_mqd(mqd_mgr, &q->mqd, &q->mqd_mem_obj);
    // MQD大小约4KB，分配在GTT或VRAM
    
    // ===== 步骤3: 初始化MQD内容 ⭐⭐⭐ =====
    err = mqd_mgr->init_mqd(mqd_mgr, 
                           &q->mqd,         // MQD指针
                           &q->mqd_mem_obj, // MQD BO
                           &q->gart_mqd_addr, // MQD GPU地址
                           properties);     // 包含ring地址等
    // 这里会调用 update_mqd()，填充：
    //   - m->cp_hqd_pq_base     = properties->queue_address  ⭐
    //   - m->cp_hqd_pq_rptr     = properties->read_ptr       ⭐
    //   - m->cp_hqd_pq_wptr     = properties->write_ptr      ⭐
    //   - m->cp_hqd_doorbell    = doorbell_offset            ⭐
    //   - m->cp_hqd_ctx_save    = cwsr_area                  ⭐
    
    // ===== 步骤4: 分配CWSR上下文保存区 ⭐⭐⭐ =====
    if (dev->kfd->cwsr_enabled) {
        size_t cwsr_size = calculate_cwsr_size(...);  // 如2MB
        void* cwsr_area = kfd_alloc_gtt_mem(cwsr_size);
        properties->ctx_save_restore_area_address = (uint64_t)cwsr_area;
        // ↑↑↑ 这个地址会写入MQD，硬件抢占时用
    }
    
    // ===== 步骤5: 分配doorbell ⭐ =====
    err = allocate_doorbell(pdd->qpd, q, &doorbell_id);
    properties->doorbell_off = doorbell_id * 8;  // 8字节对齐
    
    // ===== 步骤6: 添加到设备队列管理器 =====
    err = dev->dqm->ops.create_queue(dev->dqm, q, ...);
    // 这会：
    //   1. 设置 q->properties.is_active = true（如果立即激活）
    //   2. 调用 execute_queues_cpsch() → map_queues_cpsch()
    //   3. 通过 HIQ 发送 runlist 给 HWS
    
    // ===== 步骤7: 返回queue_id =====
    *qid = q->properties.queue_id;
    return 0;
}
```

---

## 📊 完整时序图

```
用户应用层:
  hipStreamCreate(stream)
    ↓

HIP Runtime层:
  hsa_queue_create(...)
    ↓
    
HSA Runtime层 (用户态):
  【步骤1】分配ring-buffer (mmap)
    ring_buffer = mmap(size=64KB)
    read_ptr    = mmap(size=8B)
    write_ptr   = mmap(size=8B)
    
  【步骤2】⭐⭐⭐ 调用ioctl - 关键时刻！ ⭐⭐⭐
    args.ring_base_address = ring_buffer;
    args.read_pointer_address = read_ptr;
    args.write_pointer_address = write_ptr;
    args.queue_priority = 15;
    
    ioctl(kfd_fd, AMDKFD_IOC_CREATE_QUEUE, &args);
    ↓
────────────────────────────────────────────────────
内核KFD层:
  kfd_ioctl_create_queue()
    ↓
  【步骤3】验证参数
    set_queue_properties_from_user(&q_properties, args)
    
  【步骤4】分配内核资源
    pqm_create_queue(...)
      ↓
      【4.1】分配MQD BO (4KB)
        mqd_mgr->allocate_mqd()
        
      【4.2】初始化MQD ⭐⭐⭐
        mqd_mgr->init_mqd()
          → update_mqd():
            m->cp_hqd_pq_base = args.ring_base_address  ⭐
            m->cp_hqd_pq_rptr = args.read_pointer       ⭐
            m->cp_hqd_pq_wptr = args.write_pointer      ⭐
            m->cp_hqd_doorbell = doorbell_offset        ⭐
            m->cp_hqd_ctx_save = cwsr_area (内核分配)   ⭐
            
      【4.3】分配doorbell ID
        allocate_doorbell() → doorbell_id = 5
        
      【4.4】分配CWSR保存区 (2MB)
        cwsr_area = kfd_alloc_gtt_mem(2MB)
        
      【4.5】添加到DQM
        create_queue_cpsch()
          → q->properties.is_active = true
          → execute_queues_cpsch()
            → map_queues_cpsch()
              → pm_send_runlist() ← 发送给HIQ
    
  【步骤5】返回给用户态
    args->queue_id = 123
    args->doorbell_offset = 0xABC000
    return 0;
    ↓
────────────────────────────────────────────────────
HSA Runtime层 (用户态):
  【步骤6】接收返回值
    queue_id = args.queue_id  (123)
    doorbell_off = args.doorbell_offset  (0xABC000)
    
  【步骤7】mmap doorbell
    doorbell_ptr = mmap(kfd_fd, doorbell_off)
    queue->doorbell = doorbell_ptr
    
  【步骤8】返回给HIP
    return queue;
    ↓

HIP Runtime层:
  stream = wrap_queue(queue)
  return stream;
  ↓

用户应用层:
  // 现在可以使用stream提交kernel了
  hipLaunchKernel<<<grid, block, 0, stream>>>(kernel, ...);
```

---

## 🔑 关键要点

### 1. 调用时机 ⭐⭐⭐

```
准确时机：用户态Runtime完成以下准备后立即调用

准备工作（T1）:
  ✓ 分配ring-buffer（mmap共享内存）
  ✓ 分配read/write指针（mmap）
  ✓ 准备队列参数（优先级、大小等）

ioctl调用（T2）:
  → ioctl(kfd_fd, AMDKFD_IOC_CREATE_QUEUE, &args)
  
内核处理（T3）:
  ✓ 创建MQD
  ✓ 分配doorbell
  ✓ 分配CWSR区域
  ✓ 初始化MQD（写入ring地址等）
  ✓ 发送runlist给HWS

返回用户（T4）:
  ← queue_id + doorbell_offset

后续操作（T5）:
  ✓ mmap doorbell
  ✓ 开始使用队列
```

### 2. 参数传递 ⭐⭐⭐

**用户态 → 内核态**（通过ioctl args）:
```c
struct kfd_ioctl_create_queue_args {
    uint64_t ring_base_address;      // 用户态分配的ring-buffer地址 ⭐
    uint32_t ring_size;              // ring大小（如64KB）
    uint64_t read_pointer_address;   // 用户态read_ptr地址 ⭐
    uint64_t write_pointer_address;  // 用户态write_ptr地址 ⭐
    uint32_t queue_priority;         // 优先级（0-15）
    uint32_t queue_percentage;       // 队列百分比
    uint32_t queue_type;             // 队列类型（AQL/PM4/SDMA）
    uint32_t gpu_id;                 // 目标GPU
    
    // 输出参数（内核填充）⭐
    uint32_t queue_id;               // 分配的队列ID
    uint64_t doorbell_offset;        // doorbell mmap偏移
};
```

**内核态 → 用户态**（通过ioctl返回）:
```c
args->queue_id = 123;           // 队列ID（用于destroy、update等操作）
args->doorbell_offset = 0xABC;  // doorbell地址（用于mmap）
```

### 3. MQD初始化 ⭐⭐⭐

**关键：MQD的字段直接来自ioctl参数**:

```c
// update_mqd() 中的映射关系
void update_mqd(struct v9_mqd *m, struct queue_properties *q) {
    // 用户态的ring-buffer地址 → MQD
    m->cp_hqd_pq_base_lo = lower_32_bits(q->queue_address);
    m->cp_hqd_pq_base_hi = upper_32_bits(q->queue_address);
    //   ↑↑↑ q->queue_address 来自 args->ring_base_address
    
    // 用户态的read/write指针地址 → MQD
    m->cp_hqd_pq_rptr_report_addr = (uint64_t)q->read_ptr;
    m->cp_hqd_pq_wptr_poll_addr = (uint64_t)q->write_ptr;
    //   ↑↑↑ 来自 args->read_pointer_address 和 write_pointer_address
    
    // 内核分配的doorbell → MQD
    m->cp_hqd_pq_doorbell_control = q->doorbell_off << SHIFT;
    //   ↑↑↑ 内核分配的doorbell_id，返回给用户态
    
    // 内核分配的CWSR区域 → MQD ⭐⭐⭐
    m->cp_hqd_ctx_save_base_addr = q->ctx_save_restore_area_address;
    //   ↑↑↑ 内核分配的2MB保存区，抢占时用
}
```

### 4. 资源分配责任

| 资源 | 分配方 | 时机 | 用途 |
|------|---------|------|------|
| ring-buffer | 用户态 | ioctl调用前 | 存放PM4/AQL命令 |
| read_ptr | 用户态 | ioctl调用前 | GPU更新读位置 |
| write_ptr | 用户态 | ioctl调用前 | 用户更新写位置 |
| MQD | 内核 | ioctl处理中 | 队列配置描述符 |
| doorbell | 内核 | ioctl处理中 | 通知硬件的寄存器 |
| CWSR区域 | 内核 | ioctl处理中 | Wave状态保存（抢占用）⭐ |

**关键区别**:
- **数据通道**（ring-buffer）: 用户态分配，用户态可读写
- **元数据配置**（MQD）: 内核分配，硬件读取
- **抢占资源**（CWSR）: 内核分配，硬件在抢占时自动使用

---

## 🎯 POC实施关键

### 理解调用时机的意义

**对POC的启示**:

1. **队列创建是一次性的** ⭐⭐⭐
   ```
   每个Stream/Queue只调用一次 KFD_IOC_CREATE_QUEUE
   - ring-buffer、MQD、CWSR区域在队列生命周期内持久存在
   - 抢占只是unmap/map，不需要重新创建队列
   ```

2. **优先级在创建时设置** ⭐⭐⭐
   ```
   args->queue_priority = 15;  // Online-AI
   args->queue_priority = 2;   // Offline-AI
   
   → 写入MQD.cp_hqd_priority
   → HWS根据这个优先级调度
   ```

3. **CWSR区域自动分配** ⭐⭐⭐
   ```
   POC不需要手动管理CWSR：
   ✓ 内核在create_queue时自动分配
   ✓ 写入MQD.cp_hqd_ctx_save_addr
   ✓ 硬件抢占时自动使用
   
   POC只需要：
   ✓ 确保 cwsr_enabled = true
   ✓ 调用 suspend_queues() 触发抢占
   ```

4. **Doorbell地址不可修改** ⭐⭐
   ```
   doorbell在create_queue时分配，整个生命周期固定
   - 用户态只能写doorbell值（通知新命令）
   - 不能改变doorbell地址映射
   ```

### POC不需要关心的细节

```
❌ 不需要自己创建ring-buffer（HIP Runtime处理）
❌ 不需要自己分配MQD（内核自动处理）
❌ 不需要自己管理CWSR区域（内核自动处理）
❌ 不需要自己发送runlist（内核DQM自动处理）

✅ POC只需要：
   1. 在创建队列时设置合适的优先级
   2. 调用 suspend_queues() API 触发抢占
   3. 验证抢占效果（通过时延测量）
```

---

## 📝 代码验证方法

### 验证ioctl调用时机

```bash
# 方法1: 使用strace跟踪ioctl调用
strace -e ioctl -f python your_hip_program.py 2>&1 | grep CREATE_QUEUE

# 输出示例:
# ioctl(3, AMDKFD_IOC_CREATE_QUEUE, {gpu_id=0, ring_base_address=0x7f1234000000, ...}) = 0

# 方法2: 启用KFD trace_printk
echo 1 > /sys/kernel/debug/tracing/events/kfd/enable
cat /sys/kernel/debug/tracing/trace_pipe | grep CREATE_QUEUE

# 方法3: 查看dmesg（如果开启pr_debug）
dmesg | grep "Creating queue"
```

### 验证MQD内容

```bash
# 查看创建的MQD
sudo cat /sys/kernel/debug/kfd/mqds

# 输出示例:
# Process 12345, Queue 0:
#   cp_hqd_pq_base: 0x00001234_00000000  ← ring-buffer地址
#   cp_hqd_pq_doorbell: 0x00005000       ← doorbell偏移
#   cp_hqd_ctx_save_base: 0x00007890_00000000  ← CWSR区域
```

---

## 📚 相关文档

- `AQL_QUEUE_VS_MQD_RELATIONSHIP.md` - AQL Queue与MQD的关系
- `New_MAP_UNMAP_DETAILED_PROCESS.md` - Map/Unmap详细流程
- `New_DEEP_DIVE_MI308X_QUEUE_MECHANISMS.md` - 队列机制深度分析

---

## 🔗 代码参考

**关键文件**:
- `kfd_chardev.c:311` - kfd_ioctl_create_queue()实现
- `kfd_chardev.c:190` - set_queue_properties_from_user()
- `kfd_process_queue_manager.c` - pqm_create_queue()实现
- `kfd_device_queue_manager.c` - create_queue_cpsch()实现
- `kfd_mqd_manager_v9.c:290` - update_mqd()实现

**用户态代码**（开源）:
- ROCm/ROCR-Runtime - hsa_queue_create()实现
- ROCm/ROCT-Thunk-Interface - hsaKmtCreateQueue()实现

---

**最后更新**: 2026-02-04  
**验证状态**: ✅ 基于内核代码分析  
**适用平台**: MI308X (CPSCH模式)
