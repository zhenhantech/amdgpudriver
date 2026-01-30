# Kernel提交流程追踪 (2/5) - HSA Runtime层

**范围**: HSA Runtime的Queue管理和Doorbell机制  
**代码路径**: `ROCm_keyDriver/rocm-systems/projects/rocr-runtime/`  
**关键操作**: Queue创建、AQL Packet写入、Doorbell触发

---

## 📋 本层概述

HSA Runtime是ROCm的核心运行时库，负责：
1. AQL Queue的创建和管理
2. 与KFD驱动交互（通过/dev/kfd）
3. Doorbell寄存器的映射和写入
4. Completion Signal的管理

---

## 1️⃣ HSA Runtime初始化

### 1.1 HSA Runtime初始化时机

```cpp
// 应用首次调用HIP API时，会触发HSA初始化
hipGetDeviceCount(&count);  // 或 hipMalloc、hipInit等
    ↓
HIP Runtime内部调用
    ↓
hsa_init()  // HSA Runtime初始化
```

### 1.2 hsa_init() 实现

**文件**: `ROCm_keyDriver/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/runtime.cpp`

```cpp
hsa_status_t hsa_init() {
    // 1. 检查是否已经初始化
    if (g_runtime != nullptr) {
        return HSA_STATUS_SUCCESS;
    }
    
    // 2. 创建全局Runtime对象
    g_runtime = new Runtime();
    
    // 3. 加载驱动和设备
    hsa_status_t status = g_runtime->Load();
    if (status != HSA_STATUS_SUCCESS) {
        delete g_runtime;
        g_runtime = nullptr;
        return status;
    }
    
    return HSA_STATUS_SUCCESS;
}
```

### 1.3 Runtime::Load() - 加载驱动

**文件**: `ROCm_keyDriver/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/runtime.cpp`

```cpp
hsa_status_t Runtime::Load() {
    // 1. 打开 /dev/kfd 设备文件
    // 这是与KFD驱动通信的关键！
    kfd_fd_ = open("/dev/kfd", O_RDWR | O_CLOEXEC);
    if (kfd_fd_ == -1) {
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
    
    // 2. 获取KFD版本信息
    struct kfd_ioctl_get_version_args args = {0};
    if (ioctl(kfd_fd_, AMDKFD_IOC_GET_VERSION, &args) != 0) {
        close(kfd_fd_);
        return HSA_STATUS_ERROR;
    }
    
    // 3. 枚举GPU设备
    DiscoverGpus();
    
    // 4. 为每个GPU创建Agent对象
    CreateAgents();
    
    return HSA_STATUS_SUCCESS;
}
```

**关键发现**:
- ✅ HSA Runtime会打开 `/dev/kfd` 设备文件
- ✅ 通过ioctl与KFD驱动通信
- ✅ 即使使用doorbell机制，也需要打开KFD

---

## 2️⃣ AQL Queue创建

### 2.1 Queue创建时机

```
应用首次使用GPU时：
  hipMalloc()
    ↓
  需要一个默认stream
    ↓
  Stream需要HSA queue
    ↓
  hsa_queue_create()  ← 在这里创建
```

### 2.2 hsa_queue_create() 入口

**文件**: `ROCm_keyDriver/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/hsa.cpp`

```cpp
hsa_status_t hsa_queue_create(
    hsa_agent_t agent,              // GPU agent
    uint32_t size,                  // Queue大小（必须是2的幂）
    hsa_queue_type32_t type,        // Queue类型
    void (*callback)(hsa_status_t status, hsa_queue_t* queue, void* data),
    void* data,
    uint32_t private_segment_size,
    uint32_t group_segment_size,
    hsa_queue_t** queue) {          // 返回的queue指针
    
    // 1. 验证参数
    if (queue == nullptr || agent.handle == 0) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    
    // 2. 获取Agent对象
    const core::Agent* agent_obj = core::Agent::Convert(agent);
    if (agent_obj == nullptr) {
        return HSA_STATUS_ERROR_INVALID_AGENT;
    }
    
    // 3. 调用Agent的QueueCreate方法
    core::Queue* queue_obj = nullptr;
    hsa_status_t status = agent_obj->QueueCreate(
        size, type, callback, data,
        private_segment_size, group_segment_size,
        &queue_obj);
    
    if (status != HSA_STATUS_SUCCESS) {
        return status;
    }
    
    // 4. 返回queue指针
    *queue = core::Queue::Convert(queue_obj);
    return HSA_STATUS_SUCCESS;
}
```

### 2.3 GpuAgent::QueueCreate() 实现

**文件**: `ROCm_keyDriver/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp`

```cpp
hsa_status_t GpuAgent::QueueCreate(
    size_t size,
    hsa_queue_type32_t queue_type,
    core::HsaEventCallback event_callback,
    void* data,
    uint32_t private_segment_size,
    uint32_t group_segment_size,
    core::Queue** queue) {
    
    // 1. 检查queue大小必须是2的幂
    if (!IsPowerOfTwo(size)) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    
    // 2. 检查大小范围
    if (size > maxAqlSize_ || size < minAqlSize_) {
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    
    // 3. 分配scratch内存（用于kernel的私有和组内存）
    ScratchInfo scratch;
    AllocateScratch(private_segment_size, group_segment_size, &scratch);
    
    // 4. 创建AQL Queue对象
    auto aql_queue = new AqlQueue(
        this,                    // GPU agent
        size,                    // Queue大小
        node_id(),              // NUMA节点ID
        scratch,                 // Scratch内存信息
        event_callback,          // 回调函数
        data,                   // 用户数据
        is_kv_device_);         // 是否是KV设备
    
    // 5. 添加到队列列表
    aql_queues_.push_back(aql_queue);
    
    *queue = aql_queue;
    return HSA_STATUS_SUCCESS;
}
```

### 2.4 AqlQueue构造函数

**文件**: `ROCm_keyDriver/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp`

```cpp
AqlQueue::AqlQueue(GpuAgent* agent, 
                   size_t req_size,
                   uint32_t node_id,
                   const ScratchInfo& scratch,
                   HsaEventCallback callback,
                   void* err_data,
                   bool is_kv) 
    : agent_(agent),
      queue_size_(req_size),
      is_active_(false) {
    
    // 1. 分配queue内存（用户空间）
    // 这个内存可以被GPU直接访问
    void* ring_buf = nullptr;
    amd::AllocSysMemory(req_size * sizeof(hsa_kernel_dispatch_packet_t),
                       &ring_buf);
    
    if (ring_buf == nullptr) {
        throw AMD::hsa_exception(HSA_STATUS_ERROR_OUT_OF_RESOURCES,
                                "Failed to allocate queue buffer");
    }
    
    // 清零queue内存
    memset(ring_buf, 0, req_size * sizeof(hsa_kernel_dispatch_packet_t));
    
    // 2. 设置AQL queue结构
    amd_queue_.base_address = (uint64_t)ring_buf;
    amd_queue_.size = req_size;
    amd_queue_.write_dispatch_id = 0;
    amd_queue_.read_dispatch_id = 0;
    
    // 3. 通过KFD创建硬件queue
    // 这是关键步骤！调用KFD驱动
    CreateHardwareQueue();
    
    // 4. 映射doorbell寄存器
    MapDoorbellRegister();
    
    // 5. 标记为活动状态
    is_active_ = true;
}
```

### 2.5 CreateHardwareQueue() - 调用KFD

**文件**: `ROCm_keyDriver/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp`

```cpp
void AqlQueue::CreateHardwareQueue() {
    // 准备KFD ioctl参数
    struct kfd_ioctl_create_queue_args args = {0};
    
    // 1. 设置queue类型
    args.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
    
    // 2. 设置queue地址和大小
    args.ring_base_address = (uint64_t)amd_queue_.base_address;
    args.ring_size = queue_size_;
    
    // 3. 设置其他参数
    args.gpu_id = agent_->node_id();
    args.queue_percentage = 100;  // Queue优先级
    args.queue_priority = HSA_QUEUE_PRIORITY_NORMAL;
    
    // 4. 调用KFD ioctl创建queue
    // 这是HSA Runtime与KFD驱动交互的关键！
    int ret = ioctl(agent_->kfd_fd(), 
                   AMDKFD_IOC_CREATE_QUEUE, 
                   &args);
    
    if (ret != 0) {
        throw AMD::hsa_exception(HSA_STATUS_ERROR,
                                "Failed to create KFD queue");
    }
    
    // 5. 保存KFD返回的信息
    queue_id_ = args.queue_id;            // KFD分配的queue ID
    doorbell_offset_ = args.doorbell_offset;  // Doorbell偏移
    
    // doorbell_offset用于后续映射doorbell寄存器
}

// ⭐⭐⭐ 关键问题：不同的 doorbell_offset 是否意味着不同的 hardware queue？
// 答案：是的！
```

### 2.6 Doorbell Offset 与 Hardware Queue 的关系 ⭐⭐⭐

#### 问题分析

**Q: 如果 doorbell_offset 不一样，可以认为是不同的 hardware queue 吗？**

**A: 是的！100% 可以这样判断。**

#### 代码证据

**1. Doorbell ID 的分配**（唯一性保证）

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_device_queue_manager.c:655-724`

```c
static int allocate_doorbell(struct qcm_process_device *qpd,
                             struct queue *q,
                             uint32_t const *restore_id) {
    struct kfd_node *dev = qpd->dqm->dev;
    
    // 对于 Compute Queues (SOC15+)
    if (restore_id) {
        // 恢复指定的 doorbell ID
        if (__test_and_set_bit(*restore_id, qpd->doorbell_bitmap))
            return -EINVAL;  // ID 已被占用
        q->doorbell_id = *restore_id;
    } else {
        // ⭐ 从 bitmap 中找一个空闲的 doorbell ID
        unsigned int found = find_first_zero_bit(qpd->doorbell_bitmap,
                                                 KFD_MAX_NUM_OF_QUEUES_PER_PROCESS);
        if (found >= KFD_MAX_NUM_OF_QUEUES_PER_PROCESS) {
            pr_debug("No doorbells available");
            return -EBUSY;  // 没有空闲的 doorbell
        }
        set_bit(found, qpd->doorbell_bitmap);  // 标记为已使用
        q->doorbell_id = found;  // ⭐ 分配唯一的 doorbell_id
    }
    
    // ⭐⭐ 基于 doorbell_id 计算物理偏移
    q->properties.doorbell_off = amdgpu_doorbell_index_on_bar(
        dev->adev,
        qpd->proc_doorbells,   // 进程的 doorbell BO
        q->doorbell_id,         // ⭐ 逻辑 doorbell ID
        dev->kfd->device_info.doorbell_size);
    
    return 0;
}
```

**2. Doorbell Offset 的计算**（一一映射）

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/amdgpu_doorbell_mgr.c:121-135`

```c
uint32_t amdgpu_doorbell_index_on_bar(struct amdgpu_device *adev,
                                      struct amdgpu_bo *db_bo,
                                      uint32_t doorbell_index,  // doorbell_id
                                      uint32_t db_size) {
    int db_bo_offset = amdgpu_bo_gpu_offset_no_check(db_bo);
    
    // ⭐⭐ 关键公式：
    // doorbell_offset = (db_bo_offset / 4) + doorbell_id * (db_size / 4)
    //                   ↑ 基地址偏移      ↑ 基于 ID 的偏移
    return db_bo_offset / sizeof(u32) + doorbell_index * DIV_ROUND_UP(db_size, 4);
}
```

#### 关键关系图

```
Queue 1:
  doorbell_id = 0 ──→ doorbell_offset = base + 0 * stride = 0x1000
                           ↓
                    MMIO 地址: BAR + 0x1000
                           ↓
                    GPU 硬件识别为 Queue 1

Queue 2:
  doorbell_id = 1 ──→ doorbell_offset = base + 1 * stride = 0x1008
                           ↓
                    MMIO 地址: BAR + 0x1008
                           ↓
                    GPU 硬件识别为 Queue 2

Queue 3:
  doorbell_id = 2 ──→ doorbell_offset = base + 2 * stride = 0x1010
                           ↓
                    MMIO 地址: BAR + 0x1010
                           ↓
                    GPU 硬件识别为 Queue 3
```

**stride** 通常是 8 字节（64-bit doorbell）或 4 字节（32-bit doorbell）

#### 判断依据总结

| 比较项 | 相同 | 不同 |
|--------|------|------|
| **doorbell_id** | 同一个 hardware queue | 不同的 hardware queue |
| **doorbell_offset** | 同一个 hardware queue | 不同的 hardware queue |
| **queue_id** | 同一个 queue（可能） | 可能是不同 queue |

**🎯 核心结论**:

1. ✅ **doorbell_offset 不同 → 100% 确定是不同的 hardware queue**
2. ✅ **doorbell_offset 由 doorbell_id 唯一确定**
3. ✅ **每个 queue 分配唯一的 doorbell_id（通过 bitmap 管理）**
4. ✅ **GPU 硬件通过监控不同的 doorbell MMIO 地址来区分 queue**
5. ⚠️ **queue_id 和 doorbell_id 可能不同**：
   - 旧架构（pre-SOC15）：`doorbell_id = queue_id`
   - 新架构（SOC15+）：`doorbell_id` 从 bitmap 独立分配

#### 实际验证方法

```bash
# 查看进程的所有 queue 的 doorbell 信息
sudo cat /sys/kernel/debug/dri/*/amdgpu_kfd_mqds

# 输出示例：
# Queue 0: doorbell_id=0  doorbell_offset=0x1000  queue_id=0
# Queue 1: doorbell_id=1  doorbell_offset=0x1008  queue_id=1
# Queue 2: doorbell_id=2  doorbell_offset=0x1010  queue_id=2
#         ↑ 不同           ↑ 不同                 ↑ 不同
#         → 这是 3 个不同的 hardware queue
```

#### 为什么这样设计？

1. **硬件识别**: GPU Command Processor 通过 doorbell 地址来识别是哪个 queue 被激活
2. **并发支持**: 不同的 queue 可以并发写入各自的 doorbell，互不干扰
3. **进程隔离**: 每个进程有独立的 doorbell_bitmap，防止冲突
4. **快速通知**: 用户空间直接写不同的 MMIO 地址，硬件立即识别

**🔍 调试技巧**:

```bash
# 创建多个 Stream 时观察 doorbell_offset
HIP_VISIBLE_DEVICES=0 your_program
# 在 KFD trace 中查看：
# [KFD-TRACE] CREATE_QUEUE: doorbell_offset=0x1000  # Stream 1
# [KFD-TRACE] CREATE_QUEUE: doorbell_offset=0x1008  # Stream 2
# [KFD-TRACE] CREATE_QUEUE: doorbell_offset=0x1010  # Stream 3
# → 三个不同的 doorbell_offset = 三个不同的 hardware queue
```
```

**关键ioctl结构**:
```c
struct kfd_ioctl_create_queue_args {
    uint64_t ring_base_address;    // Queue内存地址（用户空间）
    uint64_t write_pointer_address;// 写指针地址
    uint64_t read_pointer_address; // 读指针地址
    uint64_t doorbell_offset;      // OUT: Doorbell偏移（KFD返回）
    uint32_t ring_size;            // Queue大小
    uint32_t gpu_id;               // GPU ID
    uint32_t queue_type;           // Queue类型
    uint32_t queue_percentage;     // Queue优先级
    uint32_t queue_priority;       // 优先级级别
    uint64_t eop_buffer_address;   // End-of-pipe buffer地址
    uint64_t eop_buffer_size;      // EOP buffer大小
    uint64_t ctx_save_restore_address;  // Context保存恢复地址
    uint32_t ctx_save_restore_size;
    uint32_t ctl_stack_size;
    uint32_t queue_id;             // OUT: Queue ID（KFD返回）
};
```

---

## 3️⃣ Doorbell寄存器映射

### 3.1 MapDoorbellRegister() 实现

**文件**: `ROCm_keyDriver/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp`

```cpp
void AqlQueue::MapDoorbellRegister() {
    // 1. 计算doorbell地址
    // doorbell_offset是KFD返回的偏移值
    uint64_t doorbell_mmap_offset = doorbell_offset_;
    
    // 2. 通过mmap映射doorbell寄存器到用户空间
    // 这样用户空间可以直接写入doorbell！
    void* doorbell_ptr = mmap(
        NULL,                          // 让系统选择地址
        sizeof(uint64_t),              // 映射8字节（doorbell大小）
        PROT_READ | PROT_WRITE,       // 可读可写
        MAP_SHARED,                    // 共享映射
        agent_->kfd_fd(),             // KFD文件描述符
        doorbell_mmap_offset          // Doorbell偏移
    );
    
    if (doorbell_ptr == MAP_FAILED) {
        throw AMD::hsa_exception(HSA_STATUS_ERROR,
                                "Failed to map doorbell");
    }
    
    // 3. 保存doorbell地址
    doorbell_signal_.handle = (uint64_t)doorbell_ptr;
    
    // 4. 设置到queue结构中
    amd_queue_.doorbell_signal = doorbell_signal_;
}
```

**关键理解**:
- ✅ Doorbell是硬件寄存器，但被映射到用户空间
- ✅ 用户空间可以直接写入，无需系统调用
- ✅ 这就是doorbell机制低延迟的关键！

---

## 4️⃣ Kernel提交 - 写入AQL Packet

### 4.1 提交Packet的完整流程

**文件**: `ROCm_keyDriver/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp`

```cpp
uint64_t AqlQueue::AddWriteIndexAcqRel(uint64_t value) {
    // 原子增加写指针
    return __atomic_fetch_add(&amd_queue_.write_dispatch_id, 
                             value, 
                             __ATOMIC_ACQ_REL);
}

void AqlQueue::StoreRelaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    // 原子写入signal（doorbell）
    __atomic_store_n((uint64_t*)signal.handle, 
                     value, 
                     __ATOMIC_RELAXED);
}
```

### 4.2 完整的Packet提交代码

这部分在HIP Runtime层已经看到（上一章），这里再详细说明关键步骤：

```cpp
// 在 Stream::submitPacketToHsaQueue() 中

// 步骤1: 获取写指针位置（原子操作）
uint64_t write_index = queue->AddWriteIndexAcqRel(1);

// 步骤2: 计算packet在queue中的索引
const uint32_t queueMask = queue->size - 1;
uint32_t packet_index = write_index & queueMask;

// 步骤3: 获取packet地址
hsa_kernel_dispatch_packet_t* queue_packet = 
    &((hsa_kernel_dispatch_packet_t*)queue->base_address)[packet_index];

// 步骤4: 写入packet内容（除header外）
// 先写入所有字段
queue_packet->setup = packet->setup;
queue_packet->workgroup_size_x = packet->workgroup_size_x;
queue_packet->workgroup_size_y = packet->workgroup_size_y;
queue_packet->workgroup_size_z = packet->workgroup_size_z;
queue_packet->grid_size_x = packet->grid_size_x;
queue_packet->grid_size_y = packet->grid_size_y;
queue_packet->grid_size_z = packet->grid_size_z;
queue_packet->private_segment_size = packet->private_segment_size;
queue_packet->group_segment_size = packet->group_segment_size;
queue_packet->kernel_object = packet->kernel_object;
queue_packet->kernarg_address = packet->kernarg_address;
queue_packet->completion_signal = packet->completion_signal;

// 步骤5: 内存屏障（确保上面的写入对GPU可见）
__atomic_thread_fence(__ATOMIC_RELEASE);

// 步骤6: 最后写入header（激活packet）
// 使用原子操作，确保GPU看到完整的packet
__atomic_store_n(&queue_packet->header, 
                 packet->header, 
                 __ATOMIC_RELEASE);

// 步骤7: 写入doorbell（通知GPU）
// 这是最关键的一步！
queue->StoreRelaxed(queue->doorbell_signal, write_index);
```

### 4.3 Doorbell写入的底层实现

```cpp
void AqlQueue::StoreRelaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    // signal.handle 就是映射的doorbell寄存器地址
    // 直接写入即可，无需系统调用！
    volatile uint64_t* doorbell_ptr = (volatile uint64_t*)signal.handle;
    *doorbell_ptr = value;
    
    // 或者使用原子操作
    __atomic_store_n((uint64_t*)signal.handle, 
                     value, 
                     __ATOMIC_RELAXED);
}
```

**关键日志**（当设置 `AMD_LOG_LEVEL=5` 时）:

**文件**: `ROCm_keyDriver/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_kernel.cpp`

```cpp
// 在kernel提交时打印日志
void LogKernelSubmission(AqlQueue* queue, uint64_t write_index) {
    if (g_log_level >= 5) {
        fprintf(stderr, 
                ":amd_blit_kernel.cpp:1301: [***rocr***] "
                "HWq=%p, id=%u, Dispatch Header = 0x%x, "
                "rptr=%lu, wptr=%lu\n",
                (void*)queue->base_address,
                queue->queue_id,
                0x1402,  // Dispatch header (type=2)
                queue->read_dispatch_id,
                write_index);
    }
}
```

**日志示例**:
```
:amd_blit_kernel.cpp:1301: [***rocr***] HWq=0x7f40f14e4000, id=0, 
Dispatch Header = 0x1402, rptr=6, wptr=6
```

---

## 5️⃣ Completion Signal机制

### 5.1 Signal创建

```cpp
hsa_status_t hsa_signal_create(hsa_signal_value_t initial_value,
                               uint32_t num_consumers,
                               const hsa_agent_t* consumers,
                               hsa_signal_t* signal) {
    // 1. 分配signal内存
    SignalShared* signal_mem = AllocateSignalMemory();
    
    // 2. 初始化signal值
    signal_mem->value = initial_value;
    
    // 3. 返回signal handle
    signal->handle = (uint64_t)signal_mem;
    
    return HSA_STATUS_SUCCESS;
}
```

### 5.2 等待Signal完成

```cpp
hsa_signal_value_t hsa_signal_wait_acquire(hsa_signal_t signal,
                                            hsa_signal_condition_t condition,
                                            hsa_signal_value_t compare_value,
                                            uint64_t timeout_hint,
                                            hsa_wait_state_t wait_hint) {
    volatile hsa_signal_value_t* signal_ptr = 
        (volatile hsa_signal_value_t*)signal.handle;
    
    // 轮询等待signal值变化
    uint64_t start_time = GetCurrentTime();
    while (true) {
        hsa_signal_value_t current = *signal_ptr;
        
        // 检查条件是否满足
        if (CheckCondition(current, condition, compare_value)) {
            return current;
        }
        
        // 检查超时
        if (timeout_hint != UINT64_MAX) {
            if (GetCurrentTime() - start_time > timeout_hint) {
                return current;
            }
        }
        
        // CPU空转或休眠
        if (wait_hint == HSA_WAIT_STATE_ACTIVE) {
            _mm_pause();  // CPU pause指令
        } else {
            usleep(1);
        }
    }
}
```

---

## 6️⃣ 关键数据结构

### 6.1 AQL Queue结构（用户空间）

```cpp
// 在用户空间分配的queue结构
struct amd_queue_t {
    // HSA标准字段
    hsa_queue_type32_t type;          // Queue类型
    uint32_t features;                 // 特性标志
    hsa_signal_t doorbell_signal;      // Doorbell signal
    uint32_t size;                     // Queue大小
    uint32_t reserved1;
    uint64_t id;                       // Queue ID
    
    // 读写指针（在用户空间）
    volatile uint64_t write_dispatch_id;   // 写指针
    volatile uint64_t read_dispatch_id;    // 读指针
    
    // Queue内存
    uint64_t base_address;             // Packet数组基地址
    
    // AMD扩展字段
    volatile uint32_t* queue_properties;
    uint64_t reserved2[2];
};
```

### 6.2 KFD Queue创建参数

```c
// 传递给KFD驱动的参数
struct kfd_ioctl_create_queue_args {
    uint64_t ring_base_address;        // Queue内存地址
    uint64_t write_pointer_address;    // 写指针地址
    uint64_t read_pointer_address;     // 读指针地址
    uint64_t doorbell_offset;          // OUT: Doorbell偏移
    uint32_t ring_size;                // Queue大小
    uint32_t gpu_id;                   // GPU ID
    uint32_t queue_type;               // KFD_IOC_QUEUE_TYPE_COMPUTE_AQL
    uint32_t queue_percentage;         // 优先级
    uint32_t queue_priority;           // 优先级级别
    uint32_t queue_id;                 // OUT: Queue ID
    // ... 其他字段
};
```

---

## 7️⃣ 流程图

```
HSA Runtime初始化
  │
  │ hsa_init()
  ↓
Runtime::Load()
  │
  │ 1. open("/dev/kfd")  ← 打开KFD设备
  │ 2. ioctl(GET_VERSION)
  │ 3. 枚举GPU设备
  ↓
─────────────────────────────────────
Queue创建阶段
  │
  │ hsa_queue_create()
  ↓
GpuAgent::QueueCreate()
  │
  │ 1. 验证参数
  │ 2. 分配scratch内存
  ↓
AqlQueue::AqlQueue()
  │
  │ 1. 分配queue内存（用户空间）
  │ 2. 调用CreateHardwareQueue()
  ↓
CreateHardwareQueue()
  │
  │ ioctl(AMDKFD_IOC_CREATE_QUEUE)  ← 调用KFD驱动
  │ ↓
  │ KFD驱动处理（见下一章）
  │ ↓
  │ 返回: queue_id, doorbell_offset
  ↓
MapDoorbellRegister()
  │
  │ mmap(doorbell_offset)  ← 映射doorbell到用户空间
  ↓
─────────────────────────────────────
Kernel提交阶段
  │
  │ [来自HIP Runtime层]
  ↓
submitPacketToHsaQueue()
  │
  │ 1. 原子增加write_index
  │ 2. 计算packet位置
  │ 3. 写入packet内容
  │ 4. 内存屏障
  │ 5. 原子写入header
  │ 6. 写入doorbell  ← 直接写入映射的寄存器！
  ↓
GPU硬件检测doorbell更新
  ↓
[转到下一层: KFD驱动层]
```

---

## 8️⃣ 关键代码位置总结

| 功能 | 文件路径 | 关键函数/位置 |
|------|---------|-------------|
| HSA初始化 | `rocr-runtime/runtime/hsa-runtime/core/runtime/runtime.cpp` | `hsa_init()`, `Runtime::Load()` |
| 打开KFD | `rocr-runtime/runtime/hsa-runtime/core/runtime/runtime.cpp` | `open("/dev/kfd")` |
| Queue创建入口 | `rocr-runtime/runtime/hsa-runtime/core/runtime/hsa.cpp` | `hsa_queue_create()` |
| Queue创建实现 | `rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp` | `GpuAgent::QueueCreate()` |
| AQL Queue构造 | `rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp` | `AqlQueue::AqlQueue()` |
| KFD ioctl调用 | `rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp` | `CreateHardwareQueue()` |
| Doorbell映射 | `rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp` | `MapDoorbellRegister()` |
| Doorbell写入 | `rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp` | `StoreRelaxed()` |
| Kernel提交日志 | `rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_kernel.cpp` | 行1301附近 |

---

## 9️⃣ 关键发现

### 9.1 HSA Runtime与KFD的交互

```
HSA Runtime (用户空间)
    ↓ open("/dev/kfd")
KFD驱动 (内核空间)
    ↓ 返回文件描述符
HSA Runtime
    ↓ ioctl(CREATE_QUEUE)
KFD驱动
    ↓ 创建queue，返回doorbell_offset
HSA Runtime
    ↓ mmap(doorbell_offset)
用户空间可以直接写入doorbell！
```

### 9.2 为什么Doorbell机制快？

1. **Doorbell映射到用户空间**:
   - 通过mmap映射，用户空间可以直接访问
   - 不需要系统调用！

2. **直接写入硬件寄存器**:
   ```cpp
   *doorbell_ptr = write_index;  // 直接写入，无系统调用
   ```

3. **GPU直接检测**:
   - GPU硬件实时监控doorbell寄存器
   - 检测到更新立即处理

### 9.3 Queue创建vs Kernel提交

| 操作 | 频率 | 是否需要系统调用 | 性能影响 |
|------|------|---------------|---------|
| Queue创建 | 低（应用初始化时） | 是（ioctl） | 低 |
| Kernel提交 | 高（每次kernel启动） | 否（直接写doorbell） | 极低 |

---

## 🔟 下一步

在下一层（KFD驱动层），我们将看到：
- KFD如何处理CREATE_QUEUE ioctl
- Device Queue Manager的工作机制
- 如何与MES调度器交互

继续阅读: [KERNEL_TRACE_03_KFD_QUEUE.md](./KERNEL_TRACE_03_KFD_QUEUE.md)


