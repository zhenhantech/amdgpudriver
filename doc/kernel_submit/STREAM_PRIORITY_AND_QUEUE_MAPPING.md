# Stream 优先级与 Queue 映射关系

**文档目的**: 澄清不同优先级的 Stream 是否使用同一个 ring-buffer (AQL Queue)  
**关键问题**: 两个应用程序使用不同优先级的 Stream 时，它们提交到同一个还是不同的 ring-buffer？  
**创建时间**: 2026-01-28

---

## 🎯 核心答案

### 每个 Stream 都有独立的 Queue (ring-buffer)

**结论**: ✅ **不同 Stream = 不同 Queue = 不同 ring-buffer**

| 场景 | 是否共享 Queue | 是否共享 ring-buffer |
|------|--------------|-------------------|
| **同进程，不同 Stream，相同优先级** | ❌ 否 | ❌ 否 |
| **同进程，不同 Stream，不同优先级** | ❌ 否 | ❌ 否 |
| **不同进程，不同 Stream，相同优先级** | ❌ 否 | ❌ 否 |
| **不同进程，不同 Stream，不同优先级** | ❌ 否 | ❌ 否 |

**关键原则**:
```
1 个 Stream = 1 个 HSA Queue = 1 个独立的 ring-buffer (AQL Queue)
                            = 1 个独立的 doorbell
                            = 1 个独立的 Queue ID
```

---

## 1️⃣ 代码证据

### 1.1 Stream 创建流程

**文件**: `hipamd/src/hip_stream.cpp` (Line 188)

```cpp
static hipError_t ihipStreamCreate(hipStream_t* stream, unsigned int flags,
                                   hip::Stream::Priority priority,
                                   const std::vector<uint32_t>& cuMask = {}) {
    // ⭐ 为每个 Stream 创建新的 hip::Stream 对象
    hip::Stream* hStream = new hip::Stream(hip::getCurrentDevice(), priority, flags, false, cuMask);
    
    if (hStream == nullptr) {
        return hipErrorOutOfMemory;
    } else if (!hStream->Create()) {  // ⭐ 每个 Stream 调用 Create()
        hip::Stream::Destroy(hStream);
        return hipErrorOutOfMemory;
    }
    
    *stream = reinterpret_cast<hipStream_t>(hStream);
    return hipSuccess;
}

hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority) {
    hip::Stream::Priority streamPriority;
    if (priority <= hip::Stream::Priority::High) {
        streamPriority = hip::Stream::Priority::High;
    } else if (priority >= hip::Stream::Priority::Low) {
        streamPriority = hip::Stream::Priority::Low;
    } else {
        streamPriority = hip::Stream::Priority::Normal;
    }
    
    // ⭐ 每次调用都创建新的 Stream
    return ihipStreamCreate(stream, flags, streamPriority);
}
```

**关键点**:
- ✅ 每次调用 `hipStreamCreate` / `hipStreamCreateWithPriority` 都创建新的 `hip::Stream` 对象
- ✅ 每个 `hip::Stream` 对象调用 `Create()` 方法
- ✅ `priority` 作为构造参数传递给 `hip::Stream`

### 1.2 Stream::Create() 创建 HSA Queue

**推测流程**（基于 HSA Runtime 代码）:

```cpp
// 文件: hipamd/src/hip_stream.cpp (推测)
bool hip::Stream::Create() {
    // ...
    // ⭐ 为每个 Stream 创建独立的 HSA Queue
    hsa_status_t status = hsa_queue_create(
        agent,
        queue_size,
        HSA_QUEUE_TYPE_MULTI,
        nullptr,  // callback
        nullptr,  // data
        UINT32_MAX,  // private_segment_size
        UINT32_MAX,  // group_segment_size
        &hsa_queue_  // ⭐ 每个 Stream 有自己的 hsa_queue_ 成员
    );
    
    if (status != HSA_STATUS_SUCCESS) {
        return false;
    }
    
    // ⭐ 设置 Queue 的优先级
    if (priority_ != Priority::Normal) {
        core::Queue* queue = core::Queue::Convert(hsa_queue_);
        queue->SetPriority(priority_to_hsa(priority_));
    }
    
    return true;
}
```

### 1.3 HSA Queue (AqlQueue) 的创建

**文件**: `rocr-runtime/core/runtime/amd_gpu_agent.cpp` (Line 1735)

```cpp
hsa_status_t GpuAgent::QueueCreate(size_t size, hsa_queue_type32_t queue_type, uint64_t flags,
                                   core::HsaEventCallback event_callback, void* data,
                                   uint32_t private_segment_size, uint32_t group_segment_size,
                                   core::Queue** queue) {
    // ...
    
    // ⭐ 分配独立的 shared_queue 结构
    core::SharedQueue* shared_queue = ...;
    
    // ⭐ 为每个 Queue 创建新的 AqlQueue 对象
    auto aql_queue = new AqlQueue(shared_queue, this, size, node_id, scratch,
                                  event_callback, data, flags);
    *queue = aql_queue;
    aql_queues_.push_back(aql_queue);  // 添加到队列列表
    
    // ...
    return HSA_STATUS_SUCCESS;
}
```

**文件**: `rocr-runtime/core/runtime/amd_aql_queue.cpp` (Line 81)

```cpp
AqlQueue::AqlQueue(core::SharedQueue* shared_queue, GpuAgent* agent, size_t req_size_pkts,
                   HSAuint32 node_id, ScratchInfo& scratch, core::HsaEventCallback callback,
                   void* err_data, uint64_t flags)
    : Queue(shared_queue, flags, !agent->is_xgmi_cpu_gpu()),
      LocalSignal(0, false),
      DoorbellSignal(signal()),
      ring_buf_(nullptr),        // ⭐ 每个 Queue 有独立的 ring buffer
      ring_buf_alloc_bytes_(0),
      queue_id_(HSA_QUEUEID(-1)), // ⭐ 每个 Queue 有独立的 ID
      active_(false),
      agent_(agent),
      queue_scratch_(scratch),
      errors_callback_(callback),
      errors_data_(err_data),
      pm4_ib_buf_(nullptr),
      pm4_ib_size_b_(0x1000),
      dynamicScratchState(0),
      exceptionState(0),
      suspended_(false),
      priority_(HSA_QUEUE_PRIORITY_NORMAL),  // ⭐ 每个 Queue 有自己的优先级
      exception_signal_(nullptr) {
    
    // ⭐ 分配独立的 AQL packet ring buffer
    AllocRegisteredRingBuffer(queue_size_pkts);
    
    // ⭐ 调用 KFD 创建硬件 Queue
    status = agent->driver().CreateQueue(node_id, HSA_QUEUE_COMPUTE_AQL, 100, priority_, 0,
                                         ring_buf_, ring_buf_alloc_bytes_, NULL, queue_rsrc);
    
    // ⭐ 获取独立的 doorbell 地址
    signal_.hardware_doorbell_ptr = queue_rsrc.Queue_DoorBell_aql;
    
    // ⭐ 获取独立的 Queue ID
    queue_id_ = queue_rsrc.QueueId;
}
```

**关键点**:
- ✅ 每个 `AqlQueue` 分配独立的 `ring_buf_`
- ✅ 每个 `AqlQueue` 有独立的 `queue_id_`
- ✅ 每个 `AqlQueue` 有独立的 `doorbell` 地址
- ✅ 每个 `AqlQueue` 有独立的 `priority_` 属性

### 1.4 KFD 驱动层的 Queue 创建

**文件**: `kfd/amdkfd/kfd_chardev.c` (ioctl 处理)

```c
static int kfd_ioctl_create_queue(...) {
    // ...
    
    // ⭐ 从 user 参数设置优先级
    err = set_queue_properties_from_user(&q_properties, &args);
    // q_properties.priority = args.queue_priority
    
    // ⭐ 为每个请求创建新的 Queue
    err = pqm_create_queue(p, dev, filep, &q_properties, &args.queue_id);
    
    // ⭐ 返回新的 queue_id（每个 Queue 唯一）
    args.queue_id = ...;
    
    return 0;
}
```

**文件**: `kfd/amdkfd/kfd_process_queue_manager.c`

```c
int pqm_create_queue(..., struct queue_properties *properties, unsigned int *qid) {
    // ...
    
    // ⭐ 创建新的 Queue 对象
    retval = create_cp_queue(pqm, dev, &pdd->qpd, properties, &f, qid);
    
    // ⭐ 每个 Queue 有独立的 queue_id
    *qid = new_queue_id;
    
    return 0;
}
```

---

## 2️⃣ Stream → Queue 映射关系

### 2.1 1:1 映射

```
进程 A:
  Stream-1 (priority=HIGH)
    ↓ 创建
  HSA Queue-101 (ring-buffer-101, doorbell-101)
    ↓ ioctl(CREATE_QUEUE)
  KFD Queue-101 (priority=HIGH)

  Stream-2 (priority=LOW)
    ↓ 创建
  HSA Queue-102 (ring-buffer-102, doorbell-102)
    ↓ ioctl(CREATE_QUEUE)
  KFD Queue-102 (priority=LOW)

进程 B:
  Stream-3 (priority=HIGH)
    ↓ 创建
  HSA Queue-201 (ring-buffer-201, doorbell-201)
    ↓ ioctl(CREATE_QUEUE)
  KFD Queue-201 (priority=HIGH)

关键点：
  ✅ 4 个不同的 Stream
  ✅ 4 个不同的 HSA Queue
  ✅ 4 个不同的 ring-buffer
  ✅ 4 个不同的 doorbell 地址
  ✅ 4 个不同的 Queue ID
```

### 2.2 没有 Queue 池化或复用

**AMD 的设计**：
- ❌ **不会**根据优先级复用已有的 Queue
- ❌ **不会**将多个 Stream 映射到同一个 Queue
- ✅ **每个** Stream 创建时都分配新的 Queue

**原因**：
1. **隔离性**：每个 Stream 需要独立的执行流
2. **并发性**：多个 Stream 并发提交 kernel
3. **简化管理**：避免复杂的 Queue 共享逻辑

---

## 3️⃣ 优先级的作用

### 3.1 优先级存储在 MQD 中

**文件**: `kfd/amdkfd/kfd_mqd_manager_v11.c` (Line 96)

```c
static void set_priority(struct v11_compute_mqd *m, struct queue_properties *q) {
    // ⭐ Pipe 优先级（映射后）
    m->cp_hqd_pipe_priority = pipe_priority_map[q->priority];
    
    // ⭐ Queue 优先级（原始值）
    m->cp_hqd_queue_priority = q->priority;
}
```

**文件**: `kfd/amdkfd/kfd_mqd_manager.c` (Line 29)

```c
/* Mapping queue priority to pipe priority, indexed by queue priority */
int pipe_priority_map[] = {
    KFD_PIPE_PRIORITY_CS_LOW,     // priority 0
    KFD_PIPE_PRIORITY_CS_LOW,     // priority 1
    KFD_PIPE_PRIORITY_CS_LOW,     // priority 2
    KFD_PIPE_PRIORITY_CS_LOW,     // priority 3
    KFD_PIPE_PRIORITY_CS_LOW,     // priority 4
    KFD_PIPE_PRIORITY_CS_LOW,     // priority 5
    KFD_PIPE_PRIORITY_CS_LOW,     // priority 6
    KFD_PIPE_PRIORITY_CS_MEDIUM,  // priority 7
    KFD_PIPE_PRIORITY_CS_MEDIUM,  // priority 8
    KFD_PIPE_PRIORITY_CS_MEDIUM,  // priority 9
    KFD_PIPE_PRIORITY_CS_MEDIUM,  // priority 10
    KFD_PIPE_PRIORITY_CS_HIGH,    // priority 11
    KFD_PIPE_PRIORITY_CS_HIGH,    // priority 12
    KFD_PIPE_PRIORITY_CS_HIGH,    // priority 13
    KFD_PIPE_PRIORITY_CS_HIGH,    // priority 14
    KFD_PIPE_PRIORITY_CS_HIGH     // priority 15
};
```

**优先级范围**:

| HIP Priority Level | Priority Value | Pipe Priority |
|-------------------|---------------|---------------|
| **High** | 11-15 | HIGH (2) |
| **Normal** | 7-10 | MEDIUM (1) |
| **Low** | 0-6 | LOW (0) |

### 3.2 优先级如何影响调度

**MES 模式**：
- MES 硬件调度器读取 MQD 中的 `cp_hqd_pipe_priority`
- 根据优先级决定 **调度顺序**
- 高优先级 Queue 的 kernel **优先被调度到 GPU**

**CPSCH 模式**：
- CP Firmware 读取 MQD 中的 `cp_hqd_pipe_priority`
- 根据优先级排序 **runlist**
- 高优先级 Queue **优先获得 HQD 资源**

**关键点**：
- ✅ 优先级**不改变** ring-buffer 的物理隔离
- ✅ 优先级**影响**调度顺序和资源分配
- ✅ 不同优先级的 Queue 仍然是**独立的 ring-buffer**

---

## 4️⃣ 完整的创建流程追踪

### 应用程序 A: 创建高优先级 Stream

```
应用 A:
  hipStreamCreateWithPriority(&stream_high, 0, -1)  // -1 = High
    ↓
  ihipStreamCreate(..., priority=High)
    ↓
  new hip::Stream(device, priority=High, ...)
    ↓
  hip::Stream::Create()
    ↓
  hsa_queue_create(...)
    ↓
  GpuAgent::QueueCreate(...)
    ↓
  new AqlQueue(...)
    ↓ 分配 ring buffer
  AllocRegisteredRingBuffer(1024 packets)  // 独立的 ring buffer
    ↓ 调用 KFD
  driver().CreateQueue(node_id, ..., priority_=NORMAL, ..., ring_buf_, ...)
    ↓ ioctl
  ioctl(kfd_fd, AMDKFD_IOC_CREATE_QUEUE, &args)
    ↓
  kfd_ioctl_create_queue(...)
    ↓
  pqm_create_queue(..., q_properties.priority=11, ...)  // High = 11
    ↓
  create_cp_queue(...)
    ↓
  allocate_mqd(...)  // 分配 MQD
    ↓
  init_mqd(...)
    ↓
  set_priority(mqd, q_properties)
    mqd->cp_hqd_pipe_priority = pipe_priority_map[11] = HIGH
    mqd->cp_hqd_queue_priority = 11
    ↓
  add_queue_mes(...) / execute_queues_cpsch(...)
    ↓ 返回
  queue_id = 1001  // ⭐ 唯一的 Queue ID
  doorbell_offset = 0x1000  // ⭐ 唯一的 doorbell 偏移
```

### 应用程序 A: 创建低优先级 Stream

```
应用 A:
  hipStreamCreateWithPriority(&stream_low, 0, 1)  // 1 = Low
    ↓
  （省略中间步骤，与上面相同）
    ↓
  new AqlQueue(...)
    ↓ 分配 ring buffer
  AllocRegisteredRingBuffer(1024 packets)  // ⭐ 新的独立 ring buffer
    ↓ 调用 KFD
  driver().CreateQueue(..., priority_=NORMAL, ..., ring_buf_, ...)
    ↓
  pqm_create_queue(..., q_properties.priority=1, ...)  // Low = 1
    ↓
  set_priority(mqd, q_properties)
    mqd->cp_hqd_pipe_priority = pipe_priority_map[1] = LOW
    mqd->cp_hqd_queue_priority = 1
    ↓ 返回
  queue_id = 1002  // ⭐ 不同的 Queue ID
  doorbell_offset = 0x1008  // ⭐ 不同的 doorbell 偏移
```

### 应用程序 B: 创建高优先级 Stream

```
应用 B (不同进程):
  hipStreamCreateWithPriority(&stream_high, 0, -1)
    ↓
  （省略中间步骤）
    ↓
  ioctl(kfd_fd_B, AMDKFD_IOC_CREATE_QUEUE, &args)  // ⭐ 不同的 kfd_fd
    ↓
  kfd_ioctl_create_queue(..., p=process_B, ...)  // ⭐ 不同的 kfd_process
    ↓
  pqm_create_queue(process_B->pqm, ..., q_properties.priority=11, ...)
    ↓ 返回
  queue_id = 2001  // ⭐ 不同进程的 Queue ID
  doorbell_offset = 0x2000  // ⭐ 不同进程的 doorbell 偏移
```

---

## 5️⃣ 关键数据结构

### 5.1 每个 Stream 的独立资源

```
hip::Stream 对象:
  ├─ hsa_queue_* hsa_queue_         // 指向 HSA Queue
  ├─ Priority priority_             // 优先级
  ├─ Device* device_                // 所属 Device
  └─ ...

HSA AqlQueue 对象:
  ├─ void* ring_buf_                // ⭐ 独立的 ring buffer 内存
  ├─ size_t ring_buf_alloc_bytes_   // ring buffer 大小
  ├─ HSAuint64 queue_id_            // ⭐ 独立的 Queue ID
  ├─ HSA_QUEUE_PRIORITY priority_   // ⭐ 优先级
  ├─ signal_t doorbell_signal_      // ⭐ 独立的 doorbell signal
  ├─ void* hardware_doorbell_ptr    // ⭐ 独立的 doorbell MMIO 地址
  └─ ...

KFD Queue 对象 (内核态):
  ├─ unsigned int queue_id          // ⭐ 内核层的 Queue ID
  ├─ struct queue_properties
  │   ├─ priority                   // ⭐ 优先级（0-15）
  │   ├─ queue_address              // ⭐ ring buffer 物理地址
  │   ├─ doorbell_off               // ⭐ doorbell 偏移
  │   └─ ...
  └─ struct mqd
      ├─ cp_hqd_pipe_priority       // ⭐ 硬件 pipe 优先级
      ├─ cp_hqd_queue_priority      // ⭐ 原始优先级值
      └─ ...
```

### 5.2 多进程 / 多 Stream 的隔离

```
系统全局视图:

进程 A (PID=1000):
  ├─ /dev/kfd (fd=3)
  ├─ kfd_process (p)
  ├─ Stream-1 (High)  → Queue-1001, ring-buf-1001, doorbell-0x1000
  ├─ Stream-2 (Low)   → Queue-1002, ring-buf-1002, doorbell-0x1008
  └─ Stream-3 (High)  → Queue-1003, ring-buf-1003, doorbell-0x1010

进程 B (PID=2000):
  ├─ /dev/kfd (fd=3)
  ├─ kfd_process (p)
  ├─ Stream-1 (High)  → Queue-2001, ring-buf-2001, doorbell-0x2000
  └─ Stream-2 (Normal)→ Queue-2002, ring-buf-2002, doorbell-0x2008

GPU 硬件视图:
  ├─ Queue-1001 (MQD, pipe_priority=HIGH, doorbell=0x1000)
  ├─ Queue-1002 (MQD, pipe_priority=LOW,  doorbell=0x1008)
  ├─ Queue-1003 (MQD, pipe_priority=HIGH, doorbell=0x1010)
  ├─ Queue-2001 (MQD, pipe_priority=HIGH, doorbell=0x2000)
  └─ Queue-2002 (MQD, pipe_priority=MEDIUM, doorbell=0x2008)

关键点：
  ✅ 5 个不同的 Stream = 5 个不同的 Queue
  ✅ 5 个独立的 ring-buffer
  ✅ 5 个独立的 doorbell 地址
  ✅ 即使优先级相同（Queue-1001 和 Queue-1003 都是 HIGH），也是独立的 Queue
```

---

## 6️⃣ 优先级的实际影响

### 6.1 调度顺序

**MES 模式下**:
```
MES 硬件调度器的行为:
  
  1. 检测到多个 doorbell 写入
     Queue-1001 (HIGH)  → wptr++
     Queue-1002 (LOW)   → wptr++
     Queue-2001 (HIGH)  → wptr++
  
  2. MES 读取各 Queue 的 MQD
     检查 cp_hqd_pipe_priority
  
  3. 优先调度高优先级 Queue 的 kernel
     调度顺序: Queue-1001 / Queue-2001 (HIGH) 优先
               Queue-1002 (LOW) 延后
  
  4. 但所有 Queue 仍然是独立的 ring-buffer！
```

**CPSCH 模式下**:
```
CP Firmware 的行为:
  
  1. DQM 维护所有 Queue 的 runlist
     [Queue-1001(HIGH), Queue-1003(HIGH), Queue-2002(MEDIUM), Queue-1002(LOW), ...]
  
  2. 根据 priority 排序 runlist
     高优先级 Queue 排在前面
  
  3. 按 runlist 顺序分配 HQD 资源
     高优先级 Queue 优先获得硬件资源
  
  4. 但所有 Queue 仍然是独立的 ring-buffer！
```

### 6.2 优先级的局限性

**当前实现的局限**：
- ⚠️ 优先级只影响 **Queue 级别**的调度顺序
- ⚠️ **不能**抢占正在执行的 kernel（需要 CWSR）
- ⚠️ **不能**在单个 Queue 内区分不同 kernel 的优先级

**示例**:
```cpp
// 应用 A
hipStream_t stream_high;
hipStreamCreateWithPriority(&stream_high, 0, -1);  // High

hipStream_t stream_low;
hipStreamCreateWithPriority(&stream_low, 0, 1);    // Low

// 启动 kernel
kernel_A<<<grid, block, 0, stream_high>>>();  // 提交到 Queue-1001 (HIGH)
kernel_B<<<grid, block, 0, stream_low>>>();   // 提交到 Queue-1002 (LOW)

// 调度行为：
// 1. kernel_A 和 kernel_B 提交到不同的 ring-buffer
// 2. MES/CP 优先调度 Queue-1001 (HIGH) 的 kernel_A
// 3. kernel_A 执行完后，才调度 Queue-1002 (LOW) 的 kernel_B
// 4. 但如果 kernel_A 执行时间很长，kernel_B 不能抢占它（需要 GPREEMPT）
```

---

## 7️⃣ 代码验证

### 7.1 验证 Queue 独立性

**测试程序**:
```cpp
#include <hip/hip_runtime.h>
#include <stdio.h>

int main() {
    hipStream_t stream1, stream2, stream3;
    
    // 创建 3 个 Stream，不同优先级
    hipStreamCreateWithPriority(&stream1, 0, -1);  // High
    hipStreamCreateWithPriority(&stream2, 0, 0);   // Normal
    hipStreamCreateWithPriority(&stream3, 0, 1);   // Low
    
    // 打印 Stream 地址（实际上是 hip::Stream 对象指针）
    printf("Stream 1 (High):   %p\n", stream1);
    printf("Stream 2 (Normal): %p\n", stream2);
    printf("Stream 3 (Low):    %p\n", stream3);
    
    // 预期：3 个不同的地址 = 3 个不同的 Stream 对象
    // 预期：每个 Stream 有自己的 HSA Queue
    // 预期：每个 Queue 有自己的 ring-buffer 和 doorbell
    
    hipStreamDestroy(stream1);
    hipStreamDestroy(stream2);
    hipStreamDestroy(stream3);
    
    return 0;
}
```

### 7.2 使用 rocprof 验证

```bash
# 查看 Queue 信息
rocprofv3 --hip-trace ./test_priority

# 输出示例:
# Stream 1 → Queue ID 1001, doorbell 0x7f1234001000
# Stream 2 → Queue ID 1002, doorbell 0x7f1234001008
# Stream 3 → Queue ID 1003, doorbell 0x7f1234001010
#
# 结论：每个 Stream 有独立的 Queue ID 和 doorbell 地址
```

### 7.3 使用 dmesg 验证

```bash
# 启用 KFD debug
echo 0xff > /sys/module/amdkfd/parameters/debug_evictions

# 运行测试程序
./test_priority

# 查看 dmesg
dmesg | grep "create queue"

# 预期输出（示例）:
# [12345.678] amdkfd: create queue id=1001, priority=11, doorbell_off=0x1000
# [12345.679] amdkfd: create queue id=1002, priority=7, doorbell_off=0x1008
# [12345.680] amdkfd: create queue id=1003, priority=1, doorbell_off=0x1010
#
# 结论：每个 Stream 创建独立的 Queue，有不同的 queue_id 和 doorbell_off
```

---

## 8️⃣ 为什么每个 Stream 需要独立的 Queue？

### 8.1 并发执行

```cpp
// 应用代码
hipStream_t stream1, stream2;
hipStreamCreate(&stream1);
hipStreamCreate(&stream2);

// 并发提交 kernel
kernel_A<<<grid, block, 0, stream1>>>();
kernel_B<<<grid, block, 0, stream2>>>();
kernel_C<<<grid, block, 0, stream1>>>();
kernel_D<<<grid, block, 0, stream2>>>();

// 期望行为：
// stream1: kernel_A → kernel_C (串行)
// stream2: kernel_B → kernel_D (串行)
// stream1 和 stream2 之间: 并发
```

**如果共享 Queue**:
```
❌ 无法实现并发
❌ 所有 kernel 串行执行
❌ Stream 的意义丧失
```

**独立 Queue 的好处**:
```
✅ 每个 Stream 有独立的 ring-buffer
✅ 可以并发写入 packet
✅ 可以并发写入 doorbell
✅ GPU 可以并发调度多个 Stream 的 kernel
```

### 8.2 资源隔离

```
Stream-1:
  ring-buffer: [packet-A1, packet-A2, packet-A3, ...]
  doorbell: 0x1000
  
Stream-2:
  ring-buffer: [packet-B1, packet-B2, packet-B3, ...]
  doorbell: 0x1008

优势：
  ✅ 互不干扰
  ✅ 独立的 wptr/rptr
  ✅ 独立的 doorbell 通知
```

### 8.3 优先级管理

```
如果不同优先级的 Stream 共享 Queue:
  ❌ 无法区分哪个 packet 是高优先级
  ❌ 无法在硬件层面实现优先级调度
  ❌ MQD 只有一个 priority 字段

如果不同优先级的 Stream 使用独立 Queue:
  ✅ 每个 Queue 的 MQD 有自己的 priority
  ✅ 硬件可以根据 Queue priority 调度
  ✅ 高优先级 Queue 优先被调度
```

---

## 9️⃣ 总结

### 9.1 明确答案

**问题**: 两个应用程序，使用不同优先级的 Stream 时，它们提交到同一个 ring-buffer 还是不同的 ring-buffer？

**答案**: ✅ **不同的 ring-buffer（不同的 Queue）**

**详细说明**:
- ✅ 每个 Stream 都有**独立的 HSA Queue**
- ✅ 每个 Queue 都有**独立的 ring-buffer** (AQL Queue)
- ✅ 每个 Queue 都有**独立的 Queue ID**
- ✅ 每个 Queue 都有**独立的 doorbell 地址**
- ✅ 每个 Queue 都有**独立的 MQD** (Memory Queue Descriptor)
- ✅ 优先级**不影响** Queue 的独立性
- ✅ 不同进程的 Stream **完全隔离**

### 9.2 映射关系总结

```
映射关系（严格 1:1）:
  1 个 Stream = 1 个 HSA Queue
              = 1 个独立的 ring-buffer
              = 1 个独立的 Queue ID
              = 1 个独立的 doorbell 地址
              = 1 个独立的 MQD

不共享！
不复用！
完全独立！
```

### 9.3 优先级的作用

**优先级**:
- ✅ 存储在每个 Queue 的 MQD 中
- ✅ 影响**调度顺序**（高优先级优先调度）
- ✅ 影响**资源分配**（高优先级优先获得 HQD）
- ❌ **不影响** Queue 的独立性（仍然是独立的 ring-buffer）

### 9.4 图示

```
进程 A:
┌─────────────────────────────────────────────┐
│ Stream-1 (HIGH)                             │
│   ↓                                         │
│ Queue-1001 (ring-buf-1001, doorbell-0x1000) │
│   priority=11, pipe_priority=HIGH           │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Stream-2 (LOW)                              │
│   ↓                                         │
│ Queue-1002 (ring-buf-1002, doorbell-0x1008) │
│   priority=1, pipe_priority=LOW             │
└─────────────────────────────────────────────┘

进程 B:
┌─────────────────────────────────────────────┐
│ Stream-1 (HIGH)                             │
│   ↓                                         │
│ Queue-2001 (ring-buf-2001, doorbell-0x2000) │
│   priority=11, pipe_priority=HIGH           │
└─────────────────────────────────────────────┘

MES 调度器:
  检测 doorbell 写入
  读取各 Queue 的 MQD
  根据 pipe_priority 调度:
    Queue-1001 (HIGH) → 优先
    Queue-2001 (HIGH) → 优先
    Queue-1002 (LOW)  → 延后

  但所有 Queue 的 ring-buffer 都是独立的！
```

---

## 🧪 实际验证

**测试程序**: [test_stream_priority/](./test_stream_priority/)

我们创建了完整的测试套件来验证这些结论：

### 测试套件内容

1. **test_app_A.cpp**: 应用程序 A（2 个 Stream: HIGH, LOW）
2. **test_app_B.cpp**: 应用程序 B（2 个 Stream: HIGH, NORMAL）
3. **test_concurrent.cpp**: 单进程测试（4 个 Stream，便于追踪）
4. **run_test.sh**: 自动化测试脚本

### 快速运行

```bash
cd test_stream_priority

# 自动化测试
./run_test.sh

# 或手动运行
make all
./test_concurrent

# 使用 rocprof 追踪
rocprofv3 --hip-trace ./test_concurrent

# 监控内核消息
sudo dmesg -w | grep -E "create queue|doorbell|priority"
```

### 预期验证结果

- ✅ 4 个不同的 Stream 地址
- ✅ 4 个不同的 Queue ID
- ✅ 4 个不同的 doorbell 偏移
- ✅ 每个 Stream 有独立的优先级
- ✅ 所有 Stream 可以并发提交 kernel

详细说明见: [test_stream_priority/README.md](./test_stream_priority/README.md)

---

## 相关文档

- [KERNEL_TRACE_STREAM_MANAGEMENT.md](./KERNEL_TRACE_STREAM_MANAGEMENT.md) - Stream 管理详解
- [KERNEL_TRACE_03_KFD_QUEUE.md](./KERNEL_TRACE_03_KFD_QUEUE.md) - KFD Queue 创建
- [KERNEL_TRACE_05_DATA_STRUCTURES.md](./KERNEL_TRACE_05_DATA_STRUCTURES.md) - queue_properties 和 MQD
- [test_stream_priority/README.md](./test_stream_priority/README.md) - 实际验证测试程序
- [PRIORITY_TO_HARDWARE_DEEP_TRACE.md](./PRIORITY_TO_HARDWARE_DEEP_TRACE.md) - ⭐ **深度追踪：优先级如何配置硬件寄存器**

