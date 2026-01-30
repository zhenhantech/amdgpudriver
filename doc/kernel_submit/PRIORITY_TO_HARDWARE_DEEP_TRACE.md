# Stream 优先级到硬件配置深度追踪

**核心目标**: 追踪不同优先级的 Stream 如何配置不同的硬件寄存器，让 GPU 硬件根据优先级执行不同的调度策略

**创建时间**: 2026-01-29

**⚠️ 重要说明**: 本文档主要描述 MES (Micro-Engine Scheduler) 模式。如果您的 GPU 使用 CPSCH (CP Scheduler) 模式（如 MI308X），请同时参考 [PRIORITY_CPSCH_MODE_TRACE.md](./PRIORITY_CPSCH_MODE_TRACE.md)。**优先级处理的核心机制（MQD 配置）在两种模式下是相同的**。

---

## 🎯 关键发现总结

### 核心答案

**不同优先级的 Stream**:
- ✅ 使用 **不同的 ring-buffer 物理地址** (每个 Queue 独立)
- ✅ 使用 **不同的 doorbell 偏移地址** (每个 Queue 独立)
- ✅ 配置 **不同的 MQD 硬件寄存器** (优先级字段不同)
- ✅ 硬件根据 **`cp_hqd_pipe_priority`** 和 **`cp_hqd_queue_priority`** 寄存器做调度

### MQD 中的关键寄存器（优先级相关）

| 寄存器 | 作用 | 高优先级值 | 低优先级值 |
|-------|-----|-----------|-----------|
| **`cp_hqd_pipe_priority`** | 硬件 Pipe 优先级 | 2 (HIGH) | 0 (LOW) |
| **`cp_hqd_queue_priority`** | Queue 原始优先级 | 11-15 | 0-6 |
| **`cp_hqd_quantum`** | 时间片配置 | 相同 | 相同 |
| **`cp_hqd_pq_base`** | Ring Buffer 地址 | Queue-1 地址 | Queue-2 地址 |
| **`cp_hqd_pq_doorbell_control`** | Doorbell 偏移 | 0x1000 | 0x1008 |

---

## 📊 完整调用栈追踪

### Level 1: 应用层 - HIP API

```cpp
// 用户代码
hipStream_t stream_high, stream_low;

// 高优先级 Stream (-1 = HIGH)
hipStreamCreateWithPriority(&stream_high, 0, -1);

// 低优先级 Stream (1 = LOW)  
hipStreamCreateWithPriority(&stream_low, 0, 1);
```

**关键点**:
- 两个 Stream 有不同的 `priority` 参数
- 每个调用都会创建独立的 `hip::Stream` 对象

---

### Level 2: HIP Runtime 层 - Stream 创建

**文件**: `hipamd/src/hip_stream.cpp`

```cpp
// Line 299: hipStreamCreateWithPriority
hipError_t hipStreamCreateWithPriority(hipStream_t* stream, 
                                       unsigned int flags, 
                                       int priority) {
    // 映射用户优先级到内部优先级
    hip::Stream::Priority streamPriority;
    if (priority <= hip::Stream::Priority::High) {
        streamPriority = hip::Stream::Priority::High;      // priority = -1 → HIGH
    } else if (priority >= hip::Stream::Priority::Low) {
        streamPriority = hip::Stream::Priority::Low;       // priority = 1  → LOW
    } else {
        streamPriority = hip::Stream::Priority::Normal;    // priority = 0  → NORMAL
    }
    
    // 创建 Stream 对象（每个 Stream 独立）
    return ihipStreamCreate(stream, flags, streamPriority);
}

// Line 188: ihipStreamCreate
static hipError_t ihipStreamCreate(hipStream_t* stream, 
                                   unsigned int flags,
                                   hip::Stream::Priority priority, ...) {
    // ⭐ 为每个 Stream 创建新的对象
    hip::Stream* hStream = new hip::Stream(
        hip::getCurrentDevice(),  
        priority,                  // ⭐ 传递优先级
        flags, 
        false, 
        cuMask
    );
    
    // ⭐ 调用 Create() 创建底层 HSA Queue
    if (!hStream->Create()) {
        return hipErrorOutOfMemory;
    }
    
    *stream = reinterpret_cast<hipStream_t>(hStream);
    return hipSuccess;
}
```

**传递的数据**:
```
Stream-1 (HIGH):
  priority_ = hip::Stream::Priority::High
  hStream = 0x7f1234567890 (独立对象)

Stream-2 (LOW):
  priority_ = hip::Stream::Priority::Low
  hStream = 0x7f1234567a00 (独立对象)
```

---

### Level 3: HSA Runtime 层 - Queue 创建

**文件**: `rocr-runtime/core/runtime/amd_aql_queue.cpp`

```cpp
// Line 81: AqlQueue 构造函数
AqlQueue::AqlQueue(core::SharedQueue* shared_queue, 
                   GpuAgent* agent, 
                   size_t req_size_pkts,
                   HSAuint32 node_id, 
                   ScratchInfo& scratch, ...) 
    : priority_(HSA_QUEUE_PRIORITY_NORMAL),  // ⭐ 初始化优先级
      ring_buf_(nullptr),                    // ⭐ 独立的 ring buffer
      queue_id_(HSA_QUEUEID(-1)),           // ⭐ 独立的 Queue ID
      ... {
    
    // ⭐ 步骤 1: 分配独立的 ring buffer
    AllocRegisteredRingBuffer(queue_size_pkts);
    // ring_buf_ = 分配的内存地址（每个 Queue 不同）
    
    // ⭐ 步骤 2: 调用 KFD 创建 Queue
    status = agent->driver().CreateQueue(
        node_id, 
        HSA_QUEUE_COMPUTE_AQL, 
        100,           // percent (queue活跃度)
        priority_,     // ⭐ 优先级参数
        0,             // 
        ring_buf_,     // ⭐ ring buffer 地址
        ring_buf_alloc_bytes_, 
        NULL, 
        queue_rsrc     // ⭐ 返回的资源（包含 doorbell）
    );
    
    // ⭐ 步骤 3: 获取 doorbell 地址
    signal_.hardware_doorbell_ptr = queue_rsrc.Queue_DoorBell_aql;
    
    // ⭐ 步骤 4: 获取 Queue ID
    queue_id_ = queue_rsrc.QueueId;
}

// Line 634: SetPriority - 设置/更新优先级
hsa_status_t AqlQueue::SetPriority(HSA_QUEUE_PRIORITY priority) {
    if (suspended_) {
        return HSA_STATUS_ERROR_INVALID_QUEUE;
    }
    
    // ⭐ 更新内部优先级
    priority_ = priority;
    
    // ⭐ 调用 KFD 更新 Queue（会更新 MQD）
    auto err = agent_->driver().UpdateQueue(
        queue_id_, 
        100, 
        priority_,     // ⭐ 新的优先级
        ring_buf_,
        ring_buf_alloc_bytes_, 
        NULL
    );
    
    return (err == HSA_STATUS_SUCCESS ? HSA_STATUS_SUCCESS 
                                       : HSA_STATUS_ERROR_OUT_OF_RESOURCES);
}
```

**传递给 KFD 的数据**:
```
Queue-1 (HIGH):
  priority = HSA_QUEUE_PRIORITY_MAXIMUM (或对应的 HIGH 值)
  ring_buf = 0x7fabcd000000 (独立分配)
  queue_id = 1001 (KFD 分配)

Queue-2 (LOW):
  priority = HSA_QUEUE_PRIORITY_LOW
  ring_buf = 0x7fabce000000 (独立分配，不同地址)
  queue_id = 1002 (KFD 分配，不同 ID)
```

---

### Level 4: KFD Driver 层 - Queue 管理

**文件**: `kfd/amdkfd/kfd_chardev.c`

```c
// ioctl 处理
static int kfd_ioctl_create_queue(..., struct kfd_ioctl_create_queue_args *args) {
    struct queue_properties q_properties;
    
    // ⭐ 步骤 1: 从用户参数设置 queue_properties
    err = set_queue_properties_from_user(&q_properties, args);
    // q_properties.priority = args->queue_priority
    // q_properties.queue_address = ring_buf 的物理地址
    
    // ⭐ 步骤 2: 创建 Queue
    err = pqm_create_queue(
        p,                // kfd_process
        dev,              // kfd_node
        filep, 
        &q_properties,    // ⭐ 包含优先级
        &args->queue_id   // ⭐ 返回 Queue ID
    );
    
    // ⭐ 步骤 3: 分配 doorbell
    args->doorbell_offset = doorbell_off;  // ⭐ 每个 Queue 不同
    
    return 0;
}
```

**文件**: `kfd/amdkfd/kfd_process_queue_manager.c`

```c
int pqm_create_queue(..., struct queue_properties *properties, unsigned int *qid) {
    // ...
    
    // ⭐ 创建 Queue（会分配 MQD）
    retval = create_cp_queue(pqm, dev, &pdd->qpd, properties, &f, qid);
    
    return retval;
}

static int create_cp_queue(..., struct queue_properties *q_properties, ...) {
    // ...
    
    // ⭐ 关键: 调用 DQM 创建 Queue
    retval = dqm->ops.create_queue(
        dqm, 
        q, 
        &pdd->qpd,
        q_properties,    // ⭐ 包含优先级和 ring buffer 地址
        ...
    );
    
    return retval;
}
```

---

### Level 5: MQD Manager - 硬件寄存器配置 ⭐⭐⭐

**文件**: `kfd/amdkfd/kfd_mqd_manager_v11.c`

这是 **最关键** 的部分！这里配置所有硬件寄存器！

#### 5.1 MQD 初始化

```c
// Line 123: init_mqd - 初始化 MQD 结构
static void init_mqd(struct mqd_manager *mm, void **mqd,
                     struct kfd_mem_obj *mqd_mem_obj, 
                     uint64_t *gart_addr,
                     struct queue_properties *q) {
    struct v11_compute_mqd *m;
    
    // ⭐ MQD 是一个内存结构，包含所有硬件配置寄存器
    m = (struct v11_compute_mqd *) mqd_mem_obj->cpu_ptr;
    
    // 清零
    memset(m, 0, sizeof(struct v11_compute_mqd));
    
    // ═══════════════════════════════════════════════════════
    // 通用寄存器（所有 Queue 相同）
    // ═══════════════════════════════════════════════════════
    
    m->header = 0xC0310800;
    m->compute_pipelinestat_enable = 1;
    
    // CP 控制
    m->cp_hqd_pq_control = 5 << CP_HQD_PQ_CONTROL__RPTR_BLOCK_SIZE__SHIFT;
    m->cp_hqd_pq_control |= CP_HQD_PQ_CONTROL__UNORD_DISPATCH_MASK;
    
    // MQD 基地址（MQD 本身的位置）
    m->cp_mqd_base_addr_lo = lower_32_bits(addr);
    m->cp_mqd_base_addr_hi = upper_32_bits(addr);
    
    // ⭐ Quantum 配置（时间片）
    m->cp_hqd_quantum = 
        1 << CP_HQD_QUANTUM__QUANTUM_EN__SHIFT |
        1 << CP_HQD_QUANTUM__QUANTUM_SCALE__SHIFT |
        1 << CP_HQD_QUANTUM__QUANTUM_DURATION__SHIFT;
    
    // AQL 格式支持
    if (q->format == KFD_QUEUE_FORMAT_AQL)
        m->cp_hqd_aql_control = 1 << CP_HQD_AQL_CONTROL__CONTROL0__SHIFT;
    
    // CWSR (Context Save/Restore) 支持
    if (mm->dev->kfd->cwsr_enabled) {
        m->cp_hqd_persistent_state |=
            (1 << CP_HQD_PERSISTENT_STATE__QSWITCH_MODE__SHIFT);
        m->cp_hqd_ctx_save_base_addr_lo = 
            lower_32_bits(q->ctx_save_restore_area_address);
        m->cp_hqd_ctx_save_base_addr_hi = 
            upper_32_bits(q->ctx_save_restore_area_address);
        m->cp_hqd_ctx_save_size = q->ctx_save_restore_area_size;
    }
    
    // ⭐ 调用 update_mqd 设置 Queue 特定的寄存器
    mm->update_mqd(mm, m, q, NULL);
}
```

#### 5.2 Update MQD - 设置 Queue 特定寄存器

```c
// Line 222: update_mqd - 设置每个 Queue 的独立配置
static void update_mqd(struct mqd_manager *mm, void *mqd,
                       struct queue_properties *q,
                       struct mqd_update_info *minfo) {
    struct v11_compute_mqd *m;
    m = get_mqd(mqd);
    
    // ═══════════════════════════════════════════════════════
    // ⭐⭐⭐ Ring Buffer 配置（每个 Queue 不同）
    // ═══════════════════════════════════════════════════════
    
    // Ring Buffer 大小
    m->cp_hqd_pq_control &= ~CP_HQD_PQ_CONTROL__QUEUE_SIZE_MASK;
    m->cp_hqd_pq_control |=
        ffs(q->queue_size / sizeof(unsigned int)) - 1 - 1;
    
    // ⭐ Ring Buffer 基地址（每个 Queue 独立）
    m->cp_hqd_pq_base_lo = lower_32_bits((uint64_t)q->queue_address >> 8);
    m->cp_hqd_pq_base_hi = upper_32_bits((uint64_t)q->queue_address >> 8);
    
    // Read Pointer 地址
    m->cp_hqd_pq_rptr_report_addr_lo = lower_32_bits((uint64_t)q->read_ptr);
    m->cp_hqd_pq_rptr_report_addr_hi = upper_32_bits((uint64_t)q->read_ptr);
    
    // Write Pointer 地址
    m->cp_hqd_pq_wptr_poll_addr_lo = lower_32_bits((uint64_t)q->write_ptr);
    m->cp_hqd_pq_wptr_poll_addr_hi = upper_32_bits((uint64_t)q->write_ptr);
    
    // ═══════════════════════════════════════════════════════
    // ⭐⭐⭐ Doorbell 配置（每个 Queue 不同）
    // ═══════════════════════════════════════════════════════
    
    // ⭐ Doorbell 偏移（每个 Queue 独立）
    m->cp_hqd_pq_doorbell_control =
        q->doorbell_off << CP_HQD_PQ_DOORBELL_CONTROL__DOORBELL_OFFSET__SHIFT;
    
    pr_debug("cp_hqd_pq_doorbell_control 0x%x\n",
             m->cp_hqd_pq_doorbell_control);
    
    // ═══════════════════════════════════════════════════════
    // EOP (End of Pipe) Ring Buffer
    // ═══════════════════════════════════════════════════════
    
    m->cp_hqd_eop_control = min(0xA,
        ffs(q->eop_ring_buffer_size / sizeof(unsigned int)) - 1 - 1);
    m->cp_hqd_eop_base_addr_lo =
        lower_32_bits(q->eop_ring_buffer_address >> 8);
    m->cp_hqd_eop_base_addr_hi =
        upper_32_bits(q->eop_ring_buffer_address >> 8);
    
    // VMID
    m->cp_hqd_vmid = q->vmid;
    
    // ⭐ 调用 set_priority 设置优先级寄存器
    if (mm->set_priority)
        mm->set_priority(m, q);
}
```

#### 5.3 Set Priority - 配置优先级寄存器 ⭐⭐⭐

```c
// Line 96: set_priority - 设置优先级相关寄存器
static void set_priority(struct v11_compute_mqd *m, 
                        struct queue_properties *q) {
    // ═══════════════════════════════════════════════════════
    // ⭐⭐⭐ 硬件优先级寄存器（MES/CP 用于调度）
    // ═══════════════════════════════════════════════════════
    
    // ⭐ Pipe Priority（映射后的硬件优先级）
    // 这个字段直接被 MES/CP 硬件读取！
    m->cp_hqd_pipe_priority = pipe_priority_map[q->priority];
    
    // ⭐ Queue Priority（原始优先级值）
    m->cp_hqd_queue_priority = q->priority;
}
```

**优先级映射表** (`kfd/amdkfd/kfd_mqd_manager.c`):

```c
// Line 29: 优先级映射
int pipe_priority_map[] = {
    KFD_PIPE_PRIORITY_CS_LOW,     // priority 0  → LOW
    KFD_PIPE_PRIORITY_CS_LOW,     // priority 1  → LOW
    KFD_PIPE_PRIORITY_CS_LOW,     // priority 2  → LOW
    KFD_PIPE_PRIORITY_CS_LOW,     // priority 3  → LOW
    KFD_PIPE_PRIORITY_CS_LOW,     // priority 4  → LOW
    KFD_PIPE_PRIORITY_CS_LOW,     // priority 5  → LOW
    KFD_PIPE_PRIORITY_CS_LOW,     // priority 6  → LOW
    KFD_PIPE_PRIORITY_CS_MEDIUM,  // priority 7  → MEDIUM
    KFD_PIPE_PRIORITY_CS_MEDIUM,  // priority 8  → MEDIUM
    KFD_PIPE_PRIORITY_CS_MEDIUM,  // priority 9  → MEDIUM
    KFD_PIPE_PRIORITY_CS_MEDIUM,  // priority 10 → MEDIUM
    KFD_PIPE_PRIORITY_CS_HIGH,    // priority 11 → HIGH
    KFD_PIPE_PRIORITY_CS_HIGH,    // priority 12 → HIGH
    KFD_PIPE_PRIORITY_CS_HIGH,    // priority 13 → HIGH
    KFD_PIPE_PRIORITY_CS_HIGH,    // priority 14 → HIGH
    KFD_PIPE_PRIORITY_CS_HIGH     // priority 15 → HIGH
};
```

---

## 🔬 MQD 寄存器详解

### MQD (Memory Queue Descriptor) 结构

MQD 是一个 **内存中的数据结构**，包含了 GPU 硬件读取的所有 Queue 配置寄存器。

```c
struct v11_compute_mqd {
    // ═══════════════════════════════════════════════════════
    // 通用控制寄存器
    // ═══════════════════════════════════════════════════════
    uint32_t header;
    uint32_t compute_pipelinestat_enable;
    uint32_t compute_perfcount_enable;
    
    // ═══════════════════════════════════════════════════════
    // ⭐ Ring Buffer 相关寄存器（每个 Queue 不同）
    // ═══════════════════════════════════════════════════════
    uint32_t cp_hqd_pq_base_lo;           // ⭐ Ring buffer 基地址低 32 位
    uint32_t cp_hqd_pq_base_hi;           // ⭐ Ring buffer 基地址高 32 位
    uint32_t cp_hqd_pq_control;           // ⭐ Ring buffer 控制（大小等）
    
    uint32_t cp_hqd_pq_rptr_report_addr_lo;  // Read pointer 地址
    uint32_t cp_hqd_pq_rptr_report_addr_hi;
    
    uint32_t cp_hqd_pq_wptr_poll_addr_lo;    // Write pointer 地址
    uint32_t cp_hqd_pq_wptr_poll_addr_hi;
    
    // ═══════════════════════════════════════════════════════
    // ⭐ Doorbell 相关寄存器（每个 Queue 不同）
    // ═══════════════════════════════════════════════════════
    uint32_t cp_hqd_pq_doorbell_control;  // ⭐ Doorbell 偏移配置
    
    // ═══════════════════════════════════════════════════════
    // ⭐⭐⭐ 优先级寄存器（硬件调度的关键！）
    // ═══════════════════════════════════════════════════════
    uint32_t cp_hqd_pipe_priority;        // ⭐⭐⭐ 硬件 Pipe 优先级（0=LOW, 1=MEDIUM, 2=HIGH）
    uint32_t cp_hqd_queue_priority;       // ⭐⭐⭐ Queue 优先级（0-15）
    
    // ═══════════════════════════════════════════════════════
    // 时间片和调度相关
    // ═══════════════════════════════════════════════════════
    uint32_t cp_hqd_quantum;              // 时间片配置
    
    // ═══════════════════════════════════════════════════════
    // EOP, IB, VMID 等其他寄存器
    // ═══════════════════════════════════════════════════════
    uint32_t cp_hqd_eop_base_addr_lo;
    uint32_t cp_hqd_eop_base_addr_hi;
    uint32_t cp_hqd_eop_control;
    uint32_t cp_hqd_ib_control;
    uint32_t cp_hqd_vmid;
    
    // ═══════════════════════════════════════════════════════
    // CWSR (Context Save/Restore)
    // ═══════════════════════════════════════════════════════
    uint32_t cp_hqd_ctx_save_base_addr_lo;
    uint32_t cp_hqd_ctx_save_base_addr_hi;
    uint32_t cp_hqd_ctx_save_size;
    uint32_t cp_hqd_cntl_stack_size;
    uint32_t cp_hqd_cntl_stack_offset;
    uint32_t cp_hqd_wg_state_offset;
    
    // ... 更多寄存器
};
```

---

## 📊 两个不同优先级 Queue 的 MQD 对比

### 示例：High Priority vs Low Priority

```
═══════════════════════════════════════════════════════════════════════
Queue-1001 (HIGH Priority, priority=11)
═══════════════════════════════════════════════════════════════════════

MQD 寄存器配置:
  ┌─ Ring Buffer ────────────────────────────────────────────────────┐
  │ cp_hqd_pq_base_lo       = 0x12340000  ⭐ 独立的 ring buffer 地址  │
  │ cp_hqd_pq_base_hi       = 0x00007fab                            │
  │ cp_hqd_pq_control       = 0x00000205  (size=512 packets)        │
  │ cp_hqd_pq_rptr_report   = 0x...       (read ptr 地址)           │
  │ cp_hqd_pq_wptr_poll     = 0x...       (write ptr 地址)          │
  └──────────────────────────────────────────────────────────────────┘
  
  ┌─ Doorbell ───────────────────────────────────────────────────────┐
  │ cp_hqd_pq_doorbell_control = 0x00001000  ⭐ doorbell offset     │
  │   Doorbell Address: BAR + 0x1000                                │
  └──────────────────────────────────────────────────────────────────┘
  
  ┌─ Priority ⭐⭐⭐────────────────────────────────────────────────────┐
  │ cp_hqd_pipe_priority    = 2           ⭐⭐⭐ HIGH (硬件读这个！)   │
  │ cp_hqd_queue_priority   = 11          (原始 priority 值)        │
  └──────────────────────────────────────────────────────────────────┘
  
  ┌─ Quantum (时间片) ───────────────────────────────────────────────┐
  │ cp_hqd_quantum          = 0x00010101  (quantum enabled)         │
  └──────────────────────────────────────────────────────────────────┘
  
  ┌─ Other ──────────────────────────────────────────────────────────┐
  │ cp_hqd_vmid             = 1                                     │
  │ cp_hqd_eop_base         = 0x...       (EOP ring)                │
  │ cp_hqd_ctx_save_base    = 0x...       (CWSR area)               │
  └──────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════
Queue-1002 (LOW Priority, priority=1)
═══════════════════════════════════════════════════════════════════════

MQD 寄存器配置:
  ┌─ Ring Buffer ────────────────────────────────────────────────────┐
  │ cp_hqd_pq_base_lo       = 0x56780000  ⭐ 不同的 ring buffer 地址  │
  │ cp_hqd_pq_base_hi       = 0x00007fac                            │
  │ cp_hqd_pq_control       = 0x00000205  (size=512 packets)        │
  │ cp_hqd_pq_rptr_report   = 0x...       (不同的 read ptr 地址)    │
  │ cp_hqd_pq_wptr_poll     = 0x...       (不同的 write ptr 地址)   │
  └──────────────────────────────────────────────────────────────────┘
  
  ┌─ Doorbell ───────────────────────────────────────────────────────┐
  │ cp_hqd_pq_doorbell_control = 0x00001008  ⭐ 不同的 doorbell     │
  │   Doorbell Address: BAR + 0x1008                                │
  └──────────────────────────────────────────────────────────────────┘
  
  ┌─ Priority ⭐⭐⭐────────────────────────────────────────────────────┐
  │ cp_hqd_pipe_priority    = 0           ⭐⭐⭐ LOW (硬件读这个！)    │
  │ cp_hqd_queue_priority   = 1           (原始 priority 值)        │
  └──────────────────────────────────────────────────────────────────┘
  
  ┌─ Quantum (时间片) ───────────────────────────────────────────────┐
  │ cp_hqd_quantum          = 0x00010101  (quantum enabled)         │
  └──────────────────────────────────────────────────────────────────┘
  
  ┌─ Other ──────────────────────────────────────────────────────────┐
  │ cp_hqd_vmid             = 1                                     │
  │ cp_hqd_eop_base         = 0x...       (不同的 EOP ring)          │
  │ cp_hqd_ctx_save_base    = 0x...       (不同的 CWSR area)         │
  └──────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════
关键差异总结
═══════════════════════════════════════════════════════════════════════

寄存器                        Queue-1001 (HIGH)    Queue-1002 (LOW)
────────────────────────────────────────────────────────────────────
cp_hqd_pq_base             0x7fab12340000       0x7fac56780000  ⭐ 不同
cp_hqd_pq_doorbell_control 0x00001000           0x00001008      ⭐ 不同
cp_hqd_pipe_priority       2 (HIGH)             0 (LOW)         ⭐⭐⭐ 关键！
cp_hqd_queue_priority      11                   1               ⭐ 不同
cp_hqd_quantum             相同                  相同
```

---

## 🔧 硬件如何使用这些寄存器

### MES (Micro-Engine Scheduler) 调度流程

```
1. 用户空间写 Doorbell (MMIO Write)
   ├─ 写 doorbell_offset 0x1000  → Queue-1001 有新 packet
   └─ 写 doorbell_offset 0x1008  → Queue-1002 有新 packet

2. MES 硬件检测到 Doorbell 写入
   ├─ 读取 doorbell_offset → 知道是哪个 Queue
   └─ 查找对应的 MQD（Memory Queue Descriptor）

3. MES 读取 MQD 寄存器
   ├─ 读 cp_hqd_pq_base       → 知道 ring buffer 在哪里
   ├─ 读 cp_hqd_pq_wptr       → 知道 write pointer 位置
   ├─ 读 cp_hqd_pq_rptr       → 知道 read pointer 位置
   ├─ 读 cp_hqd_pipe_priority → ⭐⭐⭐ 知道这个 Queue 的优先级！
   └─ 读 cp_hqd_queue_priority

4. MES 根据优先级调度 ⭐⭐⭐
   ├─ Queue-1001: cp_hqd_pipe_priority = 2 (HIGH)
   ├─ Queue-1002: cp_hqd_pipe_priority = 0 (LOW)
   └─ 决策: 优先调度 Queue-1001

5. MES 从 Ring Buffer 读取 AQL Packet
   ├─ 使用 cp_hqd_pq_base + read_ptr 计算地址
   └─ 读取 AQL Dispatch Packet

6. MES 提交 Packet 到 CP (Command Processor)
   ├─ CP 分配 CU (Compute Unit)
   └─ 启动 Wavefront 执行

7. Packet 执行完成
   ├─ 更新 read pointer
   └─ 继续处理下一个 Packet
```

### 优先级如何影响调度

**MES 的调度逻辑**（简化版）:

```c
// 伪代码：MES 硬件的调度逻辑

while (true) {
    // 扫描所有有新 packet 的 Queue
    List<Queue> ready_queues = scan_doorbell_writes();
    
    if (ready_queues.empty()) {
        continue;  // 没有工作，等待
    }
    
    // ⭐ 根据优先级排序
    ready_queues.sort_by([](Queue q) {
        return q.mqd->cp_hqd_pipe_priority;  // ⭐⭐⭐ 读取 MQD 中的优先级
    });
    
    // 从最高优先级开始调度
    for (Queue q : ready_queues) {
        if (can_schedule(q)) {
            // 从 ring buffer 读取 packet
            packet = read_packet_from_ring(q.mqd->cp_hqd_pq_base, 
                                          q.mqd->cp_hqd_pq_rptr);
            
            // 提交到 CP
            submit_to_cp(packet);
            
            // 更新 read pointer
            q.mqd->cp_hqd_pq_rptr++;
            
            // 检查时间片
            if (quantum_expired(q)) {
                break;  // 切换到下一个 Queue
            }
        }
    }
}
```

**关键点**:
- ✅ MES 直接读取 `cp_hqd_pipe_priority` 寄存器
- ✅ 高优先级 Queue (priority=2) 优先被调度
- ✅ 低优先级 Queue (priority=0) 需要等待
- ✅ 即使低优先级 Queue 先提交，高优先级也会抢占

---

## 💡 关键洞察

### 1. Ring Buffer 和 Doorbell 的独立性

**每个 Queue 都有**:
- ✅ 独立的 `cp_hqd_pq_base` (ring buffer 基地址)
- ✅ 独立的 `cp_hqd_pq_doorbell_control` (doorbell 偏移)
- ✅ 独立的 `cp_hqd_pq_rptr` / `cp_hqd_pq_wptr` (读写指针)

**为什么需要独立**:
- 并发访问：多个 Stream 可以同时写入不同的 ring buffer
- 隔离性：一个 Queue 的 overflow 不会影响其他 Queue
- 性能：避免锁竞争

### 2. 优先级如何影响硬件行为

**关键寄存器**: `cp_hqd_pipe_priority`

**硬件行为差异**:

| 场景 | 高优先级 (priority=2) | 低优先级 (priority=0) |
|-----|---------------------|---------------------|
| **调度顺序** | 优先被调度 | 可能需要等待 |
| **时间片** | 可能获得更多时间片 | 可能获得更少时间片 |
| **抢占** | 可以抢占低优先级 Queue | 不能抢占高优先级 |
| **HQD 分配** | 优先获得 HQD 资源 | 可能需要等待 HQD 可用 |

### 3. MQD 作为硬件和软件的接口

```
Software (Driver)         Hardware (MES/CP)
     │                         │
     │  1. 写 MQD 内存          │
     ├───────────────────────→ │
     │  (配置寄存器)            │
     │                         │
     │                         │  2. 读取 MQD
     │                         ├──────────┐
     │                         │          │
     │                         │  ←───────┘
     │                         │  (获取配置)
     │                         │
     │  3. 写 Doorbell         │
     ├───────────────────────→ │
     │  (通知有新 packet)       │
     │                         │
     │                         │  4. 读 MQD 优先级
     │                         ├──────────┐
     │                         │          │
     │                         │  ←───────┘
     │                         │
     │                         │  5. 调度决策
     │                         │  (高优先级优先)
     │                         │
     │                         │  6. 从 ring buffer 读 packet
     │                         │  (使用 cp_hqd_pq_base)
     │                         │
     │                         │  7. 执行 kernel
```

---

## ⚠️ 重要发现：HSA Runtime 中的优先级被写死

### 代码位置

**文件**: `rocr-runtime/core/runtime/amd_aql_queue.cpp`  
**Line 100**: Queue 优先级被硬编码为 `HSA_QUEUE_PRIORITY_NORMAL`

```cpp
// Line ~100
AqlQueue::AqlQueue(...)
    : priority_(HSA_QUEUE_PRIORITY_NORMAL),  // ⚠️ 写死了！不管用户传什么值
      ring_buf_(nullptr),
      ...
```

### 问题说明

**当前行为**:
- 即使用户调用 `hipStreamCreateWithPriority(stream, 0, -1)` (HIGH)
- HSA Runtime 仍然创建 `priority = NORMAL` 的 Queue
- **优先级参数被忽略了！** ⚠️

**影响**:
- 所有 Queue 的 MQD 都会配置相同的优先级
- `cp_hqd_pipe_priority` 都是相同的值
- **硬件无法根据优先级调度！** ⚠️

### 修复方法

需要修改 `amd_aql_queue.cpp`，使其正确传递优先级参数：

```cpp
// 修改前 (Line ~100)
AqlQueue::AqlQueue(...)
    : priority_(HSA_QUEUE_PRIORITY_NORMAL),  // ❌ 错误：写死了
      ...

// 修改后
AqlQueue::AqlQueue(..., HSAint32 priority, ...)
    : priority_(priority),                     // ✅ 正确：使用传入的参数
      ...
```

**TODO**: 后续需要修改代码测试不同优先级的效果。详见下方"后续测试计划"。

---

## 🔍 验证方法

### 方法 1: 使用 debugfs 查看 MQD

```bash
# 查看所有 Queue 的 MQD
sudo cat /sys/kernel/debug/kfd/mqds

# ⚠️ 当前行为（优先级被写死）：
# Queue 1001:
#   cp_hqd_pq_base: 0x7fab12340000
#   cp_hqd_pipe_priority: 1  ← 都是 NORMAL！
#   cp_hqd_pq_doorbell_control: 0x1000
#
# Queue 1002:
#   cp_hqd_pq_base: 0x7fac56780000
#   cp_hqd_pipe_priority: 1  ← 都是 NORMAL！
#   cp_hqd_pq_doorbell_control: 0x1008

# ✅ 修复后的预期行为：
# Queue 1001:
#   cp_hqd_pipe_priority: 2  ← HIGH
#
# Queue 1002:
#   cp_hqd_pipe_priority: 0  ← LOW
```

### 方法 2: 添加 KFD Debug 打印

在 `kfd_mqd_manager_v11.c` 中添加：

```c
static void set_priority(struct v11_compute_mqd *m, struct queue_properties *q) {
    m->cp_hqd_pipe_priority = pipe_priority_map[q->priority];
    m->cp_hqd_queue_priority = q->priority;
    
    // ⭐ 添加 debug 打印
    pr_info("KFD: Set MQD priority - queue_priority=%u, pipe_priority=%u, "
            "pq_base=0x%llx, doorbell=0x%x\n",
            q->priority, 
            m->cp_hqd_pipe_priority,
            ((uint64_t)m->cp_hqd_pq_base_hi << 32) | m->cp_hqd_pq_base_lo,
            m->cp_hqd_pq_doorbell_control);
}
```

### 方法 3: 运行测试程序并查看 dmesg

```bash
# 运行测试
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/test_stream_priority
./test_concurrent

# 查看 dmesg（需要添加上面的 debug 打印）
sudo dmesg | grep "Set MQD priority"

# 预期输出：
# [12345.678] KFD: Set MQD priority - queue_priority=11, pipe_priority=2, 
#             pq_base=0x7fab12340000, doorbell=0x1000
# [12345.679] KFD: Set MQD priority - queue_priority=1, pipe_priority=0, 
#             pq_base=0x7fac56780000, doorbell=0x1008
```

---

## 📚 总结

### 核心发现

**问题**: 不同优先级的 Stream，ring-buffer 和 doorbell 有什么不同？有没有配置寄存器？

**答案**:

1. **Ring Buffer**: ✅ **完全不同**
   - 每个 Queue 有独立的物理地址
   - 通过 MQD 的 `cp_hqd_pq_base` 寄存器配置
   - 硬件读取此寄存器知道从哪里获取 packet

2. **Doorbell**: ✅ **完全不同**
   - 每个 Queue 有独立的 doorbell 偏移
   - 通过 MQD 的 `cp_hqd_pq_doorbell_control` 寄存器配置
   - 用户空间写不同的 doorbell 地址

3. **配置寄存器**: ✅ **有！而且很关键！**
   - `cp_hqd_pipe_priority`: ⭐⭐⭐ 硬件用于调度的优先级
   - `cp_hqd_queue_priority`: 原始优先级值
   - 这些寄存器在 MQD 中
   - MES/CP 硬件直接读取并据此调度

4. **硬件行为**: ✅ **根据优先级做不同 action**
   - 高优先级 Queue 优先被调度
   - 高优先级可能获得更多时间片
   - 高优先级优先获得 HQD 资源
   - 调度器直接读取 `cp_hqd_pipe_priority` 决策

### 调用栈总结

```
hipStreamCreateWithPriority(priority=-1 or 1)
  ↓
hip::Stream (priority=HIGH or LOW)
  ↓
AqlQueue::AqlQueue(priority=...)
  ├─ AllocRegisteredRingBuffer()  → ring_buf_ = 独立地址
  └─ driver().CreateQueue(priority, ring_buf)
      ↓
      ioctl(AMDKFD_IOC_CREATE_QUEUE)
        ↓
        pqm_create_queue(q_properties)
          ├─ q_properties.priority = priority
          ├─ q_properties.queue_address = ring_buf
          └─ q_properties.doorbell_off = 独立偏移
              ↓
              mqd_manager->init_mqd(q_properties)
                ↓
                update_mqd()
                  ├─ cp_hqd_pq_base = ring_buf 地址  ⭐
                  └─ cp_hqd_pq_doorbell_control = doorbell 偏移  ⭐
                      ↓
                      set_priority()
                        ├─ cp_hqd_pipe_priority = 映射后的优先级  ⭐⭐⭐
                        └─ cp_hqd_queue_priority = 原始优先级
                            ↓
                            MES 硬件读取 MQD
                            根据 cp_hqd_pipe_priority 调度  ⭐⭐⭐
```

---

## 🔧 后续测试计划

### Phase 1: 修复 HSA Runtime 优先级传递

**目标**: 让优先级参数真正生效

**文件**: `rocr-runtime/core/runtime/amd_aql_queue.cpp`

**修改步骤**:

1. 修改构造函数签名，接受 priority 参数
2. 将 `priority_(HSA_QUEUE_PRIORITY_NORMAL)` 改为 `priority_(priority)`
3. 确保 `SetPriority()` 函数正常工作
4. 重新编译 ROCm

**验证**: 
```bash
# 运行测试程序
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/test_stream_priority
./run_test_with_log.sh

# 查看 dmesg
sudo dmesg | grep "pipe_priority"

# 应该看到不同的优先级值 (2, 1, 0) 而不是都是 1
```

### Phase 2: 性能测试

**目标**: 验证优先级调度的实际效果

**测试场景**:
1. 高优先级 Stream + 低优先级 Stream
2. 高优先级 kernel 是否真的优先执行
3. 低优先级 kernel 是否会被延迟

**测试程序**: 需要创建一个更复杂的测试，包含：
- 长时间运行的低优先级 kernel
- 短时间运行的高优先级 kernel
- 测量执行延迟

### Phase 3: 多进程优先级测试

**目标**: 验证跨进程的优先级调度

**测试场景**:
- 进程 A: 高优先级 Stream
- 进程 B: 低优先级 Stream
- 验证进程 A 的 kernel 是否优先执行

---

**创建时间**: 2026-01-29  
**更新时间**: 2026-01-29  
**目的**: 深度追踪优先级如何配置硬件寄存器  
**重要发现**: ⚠️ HSA Runtime 中优先级被写死，需要修复  
**结论**: ✅ 不同优先级的 Queue 有不同的 ring-buffer、doorbell 和优先级寄存器，硬件据此执行不同的调度策略（**需要先修复 HSA Runtime 代码**）
