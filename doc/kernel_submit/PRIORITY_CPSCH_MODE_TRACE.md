# CPSCH 模式下的优先级处理机制

**适用场景**: 使用 CP Scheduler (CPSCH) 而非 MES 的 GPU (如 MI308X 使用 CPSCH 模式)

**创建时间**: 2026-01-29

---

## 🎯 关键区别：CPSCH vs MES

### 调度器类型

| 特性 | CPSCH (软件调度) | MES (硬件调度) |
|-----|-----------------|---------------|
| **全称** | Compute Process Scheduler | Micro-Engine Scheduler |
| **实现** | 驱动软件调度 | 硬件调度器 |
| **支持 GPU** | 较老架构 (GFX9/10/11) | 新架构 (GFX11高版本+) |
| **使用场景** | MI200/MI300 系列 | 高端 GFX11+ GPU |
| **MI308X** | ✅ **使用 CPSCH** | ❌ 不使用 |
| **调度方式** | PM4 Packet + Runlist | MES 直接读 MQD |
| **优先级支持** | ✅ 支持 (通过 MQD) | ✅ 支持 (通过 MQD) |

---

## 📊 CPSCH 模式下的完整流程

### Level 1-4: 与 MES 模式相同

前面的流程完全相同：
1. `hipStreamCreateWithPriority` → HIP Runtime
2. `hip::Stream` 创建 → HSA Runtime  
3. `AqlQueue` 创建 → KFD Driver
4. **MQD 配置**（与 MES 相同）:
   - `cp_hqd_pq_base` = ring buffer 地址
   - `cp_hqd_pq_doorbell_control` = doorbell 偏移
   - `cp_hqd_pipe_priority` = 优先级 ⭐⭐⭐
   - `cp_hqd_queue_priority` = 原始优先级

### Level 5: CPSCH 特有 - Runlist 提交 ⭐⭐⭐

**关键区别**: CPSCH 需要通过 **PM4 packet (MAP_QUEUES)** 显式告诉 CP 有哪些 Queue

```c
═══════════════════════════════════════════════════════════════════
文件: kfd_device_queue_manager.c
═══════════════════════════════════════════════════════════════════

// Line 2413: map_queues_cpsch - CPSCH 的核心函数
static int map_queues_cpsch(struct device_queue_manager *dqm)
{
    // ⭐ 步骤 1: 检查调度器状态
    if (!dqm->sched_running || dqm->sched_halt) {
        return 0;
    }
    
    if (dqm->active_queue_count <= 0 || dqm->processes_count <= 0) {
        return 0;
    }
    
    // ⭐ 步骤 2: 检查是否有正在处理的 runlist
    if (dqm->active_runlist) {
        // 有正在处理的 runlist，暂时不提交新的
        return 0;
    }
    
    // ⭐ 步骤 3: 构建 runlist（所有活跃 Queue 的列表）
    // 遍历所有进程的所有 Queue
    list_for_each_entry(cur, &dqm->queues, list) {
        qpd = cur->qpd;
        list_for_each_entry(q, &qpd->queues_list, list) {
            if (q->properties.is_active)
                runlist_size++;
                // 每个 Queue 都有自己的 MQD
                // MQD 包含 cp_hqd_pipe_priority ⭐⭐⭐
        }
    }
    
    // ⭐⭐⭐ 步骤 4: 发送 runlist 给 CP（通过 PM4 packet）
    retval = pm_send_runlist(&dqm->packet_mgr, &dqm->queues);
    
    // ⭐ 步骤 5: 标记 runlist 已激活
    dqm->active_runlist = true;
    
    return retval;
}
```

---

## 🔧 PM4 Packet - MAP_QUEUES

### Packet Manager 发送 Runlist

```c
═══════════════════════════════════════════════════════════════════
文件: kfd_packet_manager.c (推测)
═══════════════════════════════════════════════════════════════════

int pm_send_runlist(struct packet_manager *pm, 
                   struct list_head *queues_list)
{
    // ⭐ 步骤 1: 分配 PM4 packet buffer
    uint32_t *packet_buffer;
    int num_queues = 0;
    
    // ⭐ 步骤 2: 遍历所有 Queue，为每个 Queue 创建 MAP_QUEUES packet
    list_for_each_entry(qpd, queues_list, list) {
        list_for_each_entry(q, &qpd->queues_list, list) {
            if (!q->properties.is_active)
                continue;
            
            // ⭐⭐⭐ 创建 MAP_QUEUES packet
            // 这个 packet 告诉 CP:
            // 1. MQD 的位置（包含所有配置，包括优先级）
            // 2. Queue 的 ID
            // 3. Pipe ID
            pm_build_map_queues_packet(
                packet_buffer,
                q->gart_mqd_addr,    // ⭐ MQD GPU 地址
                q->queue,            // Queue ID
                q->pipe,             // Pipe ID
                q->properties.type   // Queue 类型
            );
            
            packet_buffer += packet_size;
            num_queues++;
        }
    }
    
    // ⭐ 步骤 3: 提交 packet buffer 到 CP (通过 ring buffer)
    retval = amdgpu_amdkfd_submit_ib(
        kdev->adev,
        KGD_ENGINE_MEC1,
        vmid,
        ib_base,
        packet_buffer,
        num_packets
    );
    
    return retval;
}
```

### MAP_QUEUES PM4 Packet 格式

```c
// PM4 Packet Type 3 - MAP_QUEUES
struct pm4_map_queues {
    uint32_t header;           // Packet header (opcode = MAP_QUEUES)
    
    // DW1
    uint32_t queue_sel:2;      // Queue selection
    uint32_t vmid:4;           // Virtual Machine ID
    uint32_t queue_type:3;     // Queue type (compute/sdma)
    uint32_t alloc_format:2;   // Allocation format
    uint32_t engine_sel:3;     // Engine selection
    uint32_t num_queues:4;     // Number of queues
    uint32_t check_disable:1;  // Disable checks
    uint32_t doorbell_offset:26; // ⭐ Doorbell offset
    
    // DW2
    uint32_t mqd_addr_lo;      // ⭐⭐⭐ MQD 地址低 32 位
    
    // DW3
    uint32_t mqd_addr_hi;      // ⭐⭐⭐ MQD 地址高 32 位
    
    // DW4
    uint32_t wptr_addr_lo;     // Write pointer 地址低 32 位
    
    // DW5
    uint32_t wptr_addr_hi;     // Write pointer 地址高 32 位
};

/*
 * ⭐⭐⭐ 关键点：
 * 
 * MAP_QUEUES packet 只包含 MQD 的地址，不包含优先级值本身！
 * 
 * CP 收到 MAP_QUEUES packet 后：
 * 1. 从 mqd_addr 读取整个 MQD 结构
 * 2. 从 MQD 中读取 cp_hqd_pipe_priority ⭐⭐⭐
 * 3. 从 MQD 中读取 cp_hqd_queue_priority
 * 4. 从 MQD 中读取 cp_hqd_pq_base (ring buffer 地址)
 * 5. 从 MQD 中读取所有其他配置
 * 
 * 所以优先级仍然存储在 MQD 中，CP 会读取并使用！
 */
```

---

## 🚀 CPSCH 调度流程（完整版）

```
═══════════════════════════════════════════════════════════════════
Phase 1: Queue 创建和 MQD 配置（与 MES 相同）
═══════════════════════════════════════════════════════════════════

用户:
  hipStreamCreateWithPriority(&stream_high, 0, -1)  // HIGH
  hipStreamCreateWithPriority(&stream_low, 0, 1)    // LOW

HIP/HSA Runtime:
  ├─ 创建两个 AqlQueue (独立的 ring buffer)
  ├─ Queue-1: ring_buf = 0x7fab12340000, doorbell = 0x1000
  └─ Queue-2: ring_buf = 0x7fac56780000, doorbell = 0x1008

KFD Driver:
  ├─ 创建两个 MQD (内存中)
  │
  ├─ MQD-1 (Queue-1, HIGH priority):
  │   ├─ cp_hqd_pq_base          = 0x7fab12340000
  │   ├─ cp_hqd_pq_doorbell_ctrl = 0x1000
  │   ├─ cp_hqd_pipe_priority    = 2 (HIGH)  ⭐⭐⭐
  │   └─ cp_hqd_queue_priority   = 11
  │
  └─ MQD-2 (Queue-2, LOW priority):
      ├─ cp_hqd_pq_base          = 0x7fac56780000
      ├─ cp_hqd_pq_doorbell_ctrl = 0x1008
      ├─ cp_hqd_pipe_priority    = 0 (LOW)   ⭐⭐⭐
      └─ cp_hqd_queue_priority   = 1

═══════════════════════════════════════════════════════════════════
Phase 2: Runlist 提交（CPSCH 特有）⭐⭐⭐
═══════════════════════════════════════════════════════════════════

触发时机:
  ├─ create_queue_cpsch() 调用 map_queues_cpsch()
  ├─ update_queue() 调用 map_queues_cpsch()
  └─ restore_process_queues_cpsch() 调用 execute_queues_cpsch()

map_queues_cpsch():
  ├─ 构建 runlist（活跃 Queue 列表）
  │   ├─ Queue-1 (HIGH, mqd_addr = 0xMQD_ADDR_1)
  │   └─ Queue-2 (LOW,  mqd_addr = 0xMQD_ADDR_2)
  │
  └─ pm_send_runlist() - 发送 PM4 packet

PM4 Packet 内容:
  ┌─────────────────────────────────────────────────────────────┐
  │ MAP_QUEUES Packet #1 (for Queue-1)                          │
  ├─────────────────────────────────────────────────────────────┤
  │ header          = 0xC0033000 (MAP_QUEUES opcode)           │
  │ queue_id        = 1001                                      │
  │ pipe_id         = 0                                         │
  │ mqd_addr        = 0xMQD_ADDR_1  ⭐⭐⭐ MQD-1 的地址         │
  │ doorbell_offset = 0x1000                                    │
  └─────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────┐
  │ MAP_QUEUES Packet #2 (for Queue-2)                          │
  ├─────────────────────────────────────────────────────────────┤
  │ header          = 0xC0033000 (MAP_QUEUES opcode)           │
  │ queue_id        = 1002                                      │
  │ pipe_id         = 0                                         │
  │ mqd_addr        = 0xMQD_ADDR_2  ⭐⭐⭐ MQD-2 的地址         │
  │ doorbell_offset = 0x1008                                    │
  └─────────────────────────────────────────────────────────────┘

提交到 CP:
  └─ amdgpu_amdkfd_submit_ib() - 提交到 CP ring buffer

═══════════════════════════════════════════════════════════════════
Phase 3: CP 处理 Runlist ⭐⭐⭐
═══════════════════════════════════════════════════════════════════

CP (Command Processor) 固件:
  
  1. 从 CP ring buffer 读取 MAP_QUEUES packet
  
  2. 对于每个 MAP_QUEUES packet:
     ├─ 读取 mqd_addr (MQD 地址)
     ├─ 从内存读取整个 MQD 结构
     │   ├─ cp_hqd_pq_base          (ring buffer 地址)
     │   ├─ cp_hqd_pq_doorbell_ctrl (doorbell 偏移)
     │   ├─ cp_hqd_pipe_priority    ⭐⭐⭐ 优先级！
     │   ├─ cp_hqd_queue_priority
     │   └─ ... (所有其他寄存器)
     │
     └─ 将 MQD 加载到 HQD (Hardware Queue Descriptor)
  
  3. 构建内部队列列表:
     ├─ Queue-1: priority=2 (HIGH), ring_buf=0x7fab12340000
     └─ Queue-2: priority=0 (LOW),  ring_buf=0x7fac56780000
  
  4. ⭐⭐⭐ 根据优先级排序队列
     └─ 高优先级队列会被优先调度

═══════════════════════════════════════════════════════════════════
Phase 4: 用户提交 Kernel（与 MES 相同）
═══════════════════════════════════════════════════════════════════

用户写 Doorbell:
  ├─ Queue-1: write(BAR + 0x1000, wptr)  // HIGH priority
  └─ Queue-2: write(BAR + 0x1008, wptr)  // LOW priority

CP 检测 Doorbell:
  ├─ 检测到 0x1000 和 0x1008 的写入
  ├─ 查找对应的 HQD (已加载的 MQD)
  ├─ 读取 HQD 中的 cp_hqd_pipe_priority ⭐⭐⭐
  └─ 根据优先级调度:
      ├─ Queue-1 (priority=2) 优先调度
      └─ Queue-2 (priority=0) 延后调度

CP 从 Ring Buffer 读取 AQL Packet:
  ├─ 使用 cp_hqd_pq_base + read_ptr 计算地址
  └─ 读取 Dispatch Packet

CP 提交到 CU:
  ├─ 分配 Compute Unit
  └─ 启动 Wavefront 执行
```

---

## 💡 关键差异总结

### CPSCH vs MES 的差异

| 特性 | CPSCH 模式 | MES 模式 |
|-----|-----------|---------|
| **MQD 配置** | ✅ 相同（都配置优先级寄存器） | ✅ 相同 |
| **优先级寄存器** | ✅ `cp_hqd_pipe_priority` | ✅ `cp_hqd_pipe_priority` |
| **Ring Buffer** | ✅ 独立（每个 Queue） | ✅ 独立（每个 Queue） |
| **Doorbell** | ✅ 独立（每个 Queue） | ✅ 独立（每个 Queue） |
| **Runlist 提交** | ✅ **需要**（PM4 packet） | ❌ **不需要** |
| **MAP_QUEUES** | ✅ **需要发送** | ❌ 不需要 |
| **调度触发** | Doorbell + Runlist | 仅 Doorbell |
| **CP 读取 MQD** | ✅ 通过 MAP_QUEUES | ✅ 直接读取 |
| **优先级工作方式** | ✅ **相同**（CP 读 MQD） | ✅ **相同**（MES 读 MQD） |

### 相同点 ⭐⭐⭐

**重要**: CPSCH 和 MES 在优先级处理上**本质相同**：

1. ✅ **MQD 配置相同**: 
   - 都配置 `cp_hqd_pipe_priority`
   - 都配置 `cp_hqd_queue_priority`
   - 都配置 ring buffer 和 doorbell

2. ✅ **硬件读取相同**:
   - CP/MES 都从 MQD 读取优先级
   - CP/MES 都根据优先级调度

3. ✅ **调度行为相同**:
   - 高优先级队列优先被调度
   - 低优先级队列延后调度

### 不同点

唯一的差异是 **Queue 激活方式**：

- **CPSCH**: 需要通过 PM4 `MAP_QUEUES` packet 显式告诉 CP
- **MES**: MES 硬件自动检测 doorbell 和 MQD

---

## 📝 CPSCH 模式下的重要概念

### 1. Runlist（运行列表）

```c
// Runlist 是所有活跃 Queue 的列表
struct runlist {
    struct list_head queues;  // 所有活跃的 Queue
    
    // 每个 Queue 包含:
    // - MQD 地址（包含优先级等所有配置）
    // - Queue ID
    // - Pipe ID
    // - Doorbell 偏移
};

// CPSCH 需要通过 PM4 packet 告诉 CP 这个列表
// MES 不需要，因为它自动发现
```

### 2. PM4 Packet

```c
// PM4 (Packet Manager 4) 是 AMD GPU 的命令协议
// 用于 CPU 与 GPU 通信

// MAP_QUEUES packet 告诉 CP:
// "这里有一个 Queue，它的 MQD 在这个地址，请加载它"

// CP 收到 packet 后:
// 1. 从 MQD 地址读取 MQD 结构
// 2. 将 MQD 加载到 HQD (Hardware Queue Descriptor)
// 3. 从 MQD 中读取优先级等所有配置
```

### 3. HQD (Hardware Queue Descriptor)

```c
// HQD 是 CP 内部的硬件结构
// 存储从 MQD 加载的配置

// CP 为每个活跃的 Queue 维护一个 HQD
// HQD 包含:
// - Ring buffer 地址 (从 MQD.cp_hqd_pq_base)
// - Doorbell 偏移 (从 MQD.cp_hqd_pq_doorbell_ctrl)
// - 优先级 (从 MQD.cp_hqd_pipe_priority) ⭐⭐⭐
// - 其他配置

// CP 调度时读取 HQD 的优先级字段
```

---

## 🔍 验证 CPSCH 模式

### 检查 GPU 是否使用 CPSCH

```bash
# 方法 1: 查看 sched_policy
sudo cat /sys/module/amdgpu/parameters/sched_policy
# 输出: HWS (Hardware Scheduling) = CPSCH

# 方法 2: 查看 dmesg
sudo dmesg | grep -i "scheduling policy"
# 输出: [drm] kfd: Scheduling policy: HWS (CPSCH mode)

# 方法 3: 检查 MES 是否启用
sudo dmesg | grep -i "enable_mes"
# 如果没有输出或显示 enable_mes=0，则使用 CPSCH

# 方法 4: 查看 GPU 型号
rocm-smi --showproductname
# MI308X 通常使用 CPSCH
# 高端 GFX11+ GPU 可能使用 MES
```

### 追踪 Runlist 提交

在 KFD 代码中添加打印：

```c
// 在 kfd_device_queue_manager.c 的 map_queues_cpsch() 中

static int map_queues_cpsch(struct device_queue_manager *dqm)
{
    // ... 原有代码 ...
    
    // ⭐ 添加 debug 打印
    pr_info("KFD: map_queues_cpsch - Building runlist:\n");
    
    list_for_each_entry(cur, &dqm->queues, list) {
        qpd = cur->qpd;
        list_for_each_entry(q, &qpd->queues_list, list) {
            if (q->properties.is_active) {
                pr_info("  Queue ID=%u, priority=%u, pipe_priority=%u, "
                        "mqd_addr=0x%llx, doorbell=0x%x\n",
                        q->properties.queue_id,
                        q->properties.priority,
                        // 从 MQD 读取 pipe_priority
                        ((struct v11_compute_mqd*)q->mqd)->cp_hqd_pipe_priority,
                        q->gart_mqd_addr,
                        q->properties.doorbell_off);
            }
        }
    }
    
    retval = pm_send_runlist(&dqm->packet_mgr, &dqm->queues);
    
    pr_info("KFD: map_queues_cpsch - Runlist sent, ret=%d\n", retval);
    
    // ... 原有代码 ...
}
```

### 查看 dmesg 输出

```bash
sudo dmesg | grep "map_queues_cpsch"

# 预期输出：
# [12345.678] KFD: map_queues_cpsch - Building runlist:
# [12345.678]   Queue ID=1001, priority=11, pipe_priority=2, 
#               mqd_addr=0x7fab00001000, doorbell=0x1000
# [12345.679]   Queue ID=1002, priority=1, pipe_priority=0, 
#               mqd_addr=0x7fab00002000, doorbell=0x1008
# [12345.679] KFD: map_queues_cpsch - Runlist sent, ret=0
```

---

## ⚠️ 重要提醒

**当前状态**: HSA Runtime 中优先级被写死，CPSCH 和 MES 都受影响！

**问题位置**: `rocr-runtime/core/runtime/amd_aql_queue.cpp` Line 100
```cpp
priority_(HSA_QUEUE_PRIORITY_NORMAL),  // ⚠️ 写死了！
```

**影响 CPSCH**:
- 所有 Queue 在创建时都是 NORMAL 优先级
- `pm_send_runlist()` 发送的 MQD 都有相同的 `cp_hqd_pipe_priority`
- CP 无法区分优先级

**修复后 CPSCH 会正常工作**:
- Runlist 中会包含不同优先级的 MQD
- CP 从 MQD 读取不同的 `cp_hqd_pipe_priority`
- CP 根据优先级调度

**详细修复方案**: 见 [PRIORITY_CODE_FIX_TODO.md](./PRIORITY_CODE_FIX_TODO.md)

---

## 📚 总结

### 核心要点

1. **CPSCH 仍然支持优先级** ✅
   - 优先级存储在 MQD 的 `cp_hqd_pipe_priority` 寄存器中
   - CP 从 MQD 读取并使用优先级进行调度
   - ⚠️ **但需要先修复 HSA Runtime 代码**

2. **CPSCH 的额外步骤** ⭐
   - 需要通过 PM4 `MAP_QUEUES` packet 提交 runlist
   - `MAP_QUEUES` 包含 MQD 地址，不包含优先级值本身
   - CP 从 MQD 地址读取完整的 MQD（包含优先级）

3. **与 MES 的本质相同** ✅
   - MQD 配置方式相同
   - 优先级寄存器相同
   - 硬件调度逻辑相同
   - 唯一差异是 Queue 激活方式
   - ⚠️ **都受 HSA Runtime 优先级写死的影响**

4. **Ring Buffer 和 Doorbell** ✅
   - 每个 Queue 仍然有独立的 ring buffer
   - 每个 Queue 仍然有独立的 doorbell
   - 用户空间写 doorbell 触发调度

### 调用栈总结（CPSCH 模式）

```
hipStreamCreateWithPriority(priority)
  ↓
hip::Stream::Create(priority)
  ↓
AqlQueue::AqlQueue(priority)
  ├─ AllocRegisteredRingBuffer() → ring_buf (独立)
  └─ driver.CreateQueue(priority, ring_buf)
      ↓
      pqm_create_queue(q_properties)
        ├─ init_mqd(q_properties)
        │   ├─ cp_hqd_pq_base = ring_buf  ⭐
        │   ├─ cp_hqd_pq_doorbell_control = doorbell  ⭐
        │   └─ set_priority()
        │       ├─ cp_hqd_pipe_priority = 映射后的优先级  ⭐⭐⭐
        │       └─ cp_hqd_queue_priority = 原始优先级
        │
        └─ map_queues_cpsch()  ⭐⭐⭐ CPSCH 特有！
            └─ pm_send_runlist()
                └─ 发送 PM4 MAP_QUEUES packet
                    ├─ mqd_addr = MQD 地址  ⭐⭐⭐
                    └─ doorbell_offset
                        ↓
                        CP 从 mqd_addr 读取 MQD
                        包括 cp_hqd_pipe_priority ⭐⭐⭐
                        根据优先级调度
```

---

**创建时间**: 2026-01-29  
**目的**: 说明 CPSCH 模式下优先级的完整处理机制  
**结论**: ✅ CPSCH 和 MES 在优先级支持上本质相同，都通过 MQD 配置优先级寄存器，硬件根据这些寄存器进行调度
