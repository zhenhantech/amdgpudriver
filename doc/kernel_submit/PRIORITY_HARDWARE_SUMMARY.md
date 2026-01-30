# Stream 优先级到硬件配置 - 快速参考

一页总结：不同优先级的 Stream 如何映射到不同的硬件配置

**⚠️ 适用于**: MES 和 CPSCH 两种模式（核心机制相同）  
**CPSCH 差异**: 需要额外的 PM4 packet 提交 runlist，详见 [PRIORITY_CPSCH_MODE_TRACE.md](./PRIORITY_CPSCH_MODE_TRACE.md)

---

## 🎯 核心答案（一句话）

**不同优先级的 Stream 使用不同的 ring-buffer、不同的 doorbell，并在 MQD 中配置不同的优先级寄存器，GPU 硬件直接读取这些寄存器进行调度。**

---

## 📊 优先级映射表

| 用户 API | HIP Priority | HSA Priority | KFD Priority | Pipe Priority | 硬件行为 |
|---------|-------------|-------------|-------------|--------------|---------|
| `hipStreamCreateWithPriority(s, 0, -1)` | HIGH | MAXIMUM | 11-15 | **2 (HIGH)** | 优先调度 |
| `hipStreamCreateWithPriority(s, 0, 0)` | NORMAL | NORMAL | 7-10 | **1 (MEDIUM)** | 正常调度 |
| `hipStreamCreateWithPriority(s, 0, 1)` | LOW | LOW | 0-6 | **0 (LOW)** | 延后调度 |

---

## 🔧 MQD 寄存器对比

### Queue-1 (HIGH Priority)

```
MQD 寄存器配置:
  cp_hqd_pq_base          = 0x7fab12340000  ⭐ 独立 ring buffer
  cp_hqd_pq_doorbell_ctrl = 0x00001000      ⭐ 独立 doorbell 偏移
  cp_hqd_pipe_priority    = 2               ⭐⭐⭐ HIGH (硬件读这个!)
  cp_hqd_queue_priority   = 11              ⭐ 原始优先级
  cp_hqd_quantum          = 0x00010101      时间片配置
  cp_hqd_vmid             = 1               虚拟机 ID
```

### Queue-2 (LOW Priority)

```
MQD 寄存器配置:
  cp_hqd_pq_base          = 0x7fac56780000  ⭐ 不同的 ring buffer
  cp_hqd_pq_doorbell_ctrl = 0x00001008      ⭐ 不同的 doorbell 偏移
  cp_hqd_pipe_priority    = 0               ⭐⭐⭐ LOW (硬件读这个!)
  cp_hqd_queue_priority   = 1               ⭐ 原始优先级
  cp_hqd_quantum          = 0x00010101      时间片配置
  cp_hqd_vmid             = 1               虚拟机 ID
```

---

## 🚀 完整路径（5 层）

```
Layer 1: 应用层
  hipStreamCreateWithPriority(&stream, 0, -1)  // priority = -1 (HIGH)

Layer 2: HIP Runtime
  priority_internal = HIGH
  new hip::Stream(device, priority_internal, ...)
  
Layer 3: HSA Runtime
  new AqlQueue(priority = MAXIMUM)
  AllocRegisteredRingBuffer() → ring_buf = 0x7fab12340000
  driver.CreateQueue(priority, ring_buf)
  
Layer 4: KFD Driver
  q_properties.priority = 11
  q_properties.queue_address = 0x7fab12340000
  q_properties.doorbell_off = 0x1000
  init_mqd(q_properties) → 创建 MQD
  
Layer 5: MQD Manager (硬件配置)
  mqd->cp_hqd_pq_base = 0x7fab12340000           ⭐ ring buffer 地址
  mqd->cp_hqd_pq_doorbell_control = 0x1000       ⭐ doorbell 偏移
  mqd->cp_hqd_pipe_priority = 2                  ⭐⭐⭐ 硬件优先级
  mqd->cp_hqd_queue_priority = 11
  
  MES 硬件读取 MQD → 根据 cp_hqd_pipe_priority 调度
```

---

## 🔬 硬件调度流程

```
1. 用户空间写 Doorbell
   ├─ Queue-1: write(BAR + 0x1000, wptr)  ← HIGH priority
   └─ Queue-2: write(BAR + 0x1008, wptr)  ← LOW priority

2. MES 检测 Doorbell 写入
   ├─ 检测到 0x1000 → Queue-1
   └─ 检测到 0x1008 → Queue-2

3. MES 读取每个 Queue 的 MQD
   ├─ Queue-1 MQD:
   │   ├─ cp_hqd_pq_base = 0x7fab12340000
   │   └─ cp_hqd_pipe_priority = 2 (HIGH)  ⭐⭐⭐
   └─ Queue-2 MQD:
       ├─ cp_hqd_pq_base = 0x7fac56780000
       └─ cp_hqd_pipe_priority = 0 (LOW)   ⭐⭐⭐

4. MES 调度决策 ⭐⭐⭐
   if (queue1.priority > queue2.priority) {
       schedule(queue1);  // 优先调度 Queue-1
   }

5. 从 Ring Buffer 读取 Packet
   packet = read_memory(mqd->cp_hqd_pq_base + rptr)

6. 提交到 CP 执行
   submit_to_compute_unit(packet)
```

---

## 📈 关键差异总结

| 特性 | Queue-1 (HIGH) | Queue-2 (LOW) | 备注 |
|-----|---------------|--------------|------|
| **Ring Buffer 地址** | `0x7fab12340000` | `0x7fac56780000` | ✅ 完全独立 |
| **Doorbell 偏移** | `0x1000` | `0x1008` | ✅ 不同地址 |
| **Pipe Priority** | `2 (HIGH)` | `0 (LOW)` | ⭐⭐⭐ 硬件调度关键 |
| **Queue Priority** | `11` | `1` | 原始值 |
| **调度顺序** | 优先 | 延后 | 由 pipe_priority 决定 |
| **时间片** | 可能更多 | 可能更少 | 硬件决定 |
| **HQD 资源** | 优先获得 | 可能等待 | 硬件决定 |

---

## ⚠️ 重要提醒

**当前状态**: HSA Runtime 中优先级被写死，需要先修复代码！

**问题位置**: `rocr-runtime/core/runtime/amd_aql_queue.cpp` Line 100
```cpp
priority_(HSA_QUEUE_PRIORITY_NORMAL),  // ⚠️ 写死了！
```

**影响**: 
- 所有 Queue 都是 NORMAL 优先级
- `cp_hqd_pipe_priority` 都是相同的值 (1)
- 硬件无法区分优先级

**修复方案**: 见 [PRIORITY_CODE_FIX_TODO.md](./PRIORITY_CODE_FIX_TODO.md)

---

## 🔍 验证命令

```bash
# 1. 运行测试
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/test_stream_priority
./test_concurrent

# 2. 查看 MQD（如果有 debugfs）
sudo cat /sys/kernel/debug/kfd/mqds

# 3. 查看 dmesg 日志
sudo dmesg | grep -E "create queue|priority|doorbell"

# ⚠️ 当前实际看到（修复前）：
# Queue 1001: priority=7, pipe_priority=1 (NORMAL), doorbell=0x1000
# Queue 1002: priority=7, pipe_priority=1 (NORMAL), doorbell=0x1008
# ↑ 都是 NORMAL，无法区分！

# ✅ 修复后应该看到：
# Queue 1001: priority=11, pipe_priority=2 (HIGH), doorbell=0x1000
# Queue 1002: priority=1,  pipe_priority=0 (LOW),  doorbell=0x1008
```

---

## 📚 详细文档

| 文档 | 内容 |
|-----|------|
| [PRIORITY_TO_HARDWARE_DEEP_TRACE.md](./PRIORITY_TO_HARDWARE_DEEP_TRACE.md) | 完整的深度追踪（推荐） |
| [PRIORITY_CPSCH_MODE_TRACE.md](./PRIORITY_CPSCH_MODE_TRACE.md) | CPSCH 模式（MI308X 使用） |
| [PRIORITY_CODE_FIX_TODO.md](./PRIORITY_CODE_FIX_TODO.md) | ⚠️ **代码修复方案**（必读） |
| [STREAM_PRIORITY_AND_QUEUE_MAPPING.md](./STREAM_PRIORITY_AND_QUEUE_MAPPING.md) | Stream 和 Queue 映射关系 |
| [test_stream_priority/](./test_stream_priority/) | 实际测试程序 |

---

## 💡 关键结论

1. ✅ **Ring Buffer**: 每个 Queue 完全独立的物理地址
2. ✅ **Doorbell**: 每个 Queue 不同的偏移地址
3. ✅ **MQD 寄存器**: 
   - `cp_hqd_pipe_priority` ⭐⭐⭐ 硬件用于调度
   - `cp_hqd_queue_priority` 原始优先级值
4. ✅ **硬件行为**: MES 直接读取 MQD，根据 `cp_hqd_pipe_priority` 调度

**一句话**: 不同优先级的 Stream 从头到尾都是独立的，包括内存、doorbell 和硬件寄存器，硬件根据 MQD 中的优先级字段执行不同的调度策略。

---

**创建时间**: 2026-01-29  
**用途**: 快速参考 - Stream 优先级到硬件配置的映射关系
