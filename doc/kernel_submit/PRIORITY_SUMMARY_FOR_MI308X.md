# Stream 优先级机制总结 - MI308X (CPSCH 模式)

**快速参考** - 一页了解 MI308X 上 Stream 优先级的完整机制

**创建时间**: 2026-01-29

---

## 🎯 核心结论

### ✅ 理论上支持优先级

**MI308X (CPSCH 模式)** 完全支持 Stream 优先级：

1. ✅ 每个 Stream 有独立的 Queue、ring-buffer、doorbell
2. ✅ MQD 中配置优先级寄存器 (`cp_hqd_pipe_priority`)
3. ✅ CP (Command Processor) 从 MQD 读取优先级
4. ✅ CP 根据优先级调度 Queue

### ⚠️ 实际需要修复代码

**当前问题**: HSA Runtime 中优先级被写死，需要修复才能真正生效。

---

## 📊 MI308X 使用 CPSCH 模式

### CPSCH vs MES

| 特性 | CPSCH (MI308X) | MES |
|-----|---------------|-----|
| **调度器** | 软件 (CP Scheduler) | 硬件 (Micro-Engine) |
| **Queue 激活** | PM4 MAP_QUEUES packet | 自动检测 doorbell |
| **MQD 配置** | ✅ 相同 | ✅ 相同 |
| **优先级支持** | ✅ 支持 | ✅ 支持 |
| **调度逻辑** | ✅ 相同 | ✅ 相同 |

**关键**: CPSCH 和 MES 在**优先级处理上本质相同**，都通过 MQD 配置优先级寄存器。

---

## 🚀 完整流程（MI308X）

```
用户代码:
  hipStreamCreateWithPriority(&stream_high, 0, -1)  // HIGH
  hipStreamCreateWithPriority(&stream_low, 0, 1)    // LOW
    ↓
HIP Runtime:
  hip::Stream (priority=HIGH/LOW)
    ↓
HSA Runtime:
  AqlQueue (独立的 ring buffer, doorbell)
  ⚠️ 问题: priority 被写死为 NORMAL (需要修复)
    ↓
KFD Driver:
  init_mqd() - 配置 MQD 寄存器:
    ├─ cp_hqd_pq_base          = ring buffer 地址
    ├─ cp_hqd_pq_doorbell_ctrl = doorbell 偏移
    ├─ cp_hqd_pipe_priority    = 优先级 ⭐⭐⭐
    └─ cp_hqd_queue_priority   = 原始优先级
    ↓
CPSCH 特有:
  map_queues_cpsch() - 发送 PM4 MAP_QUEUES packet
    └─ packet 包含 MQD 地址（不包含优先级值本身）
    ↓
CP (Command Processor):
  1. 接收 MAP_QUEUES packet
  2. 从 MQD 地址读取整个 MQD 结构
  3. 读取 cp_hqd_pipe_priority ⭐⭐⭐
  4. 根据优先级调度 Queue
```

---

## 🔧 MQD 寄存器对比

### 高优先级 Queue (priority=-1)

```
MQD 配置（理论上）:
  cp_hqd_pq_base          = 0x7fab12340000  (独立 ring buffer)
  cp_hqd_pq_doorbell_ctrl = 0x1000          (独立 doorbell)
  cp_hqd_pipe_priority    = 2 (HIGH)        ⭐⭐⭐
  cp_hqd_queue_priority   = 11

当前实际（⚠️ 需要修复）:
  cp_hqd_pipe_priority    = 1 (NORMAL)      ❌ 被写死了
```

### 低优先级 Queue (priority=1)

```
MQD 配置（理论上）:
  cp_hqd_pq_base          = 0x7fac56780000  (不同的 ring buffer)
  cp_hqd_pq_doorbell_ctrl = 0x1008          (不同的 doorbell)
  cp_hqd_pipe_priority    = 0 (LOW)         ⭐⭐⭐
  cp_hqd_queue_priority   = 1

当前实际（⚠️ 需要修复）:
  cp_hqd_pipe_priority    = 1 (NORMAL)      ❌ 被写死了
```

---

## ⚠️ 当前问题和修复方案

### 问题位置

**文件**: `rocr-runtime/core/runtime/amd_aql_queue.cpp`  
**Line 100**:

```cpp
AqlQueue::AqlQueue(...)
    : ...,
      priority_(HSA_QUEUE_PRIORITY_NORMAL),  // ⚠️⚠⚠️ 写死了！
      ...
```

### 修复方案

```cpp
// 1. 修改构造函数，接受 priority 参数
AqlQueue::AqlQueue(..., HSA_QUEUE_PRIORITY priority)

// 2. 修改初始化列表
    : ...,
      priority_(priority),  // ✅ 使用参数
      ...
```

### 详细步骤

见 [PRIORITY_CODE_FIX_TODO.md](./PRIORITY_CODE_FIX_TODO.md)

---

## 🔬 CP 调度流程（MI308X）

```
1. 驱动发送 PM4 MAP_QUEUES packet
   ├─ Packet 包含: MQD 地址, Queue ID, Pipe ID
   └─ 不包含: 优先级（优先级在 MQD 中）

2. CP 接收 MAP_QUEUES packet
   ├─ 读取 MQD 地址
   └─ 从内存读取整个 MQD 结构

3. CP 解析 MQD
   ├─ cp_hqd_pq_base          → ring buffer 在哪
   ├─ cp_hqd_pq_doorbell_ctrl → doorbell 在哪
   ├─ cp_hqd_pipe_priority    → ⭐⭐⭐ 优先级是多少
   └─ ... 其他配置

4. CP 维护内部队列列表
   ├─ Queue-1: priority=2 (HIGH), ring_buf=0x7fab...
   └─ Queue-2: priority=0 (LOW),  ring_buf=0x7fac...

5. 用户写 Doorbell
   ├─ Queue-1: write(BAR + 0x1000, wptr)
   └─ Queue-2: write(BAR + 0x1008, wptr)

6. CP 检测 Doorbell 写入
   ├─ 查找对应的 HQD (Hardware Queue Descriptor)
   ├─ 读取 HQD 中的 priority ⭐⭐⭐
   └─ 根据 priority 排序

7. CP 调度 ⭐⭐⭐
   ├─ 高优先级 Queue 优先调度
   └─ 低优先级 Queue 延后调度

8. CP 从 ring buffer 读取 AQL packet
   └─ 提交到 Compute Unit 执行
```

---

## 📋 验证清单

### Phase 1: 代码修复前的验证

```bash
# 运行测试
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/test_stream_priority
./test_concurrent

# 查看日志
sudo dmesg | grep "pipe_priority"

# ❌ 当前看到：
# Queue 1001: pipe_priority=1 (NORMAL)
# Queue 1002: pipe_priority=1 (NORMAL)
# ↑ 都是 NORMAL，证实了问题
```

### Phase 2: 代码修复后的验证

```bash
# 1. 修改代码（见 PRIORITY_CODE_FIX_TODO.md）

# 2. 重新编译 HSA Runtime
cd rocr-runtime/build
cmake .. && make -j$(nproc)
sudo make install

# 3. 重新运行测试
./test_concurrent

# 4. 查看日志
sudo dmesg | grep "pipe_priority"

# ✅ 应该看到：
# Queue 1001: pipe_priority=2 (HIGH)
# Queue 1002: pipe_priority=0 (LOW)
# ↑ 不同的优先级！
```

### Phase 3: 性能验证

修复后，需要验证优先级真的影响调度：

1. 创建测试：高优先级 kernel vs 低优先级 kernel
2. 测量执行延迟
3. 验证高优先级 kernel 确实优先执行

---

## 🗂️ 关键文件位置

### 需要修改的文件

```
HSA Runtime:
  rocr-runtime/core/runtime/amd_aql_queue.cpp     ⚠️ Line 100 (主要修改)
  rocr-runtime/core/runtime/amd_aql_queue.h       (添加参数)
  rocr-runtime/core/runtime/amd_gpu_agent.cpp     (传递参数)

KFD Driver (已经正确):
  kfd/amdkfd/kfd_mqd_manager_v11.c                ✅ Line 96 set_priority()
  kfd/amdkfd/kfd_device_queue_manager.c           ✅ Line 2413 map_queues_cpsch()
```

### 测试和文档

```
测试程序:
  doc/kernel_submit/test_stream_priority/test_concurrent.cpp

文档:
  doc/kernel_submit/PRIORITY_TO_HARDWARE_DEEP_TRACE.md    (完整追踪)
  doc/kernel_submit/PRIORITY_CPSCH_MODE_TRACE.md          (CPSCH 详解)
  doc/kernel_submit/PRIORITY_CODE_FIX_TODO.md             (修复方案)
  doc/kernel_submit/PRIORITY_SUMMARY_FOR_MI308X.md        (本文档)
```

---

## 💡 关键洞察

### 1. CPSCH 和 MES 本质相同

- 都通过 MQD 配置优先级寄存器
- 都由硬件读取优先级进行调度
- 唯一差异是 Queue 激活方式（MAP_QUEUES vs 自动检测）

### 2. 优先级存储在 MQD 中

- `cp_hqd_pipe_priority` = 映射后的优先级 (0=LOW, 1=MEDIUM, 2=HIGH)
- `cp_hqd_queue_priority` = 原始优先级值 (0-15)
- 硬件直接读取这些寄存器

### 3. PM4 Packet 不包含优先级

- MAP_QUEUES packet 只包含 MQD 地址
- CP 从 MQD 地址读取完整配置（包括优先级）
- 所以优先级必须正确配置在 MQD 中

### 4. 当前代码问题的影响

- HSA Runtime 创建的所有 Queue 都是 NORMAL 优先级
- MQD 中的 `cp_hqd_pipe_priority` 都是 1
- CP 看到所有 Queue 优先级相同
- 无法测试优先级调度效果

---

## 📚 参考文档

| 文档 | 用途 |
|-----|------|
| [PRIORITY_CODE_FIX_TODO.md](./PRIORITY_CODE_FIX_TODO.md) | ⚠️ **代码修复详细步骤** |
| [PRIORITY_CPSCH_MODE_TRACE.md](./PRIORITY_CPSCH_MODE_TRACE.md) | CPSCH 模式深度解析 |
| [PRIORITY_TO_HARDWARE_DEEP_TRACE.md](./PRIORITY_TO_HARDWARE_DEEP_TRACE.md) | 硬件寄存器完整追踪 |
| [PRIORITY_HARDWARE_SUMMARY.md](./PRIORITY_HARDWARE_SUMMARY.md) | 一页快速参考 |
| [STREAM_PRIORITY_AND_QUEUE_MAPPING.md](./STREAM_PRIORITY_AND_QUEUE_MAPPING.md) | Stream 和 Queue 映射 |

---

## ✅ 下一步行动

1. **阅读**: [PRIORITY_CODE_FIX_TODO.md](./PRIORITY_CODE_FIX_TODO.md)
2. **修改**: `amd_aql_queue.cpp` Line 100
3. **编译**: 重新编译 HSA Runtime
4. **测试**: 运行 `test_concurrent`
5. **验证**: 检查 dmesg 日志
6. **性能测试**: 验证优先级调度效果

---

**创建时间**: 2026-01-29  
**适用 GPU**: MI308X (CPSCH 模式)  
**状态**: ⚠️ 需要修复代码后测试  
**预计工作量**: 2.5 天（修复 + 测试 + 验证）
