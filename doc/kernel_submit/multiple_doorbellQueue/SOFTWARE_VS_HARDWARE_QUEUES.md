# 软件队列 vs 硬件队列：为什么有 1024 个软件队列但只有 32 个硬件队列？

**问题**: 硬件不是只有 32 个 queue 吗？为什么实验中分配了 1024 个 queue？

**简短回答**: 1024 是**软件队列**（Queue ID）的上限，32 是**硬件队列**（HQD）的数量。它们不是 1:1 映射的！

---

## 🔑 核心概念

### 软件队列 (Software Queue / Queue ID)

**定义**: 用户空间和 KFD 驱动层的队列抽象

**数量上限**: 1024 个（每个进程）

```c
// ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_priv.h:102
#define KFD_MAX_NUM_OF_QUEUES_PER_PROCESS 1024
```

**特点**:
- 每个进程可以创建最多 1024 个队列
- 由 KFD 驱动管理（bitmap 分配）
- 每个 Queue 有唯一的 Queue ID (0-1023)
- 对应用程序可见（通过 HIP Stream 等）

### 硬件队列 (Hardware Queue / HQD)

**定义**: GPU 硬件实际的队列资源

**数量**: 32 个（MI308X 上 KFD 可用的）

```c
// ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/gfx_v9_0.c:2272-2273
adev->gfx.mec.num_pipe_per_mec = 4;   // 4 个 Pipes
adev->gfx.mec.num_queue_per_pipe = 8; // 每个 Pipe 8 个 Queues

// MI308X 上 KFD 只使用 MEC 0:
// 1 MEC × 4 Pipes × 8 Queues = 32 个 HQD
```

**特点**:
- 这是实际的硬件资源
- 数量固定，由硬件决定
- 每个 HQD 对应一个 CP (Command Processor) slot
- 应用程序不直接感知

---

## 🔄 软件队列到硬件队列的映射

### 关键机制

**并非 1:1 映射！** 多个软件队列可以**复用**同一个硬件 HQD。

```
软件层（用户空间/KFD）:
  Process 1: Queue 0, 1, 2, 3   (4 个软件队列)
  Process 2: Queue 4, 5, 6, 7   (4 个软件队列)
  Process 3: Queue 8, 9, 10, 11 (4 个软件队列)
  ...
  Process N: Queue 1020-1023    (4 个软件队列)
  
  ↓ 映射/调度
  
硬件层（GPU MEC）:
  MEC 0:
    Pipe 0: Queue 0-7  (8 个 HQD)
    Pipe 1: Queue 0-7  (8 个 HQD)
    Pipe 2: Queue 0-7  (8 个 HQD)
    Pipe 3: Queue 0-7  (8 个 HQD)
  总共: 32 个 HQD ⚠️
```

### 映射方式

#### 方式 1: 直接映射（理想情况）

**条件**: 软件队列数量 ≤ 硬件队列数量

```
软件 Queue 0 → HQD (Pipe 0, Queue 0)
软件 Queue 1 → HQD (Pipe 0, Queue 1)
软件 Queue 2 → HQD (Pipe 0, Queue 2)
...
软件 Queue 31 → HQD (Pipe 3, Queue 7)
```

**特点**: 每个软件队列独占一个 HQD，性能最优

#### 方式 2: 复用映射（常见情况）

**条件**: 软件队列数量 > 硬件队列数量（如实验中的情况）

```
软件 Queue 0, 32, 64, ...  → HQD (Pipe 0, Queue 0)
软件 Queue 1, 33, 65, ...  → HQD (Pipe 0, Queue 1)
软件 Queue 2, 34, 66, ...  → HQD (Pipe 0, Queue 2)
...
```

**特点**: 
- 多个软件队列共享同一个 HQD
- 调度器（MES/CPSCH）负责在它们之间切换
- 性能取决于调度策略

---

## 📊 实验中的情况分析

### v5 实验的队列分配

**软件队列分配**:
```bash
进程 1 (PID=3935030): Queue ID 216-219 (4 个软件队列)
进程 2 (PID=3935031): Queue ID 220-223 (4 个软件队列)

总共: 8 个软件队列
```

**硬件队列使用**:

由于总共只有 8 个软件队列，远少于 32 个 HQD，理论上可以做到：

```
软件 Queue 216 → HQD (Pipe 0, Queue 0)  独占
软件 Queue 217 → HQD (Pipe 0, Queue 1)  独占
软件 Queue 218 → HQD (Pipe 0, Queue 2)  独占
软件 Queue 219 → HQD (Pipe 0, Queue 3)  独占
软件 Queue 220 → HQD (Pipe 0, Queue 4)  独占
软件 Queue 221 → HQD (Pipe 0, Queue 5)  独占
软件 Queue 222 → HQD (Pipe 0, Queue 6)  独占
软件 Queue 223 → HQD (Pipe 0, Queue 7)  独占
```

**理论上**: 每个软件队列应该都能独占一个 HQD，不应该有 HQD 竞争。

### 为什么性能还是没有提升？

**因为瓶颈不在 HQD 分配，而在更高层**:

1. **调度器串行化** (`active_runlist`):
   - 即使 HQD 不同，调度器也可能串行处理
   - 这是软件层的瓶颈

2. **Ring 共享**:
   - 不同的 HQD 可能共享同一个 Ring
   - Ring 层的串行化

3. **Workqueue 串行化**:
   - AMDGPU 驱动使用 Ordered Workqueue
   - 事件处理串行化

**关键洞察**:
```
硬件 HQD 层: ✅ 资源充足（32 个 HQD，只用了 8 个）
          ↕
调度器层:   ❌ 串行化瓶颈 (active_runlist)
          ↕
Ring 层:    ❌ 共享瓶颈
          ↕
Workqueue 层: ❌ 串行化瓶颈
```

---

## 🔍 深入理解：CPSCH 如何管理映射

### allocate_hqd() 函数

**代码位置**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_device_queue_manager.c`

```c
static int allocate_hqd(struct device_queue_manager *dqm, struct queue *q)
{
    bool set;
    int pipe, bit, i;
    
    set = false;
    // ⭐ 遍历所有 Pipes，找一个空闲的 HQD
    for (pipe = dqm->next_pipe_to_allocate, i = 0;
            i < get_pipes_per_mec(dqm);  // 4 个 Pipes
            pipe = ((pipe + 1) % get_pipes_per_mec(dqm)), ++i) {
        
        if (dqm->allocated_queues[pipe] != 0) {
            // ⭐ 在这个 Pipe 中找一个空闲的 Queue slot
            bit = ffs(dqm->allocated_queues[pipe]) - 1;
            dqm->allocated_queues[pipe] &= ~(1 << bit);
            
            q->pipe = pipe;      // ⭐ 分配 Pipe ID (0-3)
            q->queue = bit;      // ⭐ 分配 Queue ID in Pipe (0-7)
            set = true;
            break;
        }
    }
    
    if (!set)
        return -EBUSY;  // ⚠️ 所有 32 个 HQD 都已占用
    
    pr_debug("hqd slot - pipe %d, queue %d\n", q->pipe, q->queue);
    // ...
}
```

### 关键数据结构

```c
// 每个 Pipe 的队列分配情况
// dqm->allocated_queues[pipe] 是一个 bitmap (8 bits)
//   bit 0 = Queue 0 是否可用
//   bit 1 = Queue 1 是否可用
//   ...
//   bit 7 = Queue 7 是否可用

// 示例：
dqm->allocated_queues[0] = 0b11111111  // Pipe 0 所有 Queue 都可用
dqm->allocated_queues[1] = 0b11111111  // Pipe 1 所有 Queue 都可用
dqm->allocated_queues[2] = 0b11111111  // Pipe 2 所有 Queue 都可用
dqm->allocated_queues[3] = 0b11111111  // Pipe 3 所有 Queue 都可用
// 总共 32 个 HQD 可用
```

### 软件 Queue ID 到 HQD 的映射

**并不是简单的模运算！**

实际的映射是通过 `allocate_hqd()` 动态分配的：

```
软件 Queue 创建请求:
  1. KFD 分配 Queue ID (0-1023)
  2. 调用 allocate_hqd() 找一个空闲的 HQD
  3. HQD 由 (Pipe, Queue in Pipe) 唯一标识
  4. 记录 Queue ID → HQD 的映射

示例（按创建顺序）:
  软件 Queue 0 → HQD (Pipe 0, Queue 0)  // 第 1 个创建
  软件 Queue 1 → HQD (Pipe 1, Queue 0)  // 第 2 个创建（Round-robin）
  软件 Queue 2 → HQD (Pipe 2, Queue 0)  // 第 3 个创建
  软件 Queue 3 → HQD (Pipe 3, Queue 0)  // 第 4 个创建
  软件 Queue 4 → HQD (Pipe 0, Queue 1)  // 第 5 个创建（Pipe 0 的下一个 slot）
  ...
```

**关键点**:
- Round-robin 分配策略（跨 Pipes 轮询）
- 优先使用不同的 Pipes（提高并行性）
- 只有当所有 Pipes 的某个 Queue slot 都用完了，才会复用

---

## 💡 为什么设计成 1024 个软件队列？

### 1. 灵活性

**允许应用程序创建大量队列**:
- 复杂的应用可能需要很多 Stream/Queue
- 例如：深度学习框架可能为每个 layer 创建一个 Stream
- 软件层不需要担心硬件限制

### 2. 抽象性

**隐藏硬件细节**:
- 应用程序不需要知道硬件有多少个 HQD
- 调度器透明地处理映射和复用
- 简化编程模型

### 3. 虚拟化

**类似 CPU 的进程虚拟化**:
- CPU 只有几个核心，但可以运行成百上千个进程
- GPU 只有 32 个 HQD，但可以管理成百上千个软件队列
- 通过时间片轮转和调度实现

### 4. 向前兼容

**为未来硬件留出空间**:
- 未来的 GPU 可能有更多的 HQD
- 软件接口保持不变
- 只需要调整调度策略

---

## 🎯 关键区别总结

| 特性 | 软件队列 (Queue ID) | 硬件队列 (HQD) |
|------|---------------------|----------------|
| **数量** | 1024（每进程上限） | 32（MI308X 上 KFD 可用） |
| **定义位置** | `kfd_priv.h` | 硬件配置（Pipes × Queues） |
| **管理者** | KFD 驱动 | MEC 硬件 |
| **可见性** | 应用程序可见 | 应用程序不可见 |
| **分配方式** | Bitmap 动态分配 | Round-robin 动态分配 |
| **映射关系** | 多对一（多个软件队列可共享一个 HQD） | 一对一（每个 HQD 是唯一的硬件资源） |
| **生命周期** | 随队列创建/销毁 | 硬件固定存在 |
| **瓶颈** | 1024 上限（软件限制） | 32 上限（硬件限制） |

---

## 🔧 实际影响

### 何时会遇到 HQD 不足？

**场景 1**: 大量进程同时创建队列

```
32 个 HQD ÷ 4 个队列/进程 = 最多 8 个进程并发
```

如果有 16 个进程同时运行，每个创建 4 个队列：
- 总需求: 64 个 HQD
- 实际可用: 32 个 HQD
- **结果**: 至少一半的队列需要复用 HQD，调度器负责切换

### 何时不会遇到 HQD 不足？

**场景 2**: 少量进程（如实验中的 2 进程）

```
2 个进程 × 4 个队列/进程 = 8 个队列
8 < 32，硬件资源充足
```

**这就是为什么**:
- v5 实验中，HQD 资源不是瓶颈
- 真正的瓶颈在调度器和 Ring 层
- 即使增加 HQD 数量也不会提升性能

---

## 📚 验证方法

### 1. 查看 HQD 分配情况

```bash
# 启用 KFD debug logs
sudo bash enable_kfd_debug.sh

# 运行测试程序
./test_program

# 查看 HQD 分配
sudo dmesg | grep "hqd slot"

# 输出示例:
# kfd: hqd slot - pipe 0, queue 0  ← 软件 Queue 216 → HQD (0, 0)
# kfd: hqd slot - pipe 1, queue 0  ← 软件 Queue 217 → HQD (1, 0)
# kfd: hqd slot - pipe 2, queue 0  ← 软件 Queue 218 → HQD (2, 0)
# kfd: hqd slot - pipe 3, queue 0  ← 软件 Queue 219 → HQD (3, 0)
# kfd: hqd slot - pipe 0, queue 1  ← 软件 Queue 220 → HQD (0, 1)
# ...
```

### 2. 统计 HQD 使用情况

```bash
# 统计已分配的 HQD 数量
sudo dmesg | grep "hqd slot" | wc -l

# 查看 HQD 分布
sudo dmesg | grep "hqd slot" | awk '{print "Pipe "$5", Queue "$7}' | sort | uniq -c
```

### 3. 查看软件队列分配

```bash
# 查看 Queue ID 分配
sudo dmesg | grep "CREATE_QUEUE" | grep "queue_id"

# 输出示例:
# queue_id=216 doorbell_offset_in_process=0x1800
# queue_id=217 doorbell_offset_in_process=0x1808
# queue_id=218 doorbell_offset_in_process=0x1810
# ...
```

---

## ✅ 总结

### 问题回答

**Q**: 硬件不是只有 32 个 queue 吗？为什么分配了 1024 个 queue？

**A**: 
1. **32 个是硬件队列 (HQD)**，由 GPU 硬件决定
2. **1024 个是软件队列 (Queue ID)**，是 KFD 驱动的软件限制
3. **它们不是 1:1 映射的**！多个软件队列可以复用同一个 HQD
4. **v5 实验中**: 只用了 8 个软件队列，映射到 8 个 HQD，硬件资源充足
5. **性能瓶颈**: 不在 HQD 层，而在调度器和 Ring 层

### 关键洞察

```
软件队列 (1024 上限)
    ↓ 映射/调度
硬件队列 (32 个 HQD)  ← 资源充足，不是瓶颈
    ↓ 调度器管理
调度器层 (active_runlist) ← 串行化，关键瓶颈 🔴
    ↓ Ring 映射
Ring 层 (可能共享)      ← 串行化，次要瓶颈 🟡
```

**这就是为什么 Queue ID 优化未带来性能提升！**

---

**文档版本**: v1.0  
**创建日期**: 2026-01-19  
**适用对象**: 需要理解软件队列和硬件队列区别的研究者

