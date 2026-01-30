# Queue ID 分配实验的后续研究方向

## 📊 当前状态评估

### 已完成的工作

✅ **v4 → v5 优化**:
- 基于 PID 分配不同的 Queue ID 范围
- 避免了 Queue ID 重叠
- 代码实现正确

✅ **问题定位**:
- Queue ID 优化只解决了表面问题
- 更深层的瓶颈：HQD 复用、调度器串行化、Ring 共享
- 揭示了软件抽象和硬件实现之间的复杂映射

### 核心发现

```
Queue ID 层:  ✅ 已优化（不重叠）
     ↓
HQD 分配层:  ⚠️ 可能仍复用（未验证）
     ↓
调度器层:    ❌ active_runlist 串行化 🔴 主要瓶颈
     ↓
Ring 层:     ❌ Ring 共享 🟡 次要瓶颈
```

---

## 🎯 研究方向评估

### 方向 1: 验证 Queue ID 到 HQD 的实际映射 ⭐⭐⭐⭐⭐

**优先级**: 🔴 **最高** - **强烈推荐**

**研究问题**: 
- v5 优化后，不同的 Queue ID 是否真的映射到了不同的 HQD？
- 如果仍然复用，复用的具体机制是什么？

**研究价值**: 
- ✅ 回答核心疑问：Queue ID 不同但 HQD 相同
- ✅ 确认 v5 优化是否有效
- ✅ 为后续优化提供明确方向

**可行性**: ⭐⭐⭐⭐⭐ 非常可行

#### 具体实施方案

##### 方案 A: 使用 KFD Debug 日志

**步骤**:
```bash
# 1. 启用 KFD debug
sudo bash enable_kfd_debug.sh

# 2. 修改 KFD 代码，添加更详细的日志
# 文件: kfd_device_queue_manager.c
```

```c
// 在 allocate_hqd() 中添加详细日志
static int allocate_hqd(struct device_queue_manager *dqm, struct queue *q)
{
    // ... 现有代码 ...
    
    pr_info("KFD_HQD_ALLOC: PID=%d, QueueID=%u, Pipe=%d, Queue=%d, Doorbell=0x%llx\n",
            current->pid, 
            q->properties.queue_id,  // 软件 Queue ID
            q->pipe,                  // 硬件 Pipe
            q->queue,                 // 硬件 Queue
            q->properties.doorbell_off);  // Doorbell 偏移
    
    // ... 现有代码 ...
}
```

**预期输出**:
```bash
# 如果确实复用（怀疑的情况）
KFD_HQD_ALLOC: PID=3935030, QueueID=216, Pipe=0, Queue=0, Doorbell=0x0
KFD_HQD_ALLOC: PID=3935030, QueueID=217, Pipe=1, Queue=0, Doorbell=0x800
KFD_HQD_ALLOC: PID=3935031, QueueID=220, Pipe=0, Queue=0, Doorbell=0x0  ⚠️ 重复
KFD_HQD_ALLOC: PID=3935031, QueueID=221, Pipe=1, Queue=0, Doorbell=0x800  ⚠️ 重复

# 如果没有复用（理想情况）
KFD_HQD_ALLOC: PID=3935030, QueueID=216, Pipe=0, Queue=0, Doorbell=0x0
KFD_HQD_ALLOC: PID=3935030, QueueID=217, Pipe=1, Queue=0, Doorbell=0x800
KFD_HQD_ALLOC: PID=3935031, QueueID=220, Pipe=0, Queue=1, Doorbell=0x1000  ✅ 不同
KFD_HQD_ALLOC: PID=3935031, QueueID=221, Pipe=1, Queue=1, Doorbell=0x1800  ✅ 不同
```

**判断标准**:
- 如果看到相同的 `(Pipe, Queue)` 组合 → 确认复用
- 如果 `Doorbell` 偏移相同 → 确认共享同一个 Doorbell
- 如果都不同 → 排除 HQD 复用，转向调度器层问题

**工作量**: 1-2 天
- 修改 KFD 代码：0.5 天
- 重新编译和测试：0.5 天
- 分析结果：0.5 天

##### 方案 B: 使用 umr 工具查看硬件状态

**步骤**:
```bash
# 1. 安装 umr (User Mode Register Debugger)
git clone https://gitlab.freedesktop.org/tomstdenis/umr.git
cd umr && mkdir build && cd build
cmake .. && make && sudo make install

# 2. 在测试运行时查看 HQD 状态
sudo umr -cpc  # 查看 Compute Pipe 配置

# 3. 对比不同进程的 HQD 使用情况
```

**预期输出**:
```
Compute Pipe 0:
  Queue 0: Active, Ring Base=0x..., Write Ptr=0x..., PASID=123  ← 进程 1
  Queue 1: Active, Ring Base=0x..., Write Ptr=0x..., PASID=124  ← 进程 2
  Queue 2: Inactive
  ...

Compute Pipe 1:
  Queue 0: Active, Ring Base=0x..., Write Ptr=0x..., PASID=123  ← 进程 1
  Queue 1: Active, Ring Base=0x..., Write Ptr=0x..., PASID=124  ← 进程 2
  ...
```

**判断标准**:
- 统计 Active 的 HQD 数量
- 如果 4 进程 × 4 队列 = 16 队列，但只有 8 个 Active HQD → 确认复用
- 检查相同 PASID 的队列是否使用相同的 HQD

**工作量**: 2-3 天
- 学习 umr 使用：1 天
- 设计测试和采集数据：1 天
- 分析结果：1 天

##### 方案 C: 修改测试程序，设置不同 Queue 属性

**步骤**:
```cpp
// 测试程序：为不同进程/队列设置不同属性
int process_id = get_process_id();  // 0, 1, 2, 3
int queue_id = get_queue_id();      // 0, 1, 2, 3

// 方案 1: 设置不同优先级
int priority = (process_id * 4 + queue_id) % 3 - 1;  // -1, 0, 1 循环
hipStreamCreateWithPriority(&stream, 0, priority);

// 方案 2: 设置不同 CU Mask
uint32_t cu_mask = 1 << (process_id * 4 + queue_id);  // 不同的 CU
hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
hipExtStreamSetCUMask(&stream, cu_mask);

// 运行测试
run_benchmark();
```

**预期结果**:
- 如果设置不同属性后性能提升 → 确认原来是属性相同导致复用
- 如果性能仍无提升 → 排除属性问题，转向调度器层

**工作量**: 1 天
- 修改测试程序：0.5 天
- 运行和分析：0.5 天

#### 建议的执行顺序

1. **先执行方案 A（KFD Debug 日志）** - 最直接，最明确
2. **如果方案 A 确认复用，执行方案 C（测试不同属性）** - 验证解决方案
3. **如果方案 A 排除复用，使用方案 B（umr）** - 深入硬件层验证

---

### 方向 2: 研究 PASID + Queue ID 组合的全局唯一性 ⭐⭐⭐⭐

**优先级**: 🟠 **高** - **建议执行**（如果方向 1 确认复用）

**研究问题**: 
- 能否通过 `(PASID, Queue ID)` 组合确保全局唯一性？
- KFD 是否真的使用 PASID 来区分不同进程的队列？

**研究价值**: 
- ✅ 理解 KFD 的队列标识机制
- ✅ 为优化 v6 提供理论基础
- ✅ 可能发现 PASID 使用不当的 bug

**可行性**: ⭐⭐⭐⭐ 可行

#### 具体实施方案

##### 方案 A: 追踪 PASID 分配和使用

**步骤**:
```c
// 1. 在 KFD 代码中追踪 PASID 分配
// 文件: kfd_process.c
static int kfd_process_init_pasid(struct kfd_process *process)
{
    // ... 现有代码 ...
    pr_info("KFD_PASID_ALLOC: PID=%d, PASID=%u\n", 
            process->lead_thread->pid, process->pasid);
    // ...
}

// 2. 在 Queue 操作中追踪 PASID 使用
// 文件: kfd_device_queue_manager.c
static int create_queue_cpsch(...) {
    // ... 现有代码 ...
    pr_info("KFD_QUEUE_PASID: QueueID=%u, PASID=%u, HQD=(Pipe=%d, Queue=%d)\n",
            q->properties.queue_id, q->process->pasid, q->pipe, q->queue);
    // ...
}
```

**预期发现**:
- 每个进程有独立的 PASID
- 检查 KFD 是否在 HQD 分配时考虑 PASID
- 如果 HQD 分配只看 Queue ID 而忽略 PASID → 发现 bug

**工作量**: 2 天

##### 方案 B: 实现 PASID-aware 的 Queue ID 分配

**优化思路**:
```c
// v6 版本：基于 PASID 的全局 Queue ID
static int find_available_queue_slot_v6(struct process_queue_manager *pqm,
                                        unsigned int *qid)
{
    uint32_t pasid = pqm->process->pasid;
    uint32_t base_queue_id = pasid * 4;  // 每个 PASID 4 个队列
    
    // 在全局 Queue ID 空间中分配
    for (found = base_queue_id; found < base_queue_id + 4; found++) {
        if (!test_bit(found, global_queue_bitmap)) {  // 全局 bitmap
            set_bit(found, global_queue_bitmap);
            *qid = found;
            return 0;
        }
    }
    return -ENOMEM;
}
```

**测试**:
- 运行 4 进程测试
- 验证 Queue ID 是否基于 PASID 分配
- 测试性能是否提升

**工作量**: 3 天

---

### 方向 3: 对比 MES vs CPSCH 调度器 ⭐⭐⭐

**优先级**: 🟡 **中** - **可选执行**

**研究问题**: 
- MES (硬件调度) 和 CPSCH (软件调度) 在 Queue 管理上有何差异？
- 这个问题在 MES 架构上是否存在？

**研究价值**: 
- ✅ 理解架构差异
- ✅ 判断是否是 CPSCH 特有问题
- ⚠️ 但对 MI308X 优化可能帮助不大（MI308X 用 CPSCH）

**可行性**: ⭐⭐ 较难（需要 MES 架构的 GPU）

#### 具体实施方案

**条件**: 需要支持 MES 的 GPU (如 MI300A, MI250X, RX 7900)

**步骤**:
```bash
# 1. 检查调度器类型
cat /sys/module/amdgpu/parameters/mes
# 1 = MES, 0 = CPSCH

# 2. 在 MES GPU 上运行相同的 4 进程测试
./run_4proc_test.sh

# 3. 对比性能
```

**预期发现**:
- 如果 MES 上性能正常 → 确认是 CPSCH 特有问题
- 如果 MES 上性能也差 → 更深层的架构问题

**工作量**: 5 天（包括获取硬件、环境配置）

**建议**: **暂缓执行**，先完成方向 1 和 2

---

### 方向 4: 深入调度器层优化 ⭐⭐⭐⭐⭐

**优先级**: 🔴 **最高** - **强烈推荐**（如果方向 1 排除 HQD 复用）

**研究问题**: 
- `active_runlist` 标志导致的调度器串行化
- 如何实现真正的并发调度？

**研究价值**: 
- ✅ 攻克主要瓶颈 🔴
- ✅ 可能带来实质性性能提升
- ✅ 解决根本问题

**可行性**: ⭐⭐⭐ 中等（需要深入理解 CPSCH）

#### 核心问题

```c
// 文件: kfd_device_queue_manager.c
static int map_queues_cpsch(struct device_queue_manager *dqm)
{
    // 问题：同一时间只能有一个 active runlist
    if (dqm->active_runlist)
        return 0;  // 直接返回，不能并发！
    
    // 设置 runlist
    retval = pm_send_set_resources(&dqm->packet_mgr, &res);
    dqm->active_runlist = true;  // 设置标志
    
    return 0;
}
```

#### 优化方案

##### 方案 A: 支持多个 Runlist

```c
// v6 优化：支持进程级 Runlist
static int map_queues_cpsch_v6(struct device_queue_manager *dqm)
{
    struct kfd_process *process = current_process;
    
    // 检查该进程的 runlist
    if (process->active_runlist)
        return 0;
    
    // 为该进程创建独立的 runlist
    retval = pm_send_set_resources_for_process(&dqm->packet_mgr, 
                                                &res, 
                                                process->pasid);
    process->active_runlist = true;  // 进程级标志
    
    return 0;
}
```

**工作量**: 5-7 天（需要深入修改 CPSCH）

##### 方案 B: 异步 Runlist 提交

```c
// 不等待前一个 runlist 完成，立即提交下一个
static int map_queues_cpsch_async(struct device_queue_manager *dqm)
{
    // 不检查 active_runlist
    // 使用不同的 PM4 packet 队列
    int packet_queue_id = current->pid % NUM_PACKET_QUEUES;
    
    retval = pm_send_set_resources_async(&dqm->packet_mgr, 
                                         &res, 
                                         packet_queue_id);
    return 0;
}
```

**工作量**: 7-10 天（需要修改 PM4 机制）

**建议**: 从方案 A 开始，渐进式优化

---

### 方向 5: Ring 层优化 ⭐⭐⭐

**优先级**: 🟡 **中** - **次要优化**

**研究问题**: 
- 不同 HQD 是否共享同一个 Compute Ring？
- 能否为不同进程分配独立的 Ring？

**研究价值**: 
- ✅ 消除 Ring 层瓶颈 🟡
- ⚠️ 可能需要硬件支持

**可行性**: ⭐⭐ 较难（可能受硬件限制）

#### 具体实施方案

```c
// 文件: amdgpu_ring.c
// 当前：所有 Compute Queue 共享一个 Ring
static int amdgpu_ring_init(...) {
    // 检查是否有多个 Compute Ring 可用
    int num_rings = adev->gfx.num_compute_rings;
    
    // 为不同进程分配不同的 Ring
    int ring_id = pasid % num_rings;
    ring = &adev->gfx.compute_ring[ring_id];
    
    // ...
}
```

**挑战**: 
- 硬件可能只有有限的 Ring 数量
- 需要确认 MI308X 的 Ring 数量

**工作量**: 5 天

**建议**: **暂缓执行**，先完成方向 1、2、4

---

## 🎯 推荐的研究路线

### 短期（1-2 周）- 诊断和快速验证

**第一优先级**:
1. ✅ **方向 1 - 方案 A**: KFD Debug 日志验证 HQD 映射（1-2 天）
   - 明确是否存在 HQD 复用
   
2. ✅ **方向 1 - 方案 C**: 测试不同 Queue 属性（1 天）
   - 如果方案 A 确认复用，测试解决方案

### 中期（2-4 周）- 深度优化

**根据短期结果选择**:

**情况 A: 如果确认 HQD 复用**
1. ✅ **方向 2 - 方案 B**: 实现 PASID-aware Queue ID 分配（3 天）
2. ✅ **方向 1 - 方案 B**: umr 工具验证（2-3 天）

**情况 B: 如果排除 HQD 复用**
1. ✅ **方向 4 - 方案 A**: 支持多个 Runlist（5-7 天）
2. ✅ **方向 4 - 方案 B**: 异步 Runlist 提交（7-10 天）

### 长期（1-2 月）- 系统性优化

1. ✅ **方向 5**: Ring 层优化（5 天）
2. ✅ **方向 3**: MES vs CPSCH 对比研究（5 天，可选）
3. ✅ 综合所有优化，测试最终性能

---

## 📊 预期成果

### 最佳情况（所有优化都成功）

```
优化前: 4 进程 QPS = 59.0
    ↓
v5 (Queue ID 不重叠): QPS = 59.0 (无提升)
    ↓
v6 (修复 HQD 复用): QPS = 80-90 (提升 35-50%)
    ↓
v7 (调度器优化): QPS = 150-200 (提升 2.5-3.4x)
    ↓
v8 (Ring 优化): QPS = 200-250 (提升 3.4-4.2x)
    ↓
理论上限: 4 × 107 = 428 QPS (线性扩展)
```

### 现实预期

- **v6 (HQD 修复)**: 50% 性能提升（QPS ~90）
- **v7 (调度器优化)**: 2x 性能提升（QPS ~150-180）
- **v8 (Ring 优化)**: 额外 20-30% 提升（QPS ~200）

**总提升**: 约 3-3.5x（QPS 从 59 → ~180-200）

---

## ⚠️ 风险和挑战

### 技术风险

1. **HQD 复用可能是设计如此**
   - 硬件限制（只有 32 个 HQD）
   - CPSCH 架构的固有限制
   - 修复可能需要硬件支持

2. **调度器优化可能破坏稳定性**
   - `active_runlist` 可能有其他用途
   - 并发调度可能导致死锁或资源竞争
   - 需要大量测试

3. **性能提升可能不如预期**
   - 可能还有其他未发现的瓶颈
   - 硬件层面的限制

### 时间成本

- **诊断阶段**: 1-2 周
- **HQD 优化**: 2-3 周
- **调度器优化**: 3-4 周
- **测试和验证**: 2 周
- **总计**: 2-3 个月

### 建议

1. **先完成诊断**（方向 1）- 非常重要
   - 明确瓶颈在哪一层
   - 避免盲目优化

2. **渐进式优化**
   - 一次解决一个问题
   - 每步都验证效果

3. **设置里程碑**
   - 第 1 周：完成 HQD 映射验证
   - 第 2-3 周：尝试 HQD 优化
   - 第 4-7 周：调度器优化
   - 第 8 周：综合测试

4. **准备 Plan B**
   - 如果 CPSCH 优化太难
   - 考虑迁移到 MES 架构（MI300 系列）
   - 或使用其他并发策略（多 GPU、多进程隔离）

---

## 📚 总结

### Queue ID 分配实验的价值

**已完成的价值**: ⭐⭐⭐⭐ 高
- ✅ 揭示了多层瓶颈
- ✅ 理解了软件抽象和硬件实现的映射
- ✅ 为后续优化奠定基础

**继续研究的价值**: ⭐⭐⭐⭐⭐ 非常高
- ✅ 方向 1（HQD 映射验证）- **必须执行**
- ✅ 方向 2（PASID-aware 分配）- **高价值**
- ✅ 方向 4（调度器优化）- **核心优化**
- ⚠️ 方向 3（MES 对比）- 可选
- ⚠️ 方向 5（Ring 优化）- 次要

### 最重要的下一步

🎯 **立即执行方向 1 - 方案 A（KFD Debug 日志）**

这将明确回答：
- Queue ID 不同，HQD 是否也不同？
- 如果复用，复用的具体机制是什么？
- 优化的重点应该放在哪一层？

**预计时间**: 1-2 天  
**预计价值**: 为所有后续优化提供明确方向

---

**文档版本**: v1.0  
**创建日期**: 2026-01-19  
**下次更新**: 完成方向 1 验证后

