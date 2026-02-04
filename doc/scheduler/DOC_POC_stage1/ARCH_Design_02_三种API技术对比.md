# 三种队列抢占 API 技术深度对比

**日期**: 2026-02-03  
**目的**: 对比 POC 可用的三种 API，选择最优方案

---

## 📊 三种 API 总览

| API | 粒度 | 复杂度 | 延迟 | 状态保存 | 推荐度 |
|-----|------|--------|------|----------|--------|
| **suspend_queues** | 队列级 | ⭐ 低 | ~5ms | ✅ 自动 | ⭐⭐⭐⭐⭐ POC 首选 |
| **CWSR** | Wave级 | ⭐⭐⭐ 中 | ~10μs | ✅ 硬件 | ⭐⭐⭐⭐ 优化阶段 |
| **stop_sched** | GPU级 | ⭐ 极低 | ~1ms | ❌ 无 | ⭐⭐ 不推荐 |

---

## 🔍 API 1: KFD_IOC_DBG_TRAP_SUSPEND_QUEUES

### 基本信息

**来源**: KFD 调试接口  
**用途**: GPU 程序调试（GDB 断点时暂停队列）  
**位置**: `kfd_chardev.c:3310-3321`

### 工作原理

```
用户空间
    ↓ ioctl(KFD_IOC_DBG_TRAP_SUSPEND_QUEUES)
┌──────────────────────────────────────────────┐
│ KFD: suspend_queues()                        │
│   ↓                                          │
│   遍历 queue_array                           │
│   ↓                                          │
│   pqm_get_wave_state()                       │
│   ↓                                          │
│   evict_process_queues_cpsch()  ← DQM 层    │
│   ↓                                          │
│   ├─ unmap_queues() - 发送 PM4 UNMAP        │
│   ├─ 触发 CWSR 保存 Wave 状态               │
│   └─ 从 Runlist 移除队列                    │
└──────────────────────────────────────────────┘
    ↓ PM4 Commands
┌──────────────────────────────────────────────┐
│ CP Scheduler (CPSCH)                         │
│   • 处理 UNMAP_QUEUES                        │
│   • 触发 CWSR (Wave save)                    │
│   • 队列变为 inactive                        │
└──────────────────────────────────────────────┘
```

### 代码实现

**内核侧**:
```c
// kfd_chardev.c:3310
case KFD_IOC_DBG_TRAP_SUSPEND_QUEUES:
    r = suspend_queues(target,
            args->suspend_queues.num_queues,
            args->suspend_queues.grace_period,
            args->suspend_queues.exception_mask,
            (uint32_t *)args->suspend_queues.queue_array_ptr);
    break;

// 实际实现
static int suspend_queues(struct kfd_process *p,
                         uint32_t num_queues,
                         uint32_t grace_period,
                         uint64_t exception_mask,
                         uint32_t *queue_ids)
{
    int ret;
    
    // 遍历队列
    for (int i = 0; i < num_queues; i++) {
        struct queue *q = pqm_get_queue_by_qid(&p->pqm, queue_ids[i]);
        if (!q)
            continue;
        
        // 获取 wave 状态（触发 CWSR）
        ret = pqm_get_wave_state(p, queue_ids[i], ...);
        
        // Evict 队列（从 Runlist 移除）
        ret = evict_process_queues_cpsch(q->device->dqm, &p->pqm, ...);
    }
    
    return ret;
}
```

**用户侧**:
```c
#include <linux/kfd_ioctl.h>

int fd = open("/dev/kfd", O_RDWR);

struct kfd_ioctl_dbg_trap_args args = {0};
args.op = KFD_IOC_DBG_TRAP_SUSPEND_QUEUES;
args.suspend_queues.num_queues = 1;
args.suspend_queues.grace_period = 1000;  // 1ms
args.suspend_queues.queue_array_ptr = (uint64_t)&queue_id;

int ret = ioctl(fd, AMDKFD_IOC_DBG_TRAP, &args);
```

### 优势

✅ **已有完整实现** - 不需要修改内核代码  
✅ **自动触发 CWSR** - Wave-level 状态保存  
✅ **批量操作** - 可以一次暂停多个队列  
✅ **grace_period 支持** - 优雅停止（等待当前 wave 完成）  
✅ **快速实施** - POC 可以在 1-2 天内完成  

### 劣势

⚠️ **需要 root 权限** - 调试接口权限要求  
⚠️ **性能开销** - ioctl 系统调用（~1-10μs）  
⚠️ **可能的副作用** - 原本用于调试，可能有额外检查  

---

## 🔍 API 2: CWSR (直接使用)

### 基本信息

**来源**: MQD Manager 层  
**用途**: Checkpoint/Restore (CRIU 使用)  
**位置**: 
- `kfd_process_queue_manager.c:809` - `pqm_checkpoint_mqd()`
- `kfd_mqd_manager_v9.c:436` - `checkpoint_mqd()`
- `kfd_mqd_manager_v9.c:448` - `restore_mqd()`

### 工作原理

```
用户空间
    ↓ 自定义 ioctl (需要新增)
┌──────────────────────────────────────────────┐
│ KFD: 自定义抢占接口 (需要实现)               │
│   ↓                                          │
│   pqm_checkpoint_mqd()  ← CRIU 函数复用      │
│   ↓                                          │
│   dqm->ops.checkpoint_mqd()                  │
│   ↓                                          │
│   mqd_mgr->checkpoint_mqd()  ← MQD Manager   │
│   ├─ memcpy(mqd_backup, mqd, ...)           │
│   └─ memcpy(ctl_stack_backup, ctl_stack,..) │
│   ↓                                          │
│   mqd_mgr->destroy_mqd()                     │
│   ├─ 发送 PM4 UNMAP_QUEUES                  │
│   └─ 触发 CWSR                               │
└──────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────┐
│ CP Scheduler                                 │
│   • CWSR 机制保存 Wave 状态                  │
└──────────────────────────────────────────────┘
```

### 代码实现（需要新增）

**内核侧** (需要新增 ioctl):
```c
// kfd_chardev.c 中新增
case AMDKFD_IOC_PREEMPT_QUEUE:
{
    struct kfd_ioctl_preempt_queue_args *args = data;
    struct kfd_process *p;
    struct process_queue_node *pqn;
    int ret;
    
    p = kfd_get_process(current);
    pqn = get_queue_by_qid(&p->pqm, args->queue_id);
    if (!pqn)
        return -EINVAL;
    
    // 直接使用 CRIU 的函数
    ret = pqm_checkpoint_mqd(&p->pqm, args->queue_id,
                            pqn->q->snapshot.mqd_backup,
                            pqn->q->snapshot.ctl_stack_backup);
    
    return ret;
}
```

**用户侧**:
```c
struct kfd_ioctl_preempt_queue_args {
    uint32_t queue_id;
    uint32_t timeout_ms;
};

int fd = open("/dev/kfd", O_RDWR);
struct kfd_ioctl_preempt_queue_args args = {
    .queue_id = offline_queue_id,
    .timeout_ms = 1000
};

int ret = ioctl(fd, AMDKFD_IOC_PREEMPT_QUEUE, &args);
```

### 优势

✅ **最低延迟** - 直接调用 CWSR，~10μs  
✅ **Wave-level 精度** - 最细粒度  
✅ **可复用 CRIU 代码** - 已验证的实现  
✅ **完全控制** - 无额外的调试逻辑  

### 劣势

⚠️ **需要修改内核** - 新增 ioctl 和相关代码  
⚠️ **内存管理** - 需要管理 snapshot 内存  
⚠️ **开发周期** - 需要 3-5 天实现和测试  

---

## 🔍 API 3: amdgpu_amdkfd_stop_sched

### 基本信息

**来源**: AMDGPU KFD 接口  
**用途**: 停止/启动整个 KFD 调度器  
**位置**: `amdgpu_amdkfd.c:898-903`

### 工作原理

```
用户空间
    ↓ 自定义调用 (需要导出接口)
┌──────────────────────────────────────────────┐
│ amdgpu_amdkfd_stop_sched(adev, node_id)      │
│   ↓                                          │
│   kgd2kfd_stop_sched(kfd_dev, node_id)       │
│   ↓                                          │
│   DQM: stop_cpsch()                          │
│   ├─ unmap ALL queues                        │
│   ├─ 停止 Runlist 提交                       │
│   └─ 标记调度器为 inactive                   │
└──────────────────────────────────────────────┘
```

### 代码实现

```c
// amdgpu_amdkfd.c:898
int amdgpu_amdkfd_stop_sched(struct amdgpu_device *adev, uint32_t node_id)
{
    if (!adev->kfd.init_complete)
        return 0;
    
    return kgd2kfd_stop_sched(adev->kfd.dev, node_id);
}

// 恢复
int amdgpu_amdkfd_start_sched(struct amdgpu_device *adev, uint32_t node_id);
```

### 优势

✅ **极其简单** - 一行调用  
✅ **无需修改** - 已有接口  

### 劣势

❌ **粒度太粗** - 停止所有队列（包括 Online）  
❌ **不适合场景** - 无法区分 Online/Offline  
❌ **性能影响大** - 停止整个 GPU 调度  

**结论**: ❌ 不推荐用于 Online/Offline 场景

---

## 🎯 方案对比总结

### 按使用场景推荐

#### POC Stage 1: 快速概念验证 (1-2 周)
```
推荐: KFD_IOC_DBG_TRAP_SUSPEND_QUEUES ⭐⭐⭐⭐⭐

原因:
  ✅ 无需修改内核代码
  ✅ 已有完整实现
  ✅ 1-2 天可完成测试
  ✅ 能验证核心概念
```

#### POC Stage 2: 性能优化 (2-3 周)
```
推荐: CWSR API (直接使用) ⭐⭐⭐⭐⭐

原因:
  ✅ 更低延迟 (~10μs)
  ✅ 更精确控制
  ✅ 可复用 CRIU 代码
  ✅ 为 Stage 3 做准备
```

#### Production: 完整调度器 (1-2 月)
```
推荐: 内核态监控 + CWSR ⭐⭐⭐⭐⭐

原因:
  ✅ 无 ioctl 开销
  ✅ 自动化调度
  ✅ 完整的优先级管理
  ✅ 详见 TODOLIST.md
```

---

## 📐 具体实施对比

### 方案A: suspend_queues (POC Stage 1)

**实施复杂度**: ⭐☆☆☆☆

```python
# 实施步骤 (2-3 天)
Day 1: 
  - 编写 C 库封装 (libgpreempt_poc.so)
  - 实现 suspend/resume 函数包装

Day 2:
  - 实现队列识别机制
  - 解析 MQD debugfs

Day 3:
  - 编写 Python Test Framework
  - 运行基本测试
```

**代码量**: ~500 行 (C + Python)

**依赖**:
- ✅ 无需修改内核
- ✅ 只需用户空间代码

---

### 方案B: CWSR 直接使用 (POC Stage 2)

**实施复杂度**: ⭐⭐⭐☆☆

```python
# 实施步骤 (1-2 周)
Week 1:
  Day 1-2: 新增内核 ioctl 接口
    - AMDKFD_IOC_PREEMPT_QUEUE
    - AMDKFD_IOC_RESUME_QUEUE
  
  Day 3-4: 集成 CWSR API
    - 调用 pqm_checkpoint_mqd()
    - 分配 snapshot 内存
  
  Day 5: 用户空间库更新
    - 更新 libgpreempt_poc.so

Week 2:
  Day 6-7: 测试和调试
  Day 8-10: 性能测试和优化
```

**代码量**: ~1000 行 (内核 + 用户空间)

**依赖**:
- ⚠️ 需要修改内核代码
- ⚠️ 需要 DKMS 重新编译
- ✅ 可复用 CRIU 代码

---

### 方案C: stop_sched (不推荐)

**实施复杂度**: ⭐☆☆☆☆

```c
// 极简实现 (半天)
amdgpu_amdkfd_stop_sched(adev, 0);  // 停止所有
// Online 任务执行...
amdgpu_amdkfd_start_sched(adev, 0); // 恢复所有
```

**问题**:
- ❌ 停止所有队列（包括 Online）
- ❌ 无法区分优先级
- ❌ 不满足测试场景需求

**结论**: ❌ 不适用于 Online/Offline 场景

---

## 🚀 推荐实施路线图

### 路线图: 从简单到完整

```
┌─────────────────────────────────────────────────────────────┐
│ POC Stage 1: suspend_queues                                  │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│ 时间: 1-2 周                                                 │
│ 复杂度: ⭐☆☆☆☆                                              │
│ 目标: 验证概念可行性                                         │
│                                                              │
│ 交付物:                                                      │
│  ✅ libgpreempt_poc.so (C 库)                               │
│  ✅ test_priority_scheduling.py (测试框架)                  │
│  ✅ 性能测试报告                                             │
│  ✅ 可行性验证                                               │
└─────────────────────────────────────────────────────────────┘
                        ↓ 如果成功
┌─────────────────────────────────────────────────────────────┐
│ POC Stage 2: CWSR 直接使用                                   │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│ 时间: 2-3 周                                                 │
│ 复杂度: ⭐⭐⭐☆☆                                             │
│ 目标: 性能优化，降低延迟                                     │
│                                                              │
│ 交付物:                                                      │
│  ✅ 内核新增 ioctl 接口                                      │
│  ✅ 直接使用 pqm_checkpoint_mqd()                           │
│  ✅ 延迟 < 1ms                                              │
│  ✅ 完整性能测试                                             │
└─────────────────────────────────────────────────────────────┘
                        ↓ 如果需要自动化
┌─────────────────────────────────────────────────────────────┐
│ Production: 内核态调度器 (TODOLIST.md)                       │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│ 时间: 1-2 月                                                 │
│ 复杂度: ⭐⭐⭐⭐⭐                                            │
│ 目标: 生产级调度器                                           │
│                                                              │
│ 交付物:                                                      │
│  ✅ 完整的监控框架 (Phase 1)                                 │
│  ✅ 自动优先级调度 (Phase 2)                                 │
│  ✅ 用户态接口 (Phase 3)                                     │
│  ✅ 高级特性 (Phase 4-5)                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 💡 为什么分阶段实施？

### Stage 1 的价值

**风险最小化**:
- 无需修改内核 → 不会导致系统不稳定
- 快速验证概念 → 1-2 周就能看到结果
- 如果不可行 → 早期止损

**快速迭代**:
```
Week 1: 实现基础框架
Week 2: 测试和测量性能
  ↓
  结果: suspend_queues 延迟太高 (~5ms)
  ↓
  决策: 升级到 Stage 2 (CWSR)
```

### Stage 2 的价值

**性能优化**:
- 延迟从 ~5ms 降到 ~10μs (500倍提升)
- 直接使用 CWSR，无调试接口开销

**为 Production 铺路**:
- CWSR API 集成经验
- 性能基准数据
- 问题和限制清单

---

## 🔬 技术细节深入对比

### suspend_queues vs CWSR 的底层差异

#### suspend_queues 的额外开销

```c
// suspend_queues 调用路径（较长）
ioctl
 → kfd_ioctl_dbg_trap()
   → suspend_queues()
     → pqm_get_wave_state()        // ← 额外的状态读取
       → evict_process_queues_cpsch()
         → unmap_queues()
           → pm4_unmap_queues()
             → CP Scheduler
               → CWSR (Wave save)

// 预估开销
ioctl:                ~1-5μs
suspend_queues logic: ~100μs (遍历 + 验证)
evict_process:        ~1ms (DQM 层处理)
CWSR 本身:            ~10μs
──────────────────────────────
总计:                 ~1-5ms
```

#### CWSR 直接使用（理想情况）

```c
// 直接 CWSR 调用路径（短）
ioctl (自定义)
 → kfd_ioctl_preempt_queue()
   → pqm_checkpoint_mqd()      // ← CRIU 复用
     → dqm->ops.checkpoint_mqd()
       → mqd_mgr->checkpoint_mqd()  // 2 个 memcpy
       → mqd_mgr->destroy_mqd()     // PM4 UNMAP
         → CP Scheduler
           → CWSR (Wave save)

// 预估开销
ioctl:                ~1-5μs
checkpoint_mqd:       ~10μs (memcpy)
destroy_mqd:          ~10μs (PM4)
CWSR 本身:            ~10μs
──────────────────────────────
总计:                 ~30-100μs
```

**差异**: **Stage 1 (~1-5ms) vs Stage 2 (~100μs) = 50倍性能差距**

---

## 📋 决策矩阵

### 什么时候使用哪个方案？

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| **快速概念验证** | suspend_queues | 无需修改内核,1周完成 |
| **性能要求高** (< 1ms) | CWSR 直接使用 | 延迟最低 |
| **需要自动化调度** | 内核态调度器 | 无 ioctl 开销 |
| **调试和开发** | suspend_queues | 易于调试 |
| **生产环境** | 内核态调度器 | 完整功能 |

---

## 🛠️ POC Stage 1 实施 Checklist

### 准备阶段 (Day 1)

- [ ] 确认 KFD_IOC_DBG_TRAP_SUSPEND_QUEUES 在当前内核版本可用
- [ ] 检查权限要求（是否需要 root）
- [ ] 准备测试 GPU 和环境

### 开发阶段 (Day 2-4)

- [ ] 实现 libgpreempt_poc.so
  - [ ] gpreempt_suspend_queues()
  - [ ] gpreempt_resume_queues()
  - [ ] gpreempt_get_all_queues()

- [ ] 实现队列识别机制
  - [ ] 解析 MQD debugfs
  - [ ] 优先级过滤

- [ ] 实现 Test Framework
  - [ ] GPreemptScheduler 类
  - [ ] 监控线程
  - [ ] 测试用例

### 测试阶段 (Day 5-7)

- [ ] 功能测试
  - [ ] Test Case 1: 基本抢占
  - [ ] Test Case 2: 频繁抢占
  - [ ] Test Case 3: 边界条件

- [ ] 性能测试
  - [ ] 测量 Online 延迟
  - [ ] 测量 Offline 吞吐量
  - [ ] 测量 suspend/resume 开销

- [ ] 稳定性测试
  - [ ] 长时间运行（1小时+）
  - [ ] 异常处理

### 文档阶段 (Day 7-10)

- [ ] 测试报告
- [ ] 性能分析报告
- [ ] Stage 2 实施建议

---

## 📊 预期结果

### 性能预期

| 指标 | 预期值 | 可接受范围 |
|------|--------|-----------|
| Online 端到端延迟 | 5-10ms | < 50ms |
| Suspend 延迟 | 1-3ms | < 5ms |
| Resume 延迟 | 1-3ms | < 5ms |
| Offline 吞吐量损失 | 5-10% | < 20% |

### 如果性能不满足怎么办？

**如果 Stage 1 延迟 > 10ms**:
```
原因分析:
  1. ioctl 系统调用开销
  2. suspend_queues 内部逻辑复杂
  3. DQM 层的额外处理

解决方案:
  → 升级到 Stage 2 (CWSR 直接使用)
  → 绕过 debugfs trap 接口
  → 预期延迟降低到 ~100μs
```

---

## 🎯 成功标准

### 必须达成 (Must Have)

- [x] Online-AI 能抢占 Offline-AI
- [x] Offline-AI 正确恢复（无数据丢失）
- [x] 系统稳定运行（无崩溃）

### 应该达成 (Should Have)

- [ ] Online 延迟 < 50ms
- [ ] Offline 吞吐量损失 < 20%
- [ ] 连续运行 1 小时无错误

### 最好达成 (Nice to Have)

- [ ] Online 延迟 < 10ms
- [ ] Offline 吞吐量损失 < 10%
- [ ] 完整的性能分析报告

---

## 📚 相关文档

### Stage 1 相关

- **ARCH_Design_01_POC_Stage1_实施方案.md** - 本文档
- **test_scenaria.md** - 测试场景定义

### 技术背景

- **TODOLIST.md** - 完整实施计划（Stage 3）
- **CWSR_API_USAGE_REFERENCE.md** - CWSR API 参考
- **CRIU_CODE_REUSE_ANALYSIS.md** - CRIU 代码复用分析

---

**结论**: POC Stage 1 使用 `KFD_IOC_DBG_TRAP_SUSPEND_QUEUES` 进行快速验证，复杂度低，风险小，能在 1-2 周内完成概念验证。如果性能满足要求，可以直接使用；如果不满足，再升级到 Stage 2 (CWSR 直接使用)。✅
