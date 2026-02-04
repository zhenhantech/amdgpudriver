# 架构设计修正 - 基于 CWSR API 实际使用

**日期**: 2026-01-29  
**状态**: 🔴 关键修正（必须更新）  
**影响**: ARCH_Design_02 和 ARCH_Design_03

---

## 🎯 问题总结

通过对比 `CWSR_API_USAGE_REFERENCE.md`（基于 KFD CRIU 代码分析），发现我们的架构设计文档中 CWSR API 的使用方式存在**多处错误**。

---

## ❌ 问题 1: `destroy_mqd()` 参数错误

### 当前设计（错误）

```c
// ARCH_Design_02_AMD_GPREEMPT_redesign_v2.md:574-580
r = low_q->mqd_mgr->destroy_mqd(
    low_q->mqd_mgr,
    low_q->mqd,
    KFD_PREEMPT_TYPE_WAVEFRONT_SAVE,  // CWSR 模式
    0,                                 // timeout=0: 异步
    low_q->process->pasid              // ⚠️ 错误！不是 pasid
);
```

```c
// ARCH_Design_03_AMD_GPREEMPT_XSCHED.md: 类似问题
```

### 正确用法（基于 CRIU 代码）

```c
// 参考: kfd_queue_preempt.c:147
ret = mqd_mgr->destroy_mqd(
    mqd_mgr, 
    q->mqd, 
    KFD_PREEMPT_TYPE_WAVEFRONT_SAVE,  // type
    timeout,                           // timeout_ms
    q->pipe,                           // ⭐ pipe 编号
    q->queue                           // ⭐ queue 编号
);
```

### 修正代码

```c
// ⭐⭐⭐ 正确的 destroy_mqd 调用
static int gpreempt_preempt_queue(
    struct kfd_gpreempt_scheduler *sched,
    struct queue *low_q,
    struct queue *high_q)
{
    int r;
    ktime_t start = ktime_get();
    
    // 标记状态
    low_q->preemption_pending = true;
    low_q->preempt_start = start;
    low_q->state = QUEUE_STATE_PREEMPTED;
    
    pr_info("GPREEMPT: Preempting queue %u (prio=%d)\n",
            low_q->properties.queue_id,
            low_q->effective_priority);
    
    // ⭐⭐⭐ 关键修正：正确的参数
    r = low_q->mqd_mgr->destroy_mqd(
        low_q->mqd_mgr,
        low_q->mqd,
        KFD_PREEMPT_TYPE_WAVEFRONT_SAVE,  // preempt type
        0,                                 // timeout (0=异步)
        low_q->pipe,                       // ⭐ pipe 编号
        low_q->queue                       // ⭐ queue 编号
    );
    
    if (r == 0) {
        // 抢占立即完成
        low_q->preemption_pending = false;
    } else if (r == -EINPROGRESS) {
        // 抢占进行中（正常）
        atomic64_inc(&sched->total_preemptions);
    } else {
        // 错误
        pr_err("GPREEMPT: Preemption failed: %d\n", r);
        low_q->preemption_pending = false;
        low_q->state = QUEUE_STATE_ACTIVE;
        return r;
    }
    
    return 0;
}
```

---

## ❌ 问题 2: `restore_mqd()` 参数完全错误

### 当前设计（错误）

```c
// ARCH_Design_02:696-700
int r = q->mqd_mgr->restore_mqd(
    q->mqd_mgr,
    q->mqd,              // ⚠️ 错误！应该是 &q->mqd (double pointer)
    q->process->pasid    // ⚠️ 错误！缺少很多参数
);
```

```c
// ARCH_Design_03:572-576
int r = q->mqd_mgr->restore_mqd(
    q->mqd_mgr,
    q->mqd,
    q->process->pasid    // ⚠️ 同样的错误
);
```

### 正确签名（基于 KFD 代码）

```c
// 参考: kfd_mqd_manager_v9.c:448
void restore_mqd(
    struct mqd_manager *mm,
    void **mqd,                        // ⭐ double pointer!
    struct kfd_mem_obj *mqd_mem_obj,   // ⭐ MQD 内存对象
    uint64_t *gart_addr,               // ⭐ GART 地址
    struct queue_properties *qp,       // ⭐ 队列属性
    const void *mqd_src,               // ⭐ 源 MQD
    const void *ctl_stack_src,         // ⭐ 源 control stack
    u32 ctl_stack_size                 // ⭐ control stack 大小
);
```

### 正确用法（基于 CRIU 代码）

```c
// 参考: kfd_queue_preempt.c:236
mqd_mgr->restore_mqd(
    mqd_mgr,
    &q->mqd,                      // ⭐ double pointer
    q->mqd_mem_obj,               // MQD 内存对象
    &q->gart_mqd_addr,            // GART 地址
    &q->properties,               // 队列属性
    q->snapshot.mqd_backup,       // 保存的 MQD
    q->snapshot.ctl_stack_backup, // 保存的 control stack
    q->snapshot.ctl_stack_size    // control stack 大小
);
```

### 修正代码

```c
// ⭐⭐⭐ 正确的 restore_mqd 调用
static void gpreempt_check_resume(struct kfd_gpreempt_scheduler *sched)
{
    struct queue *q;
    unsigned long flags;
    
    spin_lock_irqsave(&sched->queue_lock, flags);
    
    list_for_each_entry(q, &sched->all_queues, sched_list) {
        if (q->state != QUEUE_STATE_PREEMPTED)
            continue;
        
        if (gpreempt_has_higher_priority_running(sched, q))
            continue;
        
        pr_info("GPREEMPT: Resuming queue %u (prio=%d)\n",
                q->properties.queue_id, q->effective_priority);
        
        // ⭐⭐⭐ 关键修正：正确的参数（8 个）
        q->mqd_mgr->restore_mqd(
            q->mqd_mgr,
            &q->mqd,                      // ⭐ double pointer
            q->mqd_mem_obj,               // MQD 内存对象
            &q->gart_mqd_addr,            // GART 地址
            &q->properties,               // 队列属性
            q->snapshot.mqd_backup,       // 保存的 MQD
            q->snapshot.ctl_stack_backup, // 保存的 control stack
            q->snapshot.ctl_stack_size    // control stack 大小
        );
        
        // ⭐⭐⭐ 关键补充：restore 后需要 load_mqd 激活队列
        int r = q->mqd_mgr->load_mqd(
            q->mqd_mgr,
            q->mqd,
            q->pipe,
            q->queue,
            &q->properties,
            q->process->mm
        );
        
        if (r == 0) {
            q->preemption_pending = false;
            q->state = QUEUE_STATE_ACTIVE;
            q->properties.is_active = true;  // ⭐ 标记为活动
            
            u64 latency = ktime_us_delta(ktime_get(), q->preempt_start);
            pr_info("GPREEMPT: Resumed (latency=%llu us)\n", latency);
            
            atomic64_inc(&q->total_resumes);
        } else {
            pr_err("GPREEMPT: Failed to load MQD: %d\n", r);
        }
    }
    
    spin_unlock_irqrestore(&sched->queue_lock, flags);
}
```

---

## ❌ 问题 3: 缺少 `checkpoint_mqd()` 调用

### 问题描述

在 `preempt` 之前，需要先调用 `checkpoint_mqd()` 保存 MQD 和 control stack，否则 `restore` 时没有数据可以恢复！

### 正确流程

```c
// ⭐⭐⭐ 完整的 Preempt 流程
static int gpreempt_preempt_queue(
    struct kfd_gpreempt_scheduler *sched,
    struct queue *low_q,
    struct queue *high_q)
{
    int r;
    
    pr_info("GPREEMPT: Preempting queue %u\n", low_q->properties.queue_id);
    
    // ⭐⭐⭐ 步骤 1: 保存 MQD 和 control stack
    //
    // 这一步非常关键！
    // 必须在 destroy_mqd 之前保存状态
    //
    low_q->mqd_mgr->checkpoint_mqd(
        low_q->mqd_mgr,
        low_q->mqd,                       // 源 MQD
        low_q->snapshot.mqd_backup,       // 目标 MQD buffer
        low_q->snapshot.ctl_stack_backup  // 目标 control stack buffer
    );
    
    pr_info("  MQD and control stack saved\n");
    pr_info("  Ring Buffer: rptr=%u, wptr=%u, pending=%u\n",
            low_q->hw_rptr, low_q->hw_wptr, low_q->pending_count);
    
    // 标记状态
    low_q->preemption_pending = true;
    low_q->preempt_start = ktime_get();
    low_q->state = QUEUE_STATE_PREEMPTED;
    
    // ⭐⭐⭐ 步骤 2: 触发硬件 CWSR
    //
    // destroy_mqd 会：
    //   1. 发送 PM4 UNMAP_QUEUES 命令
    //   2. GPU 触发 CWSR Trap Handler
    //   3. 保存 Wave 状态到 CWSR Area
    //   4. 释放 CU 资源
    //
    r = low_q->mqd_mgr->destroy_mqd(
        low_q->mqd_mgr,
        low_q->mqd,
        KFD_PREEMPT_TYPE_WAVEFRONT_SAVE,  // CWSR 模式
        0,                                 // timeout=0 (异步)
        low_q->pipe,                       // pipe 编号
        low_q->queue                       // queue 编号
    );
    
    if (r == 0 || r == -EINPROGRESS) {
        pr_info("  CWSR preemption %s\n", 
                r == 0 ? "completed" : "in progress");
        atomic64_inc(&sched->total_preemptions);
        return 0;
    } else {
        pr_err("  Preemption failed: %d\n", r);
        low_q->preemption_pending = false;
        low_q->state = QUEUE_STATE_ACTIVE;
        return r;
    }
}
```

---

## ❌ 问题 4: 缺少 `load_mqd()` 调用

### 问题描述

`restore_mqd()` 只是恢复了 MQD 数据结构，但没有激活队列。需要调用 `load_mqd()` 才能让 GPU 重新开始执行这个队列。

### 证据（CRIU 代码）

```c
// 参考: kfd_mqd_manager_v9.c:448
void restore_mqd(...)
{
    // ...
    memcpy(m, mqd_src, sizeof(*m));
    
    // ⭐ 设置为非活动状态
    m->cp_hqd_active = 0;
    qp->is_active = 0;
    
    // 需要后续调用 load_mqd 才能激活
}
```

### 正确流程

```c
// ⭐⭐⭐ 步骤 1: 恢复 MQD
q->mqd_mgr->restore_mqd(...);

// ⭐⭐⭐ 步骤 2: 加载 MQD 到 GPU（激活队列）
int r = q->mqd_mgr->load_mqd(
    q->mqd_mgr,
    q->mqd,
    q->pipe,
    q->queue,
    &q->properties,
    q->process->mm
);

if (r == 0) {
    q->properties.is_active = true;  // 标记为活动
    pr_info("Queue activated successfully\n");
}
```

---

## ❌ 问题 5: 缺少数据结构定义

### 问题描述

`checkpoint_mqd()` 和 `restore_mqd()` 需要 buffer 来保存/恢复数据，但我们的 `struct queue` 定义中缺少这些字段。

### 需要添加的字段

```c
// ============================================================================
// 文件: amd/amdkfd/kfd_priv.h
// ============================================================================

struct queue {
    // ===== 现有字段 =====
    struct queue_properties properties;
    struct mqd_manager *mqd_mgr;
    void *mqd;
    struct kfd_mem_obj *mqd_mem_obj;
    uint64_t gart_mqd_addr;
    
    // ===== GPREEMPT 新增字段 ⭐⭐⭐ =====
    
    // Ring Buffer 状态监控
    uint32_t hw_rptr;
    uint32_t hw_wptr;
    uint32_t pending_count;
    bool is_active;
    
    // ⭐⭐⭐ Snapshot（用于 checkpoint/restore）
    struct {
        void *mqd_backup;          // MQD 备份 buffer
        void *ctl_stack_backup;    // Control stack 备份 buffer
        size_t ctl_stack_size;     // Control stack 大小
        bool valid;                // Snapshot 是否有效
    } snapshot;
    
    // 抢占状态
    enum queue_state {
        QUEUE_STATE_ACTIVE,
        QUEUE_STATE_PREEMPTED,
        QUEUE_STATE_IDLE
    } state;
    
    bool preemption_pending;
    ktime_t preempt_start;
    
    // 统计
    atomic64_t total_preemptions;
    atomic64_t total_resumes;
    
    // 链表节点
    struct list_head gpreempt_list;
};
```

### 分配 Snapshot Buffer

```c
// ⭐⭐⭐ 在队列创建时分配 snapshot buffers
int kfd_gpreempt_register_queue(struct kfd_dev *dev, struct queue *q)
{
    struct kfd_gpreempt_scheduler *sched = dev->gpreempt_sched;
    
    if (!sched)
        return 0;
    
    // 获取 MQD 大小
    size_t mqd_size = sizeof(struct v9_mqd);  // 根据 GPU 版本
    
    // 获取 control stack 大小
    struct v9_mqd *mqd = (struct v9_mqd *)q->mqd;
    size_t ctl_stack_size = mqd->cp_hqd_cntl_stack_size;
    
    // 分配 MQD backup buffer
    q->snapshot.mqd_backup = kzalloc(mqd_size, GFP_KERNEL);
    if (!q->snapshot.mqd_backup) {
        pr_err("Failed to allocate MQD backup\n");
        return -ENOMEM;
    }
    
    // 分配 control stack backup buffer
    q->snapshot.ctl_stack_backup = kzalloc(ctl_stack_size, GFP_KERNEL);
    if (!q->snapshot.ctl_stack_backup) {
        kfree(q->snapshot.mqd_backup);
        pr_err("Failed to allocate control stack backup\n");
        return -ENOMEM;
    }
    
    q->snapshot.ctl_stack_size = ctl_stack_size;
    q->snapshot.valid = false;
    
    // 初始化其他字段
    q->hw_rptr = 0;
    q->hw_wptr = 0;
    q->pending_count = 0;
    q->is_active = false;
    q->state = QUEUE_STATE_IDLE;
    q->preemption_pending = false;
    atomic64_set(&q->total_preemptions, 0);
    atomic64_set(&q->total_resumes, 0);
    
    // 添加到调度器
    spin_lock(&sched->queue_lock);
    list_add_tail(&q->gpreempt_list, &sched->all_queues);
    spin_unlock(&sched->queue_lock);
    
    pr_info("GPREEMPT: Registered Q%u (prio=%d, mqd=%zu, ctl=%zu)\n",
            q->properties.queue_id, q->properties.priority,
            mqd_size, ctl_stack_size);
    
    return 0;
}
```

---

## 📊 完整的正确流程

### Preempt 流程（修正后）

```c
static int gpreempt_preempt_queue(
    struct kfd_gpreempt_scheduler *sched,
    struct queue *low_q,
    struct queue *high_q)
{
    int r;
    
    pr_info("GPREEMPT: Preempting Q%u (prio=%d) for Q%u (prio=%d)\n",
            low_q->properties.queue_id, low_q->properties.priority,
            high_q->properties.queue_id, high_q->properties.priority);
    
    // ⭐⭐⭐ 步骤 1: Checkpoint MQD
    low_q->mqd_mgr->checkpoint_mqd(
        low_q->mqd_mgr,
        low_q->mqd,
        low_q->snapshot.mqd_backup,
        low_q->snapshot.ctl_stack_backup
    );
    low_q->snapshot.valid = true;
    
    pr_info("  MQD checkpointed (mqd=%p, ctl=%p)\n",
            low_q->snapshot.mqd_backup,
            low_q->snapshot.ctl_stack_backup);
    
    // 标记状态
    low_q->preemption_pending = true;
    low_q->preempt_start = ktime_get();
    low_q->state = QUEUE_STATE_PREEMPTED;
    
    // ⭐⭐⭐ 步骤 2: 触发 CWSR
    r = low_q->mqd_mgr->destroy_mqd(
        low_q->mqd_mgr,
        low_q->mqd,
        KFD_PREEMPT_TYPE_WAVEFRONT_SAVE,
        0,                    // timeout=0 (异步)
        low_q->pipe,          // ⭐ 正确参数
        low_q->queue          // ⭐ 正确参数
    );
    
    if (r == 0 || r == -EINPROGRESS) {
        pr_info("  CWSR triggered (ret=%d)\n", r);
        atomic64_inc(&sched->total_preemptions);
        atomic64_inc(&low_q->total_preemptions);
        return 0;
    } else {
        pr_err("  CWSR failed: %d\n", r);
        low_q->preemption_pending = false;
        low_q->state = QUEUE_STATE_ACTIVE;
        low_q->snapshot.valid = false;
        return r;
    }
}
```

### Resume 流程（修正后）

```c
static void gpreempt_check_resume(struct kfd_gpreempt_scheduler *sched)
{
    struct queue *q;
    unsigned long flags;
    
    spin_lock_irqsave(&sched->queue_lock, flags);
    
    list_for_each_entry(q, &sched->all_queues, gpreempt_list) {
        if (q->state != QUEUE_STATE_PREEMPTED)
            continue;
        
        if (!q->snapshot.valid) {
            pr_warn("GPREEMPT: Q%u has no valid snapshot\n",
                    q->properties.queue_id);
            continue;
        }
        
        if (gpreempt_has_higher_priority_running(sched, q))
            continue;
        
        pr_info("GPREEMPT: Resuming Q%u (prio=%d)\n",
                q->properties.queue_id, q->properties.priority);
        
        // ⭐⭐⭐ 步骤 1: Restore MQD
        q->mqd_mgr->restore_mqd(
            q->mqd_mgr,
            &q->mqd,                      // ⭐ double pointer
            q->mqd_mem_obj,
            &q->gart_mqd_addr,
            &q->properties,
            q->snapshot.mqd_backup,       // ⭐ 从 snapshot 恢复
            q->snapshot.ctl_stack_backup, // ⭐ 从 snapshot 恢复
            q->snapshot.ctl_stack_size
        );
        
        pr_info("  MQD restored from snapshot\n");
        
        // ⭐⭐⭐ 步骤 2: Load MQD（激活队列）
        int r = q->mqd_mgr->load_mqd(
            q->mqd_mgr,
            q->mqd,
            q->pipe,
            q->queue,
            &q->properties,
            q->process->mm
        );
        
        if (r == 0) {
            // 成功
            q->preemption_pending = false;
            q->state = QUEUE_STATE_ACTIVE;
            q->properties.is_active = true;
            q->snapshot.valid = false;  // 清除 snapshot
            
            u64 latency = ktime_us_delta(ktime_get(), q->preempt_start);
            pr_info("  Q%u resumed successfully (latency=%llu us)\n",
                    q->properties.queue_id, latency);
            
            atomic64_inc(&q->total_resumes);
        } else {
            pr_err("  Failed to load MQD: %d\n", r);
            // 保持 PREEMPTED 状态，下次重试
        }
    }
    
    spin_unlock_irqrestore(&sched->queue_lock, flags);
}
```

---

## 🎯 需要更新的文档

### 1. ARCH_Design_02_AMD_GPREEMPT_redesign_v2.md

需要更新的章节：
- ✅ 核心数据结构：添加 `snapshot` 字段
- ✅ gpreempt_preempt_queue：修正 `destroy_mqd` 参数，添加 `checkpoint_mqd`
- ✅ gpreempt_check_resume：修正 `restore_mqd` 参数，添加 `load_mqd`
- ✅ 初始化：添加 snapshot buffer 分配

### 2. ARCH_Design_03_AMD_GPREEMPT_XSCHED.md

需要更新的章节：
- ✅ struct queue 定义：添加 `snapshot` 字段
- ✅ gpreempt_preempt_queue：修正参数
- ✅ gpreempt_check_resume_queues：修正参数
- ✅ kfd_gpreempt_register_queue：添加 snapshot 分配

---

## 📚 参考

### KFD CRIU 代码（正确示例）

1. **Checkpoint**: `kfd_process_queue_manager.c:800-865`
   ```c
   pqm_checkpoint_mqd() → dqm->ops.checkpoint_mqd()
   ```

2. **Restore**: `kfd_process_queue_manager.c:310-435`
   ```c
   pqm_create_queue() with restore_mqd + restore_ctl_stack
   ```

3. **MQD Manager**: `kfd_mqd_manager_v9.c:436-474`
   ```c
   checkpoint_mqd() - 简单的 memcpy
   restore_mqd() - memcpy + 设置 is_active=0
   ```

### 我们的实现参考

`CWSR_API_USAGE_REFERENCE.md` - KFD CWSR API 使用参考

---

## ✅ 总结

### 关键修正点

| 问题 | 影响 | 修正 |
|------|------|------|
| `destroy_mqd` 参数错误 | 🔴 无法正确触发 CWSR | 使用 `pipe` 和 `queue` 参数 |
| `restore_mqd` 参数错误 | 🔴 无法恢复队列 | 8 个参数，`&q->mqd` double pointer |
| 缺少 `checkpoint_mqd` | 🔴 没有数据可恢复 | preempt 前调用 |
| 缺少 `load_mqd` | 🔴 队列不会激活 | restore 后调用 |
| 缺少 `snapshot` 字段 | 🔴 无处存储备份 | 添加到 `struct queue` |

### 优先级

🔴 **P0 - 必须修正**（否则代码无法工作）:
1. `destroy_mqd` 参数
2. `restore_mqd` 参数
3. 添加 `checkpoint_mqd` 调用
4. 添加 `load_mqd` 调用
5. 添加 `snapshot` 数据结构

⚠️ **P1 - 强烈建议**:
1. 更新架构文档
2. 添加详细注释
3. 完善错误处理

---

**状态**: 等待文档更新  
**日期**: 2026-01-29  
**作者**: AI Assistant (基于 CWSR API 分析)
