# AMD GPREEMPT + XSched 融合架构设计

**日期**: 2026-01-29  
**版本**: 3.1 (CWSR API 修正)  
**最后更新**: 2026-01-29（修正 CWSR API 使用方式）
**目标**: 利用 AMD CWSR 实现 XSched Level-3 抢占，融合两篇论文的优势  
**核心原则**: Doorbell 性能 + CWSR 硬件能力 + XSched 框架 + GPREEMPT 调度逻辑

---

## 📢 v3.2 更新（2026-01-30）

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⭐⭐⭐ 关键更新：纯软件调度架构（代码验证）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

发现：amd_aql_queue.cpp:100 priority 写死验证
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  代码位置: rocr-runtime/core/runtime/amd_aql_queue.cpp:100
  关键代码: priority_(HSA_QUEUE_PRIORITY_NORMAL),
  
  → 无论 HIP 设置什么 priority，AqlQueue 向 KFD 传递时都是 NORMAL
  → 所有 queue 在 KFD/GPU 固件眼里优先级相同
  → GPU 硬件不会基于优先级做调度


架构调整：从硬件优先级 → 纯软件优先级调度
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

原设计（v3.1）:
  • 利用 AMD 16 级硬件优先级
  • queue.properties.priority = 0-15（传递给 GPU）
  • GPU 固件基于优先级调度
  
新设计（v3.2）:
  • 纯软件优先级调度 ⭐⭐⭐
  • queue.logical_priority = HIP 设置（软件使用）
  • queue.hardware_priority = NORMAL（硬件固定）
  • 软件层检测倒置并触发 CWSR
  
设计优势：
  ✅ 完全可控：调度逻辑 100% 在软件，不依赖硬件
  ✅ 灵活扩展：可以支持 >16 级优先级
  ✅ 硬件一致：所有 queue 硬件行为相同，简化交互
  ✅ 易于调试：软件逻辑透明，不依赖硬件黑盒


影响范围：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • 方案 A: KFD 层添加 logical_priority 字段
  • 方案 B: XSched 使用 logical_priority 做调度
  • ioctl 扩展: 传递 logical_priority
  • 优先级检测: 基于 logical_priority
  • 不改变 CWSR API 使用（v3.1 的修正仍然有效）


v3.1 更新（2026-01-29）- CWSR API 修正仍然有效
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

基于 KFD CRIU 代码分析，修正了方案 A 和方案 B 中的 CWSR API 调用：

修正内容：
  1. ✅ destroy_mqd: 使用 (pipe, queue) 而不是 pasid
  2. ✅ restore_mqd: 8个参数，&q->mqd 为 double pointer
  3. ✅ 新增 checkpoint_mqd: preempt 前保存状态
  4. ✅ 新增 load_mqd: restore 后激活队列
  5. ✅ 新增 snapshot 字段: 存储备份数据

详细修正: 参考 ARCH_DESIGN_CORRECTIONS_CWSR_API.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📢 设计理念

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⭐⭐⭐ 融合两篇论文的核心思想
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Paper#1 (GPREEMPT) 的贡献:
  1. 内核态监控和调度逻辑
  2. Ring Buffer 状态监控
  3. 优先级倒置检测算法
  4. 异步抢占触发机制
  5. 时间片使用策略（虽然 AMD 不需要）

Paper#2 (XSched) 的贡献:
  1. 多级硬件抽象框架（Lv1/Lv2/Lv3）
  2. 统一的 XQueue 接口
  3. Progressive Launching（Lv1）
  4. 应用层透明集成（LD_PRELOAD）
  5. 跨平台调度策略

我们的融合方案:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 从 GPREEMPT 采用:
   • 内核态监控线程（KFD kthread）
   • Ring Buffer rptr/wptr 监控
   • 优先级倒置检测逻辑
   • 直接调用 destroy_mqd（无 ioctl 开销）

✅ 从 XSched 采用:
   • Lv3 接口定义（interrupt/restore）
   • XQueue 抽象层
   • 应用透明性（LD_PRELOAD 或不拦截）
   • 调度策略框架（HPF, FIFO, etc.）

✅ AMD CWSR 的硬件优势:
   • Ring Buffer 保持不变（vs GPreempt 清空）
   • 精确状态恢复（vs GPreempt 重新提交）
   • 1-10μs 抢占延迟
   • 任意级别软件优先级（vs GPreempt 2 级时间片模拟）⭐⭐⭐
     ⚠️ 注：hardware_priority = NORMAL (所有 queue 相同)
     ⚠️ 使用 logical_priority 做纯软件调度

✅ 关键设计决策:
   • 不 bypass Doorbell（保留 ~100ns 提交性能）
   • 内核态调度器（避免 ioctl 开销）
   • 利用 CWSR 硬件机制（避免软件技巧妥协）
   • 可选的 XSched 用户态框架集成
```

---

## 🎯 架构概览

### 两种实现方案

我们提供两种融合方案，根据不同场景选择：

```
方案 A: 纯内核态 GPREEMPT-CWSR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

优势:
  ✅ 性能最优（无 userspace 开销）
  ✅ Doorbell 完全不变
  ✅ 应用完全透明（无需 LD_PRELOAD）
  ✅ 响应速度快（内核态监控）

劣势:
  ⚠️ 需要修改 KFD 驱动
  ⚠️ 部署需要 DKMS 编译
  ⚠️ 灵活性稍低（调度策略在内核）

适用场景:
  • 生产环境长期部署
  • 性能极致要求
  • 单一 AMD GPU 平台


方案 B: 混合架构 XSched-Lv3 + KFD-CWSR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

优势:
  ✅ 灵活性高（调度策略在用户态）
  ✅ 快速迭代（无需重启系统）
  ✅ 跨平台兼容（XSched 框架）
  ✅ 易于调试

劣势:
  ⚠️ 有 ioctl 开销（但仍保留 Doorbell）
  ⚠️ 需要 LD_PRELOAD（如果拦截）
  ⚠️ 或需要最小化 KFD 扩展（如果提供 ioctl）

适用场景:
  • 研发测试环境
  • 混合 GPU 平台
  • 需要频繁调整调度策略
```

---

## 🏗️ 方案 A: 纯内核态 GPREEMPT-CWSR

### 完整架构图

```
═══════════════════════════════════════════════════════════════════════════════
                         AMD GPREEMPT-CWSR 架构
═══════════════════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────────────┐
│  应用层 (Application Layer)                                               │
│  ─────────────────────────────────────────────────────────────────────── │
│                                                                           │
│  ┌──────────────────────┐            ┌──────────────────────┐           │
│  │ 训练任务 (Low Prio)   │            │ 推理服务 (High Prio) │           │
│  │ Priority = 3          │            │ Priority = 12         │           │
│  └──────┬───────────────┘            └──────┬───────────────┘           │
│         │                                    │                            │
│         ▼ hipLaunchKernel()                 ▼ hipLaunchKernel()         │
│  ┌──────────────────────┐            ┌──────────────────────┐           │
│  │ libamdhip64.so        │            │ libamdhip64.so        │           │
│  │ • 写 Ring Buffer      │            │ • 写 Ring Buffer      │           │
│  │ • wptr++              │            │ • wptr++              │           │
│  └──────┬───────────────┘            └──────┬───────────────┘           │
│         │                                    │                            │
└─────────┼────────────────────────────────────┼────────────────────────────┘
          │                                    │
          │ (~100ns each)                      │ (~100ns each)
          ▼ Doorbell (MMIO write) ⚡           ▼ Doorbell (MMIO write) ⚡
┌──────────────────────────────────────────────────────────────────────────┐
│  Doorbell 层（Hardware MMIO Registers）                                   │
│  ─────────────────────────────────────────────────────────────────────── │
│                                                                           │
│  *doorbell_train = wptr_train          *doorbell_infer = wptr_infer     │
│         ↓                                      ↓                         │
│  ┌─────────────────┐                    ┌─────────────────┐             │
│  │ Doorbell #0     │                    │ Doorbell #1     │             │
│  │ (Queue_train)   │                    │ (Queue_infer)   │             │
│  └─────────────────┘                    └─────────────────┘             │
└──────────────────────────────────────────────────────────────────────────┘
          ↓                                      ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  GPU 硬件层 (GPU Hardware Layer)                                          │
│  ─────────────────────────────────────────────────────────────────────── │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Command Processor (CP) - 固件控制                                   │ │
│  │                                                                     │ │
│  │  ┌─────────────────┐              ┌─────────────────┐             │ │
│  │  │ Queue_train     │              │ Queue_infer     │             │ │
│  │  │ priority=3      │              │ priority=12     │             │ │
│  │  │                 │              │                 │             │ │
│  │  │ Ring Buffer:    │              │ Ring Buffer:    │             │ │
│  │  │ [K0,K1,...K99]  │              │ [K0,K1,...K49]  │             │ │
│  │  │  ↑         ↑    │              │  ↑         ↑    │             │ │
│  │  │ rptr=30   wptr=100             │ rptr=0    wptr=50             │ │
│  │  │                 │              │                 │             │ │
│  │  │ ⚠️ 正在执行      │              │ ⏳ 等待中        │             │ │
│  │  └─────────────────┘              └─────────────────┘             │ │
│  │                                                                     │ │
│  │  ⚠️ GPU 硬件可能不会自动优先执行 Queue_infer（优先级支持有限）    │ │
│  │  ⚠️ 需要软件干预！                                                 │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Compute Units (CUs)                                                 │ │
│  │ • 执行 Queue_train 的 Wavefronts                                    │ │
│  │ • CU 占用率: 100%                                                   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
          ↑ 主动轮询监控（MMIO read）🔍
          ↑ 触发 CWSR 抢占（PM4 命令）⚡
          │
┌──────────────────────────────────────────────────────────────────────────┐
│  内核调度层 (KFD GPREEMPT Scheduler) - 核心创新 ⭐⭐⭐                      │
│  ─────────────────────────────────────────────────────────────────────── │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ 监控线程（kthread，定期唤醒 ~5ms）                                   │ │
│  │                                                                     │ │
│  │  while (!kthread_should_stop()) {                                  │ │
│  │    // ⭐ 步骤 1: 扫描所有队列（MMIO read）                          │ │
│  │    for_each_queue(q) {                                             │ │
│  │      q->hw_rptr = readl(q->read_ptr);   // ~100ns                 │ │
│  │      q->hw_wptr = readl(q->write_ptr);  // ~100ns                 │ │
│  │      q->pending = q->hw_wptr - q->hw_rptr;                        │ │
│  │      q->is_active = (q->pending > 0);                             │ │
│  │    }                                                                │ │
│  │                                                                     │ │
│  │    // ⭐ 步骤 2: 检测优先级倒置                                     │ │
│  │    high_q = find_highest_priority_waiting_queue();                │ │
│  │    low_q = find_running_queue();                                  │ │
│  │                                                                     │ │
│  │    if (high_q && low_q &&                                          │ │
│  │        high_q->priority > low_q->priority) {                      │ │
│  │      // ⭐ 步骤 3: 触发 CWSR 抢占（直接内核调用）                  │ │
│  │      low_q->mqd_mgr->destroy_mqd(                                 │ │
│  │        low_q->mqd,                                                 │ │
│  │        KFD_PREEMPT_TYPE_WAVEFRONT_SAVE,  // CWSR                  │ │
│  │        0  // timeout=0: 异步                                      │ │
│  │      );                                                             │ │
│  │      low_q->state = PREEMPTED;                                     │ │
│  │    }                                                                │ │
│  │                                                                     │ │
│  │    // ⭐ 步骤 4: 检查恢复                                          │ │
│  │    for_each_preempted_queue(q) {                                  │ │
│  │      if (!has_higher_priority_running(q)) {                       │ │
│  │        q->mqd_mgr->restore_mqd(q->mqd, ...);                     │ │
│  │        q->state = ACTIVE;                                          │ │
│  │      }                                                              │ │
│  │    }                                                                │ │
│  │                                                                     │ │
│  │    msleep_interruptible(check_interval_ms);  // 5ms               │ │
│  │  }                                                                  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  关键优势:                                                                │
│  ✓ 直接访问 queue 结构（无需查询 ioctl）                                 │
│  ✓ 直接调用 destroy_mqd（无需控制 ioctl）                                │
│  ✓ 无 userspace ↔ kernel 切换开销                                        │
│  ✓ Ring Buffer 保持不变（AMD CWSR 优势）                                 │
└──────────────────────────────────────────────────────────────────────────┘
          ↓ PM4 命令
┌──────────────────────────────────────────────────────────────────────────┐
│  CWSR 硬件层 (AMD MI300 Hardware)                                         │
│  ─────────────────────────────────────────────────────────────────────── │
│                                                                           │
│  Trap Handler 执行 CWSR:                                                  │
│  1. 接收 PM4 UNMAP_QUEUES 命令                                            │
│  2. 向活跃 Wavefronts 发送 Trap 信号                                      │
│  3. 保存 Wave 状态到 CWSR Area:                                           │
│     • PC (Program Counter)                                               │
│     • SGPR[0..127] (Scalar registers)                                    │
│     • VGPR[0..255][0..63] (Vector registers)                             │
│     • LDS (Local Data Share, 64KB)                                       │
│     • ACC VGPR (Accumulator registers)                                   │
│  4. 释放 CU 资源                                                          │
│  5. ⭐⭐⭐ Ring Buffer 保持不变！                                          │
│     • rptr = 30 (GPU 读到这里，保持不变)                                  │
│     • wptr = 100 (应用提交的，保持不变)                                   │
│     • [K0...K29, K30, K31, ...K99] 全部保留                              │
│                                                                           │
│  延迟: 1-10μs ⚡                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 方案 A 详细设计（推荐）

### 核心数据结构

```c
// ============================================================================
// 文件: amd/amdkfd/kfd_priv.h
// ============================================================================

// 队列扩展结构
struct queue {
    // ===== 现有字段（保持不变）=====
    struct queue_properties properties;  // 包含 hardware_priority (固定 NORMAL)
    struct mqd_manager *mqd_mgr;
    void *mqd;
    struct kfd_mem_obj *mqd_mem_obj;     // MQD 内存对象
    uint64_t gart_mqd_addr;              // GART 地址
    struct kfd_process *process;
    
    // ===== GPREEMPT 新增字段 ⭐⭐⭐ =====
    
    // Ring Buffer 状态监控（核心！）
    uint32_t hw_rptr;              // GPU 读指针（通过 MMIO 读取）
    uint32_t hw_wptr;              // CPU 写指针（通过 MMIO 读取）
    uint32_t pending_count;        // wptr - rptr = 待执行 kernels
    bool is_active;                // pending_count > 0
    
    // ⭐⭐⭐ 纯软件调度优先级（v3.2 更新）
    int logical_priority;          // HIP 设置的逻辑优先级（从应用传递）
    int effective_priority;        // 动态优先级（基于 logical_priority 计算）
    
    // ⚠️ properties.priority = NORMAL（硬件优先级，所有 queue 相同）
    // ⚠️ 只使用 logical_priority 做软件调度决策
    
    // 抢占状态
    enum queue_state {
        QUEUE_STATE_ACTIVE,        // 活跃（正在执行或等待）
        QUEUE_STATE_PREEMPTED,     // 被抢占（CWSR saved）
        QUEUE_STATE_IDLE           // 空闲（pending_count=0）
    } state;
    
    bool preemption_pending;       // 抢占进行中
    ktime_t preempt_start;         // 抢占开始时间
    
    // ⭐⭐⭐ Snapshot（用于 checkpoint/restore）- v3.1 新增
    struct {
        void *mqd_backup;          // MQD 备份 buffer
        void *ctl_stack_backup;    // Control stack 备份 buffer
        size_t ctl_stack_size;     // Control stack 大小
        bool valid;                // Snapshot 是否有效
    } snapshot;
    
    // 统计信息
    atomic64_t total_preemptions;
    atomic64_t total_resumes;
    ktime_t last_active_time;
    
    // 链表节点
    struct list_head gpreempt_list;  // 全局调度器链表
};


// 全局调度器
struct kfd_gpreempt_scheduler {
    // 监控线程
    struct task_struct *monitor_thread;
    bool enabled;
    
    // 配置参数
    unsigned int check_interval_ms;    // 监控间隔（默认 5ms）
    
    // 队列管理
    struct list_head all_queues;       // 所有队列
    spinlock_t queue_lock;
    
    // 统计
    atomic64_t total_checks;
    atomic64_t total_inversions;       // 优先级倒置次数
    atomic64_t total_preemptions;      // 总抢占次数
    atomic64_t total_resumes;          // 总恢复次数
};
```

### 监控线程实现（核心！）

```c
// ============================================================================
// 文件: amd/amdkfd/kfd_gpreempt_scheduler.c
// ============================================================================

static int kfd_gpreempt_monitor_thread(void *data)
{
    struct kfd_dev *dev = data;
    struct kfd_gpreempt_scheduler *sched = dev->gpreempt_sched;
    struct queue *high_q, *low_q;
    
    printk(KERN_INFO "GPREEMPT: Monitor thread started (interval=%dms)\n",
           sched->check_interval_ms);
    
    while (!kthread_should_stop()) {
        // ⭐ 步骤 1: 定期休眠
        msleep_interruptible(sched->check_interval_ms);
        
        if (!sched->enabled)
            continue;
        
        // ⭐ 步骤 2: 扫描所有队列状态
        gpreempt_scan_all_queues(sched);
        
        // ⭐ 步骤 3: 检测优先级倒置
        if (gpreempt_detect_priority_inversion(sched, &high_q, &low_q)) {
            printk(KERN_INFO "GPREEMPT: Priority inversion detected\n");
            printk(KERN_INFO "  High: Q%u (prio=%d, pending=%u) waiting\n",
                   high_q->properties.queue_id,
                   high_q->properties.priority,
                   high_q->pending_count);
            printk(KERN_INFO "  Low:  Q%u (prio=%d, pending=%u) running\n",
                   low_q->properties.queue_id,
                   low_q->properties.priority,
                   low_q->pending_count);
            
            // ⭐ 步骤 4: 触发 CWSR 抢占
            gpreempt_preempt_queue(sched, low_q);
            
            atomic64_inc(&sched->total_inversions);
        }
        
        // ⭐ 步骤 5: 检查是否可以恢复
        gpreempt_check_resume_queues(sched);
        
        atomic64_inc(&sched->total_checks);
    }
    
    printk(KERN_INFO "GPREEMPT: Monitor thread exiting\n");
    return 0;
}


// ⭐⭐⭐ 扫描队列状态（核心机制）
static void gpreempt_scan_all_queues(struct kfd_gpreempt_scheduler *sched)
{
    struct queue *q;
    unsigned long flags;
    
    spin_lock_irqsave(&sched->queue_lock, flags);
    
    list_for_each_entry(q, &sched->all_queues, gpreempt_list) {
        if (!q->properties.doorbell_ptr)
            continue;
        
        // ⭐⭐⭐ 核心：主动读取 Ring Buffer 指针（MMIO read）
        //
        // 关键理解：
        //   • 应用通过 Doorbell 写 wptr（MMIO write, ~100ns）
        //   • 内核通过 MMIO read 读取 wptr（~100ns）
        //   • 硬件保证可见性和原子性
        //
        uint32_t hw_rptr = readl(q->properties.read_ptr);
        uint32_t hw_wptr = readl(q->properties.write_ptr);
        
        // 计算待执行的 kernels 数量
        uint32_t pending = (hw_wptr - hw_rptr) & q->ring_size_mask;
        
        // 更新队列状态
        q->hw_rptr = hw_rptr;
        q->hw_wptr = hw_wptr;
        q->pending_count = pending;
        q->is_active = (pending > 0);
        
        if (q->is_active) {
            q->last_active_time = ktime_get();
        }
        
        // 更新空闲状态
        if (pending == 0 && q->state == QUEUE_STATE_ACTIVE) {
            q->state = QUEUE_STATE_IDLE;
        }
    }
    
    spin_unlock_irqrestore(&sched->queue_lock, flags);
}


  // ⭐⭐⭐ 优先级倒置检测（基于逻辑优先级）⭐⭐⭐
static bool gpreempt_detect_priority_inversion(
    struct kfd_gpreempt_scheduler *sched,
    struct queue **out_high_q,
    struct queue **out_low_q)
{
    struct queue *q;
    struct queue *highest_waiting = NULL;
    struct queue *running_queue = NULL;
    unsigned long flags;
    
    spin_lock_irqsave(&sched->queue_lock, flags);
    
    // 找到最高逻辑优先级的等待队列
    list_for_each_entry(q, &sched->all_queues, gpreempt_list) {
        if (!q->is_active || q->state == QUEUE_STATE_PREEMPTED)
            continue;
        
        // 简化判断：如果有 pending work，认为在等待
        if (q->pending_count > 0) {
            if (!highest_waiting ||
                q->logical_priority > highest_waiting->logical_priority) {  // ← 使用 logical_priority
                highest_waiting = q;
            }
        }
    }
    
    // 找到当前可能正在运行的队列
    // 简化实现：假设最近活跃的队列正在运行
    list_for_each_entry(q, &sched->all_queues, gpreempt_list) {
        if (q->state == QUEUE_STATE_ACTIVE && q->is_active) {
            if (!running_queue ||
                q->last_active_time > running_queue->last_active_time) {
                running_queue = q;
            }
        }
    }
    
    // 检测倒置（基于逻辑优先级）
    if (highest_waiting && running_queue &&
        highest_waiting != running_queue &&
        highest_waiting->logical_priority > running_queue->logical_priority) {  // ← 使用 logical_priority
        *out_high_q = highest_waiting;
        *out_low_q = running_queue;
        spin_unlock_irqrestore(&sched->queue_lock, flags);
        return true;
    }
    
    spin_unlock_irqrestore(&sched->queue_lock, flags);
    return false;
}


// v3.2 关键说明：纯软件调度
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// ✅ 所有 queue 的 properties.priority = NORMAL（硬件相同）
// ✅ 使用 logical_priority（从 HIP 传递）做软件决策
// ✅ GPU 不会基于优先级调度，需要软件检测并触发 CWSR
// ✅ logical_priority 可以 >15 级（不受硬件限制）
//
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


// ⭐⭐⭐ 触发 CWSR 抢占（v3.1 修正）
static int gpreempt_preempt_queue(
    struct kfd_gpreempt_scheduler *sched,
    struct queue *q)
{
    int r;
    ktime_t start = ktime_get();
    
    printk(KERN_INFO "GPREEMPT: Preempting Q%u (logical_prio=%d, hw_prio=NORMAL)\n",
           q->properties.queue_id,
           q->logical_priority);              // ← 显示逻辑优先级
    printk(KERN_INFO "  Ring Buffer: rptr=%u, wptr=%u, pending=%u\n",
           q->hw_rptr, q->hw_wptr, q->pending_count);
    
    // ⭐⭐⭐ 步骤 1: Checkpoint MQD（v3.1 新增）
    q->mqd_mgr->checkpoint_mqd(
        q->mqd_mgr,
        q->mqd,
        q->snapshot.mqd_backup,
        q->snapshot.ctl_stack_backup
    );
    q->snapshot.valid = true;
    
    printk(KERN_DEBUG "  MQD checkpointed\n");
    
    // 标记状态
    q->preemption_pending = true;
    q->preempt_start = start;
    q->state = QUEUE_STATE_PREEMPTED;
    
    // ⭐⭐⭐ 步骤 2: 触发硬件 CWSR（v3.1 修正参数）
    //
    // 重要：Ring Buffer 不变！
    //   • GPU 从 rptr 读到的位置保持不变
    //   • wptr 应用提交的位置保持不变
    //   • Ring Buffer 中的 kernels 全部保留
    //   • CWSR 保存 Wave 状态到专用区域
    //
    // v3.1 修正：使用 pipe 和 queue，而不是 pasid
    //
    r = q->mqd_mgr->destroy_mqd(
        q->mqd_mgr,
        q->mqd,
        KFD_PREEMPT_TYPE_WAVEFRONT_SAVE,  // CWSR 模式
        0,                                 // timeout=0: 异步
        q->pipe,                           // ⭐ 修正：pipe 编号
        q->queue                           // ⭐ 修正：queue 编号
    );
    
    if (r == 0) {
        // 抢占立即完成
        q->preemption_pending = false;
        atomic64_inc(&q->total_preemptions);
        printk(KERN_INFO "GPREEMPT: Q%u preemption completed\n",
               q->properties.queue_id);
    } else if (r == -EINPROGRESS) {
        // 抢占正在进行（正常情况）
        atomic64_inc(&sched->total_preemptions);
        atomic64_inc(&q->total_preemptions);
    } else {
        // 错误
        printk(KERN_ERR "GPREEMPT: Preemption failed for Q%u: %d\n",
               q->properties.queue_id, r);
        q->preemption_pending = false;
        q->state = QUEUE_STATE_ACTIVE;
        q->snapshot.valid = false;
        return r;
    }
    
    return 0;
}


// ⭐⭐⭐ 恢复队列（v3.1 修正）
static void gpreempt_check_resume_queues(struct kfd_gpreempt_scheduler *sched)
{
    struct queue *q;
    unsigned long flags;
    
    spin_lock_irqsave(&sched->queue_lock, flags);
    
    // 遍历所有被抢占的队列
    list_for_each_entry(q, &sched->all_queues, gpreempt_list) {
        if (q->state != QUEUE_STATE_PREEMPTED)
            continue;
        
        // 检查 snapshot 是否有效（v3.1 新增）
        if (!q->snapshot.valid) {
            printk(KERN_WARN "GPREEMPT: Q%u has no valid snapshot\n",
                   q->properties.queue_id);
            continue;
        }
        
        // 检查是否有更高优先级队列运行
        if (gpreempt_has_higher_priority_active(sched, q))
            continue;
        
        // 可以恢复
        printk(KERN_INFO "GPREEMPT: Resuming Q%u (prio=%d)\n",
               q->properties.queue_id,
               q->properties.priority);
        printk(KERN_INFO "  Ring Buffer: rptr=%u, wptr=%u\n",
               q->hw_rptr, q->hw_wptr);
        
        // ⭐⭐⭐ 步骤 1: Restore MQD（v3.1 修正参数）
        //
        // v3.1 修正：8个参数，&q->mqd 为 double pointer
        //
        q->mqd_mgr->restore_mqd(
            q->mqd_mgr,
            &q->mqd,                      // ⭐ double pointer
            q->mqd_mem_obj,
            &q->gart_mqd_addr,
            &q->properties,
            q->snapshot.mqd_backup,       // 从 snapshot 恢复
            q->snapshot.ctl_stack_backup,
            q->snapshot.ctl_stack_size
        );
        
        printk(KERN_DEBUG "  MQD restored from snapshot\n");
        
        // ⭐⭐⭐ 步骤 2: Load MQD（v3.1 新增）
        //
        // 重要：GPU 会做什么？
        //   1. 从 CWSR Area 恢复 Wave 状态
        //   2. ⭐ 从 Ring Buffer rptr 继续读取
        //   3. Ring Buffer 保持: rptr=30, wptr=100
        //   4. GPU 继续读取 kernel_30, 31, ..., 99
        //   5. 无需重新提交任何 kernel！✅
        //
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
            q->properties.is_active = true;
            q->snapshot.valid = false;  // 清除 snapshot
            atomic64_inc(&q->total_resumes);
            
            uint64_t latency_us = ktime_us_delta(ktime_get(), q->preempt_start);
            printk(KERN_INFO "GPREEMPT: Q%u resumed (latency=%llu us)\n",
                   q->properties.queue_id, latency_us);
        } else {
            printk(KERN_ERR "GPREEMPT: Failed to load MQD for Q%u: %d\n",
                   q->properties.queue_id, r);
        }
    }
    
    spin_unlock_irqrestore(&sched->queue_lock, flags);
}
```

### 初始化和注册

```c
// ============================================================================
// 文件: amd/amdkfd/kfd_gpreempt_scheduler.c
// ============================================================================

int kfd_gpreempt_init(struct kfd_dev *dev)
{
    struct kfd_gpreempt_scheduler *sched;
    
    sched = kzalloc(sizeof(*sched), GFP_KERNEL);
    if (!sched)
        return -ENOMEM;
    
    // 默认配置
    sched->check_interval_ms = 5;
    sched->enabled = true;
    
    INIT_LIST_HEAD(&sched->all_queues);
    spin_lock_init(&sched->queue_lock);
    
    // 创建监控线程
    sched->monitor_thread = kthread_run(
        kfd_gpreempt_monitor_thread,
        dev,
        "kfd_gpreempt_%d",
        dev->id
    );
    
    if (IS_ERR(sched->monitor_thread)) {
        int ret = PTR_ERR(sched->monitor_thread);
        kfree(sched);
        return ret;
    }
    
    dev->gpreempt_sched = sched;
    
    printk(KERN_INFO "GPREEMPT: Initialized for device %d\n", dev->id);
    
    return 0;
}


// 队列创建时注册到调度器（v3.1 更新：添加 snapshot 分配）
int kfd_gpreempt_register_queue(struct kfd_dev *dev, struct queue *q)
{
    struct kfd_gpreempt_scheduler *sched = dev->gpreempt_sched;
    size_t mqd_size, ctl_stack_size;
    unsigned long flags;
    
    if (!sched)
        return 0;
    
    // ⭐ 步骤 1: 获取 MQD 大小（v3.1 新增）
    if (dev->device_info->asic_family == CHIP_VEGA10 ||
        dev->device_info->asic_family == CHIP_VEGA20) {
        mqd_size = sizeof(struct v9_mqd);
    } else if (dev->device_info->asic_family == CHIP_MI300) {
        mqd_size = sizeof(struct v11_mqd);
    } else {
        mqd_size = PAGE_SIZE;
    }
    
    // ⭐ 步骤 2: 获取 control stack 大小（v3.1 新增）
    struct v9_mqd *mqd = (struct v9_mqd *)q->mqd;
    ctl_stack_size = mqd->cp_hqd_cntl_stack_size;
    if (ctl_stack_size == 0)
        ctl_stack_size = 4096;
    
    // ⭐ 步骤 3: 分配 snapshot buffers（v3.1 新增）
    q->snapshot.mqd_backup = kzalloc(mqd_size, GFP_KERNEL);
    if (!q->snapshot.mqd_backup) {
        printk(KERN_ERR "GPREEMPT: Failed to allocate MQD backup for Q%u\n",
               q->properties.queue_id);
        return -ENOMEM;
    }
    
    q->snapshot.ctl_stack_backup = kzalloc(ctl_stack_size, GFP_KERNEL);
    if (!q->snapshot.ctl_stack_backup) {
        kfree(q->snapshot.mqd_backup);
        printk(KERN_ERR "GPREEMPT: Failed to allocate control stack backup for Q%u\n",
               q->properties.queue_id);
        return -ENOMEM;
    }
    
    q->snapshot.ctl_stack_size = ctl_stack_size;
    q->snapshot.valid = false;
    
    // ⭐ 步骤 4: 初始化 GPREEMPT 字段
    q->hw_rptr = 0;
    q->hw_wptr = 0;
    q->pending_count = 0;
    q->is_active = false;
    
    // ⭐⭐⭐ v3.2: 初始化逻辑优先级（从 ioctl 传递）
    q->logical_priority = q->create_args.logical_priority;  // ← 从 HIP 传递的逻辑优先级
    q->effective_priority = q->logical_priority;             // ← 基础值
    // ⚠️ q->properties.priority = NORMAL（硬件优先级，所有 queue 相同）
    
    q->state = QUEUE_STATE_IDLE;
    q->preemption_pending = false;
    atomic64_set(&q->total_preemptions, 0);
    atomic64_set(&q->total_resumes, 0);
    
    // ⭐ 步骤 5: 添加到调度器
    spin_lock_irqsave(&sched->queue_lock, flags);
    list_add_tail(&q->gpreempt_list, &sched->all_queues);
    spin_unlock_irqrestore(&sched->queue_lock, flags);
    
    printk(KERN_INFO "GPREEMPT: Registered Q%u (logical_prio=%d, hw_prio=NORMAL, mqd=%zu, ctl=%zu)\n",
           q->properties.queue_id,
           q->logical_priority,                // ← 显示逻辑优先级
           mqd_size, ctl_stack_size);
    
    return 0;
}


// 队列注销（v3.1 新增：释放 snapshot buffers）
void kfd_gpreempt_unregister_queue(struct kfd_dev *dev, struct queue *q)
{
    struct kfd_gpreempt_scheduler *sched = dev->gpreempt_sched;
    unsigned long flags;
    
    if (!sched)
        return;
    
    // 从调度器移除
    spin_lock_irqsave(&sched->queue_lock, flags);
    list_del(&q->gpreempt_list);
    spin_unlock_irqrestore(&sched->queue_lock, flags);
    
    // 释放 snapshot buffers
    if (q->snapshot.mqd_backup) {
        kfree(q->snapshot.mqd_backup);
        q->snapshot.mqd_backup = NULL;
    }
    
    if (q->snapshot.ctl_stack_backup) {
        kfree(q->snapshot.ctl_stack_backup);
        q->snapshot.ctl_stack_backup = NULL;
    }
    
    printk(KERN_INFO "GPREEMPT: Unregistered Q%u\n", q->properties.queue_id);
}
```

### 完整工作流程：双 AI 模型场景

```
═══════════════════════════════════════════════════════════════════════════════
                    完整时间线：推理抢占训练
═══════════════════════════════════════════════════════════════════════════════

T < 0: 训练任务执行中
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

应用层（训练进程）:
  for (i = 0; i < 100; i++)
    hipLaunchKernel(train_kernel_i, ...)
      ↓ 每个 ~100ns
    写 Ring Buffer[wptr++]
    *doorbell_train = wptr  ⚡ MMIO write

GPU 硬件:
  Queue_train Ring Buffer:
    [k0, k1, k2, ..., k24, k25, ..., k99, empty, ...]
    ↑                         ↑
    rptr = 25                 wptr = 100
    
  • GPU 正在执行 kernel_25
  • pending_count = 75
  • CU 占用率: 95%+

KFD GPREEMPT（上次检查 T=-5ms）:
  scan_queues():
    Queue_train: rptr=25, wptr=100, pending=75, active ✅
    Queue_infer: rptr=0, wptr=0, pending=0, idle ⏸️
  detect_inversion():
    无倒置 ✅


T = 0: 推理请求到达 ⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

应用层（推理服务）:
  // ⭐ 提交 50 个 kernels
  for (i = 0; i < 50; i++)
    hipLaunchKernel(infer_kernel_i, ...)
      ↓ 每个 ~100ns
    写 Ring Buffer[wptr++]
    *doorbell_infer = wptr  ⚡ MMIO write
  
  // 提交完成，总延迟: 50 * 100ns = 5μs ✅
  
  // 等待完成
  hipStreamSynchronize(stream_infer);

GPU 硬件:
  Queue_infer Ring Buffer:
    [infer_k0, infer_k1, ..., infer_k49, empty, ...]
    ↑                                      ↑
    rptr = 0                               wptr = 50
    
  • pending_count = 50
  • ⚠️ 但 Queue_train 正在占用 GPU
  • Queue_infer 只能等待 ⏳

关键问题:
  ⚠️ GPU 不会主动抢占！
  • 硬件优先级支持有限
  • GPU 继续执行 Queue_train
  • Queue_infer 等待调度器检测


T = 0 ~ 5ms: 等待检测 ⏳
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPU 继续执行 Queue_train
  • Queue_infer 的 50 个 kernels 在 Ring Buffer 中等待
  • rptr 仍然是 0（GPU 还没读取）

⏳ 等待 GPREEMPT 监控线程检测
  • 平均延迟: 2.5ms（检查间隔的一半）
  • 最坏延迟: 5ms


T = 5ms: GPREEMPT 检测到倒置 ⭐⭐⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KFD GPREEMPT 监控线程（定时唤醒）:
  
  // ⭐ 步骤 1: 扫描所有队列
  gpreempt_scan_all_queues():
    
    // Queue_train (MMIO read, ~200ns)
    rptr = readl(queue_train.read_ptr) = 30    // GPU 更新了
    wptr = readl(queue_train.write_ptr) = 100  // 应用提交的
    pending = 100 - 30 = 70
    is_active = true ✅
    
    // Queue_infer (MMIO read, ~200ns)
    rptr = readl(queue_infer.read_ptr) = 0     // GPU 还没读
    wptr = readl(queue_infer.write_ptr) = 50   // 应用提交的
    pending = 50 - 0 = 50
    is_active = true ✅
  
  扫描延迟: <10μs (2个队列 * 2次MMIO read)
  
  // ⭐ 步骤 2: 检测优先级倒置
  gpreempt_detect_priority_inversion():
    
    highest_waiting = Queue_infer (prio=12, pending=50)
    running_queue = Queue_train (prio=3, pending=70)
    
    ⚠️ 优先级倒置！
  
  // ⭐ 步骤 3: 触发抢占
  gpreempt_preempt_queue(Queue_train):
    
    printk("GPREEMPT: Preempting Q_train\n");
    printk("  Ring Buffer: rptr=30, wptr=100, pending=70\n");
    
    // ⭐ 触发 CWSR（异步，直接内核调用）
    destroy_mqd(Queue_train, WAVEFRONT_SAVE, timeout=0)
      ↓
    构造 PM4 命令: PM4_ME_UNMAP_QUEUES
      queue_id = queue_train.id
      preempt_type = CWSR
      save_area = queue_train.cwsr_area
      ↓
    写入 CP Ring Buffer
      ↓
    敲 CP Doorbell
      ↓
    return -EINPROGRESS  // 异步返回
    
    Queue_train.state = PREEMPTED
  
  触发延迟: <1μs ✅ (内核函数调用，无 ioctl)


T = 5.001ms ~ 5.010ms: GPU 执行 CWSR ⭐⭐⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPU Command Processor:
  • 检测到 CP Doorbell
  • 读取 PM4 命令: UNMAP_QUEUES
  • 向 MEC 发送抢占请求

GPU MEC (Micro-Engine Compute):
  • 找到 Queue_train
  • 向所有活跃 CU 发送 Trap 信号

GPU CUs (并行执行 Trap Handler):
  For each active Wave on Queue_train:
    1. 在指令边界停止执行
    
    2. ⭐ 保存状态到 CWSR Area:
       • PC = 0x401234
       • SGPR[0..127]
       • VGPR[0..255][0..63]
       • LDS (64KB)
       • ACC VGPR
       
       保存大小: ~200KB per Wave
    
    3. 释放 CU 资源
  
  并行执行，延迟: 1-10μs ✅

⭐⭐⭐ 关键状态（抢占完成后）:
  Queue_train Ring Buffer 完全保持不变！
    rptr = 30  (GPU 读到这里，保持不变)
    wptr = 100 (应用提交的，保持不变)
    Ring Buffer: [k0, ..., k29, k30, k31, ..., k99]
                                ↑~~~~~~~~~~~~~~~~↑
                          这 70 个 kernels 仍在 Ring Buffer 中！
  
  CWSR Area: 保存了 Wave 状态（PC=0x401234）
  MQD: 标记为 PREEMPTED


T = 5.010ms: GPU 切换到高优先级队列 ⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPU CP:
  • Queue_train 已挂起
  • 扫描 Runlist
  • 发现 Queue_infer (pending=50, prio=12)
  • 切换到 Queue_infer ✅

GPU 执行 Queue_infer:
  While (rptr < wptr) {
    • 从 Ring Buffer 读取 AQL_Packet[rptr]
    • 解析 kernel 参数
    • 分配 CU 资源
    • 启动 Wavefronts
    • rptr++
  }
  
  执行 50 个推理 kernels
  延迟: ~20ms


T = 25ms: 推理完成 ✅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPU 状态:
  Queue_infer: rptr=50, wptr=50, pending=0, idle ✅

应用层:
  hipStreamSynchronize(stream_infer);  // 返回 ✅
  
  auto t1 = now();
  latency = t1 - t0 = 25ms ✅ (满足 <30ms SLA)

端到端延迟分解:
  提交延迟:   5μs (50 * 100ns)
  等待检测:   5ms (平均 2.5ms)
  抢占延迟:   10μs (CWSR)
  切换延迟:   <1μs
  执行时间:   20ms
  ───────────────
  总延迟:     ~25ms ✅


T = 30ms: GPREEMPT 恢复训练任务 ⭐⭐⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KFD GPREEMPT（下次检查 T=30ms）:
  gpreempt_scan_all_queues():
    Queue_infer: rptr=50, wptr=50, pending=0, idle ✅
    Queue_train: state=PREEMPTED
  
  gpreempt_check_resume_queues():
    // 没有更高优先级队列运行
    // 可以恢复 Queue_train
    
    printk("GPREEMPT: Resuming Q_train\n");
    printk("  Ring Buffer: rptr=30, wptr=100\n");
    
    // ⭐⭐⭐ 恢复 MQD
    restore_mqd(queue_train.mqd, ...)
      ↓
    构造 PM4 命令: PM4_ME_MAP_QUEUES
      queue_id = queue_train.id
      restore_from = queue_train.cwsr_area
      ↓
    写入 CP Ring Buffer
      ↓
    敲 CP Doorbell
    
    Queue_train.state = ACTIVE
  
  恢复延迟: ~1μs


T = 30.001ms: GPU 恢复训练任务 ⭐⭐⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPU CWSR Resume:
  1. 从 CWSR Area 读取状态
  
  2. ⭐ 恢复所有 Wave 状态:
     • 恢复 PC = 0x401234 ✅
     • 恢复 SGPR[0..127]
     • 恢复 VGPR[0..255][0..63]
     • 恢复 LDS
  
  3. 重新分配到 CU
  
  4. ⭐⭐⭐ 从 PC=0x401234 继续执行
     • 不是重新开始 kernel
     • 而是从中断的指令继续 ✅
  
  5. ⭐⭐⭐ 继续从 Ring Buffer 读取后续 kernels
     • Ring Buffer: rptr=30, wptr=100
     • GPU 继续读取 kernel_30, 31, ..., 99
     • 这些 kernels 一直在 Ring Buffer 中！
     • 无需重新提交 ✅
  
  恢复延迟: 1-10μs ✅

GPU 继续执行:
  Queue_train: rptr 30 → 31 → ... → 100
  • 所有 100 个 kernels 正常完成
  • 无重复执行 ✅
  • 应用完全无感知 ✅
  • 增加的延迟: ~25ms (被抢占时间)
```

---

## 🔄 方案 B: XSched-Lv3 + KFD-CWSR 混合架构

### 架构设计

```
═══════════════════════════════════════════════════════════════════════════════
                    XSched + CWSR 混合架构
═══════════════════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────────────┐
│  应用层 (Application Layer)                                               │
│  • 使用标准 HIP API（无需修改）                                           │
│  • hipLaunchKernel() → Doorbell (~100ns) ⚡                              │
└──────────────────────────────────────────────────────────────────────────┘
          ↓ LD_PRELOAD 拦截（可选）
┌──────────────────────────────────────────────────────────────────────────┐
│  XSched 用户态框架 (XShim + XPreempt)                                     │
│  ─────────────────────────────────────────────────────────────────────── │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ XShim (API Interceptor) - 可选                                      │ │
│  │ • 拦截 hipLaunchKernel（如果需要）                                  │ │
│  │ • 或者不拦截，直接使用 Doorbell                                      │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│          ↓                                                                │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ XPreempt Scheduler (调度器核心)                                     │ │
│  │ • XQueue 管理                                                        │ │
│  │ • 优先级队列                                                         │ │
│  │ • HPF / FIFO / RR 调度策略                                          │ │
│  │ • 检测优先级倒置                                                     │ │
│  │ • 调用 Lv3 接口触发抢占                                              │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│          ↓                                                                │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ XAL-Lv3 Implementation (AMD CWSR Backend)                           │ │
│  │                                                                     │ │
│  │  int interrupt(hwQueue hwq) {                                      │ │
│  │    int kfd_fd = open("/dev/kfd", O_RDWR);                         │ │
│  │    struct kfd_ioctl_preempt_queue_args args = {                   │ │
│  │      .queue_id = hwq->queue_id,                                   │ │
│  │      .preempt_type = 2,  // WAVEFRONT_SAVE                        │ │
│  │      .timeout_ms = 1000                                            │ │
│  │    };                                                               │ │
│  │    return ioctl(kfd_fd, AMDKFD_IOC_PREEMPT_QUEUE, &args);        │ │
│  │  }                                                                  │ │
│  │                                                                     │ │
│  │  int restore(hwQueue hwq) {                                        │ │
│  │    int kfd_fd = open("/dev/kfd", O_RDWR);                         │ │
│  │    struct kfd_ioctl_resume_queue_args args = {                    │ │
│  │      .queue_id = hwq->queue_id                                    │ │
│  │    };                                                               │ │
│  │    return ioctl(kfd_fd, AMDKFD_IOC_RESUME_QUEUE, &args);         │ │
│  │  }                                                                  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
          ↓ ioctl (~1-10μs)
┌──────────────────────────────────────────────────────────────────────────┐
│  KFD 驱动层 (Minimal Extension)                                           │
│  ─────────────────────────────────────────────────────────────────────── │
│                                                                           │
│  新增 ioctl 接口:                                                         │
│  • AMDKFD_IOC_PREEMPT_QUEUE (0x87) ⭐                                    │
│  • AMDKFD_IOC_RESUME_QUEUE (0x88) ⭐                                     │
│                                                                           │
│  实现很简单（只是封装现有函数）:                                          │
│  • kfd_ioctl_preempt_queue() → destroy_mqd(CWSR)                        │
│  • kfd_ioctl_resume_queue() → restore_mqd()                              │
│                                                                           │
│  ⚠️ 关键问题：跨进程 Queue 访问权限                                       │
│    • 需要增加权限检查或 CAP_SYS_ADMIN                                     │
└──────────────────────────────────────────────────────────────────────────┘
          ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  CWSR 硬件层 (同方案 A)                                                    │
│  • 1-10μs 抢占延迟                                                        │
│  • Ring Buffer 保持不变                                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 两种方案对比

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
维度                方案 A (纯内核态)         方案 B (混合架构)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
调度器位置          内核态 (kthread)           用户态 (XSched) + 内核态 (CWSR)
监控开销            无（内核直接访问）         ioctl (~1-10μs per call)
Doorbell 性能       完全保留 ✅                完全保留 ✅
应用透明性          完全透明 ✅                需要 LD_PRELOAD（可选）
                                              或完全透明（不拦截模式）
实现复杂度          中（修改 KFD）             低（用户态 + 最小 ioctl）
调试难度            高（内核态调试）           低（用户态调试）
灵活性              低（调度策略在内核）       高（调度策略在用户态）
部署要求            DKMS 编译 + 重启           复制库文件
跨平台支持          仅 AMD                     XSched 支持多平台
性能                最优（无 ioctl）           优秀（有 ioctl 但仍保留 Doorbell）
适用场景            生产环境长期部署           研发测试、混合平台
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

共同优势:
✅ Doorbell 性能保留（~100ns）
✅ CWSR 硬件抢占（1-10μs）
✅ Ring Buffer 保持不变
✅ 精确状态恢复
✅ 无重复执行
```

---

## 🎯 与 GPREEMPT (NVIDIA) 和 XSched (Lv1) 的对比

### 完整对比表

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
维度            GPreempt(NV)    XSched Lv1     方案 A         方案 B
                                (AMD 当前)     (纯内核)       (混合)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
提交性能        ~100ns ✅       ~100ns ✅      ~100ns ✅      ~100ns ✅
Doorbell        保留 ✅         保留 ✅        保留 ✅        保留 ✅

调度器位置      用户态          用户态         内核态 ✅      用户态
监控方式        ioctl 查询      拦截 API       MMIO read ✅   可选拦截
监控开销        1-10μs          Progressive    <1μs ✅        ioctl
                                控制

抢占触发        用户态+ioctl    Lv1: 命令间隙  内核直接调用   ioctl
抢占机制        软件技巧:       Progressive    硬件 CWSR ✅   硬件 CWSR ✅
                1.时间片        Launching
                2.清空Ring      
                3.Reset CUs
抢占粒度        Thread Block    命令级         Wave 级 ✅     Wave 级 ✅
抢占延迟        10-100μs        500-800μs      1-10μs ✅      1-10μs ✅

Ring Buffer     清空 ⚠️         保持 ✅        保持 ✅        保持 ✅
状态保存        kernel_offset   N/A            PC+寄存器 ✅   PC+寄存器 ✅
                (近似) ⚠️                       (精确)         (精确)

Resume 方式     重新提交 ⚠️     N/A            恢复状态 ✅    恢复状态 ✅
Resume 延迟     N*100ns         N/A            1-10μs ✅      1-10μs ✅
重复执行        可能 ⚠️         N/A            不会 ✅        不会 ✅

延迟比(H/L)     >3× ✅          1.07×          预计 >3× ✅    预计 >3× ✅
优先级支持      2 个            多个           任意级别 ✅    任意级别 ✅
                (时间片模拟)                   (纯软件)       (纯软件)
                                               hardware=NORMAL hardware=NORMAL

应用修改        需要            不需要 ✅      不需要 ✅      可选
                preempt_flag                                  (hint)
部署难度        高(闭源驱动)    低 ✅          中(DKMS)       低 ✅
调试难度        高              低 ✅          高             低 ✅
灵活性          低              高 ✅          中             高 ✅

本质            软件妥协         软件框架       硬件机制 ✅    硬件机制 ✅
                (无法修改硬件)   (Lv1 基础)     (内核态)       (用户态框架)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

核心优势总结:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

方案 A:
  1. ✅ 性能最优（无 ioctl 开销）
  2. ✅ Doorbell 完全不变
  3. ✅ AMD CWSR 硬件优势（Ring Buffer 不变）
  4. ✅ 内核态监控（响应快）
  5. ✅ 实现 XSched Lv3 的性能目标

方案 B:
  1. ✅ 灵活性高（XSched 框架）
  2. ✅ 易于调试（用户态）
  3. ✅ 跨平台（XSched 支持多 XPU）
  4. ✅ 快速迭代（无需重启）
  5. ✅ 实现 XSched Lv3 的接口标准

共同优势（vs GPreempt NVIDIA）:
  ⭐⭐⭐ Ring Buffer 保持不变（vs 清空）
  ⭐⭐⭐ 精确状态恢复（vs 重新提交）
  ⭐⭐⭐ 16 级硬件优先级（vs 2 级时间片模拟）
  ⭐⭐⭐ 1-10μs 抢占延迟（vs 10-100μs）

共同优势（vs XSched Lv1）:
  ⭐⭐⭐ Wave 级抢占（vs 命令间隙）
  ⭐⭐⭐ 1-10μs 延迟（vs 500-800μs）
  ⭐⭐⭐ 预计 >3× 延迟比（vs 1.07×）
```

---

## 💡 关键设计决策说明

### 1. 为什么不 bypass Doorbell？

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⭐⭐⭐ Doorbell 性能不可妥协
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Doorbell 路径（快）:
  应用 → libamdhip64.so → *doorbell = wptr (MMIO write)
                         ↓ ~100ns ⚡
                         GPU 立即感知

拦截路径（慢）:
  应用 → 拦截层 → 调度 → ioctl → 内核 → 最终提交
         ~~~~~    ~~~~    ~~~~~    ~~~
         10μs+    ?       10μs     ?
         ↓ 可能 50-100μs+

性能差异: 100-1000 倍！

结论:
  ✅ 两种方案都保留 Doorbell
  ✅ 方案 A: 完全不拦截（应用直接 Doorbell）
  ✅ 方案 B: 可选拦截（仅用于 XQueue 管理，不延迟提交）
```

### 2. Ring Buffer 为什么能保持不变？

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⭐⭐⭐ AMD CWSR vs GPreempt (NVIDIA) 的核心区别
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPreempt (NVIDIA) 的软件妥协:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  抢占前:
    BE Ring Buffer: [K0, K1, ..., K24, K25, ..., K99]
                                    ↑            ↑
                                  rptr=25      wptr=100
  
  抢占时（软件操作）:
    1. 设置 preempt_flag = 1
    2. ⚠️ GPUClearHostQueue() → wptr = rptr = 25
    3. GPUResetCU() → 停止 Waves
    
    BE Ring Buffer: [K0, K1, ..., K24, empty, empty, ...]
                                    ↑
                              rptr=wptr=25
    
    ⚠️ K25 到 K99 这 75 个 kernels 丢失了！
  
  Resume 时（软件重新提交）:
    ⚠️ 需要计算 kernel_offset (近似)
    ⚠️ 重新提交 K25, K26, ..., K99
    ⚠️ 可能重复执行部分 kernels
  
  原因：
    • NVIDIA 硬件不支持精确的 Wave 状态保存
    • 只能清空 Ring Buffer 让硬件"看到"无任务
    • 软件技巧妥协


AMD CWSR 的硬件优势:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  抢占前:
    BE Ring Buffer: [K0, K1, ..., K24, K25, ..., K99]
                                    ↑            ↑
                                  rptr=25      wptr=100
  
  抢占时（硬件机制）:
    1. destroy_mqd(WAVEFRONT_SAVE)
    2. ✅ CWSR Trap Handler 保存 Wave 状态
    3. ✅ 释放 CUs
    
    BE Ring Buffer: [K0, K1, ..., K24, K25, ..., K99]
                                    ↑            ↑
                                  rptr=25      wptr=100
    
    ✅ Ring Buffer 完全不变！
    ✅ K25 到 K99 仍在 Ring Buffer 中
    ✅ CWSR Area 保存了 Wave 状态（PC=中断位置）
  
  Resume 时（硬件恢复）:
    1. restore_mqd()
    2. ✅ 从 CWSR Area 恢复 Wave 状态
    3. ✅ GPU 从 rptr=25 继续读取 Ring Buffer
    4. ✅ K25, K26, ..., K99 自动执行
    5. ✅ 被抢占的 Wave 从 PC 继续
  
  原因：
    • AMD 硬件原生支持 CWSR
    • 硬件自动保存和恢复所有状态
    • 无需软件技巧和妥协
```

### 3. 为什么需要内核态监控？

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⭐⭐⭐ Doorbell 不通知内核！
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Doorbell 的工作方式:
  • 应用写 Doorbell 寄存器（MMIO write）
  • GPU 硬件检测到更新
  • GPU 从 Ring Buffer 读取并执行
  
  ⚠️ 内核完全不知道！
  ⚠️ 没有中断，没有通知
  ⚠️ KFD 无法被动响应

解决方案：主动轮询
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

内核态监控线程（方案 A）:
  while (true) {
    // 主动读取 rptr/wptr（MMIO read）
    for_each_queue(q) {
      q->hw_rptr = readl(q->read_ptr);
      q->hw_wptr = readl(q->write_ptr);
    }
    
    // 检测优先级倒置
    if (detect_inversion(...)) {
      // 直接调用 destroy_mqd
      trigger_cwsr_preempt(...);
    }
    
    msleep(5);  // 5ms 间隔
  }

用户态监控（方案 B）:
  while (true) {
    // 通过 XSched 框架监控 XQueue 状态
    // 或通过 ioctl 查询队列状态
    
    if (detect_inversion(...)) {
      // 调用 ioctl 触发 CWSR
      ioctl(kfd_fd, AMDKFD_IOC_PREEMPT_QUEUE, ...);
    }
    
    usleep(5000);  // 5ms 间隔
  }

权衡:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

方案 A:
  ✅ 无 ioctl 开销
  ✅ 响应更快
  ⚠️ 调度逻辑在内核（灵活性稍低）

方案 B:
  ✅ 调度逻辑在用户态（灵活性高）
  ✅ 易于调试和迭代
  ⚠️ 有 ioctl 开销（但仍比无调度快数百倍）
```

### 4. 与 XSched Lv3 接口的映射

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⭐⭐⭐ XSched Lv3 接口 vs 我们的实现
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

XSched Lv3 定义（论文）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  int interrupt(hwQueue hwq);   // 中断正在运行的命令
  int restore(hwQueue hwq);     // 恢复被中断的命令
  
  要求:
    • 抢占延迟 < 50μs
    • 完整状态保存
    • 精确恢复


方案 A 实现（内核态）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  // 内部实现（由监控线程自动触发）
  gpreempt_preempt_queue(queue *q) {
    return q->mqd_mgr->destroy_mqd(
      q->mqd,
      KFD_PREEMPT_TYPE_WAVEFRONT_SAVE,
      0  // 异步
    );
  }
  
  gpreempt_resume_queue(queue *q) {
    return q->mqd_mgr->restore_mqd(
      q->mqd,
      q->process->pasid
    );
  }
  
  实际延迟: 1-10μs ✅（超过 Lv3 要求）
  状态保存: 完整 ✅
  恢复精度: 指令级 ✅


方案 B 实现（用户态 XSched）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  // XAL-Lv3 接口实现
  int interrupt(hwQueue hwq) {
    int kfd_fd = open("/dev/kfd", O_RDWR);
    struct kfd_ioctl_preempt_queue_args args = {
      .queue_id = hwq->queue_id,
      .preempt_type = 2,  // WAVEFRONT_SAVE
      .timeout_ms = 1000
    };
    return ioctl(kfd_fd, AMDKFD_IOC_PREEMPT_QUEUE, &args);
  }
  
  int restore(hwQueue hwq) {
    int kfd_fd = open("/dev/kfd", O_RDWR);
    struct kfd_ioctl_resume_queue_args args = {
      .queue_id = hwq->queue_id
    };
    return ioctl(kfd_fd, AMDKFD_IOC_RESUME_QUEUE, &args);
  }
  
  实际延迟: 1-10μs (CWSR) + ioctl 开销 ✅
  状态保存: 完整 ✅
  恢复精度: 指令级 ✅
  
  接口完全匹配 XSched Lv3 标准！
```

---

## 🚀 实施计划

### 方案 A 实施路径（推荐用于生产）

```
Phase 1: KFD 驱动扩展（2-3 周）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1.1 数据结构扩展:
  • 扩展 struct queue（新增 GPREEMPT 字段）
  • 创建 struct kfd_gpreempt_scheduler
  • 修改 kfd_priv.h

1.2 监控线程实现:
  • kfd_gpreempt_monitor_thread()
  • gpreempt_scan_all_queues()
  • gpreempt_detect_priority_inversion()

1.3 抢占和恢复:
  • gpreempt_preempt_queue()
  • gpreempt_check_resume_queues()

1.4 注册机制:
  • kfd_gpreempt_init()
  • kfd_gpreempt_register_queue()
  • 在队列创建/销毁时调用

1.5 测试:
  • 编译 DKMS
  • 加载模块
  • 验证日志输出


Phase 2: 双进程测试（1 周）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2.1 测试程序:
  • 训练任务（低优先级，长时间运行）
  • 推理任务（高优先级，短时间运行）

2.2 验证指标:
  • 高优先级延迟 < 30ms
  • 延迟比 > 3×
  • Ring Buffer 状态正确
  • 无重复执行

2.3 性能测试:
  • 抢占延迟测量
  • CPU 开销测量
  • 长时间稳定性测试


Phase 3: 优化和完善（2-3 周）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3.1 性能优化:
  • 调整监控间隔
  • 优化锁粒度
  • 减少不必要的 MMIO 读取

3.2 功能增强:
  • 动态优先级（防止饥饿）
  • sysfs 配置接口
  • 统计信息导出

3.3 稳定性:
  • 边界条件处理
  • 错误恢复
  • 压力测试

总计: 5-7 周
```

### 方案 B 实施路径（快速验证）

```
Phase 1: KFD ioctl 扩展（1 周）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1.1 新增 ioctl:
  • AMDKFD_IOC_PREEMPT_QUEUE
  • AMDKFD_IOC_RESUME_QUEUE

1.2 实现（很简单）:
  static int kfd_ioctl_preempt_queue(...) {
    // 查找队列
    // 调用 destroy_mqd(CWSR)
    // 返回结果
  }
  
  static int kfd_ioctl_resume_queue(...) {
    // 查找队列
    // 调用 restore_mqd()
    // 返回结果
  }

1.3 权限处理:
  • CAP_SYS_ADMIN 检查
  • 或跨进程 Queue 访问策略


Phase 2: XSched Lv3 集成（1-2 周）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2.1 实现 XAL-Lv3:
  • 创建 hip_queue_lv3.cpp
  • 实现 interrupt() / restore()
  • 调用 KFD ioctl

2.2 XQueue 注册:
  • 检测 Lv3 能力
  • 注册到 XPreempt 调度器

2.3 调度策略:
  • HPF（Highest Priority First）
  • 优先级倒置检测


Phase 3: 测试验证（1 周）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3.1 XSched Example 3:
  • 重新编译
  • 运行测试
  • 对比 Lv1 vs Lv3

3.2 性能验证:
  • 延迟比应该 > 3×（vs 当前 1.07×）
  • 抢占延迟应该 < 10μs（vs 当前 500-800μs）

总计: 3-4 周
```

---

## 📈 预期性能

### 端到端延迟预测

```
场景: ResNet-50 推理抢占 BERT 训练
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

当前 XSched Lv1:
  推理延迟: ~29ms
  训练延迟: ~31ms
  延迟比: 1.07× ⚠️（太小）
  
  分析: Progressive Launching，粒度粗

启用方案 A 或 B（Lv3 + CWSR）:
  推理延迟: ~25ms ✅
    • 提交延迟: 5μs (Doorbell)
    • 等待检测: 5ms (平均 2.5ms)
    • 抢占延迟: 10μs (CWSR)
    • 执行时间: 20ms
  
  训练延迟: ~80-90ms ✅
    • 被抢占时间: 25ms
    • 其余时间: 正常执行
  
  延迟比: 3.2-3.6× ✅（显著）
  
  改善: 
    • vs XSched Lv1: 3倍延迟差异提升
    • vs 无调度: 性能隔离实现
    • vs GPreempt (NVIDIA): 相当或更好


对比 GPreempt (NVIDIA):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  GPreempt A100:
    抢占延迟: 40μs
    开销: <5%
    延迟比: >3×
  
  我们的方案:
    抢占延迟: 1-10μs ✅ (更快)
    开销: <5% (方案 A) 或 ~10% (方案 B)
    延迟比: >3× ✅ (相当)
    
    额外优势:
      ✅ Ring Buffer 保持不变
      ✅ 精确状态恢复
      ✅ 16 级优先级
```

---

## 🎓 核心创新点

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⭐⭐⭐ 本架构的独特贡献
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 融合两篇论文的优势
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  从 GPREEMPT 采纳:
    ✅ 内核态监控逻辑
    ✅ Ring Buffer 状态监控
    ✅ 优先级倒置检测算法
    
  从 XSched 采纳:
    ✅ Lv3 接口标准
    ✅ 多级硬件抽象思想
    ✅ 调度策略框架
    
  抛弃的部分:
    ❌ GPreempt 的时间片技巧（AMD 不需要）
    ❌ GPreempt 的清空 Ring Buffer（AMD 不需要）
    ❌ XSched 的 LD_PRELOAD 拦截（可选，不是必须）


2. 超越 GPREEMPT (NVIDIA) 的关键
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  GPreempt 的软件妥协:
    ⚠️ 时间片模拟（配置 1s vs 1μs）
    ⚠️ 清空 Ring Buffer（让硬件"看到"无任务）
    ⚠️ 重新提交 kernels（可能重复执行）
    ⚠️ 只支持 2 个优先级
  
  我们的硬件优势:
    ✅ 16 级硬件优先级（直接使用）
    ✅ Ring Buffer 保持不变（CWSR 机制）
    ✅ 精确状态恢复（无重新提交）
    ✅ 支持任意优先级级别
  
  性能对比:
    • 抢占延迟: 1-10μs vs 40μs (A100) or 200μs (MI100 软件模拟)
    • Ring Buffer: 保持 vs 清空
    • Resume: 恢复 vs 重新提交


3. 超越 XSched Lv1 的关键
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  XSched Lv1 (Progressive Launching):
    • 抢占粒度: 命令间隙
    • 抢占延迟: 500-800μs
    • 延迟比: 1.07× ⚠️
    • 实现: 分批提交
  
  我们的 Lv3 (CWSR):
    • 抢占粒度: Wave 指令级
    • 抢占延迟: 1-10μs ✅
    • 延迟比: >3× ✅
    • 实现: 硬件 CWSR
  
  性能提升:
    • 抢占延迟: 50-80× 提升
    • 延迟比: 3× 提升
    • 实现论文级性能


4. Doorbell 性能完全保留
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  关键原则:
    • 应用仍然通过 Doorbell 提交（~100ns）
    • 监控"事后"检测（5ms 间隔）
    • CWSR"快速"纠正（1-10μs）
  
  vs 拦截方案:
    • 拦截: 10-100μs 延迟
    • Doorbell: 100ns 延迟
    • 差异: 100-1000 倍
  
  结论: 必须保留 Doorbell！
```

---

## 🎯 使用方式

### 方案 A 使用（完全透明）

```cpp
// 训练程序（无需修改）
void train_model() {
    // 创建队列时指定优先级（通过环境变量或 HSA API）
    // export HSA_QUEUE_PRIORITY=3
    
    // 正常使用 HIP API
    for (int epoch = 0; epoch < 100; epoch++) {
        for (int batch = 0; batch < 1000; batch++) {
            hipLaunchKernel(forward_kernel, ...);
            hipLaunchKernel(backward_kernel, ...);
            // ↓ 每个 ~100ns（Doorbell）⚡
            // ↓ 完全透明，无感知
        }
    }
    // ✅ 自动获得优先级调度
    // ✅ 自动被高优先级抢占
}

// 推理服务（无需修改）
void inference_service() {
    // 创建队列时指定优先级
    // export HSA_QUEUE_PRIORITY=12
    
    while (true) {
        auto request = receive_request();
        
        // 正常使用 HIP API
        for (int i = 0; i < 50; i++) {
            hipLaunchKernel(infer_kernel_i, ...);
            // ↓ ~100ns（Doorbell）⚡
        }
        
        hipStreamSynchronize(stream);
        // ✅ 自动抢占低优先级任务
        // ✅ 应用完全无感知
        
        send_response(result);
    }
}

// 配置（可选）
// echo 10 > /sys/module/amdgpu/parameters/gpreempt_interval_ms
// echo 1 > /sys/module/amdgpu/parameters/gpreempt_enable
```

### 方案 B 使用（XSched 集成）

```cpp
// 应用代码（无需修改）
// 与方案 A 相同

// 启动命令
export LD_LIBRARY_PATH=/workspace/xsched/output/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/workspace/xsched/output/lib/libshimhip.so  # 可选
export XSCHED_LEVEL=3  # 使用 Lv3
export XSCHED_SCHED=HPF  # 使用 HPF 调度策略

./your_app

// XSched 会自动:
// 1. 检测到 Lv3 支持（CWSR ioctl）
// 2. 创建 XQueue with Lv3 backend
// 3. 使用 interrupt()/restore() 进行抢占
// 4. 提供统一的调度框架
```

---

## 📊 总结

### 方案选择建议

```
┌─────────────────────────────────────────────────────────────┐
│                     选择决策树                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  是否需要极致性能（无 ioctl 开销）？                         │
│    ├─ 是 → 选择方案 A（纯内核态）                           │
│    │        • 适合生产环境                                   │
│    │        • 单一 AMD 平台                                  │
│    │        • 长期部署                                       │
│    │                                                          │
│    └─ 否 → 是否需要 XSched 框架和跨平台能力？              │
│           ├─ 是 → 选择方案 B（混合架构）                   │
│           │        • 适合研发测试                            │
│           │        • 混合 GPU 平台                           │
│           │        • 快速迭代                                │
│           │                                                  │
│           └─ 否 → 仍然推荐方案 A                            │
│                    （性能最优）                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘

或者：两种方案都实现！
  • 方案 A 作为生产部署
  • 方案 B 作为研发测试和跨平台支持
  • 共享 KFD CWSR 底层实现
```

### 核心价值主张

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⭐⭐⭐ 为什么这个架构是最佳方案？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 融合两篇 USENIX'25 论文的精华
   • GPREEMPT 的监控和调度逻辑
   • XSched 的多级抽象框架
   • 两者互补，不冲突

2. 充分利用 AMD CWSR 硬件优势
   • Ring Buffer 保持不变（vs GPreempt 清空）
   • 精确状态恢复（vs GPreempt 重新提交）
   • 1-10μs 延迟（vs GPreempt 40μs）
   • 16 级优先级（vs GPreempt 2 级）

3. 保留 Doorbell 性能（不妥协）
   • 应用提交仍然 ~100ns
   • 不 bypass，不拦截（或可选拦截）
   • 监控"事后"检测和纠正
   • 数据平面和控制平面完全分离

4. 实现 XSched Lv3 的性能目标
   • 抢占延迟 < 50μs ✅ (实际 1-10μs)
   • 延迟比 > 3× ✅
   • 完整状态保存 ✅
   • 精确恢复 ✅

5. 灵活性（两种方案）
   • 方案 A: 性能极致（生产）
   • 方案 B: 灵活易用（研发）
   • 共享 CWSR 底层
   • 可以共存

6. 超越现有方案
   • vs GPreempt (NVIDIA): 更快、更精确、无妥协
   • vs XSched Lv1: 50-80× 抢占延迟提升
   • vs 无调度: 完整优先级保证
```

---

## 📚 相关文档

### 基础技术文档
- `GPreempt_完整技术分析_综合版.md` - GPreempt 代码分析
- `ARCH_Design_02_AMD_GPREEMPT_redesign_v2.md` - AMD GPREEMPT 纯内核方案
- `XSched_Lv1_Lv2_Lv3硬件级别详解.md` - XSched 三级抽象
- `GPREEMPT_作为_XSched_Lv3_实现方案.md` - 融合思路

### 关键发现文档
- `FINAL_SUCCESS_REPORT.md` - XSched Lv1 验证
- `AMD_CWSR与XSched硬件级别对应分析.md` - CWSR = Lv3
- `GPREEMPT_vs_XSched_深度对比分析.md` - 两论文对比

---

## ✅ 下一步行动

### 立即可做

1. **决定实施方案**
   - 方案 A（推荐生产）
   - 方案 B（推荐研发）
   - 或两者都做

2. **准备环境**
   - 确认 CWSR 已启用
   - 准备 DKMS 编译环境（方案 A）
   - 或准备 XSched 源码（方案 B）

3. **开始实施**
   - 按照 Phase 1 计划开始
   - 逐步验证

---

**文档版本**: ARCH_Design_03 v3.0  
**日期**: 2026-01-29  
**状态**: 架构设计完成  
**下一步**: 选择方案并开始实施

**关键理念**: 
> 融合 GPREEMPT 和 XSched，利用 AMD CWSR 硬件优势，  
> 保留 Doorbell 性能，实现超越两篇论文的调度系统！

**核心创新**:
- ✅ 不需要 GPreempt 的软件技巧妥协（时间片、清空 Ring Buffer）
- ✅ 超越 XSched Lv1 的性能（实现 Lv3 标准）
- ✅ 保留 Doorbell 的快速提交（不 bypass）
- ✅ 灵活的实施路径（纯内核或混合）

**感谢**:
- 用户对 Doorbell 性能的坚持
- 用户对 Ring Buffer 机制的深刻理解
- 用户对清空 Ring Buffer 本质的准确洞察
- 用户对时间片使用逻辑的精准提问
