# GPREEMPT (Paper#1) 作为 XSched (Paper#2) Lv3 实现方案

**日期**: 2026-01-27  
**技术设想**: 将两篇论文正确融合  
**核心原则**: 保留 Doorbell + 内核态监控

---

## 📌 技术设想概述

```
目标:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

将 GPREEMPT (Paper#1) 作为 XSched (Paper#2) Lv3 的基础 feature

关键约束:
✓ 必须保留 Doorbell 性能（不能拦截 HIP API）
✓ GPREEMPT 提供内核态调度能力
✓ XSched 提供调度框架和策略
✓ CWSR 提供硬件支持
```

---

## 🎯 4 个 Feature 的正确关系

### 技术栈分层

```
┌─────────────────────────────────────────────────────────────┐
│  调度框架层: XSched (Paper#2)                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ • 优先级队列管理                                       │  │
│  │ • 调度策略（Strict Priority, Weighted Fair, etc）    │  │
│  │ • 统计和监控                                          │  │
│  │ • Lv3 接口封装                                        │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│                  Lv3 接口调用                                │
│          interrupt() → PREEMPT_QUEUE ioctl                   │
│          restore()   → RESUME_QUEUE ioctl                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  内核调度层: GPREEMPT (Paper#1)                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ • 异步监控线程（1-10ms 间隔）                         │  │
│  │ • 读取队列状态（rptr/wptr）                           │  │
│  │ • 优先级倒置检测                                      │  │
│  │ • 触发 CWSR 抢占                                      │  │
│  │ • 提供 ioctl 接口                                     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  硬件层: CWSR (AMD MI308X)                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ • Wave-level 上下文保存/恢复                          │  │
│  │ • 1-10μs 抢占延迟                                     │  │
│  │ • 完整状态保存（PC, SGPRs, VGPRs, LDS）              │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  应用层: PyTorch / HIP Applications                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ hipLaunchKernel()                                      │  │
│  │   ↓                                                    │  │
│  │ libamdhip64.so                                         │  │
│  │   ↓                                                    │  │
│  │ *doorbell = wptr (MMIO write, ~100ns)  ← ✅ 不拦截！  │  │
│  │   ↓                                                    │  │
│  │ GPU 立即感知并执行                                     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Feature 角色定位

| Feature | 角色 | 位置 | 性能指标 | 接口 |
|---------|------|------|----------|------|
| **Doorbell** | 快速任务提交 | 用户态→GPU | ~100ns | MMIO write |
| **CWSR** | 硬件抢占能力 | GPU硬件 | 1-10μs | Trap Handler |
| **GPREEMPT** | 内核调度引擎 | KFD驱动 | 1-10ms | ioctl |
| **XSched** | 调度框架 | 用户态 | N/A | Lv3 API |

---

## 🔑 关键设计原则

### 1. 保留 Doorbell 性能（最重要！）⭐⭐⭐⭐⭐

```
为什么必须保留 Doorbell?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

性能对比:

Doorbell 路径（快）:
  应用 → libamdhip64.so → *doorbell = wptr (MMIO write)
                         ↓ ~100ns
                         GPU 立即感知

拦截路径（慢）:
  应用 → LD_PRELOAD 拦截 → 用户态调度 → ioctl → 内核
         ~~~~~~~~~~~~~~    ~~~~~~~~~~    ~~~~~    ~~~
         函数劫持开销       调度决策      系统调用  上下文切换
                         ↓ 10-100μs+
                         
性能差异: 100-1000倍！

结论: 必须保留 doorbell，不能拦截！
```

### 2. 内核态异步监控（核心思想）

```
GPREEMPT 监控机制:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

原理: 应用通过 doorbell 快速提交，内核"事后"监控和抢占

伪代码:
    // 内核监控线程
    while (true) {
        // 1. 读取所有队列状态
        for_each_queue(q) {
            q->rptr = read_hw_rptr(q);
            q->wptr = read_hw_wptr(q);
            q->pending = q->wptr - q->rptr;
            q->is_active = (q->pending > 0);
        }
        
        // 2. 检测优先级倒置
        high_prio_q = find_highest_priority_waiting_queue();
        low_prio_q = find_lowest_priority_running_queue();
        
        if (high_prio_q && low_prio_q &&
            high_prio_q->priority > low_prio_q->priority) {
            // 3. 触发抢占
            cwsr_preempt(low_prio_q);
            
            // 低优先级队列被挂起，高优先级队列自动获得资源
        }
        
        // 4. 定期检查（可配置）
        sleep(check_interval_ms);  // 1-10ms
    }

特点:
✓ 不拦截任务提交（doorbell 保持快速）
✓ 异步检测优先级倒置
✓ 使用 CWSR 快速抢占（1-10μs）
✓ 调度延迟 = 监控间隔（1-10ms）
```

### 3. XSched Lv3 接口映射

```
XSched Lv3 需求 → GPREEMPT 实现:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

XSched Lv3 接口:
    int interrupt(hwQueue hwq);   // 中断正在运行的命令
    int restore(hwQueue hwq);     // 恢复被中断的命令

GPREEMPT ioctl 实现:
    int interrupt(hwQueue hwq) {
        int kfd_fd = open("/dev/kfd", O_RDWR);
        
        struct kfd_ioctl_preempt_queue_args args = {
            .queue_id = hwq->queue_id,
            .preempt_type = 2,  // WAVEFRONT_SAVE (CWSR)
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

完美匹配！
```

---

## 🚀 Doorbell + CWSR 完整时序分析

### 关键问题：高优先级任务如何与 CWSR 交互？

```
核心机制理解:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 每个应用有自己的队列（Queue）
   - 训练任务 → Queue_train (priority=1)
   - 推理服务 → Queue_infer (priority=10)

2. 两个队列都通过 doorbell 提交任务
   - Doorbell 是"per-queue"的机制
   - 写 doorbell 只是通知 GPU："我的队列有新任务了"

3. GPU 的 Command Processor 从多个队列调度任务
   - 但 AMD GPU 的硬件调度器不支持优先级！
   - 或者支持有限的优先级（不够细粒度）

4. GPREEMPT 的作用：软件优先级调度
   - 监控所有队列状态
   - 检测优先级倒置
   - 使用 CWSR 强制抢占低优先级队列
```

### 场景：推理服务抢占训练任务（详细时序）

```
初始状态 (T < 0):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌─────────────────────────────────────────┐
│ 训练任务（低优先级）                     │
│   Queue_train (priority=1)              │
│   ├─ Ring Buffer: [kernel1, kernel2...] │
│   ├─ rptr: 100, wptr: 500              │
│   └─ 状态: ✅ 正在 GPU 上执行           │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ GPU Command Processor                   │
│   ├─ 正在执行 Queue_train 的 kernel     │
│   └─ CU 占用率: 100%                    │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ GPREEMPT 监控线程                        │
│   ├─ 上次检查: T=-5ms                   │
│   ├─ 发现: 只有 Queue_train 活跃        │
│   └─ 状态: 无需抢占 ✅                  │
└─────────────────────────────────────────┘


T=0ms: 推理请求到达
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌─ 推理服务（高优先级）─────────────────┐
│ hipLaunchKernel()                     │
│   ↓                                   │
│ libamdhip64.so:                       │
│   • 将 kernel 参数写入 Ring Buffer    │
│   • wptr += 1                         │
│   • *doorbell_infer = wptr            │  ← MMIO write (~100ns)
│     (告诉 GPU：我的队列有新任务了)    │
│                                       │
│ 用户态延迟: ~100ns ✅                 │
│ 函数返回，应用继续执行                │
└───────────────────────────────────────┘

此时 GPU 状态:
┌─────────────────────────────────────────┐
│ Queue_train (priority=1)                │
│   ├─ rptr: 120, wptr: 500              │
│   ├─ pending: 380 个 kernel            │
│   └─ 状态: ✅ 正在执行                  │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Queue_infer (priority=10) ← 新提交！    │
│   ├─ rptr: 0, wptr: 1                  │
│   ├─ pending: 1 个 kernel              │
│   └─ 状态: ⏳ 等待调度                  │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ GPU Command Processor                   │
│   ├─ 收到 doorbell 通知                │
│   ├─ 知道 Queue_infer 有新任务          │
│   ├─ 但仍在执行 Queue_train ❌          │
│   │   (因为硬件调度器不支持抢占)        │
│   └─ Queue_infer 的任务在队列中等待 ⏳   │
└─────────────────────────────────────────┘

关键点:
  ✓ 高优先级任务已通过 doorbell 提交（~100ns）
  ✓ GPU 已感知到 Queue_infer 有任务
  ❌ 但 GPU 硬件不会主动抢占 Queue_train
  ⏳ 高优先级任务被迫等待


T=5ms: GPREEMPT 监控线程检查（周期性）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌─ GPREEMPT 监控线程 ───────────────────┐
│ 定时器触发（check_interval_ms=5ms）   │
│   ↓                                   │
│ 扫描所有队列，读取硬件寄存器:          │
│                                       │
│ Queue_train:                          │
│   • rptr_hw = 120 (从 MMIO 读取)      │
│   • wptr_hw = 500                     │
│   • pending = 380 ✓ 有任务            │
│   • is_running = true ✓               │
│                                       │
│ Queue_infer:                          │
│   • rptr_hw = 0   (从 MMIO 读取)      │
│   • wptr_hw = 1                       │
│   • pending = 1 ✓ 有任务              │
│   • is_running = false ⏳ 等待        │
│                                       │
│ 优先级倒置检测:                        │
│   high_q = Queue_infer (priority=10)  │
│   low_q  = Queue_train (priority=1)   │
│   ⚠️ 倒置！high_q 在等待，low_q 在运行 │
│                                       │
│ 决策: 抢占 Queue_train                 │
│   ↓                                   │
│ 调用: kfd_queue_preempt_single(       │
│         Queue_train,                  │
│         WAVEFRONT_SAVE,  // CWSR      │
│         timeout=1000ms)               │
└───────────────────────────────────────┘
                ↓
┌─ KFD 驱动执行抢占 ────────────────────┐
│ 1. checkpoint_mqd(Queue_train)        │
│    • 备份 MQD (Memory Queue Descriptor)│
│                                       │
│ 2. destroy_mqd(Queue_train, CWSR)     │
│    • 向 GPU 发送抢占命令              │
│    • 触发 Trap Handler                │
└───────────────────────────────────────┘
                ↓
┌─ GPU 硬件执行 CWSR ───────────────────┐
│ Trap Handler (GPU 固件) 执行:         │
│                                       │
│ 1. 暂停 Queue_train 的所有 Wavefronts │
│    • 当前正在执行的 Waves             │
│                                       │
│ 2. 保存完整状态到 CWSR 内存:          │
│    ├─ PC (Program Counter)            │
│    ├─ SGPRs (Scalar GPRs, ~128 regs) │
│    ├─ VGPRs (Vector GPRs, ~256 regs) │
│    ├─ ACC VGPRs (Accumulator)         │
│    ├─ LDS (Local Data Share)          │
│    ├─ Mode registers                  │
│    └─ Status registers                │
│                                       │
│ 3. 清空 CUs，释放资源                 │
│                                       │
│ 4. 更新 Queue_train 状态:             │
│    • is_preempted = true              │
│    • save_area = CWSR 内存地址        │
│                                       │
│ 延迟: 1-10μs ✅                       │
└───────────────────────────────────────┘

T=5.01ms: GPU 自动切换到高优先级队列
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌─ GPU Command Processor ───────────────┐
│ CWSR 抢占完成后:                       │
│                                       │
│ 1. Queue_train 已被挂起               │
│    • 不再从该队列读取命令             │
│                                       │
│ 2. 扫描其他活跃队列                   │
│    • 发现 Queue_infer 有待处理任务    │
│    • wptr > rptr                      │
│                                       │
│ 3. 开始执行 Queue_infer               │
│    • 从 Ring Buffer 读取 kernel 参数  │
│    • 分配 CUs                         │
│    • 启动 Wavefronts                  │
│                                       │
│ 切换延迟: <1ms ✅                     │
└───────────────────────────────────────┘

此时 GPU 状态:
┌─────────────────────────────────────────┐
│ Queue_train (priority=1)                │
│   ├─ 状态: 🛑 已挂起 (PREEMPTED)       │
│   ├─ CWSR save area: 保存了所有状态    │
│   └─ 等待恢复                          │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Queue_infer (priority=10)               │
│   ├─ 状态: ✅ 正在执行                  │
│   ├─ rptr: 0 → 1 (逐渐增加)           │
│   └─ CU 占用率: 100%                   │
└─────────────────────────────────────────┘


T=5.01ms ~ T=25ms: 高优先级任务执行
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌─ GPU ─────────────────────────────────┐
│ 执行 Queue_infer 的 kernel             │
│   • 完全占用 GPU                       │
│   • 训练任务保持挂起状态               │
│   • 执行时间: ~20ms                    │
│                                       │
│ 推理服务端到端延迟:                    │
│   提交延迟: ~100ns                     │
│   + 等待抢占: ~5ms                     │
│   + 执行时间: ~20ms                    │
│   ────────────────                    │
│   总计: ~25ms ✅                       │
└───────────────────────────────────────┘


T=25ms: 高优先级任务完成
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌─ Queue_infer 状态 ────────────────────┐
│   rptr: 1, wptr: 1                    │
│   pending: 0 ✓ 队列空                 │
│   状态: 空闲                          │
└───────────────────────────────────────┘


T=30ms: GPREEMPT 监控线程恢复低优先级队列
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌─ GPREEMPT 监控线程 ───────────────────┐
│ 定时器再次触发（5ms 间隔）             │
│   ↓                                   │
│ 扫描队列状态:                          │
│   Queue_infer: 空闲 ✓                 │
│   Queue_train: 挂起，有待处理任务 ⏳   │
│                                       │
│ 决策: 恢复 Queue_train                 │
│   ↓                                   │
│ 调用: kfd_queue_resume_single(        │
│         Queue_train)                  │
└───────────────────────────────────────┘
                ↓
┌─ KFD 驱动执行恢复 ────────────────────┐
│ 1. restore_mqd(Queue_train)           │
│    • 从备份恢复 MQD                   │
│                                       │
│ 2. load_mqd(Queue_train)              │
│    • 重新加载到 GPU                   │
│    • 指向 CWSR save area              │
└───────────────────────────────────────┘
                ↓
┌─ GPU 硬件恢复 Wavefronts ─────────────┐
│ 1. 从 CWSR 内存读取保存的状态          │
│                                       │
│ 2. 恢复所有 Wavefronts:               │
│    ├─ 恢复 PC                         │
│    ├─ 恢复 SGPRs                      │
│    ├─ 恢复 VGPRs                      │
│    ├─ 恢复 ACC VGPRs                  │
│    ├─ 恢复 LDS                        │
│    └─ 恢复 mode/status registers      │
│                                       │
│ 3. 从断点处继续执行 ✅                │
│    • 就像从未被中断一样               │
│                                       │
│ 恢复延迟: 1-10μs ✅                   │
└───────────────────────────────────────┘


T=30ms+: 训练任务继续执行
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌─ Queue_train 状态 ────────────────────┐
│   状态: ✅ 正在执行（已恢复）          │
│   从断点处继续，应用完全无感知         │
│   总延迟增加: ~25ms (被抢占时间)      │
└───────────────────────────────────────┘
──────────────────────────────────────────────────────────────
┌─ GPREEMPT (内核态) ──────────────────────────┐
│ 监控线程周期性检查:                           │
│                                              │
│ 读取队列状态:                                │
│   Queue_train (priority=1):                 │
│     rptr=100, wptr=500, pending=400 ✓运行中 │
│   Queue_infer (priority=10):                │
│     rptr=0, wptr=50, pending=50 ⚠️等待中    │
│                                              │
│ 检测: 优先级倒置！                           │
│   高优先级队列 (10) 在等待                   │
│   低优先级队列 (1) 在运行                    │
│                                              │
│ 决策: 抢占 Queue_train                       │
│   ↓                                          │
│ kfd_queue_preempt(Queue_train, CWSR)        │
└──────────────────────────────────────────────┘
                ↓
┌─ CWSR 硬件抢占 ────────────────────┐
│ Trap Handler 执行                  │
│   • 保存所有 Wave 状态             │
│   • PC, SGPRs, VGPRs, ACC, LDS    │
│   • 1-10μs 完成 ✅                │
│                                    │
│ Queue_train 被挂起                 │
└────────────────────────────────────┘

T=5.01ms: 推理任务开始执行
──────────────────────────────────────────────────────────────
┌─ GPU ─────────────────────────────┐
│ 执行 Queue_infer 的任务            │
│   • 完全占用 GPU                   │
│   • 训练任务已挂起                 │
│   • 执行时间: 20ms                 │
└────────────────────────────────────┘

T=25ms: 推理任务完成
──────────────────────────────────────────────────────────────
┌─ GPREEMPT 监控线程 ──────────────────────┐
│ 检测到 Queue_infer 空闲                   │
│   ↓                                      │
│ 决策: 恢复 Queue_train                    │
│   ↓                                      │
│ kfd_queue_resume(Queue_train)            │
└──────────────────────────────────────────┘
                ↓
┌─ CWSR 硬件恢复 ────────────────────┐
│ 恢复 Queue_train 的 Wave 状态      │
│   • 从 CWSR 内存读取状态           │
│   • 恢复 PC, 寄存器, LDS           │
│   • 从断点处继续执行 ✅            │
└────────────────────────────────────┘

T=25ms+: 训练任务继续
──────────────────────────────────────────────────────────────
训练任务从被抢占的位置继续执行，完全无感知

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                         完整时间线总结
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

T=0ms      高优先级通过 doorbell 提交         (~100ns)
           ↓
T=0-5ms    高优先级任务在队列中等待            (等待监控检测)
           ↓
T=5ms      GPREEMPT 检测到优先级倒置
           ↓
T=5ms      触发 CWSR 抢占低优先级队列          (1-10μs)
           ↓
T=5ms      高优先级任务开始执行
           ↓
T=25ms     高优先级任务完成
           ↓
T=30ms     GPREEMPT 恢复低优先级队列           (1-10μs)
           ↓
T=30ms+    低优先级从断点继续

结果:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ 高优先级延迟: ~25ms (5ms等待 + 20ms执行)
✓ 低优先级延迟: +25ms (被抢占时间)
✓ Doorbell 性能: 完全保留（~100ns）✅
✓ CWSR 抢占延迟: 1-10μs ✅
✓ 调度检测延迟: 5ms (可配置1-10ms)
```

---

### Doorbell 和 CWSR 的关系总结

```
关键理解:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Doorbell 的作用:
   • 快速通知 GPU："我的队列有新任务了"
   • 每个队列有独立的 doorbell
   • 写 doorbell = MMIO write (~100ns)
   • 高/低优先级队列都使用 doorbell

2. GPU 硬件调度的局限:
   • GPU 会从多个队列读取任务
   • 但硬件调度器不支持细粒度优先级
   • 或者支持有限（如 NVIDIA 的 high/normal/low 3级）
   • 可能导致低优先级任务继续执行

3. CWSR 的作用:
   • 软件强制抢占机制
   • 中断正在运行的低优先级任务
   • 保存 Wave 状态，释放 GPU 资源
   • 让高优先级任务执行

4. GPREEMPT 的作用:
   • 监控线程定期检查所有队列状态
   • 检测优先级倒置（高优先级在等，低优先级在跑）
   • 触发 CWSR 抢占
   • 恢复被抢占的队列

三者关系:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Doorbell (任务提交)
    ↓
  两个队列都有任务，GPU 可能选错
    ↓
GPREEMPT (监控检测)
    ↓
  发现优先级倒置
    ↓
CWSR (强制抢占)
    ↓
  高优先级获得资源

关键点:
✓ Doorbell 不会被 bypass（保留性能）
✓ 高优先级也通过 doorbell 提交
✓ CWSR 是"事后"的抢占机制
✓ 调度延迟 = 监控间隔（1-10ms）
```

### 为什么不在 Doorbell 阶段就做调度？

```
方案对比:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

方案A: 拦截 doorbell（错误）
  应用 → 拦截层 → 调度决策 → doorbell
          ~~~~~    ~~~~~~~~    ~~~~~~~
          开销     延迟        失去快速通道
  
  问题:
  ❌ 拦截开销大（10-100μs）
  ❌ 需要 LD_PRELOAD
  ❌ 损失 doorbell 性能

方案B: doorbell + 事后监控（正确）
  应用 → doorbell (~100ns)
         ↓
       GPU 执行（可能选错队列）
         ↓
       GPREEMPT 监控检测（1-10ms 后）
         ↓
       CWSR 纠正（1-10μs 抢占）
  
  优点:
  ✅ 保留 doorbell 性能
  ✅ 应用完全透明
  ✅ 可靠的优先级保证
  
  权衡:
  ⚠️ 调度延迟 = 监控间隔（可配置）
     - 1-10ms 对大多数场景足够
     - 远小于无优先级时的延迟（可能数秒）
```

### 与 NVIDIA/XSched 的对比

```
NVIDIA GPU (支持硬件优先级):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

应用 → doorbell → GPU Command Processor
                   ↓
                 硬件调度器（支持优先级）
                   ↓
                 自动选择高优先级队列 ✅
                 
特点:
✓ 硬件级优先级支持
✓ 无需软件干预
✓ 延迟 <1ms

但:
⚠️ 只支持有限级别（如 3 级）
⚠️ 无法细粒度控制


AMD GPU + GPREEMPT (软件优先级):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

应用 → doorbell → GPU Command Processor
                   ↓
                 硬件调度器（优先级支持有限）
                   ↓
                 可能选错队列 ❌
                   ↓
         GPREEMPT 监控检测（1-10ms）
                   ↓
         CWSR 强制抢占（1-10μs）✅
         
特点:
✓ 软件实现，灵活
✓ 支持任意优先级级别
✓ 使用 CWSR 硬件能力
✓ 保留 doorbell 性能

权衡:
⚠️ 调度延迟稍高（1-10ms vs <1ms）
  但仍远优于无优先级（数秒）
```

### 多队列 + Doorbell 架构图

```
完整系统架构（包含 Doorbell）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────┐
│  用户态应用层                                                │
│                                                             │
│  ┌──────────────────┐          ┌──────────────────┐        │
│  │ 训练任务          │          │ 推理服务          │        │
│  │ (Low Priority)   │          │ (High Priority)  │        │
│  └────────┬─────────┘          └────────┬─────────┘        │
│           │                             │                  │
│           ▼ hipLaunchKernel()          ▼ hipLaunchKernel()│
│  ┌──────────────────┐          ┌──────────────────┐        │
│  │ libamdhip64.so   │          │ libamdhip64.so   │        │
│  └────────┬─────────┘          └────────┬─────────┘        │
└───────────┼──────────────────────────────┼──────────────────┘
            │                              │
            │ (~100ns each)                │
            ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Doorbell 层（MMIO）                                         │
│                                                             │
│  *doorbell_train = wptr_train    *doorbell_infer = wptr_infer
│         ↓                                  ↓                │
│  ┌─────────────────┐            ┌─────────────────┐        │
│  │ Doorbell #0     │            │ Doorbell #1     │        │
│  │ (Queue_train)   │            │ (Queue_infer)   │        │
│  └─────────────────┘            └─────────────────┘        │
└─────────────────────────────────────────────────────────────┘
            ↓                              ↓
┌─────────────────────────────────────────────────────────────┐
│  GPU 硬件层                                                  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Command Processor（硬件调度器）                        │ │
│  │                                                       │ │
│  │  ┌──────────────┐        ┌──────────────┐          │ │
│  │  │ Queue_train  │        │ Queue_infer  │          │ │
│  │  │ priority=1   │        │ priority=10  │          │ │
│  │  │              │        │              │          │ │
│  │  │ Ring Buffer  │        │ Ring Buffer  │          │ │
│  │  │ rptr wptr    │        │ rptr wptr    │          │ │
│  │  └──────┬───────┘        └──────┬───────┘          │ │
│  │         │                       │                  │ │
│  │         └───────────┬───────────┘                  │ │
│  │                     ↓                              │ │
│  │         选择队列（可能选错！）                      │ │
│  │                     ↓                              │ │
│  │         ┌─────────────────────┐                   │ │
│  │         │ Compute Units (CUs) │                   │ │
│  │         │ 执行 Wavefronts     │                   │ │
│  │         └─────────────────────┘                   │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
            ↑
            │ CWSR 抢占（由 GPREEMPT 触发）
            │
┌─────────────────────────────────────────────────────────────┐
│  内核态：GPREEMPT 监控层                                     │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ 监控线程（定期 1-10ms）                                │ │
│  │                                                       │ │
│  │  while (true) {                                      │ │
│  │    // 读取硬件寄存器（MMIO read）                     │ │
│  │    rptr_train = read_hw(Queue_train.rptr);          │ │
│  │    wptr_train = read_hw(Queue_train.wptr);          │ │
│  │    rptr_infer = read_hw(Queue_infer.rptr);          │ │
│  │    wptr_infer = read_hw(Queue_infer.wptr);          │ │
│  │                                                       │ │
│  │    // 检测优先级倒置                                  │ │
│  │    if (Queue_infer 有任务 && Queue_train 在运行 &&   │ │
│  │        Queue_infer.priority > Queue_train.priority) {│ │
│  │      // 触发 CWSR 抢占                               │ │
│  │      cwsr_preempt(Queue_train);                      │ │
│  │    }                                                 │ │
│  │                                                       │ │
│  │    sleep(check_interval_ms);                         │ │
│  │  }                                                   │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

关键点:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 每个应用都有自己的队列和 doorbell
2. 两个队列都通过 doorbell 快速提交（~100ns）
3. GPU 硬件可能选错队列（优先级支持有限）
4. GPREEMPT 监控检测到错误，触发 CWSR 纠正
5. Doorbell 性能完全保留 ✅
```

---

## 📊 性能分析

### 性能指标对比

| 指标 | 无调度 | 拦截方案（错误）| GPREEMPT方案（正确）|
|------|--------|----------------|---------------------|
| **任务提交延迟** | ~100ns | 10-100μs+ | **~100ns** ✅ |
| **Doorbell性能** | 原生 | ❌ 被bypass | **原生** ✅ |
| **抢占延迟** | N/A | N/A | **1-10μs** ✅ |
| **调度延迟** | N/A | <1ms | **1-10ms** |
| **应用透明性** | N/A | ❌ 需要LD_PRELOAD | **✅ 完全透明** |
| **优先级保证** | ❌ 无 | ✅ 有 | **✅ 有** |
| **性能开销** | 0% | 5-10% | **<2%** ✅ |

### 延迟构成分析

```
推理任务端到端延迟（高优先级）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

无优先级调度:
  提交延迟: ~100ns
  等待训练任务完成: 可能数百毫秒到数秒 ❌
  执行时间: 20ms
  ────────────────────────────────────
  总延迟: 数百毫秒+ （不可接受）

拦截方案（错误）:
  提交延迟: 10-100μs（拦截开销）❌
  调度决策: <1ms
  执行时间: 20ms
  ────────────────────────────────────
  总延迟: ~21ms（但损失 doorbell 性能）

GPREEMPT 方案（正确）:
  提交延迟: ~100ns ✅
  监控检测: 1-10ms（可配置）
  抢占延迟: 1-10μs ✅
  执行时间: 20ms
  ────────────────────────────────────
  总延迟: ~25-30ms（保留 doorbell 性能）✅
```

---

## 🛠️ 实施计划

### Phase 1: GPREEMPT 内核实现（2-3个月）

**目标**: 实现 Paper#1 的内核态调度器

#### 1.1 数据结构扩展

```c
// 文件: amd/amdkfd/kfd_priv.h

struct queue {
    // 现有字段...
    
    // 新增：优先级调度字段
    int priority;                    // 队列优先级（0-15，越大越高）
    enum queue_domain domain;        // USER_DOMAIN, SYSTEM_DOMAIN
    
    // 新增：状态监控字段
    uint64_t rptr_cached;            // 缓存的 rptr
    uint64_t wptr_cached;            // 缓存的 wptr
    uint64_t pending_work;           // wptr - rptr
    bool is_active;                  // 是否有待处理任务
    
    // 新增：统计字段
    uint64_t total_preemptions;      // 被抢占次数
    uint64_t total_resumes;          // 恢复次数
    ktime_t last_preempt_time;       // 上次抢占时间
};

// 调度器配置
struct kfd_priority_scheduler {
    struct task_struct *monitor_thread;  // 监控线程
    unsigned int check_interval_ms;      // 检查间隔（默认 5ms）
    bool enabled;                        // 是否启用
    
    // 统计
    atomic64_t total_checks;             // 总检查次数
    atomic64_t total_inversions;         // 检测到的优先级倒置次数
    atomic64_t total_preemptions;        // 总抢占次数
};
```

#### 1.2 监控线程实现

```c
// 文件: amd/amdkfd/kfd_priority_scheduler.c

int kfd_priority_scheduler_thread(void *data) {
    struct amdgpu_device *adev = data;
    struct kfd_priority_scheduler *sched = &adev->kfd->sched;
    
    pr_info("GPREEMPT: Priority scheduler thread started\n");
    
    while (!kthread_should_stop()) {
        // 1. 扫描所有队列，更新状态
        update_all_queue_states(adev);
        
        // 2. 检测优先级倒置
        struct queue *high_q = NULL, *low_q = NULL;
        if (detect_priority_inversion(adev, &high_q, &low_q)) {
            pr_info("GPREEMPT: Priority inversion detected: "
                    "Q%d (prio %d) waiting, Q%d (prio %d) running\n",
                    high_q->properties.queue_id, high_q->priority,
                    low_q->properties.queue_id, low_q->priority);
            
            // 3. 触发抢占
            kfd_queue_preempt_single(low_q, 
                                     KFD_PREEMPT_TYPE_WAVEFRONT_SAVE, 
                                     1000);
            
            atomic64_inc(&sched->total_preemptions);
        }
        
        atomic64_inc(&sched->total_checks);
        
        // 4. 休眠
        msleep_interruptible(sched->check_interval_ms);
    }
    
    return 0;
}

// 更新队列状态
void update_all_queue_states(struct amdgpu_device *adev) {
    struct queue *q;
    
    list_for_each_entry(q, &adev->kfd->queue_list, list) {
        // 从硬件读取 rptr/wptr
        q->rptr_cached = read_queue_rptr(q);
        q->wptr_cached = read_queue_wptr(q);
        q->pending_work = q->wptr_cached - q->rptr_cached;
        q->is_active = (q->pending_work > 0);
    }
}

// 优先级倒置检测
bool detect_priority_inversion(struct amdgpu_device *adev,
                               struct queue **high_q_out,
                               struct queue **low_q_out) {
    struct queue *highest_waiting = NULL;
    struct queue *lowest_running = NULL;
    
    // 找到最高优先级的等待队列
    list_for_each_entry(q, &adev->kfd->queue_list, list) {
        if (q->pending_work > 0 && !queue_is_running(q)) {
            if (!highest_waiting || 
                q->priority > highest_waiting->priority) {
                highest_waiting = q;
            }
        }
    }
    
    // 找到最低优先级的运行队列
    list_for_each_entry(q, &adev->kfd->queue_list, list) {
        if (queue_is_running(q)) {
            if (!lowest_running || 
                q->priority < lowest_running->priority) {
                lowest_running = q;
            }
        }
    }
    
    // 检测倒置
    if (highest_waiting && lowest_running &&
        highest_waiting->priority > lowest_running->priority) {
        *high_q_out = highest_waiting;
        *low_q_out = lowest_running;
        return true;
    }
    
    return false;
}
```

#### 1.3 ioctl 接口扩展

```c
// 文件: amd/amdkfd/kfd_chardev.c

// 设置队列优先级
static int kfd_ioctl_set_queue_priority(struct file *filep,
                                        struct kfd_process *p,
                                        void *data) {
    struct kfd_ioctl_set_queue_priority_args *args = data;
    struct queue *q;
    
    q = pqm_get_user_queue(&p->pqm, args->queue_id);
    if (!q)
        return -EINVAL;
    
    // 设置优先级
    q->priority = args->priority;
    
    pr_info("GPREEMPT: Set Q%d priority to %d\n", 
            args->queue_id, args->priority);
    
    return 0;
}

// 获取调度统计
static int kfd_ioctl_get_sched_stats(struct file *filep,
                                     struct kfd_process *p,
                                     void *data) {
    struct kfd_ioctl_get_sched_stats_args *args = data;
    struct amdgpu_device *adev = p->kgd->dev;
    struct kfd_priority_scheduler *sched = &adev->kfd->sched;
    
    args->total_checks = atomic64_read(&sched->total_checks);
    args->total_inversions = atomic64_read(&sched->total_inversions);
    args->total_preemptions = atomic64_read(&sched->total_preemptions);
    args->check_interval_ms = sched->check_interval_ms;
    
    return 0;
}

// 注册 ioctl
AMDKFD_IOCTL_DEF(AMDKFD_IOC_SET_QUEUE_PRIORITY,
                 kfd_ioctl_set_queue_priority, 0),
AMDKFD_IOCTL_DEF(AMDKFD_IOC_GET_SCHED_STATS,
                 kfd_ioctl_get_sched_stats, 0),
```

#### 1.4 sysfs 配置接口

```c
// 文件: amd/amdkfd/kfd_device.c

// /sys/module/amdgpu/parameters/sched_interval_ms
static int sched_interval_ms = 5;
module_param(sched_interval_ms, int, 0644);
MODULE_PARM_DESC(sched_interval_ms, 
                 "Priority scheduler check interval in milliseconds (default: 5)");

// /sys/module/amdgpu/parameters/sched_enable
static bool sched_enable = true;
module_param(sched_enable, bool, 0644);
MODULE_PARM_DESC(sched_enable, 
                 "Enable priority scheduler (default: true)");
```

### Phase 2: XSched 集成（1-2个月）

**目标**: 将 GPREEMPT 作为 XSched Lv3 实现

#### 2.1 创建 GPREEMPT Backend

```cpp
// 文件: xsched/platforms/hip/hal/src/gpreempt_backend.cpp

#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/kfd_ioctl.h>

class GpreemptBackend : public HardwareAbstractionLayer {
public:
    GpreemptBackend() : kfd_fd_(-1) {
        kfd_fd_ = open("/dev/kfd", O_RDWR);
        if (kfd_fd_ < 0) {
            throw std::runtime_error("Failed to open /dev/kfd");
        }
    }
    
    ~GpreemptBackend() {
        if (kfd_fd_ >= 0) {
            close(kfd_fd_);
        }
    }
    
    // 实现 Lv3 接口
    PreemptLevel get_preempt_level() override {
        return PreemptLevel::Lv3;
    }
    
    int interrupt(uint32_t queue_id) override {
        struct kfd_ioctl_preempt_queue_args args = {
            .queue_id = queue_id,
            .preempt_type = 2,  // WAVEFRONT_SAVE (CWSR)
            .timeout_ms = 1000
        };
        
        int ret = ioctl(kfd_fd_, AMDKFD_IOC_PREEMPT_QUEUE, &args);
        if (ret < 0) {
            fprintf(stderr, "GPREEMPT: Failed to preempt queue %d: %s\n",
                    queue_id, strerror(errno));
        }
        return ret;
    }
    
    int restore(uint32_t queue_id) override {
        struct kfd_ioctl_resume_queue_args args = {
            .queue_id = queue_id
        };
        
        int ret = ioctl(kfd_fd_, AMDKFD_IOC_RESUME_QUEUE, &args);
        if (ret < 0) {
            fprintf(stderr, "GPREEMPT: Failed to resume queue %d: %s\n",
                    queue_id, strerror(errno));
        }
        return ret;
    }
    
    int set_priority(uint32_t queue_id, int priority) {
        struct kfd_ioctl_set_queue_priority_args args = {
            .queue_id = queue_id,
            .priority = priority
        };
        
        return ioctl(kfd_fd_, AMDKFD_IOC_SET_QUEUE_PRIORITY, &args);
    }
    
private:
    int kfd_fd_;
};
```

#### 2.2 集成到 XQueue

```cpp
// 文件: xsched/core/src/xqueue.cpp

XQueue* XQueueCreate(hipStream_t stream, int priority) {
    XQueue* xq = new XQueue(stream, priority);
    
    // 检测硬件支持
    if (detect_gpreempt_support()) {
        // 使用 GPREEMPT backend
        auto backend = std::make_shared<GpreemptBackend>();
        xq->set_backend(backend);
        xq->set_preempt_level(PreemptLevel::Lv3);
        
        // 获取队列 ID 并设置优先级
        uint32_t queue_id = get_queue_id_from_stream(stream);
        backend->set_priority(queue_id, priority);
        
        printf("✅ Using GPREEMPT Lv3 for queue %d (priority=%d)\n",
               queue_id, priority);
    } else {
        // 回退到 Lv1
        xq->set_preempt_level(PreemptLevel::Lv1);
        printf("⚠️ GPREEMPT not available, using Lv1\n");
    }
    
    return xq;
}
```

### Phase 3-4: 高级功能和生产化（3-4个月）

（详见前面的实施计划）

---

## ✅ 验证和测试

### 功能测试

```bash
# 1. 验证 GPREEMPT 工作
sudo dmesg | grep GPREEMPT

# 2. 检查 sysfs 配置
cat /sys/module/amdgpu/parameters/sched_enable
cat /sys/module/amdgpu/parameters/sched_interval_ms

# 3. 运行 2 进程测试
./test_priority_scheduling

# 4. 运行 XSched Example 3
cd xsched/examples/Linux/3_intra_process_sched
./app_concurrent
```

### 性能测试

```python
# benchmark_priority.py

import time
import subprocess

def test_latency_ratio():
    """测试高/低优先级延迟比"""
    
    # 启动低优先级训练任务
    train_proc = subprocess.Popen(["./train", "--priority=1"])
    time.sleep(1)
    
    # 运行高优先级推理任务
    start = time.time()
    infer_proc = subprocess.run(["./infer", "--priority=10"])
    high_latency = time.time() - start
    
    # 运行低优先级推理任务
    start = time.time()
    infer_proc = subprocess.run(["./infer", "--priority=1"])
    low_latency = time.time() - start
    
    ratio = low_latency / high_latency
    print(f"Latency ratio: {ratio:.2f}x")
    
    # 期望: 3-4倍差异
    assert ratio >= 3.0, f"Latency ratio too low: {ratio}"

if __name__ == "__main__":
    test_latency_ratio()
```

---

## 🎯 总结

### 核心要点

1. ⭐⭐⭐⭐⭐ **保留 Doorbell 是关键**
   - 不能为了调度而牺牲性能
   - Doorbell 提供 ~100ns 的快速提交
   - 内核态监控"事后"检测和抢占

2. ⭐⭐⭐⭐⭐ **GPREEMPT 完美适配 XSched Lv3**
   - XSched Lv3 需要: interrupt() / restore()
   - GPREEMPT 提供: PREEMPT_QUEUE / RESUME_QUEUE ioctl
   - 接口完全匹配

3. ⭐⭐⭐⭐⭐ **CWSR 提供硬件基础**
   - 1-10μs 快速抢占
   - 完整 Wave 状态保存
   - AMD MI308X 原生支持

4. ⭐⭐⭐⭐ **完全透明，无需修改应用**
   - 应用继续使用原生 HIP API
   - 无需 LD_PRELOAD
   - 无需重新编译

### 技术优势

```
融合方案 vs 其他方案:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

               | 无调度 | 拦截方案 | GPREEMPT方案 |
               |--------|----------|--------------|
Doorbell性能   | ✅     | ❌       | ✅           |
优先级保证     | ❌     | ✅       | ✅           |
应用透明性     | ✅     | ❌       | ✅           |
抢占能力       | ❌     | ❌       | ✅           |
性能开销       | 0%     | 5-10%    | <2%          |
实现复杂度     | 低     | 中       | 高           |

结论: GPREEMPT 方案是最佳选择！
```

### 实施路线图

```
时间线（7-9个月）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

M0 (现在):        Phase 1.5 完成，基础 ioctl 工作
                  ↓
M1-M3 (2-3个月):  实现 GPREEMPT 内核调度器
                  • 监控线程
                  • 优先级倒置检测
                  • CWSR 抢占集成
                  ↓
M4-M5 (1-2个月):  集成到 XSched
                  • GPREEMPT backend
                  • Lv3 接口实现
                  • Example 3 测试
                  ↓
M6-M7 (2个月):    高级功能
                  • 多租户支持
                  • 高级调度策略
                  • 性能优化
                  ↓
M8-M9 (2个月):    生产化
                  • 测试和文档
                  • 上游贡献
                  • 论文发表

M9: 生产就绪！
```

---

**文档版本**: v1.0  
**创建日期**: 2026-01-27  
**作者**: AI Assistant  
**状态**: 技术方案定稿

**关键联系人**: zhehan  
**相关代码**: 
- GPREEMPT: `/mnt/md0/zhehan/code/coderampup/private_github/amdgpu_DKMS/`
- XSched: `/workspace/xsched/`

**下一步**: 开始实施 Phase 1 - GPREEMPT 内核调度器

