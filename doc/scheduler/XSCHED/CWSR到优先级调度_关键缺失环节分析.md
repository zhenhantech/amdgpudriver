# CWSR 到优先级调度：关键缺失环节分析

**日期**: 2026-01-27  
**问题**: 有了 CWSR，还需要什么才能实现完整的优先级调度？  
**核心挑战**: User Space Doorbell 直接提交绕过调度器

---

## 🎯 问题概述

您的观察非常准确！**CWSR 只是硬件抢占能力，并不等于完整的优先级调度系统。**

```
误区:
CWSR ≠ 优先级调度系统 ❌

实际:
优先级调度 = 调度决策 + 队列管理 + 抢占机制（CWSR）✅
             ^^^^^^^^   ^^^^^^^^   ^^^^^^^^
             需要添加    需要改进    已经有了
```

---

## 🔍 当前状态分析

### ✅ 已有的能力：CWSR (Hardware Preemption)

```
CWSR提供:
┌────────────────────────────────────────┐
│ ✅ Wave级别的状态保存/恢复              │
│ ✅ 微秒级抢占延迟（1-10μs）            │
│ ✅ KFD驱动接口已实现                   │
│    - checkpoint_mqd()                 │
│    - restore_mqd()                    │
│    - destroy_mqd(WAVEFRONT_SAVE)      │
└────────────────────────────────────────┘

CWSR回答的问题: "如何抢占？"
```

### ❌ 缺失的能力：调度决策与队列管理

```
需要但没有:
┌────────────────────────────────────────┐
│ ❌ 何时抢占？（调度策略）               │
│ ❌ 抢占谁？（优先级判断）               │
│ ❌ 如何拦截任务提交？（User Doorbell）  │  ← ⭐ 核心问题
│ ❌ 如何管理多个优先级队列？             │
│ ❌ 如何处理公平性？                    │
└────────────────────────────────────────┘

这些回答的问题: "什么时候、为什么、由谁来抢占？"
```

---

## 🚪 关键问题：User Space Doorbell

### 什么是 Doorbell？

**Doorbell = 用户空间直接通知 GPU 的快速通道**

```
传统路径（慢，但可控）:
┌────────────────────────────────────────┐
│ 用户应用                                │
│   ↓ ioctl() [系统调用]                 │
│ 内核驱动                                │  ← 驱动可以拦截和调度
│   ↓ 写队列                             │
│ GPU 硬件                                │
└────────────────────────────────────────┘
延迟: ~1-10μs（系统调用开销）

Doorbell 快速路径（快，但难以拦截）:
┌────────────────────────────────────────┐
│ 用户应用                                │
│   ↓ 写入 mmap 的内存区域（Doorbell）    │  ← ⭐ 绕过内核！
│ GPU 硬件（直接感知）                     │
└────────────────────────────────────────┘
延迟: ~0.1-1μs（直接MMIO写）

问题: 调度器看不到这些提交！
```

### Doorbell 的工作原理

```c
// 用户空间代码（在 HIP Runtime 中）
void submit_kernel_via_doorbell(hipStream_t stream, kernel_info* k) {
    // 1. 将 kernel 参数写入 GPU 队列（ring buffer）
    ring_buffer* rb = stream->queue->ring;
    write_packet_to_ring(rb, k);
    
    // 2. 通过 Doorbell 通知 GPU（关键：不经过内核驱动）
    volatile uint64_t* doorbell = stream->queue->doorbell_ptr;  
    //                             ↑ 这是 mmap 到用户空间的 MMIO 地址
    
    *doorbell = ring_buffer_wptr;  // ← 直接写 MMIO，GPU 立即感知
    //          ^^^^^^^^^^^^^^^^
    //          硬件 DMA 引擎读取这个地址，立即开始处理队列
}
```

**为什么用 Doorbell？**
- 性能！避免系统调用开销（~10x faster）
- 延迟敏感的 HPC/AI 应用需要这种性能

**为什么是问题？**
- **调度器看不到提交**：任务已经在 GPU 上运行了，调度器才知道
- **无法实施优先级**：高优先级任务无法在提交时就抢占低优先级任务

### 实际证据

来自您的 `2PROC_DKMS_debug` 文档：

```
问题: 为什么没有看到SDMA queue创建的日志？
答案: "用户空间直接使用已存在的SDMA ring（驱动级别），
       不需要通过KFD创建新的SDMA queue"

→ 说明: SDMA提交通过 Doorbell，绕过了 kfd_ioctl_create_queue
```

---

## 🛠️ 解决方案对比

### 方案 1：拦截 Doorbell 写入 ⚠️ （最难）

#### 思路
**在硬件/驱动层拦截 Doorbell MMIO 写入**

```
┌────────────────────────────────────────┐
│ 用户应用                                │
│   ↓ *doorbell = wptr                   │
├────────────────────────────────────────┤
│ 🔍 IOMMU / Page Fault Handler          │  ← 拦截 MMIO 写入
│   ↓ 调度器介入                         │
│   ↓ 优先级判断                         │
│   ↓ 如果需要抢占 → 触发 CWSR           │
├────────────────────────────────────────┤
│ GPU 硬件                                │
└────────────────────────────────────────┘
```

#### 实现方式

**A. 基于 IOMMU 的拦截**
```c
// 概念代码
// 1. 将 Doorbell 页面设置为不可写
void setup_doorbell_intercept(struct kfd_queue *q) {
    // 修改 Doorbell 页面的页表属性
    struct vm_area_struct *vma = q->doorbell_vma;
    
    // 设置为只读，触发 page fault
    vma->vm_page_prot = vm_get_page_prot(VM_READ);
    
    // 注册 page fault handler
    vma->vm_ops->fault = doorbell_page_fault_handler;
}

// 2. Page Fault Handler
vm_fault_t doorbell_page_fault_handler(struct vm_fault *vmf) {
    struct kfd_queue *q = get_queue_from_vma(vmf->vma);
    
    // 🎯 在这里实施调度决策
    if (should_preempt_for_priority(q)) {
        // 抢占低优先级队列
        preempt_low_priority_queues();
    }
    
    // 允许写入继续
    return VM_FAULT_NOPAGE | VM_FAULT_RETRY;
}
```

**优点**：
- ✅ 完全可控，所有提交都能拦截
- ✅ 可以在提交时就做调度决策

**缺点**：
- ❌ **巨大的性能开销**：每次 Doorbell 写入都触发 page fault（~1-2μs）
- ❌ 抵消了 Doorbell 的性能优势（10x slower）
- ❌ 实现复杂度极高（涉及 IOMMU、页表管理）
- ❌ 可能与现有硬件机制冲突

**可行性**: ⭐⭐ 理论可行，但实际不推荐

---

### 方案 2：异步监控 + 事后抢占 ⭐⭐⭐⭐ （推荐）

#### 思路
**不拦截提交，但异步监控队列状态，按需抢占**

```
┌────────────────────────────────────────┐
│ 用户应用                                │
│   ↓ *doorbell = wptr                   │
│ GPU 硬件（立即开始执行）                 │  ← 不阻塞，照常执行
└────────────────────────────────────────┘
                ↓
┌────────────────────────────────────────┐
│ 🔍 调度器 Monitoring Thread             │  ← 异步监控
│   - 定期检查所有队列状态                │
│   - 读取 Queue Read/Write Pointer      │
│   - 计算每个队列的负载                  │
│   - 优先级比较                         │
│                                        │
│   IF (高优先级队列有任务 &&             │
│       低优先级队列正在运行)             │
│   THEN:                                │
│     触发 CWSR 抢占低优先级队列          │  ← 使用已有的 CWSR
│     让出资源给高优先级                  │
└────────────────────────────────────────┘
```

#### 实现方式

```c
// 调度器监控线程
int scheduler_monitor_thread(void *data) {
    struct amdgpu_device *adev = data;
    struct priority_scheduler *sched = &adev->priority_sched;
    
    while (!kthread_should_stop()) {
        // 1. 检查所有队列的状态
        for_each_queue(q, &adev->kfd->queue_list) {
            // 读取队列的读写指针（硬件寄存器，无开销）
            u64 rptr = read_queue_rptr(q);  // GPU已处理到哪里
            u64 wptr = read_queue_wptr(q);  // 应用提交到哪里
            
            q->pending_work = wptr - rptr;  // 队列中待处理的工作
            q->is_active = (q->pending_work > 0);
        }
        
        // 2. 优先级调度决策
        struct kfd_queue *high_prio_q = find_highest_priority_queue_with_work(sched);
        struct kfd_queue *low_prio_q = find_running_low_priority_queue(sched);
        
        if (high_prio_q && low_prio_q && 
            high_prio_q->priority > low_prio_q->priority) {
            
            // 3. 触发抢占（使用现有的 CWSR）
            kfd_queue_preempt_single(low_prio_q, 
                                     KFD_PREEMPT_TYPE_WAVEFRONT_SAVE,
                                     1000 /* timeout */);
            
            // 4. 唤醒高优先级队列（如果被挂起）
            kfd_queue_resume_single(high_prio_q);
            
            pr_info("[Sched] Preempted Q%d (prio %d) for Q%d (prio %d)\n",
                    low_prio_q->properties.queue_id, low_prio_q->priority,
                    high_prio_q->properties.queue_id, high_prio_q->priority);
        }
        
        // 5. 定期检查（如 1-10ms）
        msleep_interruptible(sched->check_interval_ms);
    }
    return 0;
}
```

#### 数据结构

```c
// 优先级调度器结构
struct priority_scheduler {
    // 多优先级队列列表
    struct list_head queues[MAX_PRIORITY_LEVELS];
    
    // 监控线程
    struct task_struct *monitor_thread;
    unsigned int check_interval_ms;  // 监控间隔（1-10ms）
    
    // 统计信息
    atomic64_t preemption_count;
    atomic64_t scheduling_decisions;
    
    // 配置
    bool enabled;
    enum scheduling_policy policy;  // STRICT, FAIR, WEIGHTED
};

// 队列扩展信息
struct kfd_queue_extended {
    struct kfd_queue base;
    
    // 优先级相关
    unsigned int priority;          // 0=最低, 255=最高
    enum queue_type type;           // SYSTEM, REALTIME, NORMAL, BATCH
    
    // 运行时状态
    u64 pending_work;              // wptr - rptr
    bool is_active;
    ktime_t last_submit_time;
    
    // 统计
    u64 total_submissions;
    u64 total_preemptions;
    u64 total_exec_time_ns;
};
```

#### 优点

✅ **无性能开销**（提交路径）
- Doorbell 照常工作，无额外延迟
- 用户空间完全无感知

✅ **抢占延迟可控**
- 监控间隔可调：1-10ms
- 对于大部分应用足够快（AI 训练的 iteration 通常 >100ms）

✅ **实现简单**
- 复用现有的 CWSR 机制
- 只需要添加监控线程和调度逻辑
- 代码量：~500-1000 行

✅ **灵活的策略**
- 可以实现多种调度策略（strict priority, fair, weighted）
- 可以根据负载动态调整

#### 缺点

⚠️ **抢占不是立即的**
- 从高优先级任务提交到抢占生效，延迟 = 监控间隔（1-10ms）
- 对于延迟敏感的实时任务可能不够

⚠️ **需要周期性 CPU 开销**
- 监控线程定期唤醒（但开销很小，<1% CPU）

#### 可行性

⭐⭐⭐⭐⭐ **强烈推荐**

**为什么？**
1. **平衡性好**：性能开销小，功能完整
2. **实现简单**：复用现有 CWSR，增量开发
3. **适用性广**：满足大部分深度学习、HPC 场景的需求
4. **可演进**：未来可以优化（如中断驱动，见方案3）

---

### 方案 3：硬件中断通知 ⭐⭐⭐⭐⭐ （最优，需要硬件支持）

#### 思路
**GPU 在队列状态变化时主动通知 CPU（中断）**

```
┌────────────────────────────────────────┐
│ 用户应用                                │
│   ↓ *doorbell = wptr                   │
├────────────────────────────────────────┤
│ GPU 硬件                                │
│   ↓ 感知 Doorbell 写入                 │
│   ↓ 如果队列优先级高                    │
│   ↓ 触发 CPU 中断 🔔                   │  ← 硬件通知
├────────────────────────────────────────┤
│ CPU: 中断处理函数                        │
│   ↓ 调度器立即介入                      │
│   ↓ 判断是否需要抢占                    │
│   ↓ 触发 CWSR                          │
└────────────────────────────────────────┘
```

#### 实现方式

```c
// GPU 硬件配置（需要硬件支持）
void setup_priority_interrupt(struct amdgpu_device *adev) {
    // 配置 GPU 在高优先级队列有提交时产生中断
    writel(PRIORITY_QUEUE_INTERRUPT_ENABLE, 
           adev->mmio + GPU_INTERRUPT_CONTROL);
    
    // 注册中断处理函数
    request_irq(adev->pdev->irq, priority_queue_interrupt_handler, 
                IRQF_SHARED, "amdgpu_priority", adev);
}

// 中断处理函数
irqreturn_t priority_queue_interrupt_handler(int irq, void *arg) {
    struct amdgpu_device *adev = arg;
    
    // 1. 读取中断状态寄存器
    u32 int_status = readl(adev->mmio + GPU_INTERRUPT_STATUS);
    
    if (int_status & PRIORITY_QUEUE_SUBMIT) {
        // 2. 获取触发中断的队列
        struct kfd_queue *high_q = get_queue_from_interrupt(adev, int_status);
        
        // 3. 立即调度决策（在中断上下文，必须快）
        struct kfd_queue *low_q = find_preemptible_queue(adev, high_q->priority);
        
        if (low_q) {
            // 4. 触发抢占
            amdgpu_preempt_queue_immediate(low_q);
        }
        
        return IRQ_HANDLED;
    }
    
    return IRQ_NONE;
}
```

#### 优点

✅ **抢占延迟最小**
- <1μs 响应延迟（中断延迟）
- 适合实时任务

✅ **无周期性 CPU 开销**
- 事件驱动，只在需要时唤醒

✅ **精确性高**
- 每次提交都能精确感知

#### 缺点

❌ **需要硬件支持**
- MI308X 可能不支持（需要查硬件文档）
- 需要 GPU firmware 配合

❌ **实现复杂**
- 涉及中断管理、硬件配置
- 需要与 AMD GPU 团队合作

#### 可行性

⭐⭐⭐⭐⭐ **理想方案，但需要硬件支持**

**评估步骤**：
1. 查阅 MI308X 硬件文档（TRM, Programming Guide）
2. 检查是否支持 "Queue Submit Interrupt" 或类似机制
3. 如果支持 → 强烈推荐
4. 如果不支持 → 回退到方案 2

---

## 📊 方案对比总结

| 方案 | 抢占延迟 | CPU开销 | 实现复杂度 | 硬件要求 | 推荐度 | 适用场景 |
|-----|---------|---------|-----------|---------|-------|---------|
| **1. 拦截Doorbell** | <1μs | 极高(每次提交) | 极高 | ❌ | ⭐⭐ | ❌ 不推荐 |
| **2. 异步监控** | 1-10ms | 极低(周期性) | 低 | ✅ 当前硬件 | ⭐⭐⭐⭐⭐ | ✅ 深度学习/HPC |
| **3. 硬件中断** | <1μs | 极低(事件驱动) | 中 | ⚠️ 看硬件 | ⭐⭐⭐⭐⭐ | ✅ 实时任务 |

**结论**: 
- **短期（3-6个月）**: 实施方案2（异步监控）
- **长期（1年+）**: 评估硬件能力，尝试方案3（中断）

---

## 🛠️ 除了 Doorbell，还需要什么？

### 1. 多优先级队列管理 ⚠️ **需要添加**

**当前状态**：KFD 的队列是平等的，没有优先级概念

**需要添加**：
```c
// 扩展 kfd_queue 结构
struct kfd_queue {
    // 现有字段...
    
    // 新增：优先级相关 ⭐
    unsigned int priority;        // 0-255
    enum queue_domain {
        QUEUE_DOMAIN_SYSTEM,      // 系统库（HIPBLAS等）
        QUEUE_DOMAIN_APPLICATION  // 用户应用
    } domain;
    
    // 新增：调度状态
    enum queue_sched_state {
        QUEUE_ACTIVE,             // 正在运行
        QUEUE_PREEMPTED,          // 被抢占
        QUEUE_SUSPENDED           // 挂起
    } sched_state;
};

// 新增：优先级队列管理器
struct kfd_priority_scheduler {
    struct list_head queues[256];  // 每个优先级一个链表
    spinlock_t lock;
    
    // 调度策略
    enum sched_policy {
        SCHED_STRICT_PRIORITY,
        SCHED_WEIGHTED_FAIR,
        SCHED_TIME_SLICE
    } policy;
};
```

### 2. 队列状态监控接口 ⚠️ **需要添加**

**需要添加读取队列状态的高效接口**：

```c
// 读取队列的读写指针（无开销，直接读硬件寄存器）
static inline u64 kfd_queue_read_rptr(struct kfd_queue *q) {
    return *q->device->rptr_mmio + q->doorbell_off;
}

static inline u64 kfd_queue_read_wptr(struct kfd_queue *q) {
    return *q->device->wptr_mmio + q->doorbell_off;
}

// 计算队列负载
static inline bool kfd_queue_has_pending_work(struct kfd_queue *q) {
    u64 rptr = kfd_queue_read_rptr(q);
    u64 wptr = kfd_queue_read_wptr(q);
    return (wptr != rptr);  // 有未处理的工作
}
```

### 3. HIP Runtime 层的域标记 ⚠️ **需要添加**

**问题**：调度器怎么知道哪个队列是系统库，哪个是用户应用？

**方案**：HIP Runtime 在创建队列时标记域

```cpp
// 在 HIP Runtime (libamdhip64.so) 中
hipError_t hipStreamCreateWithPriority(hipStream_t* stream, 
                                        unsigned int flags,
                                        int priority) {
    // 创建 KFD 队列时传递优先级和域信息
    struct kfd_ioctl_create_queue_args args = {
        .queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE,
        .queue_priority = priority,        // ⭐ 新增
        .queue_domain = detect_domain(),   // ⭐ 新增：自动检测域
    };
    
    ioctl(kfd_fd, AMDKFD_IOC_CREATE_QUEUE, &args);
}

// 域检测（类似白名单机制）
static enum queue_domain detect_domain() {
    Dl_info info;
    if (dladdr(__builtin_return_address(1), &info)) {
        if (strstr(info.dli_fname, "libhipblas") ||
            strstr(info.dli_fname, "libMIOpen")) {
            return QUEUE_DOMAIN_SYSTEM;  // 系统库
        }
    }
    return QUEUE_DOMAIN_APPLICATION;     // 用户应用
}
```

### 4. 调度策略引擎 ⚠️ **需要添加**

**实现不同的调度策略**：

```c
// 调度决策引擎
struct kfd_queue* select_queue_to_preempt(
    struct kfd_priority_scheduler *sched,
    struct kfd_queue *new_queue) {
    
    switch (sched->policy) {
    case SCHED_STRICT_PRIORITY:
        // 严格优先级：抢占所有低优先级队列
        return find_lowest_priority_running_queue(sched, new_queue->priority);
        
    case SCHED_WEIGHTED_FAIR:
        // 加权公平：考虑运行时间和优先级
        return find_queue_by_weighted_fairness(sched, new_queue);
        
    case SCHED_TIME_SLICE:
        // 时间片：每个队列运行一定时间后切换
        return find_queue_exceeded_timeslice(sched);
        
    default:
        return NULL;
    }
}
```

### 5. 性能计数和统计 ✅ **可选但推荐**

**用于调试和优化**：

```c
// 调度器统计信息
struct kfd_sched_stats {
    atomic64_t total_preemptions;
    atomic64_t total_scheduling_decisions;
    atomic64_t preemption_latency_sum_ns;
    
    // 每个优先级的统计
    struct {
        atomic64_t submissions;
        atomic64_t exec_time_ns;
        atomic64_t wait_time_ns;
    } per_priority[256];
};

// 暴露给用户空间（sysfs）
// /sys/class/kfd/kfd/priority_scheduler/stats
```

---

## 🎯 完整的实施路径

### Phase 1: 基础设施（1-2个月）

```
任务清单:
├─ [1] 扩展 kfd_queue 结构（添加 priority, domain, sched_state）
├─ [2] 实现优先级队列管理器（kfd_priority_scheduler）
├─ [3] 添加队列状态读取接口（rptr/wptr）
├─ [4] 扩展 KFD ioctl 接口（支持优先级参数）
└─ [5] 测试基础功能（队列创建、状态读取）

输出: 
✓ KFD 驱动支持优先级队列
✓ 可以读取队列状态
```

### Phase 2: 调度器核心（2-3个月）

```
任务清单:
├─ [1] 实现调度监控线程
├─ [2] 实现调度决策引擎（Strict Priority策略）
├─ [3] 集成现有的 CWSR 抢占机制
├─ [4] 实现队列恢复逻辑
└─ [5] 单机测试（2个进程，不同优先级）

输出:
✓ 可以抢占低优先级队列
✓ 高优先级队列优先执行
```

### Phase 3: HIP Runtime 集成（1-2个月）

```
任务清单:
├─ [1] 修改 hipStreamCreateWithPriority API
├─ [2] 实现域检测逻辑（类似白名单）
├─ [3] 系统库默认高优先级
├─ [4] 用户应用默认普通优先级
└─ [5] 端到端测试（PyTorch + 优先级）

输出:
✓ PyTorch 可以使用优先级调度
✓ HIPBLAS/MIOpen 自动高优先级
```

### Phase 4: 优化与生产化（2-3个月）

```
任务清单:
├─ [1] 性能优化（减少监控开销）
├─ [2] 添加更多调度策略（公平性、时间片）
├─ [3] 添加统计和监控接口
├─ [4] 文档和示例代码
└─ [5] 生产环境测试

输出:
✓ 生产就绪的优先级调度系统
```

**总时间**: 6-10个月

---

## 📋 关键代码位置

### 需要修改的文件

```
ROCm 源码树:
├─ drivers/gpu/drm/amd/amdkfd/
│  ├─ kfd_priv.h                    ← 添加优先级结构体定义
│  ├─ kfd_device_queue_manager.c    ← 队列管理核心
│  ├─ kfd_priority_scheduler.c      ← ⭐ 新增文件：调度器实现
│  └─ kfd_chardev.c                 ← ioctl 接口扩展
│
└─ hip/
   └─ src/hip_stream.cpp             ← hipStreamCreateWithPriority
```

### 代码量估算

```
优先级调度器核心:          ~1500 行 C 代码
KFD 接口扩展:              ~500 行 C 代码
HIP Runtime 修改:          ~300 行 C++ 代码
测试代码:                  ~1000 行
总计:                      ~3300 行代码
```

**结论**: 这是一个中等规模的内核开发项目，技术上可行，工作量合理。

---

## 总结

### 回答您的问题

**Q: 有了 CWSR，还需要什么？**

**A: 需要以下关键组件：**

1. ✅ **CWSR**（已有）：提供抢占能力
2. ❌ **Doorbell 问题解决**（需要添加）：
   - **推荐方案**：异步监控 + 事后抢占
   - **不推荐**：拦截 Doorbell（性能损失太大）
3. ❌ **多优先级队列管理**（需要添加）
4. ❌ **调度决策引擎**（需要添加）
5. ❌ **HIP Runtime 集成**（需要添加）

### 技术可行性

✅ **完全可行！**

- CWSR 提供了硬件基础
- Doorbell 问题有现实的解决方案（异步监控）
- 实现复杂度合理（~3000行代码，6-10个月）
- 不需要新的硬件特性

### 下一步行动

**立即（本周）**：
1. 阅读 MI308X 硬件文档，确认是否支持队列提交中断
2. 如果支持 → 方案3（中断驱动）
3. 如果不支持 → 方案2（异步监控）

**短期（1个月）**：
1. 实现 Phase 1（基础设施）
2. 原型验证：简单的2进程优先级调度

**中期（3-6个月）**：
1. 完成 Phase 2-3
2. 与 XSched 集成测试

---

## 🔄 与 XSched (Paper#2) 的对比分析

### XSched 的核心架构

XSched 采用的是**用户态调度器 + LD_PRELOAD 拦截**的方案：

```
架构对比:

XSched (用户态方案):
┌────────────────────────────────────────┐
│ PyTorch Application                     │
│   ↓ HIP API (hipLaunchKernel, etc)    │
├────────────────────────────────────────┤
│ ① XShim (LD_PRELOAD)                   │  ← 拦截 HIP API
│   - 判断是否需要调度                    │
│   - 如果需要 → 提交到 XQueue            │
│   - 如果不需要 → 透传到原始 HIP         │
├────────────────────────────────────────┤
│ ② XQueue (优先级队列)                   │  ← 用户态队列管理
│   - 多个优先级队列                      │
│   - 调度决策（何时提交到GPU）           │
├────────────────────────────────────────┤
│ ③ XExecutor (执行器)                    │  ← 任务调度和执行
│   - 监控 GPU 负载                       │
│   - 按优先级提交任务                    │
│   - 处理抢占逻辑                        │
├────────────────────────────────────────┤
│ libamdhip64.so (原始 HIP)               │
│   ↓ Doorbell                           │
│ GPU Hardware                           │
└────────────────────────────────────────┘

关键特点:
✅ 完全在用户态实现
✅ 无需修改内核
✅ 通过拦截 HIP API 实现调度控制
❌ 无法拦截 Doorbell（直接提交）
❌ 依赖应用配合（需要 LD_PRELOAD）
```

### 我们的 CWSR 方案（内核态）

```
我们的方案（内核态）:
┌────────────────────────────────────────┐
│ PyTorch Application                     │
│   ↓ HIP API                            │
│ libamdhip64.so                         │
│   ↓ Doorbell（直接写 MMIO）             │  ← 绕过用户态拦截
├────────────────────────────────────────┤
│ GPU Hardware                           │
└────────────────────────────────────────┘
                ↓
┌────────────────────────────────────────┐
│ 🔍 KFD Priority Scheduler (内核态)      │  ← 异步监控
│   - 监控线程定期检查队列状态             │
│   - 读取 rptr/wptr                     │
│   - 优先级判断                         │
│   - 触发 CWSR 抢占                     │
└────────────────────────────────────────┘

关键特点:
✅ 内核态实现，更底层
✅ 不需要应用配合
✅ 可以监控所有队列（包括 Doorbell 提交）
✅ 使用硬件 CWSR 能力
❌ 抢占延迟略高（1-10ms vs 实时）
❌ 需要修改内核驱动
```

### 两种方案的互补性

```
关键洞察:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

XSched 解决了:  "任务提交前" 的调度
我们的方案解决:  "任务提交后" 的抢占

两者可以完美互补！
```

### 方案对比表

| 维度 | XSched (用户态) | CWSR (内核态) | 融合方案 |
|-----|----------------|---------------|---------|
| **实现位置** | 用户态 | 内核态 | 用户态 + 内核态 |
| **拦截点** | HIP API | 队列状态监控 | 双层拦截 |
| **Doorbell 问题** | ❌ 无法拦截 | ✅ 可以监控 | ✅ 解决 |
| **抢占能力** | ⚠️ 需要任务配合 | ✅ 硬件 CWSR | ✅ 强制抢占 |
| **调度延迟** | <1ms | 1-10ms | <1ms（预调度）+ 10ms（强制） |
| **应用透明性** | ⚠️ 需要 LD_PRELOAD | ✅ 完全透明 | ✅ 可选 |
| **实现复杂度** | 中 | 高 | 高 |
| **适用场景** | 配合的应用 | 所有应用 | 所有场景 |

---

## 🎯 正确的融合方案：保留 Doorbell + 内核态调度

### 关键约束

⚠️ **不能 bypass Doorbell！**

```
为什么 Doorbell 重要？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Doorbell 路径（快速）:
  应用 → HIP Runtime → *doorbell = wptr (MMIO write, ~100ns)
                     → GPU 立即感知，DMA 读取 ring buffer
                     
绕过 Doorbell（慢）:
  应用 → 用户态拦截 → ioctl() → KFD → 内核调度 → GPU
         ~~~~~~~~     ~~~~~~~   ~~~
         额外开销      系统调用   上下文切换
         
性能差异: 10-100倍！
         
结论: 必须保留 Doorbell 的快速提交路径！
```

### 4 个 Feature 的正确融合

```
Feature 分析:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Doorbell (硬件机制)
   作用: 快速任务提交
   位置: 用户态 → GPU (绕过内核)
   性能: ~100ns 延迟
   限制: 无法被内核拦截
   
2. CWSR (硬件能力)
   作用: Wave-level 上下文切换
   位置: GPU 硬件 + Trap Handler
   性能: 1-10μs 抢占延迟
   接口: KFD ioctl
   
3. GPREEMPT (内核驱动)
   作用: GPU 抢占机制
   位置: KFD 驱动
   基础: CWSR 硬件
   接口: PREEMPT_QUEUE / RESUME_QUEUE ioctl
   
4. XSched (调度框架)
   作用: 多级抢占调度框架
   位置: 用户态 + 硬件接口
   级别: Lv1 (软件), Lv2 (队列), Lv3 (硬件)
   需求: interrupt() / restore() 接口
```

### 正确的融合架构

**核心思想**：保留 Doorbell，使用 GPREEMPT 为 XSched Lv3 提供 interrupt/restore 能力

```
完整架构图:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌──────────────────────────────────────────────────────────────┐
│  用户态: XSched 调度框架                                      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ XQueue (优先级队列管理)                                 │  │
│  │  • 高优先级队列: XQueue_high                            │  │
│  │  • 低优先级队列: XQueue_low                             │  │
│  │  • 调度策略引擎                                         │  │
│  └────────────────────────────────────────────────────────┘  │
│                          ↓                                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ XSched Lv3 接口层 (基于 GPREEMPT)                       │  │
│  │  • interrupt(hwq) → ioctl(PREEMPT_QUEUE)               │  │
│  │  • restore(hwq)   → ioctl(RESUME_QUEUE)                │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                          ↓ ioctl (系统调用)
┌──────────────────────────────────────────────────────────────┐
│  内核态: GPREEMPT (KFD Driver)                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ 🔍 Priority Scheduler Thread (异步监控)                 │  │
│  │   • 定期扫描所有队列 (1-10ms)                           │  │
│  │   • 读取 rptr/wptr 检测活跃状态                         │  │
│  │   • 检测优先级倒置                                      │  │
│  │   • 触发抢占: preempt_queue(low_prio_q)                 │  │
│  └────────────────────────────────────────────────────────┘  │
│                          ↓                                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ GPREEMPT ioctl 处理                                     │  │
│  │  • PREEMPT_QUEUE: 触发 CWSR 抢占                        │  │
│  │  • RESUME_QUEUE: 恢复队列执行                           │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  硬件层: CWSR (GPU Hardware)                                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ • Trap Handler 执行                                     │  │
│  │ • 保存/恢复 Wave 状态 (PC, SGPRs, VGPRs, LDS)          │  │
│  │ • 1-10μs 抢占延迟                                       │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  应用层: PyTorch / HIP Applications                           │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ hipLaunchKernel()                                       │  │
│  │   ↓                                                     │  │
│  │ libamdhip64.so                                          │  │
│  │   ↓                                                     │  │
│  │ *doorbell = wptr  (MMIO write) ← ✅ 保留快速路径！      │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

**关键设计点**：

1. ✅ **保留 Doorbell**：应用继续使用 doorbell 快速提交，不拦截
2. ✅ **GPREEMPT 异步监控**：内核态监控线程检测优先级倒置
3. ✅ **XSched Lv3 映射**：XSched 的 interrupt/restore 由 GPREEMPT ioctl 实现
4. ✅ **CWSR 硬件支持**：提供 1-10μs 的快速抢占

### Doorbell 和 CWSR 的时序关系

```
关键问题：高优先级任务也通过 doorbell 提交，如何与 CWSR 配合？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

核心机制：
┌────────────────────────────────────────────────────────────┐
│ 1. 每个应用有独立的队列（Queue）                           │
│    • 训练任务 → Queue_train (priority=1)                   │
│    • 推理服务 → Queue_infer (priority=10)                  │
│                                                            │
│ 2. 两个队列都通过 doorbell 提交任务                        │
│    • *doorbell_train = wptr  (~100ns)                     │
│    • *doorbell_infer = wptr  (~100ns)                     │
│                                                            │
│ 3. GPU Command Processor 从多个队列调度任务                │
│    • 但硬件调度器优先级支持有限                            │
│    • 可能选择错误的队列                                    │
│                                                            │
│ 4. GPREEMPT 事后检测和纠正                                 │
│    • 监控线程定期检查（1-10ms）                            │
│    • 发现优先级倒置                                        │
│    • 使用 CWSR 强制抢占低优先级队列                        │
└────────────────────────────────────────────────────────────┘

时间线：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

T < 0:    低优先级任务运行中
          • Queue_train 通过 doorbell 提交 ✓
          • GPU 正在执行 Queue_train

T = 0:    高优先级任务到达
          • Queue_infer 通过 doorbell 提交 ✓ (~100ns)
          • GPU 感知到 Queue_infer 有任务
          • 但 GPU 继续执行 Queue_train ❌
          • Queue_infer 在队列中等待 ⏳

T = 5ms:  GPREEMPT 检测到优先级倒置
          • 读取硬件寄存器：
            - Queue_train: rptr=100, wptr=500 (运行中)
            - Queue_infer: rptr=0, wptr=1 (等待中)
          • 发现倒置：high_prio 等待，low_prio 运行
          • 触发抢占：kfd_queue_preempt(Queue_train, CWSR)

T = 5ms:  CWSR 执行抢占
          • Trap Handler 保存 Queue_train 的 Wave 状态
          • 1-10μs 完成 ✓
          • Queue_train 被挂起

T = 5ms:  GPU 切换到高优先级队列
          • Command Processor 发现 Queue_train 被挂起
          • 扫描其他队列，选择 Queue_infer
          • 开始执行 Queue_infer ✓

T = 25ms: 高优先级任务完成
          • Queue_infer 空闲

T = 30ms: GPREEMPT 恢复低优先级队列
          • 检测到 Queue_infer 空闲
          • 恢复 Queue_train
          • CWSR 恢复 Wave 状态（1-10μs）
          • Queue_train 从断点继续执行 ✓

结果：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ 高优先级延迟：~25ms (5ms等待 + 20ms执行)
✓ 低优先级延迟：+25ms (被抢占时间)
✓ Doorbell 性能：完全保留（~100ns）
✓ CWSR 抢占延迟：1-10μs
✓ 调度检测延迟：5ms (可配置)
```

### 为什么不在 Doorbell 阶段就做调度？

```
方案对比：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ 方案A：拦截 Doorbell
   应用 → 拦截层 → 调度决策 → doorbell
          ~~~~~    ~~~~~~~~
          10-100μs  延迟
   
   问题：
   • 拦截开销大（10-100μs vs 原生100ns）
   • 需要 LD_PRELOAD 或修改应用
   • 损失 doorbell 的快速提交性能
   • 需要应用配合

✅ 方案B：Doorbell + 事后监控
   应用 → doorbell (~100ns) → GPU
                               ↓
                        可能选错队列
                               ↓
          GPREEMPT 监控检测（1-10ms 后）
                               ↓
          CWSR 纠正（1-10μs 抢占）
   
   优点：
   • 保留 doorbell 性能 (~100ns)
   • 应用完全透明，无需修改
   • 可靠的优先级保证
   
   权衡：
   • 调度延迟 = 监控间隔（1-10ms）
   • 但远小于无优先级时的延迟（可能数秒）
```

### 详细架构设计

#### 4 个 Feature 的角色定位

```
Feature 分工:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Doorbell (快速提交通道)
   └─ 角色: 任务提交的快速通道
   └─ 保持不变: 应用直接通过 doorbell 提交任务
   └─ 性能: ~100ns MMIO write

2. CWSR (硬件抢占能力)
   └─ 角色: Wave-level 上下文保存/恢复
   └─ 提供: 1-10μs 的快速抢占
   └─ 触发: 由 GPREEMPT 控制

3. GPREEMPT (抢占调度引擎)
   └─ 角色: 内核态调度决策和抢占执行
   └─ 功能:
      ├─ 监控所有队列状态（异步）
      ├─ 检测优先级倒置
      ├─ 触发 CWSR 抢占
      └─ 提供 ioctl 接口给用户态
   
4. XSched (调度框架和策略)
   └─ 角色: 用户态调度框架和策略引擎
   └─ 功能:
      ├─ 优先级队列管理
      ├─ 调度策略（Strict Priority, Weighted Fair, etc）
      ├─ Lv3 接口封装（调用 GPREEMPT ioctl）
      └─ 统计和监控
```

#### 完整工作流程图

```
时间轴工作流程:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

T=0ms: 低优先级应用提交任务
       ┌─────────────────────────────────────────┐
       │ PyTorch App (Low Priority)              │
       │   hipLaunchKernel()                     │
       │     ↓                                   │
       │   *doorbell = wptr  (MMIO, ~100ns)      │
       │     ↓                                   │
       │   GPU 立即开始执行 ✅                    │
       └─────────────────────────────────────────┘

T=5ms: 高优先级应用提交任务
       ┌─────────────────────────────────────────┐
       │ Inference Service (High Priority)       │
       │   hipLaunchKernel()                     │
       │     ↓                                   │
       │   *doorbell = wptr  (MMIO, ~100ns)      │
       │     ↓                                   │
       │   任务提交到 GPU，等待调度 ⏳            │
       └─────────────────────────────────────────┘

T=8ms: GPREEMPT 监控线程检测到优先级倒置
       ┌─────────────────────────────────────────┐
       │ KFD Priority Scheduler (Kernel)         │
       │   • 读取所有队列的 rptr/wptr            │
       │   • 发现:                                │
       │     - Queue_low 正在运行                │
       │     - Queue_high 有待处理任务            │
       │   • 判断: 优先级倒置！                   │
       │     ↓                                   │
       │   触发抢占:                              │
       │   kfd_queue_preempt(Queue_low, CWSR)    │
       └─────────────────────────────────────────┘
                   ↓
       ┌─────────────────────────────────────────┐
       │ CWSR Hardware (GPU)                     │
       │   • Trap Handler 执行                   │
       │   • 保存 Queue_low 的所有 Wave 状态     │
       │   • 1-10μs 完成 ✅                       │
       └─────────────────────────────────────────┘

T=8.01ms: 高优先级任务开始执行
       ┌─────────────────────────────────────────┐
       │ GPU 执行 Queue_high 的任务               │
       │   • 完全占用 GPU                         │
       │   • 低优先级任务已被挂起                 │
       └─────────────────────────────────────────┘

T=50ms: 高优先级任务完成
       ┌─────────────────────────────────────────┐
       │ GPREEMPT 检测到 Queue_high 空闲         │
       │   ↓                                     │
       │ 恢复低优先级队列:                        │
       │   kfd_queue_resume(Queue_low)           │
       │     ↓                                   │
       │ CWSR 恢复 Wave 状态                     │
       │   • 从断点处继续执行 ✅                  │
       └─────────────────────────────────────────┘

结果:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ 高优先级任务: 8ms (检测延迟) + 42ms (执行) = 50ms
✓ 低优先级任务: 被抢占 42ms，总延迟增加
✓ Doorbell 性能: 完全保留 (~100ns)
✓ 抢占延迟: 1-10μs (CWSR)
✓ 调度延迟: 1-10ms (监控间隔)
```

#### 关键设计要点

**1. GPREEMPT: 内核态调度引擎（核心）**

```c
// KFD 优先级调度器（内核态）
int kfd_priority_scheduler_thread(void *data) {
    struct amdgpu_device *adev = data;
    
    while (!kthread_should_stop()) {
        // 1. 扫描所有队列（包括 Doorbell 提交的）
        for_each_queue(q, &adev->kfd->queue_list) {
            u64 rptr = read_queue_rptr(q);
            u64 wptr = read_queue_wptr(q);
            q->pending_work = wptr - rptr;
            q->is_active = (q->pending_work > 0);
        }
        
        // 2. 检测优先级倒置
        struct kfd_queue *high_q = NULL, *low_q = NULL;
        if (detect_priority_inversion(&high_q, &low_q)) {
            // 3. 强制抢占
            printk(KERN_INFO "Priority inversion detected: Q%d (prio %d) waiting, Q%d (prio %d) running\n",
                   high_q->properties.queue_id, high_q->priority,
                   low_q->properties.queue_id, low_q->priority);
            
            // 使用 CWSR 抢占低优先级队列
            kfd_queue_preempt_single(low_q, KFD_PREEMPT_TYPE_WAVEFRONT_SAVE, 1000);
            
            // 统计
            atomic64_inc(&adev->sched_stats.forced_preemptions);
        }
        
        // 4. 定期检查（可配置）
        msleep_interruptible(adev->sched_config.check_interval_ms);
    }
    return 0;
}

// 优先级倒置检测
bool detect_priority_inversion(struct kfd_queue **high_q_out,
                               struct kfd_queue **low_q_out) {
    struct kfd_queue *highest_waiting = NULL;
    struct kfd_queue *lowest_running = NULL;
    
    // 找到最高优先级的等待队列
    for_each_queue(q, &queue_list) {
        if (q->pending_work > 0 && !q->is_running) {
            if (!highest_waiting || q->priority > highest_waiting->priority)
                highest_waiting = q;
        }
    }
    
    // 找到最低优先级的运行队列
    for_each_queue(q, &queue_list) {
        if (q->is_running) {
            if (!lowest_running || q->priority < lowest_running->priority)
                lowest_running = q;
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

**优点**：
- ✅ 捕获所有优先级倒置
- ✅ 无需应用配合，完全透明
- ✅ 保留 doorbell 性能
- ✅ 使用硬件 CWSR 能力

---

**2. XSched: 用户态调度框架**

XSched 提供调度策略和管理，但**不拦截应用的 HIP API**：

```cpp
// XSched 的角色：管理和策略，而非拦截

// 1. 队列管理和优先级分配
class XSched {
public:
    // 创建优先级队列（但不拦截提交）
    XQueue* create_queue(int priority) {
        XQueue* xq = new XQueue(priority);
        
        // 如果是 AMD GPU，使用 GPREEMPT 作为 Lv3 实现
        if (detect_amd_gpu()) {
            xq->set_preempt_level(PreemptLevel::Lv3);
            xq->set_interrupt_fn(gpreempt_interrupt);
            xq->set_restore_fn(gpreempt_restore);
        }
        
        queues_.push_back(xq);
        return xq;
    }
    
    // 监控和统计（从 GPREEMPT 获取信息）
    void update_statistics() {
        // 读取 GPREEMPT 的统计信息
        // 通过 sysfs 或 ioctl 获取
        struct gpreempt_stats stats;
        ioctl(kfd_fd_, AMDKFD_IOC_GET_SCHED_STATS, &stats);
        
        total_preemptions_ = stats.total_preemptions;
        avg_preempt_latency_ = stats.avg_preempt_latency_us;
    }
    
private:
    std::vector<XQueue*> queues_;
    int kfd_fd_;
};

// 2. XSched Lv3 接口 → GPREEMPT ioctl 映射
int gpreempt_interrupt(uint32_t queue_id) {
    int kfd_fd = open("/dev/kfd", O_RDWR);
    if (kfd_fd < 0) return -1;
    
    struct kfd_ioctl_preempt_queue_args args = {
        .queue_id = queue_id,
        .preempt_type = 2,  // WAVEFRONT_SAVE (CWSR)
        .timeout_ms = 1000
    };
    
    int ret = ioctl(kfd_fd, AMDKFD_IOC_PREEMPT_QUEUE, &args);
    close(kfd_fd);
    return ret;
}

int gpreempt_restore(uint32_t queue_id) {
    int kfd_fd = open("/dev/kfd", O_RDWR);
    if (kfd_fd < 0) return -1;
    
    struct kfd_ioctl_resume_queue_args args = {
        .queue_id = queue_id
    };
    
    int ret = ioctl(kfd_fd, AMDKFD_IOC_RESUME_QUEUE, &args);
    close(kfd_fd);
    return ret;
}
```

**XSched 的功能边界**：
- ✅ 提供优先级队列管理
- ✅ 提供 Lv3 接口封装（调用 GPREEMPT）
- ✅ 提供统计和监控
- ✅ 提供调度策略（Strict Priority, Weighted Fair）
- ❌ **不拦截应用的 HIP API**（保留 doorbell）
- ❌ **不做任务提交控制**（由 GPREEMPT 在内核态控制）

---

## 🛠️ 完整实施计划（GPREEMPT + XSched 融合）

### Phase 0: 基础验证（已完成 ✅）

```
任务:
├─ [1] ✅ 验证 MI308X CWSR 支持
│  └─ 确认 cwsr_enable=1, trap handler 存在
│
├─ [2] ✅ 验证 GPREEMPT ioctl 接口
│  └─ PREEMPT_QUEUE (0x87), RESUME_QUEUE (0x88)
│
└─ [3] ✅ 基础测试
   └─ Phase 1.5 跨进程抢占测试

输出:
✓ CWSR 可用
✓ ioctl 接口已定义
✓ 基础抢占功能工作
```

### Phase 1: GPREEMPT 内核态调度器（2-3个月）⭐

**目标**: 实现内核态异步监控和优先级调度

```
任务:
├─ [1] 扩展 KFD 数据结构
│  ├─ 在 kfd_queue 中添加 priority 字段
│  ├─ 在 kfd_queue 中添加 domain 字段（user/system）
│  ├─ 添加队列状态字段（is_active, pending_work）
│  └─ 修改文件: amd/amdkfd/kfd_priv.h
│
├─ [2] 实现优先级调度器
│  ├─ 创建内核监控线程
│  ├─ 实现队列状态监控（rptr/wptr 读取）
│  ├─ 实现优先级倒置检测
│  ├─ 集成现有的 CWSR 抢占（kfd_queue_preempt_single）
│  ├─ 实现队列恢复逻辑
│  └─ 修改文件: amd/amdkfd/kfd_priority_scheduler.c (新建)
│
├─ [3] 扩展 ioctl 接口
│  ├─ 添加设置队列优先级的 ioctl
│  ├─ 添加获取调度统计的 ioctl
│  └─ 修改文件: amd/amdkfd/kfd_chardev.c
│
├─ [4] 配置接口
│  ├─ sysfs 参数（监控间隔，调度策略）
│  ├─ debugfs 调试接口
│  └─ 修改文件: amd/amdkfd/kfd_device.c
│
└─ [5] 测试
   ├─ 2 进程测试（高/低优先级）
   ├─ 测量抢占延迟（目标: 1-10ms）
   ├─ 测量 CWSR 延迟（目标: 1-10μs）
   └─ 稳定性测试（长时间运行）

输出:
✓ 工作的内核态优先级调度器
✓ 可以自动检测和抢占优先级倒置
✓ 保留 doorbell 性能
```

### Phase 2: XSched 集成 GPREEMPT Lv3（1-2个月）

**目标**: 将 GPREEMPT 作为 XSched Lv3 的实现

```
任务:
├─ [1] 修改 XSched HAL 层
│  ├─ 实现 AMD GPREEMPT backend
│  ├─ 实现 interrupt() → ioctl(PREEMPT_QUEUE)
│  ├─ 实现 restore() → ioctl(RESUME_QUEUE)
│  └─ 修改文件: xsched/platforms/hip/hal/src/gpreempt_backend.cpp (新建)
│
├─ [2] 队列 ID 获取
│  ├─ 从 HIP Stream 获取底层 HSA Queue
│  ├─ 从 HSA Queue 获取 KFD Queue ID
│  └─ 或使用 ROCr 调试接口
│
├─ [3] 优先级队列管理
│  ├─ XQueue 创建时分配优先级
│  ├─ 通过 ioctl 设置队列优先级到内核
│  └─ 修改文件: xsched/core/src/xqueue.cpp
│
├─ [4] 统计和监控
│  ├─ 从 GPREEMPT 获取统计信息
│  ├─ 集成到 XSched 的监控系统
│  └─ 修改文件: xsched/core/src/xscheduler.cpp
│
└─ [5] 测试
   ├─ 重新运行 Example 3（多优先级）
   ├─ 期望: 延迟比从 1.07× 提升到 3-4×
   └─ 期望: 抢占延迟 <10ms

输出:
✓ XSched 可以使用 GPREEMPT Lv3
✓ 达到论文级性能
✓ 保留 doorbell 性能
```

### Phase 3: 高级功能和优化（2个月）

```
任务:
├─ [1] 多租户支持
│  ├─ 每个进程/容器独立的优先级空间
│  ├─ 资源配额和限制
│  └─ QoS 保证
│
├─ [2] 高级调度策略
│  ├─ Weighted Fair Scheduling
│  ├─ Time Slice Scheduling
│  ├─ Deadline Scheduling
│  └─ 动态优先级调整
│
├─ [3] 性能优化
│  ├─ 减少监控开销（adaptive polling）
│  ├─ 优化 rptr/wptr 读取（缓存）
│  ├─ 批量抢占（多队列同时抢占）
│  └─ NUMA 感知
│
└─ [4] 监控和调试
   ├─ perf 事件支持
   ├─ tracepoint 支持
   ├─ 性能分析工具
   └─ 可视化界面

输出:
✓ 生产级功能
✓ 优化的性能
✓ 完整的工具链
```

### Phase 4: 生产化和上游贡献（2个月）

```
任务:
├─ [1] 代码质量
│  ├─ 代码审查和重构
│  ├─ 错误处理完善
│  ├─ 内存泄漏检查
│  └─ 静态分析
│
├─ [2] 文档
│  ├─ 架构设计文档
│  ├─ API 文档
│  ├─ 用户手册
│  ├─ 性能调优指南
│  └─ 故障排查指南
│
├─ [3] 测试
│  ├─ 单元测试
│  ├─ 集成测试
│  ├─ 压力测试
│  ├─ 回归测试
│  └─ CI/CD 集成
│
└─ [4] 上游贡献
   ├─ 向 Linux kernel 提交 GPREEMPT patch
   ├─ 向 XSched 项目贡献 AMD backend
   ├─ 论文撰写和发表
   └─ 社区推广

输出:
✓ 生产就绪的调度系统
✓ 完整文档和测试
✓ 开源贡献
```

**总时间**: 7-9个月

**关键里程碑**:
- M1 (3个月): GPREEMPT 内核调度器工作
- M2 (5个月): XSched Lv3 集成完成，性能达标
- M3 (7个月): 生产就绪
- M4 (9个月): 开源发布

---

## 📊 融合方案的优势

### 1. **保留 Doorbell 性能** ⭐⭐⭐⭐⭐

```
关键优势: 不损失任务提交性能
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ 应用继续使用 doorbell 快速提交 (~100ns)
✓ 不拦截 HIP API（无 LD_PRELOAD 开销）
✓ 提交路径与原生性能相同
✓ 适合所有应用（PyTorch, TensorFlow, 自定义代码）
```

### 2. **完全透明** ⭐⭐⭐⭐⭐

```
用户体验: 无需修改应用代码
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ 无需重新编译应用
✓ 无需 LD_PRELOAD 或其他环境变量
✓ 无需修改 Dockerfile 或启动脚本
✓ 就像使用普通 GPU 一样
```

### 3. **强大的抢占能力** ⭐⭐⭐⭐⭐

```
技术能力: 硬件级抢占
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ CWSR 硬件支持（1-10μs 抢占延迟）
✓ Wave-level 上下文保存/恢复
✓ 完整状态保存（PC, SGPRs, VGPRs, LDS）
✓ 可靠的优先级保证
```

### 4. **XSched 框架优势** ⭐⭐⭐⭐

```
调度框架: 成熟的调度策略和管理
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ 多种调度策略（Strict Priority, Weighted Fair, etc）
✓ 优先级队列管理
✓ 统计和监控接口
✓ 论文验证的设计
```

### 5. **增量开发** ⭐⭐⭐

```
实施路径: 可以分阶段实施
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ 先实现 GPREEMPT（内核态调度器）
✓ 再集成 XSched（调度框架）
✓ 逐步添加高级功能
✓ 风险可控
```

---

## 🎯 总结与建议

### 关键洞察：4 个 Feature 的正确关系

```
技术栈融合:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Doorbell (任务提交快速通道)
   └─ 保持原样，不拦截，保留性能
   
2. CWSR (硬件抢占能力)
   └─ 提供 1-10μs 的快速抢占
   └─ Wave-level 上下文保存/恢复
   
3. GPREEMPT (Paper#1: 内核态调度引擎)
   └─ 基于 CWSR 实现
   └─ 异步监控队列状态
   └─ 检测优先级倒置并触发抢占
   └─ 提供 ioctl 接口
   
4. XSched (Paper#2: 调度框架)
   └─ Lv3 实现 = GPREEMPT ioctl
   └─ interrupt() → PREEMPT_QUEUE
   └─ restore() → RESUME_QUEUE
   └─ 提供调度策略和管理

关系图:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    XSched (调度框架)
                         ↓
                   Lv3 接口层
                         ↓
    ioctl(PREEMPT_QUEUE / RESUME_QUEUE)
                         ↓
                    GPREEMPT (内核)
                         ↓
                   CWSR (硬件)
                         
    应用 → Doorbell → GPU (快速提交，不被拦截)
```

### 核心发现

1. ⭐⭐⭐⭐⭐ **GPREEMPT 是 XSched Lv3 的完美实现**
   - XSched Lv3 需要: `interrupt()` 和 `restore()` 接口
   - GPREEMPT 提供: `PREEMPT_QUEUE` 和 `RESUME_QUEUE` ioctl
   - 完美匹配！

2. ⭐⭐⭐⭐⭐ **必须保留 Doorbell**
   - Doorbell 是性能关键路径 (~100ns)
   - 拦截会损失 10-100倍性能
   - 内核态监控可以"事后"检测和抢占

3. ⭐⭐⭐⭐ **CWSR 提供硬件基础**
   - 1-10μs 抢占延迟
   - 完整状态保存
   - 可靠的硬件支持

4. ⭐⭐⭐⭐ **XSched 提供框架和策略**
   - 不需要拦截应用
   - 只需封装 GPREEMPT ioctl
   - 提供高级调度策略

### 实施路径

**Phase 1 (2-3个月): 实现 GPREEMPT** ← 当前阶段
```
目标: Paper#1 的内核态调度器

任务:
├─ 扩展 KFD 数据结构（priority, domain）
├─ 实现监控线程（异步检查队列状态）
├─ 实现优先级倒置检测
├─ 集成 CWSR 抢占
└─ 测试验证

输出:
✓ 工作的内核态优先级调度器
✓ 保留 doorbell 性能
✓ 1-10ms 调度延迟
```

**Phase 2 (1-2个月): 集成到 XSched**
```
目标: GPREEMPT 作为 XSched Lv3 实现

任务:
├─ 修改 XSched HAL 层
├─ 实现 GPREEMPT backend
├─ interrupt() → ioctl(PREEMPT_QUEUE)
├─ restore() → ioctl(RESUME_QUEUE)
└─ 重新测试 Example 3

输出:
✓ XSched 使用 GPREEMPT Lv3
✓ 达到论文级性能（3-4倍延迟差异）
✓ 保留 doorbell 性能
```

**Phase 3 (2个月): 高级功能**
```
任务:
├─ 多租户支持
├─ 高级调度策略
├─ 性能优化
└─ 监控工具

输出:
✓ 生产级功能
✓ 完整工具链
```

**Phase 4 (2个月): 生产化**
```
任务:
├─ 代码质量和测试
├─ 文档
└─ 上游贡献

输出:
✓ 开源发布
✓ 论文发表
```

### 最重要的认识

```
核心洞察:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Paper#1 (GPREEMPT) + Paper#2 (XSched) 的正确融合方式：

❌ 错误: 拦截 HIP API（损失 doorbell 性能）
✅ 正确: 保留 doorbell + 内核态监控

关键:
1. 不拦截应用的任务提交（保留 doorbell）
2. GPREEMPT 在内核态异步监控和抢占
3. XSched Lv3 使用 GPREEMPT ioctl
4. CWSR 提供硬件支持

结果:
✓ 保留 doorbell 性能（~100ns 提交）
✓ 强大的抢占能力（1-10μs CWSR）
✓ 灵活的调度策略（XSched 框架）
✓ 完全透明（无需修改应用）

这就是 GPREEMPT 作为 XSched Lv3 基础 feature 的正确方式！
```

---

**文档版本**: v3.0  
**更新日期**: 2026-01-27  
**重要更新**: 
- ✅ 纠正了错误的"拦截 HIP API"方案
- ✅ 确立了**保留 Doorbell**的正确架构
- ✅ 明确了 GPREEMPT (Paper#1) 作为 XSched (Paper#2) Lv3 基础 feature 的定位

**核心原则**:
1. 保留 Doorbell 快速提交（不拦截）
2. GPREEMPT 内核态异步监控和抢占
3. XSched Lv3 使用 GPREEMPT ioctl
4. CWSR 提供硬件支持

**参考文档**: 
- `2PROC_DKMS_debug`
- `GPREEMPT_MI300_Testing`  
- `XSched_方案对比与问题分析.md`
- `XSched_Lv1_Lv2_Lv3硬件级别详解.md`

