# AMD GPREEMPT 完整抢占流程：从提交到执行的时序分析

**日期**: 2026-01-28  
**核心问题**: 高优先级 kernel 如何抢占低优先级 kernel？  
**关键洞察**: 软件检测 + 软件触发 + 硬件执行

---

## 🎯 核心场景

```
初始状态:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 训练任务（低优先级）的 kernel_low_p 正在 GPU 上执行
• 占用所有 CU (Compute Units)
• 推理服务的高优先级队列空闲

目标:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 推理请求到达，提交 kernel_high_p（高优先级）
• 需要立即执行（低延迟要求）
• 必须抢占 kernel_low_p

问题:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 谁检测到需要抢占？
• 谁触发 CWSR？
• 硬件如何感知？
• 完整时序是什么？
```

---

## 📊 完整时序流程（详细版）

### 阶段 0: 初始状态（T < 0）

```
GPU 状态:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────┐
│ Queue_train (低优先级, priority=1)                      │
│   ├─ Ring Buffer: [kernel_low_p, kernel2, ...]         │
│   ├─ rptr: 100, wptr: 500 (400 个待处理)               │
│   ├─ Doorbell addr: 0xF000_0000                        │
│   └─ 状态: ✅ kernel_low_p 正在所有 CU 上执行          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Queue_infer (高优先级, priority=10)                     │
│   ├─ Ring Buffer: []                                    │
│   ├─ rptr: 0, wptr: 0 (空)                             │
│   ├─ Doorbell addr: 0xF000_1000                        │
│   └─ 状态: ⏸️ 空闲                                      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ GPU Hardware                                            │
│   ├─ Command Processor: 正在调度 Queue_train           │
│   ├─ 所有 CU: 100% 占用（执行 kernel_low_p 的 waves） │
│   └─ MEC (Micro Engine Compute): 监控所有 doorbell     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ KFD 监控线程（内核态）                                   │
│   ├─ 上次检查: T = -5ms                                │
│   ├─ 发现: 只有 Queue_train 活跃                       │
│   └─ 决策: 无需抢占 ✅                                  │
└─────────────────────────────────────────────────────────┘
```

---

### 阶段 1: 高优先级任务提交（T = 0）

```
用户态推理服务:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// 推理请求到达
void handle_inference_request(input_data) {
    // 1. 准备 kernel 参数
    hipMalloc(&d_input, size);
    hipMemcpy(d_input, input_data, size, H2D);
    
    // 2. 提交 kernel ⭐
    hipLaunchKernel(
        inference_kernel,         // kernel_high_p
        grid, block,
        d_input, d_output         // 参数
    );
    
    // ✅ hipLaunchKernel 立即返回（~100ns）
    // 实际执行：写 ring buffer + 敲 doorbell
}

hipLaunchKernel 内部（libamdhip64.so）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 构造 AQL Packet (HSA Queue Packet):
   ┌────────────────────────────────────────┐
   │ AQL Packet (64 bytes)                  │
   │   header: KERNEL_DISPATCH              │
   │   setup: dimensions, grid, block       │
   │   kernel_object: 0x1234_5678           │
   │   kernarg_address: 0xABCD_EF00         │
   │   ...                                  │
   └────────────────────────────────────────┘

2. 写入 Queue_infer 的 Ring Buffer:
   Ring_Buffer[wptr] = AQL_Packet;
   wptr++;  // wptr: 0 → 1

3. ⭐⭐⭐ 敲 Doorbell（MMIO write）:
   *doorbell_infer = wptr;  // 写 0xF000_1000 = 1
   
   // 这是一次 MMIO 写操作（~100ns）
   // 直接写到 GPU 的寄存器空间
   // GPU MEC 立即感知

4. 返回用户态
   // ✅ 延迟: ~100ns
   // 用户程序继续执行
```

**关键点**：
- ✅ 提交非常快（~100ns）
- ✅ 完全在用户态完成（绕过内核）
- ✅ GPU 硬件立即感知（doorbell 中断）
- ⚠️ **但 GPU 不会主动抢占！**

---

### 阶段 2: GPU 硬件感知新任务（T = 100ns）

```
GPU MEC (Micro Engine Compute) 状态机:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────┐
│ MEC Doorbell Monitor                                    │
│   • 检测到 doorbell_infer 写入（0xF000_1000 = 1）      │
│   • 识别: Queue_infer 有新任务                          │
│   • 读取 Queue_infer 的 MQD (Memory Queue Descriptor)   │
│     ├─ rptr: 0                                          │
│     ├─ wptr: 1                                          │
│     ├─ base_addr: 0x1000_0000                          │
│     └─ priority: 10 ⚠️（但硬件不解释优先级！）         │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ GPU Command Processor (CP)                              │
│                                                          │
│ 当前状态:                                                │
│   • 正在执行 Queue_train 的任务                         │
│   • kernel_low_p 的 waves 占用所有 CU                  │
│                                                          │
│ 检测到 Queue_infer 有新任务:                            │
│   • 将 Queue_infer 加入调度队列                         │
│   • 但继续执行 Queue_train ❌❌❌                        │
│                                                          │
│ 为什么不切换？                                           │
│   ❌ 硬件调度器不支持基于优先级的抢占                   │
│   ❌ 或者支持的优先级粒度太粗（只有 2-3 级）            │
│   ❌ 默认行为: Round-Robin 或 FIFO                      │
│                                                          │
│ Queue_infer 的状态:                                     │
│   ⏳ 在 Runlist 中，等待调度                             │
│   ⏳ 需要等待 Queue_train 释放 CU                       │
│   ⏳ 可能等待数十到数百毫秒 ❌❌❌                        │
└─────────────────────────────────────────────────────────┘

关键问题:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ GPU 硬件已经知道 Queue_infer 有任务
❌ 但硬件不会主动抢占 Queue_train
❌ kernel_high_p 被迫等待 ⏳
❌ 这违反了低延迟要求！

解决方案:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 需要软件介入！
✅ KFD 监控线程检测优先级倒置
✅ KFD 主动触发 CWSR 抢占
```

---

### 阶段 3: KFD 监控线程检测（T = 5ms）⭐⭐⭐

```
KFD 内核态监控线程（周期性检查）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// 定时器触发（每 5ms 检查一次）
static int gpreempt_monitor_thread(void *data) {
    while (!kthread_should_stop()) {
        wait_event_interruptible_timeout(
            gpreempt_wq,
            kthread_should_stop(),
            msecs_to_jiffies(5)  // 5ms 间隔
        );
        
        // ⭐ 步骤 1: 扫描所有队列，读取硬件状态
        scan_all_queues();
    }
}

步骤 1: 扫描所有队列
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void scan_all_queues(void) {
    struct kfd_queue *q;
    
    list_for_each_entry(q, &device->queues, list) {
        // 直接读取硬件寄存器（MMIO）
        u32 rptr_hw = readl(q->mqd->rptr_mmio_addr);
        u32 wptr_hw = readl(q->mqd->wptr_mmio_addr);
        
        q->hw_rptr = rptr_hw;
        q->hw_wptr = wptr_hw;
        q->pending_count = (wptr_hw - rptr_hw) & q->ring_mask;
        q->is_active = (q->pending_count > 0);
    }
}

读取结果:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Queue_train:
  • rptr_hw: 120 (从 MMIO 0xF000_0008 读取)
  • wptr_hw: 500
  • pending_count: 380
  • is_active: true ✅
  • priority: 1 (低)

Queue_infer:
  • rptr_hw: 0 (从 MMIO 0xF000_1008 读取)
  • wptr_hw: 1  ⭐ 有任务！
  • pending_count: 1
  • is_active: true ✅  但没有在执行！⏳
  • priority: 10 (高)

步骤 2: 检测优先级倒置 ⭐⭐⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

bool detect_priority_inversion(
    struct kfd_queue **high_q,
    struct kfd_queue **low_q
) {
    // 遍历所有活跃队列
    struct kfd_queue *q1, *q2;
    
    list_for_each_entry(q1, &device->active_queues, active_list) {
        if (!q1->is_active || q1->pending_count == 0)
            continue;
        
        list_for_each_entry(q2, &device->active_queues, active_list) {
            if (!q2->is_active || q2->pending_count == 0)
                continue;
            
            // 检查优先级倒置
            if (q1->priority > q2->priority &&  // q1 优先级更高
                !q1->is_running &&               // 但 q1 没在运行
                q2->is_running) {                // 而 q2 在运行
                
                *high_q = q1;
                *low_q = q2;
                
                pr_debug("Priority inversion detected:\n");
                pr_debug("  High: Queue %p (priority=%d, pending=%d) waiting\n",
                         q1, q1->priority, q1->pending_count);
                pr_debug("  Low:  Queue %p (priority=%d, pending=%d) running\n",
                         q2, q2->priority, q2->pending_count);
                
                return true;
            }
        }
    }
    
    return false;
}

检测结果:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ 优先级倒置！
  • high_q: Queue_infer (priority=10, pending=1, waiting ⏳)
  • low_q:  Queue_train (priority=1, running ✅)

决策: 抢占 Queue_train！
```

---

### 阶段 4: KFD 触发 CWSR 抢占（T = 5.1ms）⭐⭐⭐

```
KFD 抢占控制流程:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

步骤 1: 准备抢占
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void trigger_preemption(
    struct kfd_queue *high_q,
    struct kfd_queue *low_q
) {
    int r;
    
    pr_info("Preempting queue %p (priority=%d) for queue %p (priority=%d)\n",
            low_q, low_q->priority, high_q, high_q->priority);
    
    // 标记状态
    low_q->preemption_pending = true;
    low_q->preempted_by = high_q;
    low_q->preemption_start_time = ktime_get();
    
    // ⭐⭐⭐ 触发 CWSR 抢占（异步）
    r = kfd_queue_preempt_async(
        low_q,
        KFD_PREEMPT_TYPE_WAVEFRONT_SAVE  // CWSR
    );
    
    if (r == 0 || r == -EINPROGRESS) {
        pr_debug("Preemption triggered successfully\n");
    } else {
        pr_err("Preemption failed: %d\n", r);
        low_q->preemption_pending = false;
    }
}

步骤 2: KFD 抢占函数（关键！）⭐⭐⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

int kfd_queue_preempt_async(
    struct kfd_queue *q,
    enum kfd_preempt_type type
) {
    struct kfd_dev *dev = q->device;
    int r;
    
    // 1. 验证队列状态
    if (q->preemption_pending) {
        return -EINPROGRESS;
    }
    
    // 2. ⭐ 备份 MQD (Memory Queue Descriptor)
    //    包含队列的完整状态
    r = checkpoint_mqd(q);
    if (r) {
        pr_err("Failed to checkpoint MQD: %d\n", r);
        return r;
    }
    
    // 3. ⭐⭐⭐ 向 GPU 发送抢占命令
    //    这是触发 CWSR 的关键步骤！
    r = destroy_mqd_with_cwsr(q);
    if (r) {
        pr_err("Failed to trigger CWSR: %d\n", r);
        return r;
    }
    
    // 4. 异步返回（不等待 CWSR 完成）
    return -EINPROGRESS;
}

步骤 3: 向 GPU 发送抢占命令 ⭐⭐⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

int destroy_mqd_with_cwsr(struct kfd_queue *q) {
    struct kfd_dev *dev = q->device;
    struct mqd_manager *mqd_mgr = dev->mqd_mgr;
    
    // ⭐ 构造 HW 命令包
    //    这会发送到 GPU 的 CP (Command Processor)
    struct pm4_packet {
        uint32_t header;
        uint32_t cmd;
        uint32_t data[8];
    } packet;
    
    // 填充 PM4 命令（特定于 AMD GPU）
    packet.header = PM4_TYPE3_PKT;
    packet.cmd = PM4_ME_UNMAP_QUEUES;
    packet.data[0] = q->doorbell_id;
    packet.data[1] = q->queue_id;
    packet.data[2] = PREEMPT_TYPE_WAVEFRONT_SAVE;  // ⭐ CWSR 类型
    packet.data[3] = q->cwsr_area_gpu_addr;        // ⭐ CWSR 保存地址
    packet.data[4] = q->cwsr_area_size;
    
    // ⭐⭐⭐ 写入 GPU CP Ring Buffer
    //       这是软件与 GPU 通信的通道
    write_to_cp_ring(&packet, sizeof(packet));
    
    // ⭐⭐⭐ 敲响 CP Doorbell
    //       通知 GPU CP 有新命令
    writel(cp_wptr, dev->cp_doorbell_addr);
    
    pr_debug("CWSR command sent to GPU CP\n");
    
    return 0;
}

关键点解释:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PM4 (Packet Manager 4) 命令:
   • AMD GPU 的命令格式
   • ME_UNMAP_QUEUES: 取消映射队列（触发抢占）
   • PREEMPT_TYPE_WAVEFRONT_SAVE: 指定 CWSR 类型

2. CWSR 保存区域:
   • cwsr_area_gpu_addr: GPU 内存地址
   • 预先分配（队列创建时）
   • 大小足够保存所有 Wave 状态

3. CP Ring Buffer:
   • 软件向 GPU CP 发送命令的通道
   • 类似于应用的 AQL Ring Buffer
   • 但这是驱动专用的控制通道

4. CP Doorbell:
   • 通知 GPU CP 检查 Ring Buffer
   • 类似于应用 doorbell，但发给 CP
```

---

### 阶段 5: GPU 执行 CWSR（T = 5.11ms）⭐⭐⭐

```
GPU Command Processor (CP) 处理:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CP 检测到 doorbell
   • 读取 CP Ring Buffer
   • 发现 PM4_ME_UNMAP_QUEUES 命令
   • 解析参数:
     - queue_id: Queue_train 的 ID
     - preempt_type: WAVEFRONT_SAVE (CWSR)
     - save_area: 0x2000_0000 (GPU 内存)

2. CP 向 MEC 发送抢占请求
   • MEC: Micro Engine Compute
   • 负责管理 Compute 队列

3. MEC 接收抢占请求
   • 找到 Queue_train
   • 检查当前正在执行的 Wavefronts

GPU Trap Handler 执行 CWSR ⭐⭐⭐:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌────────────────────────────────────────────────────────┐
│ 对于 Queue_train 的每个活跃 Wave:                      │
│                                                         │
│ Wave 0 (CU 0, SIMD 0):                                 │
│   ├─ 1. 触发 Trap                                      │
│   │    • 硬件机制，类似 CPU 中断                       │
│   │    • Wave 暂停执行                                 │
│   │                                                     │
│   ├─ 2. 跳转到 Trap Handler (GPU 固件)                │
│   │    • 运行在 GPU 上的特殊代码                       │
│   │    • 由 AMD 提供，驱动加载到 GPU                   │
│   │                                                     │
│   ├─ 3. 保存 Wave 状态到 CWSR Area ⭐                  │
│   │    base = cwsr_area_gpu_addr + wave_id * 1KB      │
│   │                                                     │
│   │    保存内容:                                        │
│   │    • PC (Program Counter): 0x1234                 │
│   │        - 当前执行到哪条指令                        │
│   │                                                     │
│   │    • SGPRs (Scalar GPRs, 128 个):                 │
│   │        - SGPR[0..127]                             │
│   │        - 用于标量计算                              │
│   │                                                     │
│   │    • VGPRs (Vector GPRs, 256 个):                 │
│   │        - VGPR[0..255] for each lane (64 lanes)   │
│   │        - 64 threads * 256 regs = 16KB 数据        │
│   │                                                     │
│   │    • LDS (Local Data Share):                      │
│   │        - 共享内存内容                              │
│   │        - 最多 64KB per Workgroup                  │
│   │                                                     │
│   │    • Mode Registers:                              │
│   │        - 执行模式标志                              │
│   │        - FP/INT 模式                               │
│   │                                                     │
│   │    • Status Registers:                            │
│   │        - EXEC mask (64-bit)                       │
│   │        - VCC (Vector Condition Code)              │
│   │                                                     │
│   │    • ACC VGPRs (Accumulator, MI300):              │
│   │        - Matrix 加速器寄存器                       │
│   │        - 用于 WMMA 指令                            │
│   │                                                     │
│   └─ 4. 标记 Wave 为 "已保存"                          │
│        • 更新硬件状态寄存器                            │
│        • 释放 SIMD 资源                                │
│                                                         │
│ Wave 1, Wave 2, ... (并行保存):                        │
│   • 每个 CU 的 Trap Handler 独立运行                   │
│   • 多个 Wave 同时保存                                 │
│   • 延迟: ~1-10μs per wave                            │
└────────────────────────────────────────────────────────┘

假设 Queue_train 有 100 个活跃 Waves:
  • MI308X 有 304 个 CU
  • 每个 CU 最多 32 个 Waves
  • CWSR 并行执行
  • 总延迟: ~1-10μs（硬件并行）

5. MEC 更新队列状态:
   • Queue_train.status = PREEMPTED
   • Queue_train.active_waves = 0
   • 释放所有 CU 资源

6. MEC 通知 CP 抢占完成
   • 可能触发中断到驱动（可选）
   • 或驱动轮询检查

延迟总结:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• CP 处理命令: ~1μs
• Trap 触发: ~100ns per wave
• 状态保存: 1-10μs (并行)
• 资源释放: ~1μs

总计: 1-10μs ✅ 非常快！
```

---

### 阶段 6: GPU 切换到高优先级队列（T = 5.12ms）

```
GPU MEC 调度决策:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 扫描 Runlist（所有活跃队列）:
   Queue_train: status=PREEMPTED (跳过)
   Queue_infer: status=ACTIVE, pending=1 ✅

2. 选择 Queue_infer（因为它是唯一活跃的）

3. 从 Queue_infer 读取任务:
   • rptr: 0
   • wptr: 1
   • 读取 Ring_Buffer[0]: AQL Packet (kernel_high_p)

4. 加载 Context:
   • 读取 Queue_infer 的 MQD
   • 配置 CU 资源
   • 加载 Shader 代码

5. 分发到 CU:
   • 创建新的 Wavefronts
   • 分配到空闲的 CU
   • 开始执行 kernel_high_p ✅

6. 更新硬件指针:
   • Queue_infer.rptr = 1
   • 写回 MMIO (让驱动可见)

延迟: <1μs
```

---

### 阶段 7: KFD 回调处理（T = 5.13ms）

```
GPU → KFD 完成通知:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

选项 1: 中断通知（推荐）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// GPU 触发中断
static irqreturn_t kfd_interrupt_handler(int irq, void *data) {
    struct kfd_dev *dev = data;
    u32 status = readl(dev->interrupt_status_reg);
    
    if (status & KFD_IRQ_PREEMPTION_COMPLETE) {
        // 读取哪个队列完成了抢占
        u32 queue_id = readl(dev->preemption_queue_id_reg);
        struct kfd_queue *q = find_queue_by_id(dev, queue_id);
        
        if (q && q->preemption_pending) {
            // 调用完成回调
            gpreempt_preemption_complete_cb(q, KFD_PREEMPT_SUCCESS);
        }
        
        // 清除中断
        writel(KFD_IRQ_PREEMPTION_COMPLETE, dev->interrupt_clear_reg);
        
        return IRQ_HANDLED;
    }
    
    return IRQ_NONE;
}

选项 2: 轮询检查（备选）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// 监控线程下次检查时发现
void scan_all_queues(void) {
    struct kfd_queue *q;
    
    list_for_each_entry(q, &device->queues, list) {
        if (q->preemption_pending) {
            // 检查硬件状态
            u32 status = readl(q->mqd->status_reg);
            
            if (status & QUEUE_STATUS_PREEMPTED) {
                // 抢占完成
                gpreempt_preemption_complete_cb(q, KFD_PREEMPT_SUCCESS);
            }
        }
    }
}

完成回调处理:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void gpreempt_preemption_complete_cb(
    struct kfd_queue *q,
    enum kfd_preempt_result result
) {
    u64 latency_us = ktime_us_delta(
        ktime_get(),
        q->preemption_start_time
    );
    
    pr_info("Queue %p preemption completed:\n", q);
    pr_info("  Latency: %llu us\n", latency_us);
    pr_info("  Preempted by: Queue %p (priority=%d)\n",
            q->preempted_by, q->preempted_by->priority);
    
    // 更新统计
    q->total_preemption_count++;
    q->total_preemption_latency_us += latency_us;
    
    // 清除状态
    q->preemption_pending = false;
    
    // 唤醒等待的高优先级队列（如果需要）
    if (q->preempted_by) {
        wake_up_queue(q->preempted_by);
        q->preempted_by = NULL;
    }
    
    // 通知用户态（可选）
    if (q->event_fd) {
        eventfd_signal(q->event_fd, 1);
    }
}
```

---

### 阶段 8: 恢复低优先级队列（T > 100ms）

```
当高优先级任务完成后:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. kernel_high_p 执行完成
   • Queue_infer.rptr = 1
   • Queue_infer.wptr = 1
   • 队列变空

2. MEC 检测到 Queue_infer 空闲
   • 扫描 Runlist
   • 发现 Queue_train 状态=PREEMPTED

3. 决定恢复 Queue_train:
   • 读取 CWSR Area
   • 恢复所有 Wave 状态
   • 重新分配到 CU
   • 继续执行 kernel_low_p

4. kernel_low_p 从断点继续
   • 就像没被打断一样
   • 透明的抢占 ✅
```

---

## 📊 完整时间线总结

```
端到端时间线（从提交到执行）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

T = 0ms:       用户态提交 kernel_high_p
               • hipLaunchKernel() → doorbell (~100ns) ✅
               └─→ GPU 感知，但不抢占 ❌

T = 0-5ms:     高优先级任务等待 ⏳
               • GPU 继续执行 kernel_low_p
               • kernel_high_p 在队列中等待
               • 这是主要延迟来源！⚠️

T = 5ms:       KFD 监控线程检查（周期性）
               • 读取所有队列状态 (~10μs)
               • 检测优先级倒置 (~1μs)
               • 决定抢占 Queue_train
               └─→ 触发 CWSR (~1μs)

T = 5.001ms:   GPU 执行 CWSR
               • CP 处理命令 (~1μs)
               • Trap Handler 保存状态 (1-10μs) ✅
               • 释放 CU 资源 (~1μs)
               └─→ 总计 1-10μs ✅

T = 5.010ms:   GPU 切换到 Queue_infer
               • 加载 Context (<1μs)
               • 开始执行 kernel_high_p ✅

T = 5.011ms:   KFD 收到完成通知
               • 中断 或 轮询 (~10μs)
               • 更新统计
               • 清除状态

总延迟分析:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

提交延迟:     ~100ns       (Doorbell, ✅ 快)
检测延迟:     5ms          (监控周期, ⚠️ 主要延迟)
触发延迟:     ~1μs         (KFD 处理, ✅ 快)
CWSR 延迟:    1-10μs       (硬件, ✅ 快)
切换延迟:     <1μs         (硬件, ✅ 快)
────────────────────────────────────────
端到端延迟:   ~5.011ms

关键瓶颈:
  ⚠️ 监控周期（5ms）是主要延迟来源
  ✅ 硬件部分非常快（<20μs）
  ✅ 可以通过减小监控间隔优化（1ms）
```

---

## 🔑 核心洞察

### 1. 三方角色分工

```
用户态（应用）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 提交任务（hipLaunchKernel）
• 写 Ring Buffer + 敲 Doorbell
• 延迟: ~100ns
• 不参与抢占决策

KFD 驱动（软件）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 监控所有队列状态（周期性）⭐
• 检测优先级倒置 ⭐
• 触发 CWSR 抢占（发送 PM4 命令）⭐
• 处理完成通知
• 延迟: 5ms (检测) + 1μs (触发)

GPU 硬件:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 感知 Doorbell（立即）
• 执行 CWSR（Trap Handler）⭐
• 保存/恢复 Wave 状态（自动）⭐
• Context 切换（自动）
• 延迟: 1-10μs
```

### 2. 为什么需要软件监控？

```
关键问题:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❓ GPU 硬件已经知道 Queue_infer 有任务（通过 doorbell）
❓ 为什么不直接抢占？

答案:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. AMD GPU 的硬件调度器不支持细粒度优先级
   • 或者只支持 2-3 个优先级级别
   • 不足以满足复杂的调度需求

2. 优先级是软件概念
   • 由应用/框架定义（1-10）
   • 硬件不理解这些优先级值

3. 抢占决策需要复杂逻辑
   • 考虑公平性
   • 防止饥饿
   • 时间片管理
   • 这些都需要软件实现

4. 灵活性
   • 可以支持多种调度策略
   • 可以动态调整
   • 不需要修改硬件

结论:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

硬件提供能力（CWSR）：可以抢占
软件提供策略（GPREEMPT）：何时抢占

这是经典的"机制与策略分离"设计！
```

### 3. 关键优化点

```
当前瓶颈:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

监控周期（5ms）导致最多 5ms 的检测延迟

优化方案:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 减小监控间隔:
   • 5ms → 1ms: 检测延迟降到 1ms
   • 代价: CPU 消耗增加 5 倍

2. 事件驱动触发:
   • 高优先级队列提交时触发检查
   • 需要拦截 doorbell 写入（性能损失）
   • 或增加软件通知路径

3. 硬件辅助（理想）:
   • GPU 支持优先级中断
   • 硬件检测优先级倒置并通知驱动
   • 需要硬件支持（MI300 不支持）

4. 混合方案（推荐）:
   • 正常监控周期: 5ms（低 CPU 消耗）
   • 延迟敏感队列: 标记为 "latency_critical"
   • 对这些队列使用 1ms 间隔
   • 或用户态可以主动触发检查（ioctl）

当前最优:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• 正常队列: 5ms 监控（足够好）
• 关键队列: 1ms 监控（低延迟）
• 用户态触发: 允许手动抢占（调试）

端到端延迟:
  • 5ms 监控: ~5.011ms
  • 1ms 监控: ~1.011ms ✅
  • 手动触发: ~0.011ms ✅✅
```

---

## 📝 完整伪代码实现

```c
// ============================================================================
// 完整的 AMD GPREEMPT 实现伪代码
// ============================================================================

// 1. 监控线程主循环
// ============================================================================
static int gpreempt_monitor_thread(void *data) {
    struct gpreempt_scheduler *sched = data;
    struct kfd_queue *high_q, *low_q;
    
    pr_info("GPREEMPT: Monitor thread started\n");
    
    while (!kthread_should_stop()) {
        // 休眠 5ms 或被唤醒
        wait_event_interruptible_timeout(
            sched->wait_queue,
            kthread_should_stop() || atomic_read(&sched->trigger_check),
            msecs_to_jiffies(sched->check_interval_ms)  // 5ms
        );
        
        if (kthread_should_stop())
            break;
        
        atomic_set(&sched->trigger_check, 0);
        
        // ⭐ 步骤 1: 扫描所有队列
        scan_all_queues(sched);
        
        // ⭐ 步骤 2: 检测优先级倒置
        if (detect_priority_inversion(sched, &high_q, &low_q)) {
            // ⭐ 步骤 3: 触发抢占（异步）
            trigger_preemption_async(sched, high_q, low_q);
        }
        
        // 更新统计
        update_stats(sched);
    }
    
    pr_info("GPREEMPT: Monitor thread exiting\n");
    return 0;
}

// 2. 扫描所有队列
// ============================================================================
static void scan_all_queues(struct gpreempt_scheduler *sched) {
    struct kfd_queue *q;
    
    list_for_each_entry(q, &sched->device->queues, list) {
        // 直接读取硬件寄存器（MMIO）
        u32 rptr = readl(q->mqd->rptr_mmio_addr);
        u32 wptr = readl(q->mqd->wptr_mmio_addr);
        
        q->hw_rptr = rptr;
        q->hw_wptr = wptr;
        q->pending_count = (wptr - rptr) & q->ring_mask;
        q->is_active = (q->pending_count > 0);
        
        // 检查是否正在运行（可能需要额外的硬件状态）
        u32 status = readl(q->mqd->status_reg);
        q->is_running = (status & QUEUE_STATUS_ACTIVE);
    }
}

// 3. 检测优先级倒置
// ============================================================================
static bool detect_priority_inversion(
    struct gpreempt_scheduler *sched,
    struct kfd_queue **out_high_q,
    struct kfd_queue **out_low_q
) {
    struct kfd_queue *q1, *q2;
    
    list_for_each_entry(q1, &sched->device->queues, list) {
        if (!q1->is_active || q1->pending_count == 0)
            continue;
        
        list_for_each_entry(q2, &sched->device->queues, list) {
            if (!q2->is_active || q2->pending_count == 0)
                continue;
            
            // 优先级倒置条件
            if (q1->priority > q2->priority &&  // q1 优先级更高
                !q1->is_running &&               // 但 q1 没在运行
                q2->is_running) {                // 而 q2 在运行
                
                *out_high_q = q1;
                *out_low_q = q2;
                
                pr_debug("Priority inversion: high=%p(prio=%d) low=%p(prio=%d)\n",
                         q1, q1->priority, q2, q2->priority);
                
                return true;
            }
        }
    }
    
    return false;
}

// 4. 触发抢占（异步）
// ============================================================================
static void trigger_preemption_async(
    struct gpreempt_scheduler *sched,
    struct kfd_queue *high_q,
    struct kfd_queue *low_q
) {
    int r;
    
    pr_info("Preempting queue %p for queue %p\n", low_q, high_q);
    
    // 标记状态
    low_q->preemption_pending = true;
    low_q->preempted_by = high_q;
    low_q->preemption_start_time = ktime_get();
    
    // ⭐ 触发 CWSR（异步）
    r = kfd_queue_preempt_async(low_q, KFD_PREEMPT_TYPE_WAVEFRONT_SAVE);
    
    if (r == 0 || r == -EINPROGRESS) {
        atomic_inc(&sched->total_preemptions);
    } else {
        pr_err("Preemption failed: %d\n", r);
        low_q->preemption_pending = false;
    }
}

// 5. KFD 抢占函数（关键！）
// ============================================================================
static int kfd_queue_preempt_async(
    struct kfd_queue *q,
    enum kfd_preempt_type type
) {
    struct kfd_dev *dev = q->device;
    int r;
    
    // 验证
    if (q->preemption_pending)
        return -EINPROGRESS;
    
    // ⭐ 备份 MQD
    r = checkpoint_mqd(q);
    if (r)
        return r;
    
    // ⭐⭐⭐ 向 GPU 发送 CWSR 命令
    r = send_cwsr_command_to_gpu(q, type);
    if (r)
        return r;
    
    return -EINPROGRESS;  // 异步
}

// 6. 发送 CWSR 命令到 GPU
// ============================================================================
static int send_cwsr_command_to_gpu(
    struct kfd_queue *q,
    enum kfd_preempt_type type
) {
    struct kfd_dev *dev = q->device;
    struct pm4_packet pkt;
    
    // 构造 PM4 命令
    memset(&pkt, 0, sizeof(pkt));
    pkt.header = PM4_TYPE3_PKT | PM4_ME_UNMAP_QUEUES;
    pkt.data[0] = q->doorbell_id;
    pkt.data[1] = q->queue_id;
    pkt.data[2] = PREEMPT_TYPE_WAVEFRONT_SAVE;  // CWSR
    pkt.data[3] = lower_32_bits(q->cwsr_area_gpu_addr);
    pkt.data[4] = upper_32_bits(q->cwsr_area_gpu_addr);
    pkt.data[5] = q->cwsr_area_size;
    
    // ⭐ 写入 CP Ring Buffer
    write_to_cp_ring(dev, &pkt, sizeof(pkt));
    
    // ⭐⭐⭐ 敲响 CP Doorbell（通知 GPU）
    writel(dev->cp_wptr, dev->cp_doorbell_addr);
    
    pr_debug("CWSR command sent to GPU\n");
    
    return 0;
}

// 7. 完成回调
// ============================================================================
static void gpreempt_preemption_complete_cb(
    struct kfd_queue *q,
    enum kfd_preempt_result result
) {
    u64 latency_us = ktime_us_delta(ktime_get(), q->preemption_start_time);
    
    pr_info("Queue %p preemption completed in %llu us\n", q, latency_us);
    
    // 更新统计
    q->total_preemption_count++;
    q->total_preemption_latency_us += latency_us;
    
    // 清除状态
    q->preemption_pending = false;
    
    // 唤醒高优先级队列
    if (q->preempted_by) {
        wake_up_queue(q->preempted_by);
        q->preempted_by = NULL;
    }
}
```

---

**文档版本**: v1.0  
**创建日期**: 2026-01-28  
**核心价值**: 完整梳理了从提交到抢占的全流程

**关键结论**:
1. 软件（KFD）负责检测和触发
2. 硬件（CWSR）负责实际执行
3. 监控周期是主要延迟来源（可优化）
4. 硬件部分非常快（<20μs）


