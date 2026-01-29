# XSched Level 1 完整调度逻辑分析

**日志文件**: 05_logic_trace_full.log  
**测试配置**: 2线程 (1 High P10 + 1 Low P1), 每线程3任务×5 kernels  
**目的**: 追踪完整的 XSched 调度逻辑流程

---

## 📊 日志结构

完整日志包含以下层次的信息：

### 1. 应用层 (APP)
```
[APP] - 应用主逻辑
[APP-T0-P10] - Thread 0, Priority 10 的应用操作
[APP-T1-P1]  - Thread 1, Priority 1 的应用操作
```

### 2. XSched 层
```
[INFO] - XSched 框架信息
[XSCHED-REGULAR-N] - XLaunchKernel 拦截日志
[XQUEUE-SUBMIT] - Kernel 进入缓存
[XQUEUE-LAUNCH] - Kernel 实际提交到 GPU
[XQUEUE-PAUSE] - LaunchWorker 暂停
[XQUEUE-RESUME] - LaunchWorker 恢复
[HPF-SCHED] - HPF 调度器决策
```

### 3. HIP 层
```
（当 AMD_LOG_LEVEL=3 时会有 HIP runtime 的详细日志）
```

---

## 🔍 关键调度流程

### 阶段 1: 初始化

```
1. [APP] 设置 XSched 调度器
   └─> XHintSetScheduler(Local, HPF)

2. [APP] 创建线程
   └─> Thread 0 (Priority 10) - High
   └─> Thread 1 (Priority 1)  - Low
```

### 阶段 2: Thread 0 (High Priority) 启动

```
[APP-T0-P10] === Thread 0 Priority 10 START ===
  ↓
[APP-T0-P10] hipStreamCreate
  ↓
[APP-T0-P10] HipQueueCreate
  ↓
[INFO] XQueue (0x...) from process ... created
  ↓
[APP-T0-P10] XQueueCreate
  ↓
[APP-T0-P10] XQueueSetLaunchConfig(1,1)
  ↓
[APP-T0-P10] XHintPriority(10)
  ↓
[INFO] set priority 10 for XQueue 0x...
```

### 阶段 3: Thread 0 提交第一个 Task

```
[APP-T0-P10] ========== Task 1/3 START ==========
  ↓
[APP-T0-P10] Launching kernel 1/5...
  ↓
[XSCHED-REGULAR-1] XLaunchKernel stream=0x...
  ↓
[XQUEUE-SUBMIT] XQ=0x... enqueued kernel idx=1
  ↓
[HPF-SCHED] === Scheduling cycle ===
[HPF-SCHED] XQ=0x... prio=10 >= max=10 -> RESUME
  ↓
[XQUEUE-LAUNCH] Launching kernel idx=1 to GPU
```

**关键观察**: 
- Kernel 不直接提交，先进入 XQueue 缓存
- HPF 调度器检查优先级，决定 RESUME
- LaunchWorker 从缓存取出 kernel 并提交到 GPU

### 阶段 4: Thread 1 (Low Priority) 启动

```
[APP] Starting Low Priority thread 1
  ↓
[APP-T1-P1] === Thread 1 Priority 1 START ===
  ↓
（创建 stream, XQueue 等，过程同 Thread 0）
  ↓
[INFO] XQueue (0x...) from process ... created
[INFO] set priority 1 for XQueue 0x...
```

### 阶段 5: Thread 1 提交 Kernel 时被暂停 ⭐ 关键！

```
[APP-T1-P1] Launching kernel 1/5...
  ↓
[XSCHED-REGULAR-N] XLaunchKernel stream=0x...
  ↓
[XQUEUE-SUBMIT] XQ=0x... (Low) enqueued kernel idx=1
  ↓
[HPF-SCHED] === Scheduling cycle ===
[HPF-SCHED] XQ=0x... (Low) prio=1 < max=10 -> SUSPEND ⭐
[HPF-SCHED] XQ=0x... (High) prio=10 >= max=10 -> RESUME
  ↓
[XQUEUE-PAUSE] Worker paused (pause_count=1) ⭐⭐⭐
```

**关键观察**: 
- 低优先级 kernel 进入缓存
- HPF 调度器比较优先级：1 < 10
- 决策：SUSPEND 低优先级队列
- LaunchWorker 暂停，kernel 堆积在缓存中

### 阶段 6: 高优先级任务继续执行

```
（Thread 0 继续执行，不受影响）

[APP-T0-P10] Launching kernel 2/5...
[APP-T0-P10] Launching kernel 3/5...
...
[XQUEUE-SUBMIT] 连续进入缓存
[XQUEUE-LAUNCH] 连续提交到 GPU
```

### 阶段 7: 高优先级完成后，低优先级恢复

```
[APP-T0-P10] Task DONE
  ↓
（高优先级不再提交新 kernel）
  ↓
[HPF-SCHED] === Scheduling cycle ===
[HPF-SCHED] XQ=0x... (Low) prio=1 >= max=1 -> RESUME ⭐
  ↓
[XQUEUE-RESUME] Worker resumed ⭐
  ↓
[XQUEUE-LAUNCH] Launching kernel idx=1 to GPU （低优先级开始执行）
```

---

## 🎯 核心调度机制总结

### Progressive Command Launching (Level 1)

```
1. Kernel 拦截
   应用: kernel<<<>>>()
     ↓ (LD_PRELOAD 或直接链接)
   XLaunchKernel()
     ↓
   XQueue::Submit()
     ↓
   CommandBuffer::Enqueue() ← Kernel 缓存在这里

2. HPF 调度决策
   Scheduler::Sched()
     ↓ (定期执行)
   if (priority < max_priority):
       Suspend(xqueue)  ← 暂停低优先级
   else:
       Resume(xqueue)   ← 继续高优先级

3. LaunchWorker 响应
   LaunchWorker::WorkerLoop()
     ↓
   while (paused):
       wait() ← 暂停，等待 Resume
     ↓
   hw_cmd = CommandBuffer::Dequeue()
     ↓
   HwQueue::Launch(hw_cmd) ← 实际提交到 GPU
```

### 优先级效果产生机制

```
高优先级队列:
  ✓ HPF 决策: RESUME
  ✓ LaunchWorker: 持续运行
  ✓ Kernel: 快速从缓存提交到 GPU
  → 延迟低

低优先级队列:
  ✗ HPF 决策: SUSPEND
  ✗ LaunchWorker: 暂停等待
  ✗ Kernel: 堆积在缓存中
  → 延迟高（包含等待时间）
```

---

## 📂 日志文件说明

| 文件 | 内容 | 大小 |
|------|------|------|
| 01_8thread_full.log | 8线程完整测试日志 | 1.5MB |
| 02_key_scheduling.log | 关键调度日志（过滤版） | 小 |
| 03_test_results.log | 测试结果统计 | 小 |
| 04_scheduling_sequence.log | 调度序列（过滤版） | 小 |
| 05_logic_trace_full.log | 2线程逻辑追踪（完整） | 172KB |
| 06_logic_flow_analysis.md | 本文档 | - |

---

## 🔍 如何阅读日志

### 推荐阅读顺序

1. **先读本文档** (06_logic_flow_analysis.md)
   - 理解整体调度流程

2. **读 05_logic_trace_full.log**
   - 2线程简化版，容易追踪
   - 包含 APP 层和 XSched 层的完整交互

3. **读 04_scheduling_sequence.log**
   - 只看关键调度决策
   - 重点关注 SUSPEND/PAUSE/RESUME

4. **读 01_8thread_full.log**
   - 8线程完整测试
   - 看到真实的资源竞争

### 关键日志标记

```
⭐ [HPF-SCHED] ... SUSPEND
   → 调度器决定暂停低优先级

⭐⭐ [XQUEUE-PAUSE] Worker paused
   → LaunchWorker 实际暂停

⭐⭐⭐ [XQUEUE-RESUME] Worker resumed
   → LaunchWorker 恢复，开始提交堆积的 kernel
```

---

**创建时间**: 2026-01-29  
**配合文档**: FINAL_SUCCESS_REPORT.md, VERIFICATION_SUCCESS.md
