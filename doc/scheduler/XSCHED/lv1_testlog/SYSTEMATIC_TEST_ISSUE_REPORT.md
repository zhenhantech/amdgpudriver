# XSched 系统测试问题报告

**日期**: 2026-01-29  
**状态**: ⚠️ **发现重大问题 - XSched测试中kernels可能未真正执行**

---

## 🎯 测试目标

按照用户要求进行系统性测试：

1. **Test 1**: 单线程基线 → 得到 `latency_1_PROC`
2. **Test 2**: 16线程无XSched → 得到 `latency_16_PROC_concurrent`
3. **Test 3A**: 16线程 (3H+13L) with XSched → 得到 `latency_3_PROC_High`, `latency_13_PROC_Low`
4. **Test 3B**: 16线程 (1H+15L) with XSched → 得到 `latency_1_PROC_High`, `latency_15_PROC_Low`

**预期结论**:
```
latency_1_PROC < latency_3_PROC_High < latency_16_PROC_concurrent
```

---

## 📊 测试结果（Workload: 每线程10任务×30 kernels）

### Test 1: 单线程基线 ✅
```
Total time: 24.13 seconds
Avg latency: 476.50 ms
Status: ✅ 正常
```

### Test 2: 16线程无XSched ✅
```
Total time: 83.40 seconds
Avg latency: 8002.22 ms
Per-thread latency: 7696-8303 ms
Status: ✅ 正常，显示并发竞争导致延迟增加16.8倍
```

### Test 3A: 16线程 (3H+13L) with XSched ❌
```
Total time: 0.38 seconds ⚠️ 异常！
High Priority: Avg=0.37 ms
Low Priority:  Avg=0.86 ms
Status: ❌ 异常，total time只有0.38秒（vs Test 2的83秒）
```

### Test 3B: 16线程 (1H+15L) with XSched ❌
```
Total time: 0.37 seconds ⚠️ 异常！
High Priority: Avg=0.68 ms
Low Priority:  Avg=0.47 ms
Status: ❌ 异常，total time只有0.37秒
```

---

## 🚨 发现的问题

### 问题 1: XSched测试时间异常短

**观察**:
- 无XSched: 83.40 秒
- XSched:    0.38 秒
- **差距**: 220倍！

**不合理原因**:
- XSched只是改变kernel调度顺序，不应该使总执行时间减少220倍
- 这意味着kernels可能没有真正执行

### 问题 2: 之前的8线程测试也有同样问题

回顾之前被认为"成功"的8线程测试：
```
配置: 8线程 × 50任务 × 30 kernels = 12,000 kernels (vs Test 2的4,800 kernels)
Total time: 0.40 seconds ⚠️
Result: High 0.42ms vs Low 0.81ms
```

**对比分析**:
- 12,000 kernels (XSched): 0.40秒
- 4,800 kernels (无XSched): 83.4秒

**结论**: 之前的"优先级效果"(0.42ms vs 0.81ms)可能**不是真正的kernel执行延迟**，而只是XSched内部调度overhead的差异！

---

## 🔍 根本原因分析

### 可能原因 1: LaunchWrapper 失败

**代码现状** (`hip_queue.cpp:53`):
```cpp
void HipQueue::Launch(std::shared_ptr<preempt::HwCommand> hw_cmd)
{
    auto cmd = std::dynamic_pointer_cast<HipCommand>(hw_cmd);
    XASSERT(cmd != nullptr, "hw_cmd is not a HipCommand");
    if (cmd->LaunchWrapper(kStream) != hipSuccess) {
        XWARN("Failed to enqueue command, continuing...");  // ⚠️ 只是warning！
    };
}
```

**问题**:
1. 之前为了避免程序崩溃，把fatal assertion改成了warning
2. 导致LaunchWrapper失败时，kernel没有提交到GPU
3. 但程序继续执行，`hipStreamSynchronize`立即返回（因为queue是空的）
4. 结果：测量的是"没有kernel"的延迟，而不是真正的执行延迟

### 可能原因 2: Stream注册问题

XSched通过`HipQueueCreate`注册stream，但可能：
- Stream注册失败或不完整
- 后续的kernel launch没有正确路由到XQueue
- Kernels走了fallback path（直接提交到原始HIP）但没有被正确追踪

---

## ✅ 验证方法

需要验证kernels是否真的执行：

### 方法 1: 检查计算结果

添加kernel结果验证：
```cpp
// 初始化输入
float *h_in = malloc(...);
for (int i = 0; i < SIZE; i++) h_in[i] = 2.0f;

// Launch kernel: out[i] = in[i] + 1.0f
hipLaunchKernelGGL(add_kernel, ...);
hipStreamSynchronize(stream);

// 验证结果
float *h_out = malloc(...);
hipMemcpy(h_out, d_out, ...);
for (int i = 0; i < SIZE; i++) {
    if (h_out[i] != 3.0f) {
        printf("❌ Kernel did NOT execute!\n");
    }
}
```

### 方法 2: 监控GPU使用率

```bash
# 在测试期间实时监控
rocm-smi --showuse

# 预期：
# - 无XSched测试：GPU使用率 85-100%
# - XSched测试：如果kernels真的执行，也应该 85-100%
# - 如果XSched测试GPU使用率很低，说明kernels没执行
```

### 方法 3: 检查LaunchWrapper失败日志

```bash
# 查看是否有大量warning
grep "Failed to enqueue command" test_output.log | wc -l

# 如果有很多warning，说明kernels确实没有提交成功
```

---

## 🎯 推荐的修复方案

### 方案 1: 回滚到原始实现（短期）

```cpp
// 恢复fatal assertion
void HipQueue::Launch(std::shared_ptr<preempt::HwCommand> hw_cmd)
{
    auto cmd = std::dynamic_pointer_cast<HipCommand>(hw_cmd);
    XASSERT(cmd != nullptr, "hw_cmd is not a HipCommand");
    XASSERT(cmd->LaunchWrapper(kStream) == hipSuccess, "Failed to enqueue command");
}
```

**问题**: 如果LaunchWrapper真的失败，程序会崩溃

### 方案 2: 找出LaunchWrapper失败的根本原因（推荐）

调试步骤：
1. 添加详细日志，记录LaunchWrapper失败时的HIP error code
2. 检查为什么会返回非hipSuccess
3. 修复根本原因（可能是context、stream或参数问题）

### 方案 3: 使用LD_PRELOAD方式（备选）

之前测试发现直接链接方式可能有问题，尝试：
```bash
# 不链接libshimhip，运行时用LD_PRELOAD
hipcc ... -lhalhip -lpreempt  # 不链接 libshimhip
export LD_PRELOAD=libshimhip.so
./app_test
```

---

## 📋 下一步行动

### 立即行动（P0）

1. **验证kernels是否执行** 
   - 运行验证程序，检查计算结果
   - 监控GPU使用率

2. **如果kernels确实没执行**
   - 找出LaunchWrapper失败的原因
   - 修复根本问题
   - 重新运行所有测试

3. **如果kernels执行了（但时间异常短）**
   - 理解为什么XSched模式下执行这么快
   - 可能是GPU并行度更高？
   - 需要更深入分析

### 待完成的测试（P1）

在修复问题后，重新运行：
- ✅ Test 1: 单线程基线（已完成，结果正常）
- ✅ Test 2: 16线程无XSched（已完成，结果正常）
- ⏳ Test 3A: 16线程 (3H+13L) with XSched（待修复重测）
- ⏳ Test 3B: 16线程 (1H+15L) with XSched（待修复重测）

---

## 🔬 诊断命令

```bash
# 1. 编译验证程序
hipcc app_verify_execution.hip -lhalhip -lpreempt -lshimhip -o app_verify

# 2. 运行验证
./app_verify

# 3. 检查warning数量
./app_systematic_test 3 2>&1 | grep "Failed to enqueue" | wc -l

# 4. 监控GPU使用率
rocm-smi --showuse  # 在测试运行期间执行
```

---

## 📊 测试日志位置

所有测试日志已保存：
- `/tmp/test1.log` - Test 1 单线程基线
- `/tmp/test2.log` - Test 2 16线程无XSched
- `/tmp/test3.log` - Test 3A (3H+13L) with XSched
- `/tmp/test4.log` - Test 3B (1H+15L) with XSched

---

**报告时间**: 2026-01-29  
**状态**: ⚠️ **需要修复并重新验证**  
**关键问题**: XSched测试中kernels可能未真正执行，导致延迟测量不准确
