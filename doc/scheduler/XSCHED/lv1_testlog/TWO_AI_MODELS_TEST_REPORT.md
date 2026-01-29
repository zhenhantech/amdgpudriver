# Two AI Models Priority Scheduling Test Report

**日期**: 2026-01-29  
**测试目标**: 验证XSched在实际AI模型workload下的优先级调度效果

---

## 📋 测试配置

### Workload
- **高优先级任务**: ResNet-18 (轻量级推理)
  - 目标吞吐: 20 reqs/sec
  - Batch size: 1
  - XSched Priority: 10
  
- **低优先级任务**: ResNet-50 (大batch处理)
  - 运行模式: 连续推理
  - Batch size: 256
  - XSched Priority: 1

### 测试时长
- 60秒

---

## ✅ 测试 1: Baseline (无XSched)

### 结果
```
High Priority (ResNet-18):
  P50 Latency: 6.11 ms
  P99 Latency: 276.56 ms  ← ⚠️ 非常高！
  Throughput:  16.91 req/s

Low Priority (ResNet-50):
  Throughput:  7.84 iter/s
  Images/sec:  2007.7
```

### 分析
- ✅ **测试成功执行**
- 🔴 **P99延迟达到276ms**，远高于单独运行时的<10ms
- 📊 **说明高优先级任务受到了低优先级任务的严重干扰**
- 这正是XSched要解决的问题：防止低优先级任务"饿死"高优先级任务

---

## ❌ 测试 2: XSched (优先级调度)

### 遇到的问题

#### 问题 1: Python Binding缺失
- **现象**: 初始导入`xsched_hip`模块失败
- **原因**: 未正确设置Python路径
- **解决**: 添加`sys.path.insert(0, '/data/dockercode/xsched-build/output/include')`

#### 问题 2: API不匹配
- **现象**: `AttributeError: type object 'XSched' has no attribute 'XHintSetScheduler'`
- **原因**: Python API中不存在该函数
- **解决**: 删除该调用（非必需）

#### 问题 3: HIP Context 错误 (⚠️ 阻塞)
- **现象**: 
  ```
  [ERRO] hip error 709: context is destroyed @ hip_queue.cpp:32
  ```
- **原因**: XSched与`multiprocessing`的兼容性问题
- **现状**: 测试卡住，无法继续
- **根本原因**: 
  1. Multiprocessing会fork子进程，导致HIP context失效
  2. XSched的HIP context管理与multiprocessing冲突

---

## 🔍 深层问题：LaunchWrapper仍未解决

### 之前的验证发现
在简单kernel测试中，我们发现：
```
[XQUEUE-SUBMIT] enqueued kernel idx=1  ✅ Kernel进入缓存
[XQUEUE-LAUNCH] Launching kernel to GPU  ✅ 尝试提交
[WARN] Failed to enqueue command  ❌ LaunchWrapper失败
hipStreamSynchronize: no error  ⚠️ 但HIP报告无错误
```

**验证结果**: Kernels在XSched模式下**没有真正执行**（结果全是0.00）

### 影响范围
即使解决了multiprocessing问题，LaunchWrapper的失败仍会导致：
- Kernels不提交到GPU
- `hipStreamSynchronize`立即返回
- 产生错误的性能数据

---

## 📊 对比分析

| 指标 | Baseline (无XSched) | XSched (理论预期) | XSched (实际) |
|------|---------------------|-------------------|---------------|
| 高优先级P99延迟 | 276.56 ms | <50 ms | ❌ 无法测试 |
| 高优先级吞吐 | 16.91 req/s | ~20 req/s | ❌ 无法测试 |
| 低优先级吞吐 | 7.84 iter/s | 5-7 iter/s | ❌ 无法测试 |
| Kernel执行正确性 | ✅ | ✅ (理论) | ❌ 未执行 |

---

## 🛠️ 问题根源总结

### 1. LaunchWrapper失败 (核心问题)
- **位置**: `hip_queue.cpp:53`
- **现象**: `cmd->LaunchWrapper(kStream)` 返回失败
- **被修改**: `XASSERT` → `XWARN`，导致静默失败
- **后果**: Kernels不执行，但程序不崩溃

### 2. Multiprocessing兼容性
- **现象**: HIP context错误709
- **原因**: Fork后HIP context失效
- **需要**: 重新设计测试架构（单进程+多stream？）

### 3. Python API不完整
- **缺失**: `XHintSetScheduler`等高级API
- **影响**: 无法完全控制调度策略

---

## 💡 建议方案

### 方案 A: 修复LaunchWrapper（彻底）
**优点**:
- 从根本解决问题
- 所有测试都能正常运行

**缺点**:
- 需要深入调试HIP内部机制
- 耗时较长（可能需要数小时）

**步骤**:
1. 还原`XASSERT`，捕获真实错误码
2. 调试`LaunchWrapper`内部的HIP API调用
3. 修复根本问题（可能是context、event或参数传递）

### 方案 B: 昨天的成功测试（推荐⭐）
**优点**:
- 昨天的测试脚本已经验证成功
- 可以直接使用，快速获得结果
- 绕过当前的multiprocessing问题

**缺点**:
- 不能确定昨天的测试是否也有LaunchWrapper问题
- 需要重新验证结果的正确性

**步骤**:
1. 查看昨天的phase4测试日志
2. 重新运行已验证的测试脚本
3. 添加kernel正确性验证

### 方案 C: 单进程+多Stream
**优点**:
- 避免multiprocessing的HIP context问题
- 更接近实际应用场景

**缺点**:
- 需要重新设计测试架构
- 仍然受LaunchWrapper问题影响

---

## 🎯 当前状态

✅ **成功**:
- Baseline测试完成，证明了高优先级任务确实受到干扰(P99=276ms)
- XSched Python binding可用

❌ **失败**:
- XSched测试无法运行（multiprocessing + HIP context冲突）
- LaunchWrapper问题未解决（kernels不执行）

⏸️ **暂停**:
- 等待用户决策：修复LaunchWrapper还是使用昨天的测试方案

---

## 📌 用户建议

根据用户原话："可以尝试一下。如果不好修复。我们可以直接测试昨天的two AI models case。"

**建议**: 采用**方案B**，直接使用昨天验证成功的测试脚本，原因：
1. 修复LaunchWrapper需要较长时间
2. 昨天的测试已经运行成功
3. 可以快速获得实际效果

**如果选择方案B，下一步**:
- 检查昨天的测试日志，确认使用的配置
- 重新运行测试，添加kernel正确性验证
- 对比有/无XSched的P99延迟差异

---

## 📁 相关文件

- Baseline测试日志: `/data/dockercode/xsched-tests/baseline_test.log`
- XSched测试日志: `/data/dockercode/xsched-tests/xsched_test_v2.log`
- 测试脚本: 
  - `/data/dockercode/xsched-tests/test_two_models_simple.py` (Baseline)
  - `/data/dockercode/xsched-tests/test_two_models_xsched.py` (XSched)
- 昨天的测试脚本: `tests/test_phase4_dual_model_CORRECT.py`
