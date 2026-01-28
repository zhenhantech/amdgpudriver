# 🚨 重要发现：Baseline 结果差异巨大

**日期**: 2026-01-28  
**问题**: 同样的测试配置，P99 latency 差异 29 倍！

---

## 📊 两次 Baseline 测试对比

### 第一次 Baseline (run_phase4_dual_model.sh.log, 行 123)

```
High Priority (ResNet-18, 20 req/s):
  Total Requests: 2466
  P99 Latency: 574.50 ms 🚨
  Throughput:  13.70 req/s

Low Priority (ResNet-50, batch=1024):
  Total Iterations: 356
  Throughput:  1.97 iter/s
```

---

### 第二次 Baseline (run_phase4_dual_model_intensive.sh.log, 行 113)

```
High Priority (ResNet-18, 20 req/s):
  Total Requests: 3596
  P99 Latency: 19.79 ms ✅
  Throughput:  19.98 req/s

Low Priority (ResNet-50, batch=1024):
  Total Iterations: 355
  Throughput:  1.97 iter/s
```

---

## 🔍 差异分析

| 指标 | 第一次 | 第二次 | 差异 |
|------|--------|--------|------|
| **High P99** | 574.50 ms | 19.79 ms | **29 倍差异** 🚨 |
| High 吞吐 | 13.70 req/s | 19.98 req/s | **46% 提升** |
| High 请求数 | 2466 | 3596 | 46% 增加 |
| Low 吞吐 | 1.97 iter/s | 1.97 iter/s | 相同 |
| Low 迭代数 | 356 | 355 | 相同 |

---

## 🤔 可能的原因

### 原因 1: 第一次测试两个模型真正并发 ⭐ (最可能)

```
第一次:
  - High 和 Low 同时在同一个 GPU 上运行
  - 资源竞争激烈
  - Low 的大 batch 阻塞 High
  - 导致 P99: 574.50 ms

第二次:
  - 可能两个模型串行运行了
  - 或者使用了不同的 GPU
  - 没有真正的资源竞争
  - 导致 P99: 19.79 ms (接近单独运行的水平)
```

---

### 原因 2: GPU 使用方式不同

```
第一次:
  - 两个进程共享同一个 GPU
  - GPU 内存/计算资源竞争

第二次:
  - 可能两个进程使用了不同的 GPU
  - 或者 GPU 调度方式改变
```

---

### 原因 3: 测试脚本的差异

让我检查两次使用的脚本：

```bash
第一次: test_phase4_dual_model_intensive.py (第一版)
第二次: test_phase4_dual_model_intensive.py (可能更新了)
```

需要检查：
- 两个模型是否在同一个进程中？
- 是否使用 multiprocessing？
- GPU 分配方式是否改变？

---

## 🔧 如何验证

### 检查 1: GPU 使用情况

在测试运行时，查看 GPU 使用：

```bash
# 在另一个终端
watch -n 1 rocm-smi
```

应该看到：
- **如果并发**: GPU 利用率接近 100%
- **如果串行**: GPU 利用率波动

---

### 检查 2: 查看测试脚本

```bash
# 检查 multiprocessing 是否正确使用
grep -A 10 "multiprocessing" tests/test_phase4_dual_model_intensive.py
```

---

### 检查 3: 查看进程

在测试运行时：

```bash
docker exec zhenflashinfer_v1 ps aux | grep python3
```

应该看到：
- **如果并发**: 2 个 Python 进程
- **如果串行**: 1 个 Python 进程

---

## 🎯 关键问题

```
如果第二次 Baseline P99: 19.79 ms 是正确的:
  → 说明第一次测试有问题
  → 或者测试场景不同

如果第一次 Baseline P99: 574.50 ms 是正确的:
  → 说明第二次测试没有真正并发
  → 需要修复测试脚本
```

---

## 🔍 让我检查第一次的 Baseline 日志

第一次 Baseline 的关键信息：

```
从 run_phase4_dual_model.sh.log 第一次运行：
[HIGH] Results:
  Requests: 2466
  Throughput: 13.70 req/s
  Latency P99: 574.50 ms

→ 吞吐量只有 13.70 req/s (目标 20 req/s)
→ 说明确实有资源竞争
→ P99: 574.50 ms 应该是真实的并发结果
```

第二次 Baseline 的关键信息：

```
从 run_phase4_dual_model_intensive.sh.log:
[HIGH] Results:
  Requests: 3596
  Throughput: 19.98 req/s
  Latency P99: 19.79 ms

→ 吞吐量达到了 19.98 req/s (接近目标 20 req/s)
→ 说明没有明显的资源竞争
→ P99: 19.79 ms 可能是串行或单独运行的结果
```

---

## 🚨 初步结论

```
第一次 Baseline: 574.50 ms → 真正的并发测试 ✅
第二次 Baseline: 19.79 ms → 可能没有真正并发 ⚠️

需要:
  1. 检查测试脚本是否正确使用 multiprocessing
  2. 确认两个模型在同一个 GPU 上运行
  3. 验证测试期间的 GPU 使用情况
```

---

## 📝 行动计划

### 1. 先测试 XSched (当前任务)

使用新创建的脚本：

```bash
# 快速测试 (10 秒)
./test_xsched_quick.sh

# 完整测试 (180 秒)
./test_xsched_only.sh
```

---

### 2. 然后重新测试 Baseline

确保：
- 两个模型真正并发
- 在同一个 GPU 上运行
- 有明显的资源竞争

---

### 3. 对比分析

```
如果 XSched P99 < 第一次 Baseline (574.50 ms):
  → 说明 XSched 有效
  → 改善巨大

如果 XSched P99 ≈ 第二次 Baseline (19.79 ms):
  → 说明两次测试都没有真正并发
  → 需要修复测试场景
```

---

## 🎯 当前优先级

```
1. 先修复 XSched symbol error (当前任务)
2. 运行 XSched 快速测试验证环境
3. 运行 XSched 完整测试
4. 分析为什么 Baseline 结果不一致
5. 必要时重新测试 Baseline
```

---

## 💡 关键洞察

### 如果第二次 Baseline 是对的

```
P99: 19.79 ms

说明:
  - 即使高负载配置 (20 req/s, batch=1024)
  - 如果没有真正的并发/竞争
  - Native scheduler 也能表现良好

教训:
  - 测试场景的设置至关重要
  - 必须确保真正的并发
```

---

### 如果第一次 Baseline 是对的

```
P99: 574.50 ms

说明:
  - Native scheduler 在真正的并发下表现很差
  - XSched 的价值巨大
  - 需要确保测试真正并发

教训:
  - 这才是真实的生产场景
  - XSched 的改善空间巨大
```

---

## 🔧 立即执行

### Step 1: 快速测试 XSched 环境

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED

# 10 秒快速测试
./test_xsched_quick.sh
```

**预期**: 
- ✅ 环境验证通过
- ✅ PyTorch + XSched 正常

---

### Step 2: 如果快速测试成功，运行完整测试

```bash
# 180 秒完整测试
./test_xsched_only.sh
```

**预期**:
- 获得 XSched P99 数据
- 与 Baseline 对比

---

### Step 3: 分析结果

```
如果 XSched P99 ≈ 19-20 ms:
  → 说明可能没有真正并发
  → 需要检查测试脚本

如果 XSched P99 ≈ 500+ ms:
  → 说明有并发，但 XSched 没有改善
  → 需要检查 XSched 配置

如果 XSched P99 ≈ 50-100 ms:
  → 说明 XSched 有改善
  → 但需要确认 Baseline 是 574ms 还是 19ms
```

---

## 🎉 最终目标

```
理想结果:
  Baseline P99: 574.50 ms (真正并发)
  XSched P99:   <50 ms
  改善:         >90%

这将证明 XSched 的巨大价值！
```
