# Phase 4 Test 3 高负载配置

**更新日期**: 2026-01-28  
**配置类型**: Intensive (高负载)

---

## 📊 新配置 vs 原配置

### 原始配置（已完成）

```
高优先级 (ResNet-18):
  - 吞吐: 10 req/s (100ms 间隔)
  - Batch: 1
  - 时长: 60 秒

低优先级 (ResNet-50):
  - Batch: 8
  - 模式: 连续
  - 时长: 60 秒

测试结果:
  ✅ High P99: 3.47ms → 2.75ms (-20.9%)
  ✅ Low throughput: 165.40 → 163.54 iter/s (-1.1%)
```

---

### 新配置（高负载）⭐

```
高优先级 (ResNet-18):
  - 吞吐: 20 req/s (50ms 间隔) ← 2x 负载
  - Batch: 1
  - 时长: 180 秒 (3 分钟) ← 3x 时长

低优先级 (ResNet-50):
  - Batch: 1024 ← 128x batch size
  - 模式: 连续
  - 时长: 180 秒 (3 分钟)

预期影响:
  - 高优先级：更高请求频率，更具挑战性
  - 低优先级：超大 batch，GPU 占用更重
  - 测试时长：3 倍，更稳定的统计
```

---

## 🎯 测试目标

### 1. 验证高负载下的调度能力

```
原测试: 10 req/s（相对轻松）
新测试: 20 req/s（更具挑战性）

问题:
  - XSched 在高负载下能否保持优势？
  - P99 latency 能否继续降低？
  - 是否会出现饱和现象？
```

---

### 2. 验证大 Batch Size 场景

```
原测试: batch=8（轻量级批处理）
新测试: batch=1024（超大批处理）

问题:
  - 低优先级任务占用更多 GPU 资源
  - 高优先级任务能否及时插入？
  - XSched 的抢占能力如何？
```

---

### 3. 更长时间的稳定性测试

```
原测试: 60 秒（基础测试）
新测试: 180 秒（3 分钟）

优势:
  - 更多数据点（~3600 vs ~600 个请求）
  - 更稳定的统计（P99 更可靠）
  - 验证长时间运行的稳定性
```

---

## 🚀 运行新测试

### 方法 1: 使用包装脚本（推荐）

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED

# 运行高负载测试（约 6-7 分钟）
./run_phase4_dual_model_intensive.sh
```

**预计时间**:
- Baseline: 3 分钟
- XSched: 3 分钟
- 对比和报告: <1 分钟
- **总计**: ~6-7 分钟

---

### 方法 2: 单独运行 Baseline

```bash
docker exec zhenflashinfer_v1 bash -c '
  cd /data/dockercode && \
  unset LD_PRELOAD && \
  python3 test_phase4_dual_model_intensive.py \
    --duration 180 \
    --output /tmp/baseline_intensive.json
'
```

---

### 方法 3: 单独运行 XSched

```bash
docker exec zhenflashinfer_v1 bash -c '
  cd /data/dockercode && \
  export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:$LD_LIBRARY_PATH && \
  export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so && \
  python3 test_phase4_dual_model_intensive.py \
    --duration 180 \
    --output /tmp/xsched_intensive.json
'
```

---

### 方法 4: 快速验证（1 分钟）

如果只想快速验证配置是否可行:

```bash
docker exec zhenflashinfer_v1 bash -c '
  cd /data/dockercode && \
  export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:$LD_LIBRARY_PATH && \
  export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so && \
  python3 test_phase4_dual_model_intensive.py \
    --duration 60 \
    --output /tmp/xsched_intensive_quick.json
'
```

---

## 📊 预期结果

### 场景 1: XSched 继续保持优势

```
如果 XSched 在高负载下仍然有效:

High Priority P99:
  Baseline: ~5-10 ms（可能更高）
  XSched:   ~3-6 ms
  改善:     ~20-40%

Low Priority Throughput:
  Baseline: ~X iter/s
  XSched:   ~0.9X iter/s
  影响:     ~10%
```

---

### 场景 2: 高负载下性能下降

```
如果高负载导致性能下降:

High Priority P99:
  Baseline: ~10-20 ms
  XSched:   ~8-15 ms
  改善:     ~20%（仍有改善，但绝对值更高）

Low Priority Throughput:
  明显下降（可能 <50%）
```

---

### 场景 3: GPU 饱和

```
如果 GPU 已经饱和:

High Priority P99:
  Baseline: >>20 ms
  XSched:   ~类似
  改善:     最小

结论: 负载过高，需要减少请求率或增加 GPU
```

---

## ⚠️ 注意事项

### 1. GPU 内存

```
Batch=1024 需要大量 GPU 内存:
  - ResNet-50: ~4-6 GB
  - ResNet-18: ~0.5 GB
  - 总计: ~5-7 GB

检查 GPU 内存:
  docker exec zhenflashinfer_v1 rocm-smi
```

如果内存不足，可能需要减小 batch size:
- 尝试 512 或 256

---

### 2. 测试时长

```
180 秒 = 3 分钟

如果时间太长，可以调整:
  --duration 120  # 2 分钟
  --duration 90   # 1.5 分钟
```

---

### 3. 日志输出

```
低优先级任务会每 10 次迭代报告进度:
  [LOW] Progress: 10 iterations, 2.34 iter/s
  [LOW] Progress: 20 iterations, 2.45 iter/s
  ...

这是正常的，用于监控进度
```

---

## 📈 结果分析

### 关键指标对比

运行完成后，对比以下指标:

#### 高优先级 (ResNet-18)

```
原配置 (10 req/s):
  Baseline P99: 3.47 ms
  XSched P99:   2.75 ms
  改善:         -20.9%

新配置 (20 req/s):
  Baseline P99: ?
  XSched P99:   ?
  改善:         ?

问题:
  - P99 是否显著增加？
  - XSched 的改善幅度是否保持？
  - 是否出现新的瓶颈？
```

---

#### 低优先级 (ResNet-50)

```
原配置 (batch=8):
  Baseline: 165.40 iter/s
  XSched:   163.54 iter/s
  影响:     -1.1%

新配置 (batch=1024):
  Baseline: ?
  XSched:   ?
  影响:     ?

问题:
  - 大 batch 是否显著降低吞吐量？
  - XSched 的影响是否更大？
  - 是否出现饿死现象？
```

---

## 🔍 故障排查

### 问题 1: OOM (Out of Memory)

```
错误信息:
  RuntimeError: CUDA out of memory

解决:
  1. 减小 batch size
     修改 test_phase4_dual_model_intensive.py:
     batch_size = 512  # 或 256

  2. 检查 GPU 内存
     docker exec zhenflashinfer_v1 rocm-smi
```

---

### 问题 2: 测试太慢

```
如果 180 秒太长:
  ./run_phase4_dual_model_intensive.sh
  
  然后在脚本中修改 --duration 参数
  或直接运行:
  
  python3 test_phase4_dual_model_intensive.py --duration 60
```

---

### 问题 3: 进程卡住

```
如果低优先级任务卡住:
  - 可能是 batch size 太大
  - 检查 GPU 是否响应: rocm-smi
  - 考虑减小到 batch=512
```

---

## 📂 生成的文件

测试完成后，会生成以下文件:

```
/data/dockercode/test_results_phase4/
  ├─ baseline_intensive_result.json      # Baseline 结果
  ├─ xsched_intensive_result.json        # XSched 结果
  └─ phase4_dual_model_intensive_report.md  # 对比报告
```

---

## 🎯 下一步

### 1. 运行测试

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED
./run_phase4_dual_model_intensive.sh
```

---

### 2. 查看结果

```bash
# 查看 Baseline 结果
docker exec zhenflashinfer_v1 cat /data/dockercode/test_results_phase4/baseline_intensive_result.json

# 查看 XSched 结果
docker exec zhenflashinfer_v1 cat /data/dockercode/test_results_phase4/xsched_intensive_result.json

# 查看对比报告
docker exec zhenflashinfer_v1 cat /data/dockercode/test_results_phase4/phase4_dual_model_intensive_report.md
```

---

### 3. 分析和记录

- 对比原配置 vs 新配置
- 记录 P99 latency 的变化
- 评估 XSched 在高负载下的表现
- 更新 Phase 4 文档

---

## 📊 配置对比表

| 参数 | 原配置 | 新配置 | 倍数 |
|------|--------|--------|------|
| 高优先级请求率 | 10 req/s | 20 req/s | 2x |
| 高优先级间隔 | 100ms | 50ms | 0.5x |
| 低优先级 batch | 8 | 1024 | 128x |
| 测试时长 | 60s | 180s | 3x |
| 总请求数 | ~600 | ~3600 | 6x |

---

## 🎉 期待的发现

### 最佳情况

```
XSched 在高负载下仍然表现优异:
  ✅ P99 latency 显著降低
  ✅ 低优先级几乎不受影响
  ✅ 证明了 XSched 的鲁棒性
```

### 现实情况

```
XSched 仍有改善，但幅度可能变小:
  ✅ P99 latency 有所降低
  ✅ 低优先级受到一定影响
  ✅ 仍证明了 XSched 的价值
```

### 学习点

```
无论结果如何，我们都能学到:
  - XSched 的性能边界
  - 高负载下的调度挑战
  - 优化的方向
```

---

**准备好了！运行测试**: `./run_phase4_dual_model_intensive.sh` 🚀
