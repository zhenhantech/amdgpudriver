# 运行高负载测试（快速指南）

**更新**: 2026-01-28  
**测试类型**: Intensive (高负载)

---

## ⚡ 快速开始

### 立即运行

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED

# 运行高负载测试（约 6-7 分钟）
./run_phase4_dual_model_intensive.sh
```

**预计时间**: 6-7 分钟（Baseline 3分钟 + XSched 3分钟）

---

## 📊 新配置说明

### 配置对比

| 参数 | 原配置 | 新配置（高负载）|
|------|--------|----------------|
| 高优先级请求率 | 10 req/s | **20 req/s** (2x) |
| 高优先级间隔 | 100ms | **50ms** (0.5x) |
| 低优先级 batch | 8 | **1024** (128x) |
| 测试时长 | 60s | **180s** (3x) |

### 为什么改配置？

```
1. 更高请求率（20 req/s）
   → 更具挑战性的在线场景
   → 验证 XSched 在高负载下的能力

2. 超大 batch（1024）
   → 低优先级占用更多 GPU 资源
   → 测试 XSched 的抢占能力

3. 更长时间（3 分钟）
   → 更多数据点（~3600 个请求）
   → 更稳定的 P99 统计
```

---

## 🎯 测试步骤

### Step 1: 检查 GPU 内存

```bash
# 确保有足够的 GPU 内存（至少 6-7 GB）
docker exec zhenflashinfer_v1 rocm-smi
```

**需要**: ~6-7 GB GPU 内存  
**如果不够**: 减小 batch size（见下文）

---

### Step 2: 运行测试

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED

# 完整测试
./run_phase4_dual_model_intensive.sh
```

测试会自动运行：
1. ✅ Baseline 测试（3 分钟）
2. ✅ XSched 测试（3 分钟）
3. ✅ 结果对比
4. ✅ 生成报告

---

### Step 3: 查看结果

```bash
# 方法 1: 查看终端输出
# 测试完成后会自动显示对比结果

# 方法 2: 查看 JSON 结果
docker exec zhenflashinfer_v1 cat /data/dockercode/test_results_phase4/baseline_intensive_result.json
docker exec zhenflashinfer_v1 cat /data/dockercode/test_results_phase4/xsched_intensive_result.json

# 方法 3: 查看 Markdown 报告
docker exec zhenflashinfer_v1 cat /data/dockercode/test_results_phase4/phase4_dual_model_intensive_report.md
```

---

## 📈 预期结果

### 场景 A: XSched 继续保持优势

```
High Priority P99:
  Baseline: ~5-10 ms
  XSched:   ~3-6 ms
  改善:     ~20-40% ✅

Low Priority Throughput:
  影响: ~10% ✅

结论: XSched 在高负载下仍然有效
```

---

### 场景 B: 性能有所下降但仍有改善

```
High Priority P99:
  Baseline: ~10-20 ms
  XSched:   ~8-15 ms
  改善:     ~20% ✅

Low Priority Throughput:
  影响: ~20-30%

结论: XSched 有帮助，但高负载是挑战
```

---

### 场景 C: GPU 饱和

```
High Priority P99:
  Baseline: >>20 ms
  XSched:   ~类似
  改善:     最小

结论: 负载过高，需要优化配置
```

---

## ⚙️ 调整配置（如果需要）

### 减小 Batch Size

如果遇到 OOM (Out of Memory):

```bash
# 编辑测试脚本
docker exec -it zhenflashinfer_v1 bash
cd /data/dockercode
vi test_phase4_dual_model_intensive.py

# 修改这一行（第 127 行左右）:
batch_size = 1024  # 改为 512 或 256
```

---

### 减少测试时间

如果 3 分钟太长:

```bash
# 运行时指定更短的时间
docker exec zhenflashinfer_v1 bash -c '
  cd /data/dockercode && \
  export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:$LD_LIBRARY_PATH && \
  export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so && \
  python3 test_phase4_dual_model_intensive.py --duration 120
'
```

---

### 降低请求率

如果 20 req/s 太高:

```bash
# 编辑脚本，修改第 69 行左右:
sleep_time = max(0, 0.05 - elapsed)  # 改为 0.1 (10 req/s)
```

---

## 🔧 故障排查

### 问题 1: OOM 错误

```
错误: RuntimeError: CUDA out of memory

解决:
  1. 减小 batch size: 1024 → 512 → 256
  2. 关闭其他 GPU 进程
  3. 重启 Docker 容器
```

---

### 问题 2: 测试卡住

```
现象: 低优先级任务不输出进度

解决:
  1. 检查 GPU 状态: rocm-smi
  2. 减小 batch size
  3. Ctrl+C 停止并重试
```

---

### 问题 3: 性能异常低

```
现象: P99 latency 异常高 (>100ms)

可能原因:
  1. GPU 被其他进程占用
  2. 系统负载过高
  3. 配置参数不合理

解决:
  1. 检查系统负载: top, htop
  2. 检查 GPU 使用: rocm-smi
  3. 重启测试
```

---

## 📊 与原测试对比

### 原测试结果（已完成）

```
配置: 10 req/s, batch=8, 60s

High Priority:
  Baseline P99: 3.47 ms
  XSched P99:   2.75 ms
  改善:         -20.9% ✅

Low Priority:
  Baseline:     165.40 iter/s
  XSched:       163.54 iter/s
  影响:         -1.1% ✅
```

---

### 新测试（待运行）

```
配置: 20 req/s, batch=1024, 180s

High Priority:
  Baseline P99: ?
  XSched P99:   ?
  改善:         ?

Low Priority:
  Baseline:     ?
  XSched:       ?
  影响:         ?

关键问题:
  - XSched 的改善是否持续？
  - 高负载是否导致性能下降？
  - 大 batch 是否影响抢占？
```

---

## 📂 相关文件

### 测试脚本

```
tests/test_phase4_dual_model_intensive.py  # Python 测试脚本
run_phase4_dual_model_intensive.sh         # Bash 运行脚本
```

### 文档

```
INTENSIVE_TEST_CONFIG.md    # 详细配置说明
PHASE4_TEST3_PRINCIPLE.md   # 测试原理（已更新）
RUN_INTENSIVE_TEST.md       # 本文档（快速指南）
```

### 结果文件

```
/data/dockercode/test_results_phase4/
  ├─ baseline_intensive_result.json
  ├─ xsched_intensive_result.json
  └─ phase4_dual_model_intensive_report.md
```

---

## ✅ 检查清单

运行测试前，确认：

- [ ] Docker 容器 `zhenflashinfer_v1` 正在运行
- [ ] GPU 有至少 6-7 GB 可用内存
- [ ] 有至少 10 分钟时间（包括结果分析）
- [ ] 已阅读配置说明

运行测试：

- [ ] `cd` 到正确目录
- [ ] 执行 `./run_phase4_dual_model_intensive.sh`
- [ ] 等待测试完成（~6-7 分钟）
- [ ] 查看结果和对比

分析结果：

- [ ] 检查 P99 latency 的变化
- [ ] 检查低优先级吞吐量的影响
- [ ] 对比原测试结果
- [ ] 记录关键发现

---

## 🎯 成功标准

### 最低标准

```
✅ 测试成功完成（无崩溃）
✅ 有 baseline 和 XSched 结果
✅ 可以生成对比报告
```

### 理想标准

```
✅ XSched P99 latency < Baseline × 1.1
✅ 低优先级 throughput > Baseline × 0.7
✅ 无 GPU 内存溢出
✅ 测试稳定可重复
```

---

## 🚀 立即开始

```bash
# 一键运行
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED
./run_phase4_dual_model_intensive.sh

# 预计时间: 6-7 分钟
# 预期输出: 详细的对比结果和报告
```

---

**准备好了？开始测试！** 🚀

**有问题？** 查看 [INTENSIVE_TEST_CONFIG.md](INTENSIVE_TEST_CONFIG.md) 获取详细说明。
