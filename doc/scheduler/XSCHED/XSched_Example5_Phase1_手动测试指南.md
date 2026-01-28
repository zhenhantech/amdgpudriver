# XSched Example 5 - Phase 1: BERT 推理基线测试 - 手动验证指南

**目标**: 手动运行 Phase 1 基线测试，验证 BERT 推理性能  
**平台**: AMD Instinct MI308X (Docker `zhenaiter`)  
**预计时间**: 15-20 分钟

---

## 📋 测试前准备

### 1. 进入 Docker 容器

```bash
# 如果在容器外
docker exec -it zhenaiter bash

# 或者直接附加到容器
docker attach zhenaiter
```

### 2. 激活 ROCm+PyTorch 环境

```bash
# 设置 micromamba 环境变量
export MAMBA_EXE='/root/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/root/micromamba'

# 初始化 micromamba
eval "$(/root/.local/bin/micromamba shell hook --shell=bash)"

# 激活 flashinfer-rocm 环境
micromamba activate flashinfer-rocm
```

### 3. 验证环境

```bash
# 检查 PyTorch 版本
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# 期望输出: PyTorch: 2.9.1+rocm6.4

# 检查 GPU 可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# 期望输出: CUDA available: True

# 检查 GPU 数量和型号
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0)}')"
# 期望输出: 
# GPU count: 8
# GPU name: AMD Instinct MI308X
```

### 4. 检查 transformers 库

```bash
# 检查是否已安装
python -c "import transformers; print(f'transformers: {transformers.__version__}')"

# 如果未安装，执行：
pip install transformers
```

---

## 🚀 运行测试

### 步骤 1: 进入测试目录

```bash
cd /workspace
```

### 步骤 2: 检查测试脚本是否存在

```bash
ls -lh test_multi_priority_bert_simplified.py
```

**如果文件不存在**，说明需要从主机复制：

```bash
# 在主机上执行（在另一个终端）
docker cp /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/code/xsched/examples/Linux/test_multi_priority_bert_simplified.py zhenaiter:/workspace/
```

### 步骤 3: 运行完整测试（推荐）

```bash
# 运行测试，使用 bert-base-uncased 模型（更快）
python test_multi_priority_bert_simplified.py --model bert-base-uncased --requests 20
```

**预计运行时间**: 2-3 分钟（首次运行需下载模型，约 400MB）

**预期输出示例**:
```
================================================================================
Environment Check
================================================================================
PyTorch version: 2.9.1+rocm6.4
CUDA available: True
GPU count: 8
GPU name: AMD Instinct MI308X
GPU memory: 192.00 GB
================================================================================

Loading BERT model: bert-base-uncased
Downloading model files... (首次运行)
Model loaded successfully!
Warming up GPU...
Warmup complete!

================================================================================
TEST 1: Baseline Performance (Single-threaded)
================================================================================
[Baseline] Starting 20 requests...
[Baseline] Progress: 10/20, Last 10 avg: 6.38ms
[Baseline] Progress: 20/20, Last 10 avg: 6.37ms

============================================================
Baseline - Statistics
============================================================
  Mean:   6.37 ms
  Median: 6.37 ms
  P95:    6.40 ms
  P99:    6.42 ms
  Min:    6.35 ms
  Max:    6.51 ms
  Std:    0.03 ms
============================================================

...（后续还有 TEST 2 和 TEST 3）...

================================================================================
SUMMARY: All Tests Completed
================================================================================
Test 1 (Baseline):
  Mean Latency: 6.37 ms

Test 2 (Concurrent - No Priority):
  Task-A: 12.34 ms (P99: 18.47 ms)
  Task-B: 13.63 ms (P99: 18.58 ms)
  Task-C: 13.63 ms (P99: 18.52 ms)

Test 3 (Sequential - Simulated Priority):
  HIGH Priority: 6.40 ms (P99: 6.49 ms)
  LOW Priority:  6.39 ms (P99: 6.45 ms)
```

### 步骤 4: 运行简化测试（快速验证）

如果只想快速验证环境，可以减少请求数：

```bash
# 只运行 10 个请求，更快完成
python test_multi_priority_bert_simplified.py --model bert-base-uncased --requests 10
```

**预计运行时间**: 30-60 秒

---

## 📊 测试结果解读

### Test 1: Baseline（单线程）

**含义**: 测量单个推理任务的最佳性能（无竞争）

**关键指标**:
- **Mean Latency**: 应该在 **6-7ms** 左右
- **Std Dev**: 应该很小（< 0.1ms），说明性能稳定

**如果延迟过高（> 10ms）**:
- 检查是否有其他进程占用 GPU
- 确认使用的是 MI308X 而非其他 GPU
- 检查 ROCm 驱动是否正常

### Test 2: Concurrent（并发无优先级）

**含义**: 测量 3 个任务并发执行时的性能竞争

**关键指标**:
- **Mean Latency**: 应该在 **12-15ms** 左右（比 baseline 增加 2 倍）
- **P99 Latency**: 应该在 **18-20ms** 左右
- **三个任务延迟相似**: 说明没有优先级区分

**如果三个任务延迟差异很大**:
- 这是正常的，说明 GPU 调度有一定随机性
- 但平均值应该接近

### Test 3: Sequential（顺序执行）

**含义**: 测量顺序执行（无竞争）的性能

**关键指标**:
- **Mean Latency**: 应该接近 **6-7ms**（与 baseline 相同）
- **高低优先级延迟相同**: 因为是顺序执行，无竞争

**意义**: 这是 Phase 2 的理想目标 - XSched 应该让高优先级任务在并发场景下也达到这种性能

---

## 🔍 故障排查

### 问题 1: 找不到 transformers 模块

```bash
# 错误信息
ModuleNotFoundError: No module named 'transformers'

# 解决方案
pip install transformers
```

### 问题 2: GPU 不可用

```bash
# 错误信息
CUDA available: False

# 检查 ROCm
rocm-smi

# 检查环境变量
echo $HIP_VISIBLE_DEVICES

# 重新激活环境
micromamba deactivate
micromamba activate flashinfer-rocm
```

### 问题 3: 内存不足

```bash
# 错误信息
RuntimeError: CUDA out of memory

# 解决方案1: 清理 GPU 缓存
python -c "import torch; torch.cuda.empty_cache()"

# 解决方案2: 使用更小的模型
python test_multi_priority_bert_simplified.py --model bert-base-uncased --requests 10
```

### 问题 4: 模型下载失败

```bash
# 错误信息
HTTPError: 404 Client Error

# 解决方案: 设置镜像源（如果在中国）
export HF_ENDPOINT=https://hf-mirror.com

# 或者手动下载模型
python -c "from transformers import BertForQuestionAnswering; BertForQuestionAnswering.from_pretrained('bert-base-uncased')"
```

### 问题 5: 测试脚本不存在

```bash
# 在主机上（不是容器内）复制脚本
docker cp /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/code/xsched/examples/Linux/test_multi_priority_bert_simplified.py zhenaiter:/workspace/

# 在容器内验证
ls -lh /workspace/test_multi_priority_bert_simplified.py
```

---

## 📈 性能基准参考

### 预期性能（MI308X）

| 测试 | Mean Latency | P99 Latency | 说明 |
|------|-------------|-------------|------|
| **Test 1 (Baseline)** | 6-7ms | 6-7ms | 单线程最佳性能 |
| **Test 2 (Concurrent)** | 12-15ms | 18-20ms | 3 任务并发 |
| **Test 3 (Sequential)** | 6-7ms | 6-7ms | 顺序执行 |

### 与其他平台对比

| 平台 | BERT-Base 单次推理 | 说明 |
|------|-------------------|------|
| **MI308X (本测试)** | **6.37ms** | CDNA 3 架构 ⭐⭐⭐⭐⭐ |
| **NVIDIA A100** | ~10ms | Ampere 架构 |
| **NVIDIA GV100** | ~15ms | Volta 架构 |
| **MI100** | ~8-10ms | CDNA 1 架构 |

---

## 🎯 验证成功标准

### ✅ 测试成功的标志

1. **环境检查通过**
   - PyTorch 版本: 2.9.1+rocm6.4
   - GPU 检测: 8× MI308X
   - transformers 库已安装

2. **Test 1 结果正常**
   - Mean latency < 8ms
   - 标准差很小 (< 0.1ms)

3. **Test 2 结果正常**
   - Mean latency 比 Test 1 增加 1.5-2.5 倍
   - 3 个任务延迟相近

4. **Test 3 结果正常**
   - Mean latency 接近 Test 1
   - 高低优先级延迟相同

### ❌ 需要检查的情况

1. **Test 1 延迟过高** (> 15ms)
   - 检查 GPU 型号
   - 检查其他进程占用
   - 检查 ROCm 驱动

2. **Test 2 延迟没有增加**
   - 可能请求间隔太大，没有竞争
   - 减少 `delay_ms` 参数

3. **程序崩溃或错误**
   - 检查错误日志
   - 参考故障排查章节

---

## 📝 测试日志

### 保存测试输出

```bash
# 运行测试并保存日志
python test_multi_priority_bert_simplified.py --model bert-base-uncased --requests 20 2>&1 | tee bert_test_output_manual.log

# 查看日志
less bert_test_output_manual.log

# 只查看关键统计信息
grep -A 10 "Statistics" bert_test_output_manual.log
```

### 提取关键指标

```bash
# 提取所有 Mean Latency
grep "Mean:" bert_test_output_manual.log

# 提取所有 P99 Latency
grep "P99:" bert_test_output_manual.log

# 查看测试摘要
grep -A 20 "SUMMARY" bert_test_output_manual.log
```

---

## 🔗 相关文档

- **详细报告**: [XSched_Example5_Phase1_基线测试报告.md](./XSched_Example5_Phase1_基线测试报告.md)
- **适配方案**: [XSched_Example5_推理服务测试分析与AMD适配方案.md](./XSched_Example5_推理服务测试分析与AMD适配方案.md)
- **项目进度**: [XSched_Example5_项目进度总结.md](./XSched_Example5_项目进度总结.md)

---

## 💡 下一步

完成 Phase 1 验证后，可以继续进行：

1. **Phase 2: XSched 集成**
   - 集成 XSched C API
   - 实现真正的多优先级调度
   - 对比性能差异

2. **性能调优**
   - 调整模型参数
   - 优化并发策略
   - 测试不同负载

3. **扩展测试**
   - 测试 BERT-Large 模型
   - 测试更多并发任务
   - 测试不同的请求模式

---

**文档版本**: v1.0  
**创建日期**: 2026-01-27  
**最后更新**: 2026-01-27  
**状态**: ✅ 准备就绪

