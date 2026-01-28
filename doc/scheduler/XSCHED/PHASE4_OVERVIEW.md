# Phase 4: XSched Paper Tests Overview

**日期**: 2026-01-28  
**状态**: 基于 Phase 2 完成的环境

---

## 📊 项目阶段总览

```
Phase 1: PyTorch Bug Fixes                    ✅ 完成
  ├─ Bug #1: import torch 挂起               ✅
  ├─ Bug #2: tensor.cuda() 挂起              ✅
  └─ Bug #3: torch.matmul 失败 (Symbol Ver)  ✅

Phase 2: AI Models Testing                    ✅ 完成
  ├─ MLP, CNN, Transformer                   ✅
  ├─ Multi-Head Attention                    ✅
  ├─ Forward + Backward                      ✅
  └─ Mixed Precision (FP16)                  ✅

Phase 3: Real Models Testing                  ✅ 完成 (92.9%)
  ├─ 13/14 真实模型测试通过                  ✅
  ├─ ResNet, MobileNet, EfficientNet, ViT   ✅
  ├─ Training (Forward+Backward)            ✅
  └─ Batch Processing (16, 32)              ✅
  📊 详细结果: PHASE3_TEST_RESULTS.md

Phase 4: XSched Paper Tests                   🔄 当前阶段
  ├─ Test 1: Verify Existing XSched          ✅ PASSED (09:05)
  ├─ Test 2: Runtime Overhead (7.4.1)        ⏳ 待运行
  ├─ Test 3: Dual Model Priority             ✅ PASSED (09:06) 🎉
  ├─ Test 4: Multi-Tenant Scenario           ⏳ 待设计
  └─ ...更多论文测试...
  📊 Test 1: PHASE4_TEST1_RESULTS.md
  📊 Test 3: PHASE4_TEST3_RESULTS.md (P99 ↓20.9%)
```

---

## 🎯 Phase 4 的核心区别

### ❌ 不需要重新编译

**Phase 4 使用已有环境**:
```bash
# 已有的 XSched（Phase 2 编译）
/data/dockercode/xsched-official      ← 源码
/data/dockercode/xsched-build         ← 编译输出
/data/dockercode/xsched-build/output  ← 安装目录

# 已验证的功能
✅ libhalhip.so, libshimhip.so 编译完成
✅ Symbol Versioning 修复（hip_version.map）
✅ PyTorch 集成成功
✅ 7 种 AI 模型测试通过
✅ 真实模型（ResNet, ViT, etc.）测试通过
```

### ✅ Phase 4 的任务

**基于已有环境进行论文验证**:
1. 验证已有 XSched 的功能
2. 测量 Runtime Overhead（对比 baseline）
3. 测试优先级调度策略
4. 测量抢占延迟（Lv1）
5. 复现论文实验

---

## 📋 Phase 4 测试计划（简化版）

基于 `XSched_MI308X测试方案_基于论文Ch7Ch8.md`，但利用已有环境。

### Test 1: Verify Existing XSched ✅ PASSED

**目标**: 验证 Phase 2 的 XSched 安装

```bash
./run_phase4_test1.sh
```

**验证项**:
- ✅ XSched 源码和编译输出存在 (Git commit: ff5298c)
- ✅ 库文件正确安装 (libhalhip.so: 252K, libshimhip.so: 412K)
- ✅ Symbol Versioning 生效 (hipMalloc@@hip_4.2)
- ✅ PyTorch 集成正常工作 (PyTorch 2.7.1+rocm6.4.1)

**实际执行时间**: 14 秒  
**测试结果**: ✅ **PASSED** (2026-01-28 09:05:23)  
**详细报告**: [PHASE4_TEST1_RESULTS.md](PHASE4_TEST1_RESULTS.md)

---

### Test 2: Runtime Overhead（论文 7.4.1）

**目标**: 测量 XSched 的运行时开销

**测试方法**:
```python
# 1. Baseline: 不使用 XSched
unset LD_PRELOAD
python test_resnet.py  # 记录时间 T_baseline

# 2. With XSched
export LD_PRELOAD=.../libshimhip.so
python test_resnet.py  # 记录时间 T_xsched

# 3. 计算开销
Overhead = (T_xsched - T_baseline) / T_baseline * 100%
```

**成功标准**:
- ✅ Overhead < 10% (宽松)
- 🎯 Overhead < 3.4% (论文目标)

**工作量**: 利用 Phase 3 的测试脚本（`BENCHMARK.sh`）

---

### Test 3: Fixed Priority（论文 7.2.1）

**目标**: 验证优先级调度

**测试方法**:
```python
# 两个进程：
# Process 1: 高优先级（前台任务）
# Process 2: 低优先级（后台任务）

# 测量：
# - 前台任务 P99 延迟
# - 后台任务吞吐量
```

**成功标准**:
- ✅ 前台 P99 延迟 < 1.30× standalone
- ✅ 优于 Native scheduler

---

### Test 4: Preemption Latency（论文 7.3.1）

**目标**: 测量 Lv1 抢占延迟

**测试方法**:
```cpp
// 模拟场景：
// 1. 低优先级任务持续运行
// 2. 高优先级任务周期性到达
// 3. 测量抢占延迟

P99_latency = measure_preemption();
```

**成功标准**:
- ✅ Lv1 P99 延迟 ≈ 8T (T=0.5ms, 约 4ms)
- 📊 记录数据，与论文对比

---

## 🚀 立即开始

### Step 1: 验证已有环境（2 分钟）

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED

# 运行 Phase 4 Test 1
./run_phase4_test1.sh
```

**预期输出**:
```
✅ Phase 4 Test 1 PASSED

Verified XSched Installation:
  Source:  /data/dockercode/xsched-official
  Build:   /data/dockercode/xsched-build
  Install: /data/dockercode/xsched-build/output

Key Features Verified:
  ✅ Libraries compiled and installed
  ✅ Symbol versioning (Phase 2 fix)
  ✅ PyTorch integration working
```

### Step 2: Runtime Overhead（30 分钟）

```bash
# 使用 Phase 3 的 BENCHMARK.sh
cd /mnt/md0/zhehan/code/flashinfer/dockercode/xsched
./BENCHMARK.sh

# 或创建 Phase 4 的简化版本
./run_phase4_test2.sh  # 待创建
```

### Step 3: 后续测试

根据 Test 1 和 Test 2 的结果，决定是否继续：
- 如果开销 < 10%，继续 Test 3-4
- 如果开销过大，分析原因

---

## 📂 文件组织

### Phase 4 专用文件

```
XSCHED/
├── PHASE4_OVERVIEW.md                   ← 本文档
├── run_phase4_test1.sh                  ← Test 1 运行脚本
├── run_phase4_test2.sh                  ← Test 2（待创建）
├── tests/
│   ├── test_phase4_1_verify_existing.sh ← Test 1 脚本
│   ├── test_phase4_2_overhead.sh        ← Test 2（待创建）
│   └── ...
└── test_results_phase4/                 ← Phase 4 结果目录
    └── phase4_test1_report.json
```

### 复用 Phase 3 的文件

```
/mnt/md0/zhehan/code/flashinfer/dockercode/xsched/
├── BENCHMARK.sh                         ← 性能测试
├── TEST_REAL_MODELS.sh                  ← 真实模型测试
├── setup.sh                             ← 环境设置
└── ...
```

---

## 🔄 与原测试方案的关系

### 原方案（论文测试方案）

**文件**: `XSched_MI308X测试方案_基于论文Ch7Ch8.md`

**特点**:
- 完整覆盖论文所有测试
- 从零开始（假设 XSched 未编译）
- 包含 CWSR Lv3 集成

### Phase 4 方案（实际执行）

**特点**:
- 基于 Phase 2 已有环境
- 不重新编译
- 选择性测试（优先级高的先做）
- CWSR Lv3 作为未来工作

**关系**:
```
原方案 = Phase 4 的参考和长期目标
Phase 4 = 原方案的简化和渐进实现
```

---

## 💡 为什么不重新编译？

### 原因 1: Phase 2 已完成编译

```bash
# Phase 2 的成果
✅ XSched 源码：/data/dockercode/xsched-official
✅ 编译输出：/data/dockercode/xsched-build
✅ 关键修复：Symbol Versioning (hip_version.map)
✅ 验证通过：PyTorch + 7 种 AI 模型
```

### 原因 2: 重新编译会遇到问题

```bash
# 克隆新的 xsched-test 会遇到：
❌ 缺少 CLI11 子模块
❌ 缺少 Symbol Versioning 修复
❌ 需要重新应用所有 Phase 2 的修改
```

### 原因 3: 时间和资源浪费

```bash
重新编译：10-15 分钟
验证已有：2 分钟

节省时间：13 分钟 × 每次测试 = 大量时间
```

---

## 🎯 Phase 4 的目标

### 短期目标（本周）

1. ✅ 验证已有 XSched 环境
2. 📊 测量 Runtime Overhead
3. 🔍 初步测试优先级调度

### 中期目标（2 周）

1. 完成论文 Chapter 7 的核心测试
2. 对比论文数据
3. 记录 MI308X 的性能特征

### 长期目标（1-2 月）

1. 复现论文 Chapter 8 的 Case Studies
2. CWSR Lv3 集成（独立项目）
3. 技术报告或论文发表

---

## 📊 预期成果

### 对比论文数据

| 指标 | 论文值（MI50） | Phase 4 目标（MI308X） |
|------|---------------|----------------------|
| Runtime Overhead | 1.7% | < 3.4% |
| CPU Overhead | 3.6% | < 5% |
| Fixed Priority P99 | 1.30× | < 1.30× |
| Lv1 Preemption (T=0.5ms) | ~4ms | ~4ms |

### 独特贡献

- ✅ PyTorch + XSched 集成（论文未涉及）
- ✅ Symbol Versioning 修复（社区贡献）
- 📊 MI308X 性能数据（新硬件）
- ⏭️ CWSR Lv3 集成（未来，超越论文）

---

## 🚀 立即开始

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED

# Phase 4 Test 1: 验证环境（2 分钟）
./run_phase4_test1.sh

# 查看结果
docker exec zhenflashinfer_v1 cat /data/dockercode/test_results_phase4/phase4_test1_report.json
```

**预期**: ✅ PASS（因为 Phase 2 已完成）

---

**Phase 4 = 基于已有成果 + 论文验证** 🚀
