# XSched on AMD MI308X - Testing Documentation

**项目**: XSched 在 AMD MI308X 上的测试与验证  
**日期**: 2026-01-28  
**状态**: 📋 测试方案设计完成，准备执行

---

## 📚 文档索引

### Phase 1-4 文档（当前阶段）⭐

| 文档 | 描述 | 推荐度 |
|------|------|--------|
| **[PHASE1_TO_3_SUMMARY.md](PHASE1_TO_3_SUMMARY.md)** | Phase 1-3 完成总结（必读）| ⭐⭐⭐⭐⭐ |
| **[PHASE3_TEST_RESULTS.md](PHASE3_TEST_RESULTS.md)** | Phase 3 详细结果 (13/14) | ⭐⭐⭐⭐⭐ |
| **[PHASE3_LOG_SUMMARY.md](PHASE3_LOG_SUMMARY.md)** | Phase 3 日志摘要 | ⭐⭐⭐⭐ |
| **[PHASE4_PROGRESS.md](PHASE4_PROGRESS.md)** | Phase 4 进度追踪（实时更新）| ⭐⭐⭐⭐⭐ |
| **[PHASE4_TEST1_RESULTS.md](PHASE4_TEST1_RESULTS.md)** | Test 1 环境验证 (PASSED) | ⭐⭐⭐⭐⭐ |
| **[PHASE4_TEST3_RESULTS.md](PHASE4_TEST3_RESULTS.md)** | Test 3 双模型 (PASSED, P99↓20.9%) 🎉 | ⭐⭐⭐⭐⭐ |
| **[PHASE4_TEST3_PRINCIPLE.md](PHASE4_TEST3_PRINCIPLE.md)** | Test 3 测试原理详解 | ⭐⭐⭐⭐ |
| **[TEST3_QUICK_ANSWER.md](TEST3_QUICK_ANSWER.md)** | Test 3 快速问答 | ⭐⭐⭐⭐ |
| **[PHASE4_QUICKSTART.md](PHASE4_QUICKSTART.md)** | Phase 4 快速开始 | ⭐⭐⭐⭐⭐ |
| **[PHASE4_CORE_OBJECTIVES.md](PHASE4_CORE_OBJECTIVES.md)** | Phase 4 核心目标 | ⭐⭐⭐⭐⭐ |
| **[PHASE4_OVERVIEW.md](PHASE4_OVERVIEW.md)** | Phase 4 总览 | ⭐⭐⭐⭐ |

### 基础文档

| 文档 | 描述 | 推荐度 | 适用场景 |
|------|------|--------|---------|
| **[DOCKER_USAGE.md](DOCKER_USAGE.md)** | Docker 使用指南 | ⭐⭐⭐⭐⭐ | **Docker 环境运行测试** |
| **[QUICKSTART.md](QUICKSTART.md)** | 快速开始指南 | ⭐⭐⭐⭐⭐ | **立即开始，从这里入手** |
| **[PLAN_COMPARISON.md](PLAN_COMPARISON.md)** | 方案对比分析 | ⭐⭐⭐⭐⭐ | 理解两个方案的区别 |
| [XSched_MI308X_REALISTIC_TEST_PLAN.md](XSched_MI308X_REALISTIC_TEST_PLAN.md) | 实际可行测试方案 | ⭐⭐⭐⭐⭐ | **执行测试的主要依据** |
| [XSched_MI308X测试方案_基于论文Ch7Ch8.md](XSched_MI308X测试方案_基于论文Ch7Ch8.md) | 基于论文的测试方案 | ⭐⭐⭐⭐ | 学术参考，长期目标 |

### 测试脚本

| 脚本 | 测试内容 | 预计时间 | 状态 |
|------|---------|---------|------|
| [tests/test_1_1_compilation.sh](tests/test_1_1_compilation.sh) | 编译安装验证 | 10 分钟 | ✅ 已创建 |
| `tests/test_1_2_native_examples.sh` | 官方示例运行 | 30 分钟 | ⏳ 待创建 |
| `tests/test_1_3_api_coverage.cpp` | 基础 API 验证 | 1 小时 | ⏳ 待创建 |
| `tests/test_2_1_fixed_priority.cpp` | 优先级调度 | 1 天 | ⏳ 待创建 |
| `tests/test_3_1_runtime_overhead.sh` | 运行时开销 | 4 小时 | ⏳ 待创建 |
| `tests/test_4_1_pytorch_integration.sh` | PyTorch 集成 | 2 小时 | ⏳ 待创建 |

---

## 🎯 项目目标

### 短期目标（1-2 周）

✅ **验证 XSched 在 MI308X 上的基本可行性**

```
Stage 1: Baseline Verification
  ├─ 编译安装成功
  ├─ 官方示例运行
  └─ 基础 API 正常工作
```

### 中期目标（1-2 月）

⏳ **完成核心功能测试**

```
Stage 2-4: Functional Testing
  ├─ 优先级调度验证
  ├─ 性能开销测量
  ├─ PyTorch 集成测试
  └─ 真实工作负载验证
```

### 长期目标（3-6 月）

⏭️ **复现论文实验 + CWSR Lv3**

```
Stage 5: Advanced Features
  ├─ 复现论文 Chapter 7 & 8 的所有实验
  ├─ CWSR Lv3 集成（独立项目）
  └─ 性能优化与生产化
```

---

## 🚀 Quick Start

### 🎯 Phase 4: 多模型优先级调度（当前阶段）⭐⭐⭐

**核心目标**: 验证 XSched 的多 AI 模型优先级调度和 latency 保证

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED

# Test 1: 验证环境 ✅ PASSED (09:05)
./run_phase4_test1.sh

# Test 3: 双模型优先级 ✅ PASSED (09:06) 🎉
./run_phase4_dual_model.sh

# Test 3b: 双模型高负载（新配置）⭐
./run_phase4_dual_model_intensive.sh  # 20 req/s, batch=1024, 3min

# Test 2: Runtime Overhead（待运行）
./run_phase4_test2.sh  # 即将可用

# 查看结果
docker exec zhenflashinfer_v1 cat /data/dockercode/test_results_phase4/phase4_dual_model_report.md
```

**最新状态**: ✅ **2/3+ 测试通过 (66%+)**  
**Test 3 亮点**: 🎉 **P99 latency 降低 20.9%，XSched 优于 Native!**  
**Test 3b**: ⏳ **高负载配置准备就绪**（20 req/s, batch=1024）

**详细结果**: 
- [PHASE4_TEST1_RESULTS.md](PHASE4_TEST1_RESULTS.md) - Test 1: 环境验证
- [PHASE4_TEST3_RESULTS.md](PHASE4_TEST3_RESULTS.md) - Test 3: 双模型优先级 ⭐
- [RUN_INTENSIVE_TEST.md](RUN_INTENSIVE_TEST.md) - Test 3b: 高负载快速指南 ⭐⭐

**详细指南**: 阅读 [PHASE4_QUICKSTART.md](PHASE4_QUICKSTART.md)

---

### 🐳 从零开始（如果从未运行过）

```bash
# 1. 阅读 Docker 使用指南
cat DOCKER_USAGE.md

# 2. 一键运行测试
./run_test_1_1.sh

# 3. 查看结果
docker exec zhenflashinfer_v1 cat /data/dockercode/test_results/test_1_1_report.json
```

**预期时间**: 10-15 分钟  
**预期结果**: ✅ XSched successfully compiled and installed

---

### 📚 如果你是第一次接触

```bash
# 1. 先了解 Docker 环境
cat DOCKER_USAGE.md

# 2. 阅读快速开始指南
cat QUICKSTART.md

# 3. 理解两个方案的区别
cat PLAN_COMPARISON.md

# 4. 运行第一个测试
./run_test_1_1.sh
```

### 🎓 如果你熟悉 XSched

```bash
# 直接查看实际方案
cat XSched_MI308X_REALISTIC_TEST_PLAN.md

# 或参考论文方案
cat XSched_MI308X测试方案_基于论文Ch7Ch8.md

# 在 Docker 内直接运行
docker exec -it zhenflashinfer_v1 bash /data/dockercode/test_1_1_compilation_docker.sh
```

---

## 📊 两个方案概览

### 方案 A: 基于论文的测试方案

**文件**: `XSched_MI308X测试方案_基于论文Ch7Ch8.md`

**特点**:
- ✅ 完整覆盖论文 Chapter 7 & 8 的所有实验
- ✅ 详细的测试设计和性能指标
- ✅ 包含 Case Studies (Multi-tenant, Video conferencing, Inference serving)
- ⚠️  假设 XSched 已可用，直接开始复杂测试
- ⚠️  CWSR Lv3 作为测试的一部分

**适用场景**:
- 学术研究，需要复现论文
- 长期目标和路线图参考
- 详细的测试设计灵感

### 方案 B: 实际可行测试方案（推荐）

**文件**: `XSched_MI308X_REALISTIC_TEST_PLAN.md`

**特点**:
- ✅ 从编译安装开始，逐步验证
- ✅ 利用已完成的 PyTorch 集成工作
- ✅ CWSR Lv3 作为独立项目，不阻塞基础测试
- ✅ 最小化外部依赖
- ✅ 立即可执行

**适用场景**:
- 工程实施，立即开始测试
- 从零开始，没有假设
- 需要快速验证可行性

### 对比总结

| 维度 | 论文方案 | 实际方案 |
|------|---------|---------|
| **起点** | 假设 XSched 可用 | 从编译开始 |
| **复杂度** | 高（直接用复杂 workload） | 渐进（先简单后复杂） |
| **时间** | 10 周（理想） | 1 月（保守） |
| **Lv3 CWSR** | 测试一部分 | 独立项目 |
| **依赖** | Triton, Paella 等 | 最小化 |
| **推荐度** | 参考 ⭐⭐⭐⭐ | 执行 ⭐⭐⭐⭐⭐ |

---

## 🎓 使用建议

### 推荐流程

```
1. 阅读 QUICKSTART.md
   ↓
2. 理解 PLAN_COMPARISON.md
   ↓
3. 执行 REALISTIC_TEST_PLAN.md
   ├─ Stage 1: 今天
   ├─ Stage 2: 下周
   ├─ Stage 3-4: 2-3 周
   └─ Stage 5: 未来
   ↓
4. 参考论文方案，进行详细对比
   ├─ 对标论文数据
   ├─ 补充缺失测试
   └─ 复现 Case Studies
   ↓
5. CWSR Lv3 集成（独立项目）
```

### 根据角色选择

**研究人员** 🎓:
- 目标：发表论文
- 路径：实际方案（打基础）→ 论文方案（复现）→ 技术报告
- 重点：Stage 1-4 (基础) + 论文所有测试

**工程师** 🔧:
- 目标：生产集成
- 路径：实际方案 → Stage 4 PyTorch 集成 → 性能优化
- 重点：Stage 1, 3, 4（跳过复杂的 Case Studies）

**学生** 📚:
- 目标：学习理解
- 路径：论文方案（阅读）→ 实际方案（实践）→ 对比分析
- 重点：理解设计思想 > 完成所有测试

---

## 📈 项目状态

### 已完成 ✅

**Phase 1-2: PyTorch 集成** (2026-01-27 ~ 01-28):
- Bug Fixes (Symbol Versioning, Static Init, etc.)
- 基础 AI 模型测试（7/7 通过，100%）
- 测试框架（TEST.sh, TEST_AI_MODELS.sh）

**Phase 3: Real Models Testing** (2026-01-28 上午):
- 真实模型测试（13/14 通过，92.9%）
- 训练测试（2/2 通过，100%）
- 批处理测试（2/2 通过，100%）
- 详细报告：[PHASE3_TEST_RESULTS.md](PHASE3_TEST_RESULTS.md)

**Phase 4: Paper Tests** (2026-01-28):
- 论文方案文档完成
- 实际方案文档完成
- 对比分析完成
- Test 1: 环境验证 ✅ **PASSED** (09:05)
- Test 3: 双模型优先级 ✅ **PASSED** (09:06) 🎉 P99↓20.9%
- Test 3b: 双模型高负载 ⏳ **准备运行** (20 req/s, batch=1024, 3min)

### 进行中 🔄

**XSched Baseline Verification** (Stage 1):
- Test 1.1: 编译安装（脚本已创建，待运行）
- Test 1.2: 官方示例（待创建）
- Test 1.3: API 验证（待创建）

### 待开始 ⏳

- Stage 2: Scheduling Policies
- Stage 3: Performance Characterization
- Stage 4: Real Workload Integration
- Stage 5: CWSR Lv3 (独立项目)

---

## 🔗 相关资源

### XSched 官方

- **GitHub**: https://github.com/XpuOS/xsched
- **Artifacts**: https://github.com/XpuOS/xsched-artifacts
- **论文**: XSched: Preemptive Scheduling for Diverse XPUs (OSDI 2025)

### AMD 文档

- **CWSR 机制**: `/mnt/md0/zhehan/code/rampup_doc/GPREEMPT_MI300_Testing/CWSR机制简要总结.md`
- **ROCm 文档**: https://rocm.docs.amd.com/
- **MI300 系列**: https://www.amd.com/en/products/accelerators/instinct/mi300

### 相关项目

- **PyTorch ROCm**: https://pytorch.org/
- **Triton Inference Server**: https://github.com/triton-inference-server
- **Rodinia Benchmark**: https://github.com/AMDComputeLibraries/Rodinia_HIP

---

## 📞 获取帮助

### 文档导航

```
不知道从哪里开始？
  ↓
QUICKSTART.md ← 从这里

想理解两个方案的区别？
  ↓
PLAN_COMPARISON.md ← 看这里

准备执行测试？
  ↓
REALISTIC_TEST_PLAN.md ← 执行这个

需要学术参考？
  ↓
论文方案.md ← 参考这个
```

### 常见问题

**Q: 应该用哪个方案？**  
A: 从实际方案开始，论文方案作为参考。

**Q: CWSR Lv3 要做吗？**  
A: 可以做，但作为独立项目，不阻塞基础测试。

**Q: 论文方案有问题吗？**  
A: 没有！非常优秀，但假设了一些前提条件。实际方案从更基础的步骤开始。

**Q: 需要多长时间？**  
A: 基础测试 (Stage 1-4): 约 1 月；完整复现: 3-6 月；CWSR Lv3: 额外 4-6 周。

---

## 🎯 立即开始

```bash
# 进入项目目录
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED

# 阅读快速开始
cat QUICKSTART.md

# 运行第一个测试
./tests/test_1_1_compilation.sh

# 查看结果
cat test_results/test_1_1_report.json
```

**预期**: 10-15 分钟后，XSched 编译安装成功 ✅

---

**准备好了吗？开始第一个测试！** 🚀
