# XSched MI308X 测试 - 文档索引

**更新日期**: 2026-01-28  
**当前阶段**: Phase 4（多模型优先级调度）

---

## 🚀 快速开始（新用户看这里）

### 想立即开始测试？

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED

# Step 1: 验证环境（2 分钟）
./run_phase4_test1.sh

# Step 2: 双模型测试（3 分钟）
./run_phase4_dual_model.sh
```

**推荐阅读顺序**:
1. [PHASE4_QUICKSTART.md](PHASE4_QUICKSTART.md) ← 从这里开始
2. [PHASE1_TO_3_SUMMARY.md](PHASE1_TO_3_SUMMARY.md) ← 了解背景
3. [PHASE4_CORE_OBJECTIVES.md](PHASE4_CORE_OBJECTIVES.md) ← 深入目标

---

## 📚 按主题查找文档

### 我想了解项目背景

| 文档 | 用途 |
|------|------|
| [README.md](README.md) | 项目总览和文档索引 |
| [PHASE1_TO_3_SUMMARY.md](PHASE1_TO_3_SUMMARY.md) | Phase 1-3 完成总结 ⭐ |
| [PHASE3_TEST_RESULTS.md](PHASE3_TEST_RESULTS.md) | Phase 3 详细测试结果 |
| [PHASE3_LOG_SUMMARY.md](PHASE3_LOG_SUMMARY.md) | Phase 3 日志摘要 |

---

### 我想开始 Phase 4 测试

| 文档 | 用途 |
|------|------|
| [PHASE4_QUICKSTART.md](PHASE4_QUICKSTART.md) | 快速开始（推荐）⭐ |
| [PHASE4_PROGRESS.md](PHASE4_PROGRESS.md) | 实时进度追踪 ⭐ |
| [PHASE4_TEST1_RESULTS.md](PHASE4_TEST1_RESULTS.md) | Test 1 结果 (PASSED) ⭐ |
| [PHASE4_TEST3_RESULTS.md](PHASE4_TEST3_RESULTS.md) | Test 3 结果 (PASSED, P99↓20.9%) 🎉 |
| [PHASE4_READY.md](PHASE4_READY.md) | 准备就绪检查 |
| [PHASE4_CORE_OBJECTIVES.md](PHASE4_CORE_OBJECTIVES.md) | 核心目标和测试场景 |
| [PHASE4_OVERVIEW.md](PHASE4_OVERVIEW.md) | 完整总览和计划 |

---

### 我想了解测试方案对比

| 文档 | 用途 |
|------|------|
| [PLAN_COMPARISON.md](PLAN_COMPARISON.md) | 论文方案 vs 实际方案 ⭐ |
| [XSched_MI308X测试方案_基于论文Ch7Ch8.md](XSched_MI308X测试方案_基于论文Ch7Ch8.md) | 原论文测试方案 |
| [XSched_MI308X_REALISTIC_TEST_PLAN.md](XSched_MI308X_REALISTIC_TEST_PLAN.md) | 实际可行方案 |

---

### 我想了解 Docker 使用

| 文档 | 用途 |
|------|------|
| [DOCKER_USAGE.md](DOCKER_USAGE.md) | Docker 完整使用指南 ⭐ |
| [DOCKER_READY.md](DOCKER_READY.md) | Docker 脚本准备就绪 |

---

### 我想查看技术细节

| 文档 | 用途 |
|------|------|
| [QUICKSTART.md](QUICKSTART.md) | 方案选择指南 |
| Phase 1-3 的技术文档（在 flashinfer/dockercode/xsched/） | |
| - SOLUTION_SYMBOL_VERSIONING.md | Symbol Versioning 详解 |
| - VICTORY_2026-01-28.md | 三大 Bug 修复报告 |
| - PHASE1_SUMMARY.md | Phase 1 总结 |

---

## 📂 文件结构

```
XSCHED/
├── INDEX.md                              ← 本文档（文档导航）
├── README.md                             ← 项目总览
│
├── Phase 1-3 总结
│   ├── PHASE1_TO_3_SUMMARY.md            ← 综合总结 ⭐
│   ├── PHASE3_TEST_RESULTS.md            ← Phase 3 详细结果
│   └── PHASE3_LOG_SUMMARY.md             ← Phase 3 日志摘要
│
├── Phase 4 文档
│   ├── PHASE4_QUICKSTART.md              ← 快速开始 ⭐
│   ├── PHASE4_READY.md                   ← 准备就绪
│   ├── PHASE4_CORE_OBJECTIVES.md         ← 核心目标
│   └── PHASE4_OVERVIEW.md                ← 完整总览
│
├── 测试方案对比
│   ├── PLAN_COMPARISON.md                ← 方案对比 ⭐
│   ├── XSched_MI308X测试方案_基于论文Ch7Ch8.md
│   └── XSched_MI308X_REALISTIC_TEST_PLAN.md
│
├── Docker 使用
│   ├── DOCKER_USAGE.md                   ← 使用指南 ⭐
│   └── DOCKER_READY.md
│
├── 测试脚本
│   ├── run_phase4_test1.sh               ← 验证环境
│   ├── run_phase4_dual_model.sh          ← 双模型测试
│   └── tests/
│       ├── test_phase4_1_verify_existing.sh
│       ├── test_phase4_dual_model.py
│       └── ...
│
└── 其他
    ├── QUICKSTART.md
    └── ...
```

---

## 🎯 按角色推荐

### 新用户（第一次使用）

**推荐路径**:
1. [README.md](README.md) - 了解项目
2. [PHASE1_TO_3_SUMMARY.md](PHASE1_TO_3_SUMMARY.md) - 了解已完成工作
3. [PHASE4_QUICKSTART.md](PHASE4_QUICKSTART.md) - 开始测试
4. [DOCKER_USAGE.md](DOCKER_USAGE.md) - 了解 Docker 用法

---

### 研究人员（想复现论文）

**推荐路径**:
1. [XSched_MI308X测试方案_基于论文Ch7Ch8.md](XSched_MI308X测试方案_基于论文Ch7Ch8.md) - 论文方案
2. [PLAN_COMPARISON.md](PLAN_COMPARISON.md) - 理解差异
3. [PHASE1_TO_3_SUMMARY.md](PHASE1_TO_3_SUMMARY.md) - 当前进展
4. [PHASE4_CORE_OBJECTIVES.md](PHASE4_CORE_OBJECTIVES.md) - 下一步

---

### 工程师（想集成到生产）

**推荐路径**:
1. [PHASE3_TEST_RESULTS.md](PHASE3_TEST_RESULTS.md) - 验证可行性
2. [PHASE4_QUICKSTART.md](PHASE4_QUICKSTART.md) - 立即测试
3. [PHASE4_CORE_OBJECTIVES.md](PHASE4_CORE_OBJECTIVES.md) - 生产场景
4. [DOCKER_USAGE.md](DOCKER_USAGE.md) - 部署指南

---

### 调试人员（遇到问题）

**推荐路径**:
1. [PHASE1_TO_3_SUMMARY.md](PHASE1_TO_3_SUMMARY.md) - 了解已知问题
2. [PHASE3_LOG_SUMMARY.md](PHASE3_LOG_SUMMARY.md) - 查看日志示例
3. [DOCKER_USAGE.md](DOCKER_USAGE.md) - 问题排查
4. flashinfer/dockercode/xsched/ 下的技术文档

---

## 📊 项目状态一览

### 已完成 ✅

```
Phase 1: Bug Fixes              ✅ 3/3 (100%)
Phase 2: AI Models              ✅ 7/7 (100%)
Phase 3: Real Models            ✅ 13/14 (92.9%)

总计: 22/23 通过 (95.7%)
```

### 当前阶段 🔄

```
Phase 4: Multi-Model Priority Scheduling
  ├─ 测试脚本准备就绪              ✅
  ├─ 文档完善                      ✅
  ├─ Test 1: 环境验证              ✅ PASSED (09:05)
  ├─ Test 2: Runtime Overhead      ⏳ 待运行
  └─ Test 3: 双模型优先级          ✅ PASSED (09:06) 🎉

进度: 2/3+ 测试完成 (66%+)
亮点: Test 3 P99 latency ↓20.9% 🚀
```

---

## 🎯 最常用命令

```bash
# 查看 Phase 3 测试结果
cat PHASE3_TEST_RESULTS.md

# 了解 Phase 4 目标
cat PHASE4_CORE_OBJECTIVES.md

# 立即开始 Phase 4
./run_phase4_test1.sh

# 查看 Phase 3 日志
docker exec zhenflashinfer_v1 \
  grep -E "(Testing |✅|❌)" \
  /data/dockercode/xsched/testlog/TEST_REAL_MODELS.sh.log
```

---

## 🔗 外部链接

### XSched 官方
- GitHub: https://github.com/XpuOS/xsched
- 论文: XSched: Preemptive Scheduling for Diverse XPUs (OSDI 2025)

### 相关项目
- PyTorch ROCm: https://pytorch.org/
- AMD ROCm: https://rocm.docs.amd.com/

---

## 💡 文档说明

### 文档类型

**摘要类** (快速了解):
- INDEX.md (本文档)
- README.md
- PHASE4_QUICKSTART.md
- PHASE4_READY.md

**详细类** (深入理解):
- PHASE1_TO_3_SUMMARY.md
- PHASE4_CORE_OBJECTIVES.md
- PHASE4_OVERVIEW.md

**参考类** (学术/长期):
- XSched_MI308X测试方案_基于论文Ch7Ch8.md
- XSched_MI308X_REALISTIC_TEST_PLAN.md

**技术类** (实施细节):
- DOCKER_USAGE.md
- PHASE3_LOG_SUMMARY.md
- PLAN_COMPARISON.md

---

## 🎓 学习路径

### 路径 1: 快速实践（推荐）

```
1. INDEX.md (本文档)
   ↓
2. PHASE4_QUICKSTART.md
   ↓
3. 运行 ./run_phase4_test1.sh
   ↓
4. 根据需要查阅其他文档
```

### 路径 2: 系统学习

```
1. README.md (项目总览)
   ↓
2. PHASE1_TO_3_SUMMARY.md (背景)
   ↓
3. PLAN_COMPARISON.md (方案对比)
   ↓
4. PHASE4_CORE_OBJECTIVES.md (目标)
   ↓
5. PHASE4_QUICKSTART.md (实践)
```

### 路径 3: 学术研究

```
1. XSched_MI308X测试方案_基于论文Ch7Ch8.md (论文方案)
   ↓
2. PLAN_COMPARISON.md (对比分析)
   ↓
3. PHASE1_TO_3_SUMMARY.md (当前进展)
   ↓
4. XSched_MI308X_REALISTIC_TEST_PLAN.md (实际方案)
```

---

## 📞 需要帮助？

### 找不到想要的信息？

```
想了解 Phase 1-3 做了什么？
  → PHASE1_TO_3_SUMMARY.md

想看 Phase 3 的测试结果？
  → PHASE3_TEST_RESULTS.md

想开始 Phase 4 测试？
  → PHASE4_QUICKSTART.md

想了解两个测试方案的区别？
  → PLAN_COMPARISON.md

想在 Docker 里运行？
  → DOCKER_USAGE.md
```

---

**文档总数**: 15+ 个  
**核心文档**: 6 个（标记 ⭐）  
**可执行脚本**: 10+ 个

**立即开始**: `./run_phase4_test1.sh` 🚀
