# XSched MI308X Testing - Quick Start

**文档**: 基于实际可行测试方案  
**日期**: 2026-01-28

---

## 📚 文档概览

### 1. **原测试方案** (基于论文)
- 📄 `XSched_MI308X测试方案_基于论文Ch7Ch8.md`
- **特点**: 完整覆盖论文所有实验，非常详细
- **问题**: 假设 XSched 已完全可用，缺少基础验证步骤，Lv3 CWSR 作为测试一部分

### 2. **实际测试方案** (推荐)
- 📄 `XSched_MI308X_REALISTIC_TEST_PLAN.md`
- **特点**: 从编译开始，逐步递进，利用已完成的 PyTorch 工作
- **优势**: 立即可执行，避免理想化假设，CWSR Lv3 作为独立项目

---

## 🚀 立即开始

### 选项 A: 直接运行第一个测试（推荐）

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED

# 使权限可执行
chmod +x tests/*.sh

# 运行第一个测试
./tests/test_1_1_compilation.sh
```

**预期时间**: 5-10 分钟  
**预期结果**: XSched 成功编译并安装

### 选项 B: 查看对比分析

```bash
# 对比原方案和实际方案
cat XSched_MI308X_REALISTIC_TEST_PLAN.md | grep "与原方案的对比" -A 20
```

---

## 📊 测试路线图

### Stage 0: PyTorch Foundation ✅ (已完成)
我们已经完成了：
- Bug 修复（Symbol Versioning, Static Init, etc.）
- 基础 AI 模型测试（MLP, CNN, Transformer）
- 真实模型测试（ResNet, MobileNet, EfficientNet, ViT）
- 性能基准测试框架

### Stage 1: XSched Baseline (本方案起点)
```
今天可完成:
  ✅ Test 1.1: 编译安装 (1 小时)
  ✅ Test 1.2: 官方示例运行 (2 小时)
  ✅ Test 1.3: 基础 API 验证 (2 小时)
  
Total: ~1 天
```

### Stage 2: Scheduling Policies
```
下周目标:
  ⏳ Test 2.1: 优先级调度 (1 天)
  
Total: ~2 天
```

### Stage 3: Performance
```
2-3 周目标:
  ⏳ Test 3.1: 运行时开销 (4 小时)
  ⏳ Test 3.2: 抢占延迟 (1 周)
  
Total: ~2 周
```

### Stage 4: Real Workloads
```
结合已完成工作:
  ⏳ Test 4.1: PyTorch 集成
  ⏳ Test 4.2: 多进程场景
  
Total: ~1 周
```

### Stage 5: CWSR Lv3 (Future)
```
独立项目 (4-6 周):
  - KFD ioctl 调用
  - Wavefront save/restore
  - XSched Lv3 接口实现
  - 性能优化
```

---

## 🎯 关键调整说明

### 1. 阶段命名冲突解决

**原方案**: Phase 1-5（与 PyTorch 集成的 Phase 冲突）  
**新方案**: Stage 0-5（独立命名体系）

```
PyTorch Integration Timeline:
  Phase 1: Bug Fixes ✅
  Phase 2: AI Models ✅
  Phase 3: Production (进行中)

XSched Testing Timeline (独立):
  Stage 0: PyTorch Foundation ✅
  Stage 1: Baseline Verification (当前)
  Stage 2-5: 后续测试
```

### 2. CWSR Lv3 定位调整

**原方案**: 作为测试的一部分 (Test 4.1.1)  
**新方案**: 独立项目，不阻塞基础测试

**理由**:
- CWSR 集成需要深入内核开发
- 不是简单的 ioctl 调用
- 需要 4-6 周专门投入
- 不应阻塞基础功能验证

### 3. 现实性调整

**原方案假设**:
- ✅ XSched 可直接运行
- ✅ 所有依赖工具可用（Triton ROCm, Paella, etc.）
- ✅ 复杂 workload 可立即测试

**新方案现实**:
- ❓ 需要先验证编译安装
- ❓ 官方示例是否能运行
- ❓ 基础 API 是否正常工作
- ✅ 利用已验证的 PyTorch 环境

### 4. 工作复用

**新方案充分利用已完成工作**:
```python
# 我们已经有：
✅ PyTorch 2.9.1 + ROCm 6.4
✅ Symbol Versioning 修复
✅ ResNet, MobileNet, EfficientNet 等模型测试通过
✅ TEST.sh, TEST_AI_MODELS.sh, TEST_REAL_MODELS.sh

# 新方案利用：
Stage 4.1: 直接在 XSched 环境下运行这些测试
  export LD_PRELOAD=.../libshimhip.so
  ./TEST_REAL_MODELS.sh
```

---

## 📈 预期时间线

### 保守估计

| 阶段 | 时间 | 累计 |
|------|------|------|
| Stage 1: Baseline | 1 天 | 1 天 |
| Stage 2: Scheduling | 2 天 | 3 天 |
| Stage 3: Performance | 2 周 | ~3 周 |
| Stage 4: Real Workloads | 1 周 | ~4 周 |
| **Total (基础测试)** | **~1 月** | |
| Stage 5: CWSR Lv3 (独立) | 4-6 周 | - |

### 乐观估计

如果一切顺利：
- Stage 1: 4-6 小时（今天完成）
- Stage 2: 1 天
- Stage 3-4: 1-2 周

---

## 🔍 问题诊断

### 如果 Test 1.1 失败？

**编译错误**:
```bash
# 查看详细日志
cat /data/dockercode/xsched-test-build/build_output.log | tail -50

# 常见问题：
# 1. ROCm 路径
# 2. 编译器版本
# 3. CMake 配置
```

**解决方案**:
1. 检查 ROCm 环境
2. 查看 XSched GitHub Issues
3. 参考 `xsched-official` 的成功编译经验

### 如果官方示例运行失败？

**可能原因**:
1. LD_LIBRARY_PATH 未设置
2. HIP 版本不兼容
3. XSched 未正确拦截 API

**调试**:
```bash
export XLOG_LEVEL=DEBUG
export LD_PRELOAD=.../libshimhip.so
./app
```

---

## 💡 建议

### 对于原测试方案的使用

**保留原方案的价值**:
- ✅ 作为**参考基准**：对标论文实验
- ✅ 作为**长期目标**：最终要复现的内容
- ✅ 作为**技术文档**：详细的测试设计

**但不作为执行计划**:
- ❌ 不从这里开始测试
- ❌ 不假设前置条件已满足
- ❌ 不把 Lv3 作为基础测试一部分

### 推荐的工作流程

```
1. 阅读实际方案 (REALISTIC_TEST_PLAN.md) ← 开始这里
   ↓
2. 运行 Stage 1 测试 (tests/test_1_1_compilation.sh)
   ↓
3. 根据结果调整
   ↓
4. 逐步推进到 Stage 2, 3, 4
   ↓
5. 参考原方案进行详细对比和补充
   ↓
6. 如果一切顺利，考虑 Stage 5 (CWSR Lv3) 作为独立项目
```

---

## 📞 下一步

### 立即行动

```bash
# 1. 进入测试目录
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED

# 2. 准备环境
mkdir -p test_results
chmod +x tests/*.sh

# 3. 运行第一个测试
./tests/test_1_1_compilation.sh

# 4. 查看结果
cat test_results/test_1_1_report.json
```

### 如果成功

继续创建后续测试脚本：
- `test_1_2_native_examples.sh`
- `test_1_3_api_coverage.cpp`

### 如果遇到问题

1. 查看日志
2. 参考 XSched 官方文档
3. 利用我们在 `xsched-official` 的编译经验

---

**准备好开始了吗？**

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED
./tests/test_1_1_compilation.sh
```

预期时间：10 分钟  
预期结果：✅ XSched successfully compiled and installed
