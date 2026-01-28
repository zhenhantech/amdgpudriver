# XSched MI308X 测试方案对比分析

**日期**: 2026-01-28  
**目的**: 对比论文方案 vs 实际方案

---

## 📊 两个方案概览

| 方案 | 文档 | 适用场景 | 优先级 |
|------|------|---------|--------|
| **论文方案** | `XSched_MI308X测试方案_基于论文Ch7Ch8.md` | 学术复现，长期目标 | 参考 |
| **实际方案** | `XSched_MI308X_REALISTIC_TEST_PLAN.md` | 工程实施，立即执行 | **推荐** |

---

## 🔍 详细对比

### 1. 测试起点

| 维度 | 论文方案 | 实际方案 |
|------|---------|---------|
| **假设前提** | XSched 已编译安装，官方示例可运行 | 从零开始，先验证编译 |
| **第一个测试** | Test 7.1.1: 运行并测量开销 | Test 1.1: 编译安装验证 |
| **现实性** | ⚠️  理想化 | ✅ 实际可行 |

**示例对比**:

```bash
# 论文方案 Test 7.1.1 (第一个测试)
cd /workspace/xsched/examples/Linux/1_transparent_sched
./app  # 直接运行，假设已编译好

# 实际方案 Test 1.1 (第一个测试)
git clone https://github.com/XpuOS/xsched.git
cd xsched && mkdir build && cd build
cmake .. -DXSCHED_PLATFORM=hip
make -j$(nproc)  # 先验证能否编译
```

**结论**: 实际方案从更基础的步骤开始，避免假设。

---

### 2. 阶段命名

| 维度 | 论文方案 | 实际方案 |
|------|---------|---------|
| **命名** | Phase 1-5 | Stage 0-5 |
| **冲突** | ❌ 与 PyTorch Phase 冲突 | ✅ 独立命名体系 |
| **清晰度** | ⚠️  容易混淆 | ✅ 清晰区分 |

**背景**:
我们在 PyTorch 集成中已经使用了 "Phase 1-3"：
- Phase 1: Bug Fixes ✅
- Phase 2: AI Models ✅  
- Phase 3: Production (进行中)

如果 XSched 测试也用 "Phase 1-5"，会造成混淆。

**解决方案**: 实际方案使用 "Stage 0-5"，清晰独立。

---

### 3. CWSR Lv3 定位

| 维度 | 论文方案 | 实际方案 |
|------|---------|---------|
| **定位** | Test 4.1.1 (测试的一部分) | 独立项目 (Stage 5) |
| **复杂度** | 假设简单 ioctl 调用 | 承认需要 4-6 周专门工作 |
| **阻塞性** | ❌ 阻塞基础测试 | ✅ 不阻塞基础功能验证 |

**论文方案对 CWSR 的假设**:

```cpp
// Test 4.1.1 假设的实现（看起来简单）
void interrupt_lv3(uint32_t queue_id) {
    int kfd_fd = open("/dev/kfd", O_RDWR);
    struct kfd_ioctl_preempt_queue_args args = {...};
    ioctl(kfd_fd, AMDKFD_IOC_PREEMPT_QUEUE, &args);
    close(kfd_fd);
}
```

**实际情况**:
1. ❓ KFD ioctl 需要特殊权限
2. ❓ Wavefront save/restore 状态管理
3. ❓ 与 XSched 调度器的集成（200+ LoC）
4. ❓ 稳定性和性能优化
5. ❓ 硬件特性验证（CWSR 是否真的启用？）

**结论**: 实际方案将 CWSR Lv3 作为独立项目更现实。

---

### 4. 工具依赖

| 工具 | 论文方案假设 | 实际方案 |
|------|------------|---------|
| **XSched** | ✅ 已编译可用 | ❓ 需要验证 |
| **Triton (ROCm)** | ✅ 可用 | ❓ 需要寻找或适配 |
| **Paella** | ✅ 可直接对比 | ⏭️ 暂不考虑 |
| **K-EDF Policy** | ✅ 200 LoC 可实现 | ⏭️ Future work |
| **Laxity-based Policy** | ✅ 104 LoC 可实现 | ⏭️ Future work |
| **PyTorch + XSched** | ❓ 未提及 | ✅ 已有基础 |

**论文方案的复杂依赖示例**:

```python
# Test 8.3.1: Triton Integration
# 假设 Triton ROCm 版本可用
docker pull tritonserver:rocm  # ❓ 这个镜像存在吗？

# Test 8.3.2: Paella Comparison  
# 假设 Paella 可直接运行
git clone https://github.com/eniac/paella
./run_paella.sh  # ❓ 需要多少配置工作？
```

**实际方案的策略**:
```python
# Test 4.1: 利用已验证的 PyTorch 环境
export LD_PRELOAD=.../libshimhip.so
./TEST_REAL_MODELS.sh  # ✅ 我们已经有了！
```

**结论**: 实际方案最小化外部依赖，充分利用已有工作。

---

### 5. 测试复杂度

| 测试类型 | 论文方案 | 实际方案 |
|---------|---------|---------|
| **Workload** | 复杂（ResNet-152, Bert-large, etc.） | 简化（先用简单 kernel） |
| **度量指标** | 完整（P99, throughput, etc.） | 渐进（先验证功能） |
| **成功标准** | 严格（对标论文数据） | 宽松（先保证能跑） |

**示例对比**:

```python
# 论文方案 Test 7.2.1: 复杂的 ResNet-152 workload
model = torchvision.models.resnet152(pretrained=True).cuda()
# 需要：
# - 前台任务 20% throughput
# - 后台任务 100% throughput
# - P99 延迟 < 1.30× standalone
# - 与 Native scheduler 对比

# 实际方案 Test 2.1: 简化的优先级测试
__global__ void delay_kernel(int iterations) {
    // 简单的 busy wait
}
# 目标：
# - 高优先级任务能执行
# - 低优先级任务不被饿死
# - 先验证功能，再测性能
```

**结论**: 实际方案先确保功能可用，再逐步提高复杂度。

---

### 6. 时间估计

| 阶段 | 论文方案 | 实际方案 |
|------|---------|---------|
| **Phase/Stage 1** | Week 1-2 | 1 天 |
| **Phase/Stage 2** | Week 3-4 | 2 天 |
| **Phase/Stage 3** | Week 5-6 | 2 周 |
| **Phase/Stage 4** | Week 7-8 (含 Lv3) | 1 周 (不含 Lv3) |
| **Phase/Stage 5** | Week 9-10 | ⏭️ 独立项目 (4-6 周) |
| **总计（基础）** | 10 周 | ~1 月 |

**论文方案的时间假设**:

```
Phase 1 (Week 1-2): 基础验证
├── Test 7.1.1: Portability (✅ 假设 XSched 可用)
├── Test 7.4.1: Runtime overhead
└── Test 7.4.2: CPU overhead

实际可能遇到的问题：
- Week 1: 编译失败，调试 CMake 配置
- Week 1-2: 官方示例无法运行，调试 HIP 版本
- Week 2: 基础 API 有问题，调试 XSched shim
...
```

**实际方案的渐进策略**:

```
Stage 1 (Day 1): Baseline
├── Test 1.1: Compilation (可能 4-8 小时)
├── Test 1.2: Native examples (可能遇到问题)
└── Test 1.3: API coverage (如果前面成功)

如果遇到问题：
- 调试并解决 ← 不强行进入 Stage 2
- 记录问题和解决方案
- 为社区贡献 bug 报告
```

**结论**: 实际方案时间估计更保守，允许调试时间。

---

## 🎯 推荐使用策略

### 短期（本周）：使用实际方案

```bash
# 1. 阅读实际方案
cat XSched_MI308X_REALISTIC_TEST_PLAN.md

# 2. 运行 Stage 1 测试
./tests/test_1_1_compilation.sh
./tests/test_1_2_native_examples.sh
./tests/test_1_3_api_coverage.sh

# 3. 记录问题和结果
```

**原因**:
- ✅ 从基础开始，避免假设
- ✅ 立即可执行
- ✅ 遇到问题可及时调整

### 中期（下周）：对照论文方案

```bash
# 如果 Stage 1 成功，开始参考论文方案的详细设计

# Stage 2: 参考 Test 7.2.1 (Fixed Priority)
# - 使用论文的实验设置
# - 但先用简化 workload

# Stage 3: 参考 Test 7.3.1 (Preemption Latency)  
# - 使用论文的测量方法
# - 对标论文数据
```

**原因**:
- ✅ 论文方案提供了详细的测试设计
- ✅ 可以对标论文数据进行验证
- ✅ 但基础已经打好了

### 长期（未来）：复现论文全部实验

```bash
# 如果前面都成功，逐步复现论文的所有测试

# Case Study 1: Multi-tenant (Test 8.1.1)
# Case Study 2: Video conferencing (Test 8.2.1)
# Case Study 3: Inference serving (Test 8.3.1)

# 最终目标：发表论文或技术报告
```

**原因**:
- ✅ 论文方案是学术标准
- ✅ 完整复现有学术价值
- ✅ 但需要基础工作先完成

---

## 📈 决策树

```
开始测试
    ↓
是否已确认 XSched 可编译运行？
    ├─ 否 → 使用实际方案 Stage 1
    │       ↓
    │     编译安装成功？
    │       ├─ 是 → Stage 2
    │       └─ 否 → 调试，查看 GitHub Issues
    │
    └─ 是 → 可以参考论文方案，但建议先跑一遍实际方案验证
            ↓
          官方示例能运行？
            ├─ 是 → 可以开始论文方案的复杂测试
            │       (Test 7.2.1, 7.3.1, etc.)
            │
            └─ 否 → 回到实际方案，逐步调试
                    (Test 1.2, 1.3)
```

---

## 💡 两个方案的价值

### 论文方案的价值 ⭐⭐⭐⭐⭐

**学术价值**:
- ✅ 完整复现论文实验
- ✅ 可以发表技术报告
- ✅ 对标国际研究水平

**技术价值**:
- ✅ 详细的测试设计
- ✅ 明确的性能指标
- ✅ 全面的功能覆盖

**作为参考**:
- ✅ 长期目标和路线图
- ✅ 测试设计的灵感来源
- ✅ 成功标准的定义

### 实际方案的价值 ⭐⭐⭐⭐⭐

**工程价值**:
- ✅ 立即可执行
- ✅ 避免理想化假设
- ✅ 渐进式验证

**现实意义**:
- ✅ 充分利用已有工作（PyTorch）
- ✅ 最小化外部依赖
- ✅ 允许调试时间

**作为起点**:
- ✅ 从零开始的路径
- ✅ 遇到问题可及时调整
- ✅ 为后续复杂测试打基础

---

## 🎓 建议

### 对于研究人员

**如果目标是发表论文**:
1. 先用实际方案打好基础（1-2 周）
2. 再用论文方案复现实验（6-8 周）
3. 对比并分析 MI308X vs 论文硬件（MI50/GV100）
4. 撰写技术报告

### 对于工程师

**如果目标是集成到生产**:
1. 专注实际方案（4-6 周）
2. 论文方案作为参考（按需选取）
3. 重点验证 Stage 1-4
4. Stage 5 (CWSR Lv3) 作为未来优化项

### 对于学生/初学者

**如果目标是学习 XSched**:
1. 先阅读论文方案，了解全貌
2. 再用实际方案动手实践
3. 对比两个方案，理解差异
4. 逐步深入，从简单到复杂

---

## ✅ 总结

| 问题 | 答案 |
|------|------|
| **哪个方案更好？** | 各有价值，取决于目标 |
| **应该从哪个开始？** | **实际方案**（避免卡住） |
| **论文方案要丢弃吗？** | **不！** 作为参考和长期目标 |
| **两个方案冲突吗？** | 不冲突，互补 |
| **时间线是什么？** | 实际方案（1 月基础）→ 论文方案（2-3 月完整） |

---

**最终建议**:

```bash
# 今天：开始实际方案
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED
./tests/test_1_1_compilation.sh

# 本周：完成 Stage 1
# 下周：开始参考论文方案的详细设计
# 未来：逐步复现论文的所有实验
```

两个方案都是优秀的工作，关键是根据当前状态选择合适的起点！🚀
