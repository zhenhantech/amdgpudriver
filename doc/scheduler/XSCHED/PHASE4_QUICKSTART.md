# Phase 4 Quick Start

**日期**: 2026-01-28  
**核心**: 多 AI 模型优先级调度和 Latency 保证

---

## 🎯 Phase 4 目标

验证 XSched 在真实 AI 场景下的优先级调度能力：

```
✅ 多个 AI 模型并发运行
✅ 不同优先级设置
✅ 高优先级任务 Latency 保证
✅ 低优先级任务不饿死
✅ 优于 Native scheduler
```

### Phase 1-3 完成情况

**Phase 3 测试结果**: 13/14 模型通过 (92.9%) ✅

已验证的模型库：
- ResNet-18, ResNet-50, MobileNetV2, EfficientNet-B0
- Vision Transformer, DenseNet-121, VGG-16
- SqueezeNet, AlexNet
- 训练模式 (2/2) ✅
- 批处理 (2/2) ✅

**详细报告**: [PHASE3_TEST_RESULTS.md](PHASE3_TEST_RESULTS.md)  
**日志摘要**: [PHASE3_LOG_SUMMARY.md](PHASE3_LOG_SUMMARY.md)  
**完整总结**: [PHASE1_TO_3_SUMMARY.md](PHASE1_TO_3_SUMMARY.md)

---

## 📊 测试场景

### Test 4.1: 双模型优先级测试

**场景**: 在线推理 + 批处理

```
┌──────────────────────────────────────┐
│  高优先级: ResNet-18                  │
│  - 在线推理服务                       │
│  - 10 req/s                          │
│  - 目标 P99 < 50ms                   │
└──────────────────────────────────────┘
              ↓ 同时运行 ↓
┌──────────────────────────────────────┐
│  低优先级: ResNet-50                  │
│  - 批处理任务                         │
│  - 连续运行 (100% GPU)               │
│  - 尽可能高吞吐量                    │
└──────────────────────────────────────┘
```

**测试**:
1. Baseline: 不使用 XSched
2. XSched: 使用优先级调度
3. 对比: 高优先级 latency，低优先级 throughput

---

## 🚀 立即开始

### Step 1: 验证环境（2 分钟）

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED

# 验证 Phase 2 的 XSched 安装
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

---

### Step 2: 运行双模型测试（约 3 分钟）

```bash
# 运行完整测试（baseline + xsched + 对比）
./run_phase4_dual_model.sh
```

**测试流程**:
```
[1/5] 复制测试脚本               (5s)
[2/5] Baseline 测试 (无 XSched)   (60s)
[3/5] XSched 测试 (有优先级调度)  (60s)
[4/5] 对比分析                   (5s)
[5/5] 生成报告                   (5s)

Total: ~3 分钟
```

**预期输出**:
```
======================================================================
COMPARISON: XSched vs Baseline
======================================================================

High Priority Task (ResNet-18):
----------------------------------------------------------------------
  Metric             Baseline      XSched        Change
  ------------------------------------------------------------
  P99 Latency (ms)      45.23        48.56       +7.4% ✅
  Avg Latency (ms)      22.15        23.89       +7.9%
  Throughput (rps)       9.98         9.95       -0.3%

Low Priority Task (ResNet-50):
----------------------------------------------------------------------
  Metric             Baseline      XSched        Change
  ------------------------------------------------------------
  Throughput (ips)      24.56        12.34      -49.8% ✅
  Images/sec           196.5         98.7       -49.8%

======================================================================
SUMMARY
======================================================================
✅ High priority latency: GOOD (XSched P99 < 110% baseline)
✅ Low priority throughput: GOOD (XSched = 50.2% of baseline, > 30%)

🎉 Overall: PASS

Key findings:
  - High priority task maintains good latency
  - Low priority task is not starved
  - XSched priority scheduling is working
======================================================================
```

---

### Step 3: 查看详细结果

```bash
# 查看 baseline 结果
docker exec zhenflashinfer_v1 cat /data/dockercode/test_results_phase4/baseline_result.json

# 查看 xsched 结果
docker exec zhenflashinfer_v1 cat /data/dockercode/test_results_phase4/xsched_result.json

# 查看完整报告
docker exec zhenflashinfer_v1 cat /data/dockercode/test_results_phase4/phase4_dual_model_report.md
```

---

## 📊 理解结果

### 成功的标准

#### 高优先级任务（关键指标）

```
✅ P99 延迟 < baseline 的 110%
   - 说明: XSched 开销小，高优先级任务不受影响
   
✅ 吞吐量接近 10 req/s
   - 说明: 达到预期的请求频率
```

#### 低优先级任务（次要指标）

```
✅ 吞吐量 > baseline 的 30%
   - 说明: 低优先级任务不被饿死
   - 仍然能获得 GPU 资源
   
📊 吞吐量下降 50-70% 是正常的
   - 说明: 资源正确分配给高优先级
```

### 典型结果解读

#### 场景 A: 理想情况

```
高优先级 P99: 48ms (baseline: 45ms, +6.7%)  ✅ 优秀
低优先级吞吐: 12 ips (baseline: 25 ips, 48%) ✅ 合理

解读: XSched 正常工作
  - 高优先级几乎不受影响
  - 低优先级仍能获得约一半资源
```

#### 场景 B: XSched 开销较大

```
高优先级 P99: 65ms (baseline: 45ms, +44%)   ⚠️  需关注
低优先级吞吐: 8 ips (baseline: 25 ips, 32%)  ✅ 尚可

解读: XSched 有明显开销
  - 需要调优配置
  - 但低优先级不饿死
```

#### 场景 C: 优先级不生效

```
高优先级 P99: 80ms (baseline: 45ms, +78%)   ❌ 问题
低优先级吞吐: 20 ips (baseline: 25 ips, 80%) ❌ 问题

解读: 优先级可能未生效
  - 检查 XSched 配置
  - 验证 LD_PRELOAD 是否生效
```

---

## 🔧 高级选项

### 自定义测试时长

```bash
# 默认 60 秒
./run_phase4_dual_model.sh

# 如需更长测试（在脚本内修改 TEST_DURATION）
# 或手动运行
docker exec -it zhenflashinfer_v1 bash
cd /data/dockercode
export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so
python3 test_phase4_dual_model.py --duration 300 --output long_test.json
```

### 单独运行某个阶段

```bash
# 只运行 baseline
docker exec zhenflashinfer_v1 bash -c "
    cd /data/dockercode
    python3 test_phase4_dual_model.py --duration 60 --output baseline.json
"

# 只运行 xsched
docker exec zhenflashinfer_v1 bash -c "
    export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so
    cd /data/dockercode
    python3 test_phase4_dual_model.py --duration 60 --output xsched.json
"
```

---

## 🐛 问题排查

### 问题 1: Phase 2 未完成

**错误**:
```
❌ XSched source not found at /data/dockercode/xsched-official
```

**解决**:
Phase 4 需要 Phase 2 已完成。请先确认：
```bash
docker exec zhenflashinfer_v1 ls -la /data/dockercode/xsched-build/output/lib/
```

应该看到 `libhalhip.so` 和 `libshimhip.so`。

---

### 问题 2: PyTorch 错误

**错误**:
```
[HIGH] Error: CUDA out of memory
```

**解决**:
```bash
# 减小 batch size 或测试时长
# 在 test_phase4_dual_model.py 中调整
```

---

### 问题 3: 进程卡住

**现象**: 测试长时间不完成

**解决**:
```bash
# 检查 GPU 状态
docker exec zhenflashinfer_v1 rocm-smi

# 杀死进程重试
docker exec zhenflashinfer_v1 pkill -f test_phase4
```

---

## 📈 下一步

### 如果 Test 4.1 成功

```bash
# 继续 Test 4.2: 三租户场景（待实现）
# ./run_phase4_multi_tenant.sh

# 或 Test 4.3: 实时 + 批处理（待实现）
# ./run_phase4_realtime_batch.sh
```

### 如果结果不理想

1. **分析原因**:
   - 查看详细日志
   - 对比 baseline vs xsched
   - 检查 GPU 利用率

2. **调优方向**:
   - 调整优先级级别
   - 调整 in-flight command threshold
   - 分析是否是模型选择问题

3. **报告问题**:
   - 记录详细日志
   - 保存测试结果
   - 寻求社区帮助

---

## 📝 相关文档

- [PHASE4_CORE_OBJECTIVES.md](PHASE4_CORE_OBJECTIVES.md) - Phase 4 核心目标
- [PHASE4_OVERVIEW.md](PHASE4_OVERVIEW.md) - Phase 4 总览
- [XSched_MI308X测试方案_基于论文Ch7Ch8.md](XSched_MI308X测试方案_基于论文Ch7Ch8.md) - 原论文测试方案
- [DOCKER_USAGE.md](DOCKER_USAGE.md) - Docker 使用指南

---

## 🎯 快速命令参考

```bash
# Phase 4 完整流程（10 分钟）

cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED

# 1. 验证环境 (2 分钟)
./run_phase4_test1.sh

# 2. 双模型测试 (3 分钟)
./run_phase4_dual_model.sh

# 3. 查看结果
docker exec zhenflashinfer_v1 cat /data/dockercode/test_results_phase4/baseline_result.json
docker exec zhenflashinfer_v1 cat /data/dockercode/test_results_phase4/xsched_result.json
```

---

**准备好开始了吗？** 🚀

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED
./run_phase4_test1.sh
```
