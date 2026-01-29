# Test 5: 真实AI模型优先级调度测试

**日期**: 2026-01-29  
**目标**: 使用真实ResNet-18和ResNet-50验证XSched优先级调度  
**状态**: 🔄 规划中

---

## 🎯 测试目标

### 为什么需要Test 5？

**Test 1-4的局限性**:
- ❌ 使用矩阵乘法**模拟**AI模型workload
- ❌ 不包含卷积、激活、BatchNorm等真实操作
- ❌ 无法反映真实AI推理的内存访问模式

**Test 5的价值**:
- ✅ 使用真实的ResNet-18和ResNet-50模型
- ✅ 包含完整的神经网络操作（卷积、池化、BN、ReLU等）
- ✅ 真实的内存访问和计算模式
- ✅ 更接近生产环境

---

## 📊 测试配置

### 模型配置

| 角色 | 模型 | Batch Size | 目标吞吐 | 优先级 | 场景 |
|------|------|-----------|---------|--------|------|
| **High Priority** | ResNet-18 | 1 | 20 req/s | P10 | 在线推理 |
| **Low Priority** | ResNet-50 | 512 | 连续运行 | P1 | 批处理 |

### 输入数据
- **尺寸**: 224×224×3 (ImageNet标准)
- **数据类型**: FP32
- **预处理**: 随机初始化（测试性能，非精度）

### XSched配置
- **LaunchConfig**: threshold=4, batch_size=2
- **Scheduler**: Local + HPF (Highest Priority First)
- **Duration**: 60秒

---

## 🔧 实现方案

### 方案A: Python + PyTorch (已有代码)

**文件**: `/data/dockercode/xsched-tests/test_two_ai_models.py`

**优势**:
- ✅ 代码已存在
- ✅ 使用成熟的PyTorch框架
- ✅ 易于调试和修改

**挑战**:
- ⚠️ Python multiprocessing HIP context问题
- ⚠️ 之前遇到`hip error 709: context is destroyed`
- ⚠️ 需要解决或绕过

**解决方案**:
1. 先运行**Baseline版本**（不使用XSched）验证模型工作
2. 创建**单进程版本**使用Python threading
3. 或修复HIP context管理

### 方案B: C++ + LibTorch (理想方案)

**优势**:
- ✅ 避免Python HIP context问题
- ✅ 使用PyTorch C++ API (libtorch)
- ✅ 与Test 1-4一致（C++ pthread）
- ✅ 更稳定

**挑战**:
- ⚠️ 需要编写新代码
- ⚠️ LibTorch API复杂度
- ⚠️ 模型加载和推理需要实现

**所需时间**: 2-3小时开发

---

## 📋 测试步骤

### Phase 1: Baseline验证 (不使用XSched)

**目标**: 验证模型可以正常运行

```bash
# 创建baseline版本（不使用XSched）
cd /data/dockercode/xsched-tests

# 修改test_two_ai_models.py，移除XSched调用
python3 test_two_ai_models_baseline.py --duration 60
```

**预期结果**:
- ResNet-18: P50 ~XX ms, P99 ~XX ms
- ResNet-50: XX iter/s
- 验证模型推理正常

### Phase 2: XSched测试

**目标**: 验证XSched对真实模型的优化效果

```bash
# 方案2a: Python单进程版本（如果multiprocessing失败）
python3 test_two_ai_models_threading.py --duration 60

# 或方案2b: 修复后的multiprocessing版本
export LD_PRELOAD=/path/to/xsched/libshimhip.so
python3 test_two_ai_models.py --duration 60
```

**预期结果**:
- 对比Baseline，量化XSched改善
- 高优先级延迟降低 XX%
- 低优先级吞吐下降 XX%

### Phase 3: C++ LibTorch版本（如果Python失败）

```bash
# 编译C++ LibTorch版本
cd /data/dockercode/xsched-official/examples/Linux/3_intra_process_sched
hipcc app_real_ai_models.cpp -o app_real_ai_models \
  -I/path/to/libtorch/include \
  -L/path/to/libtorch/lib -ltorch -lc10 -ltorch_hip \
  -I/data/dockercode/xsched-build/output/include \
  -L/data/dockercode/xsched-build/output/lib -lhalhip -lpreempt -lshimhip

# 运行
./app_real_ai_models 60
```

---

## 📊 预期对比表

### 完整测试矩阵（Test 1-5）

| 测试 | Workload类型 | 并发 | XSched改善 | 验证目标 |
|------|-------------|------|-----------|---------|
| **Test 1-2** | 矩阵乘法 | 1/16线程 | 8-11× | 基础机制 |
| **Test 3** | 矩阵乘法 | 8线程 | 稳定<1s | 混合负载 |
| **Test 4** | 矩阵乘法intensive | 2模型 | 17-30% | 高负载 |
| **Test 5** ⭐ | 真实ResNet | 2模型 | **待验证** | **真实场景** |

### Test 5预期结果

**假设**（基于Test 4经验）:

| 指标 | Baseline | XSched | 改善 |
|------|----------|--------|------|
| **High P50** | ~XX ms | ~XX ms | -XX% |
| **High P99** | ~XX ms | ~XX ms | -XX% |
| **Low吞吐** | ~XX iter/s | ~XX iter/s | -XX% |

**关键问题**:
1. 真实模型的kernel更复杂，XSched效果如何？
2. 卷积操作是否比矩阵乘法更难调度？
3. 内存密集型操作（BN、激活）对调度的影响？

---

## 🚧 已知挑战

### 1. Python HIP Context问题 ⚠️

**问题**: `hip error 709: context is destroyed`

**根因**: 
- Python multiprocessing fork后，HIP context无效
- XSched HipQueue构造函数尝试访问parent context

**解决方案**:
- [ ] 方案A: 使用threading代替multiprocessing
- [ ] 方案B: 修改XSched HipQueue构造函数，延迟context获取
- [ ] 方案C: 使用C++ LibTorch实现

### 2. GPU内存限制 ⚠️

**ResNet-50 batch=512可能OOM**

**解决方案**:
- [ ] 降低batch size (512 → 256 → 128)
- [ ] 监控GPU内存使用
- [ ] 使用FP16混合精度

### 3. 模型加载时间 ⏰

**预训练权重很大**

**解决方案**:
- ✅ 使用随机初始化（测试性能，非精度）
- ✅ 不加载pretrained weights
- ✅ 加速启动时间

---

## ✅ 测试检查清单

### Phase 1: Baseline (不使用XSched)
- [ ] ResNet-18单独运行 (P50, P99)
- [ ] ResNet-50单独运行 (吞吐)
- [ ] 双模型并发（无XSched）
- [ ] 确认GPU内存足够
- [ ] 确认模型推理正确

### Phase 2: XSched测试
- [ ] 配置XSched (HPF调度器)
- [ ] 双模型并发（有XSched）
- [ ] 记录P50/P99延迟
- [ ] 记录低优先级吞吐
- [ ] 对比Baseline计算改善

### Phase 3: 结果分析
- [ ] 生成对比表格
- [ ] 与Test 4对比（真实vs模拟）
- [ ] 量化XSched效果差异
- [ ] 分析根本原因
- [ ] 更新文档

---

## 📈 成功标准

### 最低标准 ✅
- [ ] 模型可以成功运行（Baseline）
- [ ] 收集到基本性能数据
- [ ] 识别XSched对真实模型的适用性

### 理想标准 ⭐
- [ ] XSched改善 >10% (P50/P99)
- [ ] 低优先级trade-off <20%
- [ ] 与Test 4趋势一致
- [ ] 完整的4场景对比（类似Test 4）

---

## 📝 下一步行动

### 立即执行
1. [ ] 创建Baseline版本（移除XSched调用）
2. [ ] 运行Baseline收集数据
3. [ ] 评估HIP context问题严重性

### 如果Baseline成功
4. [ ] 尝试Python threading版本
5. [ ] 或尝试修复multiprocessing版本
6. [ ] 运行XSched测试

### 如果Python失败
7. [ ] 评估C++ LibTorch可行性
8. [ ] 开发C++ 版本
9. [ ] 运行完整测试

---

## 🎯 预期时间

| 任务 | 预估时间 |
|------|---------|
| **Baseline验证** | 30分钟 |
| **Python修复/Threading** | 1-2小时 |
| **XSched测试** | 1小时 |
| **结果分析** | 30分钟 |
| **C++ LibTorch (如需要)** | 3-4小时 |
| **总计** | **3-8小时** |

---

## 📚 参考文档

- Test 1-4结果: `SYSTEMATIC_TEST_FINAL_RESULTS.md`
- Two AI Models (矩阵乘法): `TWO_AI_MODELS_COMPLETE_RESULTS.md`
- Python测试代码: `/data/dockercode/xsched-tests/test_two_ai_models.py`
- HIP Context问题分析: `TWO_AI_MODELS_COMPLETE_RESULTS.md` 第18-27行

---

**状态**: 📋 **计划完成，等待执行**  
**优先级**: ⭐⭐⭐⭐⭐ **高** (验证真实场景适用性)
