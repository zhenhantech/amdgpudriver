# POC Stage 1 完整路线图（含实验验证）

**日期**: 2026-02-04  
**基于**: Map/Unmap机制研究 + 实验设计  
**目标**: 实现Online-AI抢占Offline-AI的队列级调度

---

## 🎯 总体目标

```
实现一个完整的POC，证明：
  1. 可以识别AI模型使用的队列
  2. 可以实时抢占低优先级队列
  3. 抢占延迟满足实时需求（<10ms）
  4. 系统稳定可靠
```

---

## 📅 实施路线图

### 🔬 阶段0: 实验验证（1周）⭐ 当前阶段

**目标**: 收集关键数据，为POC设计提供依据

#### 实验1: 队列使用分析 ⭐⭐⭐⭐⭐ 最高优先级

**位置**: `/mnt/md0/zhehan/code/flashinfer/dockercode/gpreempt_test/`

**快速开始**:
```bash
cd /mnt/md0/zhehan/code/flashinfer/dockercode/gpreempt_test
./exp01_queue_monitor.sh
python3 analyze_queue_usage.py ./exp01_results
```

**时间**: 15-20分钟  
**文档**: 
- `EXP_01_QUEUE_USAGE_ANALYSIS.md` (详细设计)
- `EXP01_QUICK_START.md` (快速指南)

**关键问题**:
1. ✅ 一个模型用几个队列？
2. ✅ Queue ID是什么？
3. ✅ 队列数量是否稳定？
4. ✅ MQD → HQD映射关系？

**输出**:
- 队列数量: X个
- Queue IDs: [...]
- 稳定性: 是/否
- 映射验证: 1 MQD → 4 HQD

**对POC的影响**:
```
如果稳定:
  → 可以简化POC（硬编码Queue ID）
  → 开发更快（1周）

如果不稳定:
  → 需要动态识别
  → 开发稍慢（1.5周）
  → 但更通用
```

---

#### 实验2: 不同模型对比（可选）⭐⭐⭐

**目标**: 验证不同模型的队列使用差异

**测试模型**:
1. 简单矩阵乘法（baseline）
2. ResNet50训练
3. BERT推理
4. 多GPU训练（如果适用）

**预期发现**:
- 不同模型队列数量可能不同
- 复杂模型可能使用更多队列
- 多GPU会成倍增加队列数

**时间**: 1小时  
**优先级**: 中（可跳过）

---

#### 实验3: 并发模型测试（可选）⭐⭐⭐⭐

**目标**: 验证多模型并发时队列是否独立

**测试方案**:
```bash
# 同时运行2个模型
python3 model_a.py &
python3 model_b.py &

# 验证队列不重叠
```

**关键验证**:
- Queue ID不重叠 ✅
- 总队列数 = 模型A + 模型B ✅
- 可以独立控制 ✅

**时间**: 30分钟  
**优先级**: 高（如果要支持多模型）

---

### 💻 阶段1: 方案选择（2-3天）

基于实验结果，选择实施方案：

#### 方案A: 传统方案（保守）⭐⭐⭐⭐

**特点**:
- 使用 `suspend_queues` / `resume_queues`
- 不修改内核
- 快速验证概念

**延迟**: ~15ms  
**开发时间**: 1周  
**风险**: 低

**适用场景**:
- 时间紧张
- 只需要概念验证
- 无法修改内核

---

#### 方案B: Map/Unmap优化方案（激进）⭐⭐⭐⭐⭐ 推荐

**特点**:
- 使用批量 `unmap` / `fast_remap`
- 需要修改内核（~400行）
- 性能优化10-150倍

**延迟**: ~1ms  
**开发时间**: 2周  
**风险**: 中（需要内核修改）

**适用场景**:
- 时间充足
- 性能要求高
- 可以修改内核

**参考文档**:
- `New_DESIGN_MAP_UNMAP_BASED_PREEMPTION.md`
- `New_IMPLEMENTATION_COMPARISON.md`

---

#### 方案C: 渐进式（推荐）⭐⭐⭐⭐⭐

```
Week 1-2: 实施方案A（传统）
  → 验证概念
  → 收集性能baseline

Week 3-4: 升级到方案B（优化）
  → 如果性能不满足
  → 实施Map/Unmap优化
  
优点: 风险最低，渐进式验证
```

---

### 🛠️ 阶段2: 内核开发（Week 1）

**仅适用于方案B或方案C的Week 3-4**

#### Day 1-2: 新增IOCTL接口

**任务清单**:
- [ ] 定义新的IOCTL命令
  - `AMDKFD_IOC_BATCH_UNMAP_QUEUES`
  - `AMDKFD_IOC_FAST_REMAP`
  - `AMDKFD_IOC_SET_HQD_RESERVATION`
  
- [ ] 实现IOCTL handler
  - `kfd_batch_unmap_queues()`
  - `kfd_fast_remap()`
  - `kfd_set_hqd_reservation()`

- [ ] 编译和基本测试

**预计代码量**: ~400行  
**复用比例**: 80% (利用已有KFD函数)

---

#### Day 3: 内核逻辑实现

**核心函数**:
```c
// 批量Unmap（利用execute_queues_cpsch）
int kfd_batch_unmap_queues(...) {
    // 1. 标记队列为inactive
    // 2. 调用execute_queues_cpsch()
    // 3. KFD自动批量unmap
}

// 快速Remap（保留MQD）
int kfd_fast_remap(...) {
    // 1. 为每个队列allocate_hqd()
    // 2. 标记为active
    // 3. 调用execute_queues_cpsch()
    // 4. MQD直接加载到新HQD
}
```

**关键**:
- 复用 `execute_queues_cpsch()` ← 批量操作核心
- 复用 `allocate_hqd()` / `load_mqd()` ← 动态分配
- 不需要从头实现 ✅

---

#### Day 4: 内核测试

**测试用例**:
1. 单队列unmap/remap
2. 批量10队列unmap/remap
3. HQD资源预留测试
4. 压力测试（100次操作）

**验证指标**:
- 延迟: <1ms ✅
- 稳定性: 无crash ✅
- 正确性: 队列恢复正常 ✅

---

### 📦 阶段3: 用户空间开发（Week 2）

#### Day 5-6: libgpreempt_poc_v2.so

**功能列表**:
- [ ] 新API封装
  - `gpreempt_batch_unmap_queues()`
  - `gpreempt_fast_remap_queues()`
  - `gpreempt_set_hqd_reservation()`
  - `gpreempt_get_hqd_status()`

- [ ] HQD监控
  - 解析 `/sys/kernel/debug/kfd/hqds`
  - 实时统计active HQD数量

- [ ] MQD解析增强
  - 按PID过滤队列
  - 提取Queue ID列表

**预计代码量**: ~500行

---

#### Day 7: Python测试框架

**组件**:
```python
class HQDResourceMonitor:
    """实时监控HQD资源"""
    def start()
    def get_status()
    def cleanup_idle_queues()

class SmartQueueScheduler:
    """智能队列调度"""
    def register_offline_queue()
    def handle_online_request()
    def print_statistics()
```

**功能**:
- 自动识别Online/Offline队列
- 批量抢占
- 快速恢复
- 统计和报告

**预计代码量**: ~500行

---

#### Day 8-9: 完整测试

**测试场景**:

1. **基本功能测试**
   - Offline训练 → Online推理 → 抢占 → 恢复
   - 验证: 延迟 < 5ms ✅

2. **性能测试**
   - 20次抢占操作
   - 统计: 平均延迟、最大延迟
   - 对比: 传统方案 vs 新方案

3. **稳定性测试**
   - 1000次抢占操作
   - 验证: 无crash、无队列泄漏

4. **资源利用率测试**
   - HQD利用率统计
   - 验证: 提升到85-90%

---

#### Day 10: 文档和报告

**交付物**:
1. **测试报告**
   - 性能数据
   - 对比分析
   - 结论

2. **用户指南**
   - 安装步骤
   - 使用示例
   - 故障排除

3. **Stage 2建议**
   - 性能瓶颈分析
   - 优化方向
   - 技术路线

---

## 📊 决策树

```
开始实验
  ↓
运行 EXP_01 (队列分析)
  ↓
结果分析
  ├─ 队列稳定? ───── 是 → 简化POC (方案A或B，硬编码)
  │                   ├─ 时间充足? ─ 是 → 方案B (Map/Unmap优化)
  │                   └─ 时间紧张? ─ 是 → 方案A (传统)
  │
  └─ 队列不稳定? ─── 是 → 需要动态识别
                        ├─ 时间充足? ─ 是 → 方案B + 动态识别
                        └─ 时间紧张? ─ 是 → 方案A + 动态识别

性能验证
  ↓
延迟满足? (<10ms)
  ├─ 是 → POC完成 ✅
  └─ 否 → 考虑升级到方案B

稳定性验证
  ↓
无crash? (1000次测试)
  ├─ 是 → POC成功 ✅
  └─ 否 → 调试和修复
```

---

## 🎯 成功标准

### 必须达成（POC验证）✅

1. **功能完整**
   - ✅ 可以识别模型队列
   - ✅ 可以抢占低优先级队列
   - ✅ 可以恢复被抢占队列

2. **性能达标**
   - ✅ 抢占延迟 < 10ms (方案A)
   - ✅ 或 < 2ms (方案B)

3. **稳定可靠**
   - ✅ 100次测试无crash
   - ✅ 队列状态正确恢复

### 希望达成（优化目标）⭐

4. **性能优异**
   - ⭐ 抢占延迟 < 1ms (方案B)
   - ⭐ 批量操作 < 0.5ms

5. **资源高效**
   - ⭐ HQD利用率 > 85%
   - ⭐ 支持队列超额订阅

---

## 📚 关键文档索引

### 实验相关
- `EXP_01_QUEUE_USAGE_ANALYSIS.md` - 实验1详细设计
- `EXP01_QUICK_START.md` - 实验1快速指南
- `analyze_queue_usage.py` - 实验1分析脚本
- `exp01_queue_monitor.sh` - 实验1执行脚本

### Map/Unmap机制
- `New_SW_QUEUE_HW_QUEUE_MAPPING_MECHANISM.md` - 机制详解
- `New_DESIGN_MAP_UNMAP_BASED_PREEMPTION.md` - 优化方案设计
- `New_IMPLEMENTATION_COMPARISON.md` - 方案对比
- `New_QUICK_START_GUIDE.md` - 5分钟速览

### 官方文档学习
- `KERNEL_DOC_MQD_HQD_ANALYSIS.md` - 官方文档分析
- `KCQ_CONFIG_GUIDE.md` - KCQ配置指南

### 原有POC文档
- `README_START_HERE.md` - 总览
- `QUICKSTART_实验立即开始.md` - 快速开始
- `TROUBLESHOOTING_常见问题解决.md` - 故障排除

---

## ⏱️ 时间线估算

### 快速路线（方案A）
```
Week 0: 实验验证          3天
Week 1: 用户空间开发      4天
Week 2: 测试和文档        3天
─────────────────────────────
总计:                   ~2周
```

### 优化路线（方案B）
```
Week 0: 实验验证          3天
Week 1: 内核开发          4天
Week 2: 用户空间开发      4天
Week 3: 测试和文档        3天
─────────────────────────────
总计:                   ~3周
```

### 渐进路线（方案C）⭐推荐
```
Week 1-2: 方案A实施      2周
Week 3:   性能评估       2天
Week 4-5: 方案B升级      2周
─────────────────────────────
总计:                   ~5周
但可以在Week 2后决定是否继续
```

---

## 🚀 立即行动

### 现在就开始！

```bash
# 步骤1: 进入实验目录
cd /mnt/md0/zhehan/code/flashinfer/dockercode/gpreempt_test

# 步骤2: 运行实验1
./exp01_queue_monitor.sh

# 步骤3: 分析结果
python3 analyze_queue_usage.py ./exp01_results

# 步骤4: 根据结果决定方案
# 阅读分析报告，选择方案A、B或C
```

**预计时间**: 20分钟  
**难度**: ⭐⭐ (简单，自动化)  
**收益**: ⭐⭐⭐⭐⭐ (关键数据)

---

## 💡 关键建议

### 1. 先实验，再设计

```
不要盲目开始编码！
先运行EXP_01，收集数据
根据数据决定实施方案
```

### 2. 选择合适的方案

```
时间紧张 → 方案A（传统，稳定）
时间充足 → 方案B（优化，高性能）
不确定   → 方案C（渐进，风险低）
```

### 3. 复用已有代码

```
方案B看似复杂，但：
  - 80%复用KFD已有代码
  - 只需新增~400行
  - 性能提升10-150倍
```

### 4. 及时调整

```
Week 2评估后：
  如果方案A满足需求 → 完成 ✅
  如果性能不足     → 升级到方案B
```

---

**创建时间**: 2026-02-04  
**维护者**: Zhehan  
**状态**: 实验阶段（阶段0）  
**下一步**: 运行 EXP_01

---

**现在就开始实验吧！** 🚀
