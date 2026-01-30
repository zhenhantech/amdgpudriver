# 通过MQD查看真实硬件Queue使用情况

**核心答案**: ✅ 是的！通过读取MQD和HQD，可以知道**真正有多少个硬件Queue在使用**，而不是理论计算。

---

## 🎯 关键区别

### 理论计算 vs 实际查看

| 方式 | 数据来源 | 准确性 | 实时性 | 难度 |
|-----|---------|-------|-------|------|
| **理论计算** | 代码中的常量 (32) | 假设性 | 静态 | ⭐ 简单 |
| **读取MQD/HQD** | `/sys/kernel/debug/kfd/` | ✅ 真实 | ✅ 实时 | ⭐⭐ 中等 |

**理论计算**（当前test程序的做法）:
```cpp
if (num_streams <= 32) {
    print("Sufficient");  // 基于硬编码的32
} else {
    print("Insufficient");
}
```

**实际查看**（通过MQD/HQD）:
```bash
# 读取实际硬件状态
sudo cat /sys/kernel/debug/kfd/mqds   # 所有软件队列的MQD
sudo cat /sys/kernel/debug/kfd/hqds   # 实际硬件队列状态

# 分析实际使用的HQD数量
```

---

## 📊 KFD Debugfs 接口

### 可用的文件

```bash
/sys/kernel/debug/kfd/
├── mqds        # Memory Queue Descriptors（软件队列配置）
├── hqds        # Hardware Queue Descriptors（硬件队列状态）⭐⭐⭐
├── rls         # RunList Status（CPSCH模式的runlist）
├── hang_hws    # 触发HWS hang（测试用）
└── mem_limit   # 内存限制配置
```

### 关键文件说明

**mqds** - 软件队列视图:
```
作用: 显示所有软件队列（AQL Queue）的MQD配置
内容:
  - 每个队列的ring buffer地址（cp_hqd_pq_base）
  - 每个队列的doorbell配置（cp_hqd_pq_doorbell_control）
  - 每个队列的优先级（cp_hqd_pipe_priority）
  - CWSR配置、量子时间片等

用途: 
  ✓ 查看有多少个软件队列
  ✓ 验证每个队列的配置是否正确
  ✓ 检查优先级设置
```

**hqds** - 硬件队列视图 ⭐⭐⭐:
```
作用: 显示实际加载到GPU硬件的队列（HQD）
内容:
  - 实际使用的Pipe ID和Queue ID
  - 哪些HQD slot正在被使用
  - 每个HQD的状态（active/inactive）

用途: 
  ✓ 查看真正有多少个硬件队列在使用  ⭐⭐⭐
  ✓ 验证HQD分配策略
  ✓ 检查HQD复用情况
```

**rls** - RunList状态:
```
作用: 显示CPSCH模式下的runlist状态
内容:
  - 当前runlist中的队列数量
  - active_runlist标志状态

用途:
  ✓ 验证CPSCH调度器状态
  ✓ 检查runlist串行化问题
```

---

## 🔬 使用方法

### 方式1: 快速查看当前状态

```bash
# 查看所有软件队列的MQD
sudo cat /sys/kernel/debug/kfd/mqds | less

# 查看实际硬件队列（HQD）⭐⭐⭐
sudo cat /sys/kernel/debug/kfd/hqds | less

# 统计队列数量
sudo cat /sys/kernel/debug/kfd/mqds | grep -c "Queue"
```

### 方式2: 使用我们的脚本（推荐）

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/test_queue_limits

# 方式A: 只查看当前状态
./read_mqd_hqd.sh

# 方式B: 运行测试并实时检查（最完整）⭐⭐⭐
./test_and_inspect_mqd.sh 16    # 16 streams
./test_and_inspect_mqd.sh 32    # 32 streams
./test_and_inspect_mqd.sh 64    # 64 streams
```

### 方式3: 手动完整测试流程

```bash
# Terminal 1: 启动测试程序（保持运行）
./test_multiple_streams 32 -t 30  # 创建32个streams，保持30秒

# Terminal 2: 在streams存活时查看MQD/HQD
sudo cat /sys/kernel/debug/kfd/mqds > mqds_32streams.txt
sudo cat /sys/kernel/debug/kfd/hqds > hqds_32streams.txt

# 分析
cat mqds_32streams.txt | grep -E "Queue|pipe|cp_hqd_pq_base|priority"
cat hqds_32streams.txt
```

---

## 📝 MQD内容示例

### MQD输出格式（推测）

```
Queue ID: 100
  Process: 12345
  Priority: 11 (HIGH)
  Pipe: 0
  Queue: 0
  cp_hqd_pq_base_lo: 0x12340000
  cp_hqd_pq_base_hi: 0x00007fab
  cp_hqd_pq_doorbell_control: 0x00001000
  cp_hqd_pipe_priority: 2
  cp_hqd_queue_priority: 11
  cp_hqd_pq_control: 0x00000205
  ...

Queue ID: 101
  Process: 12345
  Priority: 1 (LOW)
  Pipe: 0
  Queue: 0
  cp_hqd_pq_base_lo: 0x56780000
  cp_hqd_pq_base_hi: 0x00007fac
  cp_hqd_pq_doorbell_control: 0x00001008
  cp_hqd_pipe_priority: 0
  cp_hqd_queue_priority: 1
  ...
```

### 关键字段分析

**判断队列数量**:
```bash
# 统计软件队列数量
grep -c "Queue ID:" mqds.txt

# 查看不同的ring buffer（每个队列应该不同）
grep "cp_hqd_pq_base" mqds.txt | sort | uniq | wc -l
```

**判断HQD使用**:
```bash
# 在NOCPSCH模式下，可以看到实际的Pipe/Queue分配
grep -E "Pipe:|Queue:" hqds.txt

# 统计使用的HQD数量
# （需要根据实际输出格式调整）
```

---

## 🔍 CPSCH vs NOCPSCH 模式的差异

### NOCPSCH模式（直接模式）

**MQD中会包含真实的Pipe/Queue**:
```
Queue ID: 100
  Pipe: 0          ← 真实的HQD Pipe ID
  Queue: 0         ← 真实的HQD Queue ID

Queue ID: 101
  Pipe: 1          ← 不同的Pipe
  Queue: 0

Queue ID: 102
  Pipe: 2
  Queue: 0
```

**可以直接统计使用的HQD**:
```bash
# 统计不同的(Pipe, Queue)组合
grep -E "Pipe:|Queue:" mqds.txt | paste - - | sort | uniq | wc -l
```

### CPSCH模式（调度器模式）⭐ MI308X使用

**MQD中的Pipe/Queue可能都是0**:
```
Queue ID: 100
  Pipe: 0          ← 所有队列都显示0
  Queue: 0         ← 不代表实际HQD位置

Queue ID: 101
  Pipe: 0          ← 相同
  Queue: 0         ← 相同

Queue ID: 102
  Pipe: 0
  Queue: 0
```

**HQD由MEC Firmware动态分配**:
- MQD中的Pipe/Queue值无意义
- 实际HQD分配对软件层不可见
- 需要查看 `/sys/kernel/debug/kfd/hqds` 获取真实硬件状态

**关键**: 即使MQD中都是(0, 0)，每个队列仍然有**不同的ring buffer地址和doorbell**！

---

## 📊 实验：查看真实硬件Queue数量

### 实验步骤

```bash
# 1. 查看baseline状态（无活跃队列）
sudo cat /sys/kernel/debug/kfd/mqds > baseline_mqds.txt
sudo cat /sys/kernel/debug/kfd/hqds > baseline_hqds.txt

baseline_queues=$(grep -c "Queue" baseline_mqds.txt)
echo "Baseline: ${baseline_queues} queues"

# 2. 启动测试程序（后台，保持streams存活）
./test_multiple_streams 32 -t 30 &
TEST_PID=$!

# 等待streams创建完成
sleep 5

# 3. 读取活跃状态（关键！）⭐⭐⭐
sudo cat /sys/kernel/debug/kfd/mqds > active_mqds.txt
sudo cat /sys/kernel/debug/kfd/hqds > active_hqds.txt

active_queues=$(grep -c "Queue" active_mqds.txt)
echo "Active: ${active_queues} queues"

new_queues=$((active_queues - baseline_queues))
echo "New queues: ${new_queues}"

# 4. 分析MQD内容
echo ""
echo "=== MQD Analysis ==="
# 查看ring buffer地址（每个队列应该不同）
echo "Ring buffer addresses (should be unique):"
grep "cp_hqd_pq_base" active_mqds.txt | head -10

# 查看doorbell配置（每个队列应该不同）
echo ""
echo "Doorbell configurations (should be unique):"
grep "doorbell" active_mqds.txt | head -10

# 查看优先级
echo ""
echo "Priorities (currently all the same due to Line 100 bug):"
grep "cp_hqd_pipe_priority" active_mqds.txt | head -10

# 5. 分析HQD内容（真实硬件状态）⭐⭐⭐
echo ""
echo "=== HQD Analysis (Real Hardware State) ==="
cat active_hqds.txt

# 6. 等待测试完成
wait ${TEST_PID}

# 7. 验证清理后状态
sleep 2
sudo cat /sys/kernel/debug/kfd/mqds > after_mqds.txt
after_queues=$(grep -c "Queue" after_mqds.txt)
echo ""
echo "After cleanup: ${after_queues} queues (should return to baseline)"
```

### 自动化脚本（推荐）

```bash
# 使用我们创建的脚本
./test_and_inspect_mqd.sh 16
./test_and_inspect_mqd.sh 32
./test_and_inspect_mqd.sh 64
```

---

## 💡 从MQD能看到什么？

### 1. 软件队列数量（精确）

```bash
sudo cat /sys/kernel/debug/kfd/mqds | grep -c "Queue"

# 输出: 实际创建的软件队列数量
# 例如: 32 （如果创建了32个streams）
```

### 2. 每个队列的唯一配置

**Ring Buffer地址**（每个队列不同）:
```bash
sudo cat /sys/kernel/debug/kfd/mqds | grep "cp_hqd_pq_base"

# 输出: 每个队列的ring buffer地址
# 地址都不同 → 证明每个stream确实有独立的ring buffer
```

**Doorbell配置**（每个队列不同）:
```bash
sudo cat /sys/kernel/debug/kfd/mqds | grep "doorbell"

# 输出: 每个队列的doorbell偏移
# 偏移递增（+8） → 证明每个队列有独立的doorbell
```

**优先级配置**:
```bash
sudo cat /sys/kernel/debug/kfd/mqds | grep "cp_hqd_pipe_priority"

# 当前状态（未修复）: 所有队列都是1 (NORMAL)
# 修复后状态: 会看到0 (LOW), 1 (MEDIUM), 2 (HIGH)
```

### 3. 硬件队列实际使用情况 ⭐⭐⭐

**通过HQD文件**:
```bash
sudo cat /sys/kernel/debug/kfd/hqds

# 显示：
#   - 哪些HQD slot正在被使用
#   - 每个HQD对应的Pipe和Queue编号
#   - HQD的活跃状态
```

**这是唯一能看到真实硬件状态的方法！**

---

## 🧪 完整测试示例

### 一键运行（推荐）

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/test_queue_limits

# 测试32个streams，并检查真实的硬件队列使用
./test_and_inspect_mqd.sh 32
```

### 脚本会做什么

1. **读取baseline状态**（测试前）
   - 记录当前有多少队列

2. **启动测试程序**
   - 创建32个streams（32个软件队列）
   - 保持streams存活

3. **读取active状态**（测试中）⭐⭐⭐
   - 读取 `/sys/kernel/debug/kfd/mqds`
   - 读取 `/sys/kernel/debug/kfd/hqds`
   - 这是**真实的硬件状态**！

4. **分析对比**
   - 对比baseline和active状态
   - 统计新创建的队列数量
   - 验证是否与预期一致

5. **生成报告**
   - 保存所有原始数据
   - 生成分析报告

### 预期输出

```
========================================
MQD/HQD Inspection Report
========================================

Test Configuration:
  - Streams Created: 32

Results:
  - Baseline Queues: 0
  - Active Queues: 32
  - New Queues: 32
  - Match: YES ✓

Key Findings:
  1. MQDs显示所有软件队列的配置
  2. 每个MQD包含独立的ring buffer地址和doorbell配置
  3. HQDs显示实际硬件队列的使用情况
  4. 在CPSCH模式下，HQD分配是动态的

从MQD中可以看到：
  ✓ 32个不同的cp_hqd_pq_base地址（独立ring buffer）
  ✓ 32个不同的doorbell配置（独立doorbell）
  ✓ 当前优先级都是1 (NORMAL)（因为Line 100 bug）

从HQD中可以看到：
  ✓ 实际使用的硬件队列数量
  ✓ 真实的Pipe/Queue分配
  ✓ 是否发生HQD复用
```

---

## 🎯 回答您的原始问题

### 问题：是否研究MQD就知道有多少个硬件Queue？

**答案**: 是的！但需要分两个层面理解：

### 1. 通过MQD知道软件队列数量 ✅

```bash
sudo cat /sys/kernel/debug/kfd/mqds | grep -c "Queue"

→ 精确知道有多少个软件队列（AQL Queue）
→ 例如：32个（如果创建了32个streams）
```

### 2. 通过HQD知道硬件队列使用 ✅⭐⭐⭐

```bash
sudo cat /sys/kernel/debug/kfd/hqds

→ 精确知道有多少个硬件队列（HQD）正在使用
→ 例如：32个（如果32个软件队列都映射到独立HQD）
→ 或者：32个（如果64个软件队列复用32个HQD）
```

### 3. 理论上的硬件队列上限 ✅

从代码中读取：
```c
// amd/amdgpu/gfx_v9_0.c:2272-2273
adev->gfx.mec.num_pipe_per_mec = 4;
adev->gfx.mec.num_queue_per_pipe = 8;
// 总共: 4 × 8 = 32 个HQD
```

### 完整答案

```
问题：真正有多少个硬件Queue？

答案取决于您问的是哪个层面：

1. 硬件能力上限（理论）:
   ✓ 从代码读取: 32个（4 Pipes × 8 Queues）
   ✓ 这是硬件设计的固定值

2. 实际正在使用的硬件Queue（实时）:
   ✓ 从 /sys/kernel/debug/kfd/hqds 读取
   ✓ 这是真实的硬件状态
   ✓ 会随着队列创建/销毁动态变化

3. 软件队列数量（实时）:
   ✓ 从 /sys/kernel/debug/kfd/mqds 读取
   ✓ 每个MQD对应一个软件队列
   ✓ 可能 > 硬件队列数量（需要复用）
```

---

## 🔧 使用建议

### 场景1: Rampup学习阶段（当前）

**目标**: 理解概念和架构

**方法**: 
- ✅ 使用理论计算（简单）
- ✅ 查看代码中的硬件配置
- ⚠️ 可选：查看MQD/HQD验证理解

### 场景2: 调试问题

**目标**: 找到实际问题

**方法**:
- ✅ 必须查看MQD/HQD（真实状态）
- ✅ 对比预期 vs 实际
- ✅ 分析差异原因

### 场景3: 性能优化

**目标**: 优化HQD使用

**方法**:
- ✅ 实时监控HQD使用率
- ✅ 检查是否发生不必要的复用
- ✅ 验证优化效果

---

## 📚 总结

### 两种方法的价值

**理论计算**（test程序当前做法）:
- ✅ 简单快速
- ✅ 足够用于概念验证
- ✅ 适合Rampup学习
- ❌ 不够精确
- ❌ 不反映实时状态

**读取MQD/HQD**（正确方法）:
- ✅ 精确、实时
- ✅ 反映真实硬件状态
- ✅ 适合调试和优化
- ⚠️ 需要root权限
- ⚠️ 需要理解输出格式

### 推荐流程

```
Step 1: 使用理论计算快速测试
  ./test_multiple_streams 32

Step 2: 如果需要验证真实状态
  ./test_and_inspect_mqd.sh 32
  
Step 3: 深入分析MQD/HQD内容
  cat logs_mqd_inspection/active_mqds_*.txt
  cat logs_mqd_inspection/active_hqds_*.txt
```

---

## 🎉 关键发现

**您的想法完全正确！** ✅

通过研究MQD（特别是读取 `/sys/kernel/debug/kfd/mqds` 和 `hqds`），可以：

1. ✅ 知道有多少个软件队列（精确）
2. ✅ 知道每个队列的完整配置（ring buffer、doorbell、优先级）
3. ✅ 知道真正有多少个硬件队列在使用（通过hqds）
4. ✅ 验证硬件队列的实际分配策略

**这比理论计算准确得多！** 是查看真实硬件状态的正确方法。

---

**创建时间**: 2026-01-30  
**关键结论**: MQD/HQD是查看真实硬件状态的窗口，比理论计算更准确
