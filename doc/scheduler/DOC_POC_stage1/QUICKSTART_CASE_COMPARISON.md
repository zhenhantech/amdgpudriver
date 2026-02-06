# 快速开始：Case对比与抢占测试

**更新**: 2026-02-05  
**5分钟快速上手**

---

## 🎯 目标

1. **分析Queue使用差异**: Case-A (CNN) vs Case-B (Transformer)
2. **设计抢占机制**: 如何让Case-A抢占Case-B

---

## ⚡ 5分钟快速测试

### 方式1: 自动化对比测试（推荐）⭐⭐⭐⭐⭐

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 一键运行（每个case 60秒）
./run_case_comparison.sh zhen_vllm_dsv3 60
```

**自动执行**:
- ✅ 运行Case-A（CNN）
- ✅ 运行Case-B（Transformer）
- ✅ 提取Queue统计
- ✅ 生成对比报告

### 方式2: 手动测试（详细控制）

#### Step 1: 测试Case-A

```bash
docker exec -it zhen_vllm_dsv3 bash
cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

export AMD_LOG_LEVEL=5
python3 case_a_cnn.py 2>&1 | tee log/case_a.log
```

#### Step 2: 测试Case-B

```bash
export AMD_LOG_LEVEL=5
python3 case_b_transformer.py 2>&1 | tee log/case_b.log
```

#### Step 3: 分析对比

```bash
python3 analyze_queue_logs.py log/case_a.log log/case_b.log
```

---

## 📊 查看结果

### 提取Queue信息

```bash
# Case-A的Queue IDs
grep 'HWq=.*id=' log/case_a.log | grep -o 'id=[0-9]*' | sort -u

# Case-B的Queue IDs
grep 'HWq=.*id=' log/case_b.log | grep -o 'id=[0-9]*' | sort -u

# 对比
echo "=== Queue对比 ==="
echo "Case-A:"
grep 'HWq=.*id=' log/case_a.log | grep -o 'id=[0-9]*' | sort -u
echo ""
echo "Case-B:"
grep 'HWq=.*id=' log/case_b.log | grep -o 'id=[0-9]*' | sort -u
```

### 统计Kernel类型

```bash
# Case-A的Kernel
echo "Case-A Kernel类型:"
grep 'ShaderName' log/case_a.log | sed 's/.*ShaderName : //' | cut -d'_' -f1-3 | sort | uniq -c | head -10

# Case-B的Kernel
echo "Case-B Kernel类型:"
grep 'ShaderName' log/case_b.log | sed 's/.*ShaderName : //' | cut -d'_' -f1-3 | sort | uniq -c | head -10
```

---

## 🎯 抢占测试

### 运行抢占验证

```bash
docker exec -it zhen_vllm_dsv3 bash
cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 使用AMD调试日志
export AMD_LOG_LEVEL=5

# 运行抢占测试
python3 test_preemption_simple.py 2>&1 | tee log/preemption.log
```

### 查看抢占效果

```bash
# 查看延迟统计
grep -A 15 "结果分析" log/preemption.log

# 应该看到:
# Case-A (高优先级): 15ms (低延迟)
# Case-B (低优先级): 30ms (高延迟)
# ✅ Case-A延迟更低（优先级生效）
```

---

## 📊 预期输出示例

### 对比测试输出

```
╔════════════════════════════════════════════════════════════════════╗
║  Case-A (CNN) vs Case-B (Transformer) 对比测试                     ║
╚════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
测试 Case-A: CNN卷积网络
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Case-A: CNN测试
PID: 12345

预热...
开始测试 (60秒)...
  迭代   10, 已用 1.2秒
  迭代   20, 已用 2.4秒
  ...

✅ Case-A完成
   总迭代: 500
   总时间: 60.05秒
   平均延迟: 120.10ms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
测试 Case-B: Transformer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Case-B: Transformer测试
PID: 12346

预热...
开始测试 (60秒)...
  迭代   10, 已用 1.5秒
  迭代   20, 已用 3.0秒
  ...

✅ Case-B完成
   总迭代: 400
   总时间: 60.12秒
   平均延迟: 150.30ms

╔════════════════════════════════════════════════════════════════════╗
║  分析对比结果                                                       ║
╚════════════════════════════════════════════════════════════════════╝

━━━ Case-A (CNN) 统计 ━━━
  总迭代:     500
  总时间:     60.05秒
  平均延迟:   120.10ms

━━━ Case-B (Transformer) 统计 ━━━
  总迭代:     400
  总时间:     60.12秒
  平均延迟:   150.30ms

━━━ Queue使用分析 ━━━
Case-A Queue统计:
  Queue相关日志: 15000 条
  Kernel提交:    5000 次

Case-B Queue统计:
  Queue相关日志: 12000 条
  Kernel提交:    4000 次

━━━ 提取Queue ID ━━━
Case-A的Queue IDs:
  id=1
  id=2

Case-B的Queue IDs:
  id=1
```

### 分析工具输出

```
分析文件: log/case_comparison_*/case_a_cnn.log
======================================================================

━━━ Queue统计 ━━━
  不同的Queue IDs: 2
  Queue IDs: [1, 2]

  不同的HW Queue地址: 2
    0x7fad66c00000
    0x7fad66d00000

━━━ Kernel统计 ━━━
  不同的Kernel类型: 6
  Top 5 最常用Kernel:
      800x  miopenConv2dBwdWrWS2
      500x  MIOpenPooling
      200x  MIOpenBatchNorm
      ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━ 对比总结 ━━━
  Case-A Queue IDs数: 2
  Case-B Queue IDs数: 1
  共同使用的Queue: [1]
  Case-A独有: [2]

━━━ 抢占可行性分析 ━━━
  ⚠️  两个Case使用了相同的Queue [1]
     → 需要在Queue级别实施抢占
     → 或使用不同的优先级Queue
```

---

## 🎯 下一步

### 如果Queue使用不同

→ **容易实现抢占**: 可以独立控制不同的Queue

```python
# 方案: Queue优先级
stream_a = torch.cuda.Stream(priority=-1)  # Case-A高优先级
stream_b = torch.cuda.Stream(priority=0)   # Case-B低优先级
```

### 如果Queue使用相同

→ **需要更复杂的抢占机制**:

1. **Stream优先级**: 在同一Queue上设置不同优先级
2. **显式Suspend/Resume**: 暂停低优先级任务
3. **时间片轮转**: 给高优先级更多时间

---

## 📚 完整文档

- **CASE_COMPARISON_GUIDE.md** - 详细使用指南
- **PREEMPTION_DESIGN.md** - 抢占机制设计
- **AMD_DEBUG_GUIDE.md** - AMD调试日志使用

---

**5分钟流程总结**:
1. 运行: `./run_case_comparison.sh zhen_vllm_dsv3 60`
2. 分析: `python3 analyze_queue_logs.py log/case_comparison_*/case_*.log`
3. 抢占测试: `python3 test_preemption_simple.py`
4. 查看设计: `cat PREEMPTION_DESIGN.md`
