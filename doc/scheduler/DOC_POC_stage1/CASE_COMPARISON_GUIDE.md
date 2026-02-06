# Case-A vs Case-B 对比测试指南

**日期**: 2026-02-05  
**目的**: 
1. 分析不同PyTorch workload的Queue使用差异
2. 设计并测试抢占机制

---

## 📋 测试案例

| Case | 文件 | 类型 | 特点 | 预期Queue使用 |
|------|------|------|------|---------------|
| **Case-A** | `case_a_cnn.py` | CNN卷积网络 | Conv, Pool, BN | 多种Queue类型 |
| **Case-B** | `case_b_transformer.py` | Transformer | MatMul, Attention | 主要Compute Queue |

---

## 🚀 快速开始

### 步骤1: 依次运行两个Case，分析Queue差异

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 运行对比测试（每个case 60秒）
./run_case_comparison.sh zhen_vllm_dsv3 60
```

**这会**:
- ✅ 运行Case-A（CNN），保存AMD日志
- ✅ 运行Case-B（Transformer），保存AMD日志  
- ✅ 提取Queue ID和统计信息
- ✅ 对比两个Case的Queue使用

### 步骤2: 分析Queue使用差异

```bash
# 分析Case-A的日志
python3 analyze_queue_logs.py log/case_comparison_*/case_a_cnn.log

# 分析Case-B的日志  
python3 analyze_queue_logs.py log/case_comparison_*/case_b_transformer.log

# 对比两个Case
python3 analyze_queue_logs.py \
    log/case_comparison_*/case_a_cnn.log \
    log/case_comparison_*/case_b_transformer.log
```

**输出示例**:
```
━━━ Queue统计 ━━━
  不同的Queue IDs: 2
  Queue IDs: [1, 2]
  
  不同的HW Queue地址: 2
    0x7fad66c00000
    0x7fad66d00000

━━━ Kernel统计 ━━━
  不同的Kernel类型: 8
  Top 5 最常用Kernel:
    5000x  Conv2d_kernel
    5000x  MaxPool_kernel
    2500x  BatchNorm_kernel
    ...

━━━ 对比总结 ━━━
  Case-A Queue IDs数: 2
  Case-B Queue IDs数: 1
  共同使用的Queue: [1]
  Case-A独有: [2]
```

### 步骤3: 测试抢占机制

```bash
# 测试基础抢占（使用PyTorch Stream优先级）
docker exec -it zhen_vllm_dsv3 bash
cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

export AMD_LOG_LEVEL=5
python3 test_preemption_simple.py 2>&1 | tee log/preemption_test.log
```

**输出示例**:
```
GPU抢占测试
PID: 12345

预热...
等待5秒（检查lsof）...

开始测试...

结果分析
Case-A (高优先级):
  平均: 15.23ms
  P95:  17.45ms

Case-B (低优先级):
  平均: 28.67ms
  P95:  35.12ms

✅ Case-A延迟更低（优先级生效）
```

---

## 🔍 详细测试步骤

### 测试1: 分析Queue使用模式

**目标**: 了解Case-A和Case-B使用了哪些Queue

**步骤**:

1. **单独运行Case-A**
   ```bash
   docker exec zhen_vllm_dsv3 bash -c "
   export AMD_LOG_LEVEL=5
   cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
   python3 case_a_cnn.py
   " 2>&1 | tee log/case_a_solo.log
   ```

2. **单独运行Case-B**
   ```bash
   docker exec zhen_vllm_dsv3 bash -c "
   export AMD_LOG_LEVEL=5
   cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
   python3 case_b_transformer.py
   " 2>&1 | tee log/case_b_solo.log
   ```

3. **分析Queue使用**
   ```bash
   # 提取Queue ID
   echo "Case-A的Queue IDs:"
   grep 'HWq=.*id=' log/case_a_solo.log | grep -o 'id=[0-9]*' | sort -u
   
   echo "Case-B的Queue IDs:"
   grep 'HWq=.*id=' log/case_b_solo.log | grep -o 'id=[0-9]*' | sort -u
   
   # 统计Queue使用次数
   echo "Case-A Queue使用次数:"
   grep -c 'HWq=' log/case_a_solo.log
   
   echo "Case-B Queue使用次数:"
   grep -c 'HWq=' log/case_b_solo.log
   ```

4. **使用分析工具**
   ```bash
   python3 analyze_queue_logs.py \
       log/case_a_solo.log \
       log/case_b_solo.log
   ```

---

### 测试2: 并发运行，观察Queue冲突

**目标**: 看两个Case同时运行时，Queue是否冲突

**步骤**:

1. **启动Case-B（后台）**
   ```bash
   docker exec -d zhen_vllm_dsv3 bash -c "
   export AMD_LOG_LEVEL=5
   cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
   python3 case_b_transformer.py
   " > log/case_b_concurrent.log 2>&1 &
   
   PID_B=$!
   echo "Case-B PID: $PID_B"
   ```

2. **等待Case-B初始化**
   ```bash
   sleep 5
   ```

3. **启动Case-A（前台）**
   ```bash
   docker exec zhen_vllm_dsv3 bash -c "
   export AMD_LOG_LEVEL=5
   cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
   python3 case_a_cnn.py
   " 2>&1 | tee log/case_a_concurrent.log
   ```

4. **分析并发Queue使用**
   ```bash
   # 对比并发和单独运行的差异
   python3 analyze_queue_logs.py \
       log/case_a_concurrent.log \
       log/case_b_concurrent.log
   ```

---

### 测试3: 抢占效果验证

**目标**: 验证高优先级Case能否抢占低优先级Case

**步骤**:

1. **运行抢占测试**
   ```bash
   docker exec -it zhen_vllm_dsv3 bash
   cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
   
   export AMD_LOG_LEVEL=5
   python3 test_preemption_simple.py 2>&1 | tee log/preemption_test.log
   ```

2. **查看延迟对比**
   ```bash
   # 从输出中查看
   grep -A 10 "结果分析" log/preemption_test.log
   ```

3. **分析Queue调度**
   ```bash
   # 查看Queue操作顺序
   grep 'enqueued.*queue' log/preemption_test.log | head -50
   
   # 看高优先级Queue是否优先
   ```

---

## 📊 预期结果

### 结果1: Queue使用差异

**Case-A (CNN)**:
```
Queue IDs: [1, 2, 3]
主要操作: Conv2d, MaxPool, BatchNorm
特点: 多种Queue类型，操作多样
```

**Case-B (Transformer)**:
```
Queue IDs: [1]
主要操作: MatMul, Softmax, LayerNorm
特点: 单一Queue，MatMul密集
```

### 结果2: 抢占效果

**如果抢占生效**:
```
Case-A (高优先级):
  平均延迟: 15ms
  P95延迟:  18ms
  标准差:   2ms    ← 稳定

Case-B (低优先级):
  平均延迟: 30ms    ← 比Case-A高
  P95延迟:  40ms
  标准差:   8ms    ← 波动大（被抢占）

✅ Case-A延迟更低且更稳定
```

**如果抢占未生效**:
```
Case-A和Case-B延迟相近
→ 需要其他抢占策略
```

---

## 🎯 抢占机制设计

### 基于分析结果的设计

#### 情况1: 两个Case使用不同Queue

**抢占策略**: Queue级别优先级

```python
# 为Case-A分配高优先级Queue
stream_a = torch.cuda.Stream(priority=-1)

# 为Case-B分配低优先级Queue
stream_b = torch.cuda.Stream(priority=0)

# GPU硬件自动调度，高优先级优先
```

#### 情况2: 两个Case使用相同Queue

**抢占策略**: 显式Suspend/Resume

```python
# 方案A: 使用KFD Debug Trap (如果可用)
suspend_queues(pid_b, queue_ids)
wait_for_case_a()
resume_queues(pid_b, queue_ids)

# 方案B: 时间片轮转
# 80% 时间给Case-A，20%给Case-B
```

---

## 📝 测试检查清单

### 测试前

- [ ] Docker容器运行中
- [ ] PyTorch + ROCm可用
- [ ] 测试脚本有执行权限
- [ ] 有足够磁盘空间（AMD日志很大）

### 测试中

- [ ] Case-A运行成功
- [ ] Case-B运行成功
- [ ] AMD日志正常保存
- [ ] 能看到Queue相关日志

### 测试后

- [ ] 提取了Queue IDs
- [ ] 统计了Kernel类型
- [ ] 对比了Queue使用差异
- [ ] 验证了抢占效果（如果运行了抢占测试）

---

## 📚 相关文件

| 文件 | 说明 |
|------|------|
| `case_a_cnn.py` | Case-A测试脚本（CNN） |
| `case_b_transformer.py` | Case-B测试脚本（Transformer） |
| `run_case_comparison.sh` | 对比测试运行脚本 |
| `test_preemption_simple.py` | 简单抢占测试 |
| `analyze_queue_logs.py` | 日志分析工具 |
| `PREEMPTION_DESIGN.md` | 抢占机制详细设计 |
| `CASE_COMPARISON_GUIDE.md` | 本文档 |

---

## 💡 快速命令参考

```bash
# 1. 对比测试（自动化）
./run_case_comparison.sh zhen_vllm_dsv3 60

# 2. 分析日志
python3 analyze_queue_logs.py log/case_comparison_*/case_a_cnn.log
python3 analyze_queue_logs.py log/case_comparison_*/case_b_transformer.log

# 3. 对比分析
python3 analyze_queue_logs.py \
    log/case_comparison_*/case_a_cnn.log \
    log/case_comparison_*/case_b_transformer.log

# 4. 抢占测试
export AMD_LOG_LEVEL=5
python3 test_preemption_simple.py

# 5. 查看Queue IDs
grep 'HWq=.*id=' log/*.log | grep -o 'id=[0-9]*' | sort -u
```

---

**维护者**: AI Assistant  
**更新**: 2026-02-05
