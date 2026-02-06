# 快速开始：GEMM + ftrace捕获

**更新**: 2026-02-05  
**目标**: 捕获ROCm runtime和KFD的完整交互日志

---

## ⚡ 一键运行

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 运行测试（需要sudo）
sudo ./run_gemm_with_ftrace.sh zhen_vllm_dsv3
```

**输出位置**: `log/gemm_ftrace_<timestamp>/`

---

## 📊 查看结果

### 方法1: 自动分析（推荐）

```bash
cd log/gemm_ftrace_<timestamp>
./analyze.sh
```

### 方法2: 手动查看

```bash
LOG_DIR="log/gemm_ftrace_<timestamp>"

# 1. 查看AMD日志（ROCm）
less $LOG_DIR/gemm_amd_log.txt

# 2. 查看ftrace日志（KFD）
less $LOG_DIR/ftrace.txt

# 3. 搜索Queue相关
grep -i 'queue\|HWq' $LOG_DIR/gemm_amd_log.txt | head -20
grep -i 'queue\|mqd' $LOG_DIR/ftrace.txt | head -20
```

---

## 🔍 关键信息提取

### Queue创建

```bash
# ROCr层
grep 'acquireQueue' $LOG_DIR/gemm_amd_log.txt

# KFD层
grep 'kfd_create_queue\|queue.*create' $LOG_DIR/ftrace.txt
```

### MQD操作

```bash
# 如果你在KFD中添加了MQD trace point
grep -i 'mqd' $LOG_DIR/ftrace.txt | head -20
```

### Kernel提交

```bash
# ROCr层
grep 'KernelExecution.*enqueued' $LOG_DIR/gemm_amd_log.txt | head -10

# KFD层（Doorbell相关）
grep -i 'doorbell\|interrupt' $LOG_DIR/ftrace.txt | head -10
```

---

## 📈 预期结果示例

### AMD日志输出

```
:3:rocdevice.cpp:3045: 175037104827 us: [pid:157801] 
acquireQueue refCount: 0x7fad66c00000 (1)

:5:command.cpp:355: 175037138308 us: [pid:157801] 
Command (KernelExecution) enqueued: 0xd17f170 to queue: 0xbe00d60

:4:rocvirtual.cpp:1177: 175228597956 us: [pid:157801] 
SWq=0x7faf945b8000, HWq=0x7fad66c00000, id=1
```

### ftrace输出

```
python3-157801 [005] .... 175037.104830: kfd_ioctl <-do_syscall_64
python3-157801 [005] .... 175037.104831: kfd_create_queue <-kfd_ioctl
python3-157801 [005] .... 175037.104832: kfd_init_mqd <-kfd_create_queue
```

---

## 🎯 分析流程

### 步骤1: 提取Queue ID

```bash
grep 'HWq=.*id=' $LOG_DIR/gemm_amd_log.txt | grep -o 'id=[0-9]*' | sort -u
# 输出: id=1
```

### 步骤2: 分析时间关联

```bash
# AMD时间: 175037104827 us = 175037.104827 秒
# ftrace时间: 175037.104830 秒
# → 差异3微秒，说明是同一操作！
```

### 步骤3: 识别关键路径

```
ROCr: acquireQueue (175037.104827s)
  ↓
KFD: kfd_ioctl (175037.104830s)
  ↓
KFD: kfd_create_queue
  ↓
KFD: kfd_init_mqd
  ↓
完成
```

---

## 💡 常见问题

### Q1: ftrace日志为空？

**检查**:
```bash
# 1. 验证ftrace已启动
cat /sys/kernel/debug/tracing/tracing_on
# 应该输出: 1

# 2. 检查过滤器
cat /sys/kernel/debug/tracing/set_ftrace_filter
# 应该有 :mod:amdgpu

# 3. 检查KFD模块
lsmod | grep amdgpu
```

### Q2: 看不到MQD/KCQ信息？

**原因**: 需要在KFD源码中添加自定义trace point

**解决**: 参考 `FTRACE_ANALYSIS_GUIDE.md` 中的"添加trace point"章节

### Q3: AMD日志太大？

**解决**: 使用mini测试（100次迭代，约10秒）
```bash
# 已经在run_gemm_with_ftrace.sh中使用test_gemm_mini.py
# 日志大小: ~10-50MB
```

---

## 📚 完整文档

| 文档 | 说明 |
|------|------|
| **FTRACE_ANALYSIS_GUIDE.md** | 详细的ftrace分析指南 |
| **HQD_INSPECTION_GUIDE.md** | HQD查看和状态分析 |
| **CASE_COMPARISON_GUIDE.md** | Case-A vs Case-B对比 |
| **PREEMPTION_DESIGN.md** | 抢占机制设计 |

---

## 🔄 完整工作流

```bash
# 1. 捕获日志
sudo ./run_gemm_with_ftrace.sh zhen_vllm_dsv3

# 2. 快速分析
cd log/gemm_ftrace_<timestamp>
./analyze.sh

# 3. 深入分析
# 根据FTRACE_ANALYSIS_GUIDE.md进行详细分析

# 4. 运行Case对比测试
cd ../..
./run_case_comparison.sh zhen_vllm_dsv3 60

# 5. 测试抢占
docker exec -it zhen_vllm_dsv3 bash
export AMD_LOG_LEVEL=5
python3 test_preemption_simple.py
```

---

**关键命令速查**:
```bash
# 运行ftrace测试
sudo ./run_gemm_with_ftrace.sh zhen_vllm_dsv3

# 分析结果
cd log/gemm_ftrace_*/
./analyze.sh

# 提取Queue ID
grep 'HWq=.*id=' gemm_amd_log.txt | grep -o 'id=[0-9]*' | sort -u

# 查看关键函数
grep -i 'kfd_create_queue\|kfd_init_mqd' ftrace.txt
```
