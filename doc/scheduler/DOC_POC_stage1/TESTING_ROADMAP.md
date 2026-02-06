# GPU Queue测试路线图

**日期**: 2026-02-05  
**状态**: 准备就绪

---

## 🎯 测试目标

### 阶段1: 基础分析（当前）✅

**目标**: 理解ROCm + KFD的Queue管理机制

**测试内容**:
1. ✅ **GEMM + ftrace**: 捕获完整的ROCm→KFD交互
2. **Case-A vs Case-B**: 对比不同workload的Queue使用
3. **HQD状态监控**: 实时查看Hardware Queue状态

### 阶段2: 抢占机制设计（下一步）

**目标**: 实现高优先级任务抢占低优先级任务

**测试内容**:
1. PyTorch Stream优先级测试
2. 延迟对比分析
3. 抢占效果验证

### 阶段3: POC实现（未来）

**目标**: 完整的GPU抢占调度系统

**关键功能**:
1. 用户空间调度器
2. KFD IOCTL集成（如果需要）
3. 性能优化

---

## 📋 阶段1测试步骤

### 1️⃣ GEMM + ftrace测试（首先运行）⭐⭐⭐⭐⭐

**目的**: 获取ROCm runtime和KFD的完整交互日志

**运行**:
```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 运行测试
sudo ./run_gemm_with_ftrace.sh zhen_vllm_dsv3
```

**输出**:
- `log/gemm_ftrace_<timestamp>/gemm_amd_log.txt` - AMD日志（ROCr层）
- `log/gemm_ftrace_<timestamp>/ftrace.txt` - ftrace日志（KFD层）
- `log/gemm_ftrace_<timestamp>/analyze.sh` - 分析脚本

**分析**:
```bash
cd log/gemm_ftrace_<timestamp>
./analyze.sh

# 或手动分析
grep 'acquireQueue' gemm_amd_log.txt
grep 'kfd_create_queue' ftrace.txt
```

**期望发现**:
- ✅ Queue创建流程：ROCr → KFD → GPU
- ✅ MQD传递过程
- ✅ KCQ使用情况（如果有）
- ✅ Doorbell vs IOCTL（如果可见）
- ✅ 完整的调用时间线

**参考文档**:
- `QUICKSTART_FTRACE.md` - 快速开始
- `FTRACE_ANALYSIS_GUIDE.md` - 详细分析

---

### 2️⃣ Case对比测试（理解Queue差异）⭐⭐⭐⭐

**目的**: 对比CNN和Transformer的Queue使用模式

**运行**:
```bash
# 自动化测试（推荐）
./run_case_comparison.sh zhen_vllm_dsv3 60

# 或手动测试
docker exec -it zhen_vllm_dsv3 bash
export AMD_LOG_LEVEL=5
python3 case_a_cnn.py 2>&1 | tee log/case_a.log
python3 case_b_transformer.py 2>&1 | tee log/case_b.log
```

**分析**:
```bash
# 对比Queue使用
python3 analyze_queue_logs.py log/case_a.log log/case_b.log

# 提取Queue ID
grep 'HWq=.*id=' log/case_a.log | grep -o 'id=[0-9]*' | sort -u
grep 'HWq=.*id=' log/case_b.log | grep -o 'id=[0-9]*' | sort -u
```

**期望发现**:
- ✅ Case-A（CNN）Queue类型和数量
- ✅ Case-B（Transformer）Queue类型和数量
- ✅ 是否有共同的Queue
- ✅ Kernel类型差异
- ✅ 为抢占设计提供依据

**参考文档**:
- `QUICKSTART_CASE_COMPARISON.md` - 快速开始
- `CASE_COMPARISON_GUIDE.md` - 详细指南

---

### 3️⃣ HQD状态监控（实时观察）⭐⭐⭐

**目的**: 实时查看Hardware Queue的状态

**方法1: sysfs/debugfs**（最直接）
```bash
# 查看所有HQD
sudo cat /sys/kernel/debug/kfd/hqds

# 持续监控ACTIVE状态
while true; do
    echo "=== $(date) ==="
    sudo cat /sys/kernel/debug/kfd/hqds | grep "CP_HQD_ACTIVE\|CP_HQD_PQ_RPTR\|CP_HQD_PQ_WPTR"
    sleep 1
done
```

**方法2: AMD日志**
```bash
# 从AMD_LOG_LEVEL=5日志中提取
grep 'HWq=.*rptr=.*wptr=' test.log
```

**方法3: rocm-smi**
```bash
# 查看进程的Queue使用
watch -n 1 'rocm-smi --showpids --showuse'
```

**期望发现**:
- ✅ Queue是否活跃（ACTIVE=1）
- ✅ Read/Write Pointer差值（积压情况）
- ✅ 不同workload的Queue使用模式

**参考文档**:
- `HQD_INSPECTION_GUIDE.md` - 完整的HQD查看指南

---

## 📊 关键分析指标

### ROCm + KFD交互（从ftrace测试）

| 指标 | 目标值 | 分析方法 |
|------|--------|----------|
| Queue创建延迟 | < 1ms | ROCr时间 vs KFD时间 |
| MQD操作次数 | 1-2次 | `grep mqd ftrace.txt \| wc -l` |
| KCQ使用 | 0（用户Queue不用） | `grep kcq ftrace.txt` |
| Doorbell可见性 | 可能不可见 | `grep doorbell ftrace.txt` |

### Queue使用对比（从Case测试）

| 指标 | Case-A（CNN） | Case-B（Transformer） |
|------|---------------|----------------------|
| Queue数量 | 2-3个 | 1-2个 |
| 主要Queue类型 | Compute + DMA | Compute |
| Kernel类型 | Conv, Pool, BN | MatMul, Softmax |
| Kernel提交频率 | 高 | 中等 |

### HQD状态（从监控）

| 指标 | 说明 | 正常值 |
|------|------|--------|
| CP_HQD_ACTIVE | Queue活跃度 | 1（运行中） |
| RPTR - WPTR | 积压命令数 | < 100 |
| QUANTUM | 时间片 | 512-1024 |

---

## 🎯 下一步行动

### 立即执行（阶段1完成）

1. **运行GEMM + ftrace测试**
   ```bash
   sudo ./run_gemm_with_ftrace.sh zhen_vllm_dsv3
   cd log/gemm_ftrace_*/
   ./analyze.sh
   ```

2. **分析ROCm→KFD交互**
   - 提取Queue创建时间
   - 识别MQD操作
   - 确认Doorbell使用

3. **运行Case对比测试**
   ```bash
   ./run_case_comparison.sh zhen_vllm_dsv3 60
   python3 analyze_queue_logs.py log/case_comparison_*/case_*.log
   ```

4. **确定Queue使用模式**
   - Case-A使用哪些Queue？
   - Case-B使用哪些Queue？
   - 是否有重叠？

### 阶段2准备（抢占机制）

基于阶段1的发现，设计抢占策略：

**如果Queue不重叠**:
```python
# 策略: Queue优先级
stream_a = torch.cuda.Stream(priority=-1)  # 高优先级
stream_b = torch.cuda.Stream(priority=0)   # 低优先级
```

**如果Queue重叠**:
```python
# 策略: 显式Suspend/Resume或时间片
```

### 阶段3准备（POC实现）

- 基于阶段2的测试结果
- 实现完整的用户空间调度器
- 集成KFD IOCTLs（如果需要）

---

## 📚 完整文档索引

### 快速开始（⭐必读）
- **QUICKSTART_FTRACE.md** - ftrace测试快速开始
- **QUICKSTART_CASE_COMPARISON.md** - Case对比快速开始

### 详细指南
- **FTRACE_ANALYSIS_GUIDE.md** - ftrace分析详细指南
- **CASE_COMPARISON_GUIDE.md** - Case对比详细指南
- **HQD_INSPECTION_GUIDE.md** - HQD查看详细指南
- **PREEMPTION_DESIGN.md** - 抢占机制设计文档

### 工具和脚本
- **test_gemm_mini.py** - Mini GEMM测试（100次）
- **run_gemm_with_ftrace.sh** - ftrace集成脚本
- **case_a_cnn.py** - CNN测试
- **case_b_transformer.py** - Transformer测试
- **run_case_comparison.sh** - 对比测试脚本
- **analyze_queue_logs.py** - 日志分析工具
- **test_preemption_simple.py** - 简单抢占测试

### 其他参考
- **SIMPLE_TESTS_GUIDE.md** - 简单测试指南
- **AMD_DEBUG_GUIDE.md** - AMD调试日志指南
- **MONITORING_GUIDE_DSV3.md** - Docker监控指南

---

## ✅ 检查清单

### 阶段1完成标准

- [ ] ftrace测试运行成功
- [ ] AMD日志和ftrace日志已捕获
- [ ] 提取了Queue创建流程
- [ ] 识别了MQD操作
- [ ] Case-A测试完成
- [ ] Case-B测试完成
- [ ] 对比了Queue使用差异
- [ ] 查看了HQD状态
- [ ] 理解了完整的ROCm→KFD→GPU路径

### 进入阶段2的前提

- [ ] 知道Case-A使用哪些Queue
- [ ] 知道Case-B使用哪些Queue
- [ ] 理解了Queue优先级机制
- [ ] 确定了抢占策略（优先级、Suspend/Resume、时间片）
- [ ] 设计了抢占测试方案

---

## 💡 关键问题

运行完阶段1测试后，你应该能回答：

1. **Queue创建**:
   - ROCr如何请求Queue？
   - KFD如何创建Queue？
   - 延迟是多少？

2. **MQD管理**:
   - MQD在哪里分配？
   - MQD如何传递到GPU？
   - MQD包含哪些信息？

3. **Queue使用**:
   - Case-A使用几个Queue？类型？
   - Case-B使用几个Queue？类型？
   - 是否共享Queue？

4. **Doorbell机制**:
   - 是否使用Doorbell？
   - ftrace能否看到Doorbell？
   - Doorbell vs IOCTL？

5. **抢占可行性**:
   - PyTorch Stream优先级是否支持？
   - 是否需要KFD层的支持？
   - 最佳抢占策略是什么？

---

**维护者**: AI Assistant  
**更新**: 2026-02-05  
**版本**: 1.0

**下一步**: 运行 `sudo ./run_gemm_with_ftrace.sh zhen_vllm_dsv3`
