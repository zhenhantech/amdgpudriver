# 快速参考 - Case-A vs Case-B 分析结果

**日期**: 2026-02-05

---

## 🎯 核心发现（3句话总结）

1. **单Queue模型**: 两个Case都只使用1个Hardware Queue，所有Kernel通过此Queue提交
2. **Queue指针同步**: RPTR≈WPTR，说明GPU处理速度跟得上，没有积压
3. **抢占设计简化**: 只需要暂停/恢复单个Queue，不需要处理多Queue协调

---

## 📊 关键数据

| 指标 | Case-A (CNN) | Case-B (Transformer) |
|------|--------------|----------------------|
| **Queue数量** | **1** | **1** ⭐ |
| Queue地址 | 0x7f9567e00000 | 0x7f6220a00000 |
| 运行时长 | 107秒 | 246秒 |
| Kernel提交 | 127,099次 | 261,809次 |
| 日志行数 | 616万 | 1312万 |

---

## 🔧 命令速查

### 查看分析结果

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code/log/case_comparison_20260205_155247

# 运行分析脚本
./analyze_logs.sh

# 查看详细报告
cat analysis_report.txt

# 查看总结
cat ANALYSIS_SUMMARY.md
```

### 提取Queue信息

```bash
# Case-A的Queue地址
grep 'HWq=0x' case_a_cnn.log | head -1
# 输出: HWq=0x7f9567e00000

# Case-B的Queue地址
grep 'HWq=0x' case_b_transformer.log | head -1
# 输出: HWq=0x7f6220a00000

# 统计Kernel提交次数
grep -c 'KernelExecution.*enqueued' case_a_cnn.log
# 输出: 127099

grep -c 'KernelExecution.*enqueued' case_b_transformer.log
# 输出: 261809
```

### 查看Dispatch信息

```bash
# Case-A的Dispatch模式
grep 'Dispatch Header' case_a_cnn.log | head -5

# Case-B的Dispatch模式
grep 'Dispatch Header' case_b_transformer.log | head -5
```

---

## 🚀 下一步行动

### 立即开始（本周）

1. **实现Queue识别工具**
   - 从PID获取Queue地址
   - 验证Queue是否活跃

2. **实现Queue暂停/恢复**
   - 创建内核模块
   - 调用`amdgpu_amdkfd_stop_sched`
   - 调用`amdgpu_amdkfd_resume_sched`

3. **测试抢占功能**
   - Case-A抢占Case-B
   - 验证功能正确性

### 参考文档

- **详细分析**: `ANALYSIS_SUMMARY.md`
- **实现计划**: `/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/NEXT_STEPS_PREEMPTION_POC.md`
- **原始日志**: `case_a_cnn.log`, `case_b_transformer.log`

---

## 💡 关键代码片段

### 抢占伪代码

```c
// 暂停Case-B的Queue
int preempt_case_b(pid_t case_b_pid) {
    struct kfd_process *p = kfd_get_process_by_pid(case_b_pid);
    struct queue *q = get_first_queue(p);  // 只有1个Queue
    
    return amdgpu_amdkfd_stop_sched(q->device, q);
}

// 恢复Case-B的Queue
int resume_case_b(pid_t case_b_pid) {
    struct kfd_process *p = kfd_get_process_by_pid(case_b_pid);
    struct queue *q = get_first_queue(p);
    
    return amdgpu_amdkfd_resume_sched(q->device, q);
}
```

---

## 📁 文件结构

```
case_comparison_20260205_155247/
├── case_a_cnn.log              # Case-A完整日志 (616万行)
├── case_b_transformer.log      # Case-B完整日志 (1312万行)
├── pid_mapping.txt             # PID映射
├── analysis_report.txt         # 详细分析报告
├── analyze_logs.sh             # 分析脚本
├── ANALYSIS_SUMMARY.md         # 详细总结
└── QUICK_REFERENCE.md          # 本文档（快速参考）
```

---

**维护者**: AI Assistant  
**日期**: 2026-02-05

