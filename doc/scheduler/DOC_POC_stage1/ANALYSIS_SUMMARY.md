# Case-A vs Case-B 分析总结

**日期**: 2026-02-05  
**日志目录**: `/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code/log/case_comparison_20260205_155247`

---

## 📊 测试概述

### 测试目标

通过对比两个不同的PyTorch测试案例，分析：
1. **Queue使用复杂度**: 是否使用多个Queue
2. **抢占机制设计**: 如果Case-A想抢占Case-B，代码该怎么设计

### 测试案例

| 案例 | 类型 | 描述 | PID |
|------|------|------|-----|
| **Case-A** | CNN卷积网络 | 卷积、池化、批归一化等操作 | 158036 |
| **Case-B** | Transformer | 自注意力、前馈网络等操作 | 158122 |

---

## 🔍 关键发现

### 1. Queue使用情况 ⭐⭐⭐⭐⭐

**重要结论**: **两个Case都只使用了1个Hardware Queue！**

```
Case-A (CNN):
  - Hardware Queue地址: 0x7f9567e00000
  - Software Queue地址: 0x7f96fb1d4000
  - Queue ID: 0, 1 (两个ID，但同一个HWq)
  - acquireQueue调用: 1次

Case-B (Transformer):
  - Hardware Queue地址: 0x7f6220a00000
  - Software Queue地址: 0x7f63b31f6000
  - Queue ID: 0, 1 (两个ID，但同一个HWq)
  - acquireQueue调用: 1次
```

**意义**:
- ✅ PyTorch应用通常只使用**单个Hardware Queue**
- ✅ 所有Kernel都通过**同一个Queue**提交
- ✅ Queue内部通过**RPTR/WPTR**管理多个Dispatch
- ✅ 这简化了抢占机制的设计（只需要抢占一个Queue）

---

### 2. 工作负载对比

| 指标 | Case-A (CNN) | Case-B (Transformer) | 比率 |
|------|--------------|----------------------|------|
| **日志行数** | 6,165,052 | 13,122,405 | 2.1x |
| **运行时长** | 107.37秒 | 245.96秒 | 2.3x |
| **Kernel提交次数** | 127,099 | 261,809 | 2.1x |
| **内存分配次数** | 14 | 13 | ~1x |
| **Hardware Queue数量** | 1 | 1 | 1x |

**关键观察**:
- ✅ Transformer的计算量约为CNN的**2.1-2.3倍**
- ✅ 但两者的Queue使用模式**完全相同**（都是单Queue）
- ✅ 内存分配次数相近，说明初始化阶段类似

---

### 3. Dispatch模式分析

#### Case-A (CNN) Dispatch特征

```
示例Dispatch:
  grid=[262144, 1, 1], workgroup=[256, 1, 1]  ← 大规模并行
  grid=[20480, 1, 1],  workgroup=[256, 1, 1]  ← 中等规模
  grid=[512, 1, 1],    workgroup=[512, 1, 1]  ← 小规模（初始化）
```

**特点**:
- 卷积操作产生**大规模并行Dispatch** (grid=262144)
- Workgroup大小固定为256或512
- 大部分Dispatch使用`barrier=1, acquire=1, release=1`

#### Case-B (Transformer) Dispatch特征

```
示例Dispatch:
  grid=[512, 1, 1], workgroup=[512, 1, 1]  ← 小规模（频繁）
```

**特点**:
- 更多的**小规模Dispatch** (grid=512)
- 可能是Attention机制的特征（多个小Kernel）
- 使用`barrier=1, acquire=2, release=2`（更强的同步）

---

### 4. Queue指针活动

#### Case-A Queue指针示例

```
rptr=0,  wptr=0   ← 第一个Dispatch
rptr=1,  wptr=1   ← 第二个Dispatch
rptr=3,  wptr=3   ← 第三个Dispatch
rptr=5,  wptr=5   ← 第四个Dispatch
rptr=6,  wptr=6
rptr=7,  wptr=7
...
```

**观察**:
- RPTR和WPTR**同步增长**
- 说明GPU处理速度**跟得上**提交速度
- 没有明显的Queue积压

#### Case-B Queue指针示例

```
rptr=1,  wptr=1
rptr=3,  wptr=3
rptr=5,  wptr=5
rptr=7,  wptr=7
rptr=8,  wptr=8
...
```

**观察**:
- 类似Case-A，RPTR和WPTR同步
- 没有Queue积压现象

---

## 🎯 抢占机制设计建议

基于以上分析，如果**Case-A想抢占Case-B**，可以采用以下设计：

### 方案1: Queue级别抢占（推荐）⭐⭐⭐⭐⭐

由于两个Case都只使用**单个Queue**，可以直接在Queue级别实现抢占：

```c
// 伪代码
int preempt_case_b_for_case_a(pid_t case_a_pid, pid_t case_b_pid) {
    // 1. 找到Case-B的Queue
    struct kfd_process *victim_process = kfd_get_process_by_pid(case_b_pid);
    struct queue *victim_queue = get_first_queue(victim_process);  // 只有1个Queue
    
    // 2. 暂停Case-B的Queue
    int ret = amdgpu_amdkfd_stop_sched(victim_queue);
    if (ret != 0) {
        return -1;
    }
    
    // 3. 等待Case-B当前Kernel完成（或强制中断）
    wait_for_queue_idle(victim_queue);
    
    // 4. Case-A继续运行（或提升优先级）
    struct kfd_process *case_a_process = kfd_get_process_by_pid(case_a_pid);
    struct queue *case_a_queue = get_first_queue(case_a_process);
    boost_queue_priority(case_a_queue);
    
    // 5. 恢复Case-B（可选，取决于策略）
    // amdgpu_amdkfd_resume_sched(victim_queue);
    
    return 0;
}
```

**优点**:
- ✅ 简单直接（每个进程只有1个Queue）
- ✅ 不需要区分Kernel类型
- ✅ 可以完全暂停被抢占进程

**缺点**:
- ❌ 粒度较粗（整个进程被暂停）
- ❌ 需要等待当前Kernel完成

---

### 方案2: Kernel级别抢占（细粒度）⭐⭐⭐

如果需要更细粒度的控制，可以在Kernel级别抢占：

```c
// 伪代码
int preempt_kernel_in_queue(struct queue *victim_queue) {
    // 1. 读取当前Queue状态
    uint32_t rptr = read_queue_rptr(victim_queue);
    uint32_t wptr = read_queue_wptr(victim_queue);
    
    // 2. 如果Queue中有待处理的Dispatch
    if (wptr > rptr) {
        // 暂停Queue
        stop_queue_execution(victim_queue);
        
        // 3. 保存当前状态（CWSR）
        save_wave_state(victim_queue);
        
        // 4. 修改WPTR，跳过部分Dispatch（可选）
        // write_queue_wptr(victim_queue, rptr);
        
        return 0;
    }
    
    return -1;  // Queue已空
}
```

**优点**:
- ✅ 细粒度控制（可以只抢占部分Kernel）
- ✅ 可以保存/恢复Wave状态

**缺点**:
- ❌ 实现复杂（需要CWSR支持）
- ❌ 需要处理Kernel依赖关系

---

### 方案3: 优先级调度（协作式）⭐⭐⭐⭐

不直接抢占，而是通过优先级调度：

```c
// 伪代码
int boost_case_a_priority(pid_t case_a_pid, pid_t case_b_pid) {
    // 1. 降低Case-B的Queue优先级
    struct queue *case_b_queue = get_queue_by_pid(case_b_pid);
    set_queue_priority(case_b_queue, LOW_PRIORITY);
    
    // 2. 提升Case-A的Queue优先级
    struct queue *case_a_queue = get_queue_by_pid(case_a_pid);
    set_queue_priority(case_a_queue, HIGH_PRIORITY);
    
    // 3. GPU调度器会自动优先处理Case-A的Dispatch
    return 0;
}
```

**优点**:
- ✅ 实现简单
- ✅ 不需要强制中断
- ✅ 可以动态调整

**缺点**:
- ❌ 不是真正的抢占（Case-B仍在运行）
- ❌ 依赖GPU硬件调度器支持

---

## 📋 POC实现建议

### 阶段1: 验证Queue识别（已完成）✅

- ✅ 确认每个进程使用的Queue数量
- ✅ 提取Queue地址和ID
- ✅ 分析Queue指针活动

### 阶段2: 实现基础抢占（下一步）⭐

**目标**: 实现简单的Queue暂停/恢复

```bash
# 测试步骤
1. 启动Case-B (Transformer)
2. 等待Case-B进入稳定运行状态
3. 识别Case-B的Queue地址
4. 调用stop_sched暂停Case-B的Queue
5. 验证Case-B是否真的停止
6. 启动Case-A (CNN)
7. 验证Case-A可以正常运行
8. 恢复Case-B
```

**关键API**:
```c
// KFD提供的API
amdgpu_amdkfd_stop_sched(struct kfd_dev *kfd, struct queue *queue);
amdgpu_amdkfd_resume_sched(struct kfd_dev *kfd, struct queue *queue);

// 或者使用Debug API
ioctl(kfd_fd, KFD_IOC_DBG_TRAP_SUSPEND_QUEUES, &args);
```

### 阶段3: 实现CWSR保存/恢复（未来）

**目标**: 保存被抢占Kernel的Wave状态

---

## 🔧 测试工具

### 1. Queue监控脚本

```bash
#!/bin/bash
# monitor_queue.sh - 实时监控Queue状态

PID=$1
INTERVAL=${2:-1}

while true; do
    echo "=== $(date '+%H:%M:%S') ==="
    
    # 从AMD日志提取Queue信息
    docker exec zhen_vllm_dsv3 bash -c "
        ps aux | grep $PID
    "
    
    # 从debugfs读取HQD状态
    sudo cat /sys/kernel/debug/kfd/hqds | grep -A 20 "Queue 0" | \
        grep -E "ACTIVE|RPTR|WPTR"
    
    echo ""
    sleep $INTERVAL
done
```

### 2. 抢占测试脚本（待实现）

```bash
#!/bin/bash
# test_preemption.sh

# 1. 启动Case-B
docker exec -d zhen_vllm_dsv3 python3 case_b_transformer.py &
CASE_B_PID=$!

# 2. 等待Case-B稳定
sleep 5

# 3. 获取Case-B的Queue信息
CASE_B_QUEUE=$(get_queue_address $CASE_B_PID)

# 4. 暂停Case-B
echo "暂停Case-B (Queue: $CASE_B_QUEUE)"
# 调用KFD API暂停Queue

# 5. 启动Case-A
docker exec zhen_vllm_dsv3 python3 case_a_cnn.py

# 6. 恢复Case-B
echo "恢复Case-B"
# 调用KFD API恢复Queue
```

---

## 📊 性能指标

### Case-A (CNN)

| 指标 | 值 |
|------|-----|
| 运行时长 | 107.37秒 |
| Kernel提交次数 | 127,099 |
| 平均Kernel提交间隔 | 0.84ms |
| Queue地址 | 0x7f9567e00000 |

### Case-B (Transformer)

| 指标 | 值 |
|------|-----|
| 运行时长 | 245.96秒 |
| Kernel提交次数 | 261,809 |
| 平均Kernel提交间隔 | 0.94ms |
| Queue地址 | 0x7f6220a00000 |

---

## 💡 关键洞察

### 1. 单Queue模型简化了抢占设计

**发现**: 两个Case都只使用1个Hardware Queue

**意义**:
- 不需要处理多Queue协调问题
- 抢占逻辑可以简化为"暂停单个Queue"
- 不需要考虑Queue间依赖关系

### 2. Queue指针同步说明GPU性能充足

**发现**: RPTR和WPTR始终同步，没有积压

**意义**:
- GPU处理速度 >= CPU提交速度
- 抢占不会因为Queue积压而复杂化
- 可以假设Queue大部分时间处于"活跃执行"状态

### 3. Dispatch模式差异可用于优化

**发现**: CNN使用大Grid，Transformer使用小Grid

**意义**:
- 可以根据Dispatch特征识别任务类型
- 大Grid任务更适合长时间运行（抢占成本低）
- 小Grid任务更适合快速完成（抢占收益低）

---

## 🚀 下一步行动

### 立即行动（本周）

1. **实现Queue识别工具** ✅ 已完成
   - 从进程PID获取Queue地址
   - 验证Queue地址的有效性

2. **测试stop_sched API**
   - 编写内核模块调用`amdgpu_amdkfd_stop_sched`
   - 验证Queue是否真的停止
   - 测试恢复功能

3. **创建抢占测试框架**
   - 自动化测试脚本
   - 性能监控工具
   - 日志分析工具

### 中期目标（下周）

1. **实现基础抢占POC**
   - Case-A抢占Case-B
   - 验证功能正确性
   - 测量抢占延迟

2. **分析抢占开销**
   - 暂停Queue的时间
   - 恢复Queue的时间
   - 对性能的影响

### 长期目标（未来）

1. **实现CWSR支持**
   - 保存Wave状态
   - 恢复Wave状态
   - 支持Kernel级别抢占

2. **集成到调度器**
   - 优先级调度
   - 公平性保证
   - 资源隔离

---

## 📁 相关文件

```
log/case_comparison_20260205_155247/
├── case_a_cnn.log              # Case-A完整日志 (616万行)
├── case_b_transformer.log      # Case-B完整日志 (1312万行)
├── pid_mapping.txt             # PID映射信息
├── analysis_report.txt         # 详细分析报告
├── analyze_logs.sh             # 日志分析脚本
└── ANALYSIS_SUMMARY.md         # 本文档
```

---

## 📞 联系信息

**维护者**: AI Assistant  
**日期**: 2026-02-05  
**状态**: ✅ 分析完成，等待POC实现

---

## 附录: 命令速查

```bash
# 查看完整分析
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code/log/case_comparison_20260205_155247
./analyze_logs.sh

# 查看详细报告
cat analysis_report.txt

# 查看Case-A的Queue活动
grep 'HWq=0x7f9567e00000' case_a_cnn.log | less

# 查看Case-B的Queue活动
grep 'HWq=0x7f6220a00000' case_b_transformer.log | less

# 提取Dispatch信息
grep 'Dispatch Header' case_a_cnn.log | head -20
grep 'Dispatch Header' case_b_transformer.log | head -20

# 统计Kernel提交频率
grep 'KernelExecution.*enqueued' case_a_cnn.log | wc -l
grep 'KernelExecution.*enqueued' case_b_transformer.log | wc -l
```

---

**总结**: 
- ✅ 两个Case都使用**单个Hardware Queue**
- ✅ Queue使用模式**简单清晰**
- ✅ 抢占机制可以在**Queue级别**实现
- ✅ 下一步：实现**stop_sched**测试POC

