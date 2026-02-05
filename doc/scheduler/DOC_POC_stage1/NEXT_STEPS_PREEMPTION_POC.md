# POC Stage 1 - 下一步实现计划

**日期**: 2026-02-05  
**基于**: Case-A vs Case-B 日志分析结果  
**目标**: 实现基础的Queue级别抢占机制

---

## 📋 当前状态总结

### ✅ 已完成

1. **测试环境搭建**
   - ✅ 创建了Case-A (CNN) 和 Case-B (Transformer) 测试
   - ✅ 配置AMD_LOG_LEVEL=5捕获详细日志
   - ✅ 实现了自动化测试脚本

2. **日志分析**
   - ✅ 分析了600万+行Case-A日志
   - ✅ 分析了1300万+行Case-B日志
   - ✅ 确认了关键发现：**每个进程只使用1个Hardware Queue**

3. **Queue使用模式识别**
   - ✅ Case-A Queue地址: `0x7f9567e00000`
   - ✅ Case-B Queue地址: `0x7f6220a00000`
   - ✅ 两者都只有1个Queue，简化了抢占设计

---

## 🎯 核心发现

### 关键洞察 #1: 单Queue模型

```
每个PyTorch进程 → 1个Hardware Queue → 所有Kernel通过此Queue提交
```

**意义**:
- 抢占设计可以简化为"暂停/恢复单个Queue"
- 不需要处理多Queue协调问题
- 实现复杂度大大降低

### 关键洞察 #2: Queue指针同步

```
RPTR ≈ WPTR  (大部分时间)
```

**意义**:
- GPU处理速度跟得上CPU提交速度
- Queue没有明显积压
- 抢占时机容易选择（不需要等待大量积压Kernel完成）

### 关键洞察 #3: Dispatch模式差异

```
Case-A (CNN):       大Grid (262144) → 长时间运行
Case-B (Transformer): 小Grid (512)    → 短时间运行
```

**意义**:
- 可以根据Dispatch特征优化抢占策略
- 大Grid任务更适合被抢占（抢占收益高）

---

## 🚀 下一步实现计划

### Phase 1: Queue识别与监控（本周）⭐⭐⭐⭐⭐

#### 目标
实现从进程PID到Queue地址的自动识别

#### 任务

**Task 1.1: 创建Queue查询工具**

```c
// queue_finder.c
#include <linux/module.h>
#include <linux/kfd_ioctl.h>

// 功能：根据PID查找进程的所有Queue
struct queue_info {
    uint64_t queue_address;
    uint32_t queue_id;
    uint32_t queue_type;
    uint32_t priority;
};

int find_queues_by_pid(pid_t pid, struct queue_info *queues, int max_queues) {
    struct kfd_process *p = kfd_get_process_by_pid(pid);
    if (!p) {
        return -ESRCH;
    }
    
    int count = 0;
    struct process_queue_node *pqn;
    list_for_each_entry(pqn, &p->pqm.queues, process_queue_list) {
        if (count >= max_queues) {
            break;
        }
        
        queues[count].queue_address = (uint64_t)pqn->q;
        queues[count].queue_id = pqn->q->properties.queue_id;
        queues[count].queue_type = pqn->q->properties.type;
        queues[count].priority = pqn->q->properties.priority;
        count++;
    }
    
    return count;
}
```

**Task 1.2: 创建用户空间工具**

```bash
# tools/get_queue_info.sh
#!/bin/bash
# 从用户空间查询进程的Queue信息

PID=$1

# 方法1: 通过procfs（需要内核模块支持）
cat /proc/kfd/processes/$PID/queues

# 方法2: 通过AMD日志
docker exec zhen_vllm_dsv3 bash -c "
    export AMD_LOG_LEVEL=5
    # 触发Queue信息输出
    ps -p $PID
" 2>&1 | grep 'HWq='

# 方法3: 通过debugfs
sudo cat /sys/kernel/debug/kfd/hqds | grep -A 30 "Process"
```

**验收标准**:
- ✅ 能够根据PID查询到Queue地址
- ✅ 能够验证Queue是否活跃
- ✅ 能够读取Queue的RPTR/WPTR

---

### Phase 2: 实现Queue暂停/恢复（本周）⭐⭐⭐⭐⭐

#### 目标
实现基础的Queue暂停和恢复功能

#### 任务

**Task 2.1: 创建内核模块**

```c
// kfd_preempt_module.c
#include <linux/module.h>
#include <linux/kfd_ioctl.h>

// 导出的函数
extern int amdgpu_amdkfd_stop_sched(struct kfd_dev *kfd, struct queue *q);
extern int amdgpu_amdkfd_resume_sched(struct kfd_dev *kfd, struct queue *q);

// 暂停指定进程的Queue
int preempt_process_queue(pid_t pid) {
    struct kfd_process *p = kfd_get_process_by_pid(pid);
    if (!p) {
        printk(KERN_ERR "Process %d not found\n", pid);
        return -ESRCH;
    }
    
    // 获取第一个Queue（根据分析，只有1个）
    struct process_queue_node *pqn;
    pqn = list_first_entry(&p->pqm.queues, struct process_queue_node, process_queue_list);
    
    if (!pqn || !pqn->q) {
        printk(KERN_ERR "No queue found for process %d\n", pid);
        return -ENOENT;
    }
    
    struct queue *q = pqn->q;
    struct kfd_dev *kfd = q->device;
    
    printk(KERN_INFO "Stopping queue %p for process %d\n", q, pid);
    
    int ret = amdgpu_amdkfd_stop_sched(kfd, q);
    if (ret != 0) {
        printk(KERN_ERR "Failed to stop queue: %d\n", ret);
        return ret;
    }
    
    printk(KERN_INFO "Queue stopped successfully\n");
    return 0;
}

// 恢复指定进程的Queue
int resume_process_queue(pid_t pid) {
    struct kfd_process *p = kfd_get_process_by_pid(pid);
    if (!p) {
        return -ESRCH;
    }
    
    struct process_queue_node *pqn;
    pqn = list_first_entry(&p->pqm.queues, struct process_queue_node, process_queue_list);
    
    if (!pqn || !pqn->q) {
        return -ENOENT;
    }
    
    struct queue *q = pqn->q;
    struct kfd_dev *kfd = q->device;
    
    printk(KERN_INFO "Resuming queue %p for process %d\n", q, pid);
    
    int ret = amdgpu_amdkfd_resume_sched(kfd, q);
    if (ret != 0) {
        printk(KERN_ERR "Failed to resume queue: %d\n", ret);
        return ret;
    }
    
    printk(KERN_INFO "Queue resumed successfully\n");
    return 0;
}

// 模块初始化
static int __init kfd_preempt_init(void) {
    printk(KERN_INFO "KFD Preemption module loaded\n");
    return 0;
}

// 模块清理
static void __exit kfd_preempt_exit(void) {
    printk(KERN_INFO "KFD Preemption module unloaded\n");
}

module_init(kfd_preempt_init);
module_exit(kfd_preempt_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("KFD Queue Preemption Module");
```

**Task 2.2: 创建用户空间控制工具**

```bash
# tools/preempt_control.sh
#!/bin/bash
# Queue抢占控制工具

ACTION=$1  # stop | resume | status
PID=$2

case "$ACTION" in
    stop)
        echo "暂停进程 $PID 的Queue..."
        # 通过sysfs或ioctl调用内核模块
        echo "$PID" | sudo tee /sys/module/kfd_preempt/parameters/stop_pid
        ;;
    
    resume)
        echo "恢复进程 $PID 的Queue..."
        echo "$PID" | sudo tee /sys/module/kfd_preempt/parameters/resume_pid
        ;;
    
    status)
        echo "查询进程 $PID 的Queue状态..."
        cat /sys/module/kfd_preempt/parameters/queue_status
        ;;
    
    *)
        echo "用法: $0 {stop|resume|status} <PID>"
        exit 1
        ;;
esac
```

**验收标准**:
- ✅ 能够暂停指定进程的Queue
- ✅ 能够恢复被暂停的Queue
- ✅ 暂停后，进程的Kernel提交停止
- ✅ 恢复后，进程的Kernel提交继续

---

### Phase 3: 抢占功能测试（下周）⭐⭐⭐⭐

#### 目标
验证Case-A抢占Case-B的完整流程

#### 任务

**Task 3.1: 创建自动化测试脚本**

```bash
#!/bin/bash
# test_preemption.sh - 自动化抢占测试

set -e

LOG_DIR="log/preemption_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  抢占功能测试                                                       ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# 步骤1: 启动Case-B (被抢占者)
echo "━━━ 步骤1: 启动Case-B (Transformer) ━━━"
docker exec -d zhen_vllm_dsv3 bash -c "
    export AMD_LOG_LEVEL=5
    cd /workspace/code
    python3 case_b_transformer.py
" > "$LOG_DIR/case_b.log" 2>&1 &

CASE_B_PID=$!
echo "  Case-B PID: $CASE_B_PID"
echo ""

# 步骤2: 等待Case-B稳定运行
echo "━━━ 步骤2: 等待Case-B稳定运行 ━━━"
sleep 10
echo "  Case-B已稳定运行"
echo ""

# 步骤3: 获取Case-B的Queue信息
echo "━━━ 步骤3: 获取Case-B的Queue信息 ━━━"
CASE_B_QUEUE=$(grep 'HWq=0x' "$LOG_DIR/case_b.log" | head -1 | grep -o 'HWq=0x[0-9a-f]*')
echo "  Case-B Queue: $CASE_B_QUEUE"
echo ""

# 步骤4: 记录Case-B运行前的性能
echo "━━━ 步骤4: 记录Case-B基线性能 ━━━"
CASE_B_KERNELS_BEFORE=$(grep -c 'KernelExecution.*enqueued' "$LOG_DIR/case_b.log" || echo 0)
echo "  Kernel提交次数（抢占前）: $CASE_B_KERNELS_BEFORE"
echo ""

# 步骤5: 暂停Case-B
echo "━━━ 步骤5: 暂停Case-B ━━━"
PREEMPT_TIME=$(date +%s.%N)
./tools/preempt_control.sh stop $CASE_B_PID
echo "  抢占时间: $PREEMPT_TIME"
echo ""

# 步骤6: 验证Case-B是否真的停止
echo "━━━ 步骤6: 验证Case-B停止 ━━━"
sleep 2
CASE_B_KERNELS_AFTER=$(grep -c 'KernelExecution.*enqueued' "$LOG_DIR/case_b.log" || echo 0)
echo "  Kernel提交次数（抢占后2秒）: $CASE_B_KERNELS_AFTER"

if [ "$CASE_B_KERNELS_AFTER" -eq "$CASE_B_KERNELS_BEFORE" ]; then
    echo "  ✅ Case-B已停止（Kernel提交数不再增加）"
else
    echo "  ⚠️  Case-B可能未完全停止（Kernel提交数仍在增加）"
fi
echo ""

# 步骤7: 启动Case-A (抢占者)
echo "━━━ 步骤7: 启动Case-A (CNN) ━━━"
docker exec zhen_vllm_dsv3 bash -c "
    export AMD_LOG_LEVEL=5
    cd /workspace/code
    python3 case_a_cnn.py
" > "$LOG_DIR/case_a.log" 2>&1

CASE_A_EXIT=$?
echo "  Case-A退出码: $CASE_A_EXIT"
echo ""

# 步骤8: 分析Case-A性能
echo "━━━ 步骤8: 分析Case-A性能 ━━━"
CASE_A_KERNELS=$(grep -c 'KernelExecution.*enqueued' "$LOG_DIR/case_a.log" || echo 0)
echo "  Case-A Kernel提交次数: $CASE_A_KERNELS"
echo ""

# 步骤9: 恢复Case-B
echo "━━━ 步骤9: 恢复Case-B ━━━"
RESUME_TIME=$(date +%s.%N)
./tools/preempt_control.sh resume $CASE_B_PID
echo "  恢复时间: $RESUME_TIME"
echo ""

# 步骤10: 验证Case-B恢复
echo "━━━ 步骤10: 验证Case-B恢复 ━━━"
sleep 2
CASE_B_KERNELS_RESUMED=$(grep -c 'KernelExecution.*enqueued' "$LOG_DIR/case_b.log" || echo 0)
echo "  Kernel提交次数（恢复后2秒）: $CASE_B_KERNELS_RESUMED"

if [ "$CASE_B_KERNELS_RESUMED" -gt "$CASE_B_KERNELS_AFTER" ]; then
    echo "  ✅ Case-B已恢复（Kernel提交数继续增加）"
else
    echo "  ❌ Case-B未恢复（Kernel提交数未增加）"
fi
echo ""

# 步骤11: 停止Case-B
echo "━━━ 步骤11: 停止Case-B ━━━"
kill $CASE_B_PID
echo "  Case-B已停止"
echo ""

# 步骤12: 生成测试报告
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  测试报告                                                           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

PREEMPT_DURATION=$(echo "$RESUME_TIME - $PREEMPT_TIME" | bc)

cat > "$LOG_DIR/test_report.txt" << EOF
抢占功能测试报告
生成时间: $(date)
========================================================================

一、测试配置
  - Case-A (抢占者): CNN
  - Case-B (被抢占者): Transformer
  - Case-B PID: $CASE_B_PID
  - Case-B Queue: $CASE_B_QUEUE

二、测试结果
  - Case-B Kernel提交（抢占前）: $CASE_B_KERNELS_BEFORE
  - Case-B Kernel提交（抢占后）: $CASE_B_KERNELS_AFTER
  - Case-B Kernel提交（恢复后）: $CASE_B_KERNELS_RESUMED
  - Case-A Kernel提交: $CASE_A_KERNELS
  - 抢占持续时间: ${PREEMPT_DURATION}秒

三、测试结论
  - Case-B停止: $([ "$CASE_B_KERNELS_AFTER" -eq "$CASE_B_KERNELS_BEFORE" ] && echo "✅ 成功" || echo "❌ 失败")
  - Case-A运行: $([ $CASE_A_EXIT -eq 0 ] && echo "✅ 成功" || echo "❌ 失败")
  - Case-B恢复: $([ "$CASE_B_KERNELS_RESUMED" -gt "$CASE_B_KERNELS_AFTER" ] && echo "✅ 成功" || echo "❌ 失败")

四、日志文件
  - Case-A日志: $LOG_DIR/case_a.log
  - Case-B日志: $LOG_DIR/case_b.log
  - 测试报告: $LOG_DIR/test_report.txt
EOF

cat "$LOG_DIR/test_report.txt"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "测试完成！日志保存在: $LOG_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
```

**验收标准**:
- ✅ Case-B被暂停后，Kernel提交停止
- ✅ Case-A在Case-B暂停期间正常运行
- ✅ Case-B恢复后，Kernel提交继续
- ✅ 整个流程自动化，无需人工干预

---

### Phase 4: 性能分析与优化（下周）⭐⭐⭐

#### 目标
测量抢占开销，优化抢占延迟

#### 任务

**Task 4.1: 测量抢占延迟**

关键指标:
- 暂停Queue的时间（从调用到Queue真正停止）
- 恢复Queue的时间（从调用到Queue恢复执行）
- 对被抢占进程的性能影响
- 对抢占进程的性能影响

**Task 4.2: 优化抢占策略**

优化方向:
- 选择最佳抢占时机（Queue空闲时）
- 减少状态保存/恢复开销
- 实现优先级队列（避免频繁抢占）

---

## 📅 时间表

| 阶段 | 任务 | 预计时间 | 状态 |
|------|------|----------|------|
| **Phase 1** | Queue识别与监控 | 2天 | 🔄 进行中 |
| **Phase 2** | Queue暂停/恢复 | 3天 | ⏳ 待开始 |
| **Phase 3** | 抢占功能测试 | 2天 | ⏳ 待开始 |
| **Phase 4** | 性能分析优化 | 3天 | ⏳ 待开始 |

**总计**: 约2周

---

## 🔧 开发环境

### 内核模块编译

```bash
# 创建Makefile
cat > Makefile << 'EOF'
obj-m += kfd_preempt_module.o

KDIR := /usr/src/amdgpu-6.12.12-2194681.el8_preempt
PWD := $(shell pwd)

all:
	make -C $(KDIR) M=$(PWD) modules

clean:
	make -C $(KDIR) M=$(PWD) clean

install:
	sudo insmod kfd_preempt_module.ko

uninstall:
	sudo rmmod kfd_preempt_module
EOF

# 编译
make

# 加载模块
sudo insmod kfd_preempt_module.ko

# 查看日志
dmesg | tail -20
```

### 测试环境

```bash
# 容器: zhen_vllm_dsv3
# GPU: 8x AMD MI210
# 内核: 5.10.134-19.1.al8.x86_64
# ROCm: 7.x
# PyTorch: 2.x
```

---

## 📊 成功标准

### Minimum Viable Product (MVP)

- ✅ 能够识别进程的Queue
- ✅ 能够暂停指定进程的Queue
- ✅ 能够恢复被暂停的Queue
- ✅ 暂停后，被抢占进程停止提交Kernel
- ✅ 恢复后，被抢占进程继续提交Kernel

### 理想目标

- ✅ 抢占延迟 < 10ms
- ✅ 恢复延迟 < 10ms
- ✅ 对被抢占进程的性能影响 < 5%
- ✅ 支持多次抢占/恢复循环
- ✅ 完整的错误处理和日志

---

## 🚨 风险与挑战

### 技术风险

1. **stop_sched API可能不稳定**
   - 缓解: 测试多种API（stop_sched, suspend_queues, CWSR）
   - 备选: 使用优先级调度代替强制抢占

2. **Queue状态保存/恢复复杂**
   - 缓解: 先实现简单的暂停/恢复（不保存Wave状态）
   - 未来: 集成CWSR支持

3. **多GPU环境下的复杂性**
   - 缓解: 先在单GPU上测试
   - 未来: 扩展到多GPU

### 环境风险

1. **内核模块编译问题**
   - 缓解: 使用正确的内核头文件路径
   - 备选: 使用用户空间ioctl（如果支持）

2. **容器权限问题**
   - 缓解: 使用`--privileged`模式或添加必要的capabilities
   - 备选: 在主机上直接测试

---

## 📚 参考资料

### KFD API文档

```c
// 关键函数
int amdgpu_amdkfd_stop_sched(struct kfd_dev *kfd, struct queue *q);
int amdgpu_amdkfd_resume_sched(struct kfd_dev *kfd, struct queue *q);

// Debug API
ioctl(kfd_fd, KFD_IOC_DBG_TRAP_SUSPEND_QUEUES, &args);
ioctl(kfd_fd, KFD_IOC_DBG_TRAP_RESUME_QUEUES, &args);
```

### 相关文件

```
/usr/src/amdgpu-6.12.12-2194681.el8_preempt/
├── amd/amdkfd/kfd_chardev.c         # IOCTL入口
├── amd/amdkfd/kfd_process.c         # 进程管理
├── amd/amdkfd/kfd_queue.c           # Queue管理
├── amd/amdkfd/kfd_device_queue_manager.c  # DQM
└── include/uapi/linux/kfd_ioctl.h   # IOCTL定义
```

---

## 💡 下一步行动

### 本周任务（优先级排序）

1. **创建Queue查询工具** ⭐⭐⭐⭐⭐
   - 实现`find_queues_by_pid()`
   - 测试能否正确识别Queue

2. **创建内核模块框架** ⭐⭐⭐⭐⭐
   - 实现基本的模块加载/卸载
   - 添加sysfs接口

3. **实现stop_sched调用** ⭐⭐⭐⭐⭐
   - 调用`amdgpu_amdkfd_stop_sched`
   - 验证Queue是否停止

4. **实现resume_sched调用** ⭐⭐⭐⭐
   - 调用`amdgpu_amdkfd_resume_sched`
   - 验证Queue是否恢复

5. **创建测试脚本** ⭐⭐⭐⭐
   - 自动化测试流程
   - 生成测试报告

---

**总结**: 
- ✅ 分析完成，发现单Queue模型简化了设计
- 🔄 下一步：实现Queue暂停/恢复功能
- 🎯 目标：2周内完成基础POC
- 📊 成功标准：能够实现Case-A抢占Case-B

---

**维护者**: AI Assistant  
**日期**: 2026-02-05  
**状态**: 📋 计划制定完成，等待实施

