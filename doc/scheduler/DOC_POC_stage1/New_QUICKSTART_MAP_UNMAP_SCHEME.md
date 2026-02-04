# 新方案快速开始指南

**方案**: 基于Map/Unmap机制的优化抢占  
**日期**: 2026-02-04  
**预计时间**: 2周  
**难度**: ⭐⭐⭐☆☆（中等，需要内核开发经验）

---

## ⚡ 1分钟理解新方案

### 核心思想

```
传统方案：
  suspend_queues(offline_qids) → 5ms
  resume_queues(offline_qids) → 10ms
  总计：15ms ❌

新方案：
  batch_unmap(offline_qids) → 0.5ms ⭐
  fast_remap(offline_qids) → 0.5ms ⭐
  总计：1ms ✅
  
关键：
  ✅ 利用KFD已有的execute_queues_cpsch（批量操作）
  ✅ 保留MQD，只unmap/remap HQD（快速）
  ✅ HQD资源预留（无竞争）
  
性能提升：15倍 ⭐⭐⭐⭐⭐
```

---

## 🚀 快速开始（2小时上手）

### Step 1: 理解新方案（30分钟）

```bash
# 1. 阅读核心设计
cat New_DESIGN_MAP_UNMAP_BASED_PREEMPTION.md

# 重点关注：
#  - 创新点1-5
#  - 性能对比
#  - 代码示例

# 2. 阅读决策指南
cat New_IMPLEMENTATION_COMPARISON.md

# 确认：
#  - 你的场景适合新方案吗？
#  - 性能要求是什么？
#  - 可以投入2周吗？
```

### Step 2: 环境准备（30分钟）

```bash
# 1. 检查内核源码
ls /usr/src/amdgpu-*/amd/amdkfd/

# 确认文件存在：
#  - kfd_chardev.c
#  - kfd_device_queue_manager.c
#  - include/uapi/linux/kfd_ioctl.h

# 2. 准备开发环境
mkdir -p /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/src/poc_stage1_new
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/src/poc_stage1_new

# 3. 创建目录结构
mkdir -p kernel_patches
mkdir -p libgpreempt_v2
mkdir -p test_framework
mkdir -p tests
mkdir -p results
```

### Step 3: 快速原型（1小时）

```bash
# 1. 创建最小可用的内核patch
cat > kernel_patches/batch_unmap.patch << 'EOF'
# 新增BATCH_UNMAP_QUEUES ioctl的最小实现
# （完整代码见设计文档）
EOF

# 2. 创建用户空间库
cd libgpreempt_v2

# 复制C库模板
cat > gpreempt_poc_v2.h << 'EOF'
// 新增API声明
int gpreempt_batch_unmap_queues(uint32_t *qids, uint32_t num, uint32_t grace);
int gpreempt_fast_remap_queues(uint32_t *qids, uint32_t num);
EOF

cat > gpreempt_poc_v2.c << 'EOF'
// 新增API实现（调用新ioctl）
EOF

# 3. 快速测试
make
./test_batch_unmap
```

---

## 📐 完整实施路线（2周）

### Week 1: 内核开发（Day 1-5）

#### Day 1: 新增ioctl定义

```bash
# 位置：/usr/src/amdgpu-*/amd/amdkfd/include/uapi/linux/kfd_ioctl.h

# 添加ioctl编号（在现有定义后）
#define AMDKFD_IOC_BATCH_UNMAP_QUEUES        0xYY
#define AMDKFD_IOC_FAST_REMAP               0xYZ  
#define AMDKFD_IOC_SET_HQD_RESERVATION       0xZA

# 添加参数结构体
struct kfd_ioctl_batch_unmap_args {
    uint32_t num_queues;
    uint32_t grace_period_us;
    uint32_t flags;
    uint64_t queue_array_ptr;
};

struct kfd_ioctl_fast_remap_args {
    uint32_t num_queues;
    uint64_t queue_array_ptr;
};

struct kfd_ioctl_hqd_reservation_args {
    uint32_t gpu_id;
    uint32_t online_percent;
    uint32_t offline_percent;
};
```

**验证**: 编译通过

#### Day 2: 实现batch_unmap

```bash
# 位置：/usr/src/amdgpu-*/amd/amdkfd/kfd_chardev.c

# 在ioctl switch中添加：
case AMDKFD_IOC_BATCH_UNMAP_QUEUES:
{
    struct kfd_ioctl_batch_unmap_args args;
    // ... 完整实现见设计文档 ...
    
    // 核心：利用execute_queues_cpsch
    ret = execute_queues_cpsch(dqm, ...);
    break;
}
```

**验证**: 
- 编译通过
- 加载模块
- 基本ioctl调用

#### Day 3: 实现fast_remap和hqd_reservation

```bash
# 同样在kfd_chardev.c

case AMDKFD_IOC_FAST_REMAP:
    // ... 实现 ...

case AMDKFD_IOC_SET_HQD_RESERVATION:
    // ... 实现 ...
```

**验证**:
- 功能测试
- 单个队列测试
- 多个队列测试

#### Day 4: HQD预留策略

```bash
# 修改allocate_hqd()的分配策略
# 位置：kfd_device_queue_manager.c line 777

static int allocate_hqd(struct device_queue_manager *dqm, struct queue *q)
{
    // 检查队列类型
    bool is_online = (q->properties.priority >= ONLINE_PRIORITY_THRESHOLD);
    
    int start_pipe, end_pipe;
    
    if (is_online) {
        // Online队列从预留区分配
        start_pipe = 0;
        end_pipe = dqm->hqd_reservation.online_reserved_pipes;
    } else {
        // Offline队列从非预留区分配
        start_pipe = dqm->hqd_reservation.online_reserved_pipes;
        end_pipe = get_pipes_per_mec(dqm);
    }
    
    // 在指定范围内分配
    for (pipe = start_pipe; pipe < end_pipe; pipe++) {
        // ... 原有逻辑 ...
    }
}
```

#### Day 5: 内核测试和调试

```bash
# 重新编译内核模块
cd /usr/src/amdgpu-*/
make -j$(nproc)
sudo make modules_install

# 卸载旧模块
sudo modprobe -r amdgpu

# 加载新模块
sudo modprobe amdgpu

# 验证新ioctl
cd /data/test
./test_new_ioctls

# 功能测试
./test_batch_unmap_single_queue
./test_batch_unmap_multiple_queues
./test_fast_remap
./test_hqd_reservation

# 回归测试
./test_existing_functionality
```

---

### Week 2: 用户空间开发（Day 6-10）

#### Day 6: libgpreempt_poc_v2.so

```bash
cd libgpreempt_v2/

# 实现新API封装
cat > gpreempt_poc_v2.c << 'EOF'
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/kfd_ioctl.h>
#include "gpreempt_poc_v2.h"

static int kfd_fd = -1;

int gpreempt_poc_init(void) {
    kfd_fd = open("/dev/kfd", O_RDWR);
    return (kfd_fd >= 0) ? 0 : -1;
}

int gpreempt_batch_unmap_queues(uint32_t *qids, uint32_t num, uint32_t grace_us)
{
    struct kfd_ioctl_batch_unmap_args args = {
        .num_queues = num,
        .grace_period_us = grace_us,
        .queue_array_ptr = (uint64_t)qids
    };
    
    return ioctl(kfd_fd, AMDKFD_IOC_BATCH_UNMAP_QUEUES, &args);
}

int gpreempt_fast_remap_queues(uint32_t *qids, uint32_t num)
{
    struct kfd_ioctl_fast_remap_args args = {
        .num_queues = num,
        .queue_array_ptr = (uint64_t)qids
    };
    
    return ioctl(kfd_fd, AMDKFD_IOC_FAST_REMAP, &args);
}

int gpreempt_set_hqd_reservation(uint32_t gpu_id, 
                                 uint32_t online_pct,
                                 uint32_t offline_pct)
{
    struct kfd_ioctl_hqd_reservation_args args = {
        .gpu_id = gpu_id,
        .online_percent = online_pct,
        .offline_percent = offline_pct
    };
    
    return ioctl(kfd_fd, AMDKFD_IOC_SET_HQD_RESERVATION, &args);
}

// ... 其他函数 ...
EOF

# 编译
make clean && make

# 测试
./test_lib_v2
```

#### Day 7: Python Framework

```bash
cd ../test_framework/

# 创建智能调度器
cat > smart_queue_scheduler.py << 'EOF'
#!/usr/bin/env python3

import ctypes
import time
import threading
from typing import List

lib = ctypes.CDLL('../libgpreempt_v2/libgpreempt_poc_v2.so')

class HQDResourceMonitor:
    """HQD资源监控"""
    # ... 完整实现见设计文档 ...

class SmartQueueScheduler:
    """智能队列调度器"""
    # ... 完整实现见设计文档 ...
EOF

# 创建测试模型
cat > simple_models.py << 'EOF'
# Online和Offline简单模型
# ... 代码 ...
EOF

# 测试
python3 smart_queue_scheduler.py
```

#### Day 8-9: 完整测试

```bash
cd ../tests/

# Day 8: 功能测试
python3 test_basic_preemption_v2.py
python3 test_batch_operations.py
python3 test_hqd_reservation.py

# Day 9: 性能测试
python3 test_latency_comparison.py
python3 test_throughput_comparison.py

# 对比数据：新方案 vs 传统方案
python3 generate_comparison_report.py
```

#### Day 10: 文档和报告

```bash
# 生成测试报告
python3 tools/generate_test_report.py > ../results/final_report.md

# 性能对比图表
python3 tools/plot_performance.py

# 决策建议
python3 tools/stage2_recommendation.py
```

---

## 📝 最小可行实现（MVP）

如果时间紧张，可以先实现MVP（最小可行版本）：

### MVP功能清单

```
内核侧（必须）：
  ✅ BATCH_UNMAP_QUEUES ioctl
  ❌ FAST_REMAP (先用传统resume)
  ❌ HQD_RESERVATION (先用默认分配)

用户空间（必须）：
  ✅ batch_unmap封装
  ✅ 简单的调度器（无HQD监控）
  ✅ 基本测试

可选功能（后续添加）：
  □ FAST_REMAP优化
  □ HQD资源预留
  □ 实时资源监控
  □ 高级策略
```

### MVP实施时间

```
Day 1-2: 内核BATCH_UNMAP_QUEUES
Day 3:   用户空间库
Day 4:   Python调度器（简化版）
Day 5:   测试和验证
────────────────────────────
总计：1周

性能：
  batch_unmap: ~0.5ms ✓
  resume: ~10ms（用传统方式）
  总计: ~10.5ms

vs传统方案(15ms)
改进：30%

评估：
  ✓ 证明batch操作的价值
  ✓ 降低50%开发时间
  ✓ 后续可以添加fast_remap
```

---

## 🛠️ 开发工具和脚本

### 工具1: 内核patch生成器

```bash
#!/bin/bash
# generate_kernel_patch.sh

cat > /tmp/batch_unmap_ioctl.patch << 'EOFPATCH'
diff --git a/amd/amdkfd/kfd_chardev.c b/amd/amdkfd/kfd_chardev.c
index xxxxx..yyyyy 100644
--- a/amd/amdkfd/kfd_chardev.c
+++ b/amd/amdkfd/kfd_chardev.c
@@ -xxxx,6 +xxxx,50 @@ static long kfd_ioctl(struct file *filep, unsigned int cmd, unsigned long arg)
 
+       case AMDKFD_IOC_BATCH_UNMAP_QUEUES:
+       {
+               struct kfd_ioctl_batch_unmap_args args;
+               // ... 完整实现 ...
+               break;
+       }
+
 default:
     return -ENOTTY;
 }
EOFPATCH

echo "✅ Patch生成完成: /tmp/batch_unmap_ioctl.patch"
echo ""
echo "应用patch:"
echo "  cd /usr/src/amdgpu-*/"
echo "  patch -p1 < /tmp/batch_unmap_ioctl.patch"
```

### 工具2: 快速测试脚本

```bash
#!/bin/bash
# quick_test_new_api.sh

echo "🧪 快速测试新API..."

# 测试BATCH_UNMAP
echo "测试batch_unmap..."
./test_batch_unmap 0 1 2  # Queue IDs: 0,1,2

if [ $? -eq 0 ]; then
    echo "✅ batch_unmap工作正常"
else
    echo "❌ batch_unmap失败"
    exit 1
fi

# 测试FAST_REMAP
echo "测试fast_remap..."
./test_fast_remap 0 1 2

if [ $? -eq 0 ]; then
    echo "✅ fast_remap工作正常"
else
    echo "❌ fast_remap失败"
    exit 1
fi

echo ""
echo "✅ 所有API测试通过！"
```

### 工具3: 性能对比测试

```python
#!/usr/bin/env python3
# performance_comparison.py

import ctypes
import time
import numpy as np

# 加载两个库
lib_old = ctypes.CDLL('./libgpreempt_poc.so')      # 传统方案
lib_new = ctypes.CDLL('./libgpreempt_poc_v2.so')   # 新方案

def test_traditional_scheme(queue_ids, iterations=100):
    """测试传统方案"""
    latencies = []
    
    for i in range(iterations):
        start = time.time()
        
        # Suspend
        lib_old.gpreempt_suspend_queues(
            (ctypes.c_uint32 * len(queue_ids))(*queue_ids),
            len(queue_ids),
            1000
        )
        
        # 模拟Online执行
        time.sleep(0.01)  # 10ms
        
        # Resume
        lib_old.gpreempt_resume_queues(
            (ctypes.c_uint32 * len(queue_ids))(*queue_ids),
            len(queue_ids)
        )
        
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        
        time.sleep(0.1)  # 间隔
    
    return np.array(latencies)


def test_new_scheme(queue_ids, iterations=100):
    """测试新方案"""
    latencies = []
    
    for i in range(iterations):
        start = time.time()
        
        # Batch Unmap
        lib_new.gpreempt_batch_unmap_queues(
            (ctypes.c_uint32 * len(queue_ids))(*queue_ids),
            len(queue_ids),
            100  # 更短的grace period
        )
        
        # 模拟Online执行
        time.sleep(0.01)  # 10ms
        
        # Fast Remap
        lib_new.gpreempt_fast_remap_queues(
            (ctypes.c_uint32 * len(queue_ids))(*queue_ids),
            len(queue_ids)
        )
        
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        
        time.sleep(0.1)  # 间隔
    
    return np.array(latencies)


def main():
    print("╔════════════════════════════════════════╗")
    print("║  性能对比测试：传统 vs 新方案          ║")
    print("╚════════════════════════════════════════╝")
    print("")
    
    # 初始化
    lib_old.gpreempt_poc_init()
    lib_new.gpreempt_poc_init()
    
    # 测试队列
    test_queue_ids = [0, 1, 2]  # 3个队列
    
    # 测试传统方案
    print("🧪 测试传统方案（100次）...")
    traditional_latencies = test_traditional_scheme(test_queue_ids)
    
    # 测试新方案
    print("🧪 测试新方案（100次）...")
    new_latencies = test_new_scheme(test_queue_ids)
    
    # 统计分析
    print("\n╔════════════════════════════════════════╗")
    print("║  性能对比结果                           ║")
    print("╚════════════════════════════════════════╝")
    print("")
    
    print("传统方案：")
    print(f"  平均延迟: {np.mean(traditional_latencies):.2f} ms")
    print(f"  P50: {np.percentile(traditional_latencies, 50):.2f} ms")
    print(f"  P95: {np.percentile(traditional_latencies, 95):.2f} ms")
    print(f"  P99: {np.percentile(traditional_latencies, 99):.2f} ms")
    print(f"  最大: {np.max(traditional_latencies):.2f} ms")
    
    print("\n新方案：")
    print(f"  平均延迟: {np.mean(new_latencies):.2f} ms")
    print(f"  P50: {np.percentile(new_latencies, 50):.2f} ms")
    print(f"  P95: {np.percentile(new_latencies, 95):.2f} ms")
    print(f"  P99: {np.percentile(new_latencies, 99):.2f} ms")
    print(f"  最大: {np.max(new_latencies):.2f} ms")
    
    print("\n性能提升：")
    speedup = np.mean(traditional_latencies) / np.mean(new_latencies)
    print(f"  加速比: {speedup:.1f}x ⭐⭐⭐⭐⭐")
    
    improvement = (np.mean(traditional_latencies) - np.mean(new_latencies)) / np.mean(traditional_latencies) * 100
    print(f"  延迟降低: {improvement:.1f}%")
    
    # 清理
    lib_old.gpreempt_poc_cleanup()
    lib_new.gpreempt_poc_cleanup()

if __name__ == '__main__':
    main()
```

#### Day 8: AI模型集成测试

```python
#!/usr/bin/env python3
# test_real_ai_models.py

import torch
import torch.nn as nn
import subprocess
import time

def test_with_real_models():
    """使用真实AI模型测试"""
    
    print("╔════════════════════════════════════════╗")
    print("║  真实AI模型抢占测试                     ║")
    print("╚════════════════════════════════════════╝")
    print("")
    
    # 1. 启动Offline训练模型（BERT-like）
    print("🚀 启动Offline训练模型...")
    offline_proc = subprocess.Popen([
        'python3', 'models/bert_training.py',
        '--epochs', '1000',
        '--priority', '2'
    ])
    
    time.sleep(3)  # 等待模型加载
    
    # 2. 初始化新方案调度器
    from smart_queue_scheduler import SmartQueueScheduler
    sched = SmartQueueScheduler()
    
    # 扫描并注册Offline队列
    offline_queues = sched.discover_queues_by_priority(0, 5)
    print(f"✅ 发现{len(offline_queues)}个Offline队列")
    for q in offline_queues:
        sched.register_offline_queue(q)
    
    # 3. 创建Online推理模型（简单ResNet）
    print("\n🚀 加载Online推理模型...")
    online_model = torch.hub.load('pytorch/vision:v0.10.0', 
                                  'resnet18', 
                                  pretrained=True).cuda()
    online_model.eval()
    
    # 4. 模拟Online请求（20次）
    print("\n📊 开始抢占测试...")
    test_input = torch.randn(1, 3, 224, 224).cuda()
    
    for i in range(20):
        print(f"\n=== 请求 {i+1}/20 ===")
        
        start = time.time()
        
        # 触发抢占
        sched.handle_online_request()
        
        # Online推理
        with torch.no_grad():
            output = online_model(test_input)
        
        end_to_end = (time.time() - start) * 1000
        print(f"  端到端延迟: {end_to_end:.2f} ms")
        
        time.sleep(0.5)
    
    # 5. 统计和清理
    sched.print_statistics()
    sched.cleanup()
    offline_proc.terminate()
    
    print("\n✅ 测试完成！")

if __name__ == '__main__':
    test_with_real_models()
```

#### Day 10: 结果分析和报告

```bash
# 生成完整报告
cd ../

python3 tools/generate_final_report.py \
    --traditional-results results/traditional/ \
    --new-results results/new_scheme/ \
    --output results/NEW_SCHEME_FINAL_REPORT.md

# 应该包含：
# 1. 功能对比
# 2. 性能对比（表格+图表）
# 3. 资源利用率对比
# 4. 稳定性测试结果
# 5. 升级到Stage 2的建议
```

---

## ✅ 检查清单

### 开发前检查

- [ ] 已完全理解Map/Unmap机制
- [ ] 已阅读设计文档
- [ ] 内核源码可以访问和修改
- [ ] 有内核开发经验
- [ ] 有2周开发时间

### Week 1检查（Day 5）

- [ ] 3个新ioctl已实现
- [ ] 内核编译通过
- [ ] 模块加载成功
- [ ] 基本功能测试通过
- [ ] 无现有功能回归

### Week 2检查（Day 10）

- [ ] libgpreempt_poc_v2.so工作正常
- [ ] Python Framework完成
- [ ] 所有测试用例通过
- [ ] 性能数据收集完成
- [ ] 对比报告已生成

### 最终验收

- [ ] 功能：Online能抢占Offline ✓
- [ ] 性能：延迟<5ms ✓
- [ ] 稳定性：1小时无错误 ✓
- [ ] 文档：完整测试报告 ✓
- [ ] 决策：是否升级到Stage 2？

---

## 🐛 常见问题和解决

### Q1: 编译内核失败

```bash
# 检查依赖
sudo yum install kernel-devel gcc make

# 清理后重新编译
cd /usr/src/amdgpu-*/
make clean
make -j$(nproc)

# 查看错误日志
dmesg | tail -100
```

### Q2: ioctl返回-EINVAL

```bash
# 检查ioctl编号
grep "AMDKFD_IOC_BATCH_UNMAP" include/uapi/linux/kfd_ioctl.h

# 检查参数
./test_ioctl_params

# 查看内核日志
dmesg | grep -i "batch_unmap"
```

### Q3: batch_unmap没有效果

```bash
# 检查队列是否真的变inactive
cat /sys/kernel/debug/kfd/mqds | grep -A 5 "Queue ID: 0"
# 应该看到 "is active: no"

# 检查HQD是否释放
cat /sys/kernel/debug/kfd/hqds | grep "CP_HQD_ACTIVE"
# 应该看到 0x00000000
```

---

## 📊 性能基准

### 预期性能（基于理论分析）

```
单队列操作：
  batch_unmap:   0.5ms  (vs 传统5ms)
  fast_remap:    0.5ms  (vs 传统10ms)
  端到端:        11ms   (vs 传统15ms)

批量操作（10队列）：
  batch_unmap:   0.5ms  (vs 传统50ms)  ⭐⭐⭐⭐⭐
  fast_remap:    1ms    (vs 传统100ms) ⭐⭐⭐⭐⭐
  端到端:        11.5ms (vs 传统150ms) ⭐⭐⭐⭐⭐

资源利用：
  HQD利用率:     85-90% (vs 传统60-70%)
  超额订阅:      支持    (vs 传统不支持)
```

### 实际测试目标

```
必须达成：
  ✅ batch_unmap < 1ms
  ✅ fast_remap < 1ms
  ✅ Online延迟 < 15ms

应该达成：
  ✅ batch_unmap < 0.5ms
  ✅ fast_remap < 0.5ms
  ✅ Online延迟 < 10ms

最好达成：
  ✅ batch_unmap < 0.3ms
  ✅ fast_remap < 0.3ms
  ✅ Online延迟 < 5ms
```

---

## 🎯 成功的标志

### MVP成功

```
✅ batch_unmap工作
✅ 延迟<10ms（vs传统15ms）
✅ 基本稳定
✅ 证明批量操作的价值

→ 继续完成fast_remap和HQD预留
```

### 完整方案成功

```
✅ 所有3个ioctl工作
✅ 延迟<5ms
✅ 资源利用率>80%
✅ 稳定性测试通过
✅ 性能提升10-150倍

→ 可以升级到Stage 2或直接使用
```

---

## 📚 必读文档顺序

### 实施前（1小时）

```
1. New_DESIGN_MAP_UNMAP_BASED_PREEMPTION.md (30分钟)
   → 理解核心设计

2. New_IMPLEMENTATION_COMPARISON.md (20分钟)
   → 确认适合你的场景

3. New_QUICKSTART_MAP_UNMAP_SCHEME.md (10分钟)
   → 本文档，实施步骤
```

### 实施中（边做边查）

```
1. MAP_UNMAP_DETAILED_PROCESS.md
   → 理解内核函数调用链

2. SW_QUEUE_HW_QUEUE_MAPPING_MECHANISM.md
   → 理解MQD/HQD关系

3. kfd_device_queue_manager.c
   → 查看实际代码
```

---

## 🚦 立即行动

### 如果你现在就要开始：

```bash
# 1. 进入工作目录
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1

# 2. 创建代码目录
mkdir -p ../../src/poc_stage1_new
cd ../../src/poc_stage1_new

# 3. 开始Day 1任务
# 按照上面的"Week 1: 内核开发"执行

echo "🚀 新方案开发开始！"
echo ""
echo "当前任务：Day 1 - 新增ioctl定义"
echo "  1. 编辑 kfd_ioctl.h"
echo "  2. 添加3个ioctl定义"
echo "  3. 添加参数结构体"
echo "  4. 编译测试"
```

---

## 💡 专家建议

### 建议1: 先实现MVP

```
不要一次实现所有功能
先实现核心的batch_unmap
验证概念和性能
再添加fast_remap和hqd_reservation

优点：
  ✅ 降低初始复杂度
  ✅ 快速看到成果（1周）
  ✅ 渐进式风险
```

### 建议2: 充分测试

```
内核修改需要谨慎：
  1. 单元测试每个ioctl
  2. 回归测试现有功能
  3. 压力测试稳定性
  4. 在测试GPU上先运行
  5. 确认无问题再部署到生产GPU
```

### 建议3: 保留fallback

```
保留传统方案的代码：
  - 如果新方案有问题
  - 可以快速回退
  - 降低风险
  
实现：
  if (new_scheme_available && !new_scheme_failed) {
      use_new_scheme();
  } else {
      use_traditional_scheme();  // fallback
  }
```

---

## 📞 获取帮助

### 如果遇到问题：

**内核编译问题**:
- 查看 DKMS日志: `/var/lib/dkms/amdgpu/*/build/make.log`

**ioctl调用失败**:
- 查看内核日志: `dmesg | tail -100`
- 检查返回值: `strerror(errno)`

**性能未达预期**:
- 使用ftrace跟踪: 
  ```bash
  echo 1 > /sys/kernel/debug/tracing/events/kfd/enable
  cat /sys/kernel/debug/tracing/trace
  ```

**不确定如何继续**:
- 重新阅读设计文档
- 查看代码示例
- 对比传统方案

---

**创建时间**: 2026-02-04  
**难度**: ⭐⭐⭐ (需要内核开发经验)  
**时间**: 2周（完整版）或1周（MVP）  
**推荐度**: ⭐⭐⭐⭐⭐（如果需要高性能）

**准备好开始了吗？从理解设计文档开始！** 🚀
