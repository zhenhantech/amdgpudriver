# ftrace + AMD日志关联分析指南

**日期**: 2026-02-05  
**目的**: 分析ROCm runtime和KFD的完整交互流程

---

## 📋 概述

### 测试目标

通过同时捕获**AMD日志（ROCm runtime层）**和**ftrace日志（KFD内核层）**，分析：

1. **Queue创建流程**: ROCr如何通过KFD创建Hardware Queue
2. **MQD管理**: MQD在ROCr和KFD之间的传递
3. **KCQ使用**: Kernel Command Queue的分配和使用
4. **Doorbell提交**: 用户空间Doorbell如何触发KFD处理
5. **完整调用链**: 用户空间 → ROCr → KFD → GPU硬件

### 数据来源

```
┌─────────────────────┐
│  PyTorch (GEMM)     │
│  test_gemm_mini.py  │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  ROCm Runtime       │  ← AMD_LOG_LEVEL=5
│  (HIP/HSA/ROCr)     │     捕获详细日志
└──────────┬──────────┘
           │ IOCTLs
           ↓
┌─────────────────────┐
│  KFD (Kernel)       │  ← ftrace
│  - Queue管理        │     function/event trace
│  - MQD处理          │
│  - Doorbell处理     │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  GPU Hardware       │
│  - CP Scheduler     │
│  - HQD寄存器        │
└─────────────────────┘
```

---

## 🚀 快速开始

### 运行测试

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 一键运行（自动配置ftrace + 运行测试）
sudo ./run_gemm_with_ftrace.sh zhen_vllm_dsv3
```

**输出**:
- `log/gemm_ftrace_<timestamp>/gemm_amd_log.txt` - AMD日志
- `log/gemm_ftrace_<timestamp>/ftrace.txt` - ftrace日志
- `log/gemm_ftrace_<timestamp>/analyze.sh` - 快速分析脚本

### 快速分析

```bash
# 自动分析
cd log/gemm_ftrace_<timestamp>
./analyze.sh
```

---

## 🔍 手动分析步骤

### 步骤1: 提取Queue创建流程

#### AMD日志（ROCr层）

```bash
# 查找Queue获取
grep 'acquireQueue' gemm_amd_log.txt

# 示例输出:
# :3:rocdevice.cpp:3045: 175037104827 us: [pid:157801 tid: 0x7fb0621f8480] 
# acquireQueue refCount: 0x7fad66c00000 (1)
#                        ^^^^^^^^^^^^^^^^
#                        Hardware Queue地址
```

**关键信息**:
- `acquireQueue`: ROCr获取Hardware Queue
- `refCount: 0x7fad66c00000`: HQD地址
- 时间戳: `175037104827 us` (微秒)

#### ftrace日志（KFD层）

```bash
# 查找Queue创建相关函数
grep -i 'create.*queue\|queue.*create' ftrace.txt

# 或者查找你添加的自定义trace point
grep -i 'mqd\|kcq' ftrace.txt

# 示例输出:
#  python3-157801 [005] .... 175037.104830: kfd_create_queue <-kfd_ioctl
#                            ^^^^^^^^^^^^^^^
#                            Queue创建函数
```

**关键函数**:
- `kfd_create_queue`: KFD创建Queue
- `kfd_ioctl`: IOCTL入口
- `amdgpu_amdkfd_map_gtt_bo_to_kcq`: 映射KCQ

#### 时间关联

```bash
# AMD日志时间: 175037104827 us = 175037.104827 秒
# ftrace时间:   175037.104830 秒
# 差异: 3微秒 ← 几乎同时！
```

---

### 步骤2: 分析MQD传递

#### MQD结构

```c
// MQD (Memory Queue Descriptor)
struct mqd {
    uint32_t cp_hqd_pq_base;      // Queue Base Address
    uint32_t cp_hqd_pq_base_hi;
    uint32_t cp_hqd_pq_control;
    uint32_t cp_hqd_ib_control;
    uint32_t cp_hqd_vmid;
    // ... 更多字段
};
```

#### AMD日志中的MQD信息

```bash
# 查找Dispatch信息（包含Queue配置）
grep 'Dispatch Header\|grid=\|workgroup=' gemm_amd_log.txt | head -20
```

**示例输出**:
```
SWq=0x7faf945b8000,      ← Software Queue
HWq=0x7fad66c00000,      ← Hardware Queue (MQD地址)
id=1,                    ← Queue ID
grid=[20480, 1, 1], 
workgroup=[256, 1, 1],
```

#### ftrace中的MQD操作

```bash
# 查找你添加的MQD trace point
grep 'MQD\|mqd' ftrace.txt | head -20
```

**期望看到**:
- MQD allocation
- MQD initialization
- MQD写入GPU

---

### 步骤3: 追踪Kernel提交流程

#### 完整流程

```
PyTorch: torch.matmul(A, B)
    ↓
HIP: hipLaunchKernel()
    ↓
ROCr: hsa_signal_store_relaxed()    ← AMD日志可见
    ↓
Doorbell: Write to MMIO             ← 用户空间，无日志
    ↓
KFD: kfd_doorbell_interrupt()       ← ftrace可见
    ↓
GPU: CP Scheduler处理
```

#### AMD日志: Kernel提交

```bash
# 查找Kernel提交
grep 'KernelExecution.*enqueued' gemm_amd_log.txt | head -5

# 示例:
# :5:command.cpp:355: 175037138308 us: [pid:157801 tid: 0x7fb0621f8480] 
# Command (KernelExecution) enqueued: 0xd17f170 to queue: 0xbe00d60
```

**关键信息**:
- `KernelExecution enqueued`: Kernel已入队
- `queue: 0xbe00d60`: 软件Queue对象
- 时间: `175037138308 us`

#### ftrace: Doorbell处理（如果可见）

```bash
# 查找Doorbell相关函数
grep -i 'doorbell' ftrace.txt

# 可能的函数:
# - amdgpu_doorbell_get_kfd_info
# - amdgpu_doorbell_index_on_bar
# - kfd_signal_event_interrupt (Doorbell触发)
```

---

### 步骤4: 分析KCQ使用

#### 什么是KCQ？

**KCQ (Kernel Command Queue)**:
- 内核空间的命令队列
- 用于内核驱动提交命令到GPU
- 与用户空间Queue（通过Doorbell）不同

#### 查找KCQ分配

```bash
# AMD日志中的KCQ引用
grep -i 'kcq' gemm_amd_log.txt

# ftrace中的KCQ操作
grep -i 'kcq' ftrace.txt

# 期望看到:
# - map_gtt_bo_to_kcq: 映射GTT buffer到KCQ
# - kcq allocation
# - kcq使用统计
```

#### 如果你添加了自定义trace point

```bash
# 查找自定义KCQ trace
grep 'trace_kfd_kcq' ftrace.txt

# 或者使用event trace
grep 'kfd/kcq' ftrace.txt
```

---

### 步骤5: 关联分析ROCr和KFD

#### 时间戳对齐

**AMD日志时间格式**:
```
175037104827 us = 175037.104827 秒 (从系统启动开始)
```

**ftrace时间格式**:
```
175037.104830  (秒.微秒，从系统启动开始)
```

#### 对齐示例

```bash
# 1. 从AMD日志提取关键事件和时间
grep 'acquireQueue\|KernelExecution.*enqueued' gemm_amd_log.txt | \
    awk -F: '{print $3}' | \
    sed 's/ us.*//' | \
    awk '{printf "%.6f\n", $1/1000000}'

# 输出（秒）:
# 175037.104827  <- acquireQueue
# 175037.138308  <- Kernel提交

# 2. 在ftrace中查找对应时刻的事件
awk '$3 >= 175037.104 && $3 <= 175037.140' ftrace.txt
```

#### 生成时间线

```bash
#!/bin/bash
# timeline.sh - 生成时间线

echo "时间 (秒)    | 层级 | 事件"
echo "─────────────┼──────┼────────────────────────────"

# AMD事件
grep 'acquireQueue\|KernelExecution' gemm_amd_log.txt | \
    awk -F: '{print $3, "| ROCr |", $0}' | \
    sed 's/ us:.*//' | \
    awk '{printf "%.6f | ROCr | %s\n", $1/1000000, substr($0, index($0,$3))}'

# ftrace事件
grep 'kfd.*queue\|kfd.*kernel' ftrace.txt | \
    awk '{printf "%s | KFD  | %s\n", $3, $4}'
```

**输出示例**:
```
时间 (秒)      | 层级  | 事件
───────────────┼───────┼────────────────────────────
175037.104827  | ROCr  | acquireQueue refCount: 0x7fad66c00000
175037.104830  | KFD   | kfd_create_queue
175037.104835  | KFD   | kfd_init_mqd
175037.138308  | ROCr  | KernelExecution enqueued: 0xd17f170
175037.138315  | KFD   | kfd_doorbell_interrupt
```

---

## 📊 关键分析点

### 分析点1: Queue创建延迟

```bash
# ROCr请求时间
ROCR_TIME=$(grep 'acquireQueue' gemm_amd_log.txt | head -1 | awk -F: '{print $3}' | sed 's/ us.*//')

# KFD完成时间
KFD_TIME=$(grep 'kfd_create_queue' ftrace.txt | head -1 | awk '{print $3}')

# 计算延迟
echo "ROCr时间: $(echo "scale=6; $ROCR_TIME/1000000" | bc) 秒"
echo "KFD时间:  $KFD_TIME 秒"
echo "延迟: $((KFD_TIME - ROCR_TIME/1000000)) 秒"
```

### 分析点2: MQD使用情况

```bash
# 统计MQD相关操作
echo "=== MQD操作统计 ==="
grep -c 'mqd' ftrace.txt || echo "0"

# 如果有自定义trace point
echo "MQD创建:"
grep 'mqd.*create\|create.*mqd' ftrace.txt | wc -l

echo "MQD更新:"
grep 'mqd.*update\|update.*mqd' ftrace.txt | wc -l
```

### 分析点3: KCQ vs 用户Queue

```bash
# 统计KCQ使用
echo "=== Queue类型统计 ==="
echo "KCQ操作:"
grep -c 'kcq' ftrace.txt || echo "0"

echo "用户Queue操作:"
grep -c 'user.*queue\|queue.*user' ftrace.txt || echo "0"
```

### 分析点4: Doorbell频率

```bash
# 统计Doorbell
echo "=== Doorbell统计 ==="
DOORBELL_COUNT=$(grep -c 'doorbell' ftrace.txt 2>/dev/null || echo "0")
KERNEL_COUNT=$(grep -c 'KernelExecution.*enqueued' gemm_amd_log.txt)

echo "Doorbell事件: $DOORBELL_COUNT"
echo "Kernel提交:   $KERNEL_COUNT"

if [ $DOORBELL_COUNT -gt 0 ] && [ $KERNEL_COUNT -gt 0 ]; then
    echo "比率: $(echo "scale=2; $DOORBELL_COUNT/$KERNEL_COUNT" | bc)"
fi
```

---

## 🎯 期望发现

### 1. Queue创建流程

**预期**:
```
ROCr: acquireQueue()
    ↓ (< 1ms)
KFD: kfd_create_queue()
    ↓
KFD: kfd_init_mqd()      ← 初始化MQD
    ↓
KFD: program_sh_mem_settings()
    ↓
KFD: map_to_gpu()
    ↓
完成
```

### 2. MQD配置

**预期看到**:
- MQD allocation
- MQD初始化（配置Queue参数）
- MQD写入GPU寄存器

### 3. KCQ使用

**预期**:
- 系统启动时分配KCQ（如8个）
- GEMM测试不直接使用KCQ（用户Queue通过Doorbell）
- KCQ主要用于内核驱动的管理操作

### 4. Doorbell vs IOCTL

**用户空间Doorbell提交**:
- 延迟低（MMIO写入）
- ftrace可能看不到（硬件直接处理）
- 只在Doorbell中断时可见

**IOCTL提交** (如果没有Doorbell):
- 延迟高（系统调用）
- ftrace清晰可见
- 每次提交都有IOCTL

---

## 💡 故障排查

### 问题1: ftrace日志为空或很少

**可能原因**:
1. ftrace过滤器设置错误
2. KFD模块名不匹配
3. 自定义trace point未编译

**解决**:
```bash
# 检查KFD模块
lsmod | grep amdgpu

# 清空过滤器，捕获所有
sudo sh -c 'echo > /sys/kernel/debug/tracing/set_ftrace_filter'

# 检查可用的trace events
ls /sys/kernel/debug/tracing/events/ | grep -i kfd
```

### 问题2: 时间戳不匹配

**可能原因**:
- AMD日志和ftrace使用不同的时间基准

**解决**:
```bash
# 使用相对时间（第一个事件作为0点）
# 或使用进程PID关联
```

### 问题3: 看不到MQD/KCQ信息

**可能原因**:
- 自定义trace point未添加或未启用

**验证**:
```bash
# 检查是否有自定义events
ls /sys/kernel/debug/tracing/events/kfd/ 2>/dev/null

# 如果没有，需要在KFD源码中添加trace points
```

---

## 📚 参考资料

### KFD源码关键文件

```
/usr/src/amdgpu-6.12.12-2194681.el8_preempt/amd/amdkfd/
├── kfd_device_queue_manager.c    # Queue管理
├── kfd_mqd_manager.c              # MQD管理
├── kfd_packet_manager.c           # 包管理
├── kfd_doorbell.c                 # Doorbell处理
└── kfd_queue.c                    # Queue操作
```

### 添加自定义trace point示例

```c
// 在kfd_device_queue_manager.c中添加
#include <trace/events/kfd.h>

int create_queue(struct device_queue_manager *dqm, ...) {
    trace_kfd_create_queue_start(queue_id);
    
    // ... queue创建逻辑 ...
    
    trace_kfd_create_queue_end(queue_id, mqd_addr);
    return 0;
}
```

---

## ✅ 总结检查清单

分析完成后，你应该能回答：

- [ ] Queue创建流程中，ROCr和KFD的调用顺序？
- [ ] MQD在哪里分配？如何传递到GPU？
- [ ] KCQ是否被GEMM测试使用？
- [ ] Doorbell是否可见？如果不可见，为什么？
- [ ] 用户空间到GPU的完整数据路径？
- [ ] 关键操作的延迟（Queue创建、Kernel提交）？

---

**维护者**: AI Assistant  
**日期**: 2026-02-05  
**版本**: 1.0
