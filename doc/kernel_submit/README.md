# AI Kernel Submission 流程文档

**目录**: `/mnt/md0/zhehan/code/rampup_doc/Topic_kernel_submission`  
**创建时间**: 2025-01-XX  
**目的**: 详细描述ROCm/HIP环境下AI kernel从应用层到硬件执行的完整提交流程

---

## 文档概述

本目录包含AI kernel submission流程的详细技术文档，基于`/mnt/md0/zhehan/code/rampup_doc/2PORC_profiling`中的DRIVER_xx系列分析文档整理而成。

---

## 文档列表

### 1. AI_KERNEL_SUBMISSION_FLOW.md

**主要文档** - AI Kernel Submission流程详解

**内容**:
- 完整的10层调用栈（从应用层到硬件层）
- 关键机制详解（AQL Queue、Doorbell、MES调度器、驱动层Ring）
- 关键代码位置汇总
- 关键发现总结
- 性能影响分析
- 验证方法

**关键发现**:
- ✅ **90%的kernel提交使用doorbell机制**，不经过驱动层的Ring
- ✅ **MES是硬件调度器**，kernel提交直接通过doorbell
- ✅ **AQL Queue是用户空间队列**，通过doorbell通知硬件
- ✅ **驱动层Ring主要用于SDMA操作**，Compute kernel不经过驱动层Ring

---

### 2. CONTEXT_DETAILED_ANALYSIS.md

**Context详解文档** - GPU Context的概念、作用和管理机制

**内容**:
- Context基本概念和定义
- Context数据结构和生命周期
- Context与Entity的关系
- Context与硬件资源的关系
- Context与多进程的关系
- Context关键代码位置
- Context性能影响和最佳实践

**关键概念**:
- ✅ **Context是进程级的概念**：每个进程对应一个Context
- ✅ **Context管理Entity**：每个Context最多可以有4个Compute Entity和2个SDMA Entity
- ✅ **Context生命周期**：从进程创建到进程结束
- ✅ **Context与硬件资源**：通过Entity绑定到Ring和调度器

---

### 3. CODE_REFERENCE.md

**代码参考文档** - 关键代码位置和实现细节

**内容**:
- ROCm Runtime层代码
- 驱动层代码
- MES调度器代码
- HSA Runtime代码
- FlashInfer代码
- 关键数据结构
- 关键系统调用

**用途**: 提供关键代码的详细位置和实现细节，便于深入理解kernel submission机制

---

## 快速开始

### 阅读顺序

1. **首先阅读**: `AI_KERNEL_SUBMISSION_FLOW.md`
   - 了解完整的调用流程
   - 理解关键机制（AQL Queue、Doorbell、MES等）
   - 掌握关键发现和性能影响

2. **然后阅读**: `CODE_REFERENCE.md`
   - 查看关键代码位置
   - 理解具体实现细节
   - 深入分析代码逻辑

### 关键概念

#### 1. AQL Queue (Architected Queuing Language Queue)

- **位置**: 用户空间内存
- **用途**: 存储kernel dispatch命令
- **结构**: 64字节固定长度的AQL packet
- **关键字段**: rptr（读指针）、wptr（写指针）、doorbell地址

#### 2. Doorbell机制

- **用途**: 通知GPU硬件有新packet需要处理
- **工作方式**: ROCm Runtime写入doorbell寄存器，硬件检测到更新后从AQL Queue读取packet
- **关键发现**: **90%的kernel提交使用doorbell机制**，不经过驱动层Ring

#### 3. 两种调度器：MES和CPSCH

**核心概念**: ROCm驱动支持两种调度器模式，根据`enable_mes`标志选择：

##### MES调度器 (Micro-Engine Scheduler)

- **类型**: 硬件调度器（新架构）
- **启用条件**: `enable_mes = true`
- **用途**: 管理queue和kernel提交
- **工作方式**: 
  - 检测doorbell更新，从AQL Queue读取packet，调度kernel执行
  - 通过MES Ring提交管理命令（ADD_QUEUE、REMOVE_QUEUE等）
- **关键发现**: 
  - MES Ring用于管理操作（ADD_QUEUE等），不用于kernel提交
  - **90%的kernel提交使用doorbell机制**，不经过驱动层Ring
  - Compute kernel通过doorbell直接提交给MES硬件调度器

##### CPSCH调度器 (Compute Process Scheduler)

- **类型**: 软件调度器（旧架构）
- **启用条件**: `enable_mes = false`
- **用途**: 通过驱动层管理queue和kernel提交
- **工作方式**: 
  - 通过`create_queue_cpsch`创建queue
  - 通过`execute_queues_cpsch`执行queue
  - 经过驱动层Ring和GPU调度器（drm_gpu_scheduler）
- **关键发现**: 
  - 使用驱动层的GPU调度器（drm_gpu_scheduler）
  - 经过驱动层Ring提交，会触发`drm_run_job`事件
  - 主要用于旧架构GPU或MES未启用的情况

**代码判断逻辑**:
```c
if (!dqm->dev->kfd->shared_resources.enable_mes)
    retval = execute_queues_cpsch(dqm, ...);  // 使用CPSCH
else
    retval = add_queue_mes(dqm, q, qpd);      // 使用MES
```

#### 4. 驱动层Ring和调度器

- **驱动层说明**: 这里的"驱动层"主要指**KFD (Kernel Fusion Driver)**，是AMD GPU驱动层的一部分，负责queue管理、context管理等
- **Ring类型**: 软件抽象（Compute Ring、SDMA Ring、MES Ring）
- **调度器机制**: 
  - **GPU调度器（drm_gpu_scheduler）**: 每个Ring对应一个调度器，负责管理job队列和调度
  - **Entity绑定**: Entity绑定到特定的调度器（通过rq），决定使用哪个Ring
  - **调度器选择**: 通过`drm_sched_pick_best`函数选择负载最小的调度器
- **用途**: 
  - **Compute Ring**: 驱动支持但实际很少使用（Compute kernel通过doorbell提交）
  - **SDMA Ring**: 主要用于SDMA操作（内存拷贝等），经过GPU调度器调度
  - **MES Ring**: 用于MES管理操作（ADD_QUEUE、REMOVE_QUEUE等）
- **调度器优化策略**（基于2PROC_DKMS_debug文档）:
  - **V17**: PID-based initial scheduler selection（Entity初始化时基于PID选择初始调度器）
  - **V16+V18**: PID-based offset和tie-breaking（在`drm_sched_pick_best`中实现负载均衡）
  - **目的**: 实现多进程间的负载均衡，避免所有进程选择同一个Ring（如sdma0.0）
- **关键发现**: 
  - Compute kernel不经过驱动层Ring，ftrace显示只有SDMA Ring是正常的
  - SDMA操作经过GPU调度器调度，调度器选择策略影响Ring的负载分布

---

## 完整调用栈（简化版）

```
应用层 (Python)
  ↓ flashinfer.single_prefill_with_kv_cache()
FlashInfer层
  ↓ FlashInfer Python API → JIT模块 → PyTorch Extension
PyTorch层
  ↓ hipLaunchKernel()
ROCm Runtime层 (rocr)
  ↓ hsa_queue_create() → 写入AQL packet → 更新doorbell
驱动层 (KMD)
  ↓ 检测doorbell更新 → 通知MES
硬件层 (MES调度器)
  ↓ 从AQL Queue读取packet → 调度kernel执行
GPU执行单元
  ↓ 执行kernel → 更新completion signal
```

---

## 关键发现总结

### 1. Kernel提交路径（90%使用doorbell）

**主要路径**:
```
ROCm Runtime → 写入AQL packet → 更新doorbell → MES硬件调度器 → GPU执行
```

**关键特征**:
- ✅ **不经过驱动层的Ring**
- ✅ **不触发drm_run_job事件**
- ✅ **直接通过doorbell通知硬件**
- ✅ **硬件从AQL Queue读取packet并执行**
- ✅ **MES硬件调度器负责从AQL Queue读取packet并调度执行**

**调度器说明**:
- **MES (Micro-Engine Scheduler)**: 硬件调度器，负责管理AQL Queue和kernel提交
- **不经过驱动层调度器**: Compute kernel提交完全在硬件层面完成，不经过驱动层的GPU调度器（drm_gpu_scheduler）
- **与CPSCH的区别**: MES是新架构的硬件调度器，CPSCH是旧架构的软件调度器

### 2. SDMA操作路径（通过驱动层Ring和调度器）

**主要路径**:
```
ROCm Runtime → 驱动层SDMA Ring → GPU调度器 → drm_run_job事件 → GPU执行
```

**关键特征**:
- ✅ **通过驱动层SDMA Ring提交**
- ✅ **经过GPU调度器（drm_gpu_scheduler）调度**
- ✅ **触发drm_run_job事件**
- ✅ **ftrace中可以看到SDMA Ring的使用**

**调度器机制**:
- **GPU调度器（drm_gpu_scheduler）**: 每个Ring对应一个调度器，负责管理job队列和调度
  - **注意**: 这是CPSCH模式下的软件调度器，MES模式下Compute kernel不经过此调度器
- **Entity绑定**: Entity绑定到特定的调度器（通过rq），决定使用哪个Ring
- **调度器选择策略**（基于`/mnt/md0/zhehan/code/rampup_doc/2PROC_DKMS_debug`文档）:
  - **V17**: PID-based initial scheduler selection
    - **位置**: `drm_sched_entity_init`函数
    - **机制**: Entity初始化时，基于PID选择初始调度器（`initial_sched_idx = pid % num_sched_list`）
    - **目的**: 在初始化阶段就实现负载均衡，不同进程优先选择不同的调度器
  - **V16+V18**: PID-based offset和tie-breaking
    - **位置**: `drm_sched_pick_best`函数
    - **机制**: 
      - V16: 从PID-based offset开始搜索调度器（`start_idx = pid % num_sched_list`）
      - V18: 相同score时优先选择start_idx的调度器
    - **目的**: 在选择最佳调度器时实现负载均衡，避免所有进程选择同一个Ring
  - **已移除的策略**（因导致softlockup）:
    - **V19**: `drm_sched_entity_push_job`中的调度器选择（已移除）
    - **V20_fixed**: `drm_sched_entity_select_rq`中的queue-not-empty切换逻辑（已移除）
- **调度器与Ring的关系**:
  - 每个SDMA Ring对应一个GPU调度器
  - Entity通过绑定到调度器（rq）来决定使用哪个Ring
  - 调度器选择策略影响Ring的负载分布

### 3. MES vs CPSCH调度器对比

**核心区别**:

| 特性 | MES调度器 | CPSCH调度器 |
|------|----------|------------|
| **类型** | 硬件调度器 | 软件调度器 |
| **启用条件** | `enable_mes = true` | `enable_mes = false` |
| **Compute Kernel提交** | 通过doorbell，不经过KFD驱动层Ring | 经过KFD驱动层Ring和GPU调度器 |
| **SDMA操作提交** | 经过KFD驱动层SDMA Ring和GPU调度器 | 经过KFD驱动层SDMA Ring和GPU调度器 |
| **drm_run_job事件** | Compute kernel不触发 | Compute kernel会触发 |
| **适用架构** | 新架构GPU（如MI300） | 旧架构GPU或MES未启用 |
| **性能** | 低延迟，高吞吐量 | 相对较高延迟 |

**关键理解**:
- **MES模式**: Compute kernel通过doorbell直接提交给MES硬件调度器，不经过KFD驱动层GPU调度器
- **CPSCH模式**: Compute kernel经过KFD驱动层Ring和GPU调度器（drm_gpu_scheduler）调度
- **驱动层说明**: 这里的"驱动层"主要指**KFD (Kernel Fusion Driver)**，是AMD GPU的驱动层，负责管理queue、context等
- **当前系统**: 通常使用MES模式（`enable_mes = true`），因此Compute kernel不触发`drm_run_job`事件

### 4. 为什么ftrace显示只有SDMA Ring？

**根本原因**:
1. **MES模式下**: Compute kernel通过doorbell提交，不经过KFD驱动层Ring，**不触发drm_run_job事件**
2. **SDMA操作**: 无论MES还是CPSCH模式，都通过KFD驱动层SDMA Ring提交，会触发`drm_run_job`事件
3. **ftrace中的`drm_run_job`事件主要是SDMA操作**，不是Compute kernel
4. **如果使用CPSCH模式**: Compute kernel也会经过KFD驱动层Ring，会触发`drm_run_job`事件
5. **驱动层说明**: "驱动层"主要指**KFD (Kernel Fusion Driver)**，是AMD GPU驱动层的一部分，负责queue管理和调度

---

## 性能影响

### Doorbell机制的优势

- ✅ **低延迟**：直接通知硬件，无需经过驱动层
- ✅ **高吞吐量**：硬件可以直接从AQL Queue读取packet
- ✅ **减少系统调用**：不需要每次kernel提交都调用驱动层
- ✅ **硬件调度**：MES硬件调度器可以高效调度多个queue

### 驱动层Ring的限制

- ⚠️ **系统调用开销**：每次提交都需要系统调用
- ⚠️ **调度器瓶颈**：驱动层调度器可能成为瓶颈
- ⚠️ **Entity限制**：每个Context只有4个Entity，限制并行度
- ⚠️ **Ring竞争**：多个进程竞争有限的Ring资源

---

## 验证方法

### 1. 查看Doorbell日志

```bash
export AMD_LOG_LEVEL=5
export AMD_LOG_MASK=0xFFFFFFFF
./qwen2_single_prefill.py
```

**预期日志**:
```
:amd_blit_kernel.cpp:1301: [***rocr***] HWq=0x7f40f14e4000, id=0, 
Dispatch Header = 0x1402, rptr=6, wptr=6
```

### 2. 查看ftrace中的Ring使用

```bash
echo 1 > /sys/kernel/debug/tracing/events/drm/drm_run_job/enable
./qwen2_single_prefill.py
cat /sys/kernel/debug/tracing/trace | grep drm_run_job
```

**预期结果**: 如果只有`sdmaX.Y`，说明主要是SDMA操作

### 3. 查看AQL Queue创建

```bash
strace -e trace=ioctl ./qwen2_single_prefill.py 2>&1 | grep KFD_IOC_CREATE_QUEUE
```

---

## 相关文档

### 参考文档（来源）

- `DRIVER_30_COMPUTE_KERNEL_SUBMISSION_ANALYSIS.md` - Compute Kernel提交路径分析
- `DRIVER_39_KERNEL_SUBMISSION_CHANNELS_ANALYSIS.md` - Kernel submission通道分析
- `DRIVER_47_MES_KERNEL_SUBMISSION_ANALYSIS.md` - MES Kernel提交机制分析
- `DRIVER_55_SINGLE_KERNEL_SUBMISSION_ANALYSIS.md` - 单Kernel测试的Kernel提交机制分析
- `DRIVER_46_MES_ADD_HW_QUEUE_ANALYSIS.md` - MES add_hw_queue实现分析
- `DRIVER_44_KFD_IOCTL_COMPUTE_RING_MAPPING.md` - KFD IOCTL中Compute Ring映射分析
- `DRIVER_42_ROCM_CODE_ANALYSIS.md` - ROCm 6.4.3代码分析
- `DRIVER_40_HARDWARE_SOFTWARE_MAPPING_TABLE.md` - 软硬件名词概念对应表
- `DRIVER_111_LOAD_BALANCE_STRATEGY_ANALYSIS.md` - 负载均衡策略分析
- `ARCH_01_MI300_HARDWARE_QUEUE_ANALYSIS.md` - MI300硬件队列分析

### 本目录文档

- `AI_KERNEL_SUBMISSION_FLOW.md` - AI Kernel Submission流程详解
- `CONTEXT_DETAILED_ANALYSIS.md` - GPU Context详解
- `CODE_REFERENCE.md` - 关键代码参考

### 文档位置

- **参考文档**: `/mnt/md0/zhehan/code/rampup_doc/2PORC_profiling/`
- **本文档**: `/mnt/md0/zhehan/code/rampup_doc/Topic_kernel_submission/`

---

## 更新日志

- **2025-01-XX**: 创建AI Kernel Submission流程文档目录
  - 创建`AI_KERNEL_SUBMISSION_FLOW.md` - 主要流程文档
  - 创建`CODE_REFERENCE.md` - 代码参考文档
  - 创建`README.md` - 文档概述和快速开始指南

---

## 贡献

本文档基于`/mnt/md0/zhehan/code/rampup_doc/2PORC_profiling`中的DRIVER_xx系列分析文档整理而成。

如有问题或建议，请参考原始文档或联系文档维护者。

