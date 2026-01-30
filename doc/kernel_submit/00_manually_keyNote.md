
# Kernel提交流程研究笔记

## 📚 完整代码追踪文档系列 (2026-01-16创建)

已创建完整的kernel提交流程代码追踪文档，共5个文档 + 3个专题：

### 核心文档
1. **KERNEL_TRACE_INDEX.md** - 总览和快速索引
2. **KERNEL_TRACE_01_APP_TO_HIP.md** - 应用层到HIP Runtime
3. **KERNEL_TRACE_02_HSA_RUNTIME.md** - HSA Runtime层
4. **KERNEL_TRACE_03_KFD_QUEUE.md** - KFD驱动层Queue管理
5. **KERNEL_TRACE_04_MES_HARDWARE.md** - MES调度器与硬件层
6. **KERNEL_TRACE_05_DATA_STRUCTURES.md** - 关键数据结构详解

### 专题文档
7. **KERNEL_TRACE_STREAM_MANAGEMENT.md** - Stream管理机制详解
8. **ROCM_PROFILING_TOOLS_GUIDE.md** - ROCprofiler-SDK使用指南

### 文档特点
- ✅ 基于 ROCm_keyDriver 代码库
- ✅ 代码路径使用相对路径（ROCm_keyDriver/xxx）
- ✅ 包含详细的代码片段和注释
- ✅ 每个文档独立可读，按流程顺序组织
- ✅ 涵盖从应用层到硬件层的完整流程

### 核心发现
1. **90% kernel提交使用doorbell机制**，不经过KFD驱动层Ring
2. **MES是硬件调度器**，直接从AQL Queue读取packet
3. **AQL Queue在用户空间**，GPU直接访问
4. **Doorbell映射到用户空间**，无需系统调用

---

## HIP程序与/dev/kfd的关系

### 关键理解
✅ **HIP程序必须打开/dev/kfd**，即使使用doorbell机制
  - 获取GPU设备信息（AMDKFD_IOC_GET_VERSION）
  - 创建和管理Queue（AMDKFD_IOC_CREATE_QUEUE）
  - 分配GPU内存（AMDKFD_IOC_ALLOC_MEMORY_OF_GPU）
  - 管理进程的GPU资源（Context、Queue、Memory）

✅ **打开时机**: hipInit()或首次使用HIP API时

✅ **Doorbell机制不影响打开KFD**
  - Doorbell只改变kernel提交方式（不经过Ring）
  - Queue创建、内存管理仍需KFD驱动

### 验证方法
```bash
# 方法1: strace跟踪系统调用
strace -e trace=open,openat,ioctl ./app 2>&1 | grep kfd

# 方法2: lsof查看打开的文件
lsof -p <pid> | grep /dev/kfd

# 方法3: ftrace跟踪
echo 1 > /sys/kernel/debug/tracing/events/syscalls/sys_enter_openat/enable
```

---

## Kernel 提交路径（MES 模式）

### ✅ Compute Kernel (100% 走 Doorbell)
```
应用 → hipLaunchKernel → 写AQL packet → 写doorbell (MMIO)
→ MES硬件调度器 → GPU执行
```
- **100%** 通过 Doorbell，**0%** 经过 KFD Ring
- 不触发 drm_run_job 事件
- 用户空间直接 MMIO 写入，无系统调用
- 详见: [KERNEL_SUBMISSION_PATHS.md](./KERNEL_SUBMISSION_PATHS.md)

### ⚠️ SDMA 操作 (100% 走 KFD Ring)
```
应用 → hipMemcpy → HSA Runtime → KFD驱动 → SDMA Ring 
→ GPU调度器 → GPU执行
```
- 100% 经过 KFD Ring
- 触发 drm_run_job 事件（ftrace 可见）
- 用于内存拷贝等操作

### 🔧 Queue 管理 (极低频)
```
应用 → hipStreamCreate → ioctl → KFD驱动 → MES Ring
→ MES硬件注册Queue
```
- 经过 KFD MES Ring（管理命令）
- 只在 Queue 创建/销毁时

### ⚠️ "90%" 说法的澄清
- ❌ 错误理解："90% 的 kernel 走 doorbell"
- ✅ 正确理解："100% 的 compute kernel 走 doorbell"
- 📊 统计来源：典型程序中 90%+ 的操作是 compute kernel



#
image.png


特性	用户态 Queue	内核态 Ring
术语	Queue (HSA标准)	Ring (AMDGPU术语)
数据结构	Ring buffer	Ring buffer
位置	用户空间内存	内核空间内存
访问	HSA Runtime	AMDGPU Driver
用途	存储AQL Packet	存储驱动命令
Compute Kernel (MES)	✅ 使用	❌ 不使用
SDMA操作	❌ 不直接用	✅ 使用
通知机制	Doorbell (直接)	Doorbell (驱动)


是的，AQL Queue 本质上就是一个 ring buffer，但在 ROCm 中不叫它 "ring"！
详细确认
1️⃣ AQL Queue 就是 Ring Buffer

image.png


# HIP LaunchKernel()
我已经找到并在文档中添加了真实源码的位置：
文件	函数	行号	作用
hip_module.cpp	hipLaunchKernel()	823-828	公开的 API 入口函数
hip_module.cpp	hipLaunchKernel_common()	816-821	内部通用实现
hip_platform.cpp ⭐	ihipLaunchKernel()	689-736	核心启动逻辑（最重要）
hip_module.cpp	ihipModuleLaunchKernel()	见源码	最终的 module 层启动

Path:
ROCm_keyDriver/rocm-systems/projects/clr/hipamd/src/hip_platform.cpp (核心)
ROCm_keyDriver/rocm-systems/projects/clr/hipamd/src/hip_module.cpp (入口)


用户代码: hipLaunchKernelGGL(kernel, ...)
    ↓
① hipLaunchKernel()                      [hip_module.cpp:823-828]
    ↓
② hipLaunchKernel_common()               [hip_module.cpp:816-821]
    ↓
③ ihipLaunchKernel()                     [hip_platform.cpp:689-736]
    ↓
④ ihipModuleLaunchKernel() ⭐            [hip_module.cpp:443-532]  ← 您问的这个
    ↓
⑤ ihipLaunchKernelCommand() ⭐⭐         [hip_module.cpp:352-436]
    ↓
   创建 amd::NDRangeKernelCommand ⭐⭐⭐
    ↓
   command->enqueue()  （放入 Stream 队列）
    ↓
   进入 HSA Runtime 处理...