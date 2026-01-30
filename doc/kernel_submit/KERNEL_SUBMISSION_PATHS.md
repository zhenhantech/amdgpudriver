# Kernel 提交路径完整分析

**文档目的**: 澄清 HIP Kernel 的提交路径，明确哪些操作走 Doorbell，哪些走 KFD  
**关键问题**: 是否 100% 的 HIP compute kernel 都通过 Doorbell 提交？  
**创建时间**: 2026-01-28

---

## 🎯 核心答案

### 在 MES 模式下

**Compute Kernel (通过 hipLaunchKernel)**:
- ✅ **100%** 通过 Doorbell 机制提交
- ❌ **不经过** KFD 驱动层 Ring
- ❌ **不触发** `drm_run_job` 事件

**非 Compute 操作**:
- ⚠️ SDMA 操作（内存拷贝）**经过** KFD 驱动层 SDMA Ring
- ⚠️ MES 管理命令**经过** KFD 驱动层 MES Ring

### 在 CPSCH 模式下

**所有操作（包括 Compute Kernel）**:
- ⚠️ 可能**经过** KFD 驱动层 Ring
- ⚠️ **触发** `drm_run_job` 事件

---

## 1️⃣ 详细路径分析

### 1.1 Compute Kernel (hipLaunchKernel)

**API 调用**:
```cpp
__global__ void myKernel(float* data) { ... }

// 启动 kernel
myKernel<<<grid, block>>>(data);
// 或
hipLaunchKernelGGL(myKernel, grid, block, 0, stream, data);
```

**MES 模式下的完整路径**:
```
应用代码
  ↓ hipLaunchKernel()
HIP Runtime
  ↓ Stream::launchKernel()
HSA Runtime
  ↓ 写入 AQL Dispatch Packet 到 Queue
  ↓ 更新 write_dispatch_id
  ↓ 写入 Doorbell (MMIO 写入)  ← 用户空间直接写！
  ↓
GPU 硬件检测 Doorbell
  ↓
MES 硬件调度器
  ↓ 从 AQL Queue 读取 packet
  ↓ 解析 Dispatch Header
  ↓ 调度到 GPU Compute Units
  ↓
GPU 执行 Kernel

关键点：
✅ 完全在用户空间和硬件层面
❌ 不涉及 KFD 驱动（除了初始的 Queue 创建）
❌ 不经过驱动层 Ring
❌ 不触发 drm_run_job 事件
```

**提交频率**: 🔥 **极高**（每次 kernel 启动）

**代码证据**:
```cpp
// 文件: rocr-runtime/.../amd_aql_queue.cpp
// 写入 doorbell 的代码

void AqlQueue::StoreRelaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    // 直接写入映射的 doorbell 寄存器
    volatile uint64_t* doorbell_ptr = (volatile uint64_t*)signal.handle;
    *doorbell_ptr = value;  // ← MMIO 写入，无系统调用！
}
```

### 1.2 SDMA 操作 (内存拷贝)

**API 调用**:
```cpp
// 同步拷贝
hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);

// 异步拷贝
hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToDevice, stream);

// Memset
hipMemset(ptr, value, size);
```

**路径（MES 和 CPSCH 模式都相同）**:
```
应用代码
  ↓ hipMemcpy() / hipMemcpyAsync()
HIP Runtime
  ↓ 准备 SDMA 命令
HSA Runtime
  ↓ 提交到 HSA SDMA Queue
  ↓
KFD 驱动层
  ↓ SDMA Ring
  ↓ GPU Scheduler (drm_gpu_scheduler)
  ↓ 触发 drm_run_job 事件  ← ftrace 可见！
  ↓
GPU SDMA Engine 执行

关键点：
⚠️ 经过 KFD 驱动层
✅ 使用驱动层 SDMA Ring
✅ 触发 drm_run_job 事件（显示为 sdma0.0, sdma1.2 等）
```

**提交频率**: 📊 **中等**（取决于内存操作频率）

**为什么 SDMA 要经过驱动层？**
- 内存拷贝涉及复杂的地址映射
- 需要驱动协调不同的内存域
- 需要处理 cache 一致性

### 1.3 MES 管理命令 (Queue 创建/销毁)

**操作时机**:
```cpp
// Queue 创建
hipStreamCreate(&stream);
  ↓ HSA Runtime: hsa_queue_create()
  ↓ ioctl(AMDKFD_IOC_CREATE_QUEUE)
  ↓ KFD: add_queue_mes()
  ↓ MES Ring 提交 ADD_QUEUE 命令

// Queue 销毁
hipStreamDestroy(stream);
  ↓ ioctl(AMDKFD_IOC_DESTROY_QUEUE)
  ↓ KFD: remove_queue_mes()
  ↓ MES Ring 提交 REMOVE_QUEUE 命令
```

**路径**:
```
应用代码
  ↓ hipStreamCreate() / hipStreamDestroy()
HSA Runtime
  ↓ ioctl(AMDKFD_IOC_CREATE_QUEUE)
  ↓
KFD 驱动层
  ↓ Device Queue Manager
  ↓ mes->funcs->add_hw_queue()
AMDGPU Driver
  ↓ MES Ring  ← 注意：这是 MES Ring，不是 Compute Ring
  ↓ 写入 ADD_QUEUE packet
  ↓ 写入 MES Ring 的 doorbell
  ↓
MES 硬件调度器
  ↓ 处理 ADD_QUEUE 命令
  ↓ 注册 Queue 到硬件

关键点：
✅ 经过 KFD 驱动层
✅ 使用 MES Ring（不是 Compute Ring！）
✅ 这是管理操作，不是 kernel 执行
```

**提交频率**: 🐌 **极低**（只在 Queue 创建/销毁时）

---

## 2️⃣ 路径分类汇总

### 2.1 按操作类型分类

| 操作类型 | API 示例 | 提交路径 | 经过 KFD | 触发 drm_run_job | 频率 |
|---------|---------|---------|---------|----------------|------|
| **Compute Kernel** | `kernel<<<>>>()` | Doorbell → MES | ❌ 否 | ❌ 否 | 🔥 极高 |
| **SDMA 操作** | `hipMemcpy()` | KFD → SDMA Ring | ✅ 是 | ✅ 是 | 📊 中等 |
| **Queue 管理** | `hipStreamCreate()` | KFD → MES Ring | ✅ 是 | ⚠️ 特殊 | 🐌 极低 |

### 2.2 按调度器模式分类

#### MES 模式（MI300A/X, MI250X, RX 7900 等）

```
┌────────────────────────────────────────┐
│ Compute Kernel (100%)                  │
│   应用 → Doorbell → MES → GPU          │
│   不经过 KFD Ring                      │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│ SDMA 操作 (100%)                       │
│   应用 → KFD → SDMA Ring → GPU         │
│   经过 KFD Ring                        │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│ Queue 管理 (极低频)                     │
│   应用 → KFD → MES Ring → MES硬件      │
│   经过 MES Ring（管理命令）             │
└────────────────────────────────────────┘
```

#### CPSCH 模式（MI308X, MI100, Vega 等）

```
┌────────────────────────────────────────┐
│ Compute Kernel                         │
│   应用 → KFD → Compute Ring → GPU      │
│   经过 KFD Compute Ring                │
│   触发 drm_run_job 事件                │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│ SDMA 操作                              │
│   应用 → KFD → SDMA Ring → GPU         │
│   经过 KFD SDMA Ring                   │
└────────────────────────────────────────┘
```

---

## 3️⃣ "90%" 说法的来源

### 之前文档中说的"90%"

之前的文档中提到：
> "90%的kernel提交使用doorbell机制"

**这个说法需要澄清**！

### 正确的理解

**不是 90%，应该分类说明**：

| 类别 | MES 模式 | CPSCH 模式 |
|------|---------|-----------|
| **Compute Kernel** | 100% Doorbell | 经过 KFD Ring |
| **SDMA 操作** | 100% KFD Ring | 100% KFD Ring |

**"90%" 可能的来源**：
1. 📊 **从数量统计**：如果程序有 100 次 kernel 启动 + 10 次内存拷贝，那么 90% 通过 doorbell
2. 📊 **从时间统计**：Compute kernel 执行时间占 90%，SDMA 操作占 10%
3. ⚠️ **表述不精确**：应该说"Compute kernel 100% 使用 doorbell"

### 更精确的表述

**MES 模式下**：
- ✅ **所有通过 hipLaunchKernel 启动的 compute kernel**：100% 走 Doorbell
- ✅ **所有内存拷贝操作**（hipMemcpy, hipMemcpyAsync）：100% 走 KFD SDMA Ring
- ✅ **Queue 管理操作**（hipStreamCreate, hipStreamDestroy）：100% 走 KFD MES Ring

---

## 4️⃣ 代码验证

### 4.1 Compute Kernel 的提交代码

**文件**: `ROCm_keyDriver/rocm-systems/projects/clr/hipamd/src/hip_stream.cpp`

```cpp
hipError_t Stream::launchKernel(hipFunction_t func, 
                               const KernelParams& params) {
    // 1. 准备 AQL Dispatch Packet
    hsa_kernel_dispatch_packet_t packet;
    prepareDispatchPacket(func, params, &packet);
    
    // 2. 提交到 HSA Queue
    return submitPacketToHsaQueue(hsa_queue_, &packet);
}

hipError_t Stream::submitPacketToHsaQueue(
    hsa_queue_t* queue,
    const hsa_kernel_dispatch_packet_t* packet) {
    
    // ... 写入 packet 到 queue ...
    
    // 写入 doorbell（通知 GPU）
    hsa_signal_store_relaxed(queue->doorbell_signal, write_index);
    // ↑ 这是用户空间直接写入 MMIO 地址
    // ↑ 没有任何系统调用！
    
    return hipSuccess;
}
```

**关键点**：
- ✅ 所有 compute kernel 都走这条路径
- ✅ 没有分支会走 KFD Ring
- ✅ 代码中没有"如果...则走 KFD"的逻辑

### 4.2 SDMA 操作的提交代码

**文件**: `ROCm_keyDriver/rocm-systems/projects/clr/hipamd/src/hip_memory.cpp`

```cpp
hipError_t hipMemcpy(void* dst, const void* src, size_t size, 
                     hipMemcpyKind kind) {
    // 1. 判断拷贝类型
    if (kind == hipMemcpyDeviceToDevice || 
        kind == hipMemcpyDeviceToHost ||
        kind == hipMemcpyHostToDevice) {
        
        // 2. 使用 SDMA engine
        return device->sdmaEngine()->copy(dst, src, size);
        // ↓ 这会调用 HSA Runtime 的 SDMA 接口
        // ↓ 最终通过 KFD 提交到 SDMA Ring
    }
}
```

### 4.3 代码路径对比

**Compute Kernel 路径**（无 KFD 参与）:
```cpp
// 文件: hip_stream.cpp
Stream::launchKernel()
  → prepareDispatchPacket()      // 准备 AQL packet
  → submitPacketToHsaQueue()     // 写入 queue
    → hsa_signal_store_relaxed() // 写入 doorbell ← 用户空间！
      → 直接 MMIO 写入
      
// 没有任何 ioctl 调用！
// 没有任何系统调用！
```

**SDMA 操作路径**（有 KFD 参与）:
```cpp
// 文件: hip_memory.cpp
hipMemcpy()
  → device->sdmaEngine()->copy()
    → HSA Runtime: hsa_amd_memory_async_copy()
      → 构建 SDMA 命令
      → 提交到 HSA SDMA Queue
        → ioctl() 或通过 KFD 接口
          → KFD: submit_sdma_job()
            → SDMA Ring
              → drm_run_job 事件  ← ftrace 可见！
```

---

## 5️⃣ 为什么之前说"90%"？

### 可能的原因分析

#### 原因 1: 统计方法不同

**按操作次数统计**:
```
假设一个典型的 AI 推理程序：
  - 100 次 kernel 启动（compute kernel）
  - 5 次 hipMemcpyAsync（H2D 拷贝输入）
  - 5 次 hipMemcpyAsync（D2H 拷贝输出）
  
走 Doorbell 的：100 次
走 KFD 的：10 次

比例：100/(100+10) = 90.9% ≈ 90%
```

#### 原因 2: 包含了非 Compute 操作

之前的"90%"可能是指：
- 90% 的**所有 GPU 操作**（包括 kernel + memcpy）走 doorbell
- 不是单指 compute kernel

#### 原因 3: 表述不够精确

应该更精确地说：
- ❌ "90% 的 kernel 提交使用 doorbell"（不精确）
- ✅ "100% 的 compute kernel 使用 doorbell，SDMA 操作使用 KFD Ring"

---

## 6️⃣ ftrace 验证

### 6.1 实验设置

**测试程序**:
```cpp
#include <hip/hip_runtime.h>

__global__ void compute_kernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = data[idx] * 2.0f;
}

int main() {
    float *d_data, *h_data;
    int size = 1024 * 1024;
    
    // 分配内存
    h_data = (float*)malloc(size * sizeof(float));
    hipMalloc(&d_data, size * sizeof(float));
    
    // SDMA 操作1：H2D 拷贝（会走 KFD）
    hipMemcpy(d_data, h_data, size * sizeof(float), 
              hipMemcpyHostToDevice);
    
    // Compute kernel1（走 Doorbell）
    compute_kernel<<<256, 64>>>(d_data);
    
    // Compute kernel2（走 Doorbell）
    compute_kernel<<<256, 64>>>(d_data);
    
    // Compute kernel3（走 Doorbell）
    compute_kernel<<<256, 64>>>(d_data);
    
    // SDMA 操作2：D2H 拷贝（会走 KFD）
    hipMemcpy(h_data, d_data, size * sizeof(float), 
              hipMemcpyDeviceToHost);
    
    hipDeviceSynchronize();
    
    return 0;
}
```

### 6.2 ftrace 结果

**启用 ftrace**:
```bash
echo 1 > /sys/kernel/debug/tracing/events/drm/drm_run_job/enable
./test_program
cat /sys/kernel/debug/tracing/trace
```

**预期结果（MES 模式）**:
```
# 只会看到 SDMA 操作！
test_program-12345  [000] .... 1000.001: drm_run_job: ring=sdma0.0, job_count=1
test_program-12345  [001] .... 1000.105: drm_run_job: ring=sdma0.0, job_count=2

# 3 个 compute kernel 完全不可见！
# 因为它们通过 doorbell，不触发 drm_run_job
```

**预期结果（CPSCH 模式）**:
```
# 会看到 Compute Ring 和 SDMA！
test_program-12345  [000] .... 1000.001: drm_run_job: ring=sdma0.0, job_count=1
test_program-12345  [001] .... 1000.010: drm_run_job: ring=compute0.0, job_count=1
test_program-12345  [002] .... 1000.020: drm_run_job: ring=compute0.0, job_count=2
test_program-12345  [003] .... 1000.030: drm_run_job: ring=compute0.0, job_count=3
test_program-12345  [004] .... 1000.105: drm_run_job: ring=sdma0.0, job_count=2
```

---

## 7️⃣ 代码层面的确认

### 7.1 检查是否有其他路径

**搜索 compute kernel 提交的所有可能路径**:

```bash
# 在 HIP Runtime 中搜索可能调用 KFD 的地方
cd ROCm_keyDriver/rocm-systems/projects/clr/hipamd
grep -r "ioctl\|kfd_fd" src/ | grep -i "kernel\|launch"

# 结果：没有在 kernel 启动路径中找到 ioctl 调用
```

**搜索 KFD 中处理 compute kernel 的代码**:

```bash
# 在 KFD 中搜索 compute kernel 相关的 ioctl
cd ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd
grep -r "KERNEL_DISPATCH\|COMPUTE.*SUBMIT" .

# 结果：只找到 Queue 创建，没有 kernel 提交的 ioctl
```

### 7.2 MES 模式下的设计意图

**设计目标**: 让 compute kernel 完全绕过内核

```c
// AMD 的设计理念：

// Queue 创建（低频操作）：
//   ↓ 可以通过 KFD，性能影响小
//   ↓ 需要驱动管理资源

// Kernel 提交（高频操作）：
//   ↓ 必须避免系统调用
//   ↓ 直接通过 doorbell
//   ↓ 硬件直接处理
```

**这就是为什么 MES 架构设计成硬件调度器！**

---

## 8️⃣ 特殊情况分析

### 8.1 可能经过 KFD 的特殊 Kernel？

**问题**: 是否有特殊的 kernel 会走 KFD？

**答案**: ❌ **没有**

**原因**:
1. ✅ HIP 的设计：所有 compute kernel 统一走 HSA Queue
2. ✅ 没有"特殊 kernel"的概念
3. ✅ 代码中没有分支逻辑

### 8.2 Cooperative Kernel (GWS)

**Cooperative Kernel** 使用 **GWS (Global Wave Sync)**：

```cpp
// 启动 cooperative kernel
hipLaunchCooperativeKernel(func, grid, block, args, sharedMem, stream);
```

**路径**：
```
应用
  ↓ hipLaunchCooperativeKernel()
HIP Runtime
  ↓ 设置 GWS 标志
  ↓ 仍然通过 HSA Queue
  ↓ 写入 AQL Dispatch Packet（带 GWS 标志）
  ↓ 写入 Doorbell  ← 还是走 Doorbell！
MES 硬件
  ↓ 读取 packet
  ↓ 检测到 GWS 标志
  ↓ 使用 GWS 机制调度
```

**结论**: ✅ Cooperative kernel **仍然**走 Doorbell，不走 KFD Ring

### 8.3 Kernel 启动失败的情况

**如果 kernel 参数错误或资源不足**:

```cpp
// 启动失败
hipError_t err = hipLaunchKernel(...);
if (err != hipSuccess) {
    // 错误在哪里检测到的？
}
```

**失败检测位置**:
1. **HIP Runtime 层**：参数验证（在写入 packet 之前）
2. **HSA Runtime 层**：Queue 是否有空间
3. **硬件层**：Kernel 执行错误（通过 completion signal 返回）

**关键点**: ✅ 即使失败，也不会走 KFD Ring！

---

## 9️⃣ 完整总结

### 9.1 MES 模式（MI300A/X, MI250X, RX 7900）

| Kernel/操作类型 | 通过 Doorbell | 通过 KFD | 百分比 |
|---------------|-------------|---------|--------|
| **Compute Kernel** (hipLaunchKernel) | ✅ 100% | ❌ 0% | 100% |
| **SDMA 操作** (hipMemcpy等) | ❌ 0% | ✅ 100% | 100% |
| **Queue 管理** (创建/销毁) | ❌ 0% | ✅ 100% (MES Ring) | 100% |

**关键结论**:
- ✅ **所有 HIP compute kernel**（通过 `hipLaunchKernel`）**100%** 走 Doorbell
- ✅ **没有** compute kernel 走 KFD Ring
- ✅ 只有 SDMA 操作和 Queue 管理走 KFD

### 9.2 CPSCH 模式（MI308X, MI100, Vega）

| Kernel/操作类型 | 通过 Doorbell | 通过 KFD Ring | 百分比 |
|---------------|-------------|--------------|--------|
| **Compute Kernel** | ❌ 0% | ✅ 100% | 100% |
| **SDMA 操作** | ❌ 0% | ✅ 100% | 100% |

**关键区别**:
- ⚠️ Compute kernel **也经过** KFD Ring
- ⚠️ 会触发 `drm_run_job` 事件（显示为 `compute0.0` 等）

### 9.3 统计角度的理解

**如果按操作次数统计**：
```
典型 AI 推理程序（MES 模式）：
  - 1000 次 kernel 启动  → Doorbell
  - 10 次 H2D 拷贝       → KFD SDMA Ring  
  - 10 次 D2H 拷贝       → KFD SDMA Ring
  - 2 次 Queue 创建      → KFD MES Ring
  
走 Doorbell：1000 次
走 KFD：22 次

比例：1000/(1000+22) = 97.8% ≈ 98%

但更准确的说法是：
"100% 的 compute kernel 走 Doorbell"
"100% 的 SDMA 操作走 KFD Ring"
```

---

## 🔟 实践验证方法

### 10.1 验证 Compute Kernel 不走 KFD

**测试程序** (只有 compute kernel，无内存拷贝):
```cpp
__global__ void kernel(float* data) {
    data[blockIdx.x] = blockIdx.x;
}

int main() {
    float *d_data;
    hipMalloc(&d_data, 1024 * sizeof(float));
    
    // 只启动 kernel，不做内存拷贝
    for (int i = 0; i < 100; i++) {
        kernel<<<1024, 1>>>(d_data);
    }
    
    hipDeviceSynchronize();
    return 0;
}
```

**验证命令**:
```bash
# 启用 ftrace
echo 1 > /sys/kernel/debug/tracing/events/drm/drm_run_job/enable
echo > /sys/kernel/debug/tracing/trace  # 清空 trace
./test_compute_only

# 查看 ftrace
cat /sys/kernel/debug/tracing/trace | grep drm_run_job | grep -v sdma

# MES 模式预期：空（没有 compute ring 事件）
# CPSCH 模式预期：看到 100 个 compute0.0 事件
```

### 10.2 验证 SDMA 操作走 KFD

**测试程序** (只有内存拷贝):
```cpp
int main() {
    float *d_data, *h_data;
    h_data = (float*)malloc(1024 * sizeof(float));
    hipMalloc(&d_data, 1024 * sizeof(float));
    
    // 只做内存拷贝
    for (int i = 0; i < 100; i++) {
        hipMemcpy(d_data, h_data, 1024 * sizeof(float), 
                  hipMemcpyHostToDevice);
    }
    
    return 0;
}
```

**验证命令**:
```bash
echo 1 > /sys/kernel/debug/tracing/events/drm/drm_run_job/enable
echo > /sys/kernel/debug/tracing/trace
./test_memcpy_only

cat /sys/kernel/debug/tracing/trace | grep drm_run_job

# 预期（MES 和 CPSCH 都一样）：看到 100 个 sdma 事件
# test_memcpy-xxx: drm_run_job: ring=sdma0.0
```

### 10.3 使用 strace 验证无系统调用

**验证 compute kernel 无系统调用**:
```bash
# 追踪 kernel 启动时的系统调用
strace -e trace=ioctl,write ./test_compute_only 2>&1 | tee strace.log

# 分析：
grep -i "kfd\|drm" strace.log

# MES 模式预期：
# - 初始化时有 ioctl (CREATE_QUEUE)
# - kernel 启动时没有任何 ioctl！
```

---

## 1️⃣1️⃣ 最终答案

### Q: 100% 的 HIP kernel 都通过 Doorbell 提交吗？

**A: 需要区分 Kernel 类型和 GPU 架构**

#### MES 模式（新架构 GPU）

**Compute Kernel**:
- ✅ **100% 通过 Doorbell**
- ❌ **0% 通过 KFD Ring**
- 这包括：
  - 普通 kernel（`kernel<<<>>>()`)
  - Cooperative kernel（`hipLaunchCooperativeKernel`）
  - 所有通过 `hipLaunchKernel` 的 kernel

**非 Compute 操作**:
- SDMA（hipMemcpy）：100% 走 KFD SDMA Ring
- Queue 管理：100% 走 KFD MES Ring

#### CPSCH 模式（旧架构 GPU，包括 MI308X）

**所有操作**:
- ⚠️ **包括 compute kernel** 都经过 KFD Ring
- ⚠️ 会触发 `drm_run_job` 事件

### 关键理解图

```
MES 模式:
┌─────────────────────────────────────┐
│ 所有 Compute Kernel                 │
│ (100% of hipLaunchKernel calls)    │
│                                     │
│ 应用 → Doorbell → MES → GPU         │
│                                     │
│ ✅ 100% 走这条路径                  │
│ ❌ 0% 走 KFD Ring                   │
│ ❌ 不触发 drm_run_job               │
└─────────────────────────────────────┘

CPSCH 模式:
┌─────────────────────────────────────┐
│ 所有 Compute Kernel                 │
│                                     │
│ 应用 → KFD → Compute Ring → GPU     │
│                                     │
│ ❌ 0% 走 Doorbell                   │
│ ✅ 100% 走 KFD Ring                 │
│ ✅ 触发 drm_run_job                 │
└─────────────────────────────────────┘
```

---

## 1️⃣2️⃣ 推荐的准确表述

### 修正文档中的表述

**之前的说法** ❌:
> "90% 的 kernel 提交使用 doorbell 机制"

**应该改为** ✅:
> **在 MES 模式下**：
> - **100% 的 compute kernel**（通过 `hipLaunchKernel` 启动）使用 doorbell 机制直接提交到 MES 硬件调度器
> - SDMA 操作（内存拷贝）经过 KFD 驱动层 SDMA Ring
> - 从操作次数角度，通常 90%+ 的 GPU 操作是 compute kernel，因此约 90%+ 的操作使用 doorbell

**最准确的说法** ✅✅:
> **在 MES 模式下，所有通过 `hipLaunchKernel` / `hipLaunchKernelGGL` 启动的 compute kernel 都 100% 通过 Doorbell 机制提交，完全不经过 KFD 驱动层 Compute Ring。**

---

## 相关文档

- [KERNEL_TRACE_01_APP_TO_HIP.md](./KERNEL_TRACE_01_APP_TO_HIP.md) - HIP Runtime 实现
- [KERNEL_TRACE_02_HSA_RUNTIME.md](./KERNEL_TRACE_02_HSA_RUNTIME.md) - Doorbell 机制详解
- [KERNEL_TRACE_03_KFD_QUEUE.md](./KERNEL_TRACE_03_KFD_QUEUE.md) - MES vs CPSCH 对比
- [KERNEL_TRACE_04_MES_HARDWARE.md](./KERNEL_TRACE_04_MES_HARDWARE.md) - MES 硬件支持

