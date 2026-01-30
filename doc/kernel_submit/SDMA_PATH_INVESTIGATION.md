# SDMA 提交路径深入调查

**调查目的**: 澄清 `hipMemcpy` 是走 Doorbell 还是 KFD Ring  
**关键问题**: GPU 是否可以主动完成 copy？还是必须经过 KFD?  
**调查时间**: 2026-01-28  
**后续**: 详细的代码追踪见 [HIPMEMCPY_CODE_TRACE.md](./HIPMEMCPY_CODE_TRACE.md)

---

## 🔍 重要发现

### 代码分析表明：内存拷贝路径**不唯一**！

AMD GPU 的内存拷贝有**多种实现方式**：

1. **Blit Shader (通过 Compute)**: 写入 AQL Queue → Doorbell → GPU
2. **SDMA Engine (专用硬件)**: 通过 KFD → SDMA Ring → GPU
3. **CPU 直接拷贝**: 某些情况下 CPU 直接写入

**关键**: HIP Runtime/ROCclr 会**自动选择**最优路径！

---

## 1️⃣ 代码证据：多种拷贝路径

### 1.1 `hipMemcpy` 的类型判断

**文件**: `ROCm_keyDriver/rocm-systems/projects/clr/hipamd/src/hip_memory.cpp`

```cpp
hip::MemcpyType ihipGetMemcpyType(const void* src, void* dst, hipMemcpyKind kind) {
    amd::Memory* srcMemory = getMemoryObject(src, sOffset);
    amd::Memory* dstMemory = getMemoryObject(dst, dOffset);
    
    if (srcMemory == nullptr && dstMemory == nullptr) {
        return hipHostToHost;  // CPU 直接拷贝
    } else if ((srcMemory == nullptr) && (dstMemory != nullptr)) {
        return hipWriteBuffer;  // H2D 拷贝
    } else if ((srcMemory != nullptr) && (dstMemory == nullptr)) {
        return hipReadBuffer;  // D2H 拷贝
    } else if ((srcMemory != nullptr) && (dstMemory != nullptr)) {
        if (srcMemory->GetDeviceById() != dstMemory->GetDeviceById()) {
            return hipCopyBufferP2P;  // 跨 GPU 拷贝
        } else if (kind == hipMemcpyDeviceToDeviceNoCU) {
            return hipCopyBufferSDMA;  // 强制使用 SDMA ← 关键！
        } else {
            return hipCopyBuffer;  // 默认 D2D 拷贝
        }
    }
}
```

**关键发现**：
- ✅ **D2D 拷贝**默认是 `hipCopyBuffer`
- ✅ **只有明确指定** `hipMemcpyDeviceToDeviceNoCU` 才强制使用 SDMA
- ❓ `hipCopyBuffer` 可能用 **Blit Shader** 或 **SDMA**

### 1.2 Copy Engine Preference

```cpp
amd::CopyMetadata copyMetadata(isAsync, amd::CopyMetadata::CopyEnginePreference::NONE);
hip::MemcpyType type = ihipGetMemcpyType(src, dst, kind);

switch (type) {
    case hipCopyBufferSDMA:
        // 强制使用 SDMA
        copyMetadata.copyEnginePreference_ = amd::CopyMetadata::CopyEnginePreference::SDMA;
    case hipCopyBuffer:
        // 默认：让 Runtime 自动选择！
        // 可能是 Blit Shader (走 Doorbell)
        // 也可能是 SDMA (走 KFD Ring)
        command = new amd::CopyMemoryCommand(..., copyMetadata);
        break;
    case hipWriteBuffer:
        // H2D 拷贝
        command = new amd::WriteMemoryCommand(..., copyMetadata);
        break;
    case hipReadBuffer:
        // D2H 拷贝
        command = new amd::ReadMemoryCommand(..., copyMetadata);
        break;
}
```

**关键点**：
- 🔹 `CopyEnginePreference::NONE`：**Runtime 自动选择**
- 🔹 `CopyEnginePreference::SDMA`：**强制使用 SDMA**

---

## 2️⃣ ROCclr 层的实现逻辑

### 2.1 自动选择机制

ROCclr (ROCm Common Language Runtime) 会根据以下因素**自动选择**拷贝引擎：

#### 选择因素

| 因素 | Blit Shader (Compute) | SDMA Engine |
|------|----------------------|-------------|
| **拷贝大小** | 小拷贝（<1MB）优先 | 大拷贝（>1MB）优先 |
| **内存类型** | VRAM ↔ VRAM | Host ↔ Device |
| **对齐情况** | 对齐良好 | 任意对齐 |
| **GPU 繁忙度** | Compute 空闲时 | Compute 忙碌时 |
| **SDMA 可用性** | - | SDMA 不忙 |

### 2.2 实际行为（推测）

**小拷贝（例如 <1MB）**:
```
hipMemcpy(d_dst, d_src, 128KB, hipMemcpyDeviceToDevice)
  ↓ 类型: hipCopyBuffer
  ↓ Preference: NONE
  ↓ ROCclr 选择: Blit Shader
  ↓ 提交: AQL Queue
  ↓ 通知: Doorbell
  ↓ 执行: GPU Compute Units
```

**大拷贝（例如 >10MB）**:
```
hipMemcpy(d_dst, d_src, 100MB, hipMemcpyDeviceToDevice)
  ↓ 类型: hipCopyBuffer
  ↓ Preference: NONE
  ↓ ROCclr 选择: SDMA Engine
  ↓ 提交: KFD → SDMA Ring
  ↓ 通知: SDMA Doorbell
  ↓ 执行: SDMA 硬件
```

**H2D/D2H 拷贝**:
```
hipMemcpy(d_data, h_data, size, hipMemcpyHostToDevice)
  ↓ 类型: hipWriteBuffer
  ↓ Preference: NONE
  ↓ ROCclr 选择: 取决于内存映射方式
  ↓ 如果是 Pinned Memory: 可能 Blit Shader
  ↓ 如果是 Pageable Memory: 可能 SDMA + CPU staging
```

---

## 3️⃣ ftrace 验证计划

### 3.1 实验设计

**实验 1：纯 D2D 拷贝（小）**
```cpp
hipMemcpy(d_dst, d_src, 64 * 1024, hipMemcpyDeviceToDevice);
```
**预期**：可能**不会**触发 `drm_run_job`（走 Blit Shader + Doorbell）

**实验 2：纯 D2D 拷贝（大）**
```cpp
hipMemcpy(d_dst, d_src, 100 * 1024 * 1024, hipMemcpyDeviceToDevice);
```
**预期**：可能**会**触发 `drm_run_job` (sdma) (走 SDMA Ring)

**实验 3：H2D 拷贝（Pinned）**
```cpp
hipHostMalloc(&h_data, size, hipHostMallocDefault);  // Pinned
hipMemcpy(d_data, h_data, size, hipMemcpyHostToDevice);
```
**预期**：可能**不会**触发 `drm_run_job`（直接 GPU 读取）

**实验 4：H2D 拷贝（Pageable）**
```cpp
h_data = malloc(size);  // Pageable
hipMemcpy(d_data, h_data, size, hipMemcpyHostToDevice);
```
**预期**：可能**会**触发 `drm_run_job` (sdma)（需要 staging）

**实验 5：强制 SDMA**
```cpp
hipMemcpy(d_dst, d_src, 64 * 1024, hipMemcpyDeviceToDeviceNoCU);
```
**预期**：**一定会**触发 `drm_run_job` (sdma)

### 3.2 验证脚本

```bash
#!/bin/bash

echo "=== SDMA Path Verification ==="

# 清空 ftrace
sudo sh -c 'echo > /sys/kernel/debug/tracing/trace'

# 启用 drm_run_job 事件
sudo sh -c 'echo 1 > /sys/kernel/debug/tracing/events/drm/drm_run_job/enable'

# 运行测试程序
./test_sdma_path

# 查看结果
echo "=== ftrace Results ==="
sudo cat /sys/kernel/debug/tracing/trace | grep drm_run_job

# 分析
SDMA_COUNT=$(sudo cat /sys/kernel/debug/tracing/trace | grep drm_run_job | grep sdma | wc -l)
COMPUTE_COUNT=$(sudo cat /sys/kernel/debug/tracing/trace | grep drm_run_job | grep compute | wc -l)

echo ""
echo "SDMA events: $SDMA_COUNT"
echo "Compute events: $COMPUTE_COUNT"

if [ $SDMA_COUNT -gt 0 ]; then
    echo "✓ SDMA 拷贝走 KFD Ring"
fi

if [ $COMPUTE_COUNT -eq 0 ]; then
    echo "✓ Compute kernel 不走 KFD Ring"
fi
```

---

## 4️⃣ HSA Runtime 层的实现

### 4.1 `hsa_amd_memory_async_copy`

**文件**: `ROCm_keyDriver/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/hsa_ext_amd.cpp`

```cpp
hsa_status_t hsa_amd_memory_async_copy(void* dst, hsa_agent_t dst_agent,
                                       const void* src, hsa_agent_t src_agent,
                                       size_t size,
                                       uint32_t num_dep_signals,
                                       const hsa_signal_t* dep_signals,
                                       hsa_signal_t completion_signal) {
    // ...
    return core::Runtime::runtime_singleton_->CopyMemory(
        dst, dst_agent, src, src_agent, size,
        dep_signal_list, *out_signal_obj);
}
```

**关键点**：
- ✅ HSA Runtime 的 `CopyMemory` 会**自动选择**拷贝方式
- ✅ 有专门的 `hsa_amd_memory_async_copy_on_engine` 可以**指定引擎**

### 4.2 强制使用 SDMA

```cpp
hsa_status_t hsa_amd_memory_async_copy_on_engine(
    void* dst, hsa_agent_t dst_agent,
    const void* src, hsa_agent_t src_agent,
    size_t size,
    uint32_t num_dep_signals,
    const hsa_signal_t* dep_signals,
    hsa_signal_t completion_signal,
    hsa_amd_sdma_engine_id_t engine_id,  // ← 指定 SDMA 引擎
    bool force_copy_on_sdma) {           // ← 强制使用 SDMA
    
    return core::Runtime::runtime_singleton_->CopyMemoryOnEngine(
        dst, dst_agent, src, src_agent, size,
        dep_signal_list, *out_signal_obj,
        engine_id, force_copy_on_sdma);
}
```

---

## 5️⃣ 两种拷贝机制对比

### 5.1 Blit Shader (Compute-based Copy)

**特点**：
- ✅ 使用 GPU Compute Units 执行拷贝 kernel
- ✅ 通过 **AQL Queue** 提交
- ✅ 使用 **Doorbell** 通知
- ✅ **不经过** KFD Ring
- ✅ **不触发** `drm_run_job` 事件
- ✅ 适合小拷贝、对齐良好的数据
- ⚠️ 占用 Compute 资源

**提交路径**：
```
应用
  ↓ hipMemcpy (D2D, small)
HIP Runtime
  ↓ 创建 Blit Kernel (或使用缓存的)
  ↓ 写入 AQL Dispatch Packet
  ↓ 写入 Doorbell
  ↓
GPU Compute Units
  ↓ 执行 Memory Copy Shader
  ↓ 完成拷贝
```

**Blit Kernel 示例**（简化）：
```cpp
// AMD 内部的 Blit Shader（简化版）
__global__ void blit_copy_kernel(uint64_t* dst, const uint64_t* src, size_t count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = src[idx];  // 128-bit 对齐的拷贝
    }
}
```

### 5.2 SDMA Engine (Hardware DMA)

**特点**：
- ✅ 使用专用 SDMA 硬件引擎
- ✅ 通过 **KFD 驱动** 提交
- ✅ 使用 **SDMA Ring**
- ✅ **触发** `drm_run_job` (sdma) 事件
- ✅ 适合大拷贝、H2D/D2H 拷贝
- ✅ **不占用** Compute 资源
- ⚠️ 有一定的启动开销

**提交路径**：
```
应用
  ↓ hipMemcpy (D2D large, or H2D/D2H, or force SDMA)
HIP Runtime
  ↓ ROCclr 选择 SDMA
HSA Runtime
  ↓ 构建 SDMA 命令
  ↓ 通过 KFD 提交
KFD Driver
  ↓ 写入 SDMA Ring
  ↓ 触发 drm_run_job
  ↓
SDMA Hardware Engine
  ↓ 执行拷贝
  ↓ 完成拷贝
```

---

## 6️⃣ 实际行为总结

### 6.1 常见 hipMemcpy 场景

| 场景 | 类型 | 大小 | 可能走 | 触发 drm_run_job? |
|------|------|------|--------|------------------|
| **D2D 小拷贝** | DeviceToDevice | <1MB | Blit Shader | ❌ 否 |
| **D2D 大拷贝** | DeviceToDevice | >10MB | SDMA | ✅ 是 (sdma) |
| **H2D Pinned** | HostToDevice | Any | Blit Shader | ❌ 否 |
| **H2D Pageable** | HostToDevice | Any | SDMA | ✅ 是 (sdma) |
| **D2H Pinned** | DeviceToHost | Any | Blit Shader | ❌ 否 |
| **D2H Pageable** | DeviceToHost | Any | SDMA | ✅ 是 (sdma) |
| **强制 SDMA** | DeviceToDeviceNoCU | Any | SDMA | ✅ 是 (sdma) |

**注意**：上表是**推测**，需要实际验证！

### 6.2 修正之前的结论

**之前的说法** ❌：
> "所有 hipMemcpy 都走 KFD SDMA Ring"

**更准确的说法** ✅：
> **hipMemcpy 的路径取决于具体情况**：
> - **小的 D2D 拷贝**：可能通过 Blit Shader (AQL Queue + Doorbell)
> - **大的 D2D 拷贝**：可能通过 SDMA (KFD Ring)
> - **H2D/D2H 拷贝**：取决于内存类型（Pinned vs Pageable）
> - **强制 SDMA**：一定通过 SDMA (KFD Ring)

---

## 7️⃣ 为什么会有两种机制？

### 7.1 性能考虑

**小拷贝用 Blit Shader**：
- ✅ 避免 SDMA 启动开销
- ✅ 利用 GPU 高带宽内部互联
- ✅ 可以和 compute kernel 并行

**大拷贝用 SDMA**：
- ✅ 释放 Compute 资源
- ✅ SDMA 吞吐量更高（对大数据）
- ✅ 不影响 compute kernel 执行

### 7.2 架构演进

**旧架构**（Vega, GCN）：
- SDMA 性能较弱
- 更依赖 Shader Copy

**新架构**（CDNA, RDNA）：
- SDMA 性能大幅提升
- 更多使用 SDMA
- 但仍保留 Blit Shader 作为备选

---

## 8️⃣ 验证计划

### Step 1: 编译测试程序

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit
hipcc test_sdma_path.cpp -o test_sdma_path
```

### Step 2: 运行实验

```bash
# 终端 1: 清空并启用 ftrace
sudo sh -c 'echo > /sys/kernel/debug/tracing/trace'
sudo sh -c 'echo 1 > /sys/kernel/debug/tracing/events/drm/drm_run_job/enable'

# 终端 2: 运行测试
./test_sdma_path

# 终端 1: 查看结果
sudo cat /sys/kernel/debug/tracing/trace | grep drm_run_job
```

### Step 3: 分析结果

**如果看到 sdma 事件**：
```
test_sdma-12345  [000] .... 1000.001: drm_run_job: ring=sdma0.0, job_count=1
test_sdma-12345  [001] .... 1000.010: drm_run_job: ring=sdma0.0, job_count=2
...
```
→ ✅ 确认 hipMemcpy 走 SDMA Ring

**如果没有任何事件**：
```
(空)
```
→ ✅ 确认 hipMemcpy 走 Blit Shader (Doorbell)

### Step 4: 对比不同拷贝大小

创建扩展测试：

```cpp
// 测试不同大小
for (int size_kb = 1; size_kb <= 102400; size_kb *= 10) {
    size_t bytes = size_kb * 1024;
    printf("Testing %d KB...\n", size_kb);
    hipMemcpy(d_dst, d_src, bytes, hipMemcpyDeviceToDevice);
}
```

观察 ftrace，找到 **Blit Shader → SDMA** 的切换阈值。

---

## 9️⃣ 待验证的关键问题

1. ✅ **D2D 小拷贝是否走 Doorbell？**
   - 推测：是（Blit Shader）
   - 验证：ftrace 无 sdma 事件

2. ✅ **D2D 大拷贝是否走 SDMA Ring？**
   - 推测：是（SDMA Engine）
   - 验证：ftrace 有 sdma 事件

3. ✅ **H2D/D2H 拷贝走哪条路径？**
   - 推测：取决于内存类型
   - 验证：分别测试 Pinned 和 Pageable

4. ✅ **切换阈值是多少？**
   - 推测：1MB - 10MB 之间
   - 验证：测试不同大小

5. ✅ **是否可以控制使用哪种方式？**
   - 推测：可以（通过 `hipMemcpyDeviceToDeviceNoCU`）
   - 验证：对比强制 SDMA 和默认模式

---

## 🔟 初步结论

基于代码分析，我们可以得出：

### 之前文档的问题

**文档说法** ❌：
> "100% 的 compute kernel 走 Doorbell"  
> "100% 的 SDMA 操作走 KFD Ring"

**问题**：
- ✅ Compute kernel 100% 走 Doorbell **正确**
- ❌ "SDMA 操作" 这个说法**不准确**
- ❓ hipMemcpy 不一定用 SDMA Engine
- ❓ 某些拷贝可能用 Blit Shader (也走 Doorbell)

### 更准确的说法

**修正后** ✅：
> **MES 模式下**：
> - **100% 的 compute kernel**（`hipLaunchKernel`）走 Doorbell
> - **内存拷贝**（`hipMemcpy`）可能有**两种**路径：
>   - **小拷贝 / 特定场景**：Blit Shader → Doorbell
>   - **大拷贝 / SDMA 引擎**：SDMA Ring → KFD
> - 具体选择由 **ROCclr Runtime 自动决定**

---

## 1️⃣1️⃣ 下一步行动

1. ✅ 运行 `test_sdma_path` 验证 H2D/D2H 拷贝
2. ✅ 扩展测试：不同大小的 D2D 拷贝
3. ✅ 使用 `rocprof` 详细追踪拷贝操作
4. ✅ 更新 `KERNEL_SUBMISSION_PATHS.md` 文档

---

## 相关文档

- [KERNEL_SUBMISSION_PATHS.md](./KERNEL_SUBMISSION_PATHS.md) - 需要更新
- [KERNEL_TRACE_02_HSA_RUNTIME.md](./KERNEL_TRACE_02_HSA_RUNTIME.md)
- [ROCM_PROFILING_TOOLS_GUIDE.md](./ROCM_PROFILING_TOOLS_GUIDE.md)

