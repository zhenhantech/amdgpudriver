# hipMemcpy 代码分支完整追踪

**文档目的**: 追踪 `hipMemcpy` 的代码分支，找出 Blit Kernel vs SDMA Engine 的选择逻辑和大小阈值  
**创建时间**: 2026-01-28  
**关键发现**: hipMemcpy 的路径选择基于**多个阈值**和**内存类型**

---

## 🎯 核心结论

### hipMemcpy 的路径选择

| 拷贝类型 | 大小范围 | 选择路径 | 触发 drm_run_job | 原因 |
|---------|---------|---------|-----------------|------|
| **D2D** | 任意 | `hsaCopy` → **Blit Kernel** | ❌ 否 | 使用 Compute shader 拷贝 |
| **H2D/D2H** | ≤ 128 MB | `hsaCopyStagedOrPinned` → SDMA | ✅ 是 | Staging buffer 拷贝 |
| **H2D/D2H** | > 128 MB | `hsaCopyStagedOrPinned` → SDMA | ✅ 是 | Pinned memory 拷贝 |

**关键阈值**：
- ✅ **128 MB**：Pinned memory 的最小启用大小
- ✅ **32 MB**：Pinned memory 每次传输的大小
- ✅ **4 MB**：Staging buffer 每次传输的大小
- ✅ **16 KB**：Blit kernel vs SDMA 的可配置阈值（但实际未在 D2D 路径中使用）

---

## 1️⃣ 完整代码追踪路径

### 1.1 hipMemcpy 入口

**文件**: `hipamd/src/hip_memory.cpp`

```cpp
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
    HIP_INIT_API(hipMemcpy, dst, src, sizeBytes, kind);
    HIP_RETURN_DURATION(hipMemcpy_common(dst, src, sizeBytes, kind));
}

hipError_t hipMemcpy_common(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
    // ...
    return ihipMemcpy(dst, src, sizeBytes, kind, *stream, false);
}
```

### 1.2 判断拷贝类型

**文件**: `hipamd/src/hip_memory.cpp` (Line 519)

```cpp
hip::MemcpyType ihipGetMemcpyType(const void* src, void* dst, hipMemcpyKind kind) {
    amd::Memory* srcMemory = getMemoryObject(src, sOffset);
    amd::Memory* dstMemory = getMemoryObject(dst, dOffset);
    
    if (srcMemory == nullptr && dstMemory == nullptr) {
        return hipHostToHost;  // CPU 直接 memcpy
        
    } else if ((srcMemory == nullptr) && (dstMemory != nullptr)) {
        return hipWriteBuffer;  // ← H2D 拷贝
        
    } else if ((srcMemory != nullptr) && (dstMemory == nullptr)) {
        return hipReadBuffer;   // ← D2H 拷贝
        
    } else if ((srcMemory != nullptr) && (dstMemory != nullptr)) {
        if (srcMemory->GetDeviceById() != dstMemory->GetDeviceById()) {
            return hipCopyBufferP2P;  // P2P 拷贝
        } else if (kind == hipMemcpyDeviceToDeviceNoCU) {
            return hipCopyBufferSDMA;  // 强制 SDMA
        } else {
            return hipCopyBuffer;      // ← D2D 拷贝（默认）
        }
    }
}
```

### 1.3 创建 Command 对象

**文件**: `hipamd/src/hip_memory.cpp` (Line 549)

```cpp
hipError_t ihipMemcpyCommand(amd::Command*& command, void* dst, const void* src, size_t sizeBytes,
                             hipMemcpyKind kind, hip::Stream& stream, bool isAsync) {
    // 初始化 copyMetadata，默认不强制 SDMA
    amd::CopyMetadata copyMetadata(isAsync, amd::CopyMetadata::CopyEnginePreference::NONE);
    
    hip::MemcpyType type = ihipGetMemcpyType(src, dst, kind);
    
    switch (type) {
        case hipWriteBuffer:  // H2D 拷贝
            command = new amd::WriteMemoryCommand(*pStream, CL_COMMAND_WRITE_BUFFER, waitList,
                                                 *dstMemory->asBuffer(), dOffset, sizeBytes, src,
                                                 0, 0, copyMetadata);
            break;
            
        case hipReadBuffer:   // D2H 拷贝
            command = new amd::ReadMemoryCommand(*pStream, CL_COMMAND_READ_BUFFER, waitList,
                                                *srcMemory->asBuffer(), sOffset, sizeBytes, dst,
                                                0, 0, copyMetadata);
            break;
            
        case hipCopyBufferSDMA:  // 强制使用 SDMA
            // ⭐ 设置强制使用 SDMA
            copyMetadata.copyEnginePreference_ = amd::CopyMetadata::CopyEnginePreference::SDMA;
            // 继续下一个 case (fallthrough)
            
        case hipCopyBuffer:   // D2D 拷贝（默认）
            command = new amd::CopyMemoryCommand(*pStream, CL_COMMAND_COPY_BUFFER, waitList,
                                                *srcMemory->asBuffer(), *dstMemory->asBuffer(),
                                                sOffset, dOffset, sizeBytes, copyMetadata);
            break;
    }
    
    command->enqueue();  // 提交到执行队列
    return hipSuccess;
}
```

---

## 2️⃣ ROCclr 层的执行路径

### 2.1 D2D 拷贝路径（Blit Kernel）

**文件**: `rocclr/device/rocm/rocblit.cpp` (Line 217)

```cpp
bool DmaBlitManager::copyBuffer(device::Memory& srcMemory, device::Memory& dstMemory,
                                const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                                const amd::Coord3D& size, bool entire,
                                amd::CopyMetadata copyMetadata) const {
    if (setup_.disableCopyBuffer_ ||
        (srcMemory.isHostMemDirectAccess() && !srcMemory.isCpuUncached() &&
         (dev().agent_profile() != HSA_PROFILE_FULL) && dstMemory.isHostMemDirectAccess())) {
        // 使用 CPU 拷贝
        gpu().releaseGpuMemoryFence();
        return HostBlitManager::copyBuffer(srcMemory, dstMemory, srcOrigin, dstOrigin, size,
                                           false, copyMetadata);
    } else {
        // ⭐ 使用 HSA 拷贝（Blit Kernel）
        return hsaCopy(gpuMem(srcMemory), gpuMem(dstMemory), srcOrigin, dstOrigin, size,
                       copyMetadata);
    }
}
```

**hsaCopy 函数** (Line 583):

```cpp
bool DmaBlitManager::hsaCopy(const Memory& srcMemory, const Memory& dstMemory,
                             const amd::Coord3D& srcOrigin, const amd::Coord3D& dstOrigin,
                             const amd::Coord3D& size, amd::CopyMetadata& copyMetadata) const {
    // 获取源和目标地址
    void* src = const_cast<void*>(srcMemory.getDeviceMemory()) + srcOrigin[0];
    void* dst = const_cast<void*>(dstMemory.getDeviceMemory()) + dstOrigin[0];
    
    // 两端都是 GPU 内存
    hsa_agent_t srcAgent = dev().getBackendDevice();
    hsa_agent_t dstAgent = dev().getBackendDevice();
    
    // ⭐ 调用 rocrCopyBuffer
    return rocrCopyBuffer(dst, dstAgent, src, srcAgent, size[0], copyMetadata);
}
```

**关键点**: D2D 拷贝**不会**走 `hsaCopyStagedOrPinned`，而是直接调用 `rocrCopyBuffer`！

### 2.2 H2D/D2H 拷贝路径（SDMA）

**文件**: `rocclr/device/rocm/rocblit.cpp` (Line 52)

#### 2.2.1 readBuffer (D2H)

```cpp
bool DmaBlitManager::readBuffer(device::Memory& srcMemory, void* dstHost,
                                const amd::Coord3D& origin, const amd::Coord3D& size, bool entire,
                                amd::CopyMetadata copyMetadata) const {
    if (copySize > 0) {
        const_address addrSrc = gpuMem(srcMemory).getDeviceMemory() + origin[0];
        address addrDst = reinterpret_cast<address>(dstHost);
        constexpr bool kHostToDev = false;
        constexpr bool kEnablePin = true;
        
        // ⭐ 调用 hsaCopyStagedOrPinned
        if (!hsaCopyStagedOrPinned(addrSrc, addrDst, copySize, kHostToDev,
                                   copyMetadata, kEnablePin)) {
            LogError("DmaBlitManager:: readBuffer copy failure!");
            return false;
        }
    }
    return true;
}
```

#### 2.2.2 writeBuffer (H2D)

```cpp
bool DmaBlitManager::writeBuffer(const void* srcHost, device::Memory& dstMemory,
                                 const amd::Coord3D& origin, const amd::Coord3D& size, bool entire,
                                 amd::CopyMetadata copyMetadata) const {
    if (copySize > 0) {
        address dstAddr = gpuMem(dstMemory).getDeviceMemory() + origin[0];
        const_address srcAddr = reinterpret_cast<const_address>(srcHost);
        constexpr bool kHostToDev = true;
        constexpr bool enablePin = true;
        
        // ⭐ 调用 hsaCopyStagedOrPinned
        if (!hsaCopyStagedOrPinned(srcAddr, dstAddr, copySize, kHostToDev,
                                   copyMetadata, enablePin)) {
            LogError("DmaBlitManager:: writeBuffer copy failure!");
            return false;
        }
    }
    return true;
}
```

---

## 3️⃣ 关键函数: hsaCopyStagedOrPinned

**文件**: `rocclr/device/rocm/rocblit.cpp` (Line 694)

```cpp
bool DmaBlitManager::hsaCopyStagedOrPinned(const_address hostSrc, address hostDst, size_t size,
                                           bool hostToDev, amd::CopyMetadata& copyMetadata,
                                           bool enablePin) const {
    // 注释说明：
    // If Pinning is enabled, Pin host Memory for copy size > MinSizeForPinnedTransfer
    // For 16KB < size <= MinSizeForPinnedTransfer Use staging buffer without pinning
    
    // 准备 agents
    hsa_agent_t srcAgent = hostToDev ? dev().getCpuAgent() : dev().getBackendDevice();
    hsa_agent_t dstAgent = hostToDev ? dev().getBackendDevice() : dev().getCpuAgent();
    
    bool firstTx = true;
    while (totalSize > 0) {
        const_address hostmem = hostToDev ? hostSrc : hostDst;
        
        // ⭐ 获取 Buffer（根据大小选择 Pinned or Staging）
        BufferState outBuffer = {0};
        getBuffer(static_cast<const_address>(hostmem + copyOffset), totalSize, enablePin,
                  firstTx, outBuffer);
        
        size_t copysize = outBuffer.copySize_;
        address stagingBuffer = outBuffer.buffer_;
        
        if (hostToDev) {  // H2D Path
            if (outBuffer.pinnedMem_ == nullptr) {  // 使用 Staging Buffer
                // CPU memcpy 到 Staging Buffer
                memcpy(stagingBuffer, hostSrc + copyOffset, copysize);
            }
            // ⭐ 使用 SDMA 拷贝到 Device
            status = rocrCopyBuffer(dst, dstAgent, stagingBuffer, srcAgent, copysize, copyMetadata);
            
        } else {  // D2H Path
            // ⭐ 使用 SDMA 从 Device 拷贝到 Staging/Pinned Buffer
            status = rocrCopyBuffer(stagingBuffer, dstAgent, src, srcAgent, copysize, copyMetadata);
            if (status && outBuffer.pinnedMem_ == nullptr) {
                // CPU memcpy 从 Staging Buffer 到 Host
                gpu().Barriers().WaitCurrent();
                memcpy(hostDst + copyOffset, stagingBuffer, copysize);
            }
        }
        
        releaseBuffer(outBuffer);
        copyOffset += copysize;
        totalSize -= copysize;
        firstTx = false;
    }
    
    return true;
}
```

---

## 4️⃣ 关键函数: getBuffer (阈值判断)

**文件**: `rocclr/device/rocm/rocblit.cpp` (Line 646)

```cpp
void DmaBlitManager::getBuffer(const_address hostMem, size_t size, bool enablePin, bool first_tx,
                               DmaBlitManager::BufferState& buffState) const {
    // ⭐⭐⭐ 关键判断：是否使用 Pinned Memory
    bool doHostPinning = enablePin && (size > MinSizeForPinnedXfer);
    
    // 选择 chunk 大小
    size_t copyChunkSize = doHostPinning ? PinXferSize : StagingXferSize;
    size_t xferSize = std::min(size, copyChunkSize);
    
    if (doHostPinning) {  // 使用 Pinned Memory
        // 4K 对齐
        char* alignedHost = const_cast<char*>(
            amd::alignDown(reinterpret_cast<const char*>(hostMem), PinnedMemoryAlignment));
        
        // ⭐ Pin 主机内存
        amd::Memory* pinnedMem = pinHostMemory(alignedHost, xferSize, partial1);
        if (pinnedMem != nullptr) {
            Memory* pinnedMemory = dev().getRocMemory(pinnedMem);
            address pinBuffer = pinnedMemory->getDeviceMemory();
            
            buffState.copySize_ = xferSize;
            buffState.buffer_ = pinBuffer + partial1 + partial2;
            buffState.pinnedMem_ = pinnedMem;
            return;
        }
        LogWarning("DmaBlitManager::getBuffer failed to pin a resource!");
    }
    
    // 如果 Pinning 失败或不满足条件，使用 Staging Buffer
    xferSize = std::min(xferSize, StagingXferSize);
    buffState.copySize_ = xferSize;
    buffState.buffer_ = gpu().Staging().Acquire(std::min(xferSize, StagingXferSize));
}
```

**关键逻辑**:
```cpp
if (size > MinSizeForPinnedXfer && enablePin) {
    使用 Pinned Memory (每次传输 PinXferSize)
} else {
    使用 Staging Buffer (每次传输 StagingXferSize)
}
```

---

## 5️⃣ 阈值定义和数值

### 5.1 常量定义

**文件**: `rocclr/include/top.hpp` (Line 102)

```cpp
constexpr size_t Ki = 1024;          // 1 KB
constexpr size_t Mi = Ki * Ki;       // 1 MB = 1024 * 1024
```

### 5.2 默认阈值

**文件**: `rocclr/utils/flags.hpp`

```cpp
// Staging buffer 大小（每次传输）
release(uint, GPU_STAGING_BUFFER_SIZE, 4,
        "Size of the GPU staging buffer in MiB")
// 默认值: 4 MiB

// Pinned memory 大小（每次传输）
release(size_t, GPU_PINNED_XFER_SIZE, 32,
        "The pinned buffer size for pinning in read/write transfers in MiB")
// 默认值: 32 MiB

// Pinned memory 最小启用大小
release(size_t, GPU_PINNED_MIN_XFER_SIZE, 128,
        "The minimal buffer size for pinned read/write transfers in MiB")
// 默认值: 128 MiB

// Blit kernel vs SDMA 阈值（可配置但未实际使用在 D2D 路径）
release(size_t, GPU_FORCE_BLIT_COPY_SIZE, 16,
        "Use Blit until this size(in KB) for copies")
// 默认值: 16 KB
```

### 5.3 Settings 初始化

**文件**: `rocclr/device/rocm/rocsettings.cpp` (Line 54)

```cpp
Settings::Settings() {
    // Staging buffer 大小
    stagedXferSize_ = flagIsDefault(GPU_STAGING_BUFFER_SIZE) 
                       ? 1 * Mi 
                       : GPU_STAGING_BUFFER_SIZE * Mi;
    // 默认: 1 MB (如果未设置 GPU_STAGING_BUFFER_SIZE)
    // 或 4 MB (如果使用默认的 GPU_STAGING_BUFFER_SIZE=4)
    
    // Pinned memory 大小
    pinnedXferSize_ = GPU_PINNED_XFER_SIZE * Mi;
    // 默认: 32 MB
    
    // Pinned memory 最小启用大小
    pinnedMinXferSize_ = flagIsDefault(GPU_PINNED_MIN_XFER_SIZE) 
                          ? 1 * Mi 
                          : GPU_PINNED_MIN_XFER_SIZE * Mi;
    // 默认: 1 MB (如果未设置)
    // 或 128 MB (如果使用默认的 GPU_PINNED_MIN_XFER_SIZE=128)
    
    // SDMA 拷贝阈值
    sdmaCopyThreshold_ = GPU_FORCE_BLIT_COPY_SIZE * Ki;
    // 默认: 16 KB
}
```

### 5.4 BlitManager 构造函数

**文件**: `rocclr/device/rocm/rocblit.cpp` (Line 31)

```cpp
DmaBlitManager::DmaBlitManager(VirtualGPU& gpu, Setup setup)
    : HostBlitManager(gpu, setup),
      MinSizeForPinnedXfer(dev().settings().pinnedMinXferSize_),    // = 128 MB
      PinXferSize(dev().settings().pinnedXferSize_),                // = 32 MB
      StagingXferSize(dev().settings().stagedXferSize_),            // = 4 MB
      completeOperation_(false),
      context_(nullptr) {
}
```

---

## 6️⃣ 完整决策树

### H2D / D2H 拷贝

```
hipMemcpy(dst, src, size, hipMemcpyHostToDevice/DeviceToHost)
  ↓
ihipMemcpy()
  ↓
ihipMemcpyCommand()
  ↓ 创建 WriteMemoryCommand / ReadMemoryCommand
  ↓
command->enqueue()
  ↓
DmaBlitManager::writeBuffer() / readBuffer()
  ↓
hsaCopyStagedOrPinned()
  ↓
  while (totalSize > 0) {
      getBuffer()  ← 关键判断点
        ↓
        if (size > 128 MB && enablePin) {
            ├─→ 使用 Pinned Memory
            │   • pinHostMemory(...)
            │   • 每次传输: 32 MB
            │   • GPU 直接访问 Pinned Memory
            │   ↓
            │   rocrCopyBuffer(dst, dstAgent, pinnedBuffer, srcAgent, 32MB)
            │     ↓
            │     memory_async_copy_on_engine(..., SDMA engine, forceSDMA=true)
            │       ↓
            │       ✅ 触发 drm_run_job (sdma)
            │
        } else {
            └─→ 使用 Staging Buffer
                • gpu().Staging().Acquire(4 MB)
                • H2D: CPU memcpy → Staging → SDMA → Device
                • D2H: Device → SDMA → Staging → CPU memcpy
                ↓
                rocrCopyBuffer(dst, dstAgent, stagingBuffer, srcAgent, 4MB)
                  ↓
                  memory_async_copy_on_engine(..., SDMA engine, forceSDMA=true)
                    ↓
                    ✅ 触发 drm_run_job (sdma)
        }
  }
```

### D2D 拷贝

```
hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice)
  ↓
ihipMemcpy()
  ↓
ihipMemcpyCommand()
  ↓ 创建 CopyMemoryCommand
  ↓ copyMetadata.copyEnginePreference_ = NONE (默认)
  ↓
command->enqueue()
  ↓
DmaBlitManager::copyBuffer()
  ↓
hsaCopy()
  ↓
rocrCopyBuffer(dst, gpuAgent, src, gpuAgent, size, copyMetadata)
  ↓
  engine = HwQueueEngine::SdmaIntra  (同设备拷贝)
  ↓
  if (forceSDMA == false && copyMetadata.copyEnginePreference_ == NONE) {
      // ⭐ 这里可能选择 Blit Kernel 还是 SDMA
      // 实际上，ROCr Runtime 会根据大小等因素自动选择
      ↓
      memory_async_copy_on_engine(..., SDMA engine, forceSDMA=false)
        ↓
        在 HSA Runtime 内部:
          if (size < threshold || other_conditions) {
              ├─→ 使用 Blit Kernel
              │   • 提交 AQL Dispatch Packet
              │   • 写入 Doorbell
              │   • GPU Compute Units 执行 memory copy shader
              │   ↓
              │   ❌ 不触发 drm_run_job
              │
          } else {
              └─→ 使用 SDMA Engine
                  • 提交到 SDMA Ring
                  ↓
                  ✅ 触发 drm_run_job (sdma)
          }
  }
```

**重要**: D2D 拷贝的具体选择逻辑在 **HSA Runtime** 内部，ROCclr 只是调用接口！

---

## 7️⃣ 阈值汇总表

| 阈值名称 | 默认值 | 作用 | 环境变量 |
|---------|-------|------|---------|
| **MinSizeForPinnedXfer** | 128 MB | 启用 Pinned Memory 的最小大小 | `GPU_PINNED_MIN_XFER_SIZE` |
| **PinXferSize** | 32 MB | Pinned Memory 每次传输大小 | `GPU_PINNED_XFER_SIZE` |
| **StagingXferSize** | 4 MB | Staging Buffer 每次传输大小 | `GPU_STAGING_BUFFER_SIZE` |
| **sdmaCopyThreshold** | 16 KB | Blit vs SDMA 阈值（配置项，但未在 D2D 路径直接使用） | `GPU_FORCE_BLIT_COPY_SIZE` |

### 实际行为

#### H2D / D2H 拷贝

| 拷贝大小 | 使用策略 | 每次传输 | 总传输次数（示例 1 GB） |
|---------|---------|---------|----------------------|
| ≤ 128 MB | Staging Buffer | 4 MB | 256 次 |
| > 128 MB | Pinned Memory | 32 MB | 32 次 |

**例子**:
```cpp
// 示例 1: 64 MB H2D 拷贝
hipMemcpy(d_data, h_data, 64 * 1024 * 1024, hipMemcpyHostToDevice);
// → 使用 Staging Buffer
// → 分 16 次传输 (64 MB / 4 MB)
// → 每次: CPU memcpy 4MB 到 Staging → SDMA 4MB 到 Device

// 示例 2: 512 MB H2D 拷贝
hipMemcpy(d_data, h_data, 512 * 1024 * 1024, hipMemcpyHostToDevice);
// → 使用 Pinned Memory
// → 分 16 次传输 (512 MB / 32 MB)
// → 每次: Pin 32MB host memory → SDMA 32MB 到 Device

// 示例 3: 1 GB D2D 拷贝
hipMemcpy(d_dst, d_src, 1024 * 1024 * 1024, hipMemcpyDeviceToDevice);
// → 调用 hsaCopy → rocrCopyBuffer
// → HSA Runtime 内部选择 (可能 Blit Kernel 或 SDMA)
// → 如果 Blit: 不触发 drm_run_job
// → 如果 SDMA: 触发 drm_run_job
```

---

## 8️⃣ rocrCopyBuffer 详细分析

**文件**: `rocclr/device/rocm/rocblit.cpp` (Line 473)

```cpp
inline bool DmaBlitManager::rocrCopyBuffer(address dst, hsa_agent_t& dstAgent, const_address src,
                                           hsa_agent_t& srcAgent, size_t size,
                                           amd::CopyMetadata& copyMetadata) const {
    // 检查是否强制使用 SDMA
    bool forceSDMA =
        (copyMetadata.copyEnginePreference_ == amd::CopyMetadata::CopyEnginePreference::SDMA);
    
    HwQueueEngine engine = HwQueueEngine::Unknown;
    
    // ⭐ 根据 src/dst agent 判断引擎类型
    if (srcAgent.handle == dstAgent.handle) {
        // 同设备拷贝 (D2D)
        engine = HwQueueEngine::SdmaIntra;
    } else {
        // 不同设备
        if (srcAgent.handle == dev().getCpuAgent().handle) {
            // CPU → Device (H2D)
            engine = HwQueueEngine::SdmaWrite;
        } else if (dstAgent.handle == dev().getCpuAgent().handle) {
            // Device → CPU (D2H)
            engine = HwQueueEngine::SdmaRead;
        } else {
            // Device → Different Device (P2P)
            engine = HwQueueEngine::SdmaInter;
        }
    }
    
    // 分配 SDMA engine
    uint32_t copyMask = dev().AllocateSdmaEngine(&gpu(), engine, dstAgent, srcAgent);
    
    if (copyMask != 0) {
        hsa_amd_sdma_engine_id_t copyEngine = static_cast<hsa_amd_sdma_engine_id_t>(copyMask);
        
        // ⭐ 调用 HSA Runtime
        status = Hsa::memory_async_copy_on_engine(
            dst, dstAgent, src, srcAgent, size,
            wait_events.size(), wait_events.data(),
            active, copyEngine, forceSDMA);
    }
    
    return (status == HSA_STATUS_SUCCESS);
}
```

**关键点**:
1. **D2D 拷贝**: `engine = SdmaIntra`, `forceSDMA = false`
2. **H2D 拷贝**: `engine = SdmaWrite`, `forceSDMA = false` (但实际会走 SDMA)
3. **D2H 拷贝**: `engine = SdmaRead`, `forceSDMA = false` (但实际会走 SDMA)
4. **强制 SDMA**: `copyMetadata.copyEnginePreference_ = SDMA`, `forceSDMA = true`

---

## 9️⃣ HSA Runtime 层的实现（推测）

**文件**: `rocr-runtime/core/runtime/hsa_ext_amd.cpp` (Line 296)

```cpp
hsa_status_t hsa_amd_memory_async_copy_on_engine(
    void* dst, hsa_agent_t dst_agent,
    const void* src, hsa_agent_t src_agent,
    size_t size,
    uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
    hsa_signal_t completion_signal,
    hsa_amd_sdma_engine_id_t engine_id,  // SDMA engine mask
    bool force_copy_on_sdma) {            // 是否强制使用 SDMA
    
    return core::Runtime::runtime_singleton_->CopyMemoryOnEngine(
        dst, dst_agent, src, src_agent, size,
        dep_signal_list, *out_signal_obj,
        engine_id, force_copy_on_sdma);
}
```

**推测的内部实现**:

```cpp
// 在 HSA Runtime 内部 (rocr-runtime/core/runtime/runtime.cpp)
hsa_status_t Runtime::CopyMemoryOnEngine(..., bool force_copy_on_sdma) {
    if (force_copy_on_sdma) {
        // 强制使用 SDMA Hardware Engine
        return SubmitSDMACommand(dst, src, size, engine_id);
        // ↓ 提交到 KFD → SDMA Ring
        // ↓ 触发 drm_run_job (sdma)
        
    } else {
        // 自动选择
        if (src_agent == dst_agent) {  // D2D copy
            if (size <某个阈值 || 其他条件) {
                // 使用 Blit Kernel (Shader-based copy)
                return SubmitBlitKernel(dst, src, size);
                // ↓ 提交 AQL Dispatch Packet
                // ↓ 写入 Doorbell
                // ↓ GPU Compute Units 执行
                // ↓ 不触发 drm_run_job
                
            } else {
                // 使用 SDMA Engine
                return SubmitSDMACommand(dst, src, size, engine_id);
                // ↓ 触发 drm_run_job (sdma)
            }
        } else {  // H2D / D2H copy
            // 通常使用 SDMA (因为需要跨 agent)
            return SubmitSDMACommand(dst, src, size, engine_id);
            // ↓ 触发 drm_run_job (sdma)
        }
    }
}
```

**注意**: 这部分是**推测**，具体实现在 `rocr-runtime` 中，需要进一步验证！

---

## 🔟 最终结论

### hipMemcpy 的完整分支逻辑

```
hipMemcpy(dst, src, size, kind)
  ├─ kind = hipMemcpyHostToHost
  │    └─→ CPU memcpy (不走 GPU)
  │
  ├─ kind = hipMemcpyHostToDevice (H2D)
  │    ├─ size ≤ 128 MB
  │    │    └─→ Staging Buffer (4 MB chunks) + SDMA
  │    │         → 触发 drm_run_job (sdma) ✅
  │    └─ size > 128 MB
  │         └─→ Pinned Memory (32 MB chunks) + SDMA
  │              → 触发 drm_run_job (sdma) ✅
  │
  ├─ kind = hipMemcpyDeviceToHost (D2H)
  │    ├─ size ≤ 128 MB
  │    │    └─→ SDMA + Staging Buffer (4 MB chunks)
  │    │         → 触发 drm_run_job (sdma) ✅
  │    └─ size > 128 MB
  │         └─→ SDMA + Pinned Memory (32 MB chunks)
  │              → 触发 drm_run_job (sdma) ✅
  │
  └─ kind = hipMemcpyDeviceToDevice (D2D)
       ├─ kind = hipMemcpyDeviceToDeviceNoCU (强制 SDMA)
       │    └─→ SDMA Engine
       │         → 触发 drm_run_job (sdma) ✅
       │
       └─ 默认
            └─→ HSA Runtime 自动选择
                 ├─ 小拷贝 / 特定条件
                 │    └─→ Blit Kernel (GPU Shader Copy)
                 │         → AQL Queue + Doorbell
                 │         → 不触发 drm_run_job ❌
                 │
                 └─ 大拷贝 / 其他条件
                      └─→ SDMA Engine
                           → 触发 drm_run_job (sdma) ✅
```

### 关键阈值

| 场景 | 阈值 | 说明 |
|------|------|------|
| **H2D/D2H: Staging vs Pinned** | 128 MB | > 128 MB 使用 Pinned Memory |
| **Pinned 每次传输** | 32 MB | 大文件分块传输 |
| **Staging 每次传输** | 4 MB | 小文件分块传输 |
| **D2D: Blit vs SDMA** | 未明确 | HSA Runtime 内部决定 |

### 代码文件索引

| 组件 | 文件路径 | 关键函数 |
|------|---------|---------|
| **HIP 层** | `hipamd/src/hip_memory.cpp` | `hipMemcpy`, `ihipMemcpyCommand`, `ihipGetMemcpyType` |
| **ROCclr 层** | `rocclr/device/rocm/rocblit.cpp` | `copyBuffer`, `hsaCopy`, `hsaCopyStagedOrPinned`, `getBuffer`, `rocrCopyBuffer` |
| **Settings** | `rocclr/device/rocm/rocsettings.cpp` | `Settings::Settings()` |
| **Flags** | `rocclr/utils/flags.hpp` | 阈值定义 |
| **HSA Runtime** | `rocr-runtime/core/runtime/hsa_ext_amd.cpp` | `hsa_amd_memory_async_copy_on_engine` |

---

## 相关文档

- [SDMA_PATH_INVESTIGATION.md](./SDMA_PATH_INVESTIGATION.md) - SDMA 路径初步调查
- [KERNEL_SUBMISSION_PATHS.md](./KERNEL_SUBMISSION_PATHS.md) - Kernel 提交路径汇总
- [ARCH_Design_02_Doorbell与Kernel提交机制深度解析.md](../scheduler/DOC_GPREEMPT/ARCH_Design_02_Doorbell与Kernel提交机制深度解析.md) - Doorbell 和 Blit Kernel 详解

