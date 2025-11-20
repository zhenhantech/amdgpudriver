# ROCm 内存池 (Memory Pool) 详解

## 概述

**内存池 (Memory Pool)** 是ROCm内存管理体系中的核心概念，它是HSA标准定义的一种内存抽象，用于封装物理内存分区及其访问模型。

---

## 为什么需要内存池？

### 传统问题

在没有内存池概念之前：
- ❌ 应用不知道哪些内存在哪个物理位置
- ❌ 不清楚不同内存的访问特性（延迟、带宽、一致性）
- ❌ 难以针对NUMA架构优化
- ❌ 无法查询内存拓扑信息

### 内存池的优势

- ✅ **拓扑可见性**：应用可以查询系统内存拓扑
- ✅ **特性明确**：每个pool有明确的访问模型和性能特征
- ✅ **显式控制**：应用可以选择从哪个pool分配
- ✅ **NUMA优化**：可以优先分配到特定NUMA节点
- ✅ **异构支持**：统一管理CPU和GPU内存

---

## 内存池的定义

### HSA标准定义

```c
// rocr-runtime/runtime/hsa-runtime/inc/hsa_ext_amd.h

/**
 * @brief A memory pool encapsulates physical storage on an agent
 * along with a memory access model.
 *
 * @details 内存池封装了agent内存系统的物理分区及其访问模型。
 * 将单个内存系统划分为多个池，允许查询每个分区的访问路径属性。
 * 从池中分配的内存优先绑定到该池的物理分区。
 */
typedef struct hsa_amd_memory_pool_s {
  uint64_t handle;  // 不透明句柄
} hsa_amd_memory_pool_t;
```

### 关键特性

1. **物理分区**：每个pool代表一个物理内存分区
2. **访问模型**：定义了该内存的一致性和原子性语义
3. **优先绑定**：分配倾向于绑定到该pool的物理位置
4. **可查询**：可以查询大小、位置、标志等属性

---

## 内存池的类型

### 1. 按内存段 (Segment) 分类

```c
typedef enum {
  HSA_AMD_SEGMENT_GLOBAL = 0,    // 全局内存（VRAM或系统内存）
  HSA_AMD_SEGMENT_READONLY = 1,  // 只读内存
  HSA_AMD_SEGMENT_PRIVATE = 2,   // 私有内存（每个work-item）
  HSA_AMD_SEGMENT_GROUP = 3,     // 组内存（LDS - Local Data Share）
} hsa_amd_segment_t;
```

### 2. 按物理位置分类

```c
typedef enum {
  HSA_AMD_MEMORY_POOL_LOCATION_CPU = 0,  // CPU端（系统内存）
  HSA_AMD_MEMORY_POOL_LOCATION_GPU = 1   // GPU端（VRAM）
} hsa_amd_memory_pool_location_t;
```

### 3. 按访问模型分类（最重要）

#### Fine-Grained（细粒度）

```
特性：
  • CPU和GPU都可以直接访问
  • 支持原子操作
  • 缓存一致性由硬件保证
  • 适合频繁CPU-GPU交互

典型用途：
  • Kernel参数（kernarg）
  • 共享缓冲区
  • CPU-GPU同步变量
```

#### Coarse-Grained（粗粒度）

```
特性：
  • 主要由GPU访问
  • 需要显式同步
  • 更高的GPU带宽
  • 适合GPU密集计算

典型用途：
  • GPU本地VRAM
  • 大型计算缓冲区
  • GPU专用数据结构
```

#### Extended Scope Fine-Grained（扩展细粒度）

```
特性：
  • 设备级原子操作提升为系统级
  • 支持跨设备原子
  • 可能需要HDP刷新

典型用途：
  • 多GPU同步
  • 跨设备原子计数器
```

---

## 典型的内存池布局

### 单GPU系统

```
┌─────────────────────────────────────────────────────────────┐
│  CPU Agent (gfx000)                                         │
├─────────────────────────────────────────────────────────────┤
│  Pool 1: System Fine-Grained (可缓存)                       │
│    • 位置: CPU                                              │
│    • 大小: 32 GB                                            │
│    • 标志: FINE_GRAINED | KERNARG_INIT                      │
│    • 用途: 内核参数、共享数据                                │
├─────────────────────────────────────────────────────────────┤
│  Pool 2: System Coarse-Grained (不可缓存)                   │
│    • 位置: CPU                                              │
│    • 大小: 32 GB                                            │
│    • 标志: COARSE_GRAINED                                   │
│    • 用途: 大块CPU内存                                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  GPU Agent (gfx908)                                         │
├─────────────────────────────────────────────────────────────┤
│  Pool 3: GPU VRAM Coarse-Grained                            │
│    • 位置: GPU                                              │
│    • 大小: 16 GB                                            │
│    • 标志: COARSE_GRAINED                                   │
│    • 用途: GPU计算缓冲区、纹理                               │
├─────────────────────────────────────────────────────────────┤
│  Pool 4: GPU Fine-Grained (可选，不是所有GPU都有)            │
│    • 位置: GPU                                              │
│    • 大小: 256 MB                                           │
│    • 标志: FINE_GRAINED                                     │
│    • 用途: CPU可访问的GPU内存（BAR）                         │
├─────────────────────────────────────────────────────────────┤
│  Pool 5: LDS (Local Data Share)                            │
│    • 位置: GPU (on-chip)                                    │
│    • 大小: 64 KB per CU                                     │
│    • 段: GROUP                                              │
│    • 用途: Workgroup共享内存                                 │
└─────────────────────────────────────────────────────────────┘
```

### 多GPU系统（NUMA）

```
┌─────────────────────────────────────────────────────────────┐
│  NUMA Node 0                                                │
│  ├─ CPU Agent 0                                             │
│  │   ├─ System Fine-Grained Pool (16 GB)                    │
│  │   └─ System Coarse-Grained Pool (16 GB)                  │
│  └─ GPU Agent 0 (gfx908)                                    │
│      └─ VRAM Coarse-Grained Pool (16 GB)                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  NUMA Node 1                                                │
│  ├─ CPU Agent 1                                             │
│  │   ├─ System Fine-Grained Pool (16 GB)                    │
│  │   └─ System Coarse-Grained Pool (16 GB)                  │
│  └─ GPU Agent 1 (gfx908)                                    │
│      └─ VRAM Coarse-Grained Pool (16 GB)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 内存池的查询和遍历

### 遍历Agent的内存池

```c
// 应用代码：查询GPU的内存池
hsa_status_t pool_callback(hsa_amd_memory_pool_t pool, void* data) {
  // 查询池的段类型
  hsa_amd_segment_t segment;
  hsa_amd_memory_pool_get_info(pool, 
                                HSA_AMD_MEMORY_POOL_INFO_SEGMENT, 
                                &segment);
  
  if (segment != HSA_AMD_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;  // 跳过非全局段
  }
  
  // 查询池的标志
  uint32_t flags;
  hsa_amd_memory_pool_get_info(pool, 
                                HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, 
                                &flags);
  
  // 查询池的大小
  size_t size;
  hsa_amd_memory_pool_get_info(pool, 
                                HSA_AMD_MEMORY_POOL_INFO_SIZE, 
                                &size);
  
  // 查询是否可以分配
  bool alloc_allowed;
  hsa_amd_memory_pool_get_info(pool, 
                                HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, 
                                &alloc_allowed);
  
  printf("Pool: size=%lu MB, flags=0x%x, alloc_allowed=%d\n", 
         size / (1024*1024), flags, alloc_allowed);
  
  // 保存VRAM粗粒度池
  if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
    *(hsa_amd_memory_pool_t*)data = pool;
  }
  
  return HSA_STATUS_SUCCESS;
}

// 遍历GPU的所有内存池
hsa_amd_memory_pool_t vram_pool;
hsa_amd_agent_iterate_memory_pools(gpu_agent, pool_callback, &vram_pool);
```

### 查询内存池的详细信息

```c
// 内存池属性枚举
typedef enum {
  HSA_AMD_MEMORY_POOL_INFO_SEGMENT = 0,              // 内存段
  HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS = 1,         // 标志
  HSA_AMD_MEMORY_POOL_INFO_SIZE = 2,                 // 大小
  HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED = 5,// 是否可分配
  HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE = 6,// 分配粒度
  HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT = 7,// 对齐要求
  HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL = 8,    // 是否所有agent可访问
  HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE = 10,      // 最大单次分配
  HSA_AMD_MEMORY_POOL_INFO_LOCATION = 12,            // CPU还是GPU
} hsa_amd_memory_pool_info_t;
```

---

## 从内存池分配内存

### HSA Runtime层

```c
// rocr-runtime/runtime/hsa-runtime/core/runtime/runtime.cpp

hsa_status_t hsa_amd_memory_pool_allocate(
    hsa_amd_memory_pool_t memory_pool,  // 内存池
    size_t size,                        // 大小
    uint32_t flags,                     // 标志
    void** ptr)                         // 返回指针
{
  // 1. 验证参数
  if (ptr == nullptr || size == 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  
  // 2. 获取内存池（实际是MemoryRegion对象）
  core::MemoryRegion* region = core::MemoryRegion::Convert(memory_pool);
  
  // 3. 检查是否允许分配
  bool alloc_allowed;
  region->GetPoolInfo(HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, 
                      &alloc_allowed);
  if (!alloc_allowed) {
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  }
  
  // 4. 选择分配策略
  void* allocated_ptr = nullptr;
  
  if (region->IsSystem()) {
    // 系统内存 - 直接malloc或通过libhsakmt
    allocated_ptr = malloc(size);
  } else {
    // GPU内存 - 调用libhsakmt
    HsaMemFlags hsa_flags = {0};
    hsa_flags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
    hsa_flags.ui32.NoSubstitute = 1;
    
    HSAKMT_STATUS status = hsaKmtAllocMemory(
        region->node_id(),
        size,
        hsa_flags,
        &allocated_ptr);
    
    if (status != HSAKMT_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }
  }
  
  // 5. 记录分配信息（用于后续查询和释放）
  core::Runtime::runtime_singleton_->memory_allocations_[allocated_ptr] = {
    .size = size,
    .region = region,
    .agent = region->owner_agent()
  };
  
  *ptr = allocated_ptr;
  return HSA_STATUS_SUCCESS;
}
```

### ROCclr层的内存池选择

```cpp
// clr/rocclr/device/rocm/rocdevice.cpp

hsa_amd_memory_pool_t Device::getHostMemoryPool(
    MemorySegment mem_seg,
    const AgentInfo* agentInfo) const 
{
  if (agentInfo == nullptr) {
    agentInfo = cpu_agent_info_;
  }
  
  hsa_amd_memory_pool_t segment{0};
  
  switch (mem_seg) {
    case kKernArg:
      // 内核参数：需要fine-grained
      if (settings().fgs_kernel_arg_) {
        segment = agentInfo->kern_arg_pool;
        break;
      }
      // Falls through
      
    case kNoAtomics:
      // 不需要原子操作：使用coarse-grained（更快）
      if (agentInfo->coarse_grain_pool.handle != 0) {
        segment = agentInfo->coarse_grain_pool;
        break;
      }
      // Falls through
      
    case kAtomics:
      // 需要原子操作：使用fine-grained
      segment = agentInfo->fine_grain_pool;
      break;
      
    case kUncachedAtomics:
      // 需要uncached原子：使用extended fine-grained
      if (agentInfo->ext_fine_grain_pool.handle != 0) {
        segment = agentInfo->ext_fine_grain_pool;
        break;
      }
      
    default:
      guarantee(false, "Invalid Memory Segment");
      break;
  }
  
  assert(segment.handle != 0);
  return segment;
}

void* Device::deviceLocalAlloc(size_t size, const AllocationFlags& flags) const {
  // 根据标志选择GPU内存池
  const hsa_amd_memory_pool_t& pool =
      (flags.pseudo_fine_grain_ && gpu_ext_fine_grained_segment_.handle)
          ? gpu_ext_fine_grained_segment_         // 扩展细粒度
      : (flags.atomics_ && gpu_fine_grained_segment_.handle) 
          ? gpu_fine_grained_segment_             // 细粒度
          : gpuvm_segment_;                       // 粗粒度（默认）
  
  if (pool.handle == 0 || gpuvm_segment_max_alloc_ == 0) {
    return nullptr;
  }
  
  uint32_t hsa_mem_flags = 0;
  if (flags.executable_) {
    hsa_mem_flags |= HSA_AMD_MEMORY_POOL_ALLOC_EXECUTABLE;
  }
  
  void* ptr = nullptr;
  hsa_status_t status = Hsa::memory_pool_allocate(pool, size, &ptr);
  
  if (status != HSA_STATUS_SUCCESS) {
    return nullptr;
  }
  
  return ptr;
}
```

---

## 内存池的实际应用

### 1. hipMalloc() 的内存池选择

```
用户调用：
  hipMalloc(&ptr, 1024);
  
↓ HIP层
  • 决定分配类型：Device内存

↓ ROCclr层
  • 选择内存池：
    if (需要CPU访问)
      → gpu_fine_grained_segment_  (如果有)
    else
      → gpuvm_segment_  (VRAM粗粒度，默认)

↓ HSA Runtime层
  • hsa_amd_memory_pool_allocate(gpuvm_segment_, 1024, &ptr)

↓ ROCt层
  • hsaKmtAllocMemory(gpu_id, 1024, VRAM_FLAGS, &ptr)

↓ KFD层
  • 在VRAM中分配1024字节
```

### 2. hipHostMalloc() 的内存池选择

```
用户调用：
  hipHostMalloc(&ptr, 1024, hipHostMallocDefault);
  
↓ HIP层
  • 决定分配类型：Host pinned内存

↓ ROCclr层
  • 选择内存池：
    cpu_agent_info_->fine_grain_pool  (系统细粒度)

↓ HSA Runtime层
  • hsa_amd_memory_pool_allocate(fine_grain_pool, 1024, &ptr)

↓ ROCt层
  • hsaKmtAllocMemory(0, 1024, SYSTEM_FLAGS, &ptr)
  • hsaKmtRegisterMemory(ptr, 1024)  // 固定内存

↓ KFD层
  • 在系统内存中分配
  • 设置GPU页表映射
```

### 3. 内核参数内存

```
用户代码：
  myKernel<<<grid, block>>>(arg1, arg2, arg3);
  
↓ ROCclr层
  • 需要分配kernarg缓冲区
  • 选择内存池：
    cpu_agent_info_->kern_arg_pool  (带KERNARG_INIT标志)
  
特性：
  • 必须是fine-grained（GPU可直接读取）
  • 必须支持kernarg标志
  • 通常CPU写入，GPU读取
```

---

## 内存池的性能特性

### 各类内存池的性能对比

| 内存池类型 | CPU访问延迟 | GPU访问延迟 | CPU-GPU带宽 | GPU内部带宽 | 原子操作 |
|-----------|------------|------------|------------|------------|---------|
| **System Fine-Grained** | ~100 ns | ~500 ns | ~16 GB/s | ~16 GB/s | ✅ 支持 |
| **System Coarse-Grained** | ~100 ns | ~500 ns | ~16 GB/s | ~16 GB/s | ❌ 不支持 |
| **GPU VRAM Coarse** | N/A (不可直接访问) | ~5 ns | ~16 GB/s (PCIe) | **~1 TB/s** | ❌ 不支持 |
| **GPU Fine-Grained** | ~500 ns (通过BAR) | ~5 ns | ~16 GB/s | ~1 TB/s | ✅ 支持 |
| **GPU Extended FG** | ~500 ns | ~5 ns | ~16 GB/s | ~1 TB/s | ✅ 系统级 |

### 选择建议

```
用例 1：GPU密集计算，CPU不访问
  → GPU VRAM Coarse-Grained
  优势：最高GPU带宽，最低GPU延迟

用例 2：频繁CPU-GPU数据交换
  → System Fine-Grained
  优势：CPU和GPU都能高效访问

用例 3：CPU写入，GPU读取（如kernel参数）
  → System Fine-Grained (KernArg pool)
  优势：CPU写入快，GPU可直接读取

用例 4：多GPU原子同步
  → GPU Extended Fine-Grained
  优势：跨设备原子操作

用例 5：大块传输缓冲区
  → System Coarse-Grained
  优势：无缓存开销，适合DMA
```

---

## 内存池与NUMA

### NUMA系统的内存池

在多CPU/GPU的NUMA系统中，每个NUMA节点都有自己的内存池：

```
NUMA拓扑：
┌──────────────┬──────────────┐
│  Node 0      │  Node 1      │
│  CPU 0       │  CPU 1       │
│  GPU 0       │  GPU 1       │
│  32GB RAM    │  32GB RAM    │
│  16GB VRAM   │  16GB VRAM   │
└──────────────┴──────────────┘

内存池分布：
  Node 0:
    • System Pool 0 (32GB)
    • GPU VRAM Pool 0 (16GB)
  
  Node 1:
    • System Pool 1 (32GB)
    • GPU VRAM Pool 1 (16GB)
```

### NUMA感知分配

```c
// 应用可以选择从特定NUMA节点分配
hsa_amd_memory_pool_t numa0_pool;
hsa_amd_memory_pool_t numa1_pool;

// 从NUMA Node 0分配（靠近GPU 0）
hsa_amd_memory_pool_allocate(numa0_pool, size, 0, &ptr0);

// 从NUMA Node 1分配（靠近GPU 1）
hsa_amd_memory_pool_allocate(numa1_pool, size, 0, &ptr1);

// 优化：数据分配在使用它的GPU所在的NUMA节点
// 减少跨NUMA访问开销
```

---

## HIP 5.0+ 的流内存池 (Stream Memory Pool)

### 概念

HIP 5.0+引入了**流内存池**，用于减少分配开销：

```cpp
// HIP 5.0+ 异步分配API
hipError_t hipMallocAsync(void** ptr, size_t size, hipStream_t stream);
hipError_t hipFreeAsync(void* ptr, hipStream_t stream);
```

### 工作原理

```
传统 hipMalloc():
  每次调用 → ioctl → 内核分配 → 返回
  延迟：~10-100 μs

流内存池 hipMallocAsync():
  首次：预分配大块内存到pool
  后续：从pool快速分配
  延迟：~1 μs
  
流程：
  ┌─────────────┐
  │ hipMallocAsync(1MB, stream1) │
  └──────┬──────┘
         │ 首次调用
         ↓
  ┌─────────────────────────────┐
  │ 预分配 64MB 到 stream1 pool │
  └──────┬──────────────────────┘
         │
         ↓
  ┌─────────────────────┐
  │ 从pool快速分配 1MB  │
  └─────────────────────┘
  
  ┌─────────────┐
  │ hipMallocAsync(2MB, stream1) │  ← 第二次调用
  └──────┬──────┘
         │ 无需ioctl！
         ↓
  ┌─────────────────────┐
  │ 从已有pool分配 2MB  │
  └─────────────────────┘
```

### 内部实现

```cpp
// clr/hipamd/src/hip_memory.cpp
hipError_t hipMallocAsync(void** ptr, size_t size, hipStream_t stream) {
  HIP_INIT_API(hipMallocAsync, ptr, size, stream);
  
  // 1. 获取stream对应的内存池
  hip::Stream* hip_stream = hip::Stream::Convert(stream);
  MemoryPool* pool = hip_stream->getMemoryPool();
  
  // 2. 从pool分配（无需ioctl，如果pool有空间）
  void* allocated = pool->allocate(size);
  
  if (allocated == nullptr) {
    // 3. Pool空间不足，扩展pool
    pool->expand(size * 2);  // 扩展为2倍，减少后续扩展
    allocated = pool->allocate(size);
  }
  
  // 4. 记录分配到stream的依赖
  hip_stream->recordAllocation(allocated, size);
  
  *ptr = allocated;
  HIP_RETURN(hipSuccess);
}

hipError_t hipFreeAsync(void* ptr, hipStream_t stream) {
  HIP_INIT_API(hipFreeAsync, ptr, stream);
  
  // 1. 不立即释放，而是标记为"可回收"
  hip::Stream* hip_stream = hip::Stream::Convert(stream);
  MemoryPool* pool = hip_stream->getMemoryPool();
  
  // 2. 添加到stream的释放队列
  // 只有在stream的所有操作完成后才真正回收
  hip_stream->deferredFree(ptr);
  
  HIP_RETURN(hipSuccess);
}
```

### 优势

| 特性 | hipMalloc() | hipMallocAsync() |
|-----|------------|------------------|
| **延迟** | ~10-100 μs | ~1 μs |
| **ioctl** | 每次都需要 | 仅在pool扩展时 |
| **碎片** | 可能严重 | pool内部管理 |
| **并发** | 需要全局锁 | 每stream独立 |
| **适用场景** | 长生命周期 | 短生命周期、频繁分配 |

---

## 内存池的底层实现

### MemoryRegion类（HSA Runtime）

```cpp
// rocr-runtime/runtime/hsa-runtime/core/inc/memory_region.h

namespace rocr {
namespace core {

class MemoryRegion {
public:
  MemoryRegion(bool fine_grain, bool kernarg, bool full_profile, 
               bool extended_scope_fine_grain, bool user_visible, 
               core::Agent* owner);
  
  // 分配内存
  virtual hsa_status_t Allocate(
      size_t& size, 
      AllocateFlags alloc_flags, 
      void** address, 
      int agent_node_id) const = 0;
  
  // 释放内存
  virtual hsa_status_t Free(void* address, size_t size) const = 0;
  
  // 查询属性
  virtual hsa_status_t GetInfo(
      hsa_region_info_t attribute, 
      void* value) const = 0;
  
  // 访问器
  bool fine_grain() const { return fine_grain_; }
  bool kernarg() const { return kernarg_; }
  core::Agent* owner() const { return owner_; }
  
private:
  bool fine_grain_;                    // 是否细粒度
  bool kernarg_;                       // 是否支持kernarg
  bool full_profile_;                  // 是否full profile
  bool extended_scope_fine_grain_;     // 是否扩展细粒度
  bool user_visible_;                  // 是否用户可见
  core::Agent* owner_;                 // 所属agent
  
  // Fragment分配器（用于小对象）
  mutable BlockAllocator fragment_allocator_;
};

}  // namespace core
}  // namespace rocr
```

### AMD::MemoryRegion实现

```cpp
// rocr-runtime/runtime/hsa-runtime/core/runtime/amd_memory_region.cpp

namespace rocr {
namespace AMD {

MemoryRegion::MemoryRegion(
    bool fine_grain, bool kernarg, bool full_profile,
    bool extended_scope_fine_grain, bool user_visible,
    core::Agent* owner, const HsaMemoryProperties& mem_props)
    : core::MemoryRegion(fine_grain, kernarg, full_profile,
                        extended_scope_fine_grain, user_visible, owner),
      mem_props_(mem_props),
      fragment_allocator_(BlockAllocator(*this)) 
{
  // 设置内存标志
  mem_flag_.ui32.CoarseGrain = (fine_grain || extended_scope_fine_grain) ? 0 : 1;
  mem_flag_.ui32.ExtendedCoherent = extended_scope_fine_grain ? 1 : 0;
  
  if (IsLocalMemory()) {
    // GPU VRAM
    mem_flag_.ui32.PageSize = HSA_PAGE_SIZE_4KB;
    mem_flag_.ui32.NoSubstitute = 1;
    mem_flag_.ui32.HostAccess = 
        (mem_props_.HeapType == HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE) ? 0 : 1;
    mem_flag_.ui32.NonPaged = 1;
    virtual_size_ = kGpuVmSize;
  } else if (IsSystem()) {
    // 系统内存
    mem_flag_.ui32.PageSize = GetPageSize();
    mem_flag_.ui32.NoSubstitute = 0;
    mem_flag_.ui32.HostAccess = 1;
    mem_flag_.ui32.CachePolicy = HSA_CACHING_CACHED;
    if (kernarg) mem_flag_.ui32.Uncached = 1;
    virtual_size_ = os::GetUserModeVirtualMemorySize();
  }
  
  max_single_alloc_size_ = AlignDown(GetPhysicalSize(), GetPageSize());
}

hsa_status_t MemoryRegion::Allocate(
    size_t& size, AllocateFlags alloc_flags, 
    void** address, int agent_node_id) const 
{
  // 1. 验证
  if (address == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  
  if (size > max_single_alloc_size_) {
    return HSA_STATUS_ERROR_INVALID_ALLOCATION;
  }
  
  // 2. 页面对齐
  size = AlignUp(size, GetPageSize());
  
  // 3. 调用driver分配
  return owner()->driver().AllocateMemory(
      *this, alloc_flags, address, size, agent_node_id);
}

hsa_status_t MemoryRegion::Free(void* address, size_t size) const {
  // 尝试从fragment allocator释放
  if (fragment_allocator_.free(address)) {
    return HSA_STATUS_SUCCESS;
  }
  
  // 否则调用driver释放
  return owner()->driver().FreeMemory(address, size);
}

}  // namespace AMD
}  // namespace rocr
```

---

## 调试和工具

### 1. 查询系统内存池

```bash
# 使用rocminfo查看内存池
rocminfo | grep -A 20 "Pool"

# 输出示例：
  Pool 1                    
    Segment:                  GLOBAL; FLAGS: COARSE GRAINED
    Size:                     16777216(0x1000000) KB
    Allocatable:              TRUE
    Alloc Granule:            4KB
    Alloc Alignment:          4KB
    Accessible by all:        FALSE
    Location:                 GPU
  Pool 2                    
    Segment:                  GLOBAL; FLAGS: FINE GRAINED KERNARG
    Size:                     33554432(0x2000000) KB
    Allocatable:              TRUE
    Alloc Granule:            4KB
    Alloc Alignment:          4KB
    Accessible by all:        TRUE
    Location:                 CPU
```

### 2. 应用级查询

```cpp
// 查询内存池并打印信息
void print_memory_pools(hsa_agent_t agent) {
  auto callback = [](hsa_amd_memory_pool_t pool, void* data) {
    // 查询segment
    hsa_amd_segment_t segment;
    hsa_amd_memory_pool_get_info(pool, 
        HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
    
    if (segment == HSA_AMD_SEGMENT_GLOBAL) {
      // 查询详细信息
      uint32_t flags;
      size_t size;
      hsa_amd_memory_pool_location_t location;
      bool alloc_allowed;
      
      hsa_amd_memory_pool_get_info(pool, 
          HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
      hsa_amd_memory_pool_get_info(pool, 
          HSA_AMD_MEMORY_POOL_INFO_SIZE, &size);
      hsa_amd_memory_pool_get_info(pool, 
          HSA_AMD_MEMORY_POOL_INFO_LOCATION, &location);
      hsa_amd_memory_pool_get_info(pool, 
          HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed);
      
      printf("Pool: %s, Size: %lu MB, Flags: ", 
             location == HSA_AMD_MEMORY_POOL_LOCATION_GPU ? "GPU" : "CPU",
             size / (1024*1024));
      
      if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)
        printf("FINE_GRAINED ");
      if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)
        printf("COARSE_GRAINED ");
      if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT)
        printf("KERNARG ");
      
      printf(", Alloc: %s\n", alloc_allowed ? "YES" : "NO");
    }
    
    return HSA_STATUS_SUCCESS;
  };
  
  hsa_amd_agent_iterate_memory_pools(agent, callback, nullptr);
}
```

### 3. 环境变量

```bash
# 启用HSA调试
export HSA_ENABLE_DEBUG=1

# 查看内存分配
export AMD_LOG_LEVEL=4

# 跟踪内存池分配
export HSA_MEMORY_POOL_DEBUG=1
```

---

## 总结

### 内存池的关键点

```
1. 概念
   • 物理内存分区 + 访问模型
   • HSA标准定义
   • 拓扑可见

2. 分类
   • Fine-Grained vs Coarse-Grained
   • CPU vs GPU
   • System vs VRAM

3. 优势
   • 显式控制
   • 性能优化
   • NUMA支持

4. 使用
   • 查询：hsa_amd_agent_iterate_memory_pools()
   • 分配：hsa_amd_memory_pool_allocate()
   • 释放：hsa_amd_memory_pool_free()

5. 性能
   • VRAM Coarse: 最快GPU访问
   • System Fine: CPU-GPU都快
   • Pool选择影响性能
```

### 设计哲学

```
传统方式：
  malloc() → 不知道在哪
  
内存池方式：
  query_pools() → 知道有哪些池
  select_pool() → 选择合适的池
  allocate(pool) → 从指定池分配
  
优势：
  ✅ 拓扑感知
  ✅ 性能可预测
  ✅ 显式控制
  ✅ 异构友好
```

### 与其他概念的关系

```
内存池 (Memory Pool)
  ↓ 是
内存域 (Memory Region)
  ↓ 包含
内存分配 (Memory Allocation)
  ↓ 使用
物理内存 (Physical Memory)

层次关系：
  Pool → Region → Allocation → Physical
```

希望这份详细的解释能帮助您深入理解ROCm的内存池概念！
