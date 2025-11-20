# ROCm 内存管理分层详解

## 概述

以**内存管理**为例，展示 HIP、CLR、rocr-runtime、ROCt 和 KFD 五个组件在同一功能领域的：
- **相同点**：都处理 GPU 内存
- **不同点**：抽象层次、职责范围、接口形式
- **调用关系**：自上而下的依赖链

---

## 五层架构总览

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 5: HIP API (用户接口层)                               │
│  hip/include + clr/hipamd/src                               │
│  • hipMalloc(void** ptr, size_t size)                       │
│  • hipMemcpy(dst, src, size, kind)                          │
│  • hipFree(void* ptr)                                       │
│  职责: 提供类CUDA的C++ API                                   │
└────────────────────────┬────────────────────────────────────┘
                         │ 调用
┌────────────────────────▼────────────────────────────────────┐
│  Layer 4: CLR/ROCclr (设备抽象层)                           │
│  clr/rocclr/device + clr/rocclr/platform                    │
│  • amd::Memory (内存对象)                                    │
│  • roc::Memory (ROCm内存实现)                                │
│  • SVM、Buffer、Image管理                                    │
│  职责: 统一的内存对象抽象，支持HIP和OpenCL                    │
└────────────────────────┬────────────────────────────────────┘
                         │ 调用
┌────────────────────────▼────────────────────────────────────┐
│  Layer 3: HSA Runtime (标准运行时层)                        │
│  rocr-runtime/runtime/hsa-runtime                           │
│  • hsa_memory_allocate(region, size, ptr)                   │
│  • hsa_memory_copy(dst, src, size)                          │
│  • hsa_memory_register/deregister                           │
│  • hsa_amd_memory_pool_*                                     │
│  职责: 实现HSA标准的内存管理API                               │
└────────────────────────┬────────────────────────────────────┘
                         │ 调用
┌────────────────────────▼────────────────────────────────────┐
│  Layer 2: ROCt (Thunk Library)                              │
│  rocr-runtime/libhsakmt                                     │
│  • hsaKmtAllocMemory()                                       │
│  • hsaKmtMapMemoryToGPU()                                    │
│  • hsaKmtUnmapMemoryToGPU()                                  │
│  • hsaKmtRegisterMemory()                                    │
│  职责: 封装ioctl系统调用                                      │
└────────────────────────┬────────────────────────────────────┘
                         │ ioctl
┌────────────────────────▼────────────────────────────────────┐
│  Layer 1: KFD (Kernel Driver)                               │
│  kfd/amdkfd                                                 │
│  • kfd_ioctl_alloc_memory_of_gpu()                          │
│  • kfd_ioctl_map_memory_to_gpu()                            │
│  • GPU页表管理                                               │
│  • 物理内存分配                                              │
│  职责: 内核态内存管理，硬件控制                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 详细对比分析

### 1. HIP 层内存管理

**位置**：`clr/hipamd/src/hip_memory.cpp`

#### 主要API

```cpp
// 基本内存分配
hipError_t hipMalloc(void** ptr, size_t size);
hipError_t hipFree(void* ptr);

// 内存拷贝
hipError_t hipMemcpy(void* dst, const void* src, size_t size, 
                     hipMemcpyKind kind);
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t size,
                          hipMemcpyKind kind, hipStream_t stream);

// 主机内存
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags);
hipError_t hipHostFree(void* ptr);

// 高级内存分配
hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int flags);
hipError_t hipMallocPitch(void** ptr, size_t* pitch, 
                          size_t width, size_t height);

// 内存属性
hipError_t hipMemGetInfo(size_t* free, size_t* total);
hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attr, void* ptr);

// 内存池（HIP 5.0+）
hipError_t hipMallocAsync(void** ptr, size_t size, hipStream_t stream);
hipError_t hipFreeAsync(void* ptr, hipStream_t stream);
```

#### 功能特点

| 特性 | 说明 |
|-----|------|
| **抽象层次** | 最高层，用户友好 |
| **接口风格** | 类CUDA的C API |
| **内存类型** | Device、Host、Managed、Pinned |
| **同步性** | 同步和异步接口都有 |
| **职责** | 参数验证、类型转换、调用ROCclr |

#### 实现示例

```cpp
// clr/hipamd/src/hip_memory.cpp
hipError_t hipMalloc(void** ptr, size_t sizeBytes) {
  HIP_INIT_API(hipMalloc, ptr, sizeBytes);
  
  // 1. 参数验证
  if (ptr == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  
  // 2. 获取当前设备
  hip::Device* device = hip::getCurrentDevice();
  if (device == nullptr) {
    HIP_RETURN(hipErrorNoDevice);
  }
  
  // 3. 调用内部实现
  hipError_t status = ihipMalloc(ptr, sizeBytes, 0);
  
  HIP_RETURN(status);
}

static hipError_t ihipMalloc(void** ptr, size_t size, unsigned int flags) {
  // 获取设备
  hip::Device* device = hip::getCurrentDevice();
  
  // 调用 ROCclr 层
  amd::Memory* mem = device->createMemory(size, flags);
  
  if (mem == nullptr) {
    return hipErrorOutOfMemory;
  }
  
  // 返回设备指针
  *ptr = mem->getDevicePointer();
  
  return hipSuccess;
}
```

**特点**：
- ✅ 用户友好的API
- ✅ 错误检查和日志
- ✅ 性能分析钩子
- ✅ 不直接操作硬件

---

### 2. CLR/ROCclr 层内存管理

**位置**：
- `clr/rocclr/platform/memory.cpp` (通用抽象)
- `clr/rocclr/device/rocm/rocmemory.cpp` (ROCm实现)

#### 主要类和接口

```cpp
namespace amd {
  // 通用内存对象基类
  class Memory : public RuntimeObject {
  public:
    enum Type {
      Buffer,        // 线性缓冲区
      Image1D,       // 1D图像
      Image2D,       // 2D图像
      Image3D,       // 3D图像
      Pipe           // 管道
    };
    
    // 内存标志
    enum Flags {
      HostPtr        = 1 << 0,  // 使用主机指针
      AllocHostPtr   = 1 << 1,  // 分配主机内存
      CopyHostPtr    = 1 << 2,  // 拷贝主机数据
      WriteOnly      = 1 << 3,  // 只写
      ReadOnly       = 1 << 4,  // 只读
      // ... 更多标志
    };
    
    virtual bool create() = 0;
    virtual void* map(MapFlags flags) = 0;
    virtual void unmap() = 0;
    
  protected:
    void* hostMemory_;         // 主机内存指针
    device::Memory* deviceMemory_; // 设备内存对象
    size_t size_;              // 大小
    Context& context_;         // 上下文
  };
}

namespace roc {
  // ROCm 内存实现
  class Memory : public device::Memory {
  public:
    // 创建内存
    static Memory* create(size_t size, unsigned int flags);
    
    // 映射到主机
    void* map(amd::Device& device, MapFlags flags);
    
    // 获取GPU虚拟地址
    void* getDevicePointer() const { return dev_ptr_; }
    
  private:
    void* dev_ptr_;           // GPU虚拟地址
    hsa_amd_memory_pool_t pool_; // HSA内存池
  };
}
```

#### 功能特点

| 特性 | 说明 |
|-----|------|
| **抽象层次** | 中间层，设备无关 |
| **接口风格** | C++面向对象 |
| **内存类型** | Buffer、Image、SVM |
| **管理范围** | 内存对象生命周期 |
| **职责** | 内存对象抽象、状态管理、设备绑定 |

#### 实现示例

```cpp
// clr/rocclr/device/rocm/rocmemory.cpp
roc::Memory* roc::Memory::create(size_t size, unsigned int flags) {
  // 1. 创建内存对象
  Memory* memory = new Memory();
  
  // 2. 选择内存池
  hsa_amd_memory_pool_t pool;
  if (flags & CL_MEM_SVM_FINE_GRAIN_BUFFER) {
    pool = fine_grain_pool_;  // 细粒度（CPU可见）
  } else {
    pool = coarse_grain_pool_; // 粗粒度（GPU本地VRAM）
  }
  
  // 3. 调用 HSA Runtime 分配
  hsa_status_t status = Hsa::memory_allocate(pool, size, &memory->dev_ptr_);
  
  if (status != HSA_STATUS_SUCCESS) {
    delete memory;
    return nullptr;
  }
  
  // 4. 记录内存信息
  memory->size_ = size;
  memory->pool_ = pool;
  
  return memory;
}

void* roc::Memory::map(amd::Device& device, MapFlags flags) {
  // 如果是VRAM，需要先拷贝到系统内存
  if (pool_ == coarse_grain_pool_) {
    // 分配临时主机内存
    void* host_ptr = malloc(size_);
    
    // 从GPU拷贝到主机
    Hsa::memory_copy(host_ptr, dev_ptr_, size_);
    
    return host_ptr;
  } else {
    // 细粒度内存，CPU可直接访问
    return dev_ptr_;
  }
}
```

**特点**：
- ✅ 设备无关的抽象
- ✅ 支持多种内存类型
- ✅ 管理内存状态和迁移
- ✅ 对接HSA Runtime

---

### 3. HSA Runtime 层内存管理

**位置**：`rocr-runtime/runtime/hsa-runtime/core/runtime/`

#### 主要API

```cpp
// 内存域和池
hsa_status_t hsa_agent_iterate_regions(
    hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_region_t, void*),
    void* data);

hsa_status_t hsa_region_get_info(
    hsa_region_t region,
    hsa_region_info_t attribute,
    void* value);

// 内存分配（基础）
hsa_status_t hsa_memory_allocate(
    hsa_region_t region,
    size_t size,
    void** ptr);

hsa_status_t hsa_memory_free(void* ptr);

// AMD 扩展：内存池
hsa_status_t hsa_amd_memory_pool_allocate(
    hsa_amd_memory_pool_t memory_pool,
    size_t size,
    uint32_t flags,
    void** ptr);

hsa_status_t hsa_amd_memory_pool_free(void* ptr);

// 内存注册（固定系统内存）
hsa_status_t hsa_memory_register(
    void* ptr,
    size_t size);

hsa_status_t hsa_memory_deregister(
    void* ptr,
    size_t size);

// 内存拷贝
hsa_status_t hsa_memory_copy(
    void* dst,
    const void* src,
    size_t size);

// 内存属性
hsa_status_t hsa_amd_memory_fill(
    void* ptr,
    uint32_t value,
    size_t count);

hsa_status_t hsa_amd_agents_allow_access(
    uint32_t num_agents,
    const hsa_agent_t* agents,
    const uint32_t* flags,
    const void* ptr);
```

#### 内存池类型

```cpp
// HSA 内存池类型
typedef enum {
  HSA_AMD_SEGMENT_GLOBAL = 0,        // 全局内存（VRAM）
  HSA_AMD_SEGMENT_READONLY = 1,      // 只读内存
  HSA_AMD_SEGMENT_PRIVATE = 2,       // 私有内存
  HSA_AMD_SEGMENT_GROUP = 3,         // 组内存（LDS）
} hsa_amd_segment_t;

// 内存池标志
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT    // 内核参数
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED    // 细粒度
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED  // 粗粒度
```

#### 功能特点

| 特性 | 说明 |
|-----|------|
| **抽象层次** | HSA标准层 |
| **接口风格** | C标准API |
| **内存类型** | Region、Pool、Segment |
| **管理范围** | 内存域、拓扑、访问权限 |
| **职责** | HSA标准实现，调用ROCt |

#### 实现示例

```cpp
// rocr-runtime/runtime/hsa-runtime/core/runtime/runtime.cpp
hsa_status_t hsa_amd_memory_pool_allocate(
    hsa_amd_memory_pool_t memory_pool,
    size_t size,
    uint32_t flags,
    void** ptr) {
  
  // 1. 验证参数
  if (ptr == nullptr || size == 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  
  // 2. 获取内存池信息
  core::MemoryRegion* region = core::MemoryRegion::Convert(memory_pool);
  
  // 3. 选择分配策略
  void* allocated_ptr = nullptr;
  
  if (region->IsSystem()) {
    // 系统内存 - 使用 malloc
    allocated_ptr = malloc(size);
  } else {
    // GPU内存 - 调用 libhsakmt
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
  
  // 4. 记录分配信息
  core::Runtime::runtime_singleton_->memory_allocations_[allocated_ptr] = {
    .size = size,
    .region = region,
    .agent = region->owner_agent()
  };
  
  *ptr = allocated_ptr;
  
  return HSA_STATUS_SUCCESS;
}
```

**特点**：
- ✅ 实现HSA标准
- ✅ 管理内存拓扑和域
- ✅ 处理跨设备访问
- ✅ 调用libhsakmt

---

### 4. ROCt (libhsakmt) 层内存管理

**位置**：`rocr-runtime/libhsakmt/src/memory.c`

#### 主要API

```c
// 分配GPU内存
HSAKMT_STATUS hsaKmtAllocMemory(
    HSAuint32 NodeId,           // GPU节点ID
    HSAuint64 Size,             // 大小
    HsaMemFlags MemFlags,       // 标志
    void** MemoryAddress);      // 返回地址

// 释放GPU内存
HSAKMT_STATUS hsaKmtFreeMemory(
    void* MemoryAddress,
    HSAuint64 Size);

// 映射内存到GPU
HSAKMT_STATUS hsaKmtMapMemoryToGPU(
    void* MemoryAddress,
    HSAuint64 Size,
    HSAuint64* AlternateVAGPU);

// 从GPU解映射
HSAKMT_STATUS hsaKmtUnmapMemoryToGPU(
    void* MemoryAddress);

// 注册用户内存（固定）
HSAKMT_STATUS hsaKmtRegisterMemory(
    void* MemoryAddress,
    HSAuint64 Size);

// 注销用户内存
HSAKMT_STATUS hsaKmtDeregisterMemory(
    void* MemoryAddress);

// 内存拷贝
HSAKMT_STATUS hsaKmtMemoryCopy(
    void* Dest,
    void* Src,
    HSAuint64 Size);
```

#### 内存标志

```c
typedef union {
  struct {
    unsigned int NonPaged : 1;       // 不可换页
    unsigned int CachePolicy : 2;    // 缓存策略
    unsigned int ReadOnly : 1;       // 只读
    unsigned int PageSize : 2;       // 页面大小
    unsigned int HostAccess : 1;     // 主机访问
    unsigned int NoSubstitute : 1;   // 不替换
    unsigned int GDSMemory : 1;      // GDS内存
    unsigned int Scratch : 1;        // Scratch内存
    unsigned int CoarseGrain : 1;    // 粗粒度
    unsigned int Uncached : 1;       // 不缓存
    unsigned int AQLQueue : 1;       // AQL队列
  } ui32;
  HSAuint32 Value;
} HsaMemFlags;
```

#### 功能特点

| 特性 | 说明 |
|-----|------|
| **抽象层次** | 系统调用封装层 |
| **接口风格** | C API |
| **内存类型** | 按标志位区分 |
| **管理范围** | ioctl参数准备 |
| **职责** | 封装ioctl，管理文件描述符 |

#### 实现示例

```c
// rocr-runtime/libhsakmt/src/memory.c
HSAKMT_STATUS hsaKmtAllocMemory(
    HSAuint32 NodeId,
    HSAuint64 Size,
    HsaMemFlags MemFlags,
    void** MemoryAddress) {
  
  // 1. 准备ioctl参数
  struct kfd_ioctl_alloc_memory_of_gpu_args args = {0};
  args.gpu_id = get_gpu_id(NodeId);
  args.size = Size;
  args.flags = 0;
  
  // 2. 转换标志
  if (MemFlags.ui32.NonPaged) {
    args.flags |= KFD_IOC_ALLOC_MEM_FLAGS_VRAM;
  }
  if (MemFlags.ui32.HostAccess) {
    args.flags |= KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC;
  }
  if (MemFlags.ui32.CoarseGrain) {
    args.flags |= KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
  }
  
  // 3. 调用KFD ioctl
  int ret = ioctl(kfd_fd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &args);
  
  if (ret != 0) {
    return HSAKMT_STATUS_ERROR;
  }
  
  // 4. 返回GPU虚拟地址
  *MemoryAddress = (void*)args.va_addr;
  
  // 5. 记录分配（用于后续管理）
  mem_record_t record = {
    .va_addr = args.va_addr,
    .size = Size,
    .gpu_id = args.gpu_id,
    .handle = args.handle
  };
  add_mem_record(&record);
  
  return HSAKMT_STATUS_SUCCESS;
}
```

**特点**：
- ✅ 直接封装ioctl
- ✅ 管理/dev/kfd文件描述符
- ✅ 维护分配记录
- ✅ 参数转换和验证

---

### 5. KFD (Kernel Driver) 层内存管理

**位置**：`kfd/amdkfd/`

#### 主要ioctl命令

```c
// 分配GPU内存
#define AMDKFD_IOC_ALLOC_MEMORY_OF_GPU \
    AMDKFD_IOWR(0x1b, struct kfd_ioctl_alloc_memory_of_gpu_args)

// 释放GPU内存
#define AMDKFD_IOC_FREE_MEMORY_OF_GPU \
    AMDKFD_IOWR(0x1c, struct kfd_ioctl_free_memory_of_gpu_args)

// 映射内存到GPU
#define AMDKFD_IOC_MAP_MEMORY_TO_GPU \
    AMDKFD_IOWR(0x1d, struct kfd_ioctl_map_memory_to_gpu_args)

// 从GPU解映射内存
#define AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU \
    AMDKFD_IOWR(0x1e, struct kfd_ioctl_unmap_memory_from_gpu_args)
```

#### ioctl参数结构

```c
// 分配内存参数
struct kfd_ioctl_alloc_memory_of_gpu_args {
  __u64 va_addr;          // GPU虚拟地址（输出）
  __u64 size;             // 大小（输入）
  __u64 handle;           // 句柄（输出）
  __u64 mmap_offset;      // mmap偏移（输出）
  __u32 gpu_id;           // GPU ID（输入）
  __u32 flags;            // 标志（输入）
};

// 内存标志
#define KFD_IOC_ALLOC_MEM_FLAGS_VRAM          (1 << 0)  // VRAM
#define KFD_IOC_ALLOC_MEM_FLAGS_GTT           (1 << 1)  // GTT
#define KFD_IOC_ALLOC_MEM_FLAGS_USERPTR       (1 << 2)  // 用户指针
#define KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL      (1 << 3)  // Doorbell
#define KFD_IOC_ALLOC_MEM_FLAGS_COHERENT      (1 << 4)  // 一致性
#define KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC        (1 << 5)  // CPU可见
#define KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE (1 << 6)  // 不替换
#define KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM (1 << 7)  // AQL队列
```

#### 功能特点

| 特性 | 说明 |
|-----|------|
| **抽象层次** | 内核态，最底层 |
| **接口风格** | ioctl系统调用 |
| **内存类型** | VRAM、GTT、Doorbell、SVM |
| **管理范围** | 物理内存、页表、硬件寄存器 |
| **职责** | 实际的内存分配和硬件配置 |

#### 实现示例

```c
// kfd/amdkfd/kfd_chardev.c
static int kfd_ioctl_alloc_memory_of_gpu(
    struct file *filep,
    struct kfd_process *p,
    void *data) {
  
  struct kfd_ioctl_alloc_memory_of_gpu_args *args = data;
  struct kfd_dev *dev;
  struct kgd_mem *mem;
  uint64_t va_addr;
  int err;
  
  // 1. 查找设备
  dev = kfd_device_by_id(args->gpu_id);
  if (!dev)
    return -EINVAL;
  
  // 2. 根据标志选择内存类型
  uint32_t domain;
  if (args->flags & KFD_IOC_ALLOC_MEM_FLAGS_VRAM) {
    domain = AMDGPU_GEM_DOMAIN_VRAM;  // GPU本地VRAM
  } else if (args->flags & KFD_IOC_ALLOC_MEM_FLAGS_GTT) {
    domain = AMDGPU_GEM_DOMAIN_GTT;   // 系统内存GTT
  } else {
    domain = AMDGPU_GEM_DOMAIN_CPU;   // 系统内存
  }
  
  // 3. 调用AMDGPU驱动分配内存
  err = amdgpu_amdkfd_gpuvm_alloc_memory_of_gpu(
      dev->adev,              // amdgpu设备
      args->size,             // 大小
      p->vm,                  // 进程VM
      &mem,                   // 返回内存对象
      &va_addr,               // 返回VA地址
      domain,                 // 内存域
      args->flags);           // 标志
  
  if (err)
    return err;
  
  // 4. 设置GPU页表
  err = amdgpu_amdkfd_gpuvm_map_memory_to_gpu(
      dev->adev,
      mem,
      p->vm);
  
  if (err) {
    amdgpu_amdkfd_gpuvm_free_memory_of_gpu(dev->adev, mem);
    return err;
  }
  
  // 5. 添加到进程的内存列表
  mutex_lock(&p->mutex);
  list_add(&mem->list, &p->allocated_memory);
  mutex_unlock(&p->mutex);
  
  // 6. 返回结果
  args->va_addr = va_addr;
  args->handle = (uint64_t)mem;
  
  return 0;
}
```

**特点**：
- ✅ 直接操作硬件
- ✅ 管理GPU页表
- ✅ 分配物理内存
- ✅ 进程隔离和权限控制

---

## 完整调用链示例：`hipMalloc()`

### 代码流转

```
应用代码：
  hipMalloc(&ptr, 1024);
  
  ↓ 第1层：HIP API
  
clr/hipamd/src/hip_memory.cpp::hipMalloc()
  ├─ HIP_INIT_API()           // 初始化、日志
  ├─ 参数验证
  └─ ihipMalloc(ptr, size, 0)
      └─ hip::Device::createMemory()
      
  ↓ 第2层：ROCclr
  
clr/rocclr/platform/context.cpp::createMemory()
  └─ amd::Memory::create()
      └─ roc::Memory::create()
      
clr/rocclr/device/rocm/rocmemory.cpp::create()
  ├─ 选择内存池（VRAM/GTT）
  └─ Hsa::memory_allocate(pool, size, &ptr)
  
  ↓ 第3层：HSA Runtime
  
rocr-runtime/runtime/hsa-runtime/core/runtime/runtime.cpp
  └─ hsa_amd_memory_pool_allocate()
      ├─ 获取内存域信息
      └─ hsaKmtAllocMemory(node_id, size, flags, &ptr)
      
  ↓ 第4层：ROCt
  
rocr-runtime/libhsakmt/src/memory.c::hsaKmtAllocMemory()
  ├─ 准备ioctl参数
  ├─ 转换标志位
  └─ ioctl(kfd_fd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &args)
  
  ↓ 第5层：KFD
  
kfd/amdkfd/kfd_chardev.c::kfd_ioctl_alloc_memory_of_gpu()
  ├─ 验证进程和设备
  ├─ amdgpu_amdkfd_gpuvm_alloc_memory_of_gpu()
  │   ├─ 分配物理内存（VRAM或系统内存）
  │   ├─ 创建BO (Buffer Object)
  │   └─ 分配GPU虚拟地址
  ├─ amdgpu_amdkfd_gpuvm_map_memory_to_gpu()
  │   └─ 设置GPU页表
  └─ 返回VA地址和句柄
  
  ↓ 返回路径
  
KFD → ROCt → HSA Runtime → ROCclr → HIP → 应用
```

### 时序图

```
应用    HIP     ROCclr   HSA-RT   ROCt     KFD
 │       │        │        │        │        │
 │ hipMalloc()    │        │        │        │
 ├──────>│        │        │        │        │
 │       │ create │        │        │        │
 │       ├───────>│        │        │        │
 │       │        │ alloc  │        │        │
 │       │        ├───────>│        │        │
 │       │        │        │hsaKmt  │        │
 │       │        │        ├───────>│        │
 │       │        │        │        │ ioctl  │
 │       │        │        │        ├───────>│
 │       │        │        │        │        │ 分配物理内存
 │       │        │        │        │        │ 设置页表
 │       │        │        │        │<───────┤
 │       │        │        │<───────┤        │
 │       │        │<───────┤        │        │
 │       │<───────┤        │        │        │
 │<──────┤        │        │        │        │
 │       │        │        │        │        │
```

---

## 功能对比表

### 相同点

所有层都处理：
- ✅ GPU内存
- ✅ 内存生命周期（分配/释放）
- ✅ 内存映射和访问权限
- ✅ 错误处理

### 不同点

| 维度 | HIP | ROCclr | HSA Runtime | ROCt | KFD |
|-----|-----|--------|-------------|------|-----|
| **抽象层次** | 最高 | 中高 | 中等 | 中低 | 最低 |
| **接口风格** | C++ API | C++对象 | C API | C API | ioctl |
| **内存抽象** | Device/Host | Memory对象 | Region/Pool | 按标志 | VRAM/GTT |
| **用户** | 应用开发者 | Runtime开发者 | Runtime内部 | HSA Runtime | libhsakmt |
| **职责** | 易用性 | 对象管理 | 标准实现 | 系统调用 | 硬件控制 |
| **错误类型** | hipError_t | bool/指针 | hsa_status_t | HSAKMT_STATUS | errno |
| **同步性** | 同步/异步 | 同步为主 | 同步 | 同步 | 同步 |

### 特有功能

**HIP独有**：
- `hipMallocManaged()` - 统一内存
- `hipMallocAsync()` - 流异步分配
- `hipMemcpyAsync()` - 异步拷贝

**ROCclr独有**：
- Image对象（1D/2D/3D）
- SVM（Shared Virtual Memory）
- OpenCL兼容性

**HSA Runtime独有**：
- 内存域拓扑
- 跨Agent访问控制
- Fine/Coarse grain区分

**ROCt独有**：
- /dev/kfd文件描述符管理
- ioctl参数打包

**KFD独有**：
- GPU页表管理
- DMA映射
- 物理内存分配
- 进程隔离

---

## 内存类型映射

### HIP → ROCclr → HSA → KFD

```
HIP                  ROCclr              HSA                 KFD
────────────────────────────────────────────────────────────────────
hipMalloc()          Buffer              VRAM Pool           VRAM
                     Coarse Grain

hipHostMalloc()      Buffer              System Pool         GTT
                     Fine Grain

hipMallocManaged()   Buffer              SVM Pool            SVM
                     SVM

hipMallocPitch()     Buffer              VRAM Pool           VRAM
                     2D Layout

Image<T>             Image2D             Image Region        VRAM
                                                             +Metadata
```

---

## 性能考虑

### 各层开销

```
┌─────────────┬──────────────┬─────────────────┐
│    层次      │   开销类型    │    典型时间     │
├─────────────┼──────────────┼─────────────────┤
│ HIP         │ 函数调用     │ ~10 ns          │
│ ROCclr      │ 对象管理     │ ~50 ns          │
│ HSA Runtime │ 表查询       │ ~100 ns         │
│ ROCt        │ 参数打包     │ ~50 ns          │
│ KFD (ioctl) │ 上下文切换   │ ~1-2 μs         │
│ 物理分配     │ 硬件操作     │ ~10-100 μs      │
└─────────────┴──────────────┴─────────────────┘

总开销（首次分配）：~15-105 μs
总开销（缓存命中）：~200 ns（不进入内核）
```

### 优化策略

1. **内存池**（HIP 5.0+）
   ```cpp
   // 预分配，减少ioctl
   hipMallocAsync(stream);  // 从流内存池分配
   ```

2. **缓存**（ROCclr）
   ```cpp
   // ROCclr维护已分配内存的缓存
   // 释放后不立即释放，保留给后续使用
   ```

3. **批量操作**
   ```cpp
   // 一次分配大块，应用内部子分配
   hipMalloc(&big_ptr, large_size);
   ```

---

## 调试技巧

### 各层调试方法

```bash
# 1. HIP层：启用API跟踪
export HIP_TRACE_API=1
./myapp
# 输出：hipMalloc(0x7f..., 1024) = hipSuccess

# 2. ROCclr层：启用调试日志
export AMD_LOG_LEVEL=4
./myapp
# 输出：roc::Memory::create(1024) -> 0x7f...

# 3. HSA Runtime层：启用HSA调试
export HSA_ENABLE_DEBUG=1
./myapp
# 输出：hsa_memory_allocate() -> 0x7f...

# 4. ROCt层：跟踪系统调用
strace -e ioctl ./myapp
# 输出：ioctl(3, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, ...) = 0

# 5. KFD层：查看内核日志
dmesg | grep kfd
# 输出：[kfd] allocated 1024 bytes at 0x...
```

### GDB断点设置

```bash
gdb ./myapp

# 各层断点
(gdb) break hipMalloc                    # HIP层
(gdb) break roc::Memory::create          # ROCclr层
(gdb) break hsa_memory_allocate          # HSA层
(gdb) break hsaKmtAllocMemory            # ROCt层
# KFD层需要内核调试器

(gdb) run
# 会依次在各层停下
```

---

## 总结

### 五层职责总结

```
┌──────────┬──────────────┬────────────────┬──────────────┐
│   层次    │     职责      │     接口特点    │   典型用户   │
├──────────┼──────────────┼────────────────┼──────────────┤
│ HIP      │ 用户友好API  │ 类CUDA，易用   │ 应用开发者   │
│ ROCclr   │ 对象抽象     │ OOP，灵活      │ HIP/OpenCL   │
│ HSA-RT   │ 标准实现     │ HSA标准，严谨  │ ROCclr       │
│ ROCt     │ 系统调用封装 │ C API，直接    │ HSA Runtime  │
│ KFD      │ 硬件控制     │ ioctl，底层    │ libhsakmt    │
└──────────┴──────────────┴────────────────┴──────────────┘
```

### 设计原则

1. **分层清晰**：每层有明确边界
2. **抽象递进**：从高级到底层逐步具体化
3. **复用性**：ROCclr被HIP和OpenCL共享
4. **标准化**：HSA Runtime遵循HSA标准
5. **隔离性**：应用不直接接触硬件

### 关键理解

```
内存管理就像一个"快递系统"：

HIP      = 用户下单（简单界面）
ROCclr   = 快递公司（管理包裹）
HSA-RT   = 物流网络（标准流程）
ROCt     = 配送站（本地操作）
KFD      = 快递员（直达门户）

每层都在处理"包裹"（内存），但：
- 抽象层次不同
- 管理粒度不同
- 接口形式不同
- 但最终目标一致：高效可靠地交付
```

这就是ROCm内存管理的完整分层设计！每一层都有其不可或缺的作用。
