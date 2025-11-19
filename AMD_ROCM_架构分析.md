# AMD ROCm 软件架构及组件调用关系分析

## 目录
1. [架构概览](#架构概览)
2. [核心组件介绍](#核心组件介绍)
3. [调用关系与数据流](#调用关系与数据流)
4. [关键接口定义](#关键接口定义)
5. [内存管理](#内存管理)
6. [队列与任务调度](#队列与任务调度)

---

## 架构概览

AMD ROCm的软件栈采用分层架构，从上到下分为以下几层：

```
┌─────────────────────────────────────────────────────────┐
│              应用层 (Application)                        │
│         (CUDA/HIP Applications, OpenCL Apps)            │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                  HIP Runtime API                        │
│              (clr/hipamd - HIP AMD实现)                 │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              ROCclr (ROCm Common Language Runtime)      │
│           Device抽象层、内存管理、命令队列               │
│              (clr/rocclr - OpenCL基础)                  │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│            HSA Runtime (rocr-runtime/runtime)           │
│        HSA标准实现、代理管理、信号同步、AQL包调度        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│           libhsakmt / ROCt (Thunk Library)              │
│           (rocr-runtime/libhsakmt)                      │
│            用户态接口库，封装ioctl调用                   │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              KFD (Kernel Fusion Driver)                 │
│                   (kfd/amdkfd)                          │
│         Linux内核驱动，设备管理、内存管理、调度          │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                  AMD GPU 硬件                           │
└─────────────────────────────────────────────────────────┘
```

---

## 核心组件介绍

### 1. HIP (clr/hipamd)

**功能定位**：
- HIP (Heterogeneous-Compute Interface for Portability) 是AMD的GPU编程模型
- 提供类CUDA的C++ Runtime API和Kernel Language
- 允许开发者从单一源代码创建AMD和NVIDIA GPU的可移植应用

**主要模块**：
- `clr/hipamd/src/`: HIP API实现
  - `hip_memory.cpp`: 内存管理API (hipMalloc, hipMemcpy等)
  - `hip_platform.cpp`: 平台初始化和设备管理
  - `hip_module.cpp`: 模块加载和内核启动
  - `hip_stream.cpp`: 流管理
  - `hip_event.cpp`: 事件同步
  - `hip_graph*.cpp`: 计算图相关功能
  
- `clr/hipamd/include/hip/`: 公共头文件
  - 定义HIP API接口
  - 兼容层定义

**对外接口**：
```cpp
// 核心API示例
hipError_t hipMalloc(void** ptr, size_t size);
hipError_t hipMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind);
hipError_t hipLaunchKernel(const void* function_address, dim3 gridDim, 
                           dim3 blockDim, void** args, size_t sharedMemBytes, 
                           hipStream_t stream);
```

### 2. ROCclr (clr/rocclr)

**功能定位**：
- ROCm Common Language Runtime - ROCm的通用语言运行时
- 为HIP和OpenCL提供统一的设备抽象层
- 管理设备资源、内存、命令队列和内核执行

**主要模块**：

- `device/rocm/`: ROCm后端实现
  - `rocdevice.cpp`: GPU设备抽象和初始化
  - `rocmemory.cpp`: 内存对象管理
  - `rocvirtual.cpp`: 虚拟内存管理
  - `rockernel.cpp`: 内核对象管理
  - `rocrctx.cpp`: HSA Runtime动态加载和包装
  - `rocblit.cpp`: Blit操作（内存拷贝、填充）

- `device/`: 通用设备抽象
  - `device.cpp`: 设备基类
  - 内存管理、命令队列、上下文管理

- `platform/`: 平台层
  - 运行时初始化
  - 程序对象管理

**关键设计**：
```cpp
// ROCclr通过动态加载HSA Runtime
namespace roc {
  class Hsa {
    static RocrEntryPoints cep_;  // HSA函数入口点
    static bool LoadLib();        // 动态加载libhsa-runtime64.so
    // 包装HSA API调用
    static hsa_status_t hsa_init() { return cep_.hsa_init_fn(); }
    static hsa_status_t hsa_queue_create(...);
    static hsa_status_t hsa_memory_allocate(...);
    // ... 更多HSA API包装
  };
}
```

### 3. HSA Runtime (rocr-runtime/runtime)

**功能定位**：
- 实现HSA (Heterogeneous System Architecture) 标准
- 提供细粒度的GPU访问和控制
- 管理HSA代理(Agents)、队列(Queues)、信号(Signals)和内存

**主要模块**：

- `hsa-runtime/core/runtime/`: 核心运行时
  - `runtime.cpp`: 运行时初始化和系统管理
  - `agent.cpp`: HSA代理管理（GPU、CPU节点）
  - `queue.cpp`: AQL队列管理
  - `signal.cpp`: 信号同步机制
  - `memory_region.cpp`: 内存域管理
  - `amd_topology.cpp`: 拓扑发现

- `hsa-runtime/core/driver/kfd/`: KFD驱动接口
  - `amd_kfd_driver.cpp`: KFD驱动适配器
  - 通过libhsakmt与内核通信

- `hsa-runtime/inc/`: 公共头文件
  - `hsa.h`: HSA核心API
  - `hsa_ext_amd.h`: AMD扩展API
  - `amd_hsa_queue.h`: AQL队列定义

**核心API**：
```c
// HSA初始化和设备枚举
hsa_status_t hsa_init(void);
hsa_status_t hsa_iterate_agents(hsa_status_t (*callback)(...), void* data);

// 队列操作
hsa_status_t hsa_queue_create(hsa_agent_t agent, uint32_t size, 
                               hsa_queue_type_t type, ...);

// 内存操作
hsa_status_t hsa_memory_allocate(hsa_region_t region, size_t size, void** ptr);
hsa_status_t hsa_memory_copy(void* dst, const void* src, size_t size);

// 信号同步
hsa_status_t hsa_signal_create(hsa_signal_value_t initial_value, ...);
hsa_signal_value_t hsa_signal_wait_acquire(...);
```

**AQL (Architected Queuing Language)**：
- 用于GPU命令提交的标准化包格式
- 支持内核调度、内存操作、同步

### 4. libhsakmt / ROCt (rocr-runtime/libhsakmt)

**功能定位**：
- Thunk Library - "垫片"库
- 用户态和KFD内核驱动之间的接口层
- 封装所有的ioctl系统调用

**主要模块**：

- `src/`: 实现文件
  - `openclose.c`: 打开/关闭KFD设备 (/dev/kfd)
  - `topology.c`: 拓扑发现和系统属性查询
  - `memory.c`: 内存分配和映射
  - `queues.c`: 队列创建和管理
  - `events.c`: 事件创建和等待
  - `debug.c`: 调试支持

- `include/hsakmt/`: 头文件
  - `hsakmt.h`: Thunk API定义
  - `linux/kfd_ioctl.h`: KFD ioctl命令定义

**核心功能**：
```c
// 打开KFD设备
HSAKMT_STATUS hsaKmtOpenKFD(void);  // 打开 /dev/kfd

// 拓扑发现
HSAKMT_STATUS hsaKmtAcquireSystemProperties(HsaSystemProperties* SystemProperties);
HSAKMT_STATUS hsaKmtGetNodeProperties(HSAuint32 NodeId, HsaNodeProperties* NodeProperties);

// 内存管理
HSAKMT_STATUS hsaKmtAllocMemory(HSAuint32 NodeId, HSAuint64 Size, 
                                 HsaMemFlags MemFlags, void** MemoryAddress);
HSAKMT_STATUS hsaKmtMapMemoryToGPU(void* MemoryAddress, HSAuint64 Size, 
                                    HSAuint64* AlternateVAGPU);

// 队列操作
HSAKMT_STATUS hsaKmtCreateQueue(HSAuint32 NodeId, HSA_QUEUE_TYPE Type, ...);
```

### 5. KFD (kfd/amdkfd) - Kernel Fusion Driver

**功能定位**：
- Linux内核驱动模块
- 管理AMD GPU的计算功能
- 处理进程管理、内存管理、队列调度、中断处理

**主要模块**：

- `kfd_chardev.c`: 字符设备接口，ioctl处理
- `kfd_device.c`: 设备初始化和管理
- `kfd_process.c`: 进程上下文管理
- `kfd_topology.c`: 拓扑信息管理
- `kfd_queue.c`: 队列管理
- `kfd_device_queue_manager*.c`: 设备队列管理器（不同GPU架构）
- `kfd_memory.c`: 内存管理
- `kfd_doorbell.c`: Doorbell机制（用户态通知GPU）
- `kfd_events.c`: 事件和信号
- `kfd_interrupt.c`: 中断处理
- `kfd_svm.c`: Shared Virtual Memory支持
- `kfd_debug.c`: 调试支持

**ioctl命令** (定义在 `kfd_ioctl.h`):
```c
// 主要的ioctl命令
#define AMDKFD_IOC_GET_VERSION         // 获取驱动版本
#define AMDKFD_IOC_CREATE_QUEUE        // 创建计算队列
#define AMDKFD_IOC_DESTROY_QUEUE       // 销毁队列
#define AMDKFD_IOC_SET_MEMORY_POLICY   // 设置内存策略
#define AMDKFD_IOC_ALLOC_MEMORY_OF_GPU // 分配GPU内存
#define AMDKFD_IOC_FREE_MEMORY_OF_GPU  // 释放GPU内存
#define AMDKFD_IOC_MAP_MEMORY_TO_GPU   // 映射内存到GPU
#define AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU // 从GPU解映射内存
#define AMDKFD_IOC_CREATE_EVENT        // 创建事件
#define AMDKFD_IOC_WAIT_EVENTS         // 等待事件
#define AMDKFD_IOC_SVM                 // SVM操作
#define AMDKFD_IOC_DBG_TRAP            // 调试陷阱
// ... 还有约40+个ioctl命令
```

**队列类型**：
- `KFD_IOC_QUEUE_TYPE_COMPUTE`: 计算队列
- `KFD_IOC_QUEUE_TYPE_COMPUTE_AQL`: AQL格式计算队列
- `KFD_IOC_QUEUE_TYPE_SDMA`: DMA队列
- `KFD_IOC_QUEUE_TYPE_SDMA_XGMI`: XGMI DMA队列

---

## 调用关系与数据流

### 典型的内核启动流程

```
1. 应用调用 HIP API
   ↓
   hipLaunchKernel(kernel, grid, block, args, stream)
   
2. HIP Runtime (hipamd)
   ↓
   - 验证参数
   - 获取kernel元数据
   - 准备kernel参数
   - 调用ROCclr接口
   
3. ROCclr
   ↓
   - 创建amd::Command对象
   - 提交到CommandQueue
   - 调用roc::VirtualGPU::submitKernel()
   - 构建AQL包 (Architected Queuing Language packet)
   
4. HSA Runtime
   ↓
   - 调用 hsa_queue_add_write_index()
   - 填充AQL包到队列环形缓冲区
   - AQL包结构：
     * header: 包类型、控制标志
     * setup: 工作组大小、网格大小
     * kernel_object: 内核代码地址
     * kernarg_address: 内核参数地址
     * completion_signal: 完成信号
   - 写入Doorbell寄存器通知GPU
   
5. GPU硬件
   ↓
   - 从队列读取AQL包
   - 调度计算单元执行kernel
   - 完成后更新completion_signal
   
6. 同步返回
   ↓
   - HSA Runtime监控signal
   - 通知ROCclr命令完成
   - HIP Runtime返回给应用
```

### 内存分配流程

```
1. 应用调用
   ↓
   hipMalloc(&ptr, size)
   
2. HIP Runtime
   ↓
   ihipMalloc(ptr, size, flags)
   
3. ROCclr
   ↓
   - amd::Device::createMemory()
   - roc::Memory::create()
   - 调用 HSA Runtime
   
4. HSA Runtime
   ↓
   - hsa_amd_memory_pool_allocate()
   - 选择合适的内存池 (VRAM/GTT/System)
   - 调用 libhsakmt
   
5. libhsakmt
   ↓
   - hsaKmtAllocMemory()
   - 封装参数，调用ioctl
   - ioctl(fd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &args)
   
6. KFD驱动
   ↓
   - kfd_ioctl_alloc_memory_of_gpu()
   - 调用 amdgpu_amdkfd_gpuvm_alloc_memory_of_gpu()
   - 分配GPU页表
   - 映射到进程地址空间
   - 返回GPU虚拟地址
   
7. 返回路径
   ↓
   KFD → libhsakmt → HSA Runtime → ROCclr → HIP → 应用
```

### 队列创建流程

```
1. HSA Runtime (首次使用或显式创建)
   ↓
   hsa_queue_create(agent, size, type, callback, ...)
   
2. HSA Runtime内部
   ↓
   - 分配队列环形缓冲区 (ring buffer)
   - 分配read/write索引存储
   - 调用 libhsakmt
   
3. libhsakmt
   ↓
   - hsaKmtCreateQueue()
   - ioctl(fd, AMDKFD_IOC_CREATE_QUEUE, &args)
   
4. KFD驱动
   ↓
   - kfd_ioctl_create_queue()
   - 创建 kfd_queue 对象
   - 分配 MQD (Memory Queue Descriptor)
   - 配置硬件队列寄存器
   - 分配 Doorbell 页
   - 将队列添加到设备队列管理器
   
5. 返回
   ↓
   - 返回队列ID和Doorbell地址
   - HSA Runtime保存队列句柄
```

---

## 关键接口定义

### 1. HIP → ROCclr 接口

HIP通过ROCclr的C++对象接口操作：

```cpp
// 命令提交
namespace amd {
  class Command;      // 命令基类
  class KernelCommand;  // 内核命令
  class CopyCommand;    // 拷贝命令
  
  class HostQueue {
    bool append(Command& command);  // 提交命令
  };
  
  class Device {
    Memory* createMemory(...);      // 创建内存对象
    Program* createProgram(...);    // 创建程序对象
  };
}
```

### 2. ROCclr → HSA Runtime 接口

ROCclr通过动态加载的HSA C API调用：

```cpp
// clr/rocclr/device/rocm/rocrctx.cpp
namespace roc {
  class Hsa {
    // 动态加载的函数指针
    static decltype(::hsa_init)* hsa_init_fn;
    static decltype(::hsa_queue_create)* hsa_queue_create_fn;
    static decltype(::hsa_memory_allocate)* hsa_memory_allocate_fn;
    // ... 更多
    
    // 包装函数
    static hsa_status_t hsa_init() {
      return cep_.hsa_init_fn();
    }
  };
}
```

### 3. HSA Runtime → libhsakmt 接口

HSA Runtime直接链接libhsakmt，调用其C接口：

```c
// HSA Runtime内部调用示例
HSAKMT_STATUS status = hsaKmtAllocMemory(node_id, size, flags, &addr);
HSAKMT_STATUS status = hsaKmtCreateQueue(node_id, type, ...);
HSAKMT_STATUS status = hsaKmtMapMemoryToGPU(addr, size, &gpu_addr);
```

### 4. libhsakmt → KFD 接口

通过Linux ioctl系统调用：

```c
// libhsakmt/src/memory.c 示例
int fd = kfd_fd;  // /dev/kfd的文件描述符
struct kfd_ioctl_alloc_memory_of_gpu_args args = {...};
int ret = ioctl(fd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &args);
```

### 5. KFD ioctl处理

```c
// kfd/amdkfd/kfd_chardev.c
static const struct amdkfd_ioctl_desc amdkfd_ioctls[] = {
  AMDKFD_IOCTL_DEF(AMDKFD_IOC_CREATE_QUEUE, kfd_ioctl_create_queue, 0),
  AMDKFD_IOCTL_DEF(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, kfd_ioctl_alloc_memory_of_gpu, 0),
  AMDKFD_IOCTL_DEF(AMDKFD_IOC_MAP_MEMORY_TO_GPU, kfd_ioctl_map_memory_to_gpu, 0),
  // ... 40+个ioctl命令
};

static long kfd_ioctl(struct file *filep, unsigned int cmd, unsigned long arg) {
  // 验证命令
  // 拷贝用户态参数
  // 调用对应的处理函数
  // 返回结果
}
```

---

## 内存管理

### 内存类型

AMD ROCm支持多种内存类型：

1. **VRAM (Device Local Memory)**
   - GPU本地显存
   - 最快的GPU访问速度
   - 分配标志: `HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED`

2. **GTT (Graphics Translation Table)**
   - 系统内存，可被GPU访问
   - CPU/GPU都可访问
   - 分配标志: `HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED`

3. **System Memory**
   - 普通系统RAM
   - 需要固定(pin)后才能被GPU访问

4. **SVM (Shared Virtual Memory)**
   - CPU和GPU共享同一虚拟地址空间
   - 透明迁移和页错误处理

### 内存管理层次

```
┌─────────────────────────────────────────┐
│  HIP API (hipMalloc, hipMemcpy, etc.)  │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│     ROCclr Memory Object (amd::Memory)  │
│  - 内存属性管理                          │
│  - 多设备映射                            │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│   HSA Runtime Memory Management         │
│  - Memory Pools (VRAM/GTT/System)       │
│  - Memory Regions                       │
│  - hsa_amd_memory_pool_allocate()       │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│   libhsakmt (hsaKmtAllocMemory)         │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│   KFD Driver                            │
│  - kfd_process_device_apertures         │
│  - amdgpu_amdkfd_gpuvm_*                │
│  - GPU页表管理                          │
└─────────────────────────────────────────┘
```

### SVM支持

KFD提供完整的SVM支持：

```c
// KFD SVM ioctl
#define AMDKFD_IOC_SVM  // SVM操作

// SVM操作类型
enum svm_op {
  KFD_SVM_OP_REGISTER,      // 注册SVM范围
  KFD_SVM_OP_UNREGISTER,    // 注销SVM范围
  KFD_SVM_OP_SET_ATTR,      // 设置属性
  KFD_SVM_OP_GET_ATTR,      // 获取属性
};

// SVM页面迁移
- kfd_svm.c 实现页错误处理
- 自动在CPU/GPU之间迁移页面
- 支持预取和驻留控制
```

---

## 队列与任务调度

### 队列类型

1. **User-mode Queues (用户态队列)**
   - 应用直接写入AQL包
   - 低延迟提交
   - 硬件直接读取

2. **Compute Queues**
   - 执行计算kernel
   - 支持AQL格式

3. **SDMA Queues**
   - 专用DMA引擎
   - 异步内存拷贝
   - 不占用计算资源

### AQL包结构

```c
// HSA AQL Dispatch Packet
typedef struct hsa_kernel_dispatch_packet_s {
  uint16_t header;              // 包类型和控制位
  uint16_t setup;               // 维度、屏障等
  uint16_t workgroup_size_x;    // 工作组大小
  uint16_t workgroup_size_y;
  uint16_t workgroup_size_z;
  uint16_t reserved0;
  uint32_t grid_size_x;         // 网格大小
  uint32_t grid_size_y;
  uint32_t grid_size_z;
  uint32_t private_segment_size;
  uint32_t group_segment_size;
  uint64_t kernel_object;       // 内核代码地址
  uint64_t kernarg_address;     // 内核参数地址
  uint64_t reserved2;
  hsa_signal_t completion_signal;  // 完成信号
} hsa_kernel_dispatch_packet_t;
```

### 队列提交流程

```
1. 应用准备AQL包
   ↓
2. HSA Runtime:
   - write_index = atomic_add(queue->write_index, 1)
   - packet_ptr = &queue->base[write_index % queue->size]
   - 填充AQL包字段
   - 原子写入header激活包
   
3. 写入Doorbell寄存器:
   - doorbell_ptr = queue->doorbell
   - *doorbell_ptr = write_index
   
4. GPU硬件:
   - Doorbell触发GPU
   - Command Processor读取AQL包
   - 解析包并调度
   - 启动Compute Units
   
5. 完成:
   - GPU写入completion_signal
   - 可选触发中断
   - 应用或runtime轮询/等待signal
```

### Doorbell机制

Doorbell是用户态通知GPU的高效机制：

```
┌──────────────┐
│  User Space  │
│  (HSA RT)    │
└──────────────┘
       │ 写入doorbell地址
       ↓
┌──────────────┐
│  Doorbell    │  映射到用户空间的MMIO区域
│  Page        │  每个队列有独立的doorbell
└──────────────┘
       │ 硬件监控
       ↓
┌──────────────┐
│  GPU CP      │  Command Processor
│ (Hardware)   │  检测到doorbell写入
└──────────────┘
       │
       ↓ 从队列读取AQL包
```

KFD负责分配和映射doorbell页面到用户空间。

---

## 总结

### 软件栈分层职责

| 层次 | 组件 | 主要职责 |
|------|------|----------|
| **应用层** | 用户代码 | 业务逻辑、算法实现 |
| **编程模型层** | HIP / OpenCL | 统一的编程接口、可移植性 |
| **运行时层** | ROCclr | 设备抽象、资源管理、命令调度 |
| **HSA层** | HSA Runtime | HSA标准实现、底层资源管理 |
| **Thunk层** | libhsakmt | 用户态/内核态接口封装 |
| **内核层** | KFD | 设备驱动、硬件访问、进程隔离 |
| **硬件层** | AMD GPU | 实际执行计算 |

### 关键设计特点

1. **分层清晰**：每层职责明确，便于维护和扩展

2. **标准化**：遵循HSA标准，保证互操作性

3. **高性能**：
   - 用户态队列，低延迟提交
   - Doorbell机制，高效通知
   - AQL包，标准化调度接口
   - DMA队列，异步传输

4. **灵活性**：
   - 支持多种编程模型（HIP、OpenCL）
   - 多种内存类型
   - SVM支持

5. **强隔离**：
   - 每个进程独立的GPU地址空间
   - KFD管理资源隔离和调度

### 调试与监控

- `kfd_debugfs.c`: debugfs接口，暴露内核状态
- `kfd_debug.c`: 调试陷阱支持
- `kfd_smi_events.c`: 系统管理接口事件
- HSA Runtime提供API trace支持
- ROCm提供rocprof等性能分析工具

---

## 参考资料

本文档基于以下代码库分析：
- `clr/hipamd`: HIP AMD实现
- `clr/rocclr`: ROCm通用运行时
- `rocr-runtime/runtime`: HSA Runtime
- `rocr-runtime/libhsakmt`: ROCt Thunk库
- `kfd/amdkfd`: KFD内核驱动

相关标准和文档：
- HSA Foundation: https://hsafoundation.com/
- ROCm Documentation: https://rocm.docs.amd.com/
- HIP Programming Guide
- HSA Runtime Programmer's Reference Manual

