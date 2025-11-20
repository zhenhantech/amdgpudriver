# HIP与CLR架构关系详解

## 核心概念理解

### HIP vs CLR 的本质区别

```
┌─────────────────────────────────────────────────────────────┐
│                    hip/ 文件夹                               │
│              (HIP API 规范 - Interface)                      │
│                                                              │
│  • 定义HIP的公共API接口                                       │
│  • 纯头文件（.h），只有声明，没有实现                          │
│  • 平台无关的API规范                                          │
│  • 编译时依赖                                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ 应用程序 #include <hip/hip_runtime.h>
                            │ 编译时引用这些头文件
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  clr/ 文件夹                                 │
│            (CLR - Compute Language Runtime)                 │
│              HIP的实际运行时实现库                            │
│                                                              │
│  ┌───────────────────────────────────────────────┐          │
│  │  clr/hipamd/ (HIP AMD Implementation)         │          │
│  │  • 实现HIP API的所有函数                       │          │
│  │  • 编译成 libamdhip64.so 动态库                │          │
│  │  • 运行时加载                                  │          │
│  └───────────────────────────────────────────────┘          │
│                          │                                   │
│                          ▼ 依赖                              │
│  ┌───────────────────────────────────────────────┐          │
│  │  clr/rocclr/ (ROCm Common Language Runtime)   │          │
│  │  • 设备抽象层                                  │          │
│  │  • OpenCL和HIP的共同基础                       │          │
│  │  • 内存管理、命令队列、设备管理                 │          │
│  └───────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## 详细架构分析

### 1. hip/ 文件夹 - API 规范层

**职责**：定义HIP编程模型的接口规范

**主要内容**：

```
hip/
├── include/hip/          # HIP公共头文件
│   ├── hip_runtime_api.h       # HIP Runtime API声明（C接口）
│   ├── hip_runtime.h           # HIP Runtime总头文件
│   ├── hip_vector_types.h      # 向量类型定义
│   ├── hip_complex.h           # 复数类型
│   ├── math_functions.h        # 数学函数声明
│   ├── device_functions.h      # 设备函数
│   ├── hip_cooperative_groups.h # 协作组
│   └── amd_detail/             # AMD平台相关（转接）
│       └── amd_hip_runtime.h   # 转接到CLR实现
│
├── bin/                  # 工具
│   └── hipcc             # HIP编译器包装器
│
├── cmake/                # CMake配置
│   └── FindHIP.cmake     # CMake查找脚本
│
└── docs/                 # 文档
    └── reference/        # API参考文档
```

**关键设计**：

```cpp
// hip/include/hip/hip_runtime.h
#ifndef HIP_INCLUDE_HIP_HIP_RUNTIME_H
#define HIP_INCLUDE_HIP_HIP_RUNTIME_H

// 根据平台选择不同的实现
#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
    // AMD平台：包含AMD实现的头文件
    #include <hip/amd_detail/amd_hip_runtime.h>
#elif !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
    // NVIDIA平台：包含NVIDIA实现的头文件
    #include <hip/nvidia_detail/nvidia_hip_runtime.h>
#else
    #error ("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif

#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>

#endif
```

**特点**：
1. ✅ **只有接口声明**：函数原型、类型定义、宏定义
2. ✅ **平台无关**：通过条件编译支持AMD和NVIDIA
3. ✅ **编译时依赖**：应用编译时需要这些头文件
4. ✅ **无实现代码**：所有实现都在CLR中

### 2. clr/ 文件夹 - 实现层

**职责**：实现HIP和OpenCL的运行时

**架构组成**：

```
clr/
├── hipamd/               # HIP AMD平台实现
│   ├── src/              # 实现文件（.cpp）
│   │   ├── hip_platform.cpp      # 平台初始化
│   │   ├── hip_memory.cpp        # 内存管理实现
│   │   ├── hip_module.cpp        # 模块加载和内核启动
│   │   ├── hip_stream.cpp        # 流管理
│   │   ├── hip_event.cpp         # 事件同步
│   │   ├── hip_device.cpp        # 设备管理
│   │   └── hip_graph*.cpp        # 计算图实现
│   │
│   ├── include/hip/amd_detail/   # AMD实现的内部头文件
│   │   ├── amd_hip_runtime.h     # AMD运行时接口
│   │   └── hip_internal.hpp      # 内部实现
│   │
│   └── CMakeLists.txt            # 构建配置
│       # 生成 libamdhip64.so
│
├── rocclr/               # ROCm Common Language Runtime
│   ├── device/           # 设备抽象层
│   │   ├── device.cpp            # 设备基类
│   │   └── rocm/                 # ROCm后端
│   │       ├── rocdevice.cpp     # ROCm设备实现
│   │       ├── rocmemory.cpp     # 内存对象
│   │       ├── rockernel.cpp     # 内核对象
│   │       ├── rocrctx.cpp       # HSA Runtime接口
│   │       └── rocvirtual.cpp    # 虚拟GPU（命令提交）
│   │
│   ├── platform/         # 平台层
│   │   ├── command.cpp           # 命令对象
│   │   ├── context.cpp           # 上下文管理
│   │   └── program.cpp           # 程序对象
│   │
│   └── CMakeLists.txt            # 构建rocclr静态库
│
└── opencl/               # OpenCL实现（共享rocclr）
    └── amdocl/
```

### 3. 编译和链接关系

#### 编译时（应用编译）

```bash
# 应用代码
# myapp.cpp
#include <hip/hip_runtime.h>  // 引用 hip/ 文件夹的头文件

int main() {
    hipMalloc(&ptr, size);     // 只是声明，此时不需要实现
    hipLaunchKernel(...);
}

# 编译命令
hipcc myapp.cpp -o myapp
# 或者
clang++ myapp.cpp \
    -I/opt/rocm/include \           # 包含 hip/ 头文件
    -L/opt/rocm/lib \
    -lamdhip64 \                    # 链接 clr/hipamd 编译的库
    -o myapp
```

#### 运行时（应用执行）

```
应用启动
    ↓
加载 libamdhip64.so (由clr/hipamd编译)
    ↓
libamdhip64.so 内部依赖 rocclr (静态链接或动态链接)
    ↓
rocclr 加载 libhsa-runtime64.so
    ↓
连接到KFD驱动和GPU硬件
```

### 4. 代码流转示例

#### 示例：hipMalloc 的完整路径

**Step 1: 应用代码（使用hip/头文件）**

```cpp
// myapp.cpp
#include <hip/hip_runtime.h>  // hip/ 文件夹

int main() {
    void* ptr;
    hipError_t err = hipMalloc(&ptr, 1024);  // 调用API
    return 0;
}
```

**Step 2: hip/ 头文件（接口声明）**

```cpp
// hip/include/hip/hip_runtime_api.h
// 只有声明，没有实现
hipError_t hipMalloc(void** ptr, size_t size);
```

**Step 3: clr/hipamd 实现（运行时库）**

```cpp
// clr/hipamd/src/hip_memory.cpp
// 实际实现
hipError_t hipMalloc(void** ptr, size_t sizeBytes) {
  HIP_INIT_API(hipMalloc, ptr, sizeBytes);
  
  // 调用内部实现
  hipError_t status = ihipMalloc(ptr, sizeBytes, 0);
  
  HIP_RETURN(status);
}

static hipError_t ihipMalloc(void** ptr, size_t sizeBytes, 
                              unsigned int flags) {
  // 获取当前设备
  hip::Device* device = hip::getCurrentDevice();
  
  // 调用rocclr层的内存分配
  amd::Memory* mem = device->createMemory(sizeBytes, flags);
  
  *ptr = mem->getDevicePointer();
  return hipSuccess;
}
```

**Step 4: clr/rocclr 设备抽象层**

```cpp
// clr/rocclr/device/rocm/rocmemory.cpp
amd::Memory* Device::createMemory(size_t size, unsigned int flags) {
  // 调用HSA Runtime
  hsa_status_t status = Hsa::memory_allocate(region, size, &ptr);
  
  return new roc::Memory(ptr, size);
}
```

### 5. 构建产物

```
编译后的文件结构:

/opt/rocm/
├── include/              # 来自 hip/ 文件夹
│   └── hip/
│       ├── hip_runtime.h
│       ├── hip_runtime_api.h
│       └── ...
│
└── lib/                  # 来自 clr/ 文件夹
    ├── libamdhip64.so          # clr/hipamd 编译生成
    │   └── (包含 rocclr 的代码)
    │
    ├── libhsa-runtime64.so     # rocr-runtime 编译生成
    │
    └── cmake/hip/
        └── hip-config.cmake    # CMake配置
```

## CLR的三层架构

```
┌─────────────────────────────────────────────────────────────┐
│                    CLR (Compute Language Runtime)           │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Language Frontend Layer (语言前端层)                  │  │
│  │                                                         │  │
│  │  ┌──────────────┐        ┌──────────────┐            │  │
│  │  │  clr/hipamd  │        │ clr/opencl   │            │  │
│  │  │  (HIP API)   │        │ (OpenCL API) │            │  │
│  │  └──────┬───────┘        └──────┬───────┘            │  │
│  │         │                       │                     │  │
│  │         └───────────┬───────────┘                     │  │
│  └─────────────────────┼─────────────────────────────────┘  │
│                        │                                     │
│  ┌────────────────────▼──────────────────────────────────┐  │
│  │  Common Runtime Layer (通用运行时层)                  │  │
│  │                clr/rocclr                              │  │
│  │                                                         │  │
│  │  • amd::Device - 设备抽象                              │  │
│  │  • amd::Memory - 内存对象                              │  │
│  │  • amd::Command - 命令对象                             │  │
│  │  • amd::HostQueue - 命令队列                           │  │
│  │  • amd::Context - 上下文管理                           │  │
│  │  • amd::Program - 程序对象                             │  │
│  └────────────────────┬────────────────────────────────┘  │
│                       │                                     │
│  ┌────────────────────▼──────────────────────────────────┐  │
│  │  Backend Layer (后端层)                               │  │
│  │         clr/rocclr/device/rocm/                        │  │
│  │                                                         │  │
│  │  • roc::Device - ROCm设备实现                          │  │
│  │  • roc::Memory - ROCm内存实现                          │  │
│  │  • roc::Kernel - ROCm内核实现                          │  │
│  │  • roc::VirtualGPU - AQL包提交                         │  │
│  │  • Hsa::* - HSA Runtime包装                            │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
            HSA Runtime (libhsa-runtime64.so)
```

## 为什么要这样设计？

### 1. 接口与实现分离

```
好处：
✅ 应用代码只依赖接口（hip/），不依赖实现细节
✅ 可以有多个实现（AMD、NVIDIA、模拟器等）
✅ 实现可以独立演进和优化
✅ 二进制兼容性：实现改变不影响已编译的应用
```

### 2. 代码复用

```
┌─────────────┐
│   HIP API   │─────┐
└─────────────┘     │
                    ├─→ clr/rocclr (共享底层实现)
┌─────────────┐     │
│ OpenCL API  │─────┘
└─────────────┘

优势：
✅ HIP和OpenCL共享设备管理、内存管理等核心代码
✅ 减少代码重复
✅ 一致的性能特性
```

### 3. 平台可移植性

```cpp
// 同一份应用代码
#include <hip/hip_runtime.h>

// AMD平台
#define __HIP_PLATFORM_AMD__
→ 链接 libamdhip64.so (clr/hipamd)
→ 使用 ROCm 后端

// NVIDIA平台
#define __HIP_PLATFORM_NVIDIA__
→ 链接 hip_nvidia_runtime.so
→ 使用 CUDA 后端
```

## 关键文件对应关系

### API声明 vs 实现

| hip/ (接口声明) | clr/hipamd/ (实现) | 说明 |
|----------------|-------------------|------|
| `hip/include/hip/hip_runtime_api.h` | `clr/hipamd/src/hip_*.cpp` | API函数实现 |
| 声明: `hipMalloc(...)` | `clr/hipamd/src/hip_memory.cpp::hipMalloc()` | 内存分配 |
| 声明: `hipLaunchKernel(...)` | `clr/hipamd/src/hip_module.cpp::hipLaunchKernel()` | 内核启动 |
| 声明: `hipMemcpy(...)` | `clr/hipamd/src/hip_memory.cpp::hipMemcpy()` | 内存拷贝 |
| 声明: `hipStreamCreate(...)` | `clr/hipamd/src/hip_stream.cpp::hipStreamCreate()` | 流创建 |

### 内部头文件引用

```cpp
// hip/ 公共头文件
hip/include/hip/hip_runtime.h
    ↓ #include
hip/include/hip/amd_detail/amd_hip_runtime.h  (转接到实现)
    ↓ (编译时，实际实现在clr中)

// clr/ 实现头文件
clr/hipamd/include/hip/amd_detail/amd_hip_runtime.h
    ↓ 
clr/hipamd/src/hip_internal.hpp  (内部实现)
    ↓
clr/rocclr/device/device.hpp     (设备抽象)
    ↓
clr/rocclr/device/rocm/rocdevice.hpp  (ROCm实现)
```

## 完整调用链（从应用到硬件）

```
┌──────────────────────────────────────────────────────────────┐
│ 1. 应用代码                                                   │
│    #include <hip/hip_runtime.h>  // 来自 hip/                │
│    hipMalloc(&ptr, size);                                    │
└────────────────────────┬─────────────────────────────────────┘
                         │ 编译时：引用 hip/ 头文件
                         │ 运行时：调用 libamdhip64.so
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. HIP Runtime (clr/hipamd)                                  │
│    hip_memory.cpp::hipMalloc()                               │
│    → ihipMalloc()                                            │
│    → hip::Device::createMemory()                             │
└────────────────────────┬─────────────────────────────────────┘
                         │ 调用 rocclr 层
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. ROCclr (clr/rocclr)                                       │
│    amd::Device::createMemory()                               │
│    → roc::Memory::create()                                   │
│    → roc::Device::allocMemory()                              │
└────────────────────────┬─────────────────────────────────────┘
                         │ 调用 HSA Runtime
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. HSA Runtime                                               │
│    Hsa::memory_allocate()                                    │
│    → hsa_amd_memory_pool_allocate()                          │
└────────────────────────┬─────────────────────────────────────┘
                         │ 调用 libhsakmt
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. libhsakmt                                                 │
│    hsaKmtAllocMemory()                                       │
│    → ioctl(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU)                   │
└────────────────────────┬─────────────────────────────────────┘
                         │ 系统调用
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ 6. KFD驱动                                                    │
│    kfd_ioctl_alloc_memory_of_gpu()                           │
│    → GPU页表分配                                              │
└──────────────────────────────────────────────────────────────┘
```

## 实际开发场景

### 场景1: 应用开发者

```cpp
// 开发者视角
#include <hip/hip_runtime.h>  // 只需要 hip/ 头文件

int main() {
    void* ptr;
    hipMalloc(&ptr, 1024);      // 调用API
    hipFree(ptr);
    return 0;
}

// 编译
$ hipcc myapp.cpp -o myapp
// hipcc会自动找到：
// - 头文件：/opt/rocm/include/hip/
// - 库文件：/opt/rocm/lib/libamdhip64.so

// 运行
$ ./myapp
// 自动加载 libamdhip64.so (来自clr/hipamd)
```

### 场景2: HIP Runtime开发者

```cpp
// 在 clr/hipamd/src/hip_memory.cpp 中实现新功能

hipError_t hipMallocNew(void** ptr, size_t size, int hint) {
  HIP_INIT_API(hipMallocNew, ptr, size, hint);
  
  // 使用 rocclr 提供的设备抽象
  amd::Device* device = hip::getCurrentDevice();
  amd::Memory* mem = device->createMemory(size, hint);
  
  *ptr = mem->getDevicePointer();
  
  HIP_RETURN(hipSuccess);
}

// 同时在 hip/include/hip/hip_runtime_api.h 中添加声明
hipError_t hipMallocNew(void** ptr, size_t size, int hint);
```

### 场景3: ROCclr开发者

```cpp
// 在 clr/rocclr/device/rocm/rocmemory.cpp 优化内存分配

bool Memory::allocate(size_t size) {
  // 优化内存池选择策略
  hsa_amd_memory_pool_t pool = selectOptimalPool(size);
  
  // 调用HSA Runtime
  hsa_status_t status = Hsa::memory_allocate(pool, size, &ptr_);
  
  return (status == HSA_STATUS_SUCCESS);
}

// 这个优化会自动惠及HIP和OpenCL
```

## 总结

### HIP文件夹 (hip/)
- **角色**：API规范和接口定义
- **内容**：头文件（.h）
- **依赖时机**：编译时
- **对应物**：类似于C标准库的头文件（stdio.h），只有声明

### CLR文件夹 (clr/)
- **角色**：运行时实现
- **内容**：
  - `clr/hipamd/`: HIP API的完整实现（编译成libamdhip64.so）
  - `clr/rocclr/`: 通用设备抽象层（HIP和OpenCL共享）
  - `clr/opencl/`: OpenCL API实现
- **依赖时机**：运行时
- **对应物**：类似于C标准库的实现（libc.so），有实际代码

### 关系总结

```
应用代码
    │ 编译时
    ├─ 引用: hip/ 头文件 (接口)
    │
    │ 运行时
    └─ 链接: clr/hipamd 库 (实现)
           └─ 依赖: clr/rocclr (设备抽象)
                  └─ 依赖: HSA Runtime
                         └─ 依赖: KFD驱动
```

这种"接口与实现分离"的设计是现代软件工程的最佳实践，使得HIP具有良好的可维护性、可扩展性和平台可移植性。

---

## 快速对比表

| 对比项 | hip/ 文件夹 | clr/ 文件夹 |
|--------|-------------|-------------|
| **本质** | API规范/接口 | 运行时实现 |
| **内容** | 头文件（.h） | 源码和库（.cpp, .so） |
| **类比** | 建筑设计图纸 | 实际建筑物 |
| **代码量** | 很少（只有声明） | 很大（完整实现） |
| **编译产物** | 无（只在编译时使用） | libamdhip64.so |
| **依赖阶段** | 编译时（#include） | 运行时（动态链接） |
| **修改影响** | 需要重新编译应用 | 不需要重新编译应用 |
| **平台支持** | AMD + NVIDIA（条件编译） | 只有AMD实现在CLR中 |
| **开发者** | API设计者 | 运行时工程师 |

## 类比理解

### 类比1: 餐厅菜单 vs 后厨

```
hip/ (菜单)                      clr/ (后厨)
├── 宫保鸡丁                     ├── 准备食材
├── 麻婆豆腐                     ├── 烹饪流程
└── 鱼香肉丝                     ├── 调味配方
                                └── 装盘技巧

顾客（应用）：只看菜单点菜
厨师（运行时）：根据菜单做出实际菜品
```

### 类比2: 汽车仪表盘 vs 引擎

```
hip/ (仪表盘)                    clr/ (引擎系统)
├── 油门踏板接口                 ├── 燃油喷射
├── 刹车踏板接口                 ├── 点火系统
├── 方向盘接口                   ├── 动力传输
└── 仪表显示                     └── 电子控制

驾驶员（应用）：通过仪表盘操作
系统（运行时）：实际驱动汽车
```

### 类比3: 遥控器 vs 电视机

```
hip/ (遥控器按键)                clr/ (电视内部电路)
├── 开关按钮                     ├── 电源管理
├── 音量按钮                     ├── 音频处理
├── 频道按钮                     ├── 视频解码
└── 菜单按钮                     └── 信号接收

用户（应用）：按遥控器按钮
电视（运行时）：执行实际操作
```

## 技术术语对应

| 通用术语 | hip/ | clr/ |
|----------|------|------|
| **软件工程** | Interface（接口） | Implementation（实现） |
| **面向对象** | Abstract Class（抽象类） | Concrete Class（具体类） |
| **C语言** | Header File（.h） | Source File（.c）+ Library（.so） |
| **操作系统** | System Call Interface | Kernel Implementation |
| **网络协议** | Protocol Specification | Protocol Stack |
| **数据库** | SQL Standard | Database Engine |

## 文件系统对应关系示意图

```
/opt/rocm/                          # 安装目录
│
├── include/                        # 来自 hip/ 
│   └── hip/
│       ├── hip_runtime.h           ← hip/include/hip/hip_runtime.h
│       ├── hip_runtime_api.h       ← hip/include/hip/hip_runtime_api.h
│       └── amd_detail/             ← hip/include/hip/amd_detail/
│
├── lib/                            # 来自 clr/
│   ├── libamdhip64.so              ← clr/hipamd/ 编译生成
│   │   (包含 clr/rocclr 的代码)
│   │
│   └── cmake/hip/
│       └── hip-config.cmake        ← clr/hipamd/hip-config.cmake.in
│
├── bin/                            # 来自 hip/ 和 clr/
│   ├── hipcc                       ← hip/bin/hipcc (编译器包装器)
│   └── hipconfig                   ← clr/hipamd/bin/hipconfig
│
└── share/doc/hip/                  # 来自 hip/
    └── ...                         ← hip/docs/
```

## 调试和开发场景

### 场景1: 应用崩溃，如何定位？

```bash
# 应用崩溃在 hipMalloc
$ gdb ./myapp
(gdb) break hipMalloc
(gdb) run

# 断点会停在哪里？
Breakpoint 1, hipMalloc ()
    at clr/hipamd/src/hip_memory.cpp:758
                ^^^^
                实现在 clr/ 中！

# 调用栈
#0  hipMalloc (clr/hipamd/src/hip_memory.cpp:758)
#1  ihipMalloc (clr/hipamd/src/hip_memory.cpp:642)
#2  amd::Device::createMemory (clr/rocclr/device/device.cpp:1234)
#3  roc::Memory::create (clr/rocclr/device/rocm/rocmemory.cpp:567)
#4  Hsa::memory_allocate (clr/rocclr/device/rocm/rocrctx.cpp:89)
```

### 场景2: 想要理解API行为

```
问题：hipMalloc是如何选择内存类型的？

❌ 错误：查看 hip/include/hip/hip_runtime_api.h
   → 只有声明，没有逻辑

✅ 正确：查看 clr/hipamd/src/hip_memory.cpp
   → 有完整的实现逻辑和注释
```

### 场景3: 想要添加新功能

```
需求：添加新API hipMallocAdvanced()

步骤：
1️⃣ 在 hip/include/hip/hip_runtime_api.h 添加声明
   hipError_t hipMallocAdvanced(void** ptr, size_t size, int flags);

2️⃣ 在 clr/hipamd/src/hip_memory.cpp 添加实现
   hipError_t hipMallocAdvanced(void** ptr, size_t size, int flags) {
       // 实现代码
   }

3️⃣ 重新编译 clr/ 生成新的 libamdhip64.so

4️⃣ 用户重新编译应用（因为头文件改了）
```

## 最后的总结

### 简单记忆法

```
┌──────────────────────────────────────────┐
│  hip/ = "说明书"（告诉你有什么功能）      │
│  clr/ = "实物"（实际执行这些功能）        │
└──────────────────────────────────────────┘

应用开发：只需看"说明书"（hip/）
运行时开发：需要制造"实物"（clr/）
应用运行：实际使用"实物"（clr/的库）
```

### 核心要点

1. **hip/** = **接口规范层**
   - 纯头文件，定义API
   - 编译时依赖
   - 平台无关（AMD/NVIDIA通用）

2. **clr/hipamd/** = **HIP实现层** 
   - 实现所有HIP API
   - 编译成libamdhip64.so
   - AMD平台专用

3. **clr/rocclr/** = **设备抽象层**
   - HIP和OpenCL共享
   - 设备/内存/命令管理
   - 对接HSA Runtime

这就是为什么你会看到两个看似重复的"HIP"目录：一个是接口定义（hip/），另一个是实现代码（clr/hipamd/）。它们是同一个API的不同层面！

