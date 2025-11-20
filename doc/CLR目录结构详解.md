# CLR (Compute Language Runtimes) 目录结构详解

## 概述

**CLR** (Compute Language Runtimes) 是 AMD 的计算语言运行时项目，包含 **HIP** 和 **OpenCL** 两种编程模型的完整实现代码。

> **重要**：CLR 是实现层，与 `hip/` 目录（接口层）形成对比。

## 目录树总览

```
clr/
├── hipamd/             # HIP AMD 平台实现（最核心）
├── rocclr/             # ROCm Common Language Runtime（共享基础）
├── opencl/             # OpenCL 实现
├── CMakeLists.txt      # 主构建脚本
├── README.md           # 项目说明
├── CHANGELOG.md        # 变更日志
├── LICENSE.md          # 许可证
└── CONTRIBUTING.md     # 贡献指南
```

---

## 三大核心目录详解

### 1. 🎯 `hipamd/` - HIP AMD 平台实现

**定位**：HIP Runtime API 在 AMD 平台上的完整实现

**主要职责**：
- 实现所有 HIP API 函数（200+ 个函数）
- 提供 AMD GPU 特定的优化
- 编译成 `libamdhip64.so` 动态库

**目录结构**：
```
hipamd/
├── src/                     # HIP 实现源码（最核心）
│   ├── 内存管理：
│   │   ├── hip_memory.cpp           # hipMalloc, hipMemcpy 等
│   │   ├── hip_mempool.cpp          # 内存池管理
│   │   ├── hip_mempool_impl.cpp     # 内存池实现细节
│   │   └── hip_vm.cpp               # 虚拟内存管理
│   │
│   ├── 设备管理：
│   │   ├── hip_device.cpp           # hipGetDevice, hipSetDevice 等
│   │   ├── hip_platform.cpp         # 平台初始化
│   │   └── hip_peer.cpp             # P2P 设备访问
│   │
│   ├── 流和事件：
│   │   ├── hip_stream.cpp           # hipStreamCreate 等
│   │   ├── hip_stream_ops.cpp       # 流操作
│   │   ├── hip_event.cpp            # hipEventCreate 等
│   │   └── hip_event_ipc.cpp        # IPC 事件
│   │
│   ├── 内核启动和模块：
│   │   ├── hip_module.cpp           # hipLaunchKernel, hipModuleLoad 等
│   │   ├── hip_code_object.cpp      # 代码对象加载
│   │   ├── hip_fatbin.cpp           # Fat Binary 处理
│   │   └── hip_library.cpp          # 库管理
│   │
│   ├── 计算图：
│   │   ├── hip_graph.cpp            # hipGraphCreate 等
│   │   ├── hip_graph_internal.cpp   # 图内部实现
│   │   ├── hip_graph_capture.hpp    # 流捕获
│   │   └── hip_graph_helper.hpp     # 辅助函数
│   │
│   ├── 纹理和表面：
│   │   ├── hip_texture.cpp          # 纹理 API
│   │   └── hip_surface.cpp          # 表面 API
│   │
│   ├── OpenGL 互操作：
│   │   └── hip_gl.cpp               # OpenGL 互操作
│   │
│   ├── 性能分析和调试：
│   │   ├── hip_profile.cpp          # 性能分析
│   │   ├── hip_activity.cpp         # 活动跟踪
│   │   ├── hip_api_trace.cpp        # API 跟踪
│   │   └── hip_intercept.cpp        # API 拦截
│   │
│   ├── 其他核心功能：
│   │   ├── hip_context.cpp          # 上下文管理
│   │   ├── hip_error.cpp            # 错误处理
│   │   ├── hip_global.cpp           # 全局变量
│   │   ├── hip_runtime.cpp          # Runtime 初始化
│   │   ├── hip_hmm.cpp              # Heterogeneous Memory Management
│   │   └── hip_device_runtime.cpp   # 设备端 runtime
│   │
│   ├── 辅助模块：
│   │   ├── hip_comgr_helper.cpp     # Code Object Manager 辅助
│   │   ├── hip_conversions.hpp      # 类型转换
│   │   ├── hip_formatting.hpp       # 格式化工具
│   │   ├── hip_internal.hpp         # 内部定义
│   │   └── hip_table_interface*.cpp # 函数表接口
│   │
│   └── hiprtc/              # HIP RTC (Runtime Compilation)
│       ├── hiprtc.cpp               # RTC API 实现
│       └── hiprtcInternal.cpp       # RTC 内部实现
│
├── include/hip/             # AMD 平台头文件
│   └── amd_detail/          # AMD 实现细节头文件（51个）
│       ├── amd_hip_runtime.h        # AMD Runtime 接口
│       ├── amd_hip_atomic.h         # 原子操作
│       ├── amd_hip_cooperative_groups.h # 协作组
│       ├── amd_math_functions.h     # 数学函数
│       ├── amd_device_functions.h   # 设备函数
│       ├── device_library_decls.h   # 设备库声明
│       ├── grid_launch*.hpp         # Grid 启动宏/模板
│       ├── hip_prof_str.h           # 性能分析字符串
│       └── ...                      # 更多辅助头文件
│
├── bin/                     # 工具脚本
│   ├── roc-obj                      # ROC 对象工具
│   ├── roc-obj-extract              # 提取代码对象
│   └── roc-obj-ls                   # 列出代码对象
│
├── packaging/               # 打包配置
│   ├── hip-runtime-amd.*            # AMD Runtime 包
│   └── hip-devel.*                  # 开发包
│
├── CMakeLists.txt           # 构建配置
└── hip-config*.cmake.in     # CMake 配置模板
```

**关键实现文件说明**：

| 文件 | 代码行数 | 主要功能 |
|-----|---------|---------|
| `hip_memory.cpp` | ~2000 | 内存分配、拷贝、释放 |
| `hip_module.cpp` | ~1500 | 内核启动、模块加载 |
| `hip_stream.cpp` | ~1000 | 流管理、同步 |
| `hip_graph.cpp` | ~2500 | 计算图 API |
| `hip_graph_internal.cpp` | ~3000 | 计算图内部实现 |
| `hip_device.cpp` | ~800 | 设备管理 |
| `hip_event.cpp` | ~600 | 事件管理 |

**编译产物**：
```bash
# 编译后生成：
/opt/rocm/lib/
├── libamdhip64.so       # 主要的 HIP Runtime 库
├── libamdhip64.so.6     # 版本链接
└── libhiprtc.so         # HIP RTC 库
```

**调用示例**：
```cpp
// 应用代码：
#include <hip/hip_runtime.h>
hipMalloc(&ptr, size);

// 实际调用到：
// clr/hipamd/src/hip_memory.cpp::hipMalloc()
hipError_t hipMalloc(void** ptr, size_t sizeBytes) {
  HIP_INIT_API(hipMalloc, ptr, sizeBytes);
  hipError_t status = ihipMalloc(ptr, sizeBytes, 0);
  HIP_RETURN(status);
}

// 内部调用 ROCclr：
static hipError_t ihipMalloc(void** ptr, size_t size, unsigned int flags) {
  hip::Device* device = hip::getCurrentDevice();
  amd::Memory* mem = device->createMemory(size, flags); // ← 调用 ROCclr
  *ptr = mem->getDevicePointer();
  return hipSuccess;
}
```

---

### 2. 🏗️ `rocclr/` - ROCm Common Language Runtime

**定位**：HIP 和 OpenCL 共享的底层运行时基础设施

**主要职责**：
- 提供设备抽象层（Device Abstraction Layer）
- 统一的内存管理、命令队列、程序对象
- 对接 HSA Runtime
- 被 HIP 和 OpenCL 共同依赖

**目录结构**：
```
rocclr/
├── device/                  # 设备抽象层（最核心）
│   ├── 通用设备抽象：
│   │   ├── device.cpp               # 设备基类
│   │   ├── device.hpp
│   │   ├── devkernel.cpp            # 内核对象
│   │   ├── devprogram.cpp           # 程序对象
│   │   ├── blit.cpp                 # Blit 操作（拷贝、填充）
│   │   ├── comgrctx.cpp             # Code Object Manager 上下文
│   │   └── appprofile.cpp           # 应用性能配置
│   │
│   ├── rocm/                # ROCm 后端实现（AMD GPU）
│   │   ├── rocdevice.cpp            # ROCm 设备实现
│   │   ├── rocdevice.hpp
│   │   ├── rocmemory.cpp            # ROCm 内存管理
│   │   ├── rockernel.cpp            # ROCm 内核
│   │   ├── rocprogram.cpp           # ROCm 程序
│   │   ├── rocvirtual.cpp           # 虚拟 GPU（命令提交）
│   │   ├── rocrctx.cpp              # HSA Runtime 包装
│   │   ├── rocrctx.hpp              # HSA API 动态加载
│   │   ├── rocblit.cpp              # ROCm Blit 实现
│   │   ├── rocsettings.cpp          # ROCm 设置
│   │   ├── rocsignal.cpp            # 信号实现
│   │   ├── rocprintf.cpp            # printf 支持
│   │   ├── rocglinterop.cpp         # OpenGL 互操作
│   │   └── roccounters.cpp          # 性能计数器
│   │
│   └── pal/                 # PAL 后端（已废弃，但代码仍在）
│       └── ...                      # PAL 设备实现
│
├── platform/                # 平台层（OpenCL 语义）
│   ├── runtime.cpp                  # Runtime 初始化
│   ├── context.cpp                  # 上下文管理
│   ├── commandqueue.cpp             # 命令队列
│   ├── command.cpp                  # 命令对象（NDRange、Copy等）
│   ├── memory.cpp                   # 内存对象（Buffer、Image）
│   ├── kernel.cpp                   # 内核对象
│   ├── program.cpp                  # 程序对象
│   ├── agent.cpp                    # Agent（设备代理）
│   ├── ndrange.cpp                  # NDRange 执行
│   └── activity.cpp                 # 活动跟踪
│
├── compiler/                # 编译器接口
│   └── lib/
│       ├── backends/                # 编译器后端
│       ├── include/                 # 编译器接口
│       └── utils/                   # 工具
│
├── os/                      # 操作系统抽象
│   ├── os.cpp                       # OS 通用接口
│   ├── os_posix.cpp                 # Linux 实现
│   ├── os_win32.cpp                 # Windows 实现
│   └── alloc.cpp                    # 内存分配
│
├── utils/                   # 工具类
│   ├── flags.cpp                    # 环境变量标志
│   ├── debug.cpp                    # 调试工具
│   ├── concurrent.hpp               # 并发工具
│   └── util.hpp                     # 通用工具
│
├── thread/                  # 线程管理
│   ├── thread.cpp                   # 线程抽象
│   └── monitor.hpp                  # 监控器
│
├── elf/                     # ELF 处理
│   ├── elf.cpp                      # ELF 文件解析
│   └── elfio/                       # ELF I/O 库
│
├── include/                 # 公共头文件
│   ├── top.hpp                      # 顶层定义
│   └── vdi_common.hpp               # VDI 通用定义
│
└── cmake/                   # CMake 配置
    ├── ROCclr.cmake                 # ROCclr 配置
    ├── ROCclrHSA.cmake              # HSA 支持
    └── ...                          # 其他配置
```

**关键设计模式**：

```cpp
// 设备抽象层架构
namespace amd {
  // 通用设备基类
  class Device {
    virtual Memory* createMemory(...) = 0;
    virtual Kernel* createKernel(...) = 0;
    virtual Program* createProgram(...) = 0;
  };
  
  // ROCm 设备实现
  namespace roc {
    class Device : public amd::Device {
      Memory* createMemory(...) override;  // ROCm 特定实现
      // 内部调用 HSA Runtime
    };
  }
}
```

**核心功能**：

1. **设备管理** (`device/rocm/rocdevice.cpp`)
   - 初始化 HSA Runtime
   - 枚举 GPU Agents
   - 管理设备属性

2. **内存管理** (`device/rocm/rocmemory.cpp`)
   - 分配 GPU 内存（VRAM）
   - 系统内存固定（pinning）
   - SVM（Shared Virtual Memory）

3. **命令提交** (`device/rocm/rocvirtual.cpp`)
   - 构建 AQL 包
   - 提交到 HSA 队列
   - Doorbell 通知

4. **HSA 接口** (`device/rocm/rocrctx.cpp`)
   ```cpp
   // 动态加载 HSA Runtime
   class Hsa {
     static bool LoadLib() {
       cep_.handle = dlopen("libhsa-runtime64.so.1");
       GET_ROCR_SYMBOL(hsa_init);
       GET_ROCR_SYMBOL(hsa_queue_create);
       // ... 加载所有 HSA API
     }
     
     static hsa_status_t hsa_init() {
       return cep_.hsa_init_fn();
     }
   };
   ```

**编译产物**：
```
rocclr 编译成静态库，链接到 hipamd 和 opencl 中
```

---

### 3. 🌐 `opencl/` - OpenCL 实现

**定位**：OpenCL API 标准的完整实现

**主要职责**：
- 实现 OpenCL 1.2 / 2.0 / 2.1 / 2.2 标准
- 提供 ICD (Installable Client Driver) 支持
- 共享 ROCclr 作为底层实现

**目录结构**：
```
opencl/
├── amdocl/                  # AMD OpenCL 实现
│   ├── cl_platform_amd.cpp          # clGetPlatformInfo 等
│   ├── cl_device.cpp                # clGetDeviceInfo 等
│   ├── cl_context.cpp               # clCreateContext 等
│   ├── cl_command.cpp               # 命令队列操作
│   ├── cl_memobj.cpp                # clCreateBuffer 等
│   ├── cl_program.cpp               # clCreateProgram 等
│   ├── cl_kernel.cpp                # clCreateKernel 等
│   ├── cl_event.cpp                 # clCreateEvent 等
│   ├── cl_execute.cpp               # clEnqueueNDRangeKernel 等
│   ├── cl_svm.cpp                   # SVM API
│   ├── cl_pipe.cpp                  # Pipe API
│   ├── cl_gl.cpp                    # OpenGL 互操作
│   ├── cl_d3d9.cpp / cl_d3d10.cpp / cl_d3d11.cpp  # Direct3D 互操作
│   ├── cl_icd.cpp                   # ICD 支持
│   ├── cl_profile_amd.cpp           # AMD 性能分析扩展
│   ├── cl_thread_trace_amd.cpp      # 线程跟踪
│   ├── cl_p2p_amd.cpp               # P2P 扩展
│   └── CMakeLists.txt
│
├── khronos/                 # Khronos 标准文件
│   ├── headers/             # OpenCL 标准头文件
│   │   ├── opencl1.2/CL/
│   │   ├── opencl2.0/CL/
│   │   ├── opencl2.1/CL/
│   │   └── opencl2.2/CL/
│   │       ├── cl.h                 # OpenCL 核心 API
│   │       ├── cl_platform.h        # 平台定义
│   │       ├── cl_gl.h              # OpenGL 互操作
│   │       ├── cl_ext.h             # 扩展
│   │       └── ...
│   │
│   └── icd/                 # ICD Loader（可选）
│       └── loader/                  # ICD 加载器实现
│
├── tests/                   # OpenCL 测试套件
│   └── ocltst/              # 301 个测试
│       ├── module/
│       │   ├── runtime/             # Runtime 测试（89个）
│       │   ├── perf/                # 性能测试（143个）
│       │   ├── common/              # 通用测试
│       │   ├── gl/                  # OpenGL 测试
│       │   └── dx/                  # DirectX 测试
│       └── env/                     # 测试环境
│
├── tools/                   # OpenCL 工具
│   ├── clinfo/                      # clinfo 设备信息工具
│   │   └── clinfo.cpp
│   └── cltrace/                     # OpenCL API 跟踪工具
│       └── cltrace.cpp
│
├── config/                  # ICD 配置
│   ├── amdocl64.icd                 # ICD 注册文件
│   └── amdocl32.icd
│
├── packaging/               # 打包配置
│   ├── rocm-opencl.*                # OpenCL 运行时包
│   └── rocm-ocl-icd.*               # ICD 加载器包
│
├── CMakeLists.txt           # 构建配置
└── README.md                # 说明文档
```

**API 实现示例**：

```cpp
// opencl/amdocl/cl_memobj.cpp
CL_API_ENTRY cl_mem CL_API_CALL
clCreateBuffer(cl_context context,
               cl_mem_flags flags,
               size_t size,
               void* host_ptr,
               cl_int* errcode_ret) {
  // 1. 验证参数
  if (!is_valid(context)) {
    *errcode_ret = CL_INVALID_CONTEXT;
    return nullptr;
  }
  
  // 2. 调用 ROCclr 层
  amd::Context* amdContext = as_amd(context);
  amd::Memory* amdMemory = amdContext->createBuffer(size, flags, host_ptr);
  
  // 3. 返回 OpenCL 对象
  return as_cl(amdMemory);
}
```

**编译产物**：
```bash
/opt/rocm/lib/
├── libamdocl64.so          # AMD OpenCL 实现
└── libOpenCL.so            # OpenCL ICD Loader（可选）

/etc/OpenCL/vendors/
└── amdocl64.icd            # ICD 注册文件
```

**使用场景**：
- 运行 OpenCL 应用程序
- 与其他 OpenCL 实现共存（通过 ICD）
- 科学计算、图像处理等传统 OpenCL 应用

---

## 三大目录的关系

### 层次关系图

```
┌─────────────────────────────────────────────────────────┐
│                    应用层                                │
│         HIP 应用          │        OpenCL 应用           │
└──────────┬────────────────┴──────────────┬──────────────┘
           │                               │
           │ HIP API                       │ OpenCL API
           ▼                               ▼
┌──────────────────────┐        ┌──────────────────────┐
│   clr/hipamd/        │        │   clr/opencl/        │
│   HIP 实现           │        │   OpenCL 实现        │
│   (libamdhip64.so)   │        │   (libamdocl64.so)   │
└──────────┬───────────┘        └──────────┬───────────┘
           │                               │
           │ 共享依赖                       │
           └───────────────┬───────────────┘
                           ▼
           ┌────────────────────────────────┐
           │      clr/rocclr/               │
           │      通用运行时基础             │
           │      (静态库)                   │
           │                                │
           │  • Device 抽象                 │
           │  • Memory 管理                 │
           │  • Command 队列                │
           │  • HSA Runtime 接口            │
           └────────────────┬───────────────┘
                            │
                            ▼
           ┌────────────────────────────────┐
           │    HSA Runtime                 │
           │    (libhsa-runtime64.so)       │
           └────────────────────────────────┘
```

### 代码共享示例

```cpp
// HIP 和 OpenCL 都使用 ROCclr 的设备抽象

// HIP 路径：
hipMalloc()                          // clr/hipamd/src/hip_memory.cpp
  → hip::Device::createMemory()      // clr/hipamd/
    → amd::Memory::create()          // clr/rocclr/platform/memory.cpp
      → roc::Memory::create()        // clr/rocclr/device/rocm/rocmemory.cpp
        → Hsa::memory_allocate()     // clr/rocclr/device/rocm/rocrctx.cpp
          → hsa_memory_allocate()    // HSA Runtime

// OpenCL 路径：
clCreateBuffer()                     // clr/opencl/amdocl/cl_memobj.cpp
  → amd::Context::createBuffer()     // clr/rocclr/platform/context.cpp
    → amd::Memory::create()          // clr/rocclr/platform/memory.cpp
      → roc::Memory::create()        // clr/rocclr/device/rocm/rocmemory.cpp
        → Hsa::memory_allocate()     // 同一个实现！
```

---

## 编译流程和产物

### 编译命令

```bash
cd clr
mkdir build && cd build

# 配置（启用 HIP 和 OpenCL）
cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm \
      -DCLR_BUILD_HIP=ON \
      -DCLR_BUILD_OCL=ON \
      ..

# 编译
make -j$(nproc)

# 安装
sudo make install
```

### 编译产物

```
/opt/rocm/
├── lib/
│   ├── libamdhip64.so           ← hipamd 编译生成
│   │   (包含 rocclr 代码)
│   │
│   ├── libamdocl64.so           ← opencl 编译生成
│   │   (包含 rocclr 代码)
│   │
│   ├── libhiprtc.so             ← hipamd/hiprtc 编译生成
│   │
│   └── cmake/
│       ├── hip/                 ← hipamd CMake 配置
│       └── AMDDeviceLibs/
│
├── include/
│   ├── hip/
│   │   └── amd_detail/          ← hipamd/include/
│   │
│   └── CL/                      ← opencl/khronos/headers/
│       ├── cl.h
│       └── ...
│
├── bin/
│   ├── clinfo                   ← opencl/tools/clinfo/
│   ├── roc-obj                  ← hipamd/bin/
│   └── roc-obj-ls
│
└── etc/OpenCL/vendors/
    └── amdocl64.icd             ← opencl/config/
```

---

## 代码量统计

```
┌──────────────┬──────────┬─────────┬─────────────┐
│   目录        │  文件数   │ 代码行数 │   主要语言  │
├──────────────┼──────────┼─────────┼─────────────┤
│ hipamd/src/  │   ~70    │ ~40K    │   C++11     │
│ rocclr/      │   ~160   │ ~80K    │   C++11     │
│ opencl/      │   ~350   │ ~50K    │   C/C++     │
├──────────────┼──────────┼─────────┼─────────────┤
│ 总计          │   ~580   │ ~170K   │             │
└──────────────┴──────────┴─────────┴─────────────┘
```

---

## 常见问题

### Q1: 为什么 HIP 和 OpenCL 共享 ROCclr？

**答**：避免代码重复，统一底层实现：
- ✅ 减少维护成本
- ✅ 保证一致的性能特性
- ✅ 共享设备管理、内存管理等核心功能

### Q2: hipamd 和 hip/ 的区别？

```
hip/        = 接口定义（头文件）
hipamd/     = 实现代码（源文件 + 库）

类比：
hip/        = 建筑设计图
hipamd/     = 实际建筑物
```

### Q3: 如何调试 HIP 应用？

```bash
# 1. 启用调试日志
export AMD_LOG_LEVEL=3
export HIP_TRACE_API=1

# 2. 使用 GDB
gdb --args ./myapp
(gdb) break hipMalloc
(gdb) run

# 断点会停在：
# clr/hipamd/src/hip_memory.cpp:hipMalloc()
```

### Q4: OpenCL 和 HIP 能同时使用吗？

**答**：可以，但会加载两个运行时库：
```
应用
├─ libamdhip64.so    (HIP)
└─ libamdocl64.so    (OpenCL)
     ↓
   共同依赖 HSA Runtime
```

---

## 学习路径建议

### 对于 HIP 开发者

```
1. 熟悉 API：
   📖 hip/include/hip/hip_runtime_api.h

2. 理解实现：
   📝 clr/hipamd/src/hip_memory.cpp
   📝 clr/hipamd/src/hip_module.cpp

3. 深入底层：
   🏗️ clr/rocclr/device/rocm/rocmemory.cpp
   🏗️ clr/rocclr/device/rocm/rocvirtual.cpp

4. 研究 HSA 接口：
   🔌 clr/rocclr/device/rocm/rocrctx.cpp
```

### 对于 OpenCL 开发者

```
1. 熟悉标准：
   📖 clr/opencl/khronos/headers/opencl2.2/CL/cl.h

2. 理解实现：
   📝 clr/opencl/amdocl/cl_memobj.cpp
   📝 clr/opencl/amdocl/cl_execute.cpp

3. 测试参考：
   ✅ clr/opencl/tests/ocltst/
```

### 对于 Runtime 开发者

```
1. ROCclr 架构：
   🏗️ clr/rocclr/device/device.hpp
   🏗️ clr/rocclr/platform/

2. ROCm 后端：
   🎯 clr/rocclr/device/rocm/

3. 命令提交：
   📤 clr/rocclr/device/rocm/rocvirtual.cpp
```

---

## 总结图示

### CLR 的三层架构

```
┌─────────────────────────────────────────────────────┐
│              语言前端层                              │
│  ┌──────────────┐        ┌──────────────┐          │
│  │ clr/hipamd/  │        │ clr/opencl/  │          │
│  │  HIP API     │        │  OpenCL API  │          │
│  └──────┬───────┘        └──────┬───────┘          │
└─────────┼──────────────────────┼──────────────────┘
          │                      │
          └──────────┬───────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│              通用运行时层                            │
│           clr/rocclr/                               │
│                                                      │
│  • amd::Device    - 设备抽象                         │
│  • amd::Memory    - 内存对象                         │
│  • amd::Command   - 命令对象                         │
│  • roc::Device    - ROCm 后端                        │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
           HSA Runtime (外部)
```

### 快速记忆

```
CLR 三剑客：

🎯 hipamd/    - HIP 实现（libamdhip64.so）
🏗️ rocclr/    - 共享基础（静态库）
🌐 opencl/    - OpenCL 实现（libamdocl64.so）

关系：
hipamd → rocclr → HSA Runtime
opencl → rocclr → HSA Runtime
```

这就是 `clr/` 目录的完整结构！它是 AMD 计算语言运行时的核心实现，为 HIP 和 OpenCL 提供了统一的底层基础设施。

