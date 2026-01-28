# XSched Phase 2: 实施进展与问题分析

## 📋 文档信息

- **创建时间**: 2026-01-27
- **测试平台**: AMD MI308X (Docker: zhenaiter)
- **状态**: 遇到 HIP Context 冲突问题

---

## ✅ 已完成的工作

### 1. XSched C API Python 绑定

成功使用 `ctypes` 创建了完整的 XSched C API 绑定：

```python
# 加载 XSched 库
libpreempt = ctypes.CDLL("/workspace/xsched/output/lib/libpreempt.so")
libhalhip = ctypes.CDLL("/workspace/xsched/output/lib/libhalhip.so")

# 定义类型
XQueueHandle = ctypes.c_uint64
HwQueueHandle = ctypes.c_uint64
Priority = ctypes.c_int32

# 定义函数签名
libpreempt.XHintSetScheduler.argtypes = [XSchedulerType, XPolicyType]
libpreempt.XHintPriority.argtypes = [XQueueHandle, Priority]
libpreempt.XQueueCreate.argtypes = [...]
libhalhip.HipQueueCreate.argtypes = [...]
```

**状态**: ✅ 完成

### 2. XSchedQueue 包装类

创建了 Python 类来管理 XSched 队列：

```python
class XSchedQueue:
    def __init__(self, stream: torch.cuda.Stream, priority: int):
        # 1. 获取 HIP Stream 句柄
        hip_stream = ctypes.c_void_p(stream.cuda_stream)
        
        # 2. 创建 HwQueue
        libhalhip.HipQueueCreate(ctypes.byref(self.hwq), hip_stream)
        
        # 3. 创建 XQueue
        libpreempt.XQueueCreate(ctypes.byref(self.xq), self.hwq, ...)
        
        # 4. 设置优先级
        libpreempt.XHintPriority(self.xq, priority)
```

**状态**: ✅ 完成

### 3. 完整的测试脚本

创建了两个测试脚本：

1. **`test_xsched_integration.py`**: 完整的 BERT 推理测试，包含 Baseline 和 XSched 对比
2. **`test_xsched_simple.py`**: 简单的 XSched 集成验证测试

**状态**: ✅ 完成

### 4. 技术文档

创建了详细的技术文档：

- **`XSched_Phase2_真实GPU优先级调度实现报告.md`**: 完整的技术说明和使用指南
- **`XSched_Phase2_实施进展与问题分析.md`**: 本文档

**状态**: ✅ 完成

---

## ❌ 遇到的问题

### 问题描述

在运行测试时，遇到以下错误：

```
[INFO @ T19594 @ 05:15:13.467276] using local scheduler with policy HPF
[ERRO @ T19594 @ 05:15:13.481502] hip error 709: context is destroyed @ /workspace/xsched/platforms/hip/hal/src/hip_queue.cpp:32
```

**错误代码**: `hip error 709` - `hipErrorContextIsDestroyed`

### 问题分析

#### 1. HIP Context 冲突

PyTorch 和 XSched 都需要管理 HIP context：

```
┌─────────────────────────────────────────────────────────────┐
│ PyTorch (torch.cuda)                                        │
│  - 创建并管理自己的 HIP context                             │
│  - 创建 HIP Streams                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ XSched (libhalhip.so)                                       │
│  - 尝试访问 PyTorch 创建的 HIP Stream                       │
│  - 可能在 context 被销毁后访问                              │
└─────────────────────────────────────────────────────────────┘
```

**可能的原因**：
1. **Context 生命周期管理**: PyTorch 的 HIP context 可能在 XSched 访问之前或期间被销毁
2. **多线程问题**: HIP context 是线程局部的，XSched 可能在不同的线程中访问
3. **初始化顺序**: XSched 初始化时，PyTorch 的 HIP context 可能还未完全建立

#### 2. 错误发生位置

错误发生在 `/workspace/xsched/platforms/hip/hal/src/hip_queue.cpp:32`：

```cpp
// hip_queue.cpp (推测)
XResult HipQueueCreate(HwQueueHandle *hwq, hipStream_t stream) {
    // Line 32: 尝试访问 HIP context
    hipError_t err = hipStreamQuery(stream);  // 或类似的 HIP API 调用
    if (err == hipErrorContextIsDestroyed) {
        // 错误：context 已被销毁
        return kXSchedErrorHardware;
    }
    ...
}
```

---

## 🔍 可能的解决方案

### 方案 1: 使用 LD_PRELOAD 透明拦截（推荐）⭐⭐⭐⭐⭐

XSched 的设计初衷就是通过 `LD_PRELOAD` 透明地拦截 HIP API 调用，而不是直接在 Python 中调用 C API。

#### 实现步骤

1. **使用 XSched 的 Shim 库**:

```bash
export LD_PRELOAD=/workspace/xsched/output/lib/libshimhip.so
```

2. **运行普通的 PyTorch 代码**:

```python
# 不需要任何 XSched C API 调用！
import torch

# 设置环境变量来配置 XSched
os.environ['XSCHED_SCHEDULER'] = 'local'
os.environ['XSCHED_POLICY'] = 'HPF'  # Highest Priority First

# 正常使用 PyTorch
stream1 = torch.cuda.Stream(priority=-1)  # 高优先级
stream2 = torch.cuda.Stream(priority=0)   # 低优先级

with torch.cuda.stream(stream1):
    output1 = model(input1)  # XSched 会自动拦截并管理

with torch.cuda.stream(stream2):
    output2 = model(input2)
```

3. **XSched 自动拦截**:

```
┌─────────────────────────────────────────────────────────────┐
│ Python 代码                                                  │
│  stream = torch.cuda.Stream(priority=-1)                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ libshimhip.so (LD_PRELOAD)                                  │
│  - 拦截 hipStreamCreate() 调用                              │
│  - 自动创建 XQueue 并设置优先级                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ XSched 调度器                                               │
│  - 管理所有 XQueue                                          │
│  - 根据优先级调度                                           │
└─────────────────────────────────────────────────────────────┘
```

**优点**：
- ✅ 完全透明，不需要修改 Python 代码
- ✅ 避免 context 管理冲突
- ✅ 这是 XSched 设计的正确使用方式

**缺点**：
- ⚠️ 需要设置 `LD_PRELOAD`
- ⚠️ 调试可能更困难

### 方案 2: 修复 Context 管理

直接修复 Python 代码中的 context 管理问题。

#### 可能的修复方法

1. **确保 Context 持久化**:

```python
# 在全局作用域创建一个持久的 CUDA tensor
_cuda_context_holder = torch.zeros(1, device='cuda:0')

def create_xsched_queue(stream, priority):
    # 确保 context 存在
    torch.cuda.synchronize()
    _cuda_context_holder.cpu()  # 强制访问 context
    
    # 创建 XQueue
    ...
```

2. **使用 CUDA Primary Context**:

```python
import ctypes
libcuda = ctypes.CDLL('libamdhip64.so')

# 获取并保持 primary context
device = 0
ctx = ctypes.c_void_p()
libcuda.hipDevicePrimaryCtxRetain(ctypes.byref(ctx), device)

# 创建 XQueue
...

# 在程序结束时释放
libcuda.hipDevicePrimaryCtxRelease(device)
```

3. **延迟 XQueue 创建**:

```python
# 在第一次使用时才创建 XQueue，而不是在初始化时
class LazyXSchedQueue:
    def __init__(self, stream, priority):
        self.stream = stream
        self.priority = priority
        self._xq = None
    
    def ensure_created(self):
        if self._xq is None:
            # 在这里创建 XQueue
            ...
```

**优点**：
- ✅ 保持直接调用 C API 的方式
- ✅ 更精细的控制

**缺点**：
- ❌ 需要深入理解 HIP context 管理
- ❌ 可能仍然有其他隐藏的冲突

### 方案 3: 使用 XSched 的 C++ 示例作为参考

直接使用 C++ 编写测试程序，避免 Python/PyTorch 的 context 管理问题。

```cpp
// test_xsched_bert.cpp
#include <hip/hip_runtime.h>
#include "xsched/xsched.h"
#include "xsched/hip/hal.h"

int main() {
    // 初始化 HIP
    hipSetDevice(0);
    
    // 创建 Stream
    hipStream_t stream;
    hipStreamCreate(&stream);
    
    // 创建 XQueue
    HwQueueHandle hwq;
    HipQueueCreate(&hwq, stream);
    
    XQueueHandle xq;
    XQueueCreate(&xq, hwq, kPreemptLevelBlock, kQueueCreateFlagNone);
    
    // 设置优先级
    XHintPriority(xq, 3);
    
    // 运行 kernel
    ...
    
    return 0;
}
```

**优点**：
- ✅ 避免 Python/PyTorch 的复杂性
- ✅ 更接近 XSched 的原生使用方式

**缺点**：
- ❌ 无法直接使用 PyTorch 的模型
- ❌ 需要重新实现 BERT 推理逻辑

---

## 📊 Example 3 的成功经验

我们之前成功运行了 Example 3 (`app_concurrent.hip`)，它也是使用 XSched C API。让我们分析它为什么能成功：

### Example 3 的关键点

1. **纯 HIP 代码**:
   - 不依赖 PyTorch
   - 直接使用 HIP Runtime API

2. **Context 管理**:
   ```cpp
   // 在 main 函数开始时
   hipSetDevice(0);  // 显式设置设备
   
   // 在 run 函数中
   hipStreamCreate(&stream);  // 创建 stream
   HipQueueCreate(&hwq, stream);  // 立即创建 HwQueue
   XQueueCreate(&xq, hwq, ...);  // 立即创建 XQueue
   ```

3. **生命周期管理**:
   - Stream, HwQueue, XQueue 都在同一个作用域内
   - 没有跨线程访问

### 与我们的代码的差异

| 维度 | Example 3 (成功) | 我们的代码 (失败) |
|------|-----------------|------------------|
| **语言** | C++ | Python |
| **HIP 管理** | 直接使用 HIP API | 通过 PyTorch 间接使用 |
| **Context** | 显式管理 | PyTorch 自动管理 |
| **生命周期** | 简单明确 | 复杂（Python GC + PyTorch） |

---

## 🎯 推荐的下一步

### 短期方案（立即可行）

**使用方案 1: LD_PRELOAD 透明拦截**

1. 创建一个新的测试脚本 `test_xsched_preload.py`
2. 使用 `LD_PRELOAD` 加载 `libshimhip.so`
3. 运行普通的 PyTorch 代码，让 XSched 自动拦截

```bash
# 运行脚本
LD_PRELOAD=/workspace/xsched/output/lib/libshimhip.so \
    python3 test_bert_inference.py
```

### 中期方案（需要调试）

**修复 Context 管理问题**

1. 深入研究 XSched 的 HIP HAL 实现
2. 理解 PyTorch 的 HIP context 管理机制
3. 找到两者兼容的方式

### 长期方案（最彻底）

**贡献给 XSched 项目**

1. 向 XSched 项目报告这个问题
2. 提供 PyTorch 集成的补丁
3. 添加 PyTorch 示例到 XSched 仓库

---

## 📝 相关文件

1. **测试脚本**:
   - `/mnt/md0/zhehan/code/flashinfer/dockercode/xsched/test_xsched_integration.py`
   - `/mnt/md0/zhehan/code/flashinfer/dockercode/xsched/test_xsched_simple.py`

2. **文档**:
   - `/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED/XSched_Phase2_真实GPU优先级调度实现报告.md`
   - `/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED/XSched_Phase2_实施进展与问题分析.md`

3. **参考代码**:
   - `/workspace/xsched/examples/Linux/3_intra_process_sched/app_concurrent.hip` (成功的例子)
   - `/workspace/xsched/platforms/hip/hal/src/hip_queue.cpp` (错误发生位置)

---

## 💡 总结

### 已完成
- ✅ XSched C API Python 绑定
- ✅ XSchedQueue 包装类
- ✅ 完整的测试脚本
- ✅ 详细的技术文档

### 当前问题
- ❌ HIP Context 冲突 (`hipErrorContextIsDestroyed`)
- ❌ PyTorch 和 XSched 的 context 管理不兼容

### 推荐方案
- 🎯 **首选**: 使用 `LD_PRELOAD` 方式（方案 1）
- 🎯 **备选**: 修复 context 管理（方案 2）
- 🎯 **最后**: 使用纯 C++ 实现（方案 3）

### 关键洞察
**XSched 的设计初衷是通过 `LD_PRELOAD` 透明拦截，而不是直接在应用代码中调用 C API。** 我们应该遵循这个设计理念，使用 Shim 库的方式来集成 XSched。

---

## 🔗 参考资料

1. **XSched 论文**: `/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/papers/XSched_Preemptive Scheduling for Diverse XPUs.pdf`
2. **XSched README**: `/workspace/xsched/README.md`
3. **HIP Error Codes**: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html#error-codes
4. **PyTorch CUDA Streams**: https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams

---

**创建时间**: 2026-01-27  
**最后更新**: 2026-01-27  
**状态**: 问题分析完成，等待实施解决方案

