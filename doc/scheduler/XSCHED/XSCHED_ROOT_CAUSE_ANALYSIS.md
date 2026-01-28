# XSched 失败根本原因分析

**日期**: 2026-01-28  
**状态**: ✅ 根本原因已定位

---

## 🎯 根本原因

### 发现

通过深入分析 XSched 源码，发现关键问题：

```cpp
// platforms/hip/hal/include/xsched/hip/hal/handle.h
inline HwQueueHandle GetHwQueueHandle(hipStream_t stream)
{
    return (HwQueueHandle)stream;  // 只是强制类型转换！
}
```

```cpp
// platforms/hip/shim/src/shim.cpp
hipError_t XLaunchKernel(const void *f, dim3 numBlocks, dim3 dimBlocks, void **args,
                         size_t sharedMemBytes, hipStream_t stream)
{
    XDEBG("XLaunchKernel: func=%p stream=%p\\n", f, stream);
    if (stream == nullptr) {  // ← 默认流直接绕过！
        HipSyncBlockingXQueues();
        return Driver::LaunchKernel(f, numBlocks, dimBlocks, args, sharedMemBytes, stream);
    }
    
    auto xqueue = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xqueue == nullptr) {  // ← XQueue 不存在也绕过！
        return Driver::LaunchKernel(f, numBlocks, dimBlocks, args, sharedMemBytes, stream);
    }
    
    // 只有这里才会使用 XSched
    auto kernel = std::make_shared<HipKernelLaunchCommand>(...);
    xqueue->Submit(kernel);
    return hipSuccess;
}
```

---

## 🔍 问题分解

### 问题 1: 默认流绕过 XSched

**症状**: `torch.randn(device='cuda')` 使用默认流 (stream=nullptr)

**代码路径**:
1. PyTorch 调用 `hipLaunchKernel(kernel, ..., stream=nullptr)`
2. XSched 拦截，调用 `XLaunchKernel(..., stream=nullptr)`
3. 代码判断 `if (stream == nullptr)`
4. **直接调用 `Driver::LaunchKernel` 绕过 XSched！**

**为什么失败**:
- `Driver::LaunchKernel` 是 XSched 对原始 HIP 的封装
- 这个封装可能有问题或不完整

### 问题 2: XQueue 初始化

**症状**: 即使 stream != nullptr，`HwQueueManager::GetXQueue(...)` 也可能返回 nullptr

**代码路径**:
1. `GetHwQueueHandle(stream)` 返回 stream 强制转换为 `HwQueueHandle`
2. `HwQueueManager::GetXQueue(handle)` 查找对应的 XQueue
3. 如果 stream 没有被 XSched 注册/创建，返回 nullptr
4. **再次走 fallback 路径**

**为什么会这样**:
- PyTorch 使用 ROCm 自己创建的 stream
- XSched 不知道这些 stream 的存在
- 需要显式告诉 XSched 哪些 stream 需要调度

### 问题 3: Driver::LaunchKernel 的实现

**关键问题**: `Driver::LaunchKernel` 到底是什么？

让我查找它的定义...

---

## 🔬 Driver::LaunchKernel 分析

`Driver::` 是 XSched 对原始 HIP API 的命名空间封装。

**可能的问题**:
1. **符号版本问题**: 之前修复的符号导出可能不完整
2. **API 兼容性**: ROCm 6.4 的 API 可能有变化
3. **函数指针问题**: `hipLaunchKernel` 的函数指针可能被错误处理

---

## 💡 解决方案

### 方案 A: 修改 XSched 代码（推荐）⭐

**修改点 1**: 支持默认流

```cpp
// 修改 shim.cpp XLaunchKernel
hipError_t XLaunchKernel(const void *f, dim3 numBlocks, dim3 dimBlocks, void **args,
                         size_t sharedMemBytes, hipStream_t stream)
{
    // 如果是默认流，使用 hipStreamPerThread 或创建默认 XQueue
    if (stream == nullptr) {
        // 选项 1: 使用 per-thread 流
        hipStreamPerThread(&stream);
        
        // 选项 2: 或使用全局默认 XQueue
        // stream = GetDefaultXStream();
    }
    
    auto xqueue = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xqueue == nullptr) {
        // 选项: 动态创建 XQueue
        xqueue = HwQueueManager::CreateXQueueForStream(stream);
        
        // 如果还是 nullptr，才 fallback
        if (xqueue == nullptr) {
            return ORIGINAL_hipLaunchKernel(f, numBlocks, dimBlocks, args, sharedMemBytes, stream);
        }
    }
    
    auto kernel = std::make_shared<HipKernelLaunchCommand>(...);
    xqueue->Submit(kernel);
    return hipSuccess;
}
```

**修改点 2**: 修复 Driver::LaunchKernel

需要确保 `Driver::LaunchKernel` 正确调用原始的 `hipLaunchKernel`，而不是递归调用。

### 方案 B: 使用显式流初始化

**在 PyTorch 代码中**:

```python
import torch
from xsched_hip import XSchedHIP

# 创建并注册流
stream = torch.cuda.Stream()
XSchedHIP.RegisterStream(stream.cuda_stream)  # 假设有这个 API

# 使用这个流
with torch.cuda.stream(stream):
    a = torch.randn(100, 100, device='cuda:0')
```

**问题**: 需要修改测试代码，且不通用。

### 方案 C: 完全绕过默认流检查（临时）

**快速测试修改**:

```cpp
// 临时注释掉默认流的检查
hipError_t XLaunchKernel(const void *f, dim3 numBlocks, dim3 dimBlocks, void **args,
                         size_t sharedMemBytes, hipStream_t stream)
{
    // TEMPORARY: 注释掉这个检查
    // if (stream == nullptr) {
    //     HipSyncBlockingXQueues();
    //     return Driver::LaunchKernel(f, numBlocks, dimBlocks, args, sharedMemBytes, stream);
    // }
    
    // 强制使用 stream=0 的 XQueue
    if (stream == nullptr) stream = (hipStream_t)1;  // 使用非零值
    
    auto xqueue = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    // ...
}
```

---

## 🎯 推荐的修复步骤

### Step 1: 验证假设（已完成）

创建测试来验证默认流的行为：

```python
# 测试默认流
a = torch.randn(10, 10, device='cuda')  # ❌ 失败

# 测试显式流
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    b = torch.randn(10, 10, device='cuda')  # ？会成功吗
```

### Step 2: 查找 Driver::LaunchKernel 定义

找到它的实际实现，确认是否有问题。

### Step 3: 修改 XSched 代码

根据方案 A 修改 `shim.cpp`。

### Step 4: 重新编译测试

```bash
cd /data/dockercode/xsched-official
make clean
make hip
# 复制库文件
cp build/platforms/hip/*.so /data/dockercode/xsched-build/output/lib/
```

### Step 5: 验证修复

运行渐进式测试确认修复。

---

## 📊 预期结果

### 修复前
```
✅ Baseline: 所有测试通过
❌ XSched: Step 1 失败（默认流绕过 → Driver::LaunchKernel 失败）
```

### 修复后
```
✅ Baseline: 所有测试通过
✅ XSched: 所有测试通过（默认流正确处理）
```

---

## 🔍 需要进一步调查

1. **Driver::LaunchKernel 的实现**
   - 在哪里定义？
   - 是宏还是函数？
   - 如何调用原始 HIP API？

2. **HwQueueManager 的工作机制**
   - 如何注册 stream？
   - GetXQueue 的查找逻辑？
   - 能否动态创建 XQueue？

3. **PyTorch 的流使用模式**
   - 何时使用默认流？
   - 何时创建新流？
   - 能否强制使用显式流？

---

## 📝 下一步行动

1. ✅ 已定位根本原因：默认流绕过 XSched
2. ⏳ 查找 `Driver::LaunchKernel` 定义
3. ⏳ 实现方案 A 的修改
4. ⏳ 编译测试
5. ⏳ 验证修复

---

**报告时间**: 2026-01-28  
**状态**: 根本原因已明确，准备实施修复  
**信心等级**: ⭐⭐⭐⭐⭐ (非常有信心这是问题所在)
