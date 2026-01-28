# XSched 修复尝试状态报告

**日期**: 2026-01-28 15:43  
**状态**: 部分进展，需要进一步调查

---

## ✅ 已完成的工作

### 1. 根本原因定位 ⭐⭐⭐⭐⭐
- 通过源码分析找到关键问题
- 定位到默认流处理逻辑
- 理解了 XSched 的 kernel launch 拦截机制

### 2. 第一次修复尝试
- 修改了 `shim.cpp` 中的 XLaunchKernel
- 注释掉了默认流的特殊处理
- 成功重新编译

### 3. 完整的 Baseline 测试数据
- ✅ Test 1-4 全部完成
- ✅ 高负载 P99 = 19.62ms（增加 7.4 倍）
- ✅ 证明了 Native scheduler 的问题

---

## 🎯 发现的根本原因

```cpp
// platforms/hip/shim/src/shim.cpp
hipError_t XLaunchKernel(..., hipStream_t stream)
{
    if (stream == nullptr) {  // ← PyTorch 使用默认流!
        HipSyncBlockingXQueues();
        return Driver::LaunchKernel(...);  // ← 这里失败!
    }
    
    auto xqueue = HwQueueManager::GetXQueue(GetHwQueueHandle(stream));
    if (xqueue == nullptr) {  // ← 或者这里没有 XQueue
        return Driver::LaunchKernel(...);  // ← 也失败!
    }
    
    // 只有这里才使用 XSched
    auto kernel = std::make_shared<HipKernelLaunchCommand>(...);
    xqueue->Submit(kernel);
    return hipSuccess;
}
```

### 问题链条

1. **PyTorch 使用默认流** (stream = nullptr)
2. **代码绕过 XSched** → 直接调用 `Driver::LaunchKernel`
3. **Driver::LaunchKernel 失败** → `HIP error: invalid device function`

### Driver::LaunchKernel 的本质

```cpp
// platforms/hip/hal/include/xsched/hip/hal/driver.h
DEFINE_STATIC_ADDRESS_CALL(GetSymbol("hipLaunchKernel"), 
                           hipError_t, LaunchKernel, ...)
```

- 通过 `dlsym` 动态加载原始的 `hipLaunchKernel`
- 理论上应该工作，但实际失败
- **可能原因**:
  - 符号加载失败
  - 函数指针错误
  - 参数传递问题

---

## 🔧 修复尝试 1：注释默认流检查

### 修改内容

```cpp
// 原来的代码
if (stream == nullptr) {
    HipSyncBlockingXQueues();
    return Driver::LaunchKernel(...);  // ← 失败的路径
}

// 修改后
// 注释掉这个检查，让默认流也走 XQueue 路径
// if (stream == nullptr) { ... }
```

### 测试结果

❌ **仍然失败** - 同样的错误

**原因分析**:
- 即使绕过了默认流检查
- `GetXQueue(GetHwQueueHandle(nullptr))` 返回 nullptr
- 还是会走到 `Driver::LaunchKernel`
- 问题依然存在

---

## 💡 深层问题：Driver::LaunchKernel 为什么失败？

### 可能性 1: 符号加载失败

```cpp
DEFINE_STATIC_ADDRESS_CALL(GetSymbol("hipLaunchKernel"), ...)
```

**验证方法**:
- 在 Driver::LaunchKernel 调用前添加日志
- 检查函数指针是否为 nullptr
- 确认 dlsym 是否成功

### 可能性 2: LD_PRELOAD 干扰

**问题**:
- XSched 通过 LD_PRELOAD 拦截 HIP API
- `libshimhip.so` 本身导出了 `hipLaunchKernel`
- `Driver::LaunchKernel` 通过 dlsym 查找 "hipLaunchKernel"
- **可能找到了 libshimhip.so 中的版本，而不是 libamdhip64.so 中的原始版本**
- **造成递归调用或无限循环！**

### 可能性 3: XSched 需要显式初始化

**观察**:
- XSched 可能需要预先创建 XQueue
- 默认流可能需要特殊注册
- 缺少初始化步骤

---

## 🎯 建议的下一步修复方案

### 方案 A: 修复 Driver::LaunchKernel（推荐）⭐

**目标**: 让 Driver::LaunchKernel 正确调用原始 HIP API

**步骤**:
1. **使用 RTLD_NEXT** 而不是查找符号名
   ```cpp
   // 修改 GetSymbol 实现
   void* GetSymbol(const char* name) {
       return dlsym(RTLD_NEXT, name);  // 查找下一个库中的符号
   }
   ```

2. **或者直接打开 libamdhip64.so**
   ```cpp
   void* handle = dlopen("/opt/rocm/lib/libamdhip64.so", RTLD_NOW);
   void* func = dlsym(handle, "hipLaunchKernel");
   ```

### 方案 B: 为默认流创建 XQueue

**目标**: 确保所有流都有对应的 XQueue

**步骤**:
1. 在 XSched 初始化时创建默认流的 XQueue
2. 修改 `GetHwQueueHandle(nullptr)` 返回有效句柄
3. 确保 `HwQueueManager::GetXQueue(0)` 返回默认 XQueue

### 方案 C: 完全绕过 Driver 封装

**目标**: 直接使用原始 HIP API

**步骤**:
1. 声明原始 HIP 函数指针
2. 在初始化时通过 RTLD_NEXT 获取
3. 在 fallback 路径直接调用原始函数

---

## 📊 当前测试状态

### Baseline (无 XSched)
```
✅ Step 1: Basic tensor operations - PASSED
✅ Step 2: Matrix multiplication   - PASSED
✅ Step 3: Convolution (MIOpen)    - PASSED
✅ Step 4: Simple model            - PASSED
✅ Step 5: ResNet                  - PASSED
```

### XSched (修复尝试 1)
```
✅ Step 1.1: CPU tensor            - PASSED
✅ Step 1.2: to('cuda')            - PASSED
❌ Step 1.3: randn(device='cuda')  - FAILED
    └─ HIP error: invalid device function
```

---

## 🔬 需要的调试信息

### 1. 验证 Driver::LaunchKernel 函数指针

```cpp
// 在 shim.cpp 中添加
XINFO("Driver::LaunchKernel address: %p", &Driver::LaunchKernel);
void* direct_func = dlsym(RTLD_NEXT, "hipLaunchKernel");
XINFO("RTLD_NEXT hipLaunchKernel: %p", direct_func);
```

### 2. 检查符号查找顺序

```bash
LD_DEBUG=symbols LD_PRELOAD=... python3 test.py 2>&1 | grep hipLaunchKernel
```

### 3. 验证 XQueue 创建

```cpp
XINFO("GetHwQueueHandle(stream=%p) = %p", stream, GetHwQueueHandle(stream));
XINFO("GetXQueue result: %p", xqueue.get());
```

---

## 📝 文件清单

### 分析文档
- `DEBUG_XSCHED_FINDINGS.md` - 初步调查
- `XSCHED_ROOT_CAUSE_ANALYSIS.md` - 根本原因分析 ⭐
- `FIX_ATTEMPT_STATUS.md` - 本文档

### 修复相关
- `fix_xsched_default_stream.patch` - 补丁文件
- `apply_fix_and_rebuild.sh` - 应用修复脚本
- `phase4_log/fix_apply_*.log` - 编译日志
- `phase4_log/debug_after_fix_*.log` - 测试日志

### 备份
- Docker 容器内：`/data/dockercode/xsched-official/platforms/hip/shim/src/shim.cpp.backup`

---

## 💪 继续前进的策略

### 短期（立即）
1. **尝试方案 A** - 修复 Driver::LaunchKernel 符号查找
2. 添加详细日志验证假设
3. 重新编译测试

### 中期（如果方案 A 失败）
4. **尝试方案 B** - 为默认流创建 XQueue
5. 研究 HwQueueManager 的初始化机制
6. 查看是否有配置文件或环境变量

### 长期（备选）
7. **联系 XSched 开发者** - 提供详细的调试信息
8. **探索替代方案** - AMD 原生优先级 API
9. **基于 Baseline 完成分析** - 理论推导 XSched 效果

---

## ✅ 已证明的价值（即使 XSched 未运行）

1. **问题明确存在**
   - Native scheduler 高负载 P99 增加 **7.4 倍**
   - 从 2.65ms → 19.62ms

2. **测试方法有效**
   - 数据一致性验证通过
   - 多线程真正并发

3. **XSched 架构理解**
   - 完整的代码流程分析
   - 识别了关键瓶颈点

---

## 🎯 下一个动作项

**选择**:
1. **继续深入修复** (推荐方案 A)
2. **暂时停止，基于 Baseline 完成报告**
3. **寻求外部帮助** (XSched 开发者)

**时间估算**:
- 方案 A 修复: 2-4 小时
- Baseline 报告: 1 小时
- 联系开发者: 响应时间未知

---

**报告时间**: 2026-01-28 15:43  
**状态**: 修复尝试 1 失败，准备尝试方案 A  
**信心等级**: ⭐⭐⭐ (方案 A 有 60% 成功概率)
