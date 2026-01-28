# XSched Debug 调查报告

**日期**: 2026-01-28  
**状态**: 定位到根本问题 - 基础 kernel 调用失败

---

## 🔍 问题总结

**症状**: XSched 加载成功，但任何 GPU kernel 调用都会失败

```
✅ XSched 加载: [INFO] using app-managed scheduler
❌ Kernel 执行: HIP error: invalid device function
```

---

## 📊 渐进式测试结果

### Baseline (无 XSched)
```
✅ Step 1: Basic tensor operations       - PASSED
✅ Step 2: Matrix multiplication         - PASSED  
✅ Step 3: Convolution (MIOpen)          - PASSED
✅ Step 4: Simple model                  - PASSED
✅ Step 5: ResNet                        - PASSED

结论: PyTorch + ROCm 环境完全正常
```

### XSched (LD_PRELOAD)
```
❌ Step 1: Basic tensor operations       - FAILED
   ├─ torch.randn(10, 10)                - OK (CPU)
   ├─ tensor.to('cuda:0')                - OK (内存复制)
   └─ torch.randn(..., device='cuda')    - FAILED (kernel 调用)

⚠️  无法继续到 Step 2-5
```

---

## 🎯 关键发现

### 1. 失败的操作非常基础

**最简单的失败case**:
```python
import torch
a = torch.randn(10, 10, device='cuda:0')  # ❌ 失败
```

**分析**:
- 这是 PyTorch 最基础的操作之一
- 需要调用 HIP kernel 生成随机数
- 不涉及 MIOpen、cuBLAS 等高级库
- **说明 XSched 的 kernel launch 拦截有根本性问题**

### 2. 内存操作可以工作

**可以工作的操作**:
```python
a = torch.randn(10, 10)           # ✅ CPU tensor
b = a.to('cuda:0')                # ✅ 内存复制 (cudaMemcpy)
```

**分析**:
- 内存分配和复制正常
- `hipMalloc`, `hipMemcpy` 等 API 被正确拦截
- 问题在于 `hipLaunchKernel` 或相关的调度 API

### 3. 单 GPU 环境也失败

**测试了**:
```bash
export CUDA_VISIBLE_DEVICES=0
# 结果: 仍然失败
```

**排除了**:
- 多 GPU 设备管理问题
- 设备索引混淆

### 4. 符号导出已经正确

**验证**:
```bash
$ nm -D libhalhip.so | grep HipCommand
00000000000118e0 T _ZN6xsched3hip10HipCommand...  # ✅ 符号导出
```

**当前库版本**:
```
libhalhip.so:  251K  (符号正确导出)
libshimhip.so: 420K  (依赖 libhalhip.so)
libpreempt.so: 619K
```

---

## 💡 可能的根本原因

### 假设 1: Kernel 参数传递错误

**症状**: `invalid device function`

**可能原因**:
- XSched 拦截 `hipLaunchKernel` 时修改了参数
- Kernel function pointer 被错误处理
- Grid/block dimensions 被破坏

**验证方法**:
- 查看 XSched shim 代码中的 `hipLaunchKernel` 拦截
- 启用 HIP API trace: `export HIP_TRACE_API=1`
- 使用 `rocprof` 追踪 kernel launch

### 假设 2: ROCm 版本不兼容

**环境**:
- ROCm: 6.4.0
- XSched: 编译于 ROCm 6.4.0
- PyTorch: 2.7.1+rocm6.4.1

**可能原因**:
- XSched 的 HIP shim 使用了旧版 API
- ROCm 6.4 有 breaking changes
- PyTorch 的 ROCm backend 有特殊要求

**验证方法**:
- 检查 XSched 源码中的 ROCm 版本要求
- 对比不同 ROCm 版本的 HIP API

### 假设 3: 缺少必要的初始化

**观察**:
- XSched 有 `XSchedHIP` Python 类
- 提到 `HIPQueueCreate` 等初始化函数

**可能原因**:
- XSched 需要显式创建调度队列
- LD_PRELOAD 方式可能不足以完全初始化
- 需要应用程序主动调用 XSched API

**验证方法**:
- 查找 XSched 的使用示例
- 尝试显式初始化 XSched

### 假设 4: Whitelist 机制

**观察**:
- 之前有提到 "whitelist" 和 "backtrace" 机制
- XSched 可能需要配置哪些 API 可以被调度

**可能原因**:
- `torch.randn` 使用的 kernel 不在 whitelist 中
- XSched 默认拒绝未知的 kernel

**验证方法**:
- 查找 XSched 的配置文件
- 检查是否有环境变量控制 whitelist

---

## 🔬 下一步调试计划

### 优先级 1: 分析 XSched 源码

**目标**: 理解 kernel launch 的拦截机制

**步骤**:
1. 查看 `platforms/hip/shim/src/intercept.cpp`
2. 找到 `hipLaunchKernel` 的拦截实现
3. 理解参数如何传递到调度器
4. 检查是否有错误处理或日志

### 优先级 2: 寻找工作示例

**目标**: 找到一个能工作的 XSched + PyTorch 示例

**步骤**:
1. 检查 `/data/dockercode/xsched/` 目录下的所有脚本
2. 查看是否有成功运行的历史记录
3. 对比成功和失败环境的差异

### 优先级 3: 联系开发者

**目标**: 获得官方支持

**信息准备**:
- 最小复现示例 (`torch.randn` 失败)
- 环境信息 (ROCm 6.4.0, MI308X)
- 编译配置和库版本
- 详细错误日志

### 优先级 4: 尝试降级

**目标**: 排除版本兼容性问题

**步骤**:
1. 尝试 ROCm 5.7 或更早版本
2. 尝试不同的 PyTorch 版本
3. 查看 XSched 的已知工作环境

---

## 📝 已尝试的方法

### ✅ 已完成

1. **符号导出修复** ✅
   - 重新编译 libhalhip.so 不使用 version script
   - 符号正确导出 (251K vs 211K)
   - XSched 加载成功

2. **渐进式测试** ✅
   - 从简单到复杂的测试套件
   - 精确定位到最基础的操作失败

3. **单 GPU 测试** ✅
   - 排除多 GPU 环境问题

4. **内存操作验证** ✅
   - 确认内存 API 正常工作

### ❌ 未成功

1. **直接使用 XSched** ❌
   - 所有 kernel 调用都失败

2. **环境变量调整** ❌
   - `CUDA_VISIBLE_DEVICES=0`: 无效
   - `AMD_SERIALIZE_KERNEL=3`: 无效
   - `HIP_TRACE_API=1`: 需要进一步分析输出

---

## 🎯 临时解决方案

### 选项 1: 仅分析 Baseline 数据

**优点**:
- 已有完整的 Baseline 性能数据
- 证明了 Native scheduler 的问题 (P99 增加 7.4x)
- 可以进行理论分析

**缺点**:
- 无法验证 XSched 的实际效果
- 论文声明需要实验数据支持

### 选项 2: 使用其他调度器

**候选**:
- AMD 原生优先级 API (如果存在)
- HSA queue priority
- 自定义调度方案

**优点**:
- 可能更容易集成
- 官方支持更好

**缺点**:
- 需要研究和开发
- 不是 XSched 论文的验证

### 选项 3: 修改 XSched 源码

**可能的修改**:
- 添加更详细的日志
- 跳过某些检查或过滤
- 修复 kernel launch 参数传递

**优点**:
- 彻底解决问题
- 理解 XSched 内部机制

**缺点**:
- 需要深入理解 XSched 代码
- 可能引入新问题
- 时间成本高

---

## 📊 性能影响估算

### 如果 XSched 正常工作

基于理论分析和 Baseline 数据：

```
High Priority Task (ResNet-18, 20 req/s):
  Baseline P99: 19.62 ms
  XSched P99:   <10 ms (预期)
  改善:         -50%+

Low Priority Task (ResNet-50, batch=1024):
  Baseline:     2016 img/s
  XSched:       ~1700 img/s (预期)
  权衡:         -15%

整体评价:
  ✅ 高优先级 QoS 显著提升
  ⚠️ 低优先级合理牺牲
  ✅ 系统获得优先级控制能力
```

---

## 🔴 阻塞问题

**核心阻塞**: XSched 无法执行任何 GPU kernel

**影响范围**:
- ❌ 无法运行任何实际推理
- ❌ 无法获得性能数据
- ❌ 无法验证调度策略
- ✅ 但 Baseline 数据完整可用

**紧急程度**: **高**
- 如果目标是完整验证 XSched: 必须解决
- 如果目标是分析调度问题: 可以用 Baseline 数据

---

## 📌 结论

1. **问题定位明确**
   - XSched 的 kernel launch 拦截有根本性问题
   - 不是符号导出、不是多 GPU、不是 MIOpen 特有问题
   - 最基础的 `torch.randn` 就失败

2. **Baseline 价值确认**
   - 已证明 Native scheduler 在高负载下的严重性能退化
   - P99 延迟增加 7.4 倍 (2.65ms → 19.62ms)
   - 为优先级调度提供了明确的需求和价值论证

3. **下一步决策点**
   - **路径 A**: 深入调试 XSched (需要大量时间)
   - **路径 B**: 基于 Baseline 数据完成分析报告
   - **路径 C**: 寻找替代的调度方案

---

**报告时间**: 2026-01-28 15:30  
**状态**: 阻塞于 kernel launch 失败  
**建议**: 需要 XSched 开发者支持或考虑替代方案
