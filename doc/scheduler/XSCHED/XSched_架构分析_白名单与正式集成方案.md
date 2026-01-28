# XSched 白名单机制分析与 ROCm/HIP 正式集成方案

**日期**: 2026-01-27  
**作者**: 技术分析团队  
**主题**: 白名单机制的性质分析与长期集成建议

---

## 目录

1. [白名单机制的性质：是Workaround吗？](#白名单机制的性质)
2. [当前架构的局限性](#当前架构的局限性)
3. [正式集成到ROCm/HIP的方案对比](#正式集成方案对比)
4. [驱动层集成的优势与挑战](#驱动层集成)
5. [推荐的长期架构](#推荐架构)

---

## 白名单机制的性质

### 直接回答：**是的，白名单机制是一个Workaround！**

#### 1.1 为什么说是Workaround？

**定义对比**：

| 特征 | 正式解决方案 | Workaround（权宜之计） | 白名单机制 |
|-----|-------------|---------------------|----------|
| **解决根本问题** | ✅ 是 | ❌ 否，绕过问题 | ❌ 绕过冲突 |
| **架构优雅性** | ✅ 自然融入系统 | ❌ 需要额外逻辑 | ❌ 需要调用者检测 |
| **维护成本** | ✅ 低 | ⚠️ 中到高 | ⚠️ 需要持续维护白名单 |
| **性能开销** | ✅ 最优 | ⚠️ 可能有额外开销 | ⚠️ 每次调用都需要检测 |
| **可扩展性** | ✅ 好 | ❌ 差 | ❌ 新库需要手动添加 |

**具体表现**：

```cpp
// 白名单机制的"不优雅"之处
bool should_passthrough() {
    // 1. 每次HIP API调用都需要检查调用者
    Dl_info info;
    dladdr(__builtin_return_address(0), &info);  // ← 性能开销
    
    // 2. 硬编码的库名列表
    if (lib.find("libhipblas.so") != std::string::npos) {  // ← 维护负担
        return true;
    }
    
    // 3. 每个新的深度学习库都需要手动添加
    if (lib.find("libMIOpen.so") != std::string::npos) {   // ← 不可扩展
        return true;
    }
    
    // 4. 深层调用链无法正确识别（PyTorch问题）
    // ← 机制本身有缺陷
}
```

#### 1.2 为什么现在需要这个Workaround？

**根本原因**：XSched的设计假设与现代深度学习框架的现实不符。

```
XSched 的假设（2022年论文设计）:
┌─────────────────────────────────────┐
│ 应用代码直接调用 HIP API             │
│ → XSched拦截 → 调度                  │
│ 简单、直接、可控                      │
└─────────────────────────────────────┘

现代深度学习框架的现实（2026年）:
┌─────────────────────────────────────┐
│ PyTorch/TensorFlow                  │
│   ↓                                  │
│ 高层抽象（libtorch, etc）            │
│   ↓                                  │
│ 中间优化层（operator fusion）        │
│   ↓                                  │
│ BLAS/DNN库（libhipblas, MIOpen）    │
│   ↓ （内部调用HIP API）              │
│ HIP Runtime                          │
│ 复杂、多层、有状态                    │
└─────────────────────────────────────┘

冲突: XSched拦截了BLAS/DNN库的内部调用
     → 破坏了这些库的状态管理
     → 导致错误（HIPBLAS_STATUS_INTERNAL_ERROR）
```

---

## 当前架构的局限性

### 2.1 LD_PRELOAD + 白名单的问题

#### 问题 1: 性能开销

```cpp
// 每次HIP API调用的开销
hipMemcpy(...) 被调用
  ↓
  1. 进入 libshimhip.so 的包装函数        [1-2 CPU cycles]
  2. 调用 should_passthrough()            [开始性能损耗]
  3.   调用 dladdr(__builtin_return_address(0))  [~100-500 cycles]
  4.   查询调用者库信息（系统调用）
  5.   字符串匹配检查白名单              [~10-50 cycles]
  6. 决策：透传 or 调度                  
  7. 如果透传：调用 dlsym(RTLD_NEXT)     [~50-100 cycles]
  8. 调用原始函数

总开销: 150-650 CPU cycles PER API CALL
```

**影响**：
- 对于频繁调用的API（如 `hipMemcpy`），这是显著的开销
- 深度学习训练中，每个 mini-batch 可能调用数千次 HIP API

#### 问题 2: 可维护性

**白名单需要持续维护**：

| 时间 | ROCm版本 | 新增库 | 白名单更新 |
|-----|---------|--------|----------|
| 2024 | ROCm 5.7 | - | libhipblas, MIOpen |
| 2025 | ROCm 6.0 | libhipblaslt | ➕ 需要添加 |
| 2026 | ROCm 6.4 | libhipsparse, librccl | ➕ 需要添加 |
| 2027 | ROCm 7.0 | ??? | ➕ 持续添加 |

**问题**：
- 每次 ROCm 更新，都需要检查是否有新库
- 第三方库（如 Triton, Flash Attention）需要用户自己添加
- 没有自动发现机制

#### 问题 3: 语义不清晰

```bash
# 用户困惑：为什么需要设置这些？
export XSCHED_PASSTHROUGH_LIBS="libhipblas.so,libMIOpen.so,librocblas.so"

# 如何知道要添加哪些库？
# → 只能通过试错（运行失败 → 添加库 → 重试）
```

**对比理想情况**：
```bash
# 理想：用户无需关心底层细节
python3 train.py  # 直接工作，自动处理
```

### 2.2 深层调用链检测失败

**PyTorch的问题**（我们遇到的）：

```
调用链深度: 7层
Level 0: libshimhip.so (XSched拦截层)
Level 1: libamdhip64.so (原始HIP驱动)
Level 2: rocBLAS 内部函数
Level 3: libhipblaslt.so ← 白名单目标，但检测不到
Level 4: libtorch_hip.so
Level 5: libtorch_cpu.so
Level 6: Python C extension
Level 7: Python 解释器

问题: __builtin_return_address(0) 只能看到 Level 1
     无法识别这是来自 PyTorch 的调用
```

**可能的解决方案（都不理想）**：

| 方案 | 优点 | 缺点 | 可行性 |
|-----|------|------|-------|
| libunwind全栈分析 | 准确 | **巨大性能开销** | ⚠️ 不推荐 |
| 硬编码检查N层 | 简单 | **不灵活，N取多少？** | ⚠️ 有限 |
| 进程名检测 | 快速 | **过于粗糙** | ❌ 不精确 |
| 全局透传模式 | 100%兼容 | **失去调度能力** | ⚠️ 临时方案 |

**结论**：在当前LD_PRELOAD架构下，**没有完美的解决方案**。

---

## 正式集成方案对比

### 3.1 方案对比总览

| 方案 | 实施位置 | 侵入性 | 性能 | 可维护性 | 推荐度 |
|-----|---------|-------|------|---------|-------|
| **A. 当前方案** | 用户空间LD_PRELOAD | ✅ 无侵入 | ⚠️ 中等 | ❌ 差 | ⭐⭐ 临时方案 |
| **B. HIP Runtime层** | libamdhip64.so | ⚠️ 低侵入 | ✅ 好 | ✅ 好 | ⭐⭐⭐⭐ 推荐 |
| **C. ROCm驱动层** | amdgpu内核驱动 | ⚠️⚠️ 中侵入 | ✅✅ 最优 | ✅ 好 | ⭐⭐⭐⭐⭐ 最佳 |
| **D. 应用层API** | PyTorch/TF插件 | ✅ 无侵入 | ⚠️ 中等 | ❌ 差 | ⭐⭐⭐ 特定场景 |

---

### 3.2 方案 A：当前方案（LD_PRELOAD + 白名单）

**架构**：
```
┌─────────────────────────────────────┐
│ 应用层 (PyTorch/TensorFlow)          │
├─────────────────────────────────────┤
│ 深度学习库 (HIPBLAS/MIOpen)          │
├─────────────────────────────────────┤
│ libshimhip.so (LD_PRELOAD)          │  ← 拦截层
│   ├─ 白名单检测                       │  ← Workaround
│   └─ XSched 调度                     │
├─────────────────────────────────────┤
│ HIP Runtime (libamdhip64.so)        │
├─────────────────────────────────────┤
│ ROCm 驱动 (amdgpu.ko)                │
└─────────────────────────────────────┘
```

**优点**：
- ✅ 无需修改 ROCm/HIP
- ✅ 快速原型验证
- ✅ 用户可控（环境变量配置）

**缺点**：
- ❌ 白名单维护成本高
- ❌ 性能开销（每次API调用检测）
- ❌ 深层调用链检测失败
- ❌ 用户体验差（需要配置）

**适用场景**：
- 研究原型
- 概念验证
- **不适合生产环境**

---

### 3.3 方案 B：HIP Runtime 层集成 ⭐⭐⭐⭐ **推荐**

#### 架构设计

```
┌─────────────────────────────────────────────────────────┐
│ 应用层 (PyTorch/TensorFlow)                              │
├─────────────────────────────────────────────────────────┤
│ 深度学习库 (HIPBLAS/MIOpen)                              │
├─────────────────────────────────────────────────────────┤
│ HIP Runtime (libamdhip64.so) ← 集成点                   │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ 调度上下文管理 (Scheduling Context)                  │ │
│ │                                                       │ │
│ │ 每个应用/库有自己的调度域 (Scheduling Domain)         │ │
│ │   ├─ 系统库域 (System Libraries)                     │ │
│ │   │   → HIPBLAS, MIOpen, rocBLAS                    │ │
│ │   │   → 策略: 高优先级, 无抢占                       │ │
│ │   │                                                   │ │
│ │   └─ 应用域 (User Applications)                      │ │
│ │       → 用户 kernel, 训练/推理代码                   │ │
│ │       → 策略: 可调度, 支持抢占                       │ │
│ └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│ ROCm 驱动 (amdgpu.ko)                                    │
└─────────────────────────────────────────────────────────┘
```

#### 核心概念：调度域（Scheduling Domain）

```cpp
// 在 HIP Runtime 中定义
enum hipSchedulingDomain {
    HIP_SCHED_DOMAIN_SYSTEM,      // 系统库，不调度
    HIP_SCHED_DOMAIN_APPLICATION, // 应用代码，可调度
    HIP_SCHED_DOMAIN_AUTO         // 自动检测
};

// 新增 HIP API
hipError_t hipSetSchedulingDomain(hipSchedulingDomain domain);
hipError_t hipGetSchedulingDomain(hipSchedulingDomain* domain);
```

#### 自动域检测机制

```cpp
// HIP Runtime 内部实现
class HipSchedulingContext {
private:
    // 系统库的动态库路径前缀
    static const std::set<std::string> SYSTEM_LIB_PREFIXES = {
        "/opt/rocm/lib/libhipblas",
        "/opt/rocm/lib/libMIOpen",
        "/opt/rocm/lib/librocblas",
        "/opt/rocm/lib/librocfft"
    };
    
public:
    hipSchedulingDomain detectDomain() {
        // 1. 检查是否显式设置
        if (explicitDomain != HIP_SCHED_DOMAIN_AUTO) {
            return explicitDomain;
        }
        
        // 2. 检查调用栈（只在第一次调用时）
        static thread_local hipSchedulingDomain cachedDomain = 
            HIP_SCHED_DOMAIN_AUTO;
        
        if (cachedDomain != HIP_SCHED_DOMAIN_AUTO) {
            return cachedDomain;  // 使用缓存结果
        }
        
        // 3. 分析调用者
        Dl_info info;
        if (dladdr(__builtin_return_address(1), &info)) {
            for (const auto& prefix : SYSTEM_LIB_PREFIXES) {
                if (strstr(info.dli_fname, prefix.c_str())) {
                    cachedDomain = HIP_SCHED_DOMAIN_SYSTEM;
                    return cachedDomain;
                }
            }
        }
        
        // 4. 默认为应用域
        cachedDomain = HIP_SCHED_DOMAIN_APPLICATION;
        return cachedDomain;
    }
};
```

#### API 使用示例

```cpp
// 系统库（如 HIPBLAS）的初始化代码
// ROCm 可以在编译时为系统库自动添加这个调用
hipblasStatus_t hipblasCreate(hipblasHandle_t* handle) {
    // 设置为系统域，所有后续调用都不会被调度
    hipSetSchedulingDomain(HIP_SCHED_DOMAIN_SYSTEM);
    
    // ... 原有的 HIPBLAS 初始化代码
}

// 用户应用代码
int main() {
    // 默认就是应用域，无需手动设置
    // 或者显式设置
    hipSetSchedulingDomain(HIP_SCHED_DOMAIN_APPLICATION);
    
    // 这些调用会被调度系统管理
    hipMalloc(&ptr, size);
    hipLaunchKernel(...);
}
```

#### 优势

1. **无需白名单维护** ✅
   - 系统库在编译时就标记了域
   - 新增库由 ROCm 官方维护
   - 用户无需配置环境变量

2. **性能优化** ✅
   ```cpp
   // 域检测结果会被缓存（thread_local）
   // 每个线程只检测一次，后续调用直接使用缓存
   // 性能开销: ~5-10 cycles (vs 150-650 cycles for whitelist)
   ```

3. **语义清晰** ✅
   ```cpp
   // API 名称直接表达意图
   hipSetSchedulingDomain(HIP_SCHED_DOMAIN_SYSTEM);
   ```

4. **向后兼容** ✅
   ```cpp
   // 默认行为：自动检测
   // 旧代码无需修改即可工作
   ```

#### 实施路径

**阶段 1: HIP Runtime 增强（3-6个月）**
```cpp
// 1. 添加调度域 API
hipError_t hipSetSchedulingDomain(hipSchedulingDomain domain);

// 2. 修改 HIP Runtime 的队列管理
// 在 hipStreamCreate 等函数中添加域检测逻辑

// 3. 集成调度器接口
class HipScheduler {
    virtual void submitKernel(hipStream_t stream, ...);
    virtual void preempt(hipStream_t stream);
    // ...
};
```

**阶段 2: ROCm 系统库更新（并行进行）**
```cpp
// 修改 HIPBLAS, MIOpen, rocBLAS 的初始化函数
// 添加域设置调用（一行代码）
hipblasCreate(...) {
    hipSetSchedulingDomain(HIP_SCHED_DOMAIN_SYSTEM);
    // ...
}
```

**阶段 3: 调度器插件化（扩展性）**
```cpp
// 允许第三方调度器（如 XSched）通过插件接口接入
hipError_t hipRegisterScheduler(HipScheduler* scheduler);
```

---

### 3.4 方案 C：ROCm 驱动层集成 ⭐⭐⭐⭐⭐ **最佳长期方案**

#### 架构设计

```
┌─────────────────────────────────────────────────────────┐
│ 应用层 (PyTorch/TensorFlow)                              │
├─────────────────────────────────────────────────────────┤
│ 深度学习库 (HIPBLAS/MIOpen)                              │
├─────────────────────────────────────────────────────────┤
│ HIP Runtime (libamdhip64.so)                            │
│   ├─ hipStreamCreateWithPriority(stream, priority)      │
│   └─ 调度提示传递给驱动                                  │
├─────────────────────────────────────────────────────────┤
│ ROCm 驱动 (amdgpu.ko) ← 集成点                          │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ GPU 硬件调度器 (HW Scheduler)                        │ │
│ │                                                       │ │
│ │ ┌─────────────┬─────────────┬─────────────┐         │ │
│ │ │ 高优先级队列  │ 中优先级队列  │ 低优先级队列  │         │ │
│ │ │ (System)    │ (Interactive)│ (Batch)     │         │ │
│ │ └─────────────┴─────────────┴─────────────┘         │ │
│ │                                                       │ │
│ │ 抢占控制单元 (Preemption Controller)                  │ │
│ │   ├─ CWSR (Context Save/Restore)                    │ │
│ │   ├─ 中间抢占 (Mid-Wave Preemption)                  │ │
│ │   └─ 公平性控制 (Fairness Control)                   │ │
│ └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│ GPU 硬件 (MI308X, MI300, ...)                            │
└─────────────────────────────────────────────────────────┘
```

#### 核心特性

**1. 硬件级优先级队列**
```c
// 内核驱动实现
struct amdgpu_sched_queue {
    enum queue_priority {
        AMDGPU_PRIORITY_SYSTEM,     // 最高，用于系统库
        AMDGPU_PRIORITY_REALTIME,   // 实时任务
        AMDGPU_PRIORITY_HIGH,       // 高优先级
        AMDGPU_PRIORITY_NORMAL,     // 默认
        AMDGPU_PRIORITY_LOW         // 低优先级/批处理
    };
    
    // 硬件队列 ID
    uint32_t hw_queue_id;
    
    // 抢占控制
    bool preemptible;
    uint32_t quantum_us;  // 时间片（微秒）
};
```

**2. 自动域识别（驱动层）**
```c
// 驱动根据进程/线程上下文自动分配优先级
static enum queue_priority amdgpu_detect_priority(struct task_struct *task) {
    // 1. 检查进程的 cgroup
    if (task_in_cgroup(task, "system.slice")) {
        return AMDGPU_PRIORITY_SYSTEM;
    }
    
    // 2. 检查调用者库（通过用户空间传递的提示）
    if (task->gpu_context->domain == GPU_DOMAIN_SYSTEM) {
        return AMDGPU_PRIORITY_SYSTEM;
    }
    
    // 3. 检查命令队列的显式优先级设置
    if (task->gpu_context->explicit_priority != -1) {
        return task->gpu_context->explicit_priority;
    }
    
    // 4. 默认为普通优先级
    return AMDGPU_PRIORITY_NORMAL;
}
```

**3. 硬件级抢占支持**
```c
// 利用 AMD CWSR (Compute Wave Save/Restore) 机制
static void amdgpu_preempt_low_priority_queue(struct amdgpu_device *adev,
                                                struct amdgpu_sched_queue *high_prio_queue) {
    // 1. 触发 CWSR
    amdgpu_trigger_cwsr(adev, low_priority_queue);
    
    // 2. 保存低优先级队列的上下文到内存
    amdgpu_save_queue_context(low_priority_queue);
    
    // 3. 将 GPU 资源分配给高优先级队列
    amdgpu_schedule_queue(high_prio_queue);
    
    // 4. 低优先级队列挂起，等待恢复
    low_priority_queue->state = QUEUE_PREEMPTED;
}
```

#### 优势

1. **零用户空间开销** ✅✅
   - 域检测和调度决策都在内核驱动中
   - 无需用户空间的额外函数调用
   - 性能开销: <1 cycle

2. **硬件级抢占** ✅✅
   - 利用 GPU 硬件特性（CWSR, mid-wave preemption）
   - 抢占延迟 <100us（相比软件抢占的ms级）
   - 更精细的控制

3. **系统级集成** ✅✅
   - 可以与 Linux cgroup, namespace 集成
   - 支持容器化部署
   - 统一的资源管理

4. **无需应用修改** ✅✅
   - 完全透明
   - 系统库自动享有高优先级
   - 用户应用默认可调度

#### 实施路径

**阶段 1: 驱动增强（6-12个月）**
```c
// 1. 添加多优先级队列支持
// 在 amdgpu_ctx.c, amdgpu_cs.c 中实现

// 2. 实现队列抢占逻辑
// 在 amdgpu_job.c 中添加抢占控制

// 3. 集成 CWSR 机制
// 利用现有的 MI300 CWSR 支持
```

**阶段 2: HIP Runtime 适配（3-6个月，并行）**
```cpp
// HIP 传递调度提示给驱动
hipError_t hipStreamCreateWithPriority(
    hipStream_t* stream,
    unsigned int flags,
    int priority  // ← 新参数
) {
    // 创建带优先级的内核驱动队列
    drm_amdgpu_ctx_create_args args = {
        .priority = priority,
        .flags = AMDGPU_CTX_PRIORITY_EXPLICIT
    };
    ioctl(fd, DRM_IOCTL_AMDGPU_CTX_CREATE, &args);
}
```

**阶段 3: ROCm 生态适配（并行）**
```cpp
// 修改系统库使用高优先级流
hipblasCreate(...) {
    hipStreamCreateWithPriority(&internal_stream, 0, 
                                 HIP_PRIORITY_SYSTEM);
    // ...
}
```

#### 与 NVIDIA 的对比

| 特性 | NVIDIA CUDA | AMD ROCm (建议) |
|-----|-------------|-----------------|
| **优先级队列** | ✅ `cudaStreamCreateWithPriority` | ✅ `hipStreamCreateWithPriority` |
| **抢占支持** | ✅ Pascal+ (compute preemption) | ✅ MI300+ (CWSR) |
| **MPS (多进程服务)** | ✅ CUDA MPS | ⚠️ 可以借鉴 |
| **时间片调度** | ✅ 支持 | ✅ 可实现 |

---

### 3.5 方案 D：应用层API（辅助方案）

**架构**：
```python
# PyTorch 插件方式
import torch
from torch_xsched import XSchedScheduler

# 创建调度器
scheduler = XSchedScheduler(
    policy='priority',
    domains={
        'system': ['hipblas', 'miopen'],  # 系统库
        'user': ['default']                # 用户代码
    }
)

# 启用调度
with scheduler.enabled():
    model(input)  # 自动应用调度
```

**优点**：
- ✅ 无需修改 ROCm
- ✅ 灵活性高

**缺点**：
- ❌ 每个框架都需要单独适配
- ❌ 不够通用
- ❌ 仍需底层支持

**适用场景**：特定应用优化

---

## 驱动层集成的优势与挑战

### 4.1 为什么驱动层是最佳选择？

#### 原因 1: 信息优势

```
驱动层拥有的信息：
┌────────────────────────────────────────┐
│ • 所有GPU队列的状态（空闲/繁忙）        │
│ • 每个队列的优先级                      │
│ • 硬件资源使用情况（SM, memory）       │
│ • 进程/线程上下文信息                   │
│ • cgroup, namespace 信息               │
│ • 历史调度统计                         │
└────────────────────────────────────────┘

用户空间（LD_PRELOAD）只能看到：
┌────────────────────────────────────────┐
│ • 当前进程的HIP API调用                 │
│ • 调用者库名（通过dladdr）              │
│ • ❌ 无法看到其他进程                   │
│ • ❌ 无法看到全局GPU状态                │
│ • ❌ 无法做全局最优调度                 │
└────────────────────────────────────────┘
```

#### 原因 2: 性能

```
性能对比（单次 API 调用开销）:

用户空间白名单:
  dladdr() 调用:           ~200 cycles
  字符串匹配:              ~50 cycles
  dlsym() 查找原始函数:     ~100 cycles
  总计:                    ~350 cycles

HIP Runtime 域缓存:
  thread_local 缓存查询:   ~5 cycles
  域决策:                  ~10 cycles
  总计:                    ~15 cycles
  
驱动层（内核空间）:
  上下文查询:              <1 cycle (已在内存中)
  调度决策:                ~3 cycles
  总计:                    ~4 cycles
  
性能比: 350 : 15 : 4 ≈ 87x : 3.75x : 1x
```

#### 原因 3: 可靠性

```
用户空间方案的脆弱性:
┌────────────────────────────────────────┐
│ • LD_PRELOAD 可能被其他库覆盖           │
│ • dladdr() 在某些情况下失败             │
│ • 符号解析可能冲突                      │
│ • 深层调用链检测失败                    │
└────────────────────────────────────────┘

驱动层的鲁棒性:
┌────────────────────────────────────────┐
│ • 完全控制GPU资源                       │
│ • 不依赖用户空间符号解析                │
│ • 硬件级保证                            │
│ • 不受应用层干扰                        │
└────────────────────────────────────────┘
```

### 4.2 挑战与对策

#### 挑战 1: 开发周期长

**问题**: 内核驱动开发需要6-12个月

**对策**: 分阶段实施
```
Phase 1 (3个月): HIP Runtime层 - 快速原型
  ↓ 验证设计
Phase 2 (6个月): 驱动层基础 - 核心功能
  ↓ 逐步迁移
Phase 3 (3个月): 优化 & 生态集成
```

#### 挑战 2: 兼容性风险

**问题**: 驱动变更可能影响现有应用

**对策**: 
- 默认行为保持不变（向后兼容）
- 新特性通过显式API启用
- 充分的回归测试

#### 挑战 3: 跨版本维护

**问题**: 需要支持多个 ROCm 版本

**对策**:
- 在 HIP Runtime 层提供抽象接口
- 旧驱动降级到用户空间实现
- 新驱动使用硬件加速

---

## 推荐的长期架构

### 5.1 混合架构（短期+长期结合）

```
阶段 1: 当前（2026 Q1）
┌─────────────────────────────────────┐
│ LD_PRELOAD + 白名单 (Workaround)     │  ← 临时方案
│ ✓ 快速验证概念                       │
│ ✓ 收集使用反馈                       │
└─────────────────────────────────────┘

阶段 2: 短期（2026 Q2-Q3）
┌─────────────────────────────────────┐
│ HIP Runtime 域管理                   │  ← 过渡方案
│ ✓ 性能显著提升                       │
│ ✓ 用户体验改善                       │
│ ✓ 向驱动层集成铺路                   │
└─────────────────────────────────────┘

阶段 3: 长期（2026 Q4 - 2027）
┌─────────────────────────────────────┐
│ ROCm 驱动层原生支持                  │  ← 最终方案
│ ✓ 硬件级调度                         │
│ ✓ 零开销                             │
│ ✓ 生态完整集成                       │
└─────────────────────────────────────┘
```

### 5.2 API 演进路径

```cpp
// 阶段 1: 环境变量（当前）
export XSCHED_PASSTHROUGH_LIBS="libhipblas.so,..."
// ↓ 用户需要理解底层细节

// 阶段 2: HIP API
hipSetSchedulingDomain(HIP_SCHED_DOMAIN_SYSTEM);
// ↓ 显式但清晰

// 阶段 3: 自动透明（驱动层）
// 无需任何配置，自动工作
python3 train.py  # Just works™
```

### 5.3 投入产出分析

| 方案 | 开发成本 | 维护成本 | 性能收益 | 用户体验 | ROI |
|-----|---------|---------|---------|---------|-----|
| **LD_PRELOAD** | 1人月 | 高（持续） | ⚠️ 中等 | ❌ 差 | ⭐⭐ |
| **HIP Runtime** | 3-6人月 | 中等 | ✅ 好 | ✅ 好 | ⭐⭐⭐⭐ |
| **驱动层** | 12-18人月 | 低 | ✅✅ 最优 | ✅✅ 最优 | ⭐⭐⭐⭐⭐ |

**结论**: 
- 短期（6个月内）：HIP Runtime 方案是最佳平衡
- 长期（1年+）：驱动层集成是战略目标

---

## 总结与建议

### 关键结论

1. **白名单机制确实是 Workaround** ✅
   - 这是临时的权宜之计，不是长期方案
   - 存在性能、可维护性、可扩展性问题

2. **驱动层集成是正确方向** ✅
   - 性能最优（~87x 快于当前方案）
   - 最可靠（硬件级保证）
   - 最透明（用户无需配置）

3. **但不应直接跳到驱动层** ⚠️
   - 先在 HIP Runtime 层实现是更明智的策略
   - 快速验证设计，收集反馈
   - 为驱动层集成打基础

### 行动建议

#### 立即行动（1个月）
- [ ] 完成当前白名单方案的测试和文档
- [ ] 提交 Issue 给 XSched 社区
- [ ] 与 AMD ROCm 团队建立联系

#### 短期计划（3-6个月）
- [ ] 提议 HIP Runtime 层的域管理机制
- [ ] 开发概念验证原型
- [ ] 编写 ROCm RFC (Request for Comments)

#### 长期规划（1年+）
- [ ] 推动 ROCm 驱动层支持
- [ ] 参与 amdgpu 内核驱动开发
- [ ] 贡献到 upstream Linux kernel

---

**文档版本**: v1.0  
**创建日期**: 2026-01-27  
**下次审查**: 2026-03-01（2个月后评估进展）

