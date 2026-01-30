# HIP Stream 管理机制详解

**专题文档**: HIP Stream 的概念、实现和管理  
**代码路径**: `ROCm_keyDriver/rocm-systems/projects/clr/hipamd/`  
**相关文档**: [KERNEL_TRACE_01_APP_TO_HIP.md](./KERNEL_TRACE_01_APP_TO_HIP.md)

---

## 📋 文档概述

本文档深入讲解 HIP Stream 的管理机制，包括：
1. Stream 的概念和作用
2. Stream 的创建和生命周期
3. Stream 与 HSA Queue 的映射关系
4. 默认 Stream vs 用户创建的 Stream
5. Stream 的同步机制
6. 多 Stream 并发执行原理
7. Stream 的底层实现代码追踪

---

## 1️⃣ Stream 概念

### 1.1 什么是 Stream？

**Stream（流）** 是 GPU 编程中的一个重要概念，表示一个**命令执行序列**。

```
传统CPU执行:
  命令1 → 命令2 → 命令3  (串行执行)

GPU Stream执行:
  Stream 1: Kernel A → Memcpy D → Kernel B  (串行)
  Stream 2: Kernel C → Memcpy E → Kernel D  (串行)
  ↓
  Stream 1 和 Stream 2 可以并发执行！（前提是映射到不同的底层Queue）
```

**关键特性**:
- ✅ 同一个 Stream 中的操作**按顺序执行**
- ✅ 不同 Stream 中的操作**可以并发执行**（需要映射到不同的底层Queue）
- ✅ 提供了细粒度的并发控制

**⚠️ 重要发现**：基于实际研究（详见下文 "Stream 到 Queue 的映射关系"），多个 HIP Stream（即使在不同进程中）**可能映射到同一个底层 KFD Queue**，这会导致这些 Stream 中的任务**串行执行**而非并发执行，造成性能瓶颈。

### 1.2 Stream 的作用

**1. 并发执行多个 Kernel**:
```cpp
// 串行执行（使用默认stream）
kernel1<<<grid, block>>>(data1);
kernel2<<<grid, block>>>(data2);  // 等待kernel1完成

// 并发执行（使用不同stream）
kernel1<<<grid, block, 0, stream1>>>(data1);
kernel2<<<grid, block, 0, stream2>>>(data2);  // 可以与kernel1并发
```

**2. 隐藏数据传输延迟**:
```cpp
// 重叠计算和数据传输
hipMemcpyAsync(d_data1, h_data1, size, ..., stream1);  // 传输数据
kernel<<<..., stream2>>>(d_data2);                     // 同时执行kernel
```

**3. 流水线执行**:
```cpp
// Batch处理的流水线
for (int i = 0; i < N; i++) {
    hipMemcpyAsync(d_data, h_data[i], ..., streams[i % 2]);
    kernel<<<..., streams[i % 2]>>>(d_data);
    hipMemcpyAsync(h_result[i], d_result, ..., streams[i % 2]);
}
```

### 1.3 Stream 的类型

**默认 Stream (Null Stream)**:
```cpp
// 以下操作都在默认stream中执行
kernel<<<grid, block>>>(data);
hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
```

**用户创建的 Stream**:
```cpp
hipStream_t stream;
hipStreamCreate(&stream);

kernel<<<grid, block, 0, stream>>>(data);

hipStreamDestroy(stream);
```

---

## 2️⃣ Stream 的创建和管理

### 2.1 hipStreamCreate() 实现

**文件**: `ROCm_keyDriver/rocm-systems/projects/clr/hipamd/src/hip_stream.cpp`

```cpp
hipError_t hipStreamCreate(hipStream_t* stream) {
    return hipStreamCreateWithFlags(stream, hipStreamDefault);
}

hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags) {
    HIP_INIT_API(hipStreamCreateWithFlags, stream, flags);
    
    if (stream == nullptr) {
        return hipErrorInvalidValue;
    }
    
    // 1. 获取当前设备
    hip::Device* device = hip::getCurrentDevice();
    if (device == nullptr) {
        return hipErrorInvalidDevice;
    }
    
    // 2. 创建 Stream 对象
    hip::Stream* hip_stream = new hip::Stream(device, flags);
    if (hip_stream == nullptr) {
        return hipErrorOutOfMemory;
    }
    
    // 3. 初始化 Stream
    hipError_t err = hip_stream->initialize();
    if (err != hipSuccess) {
        delete hip_stream;
        return err;
    }
    
    // 4. 返回 stream handle
    *stream = reinterpret_cast<hipStream_t>(hip_stream);
    
    return hipSuccess;
}
```

**Stream Flags**:
```cpp
// Stream创建标志
#define hipStreamDefault       0x00  // 默认stream行为
#define hipStreamNonBlocking   0x01  // 非阻塞stream
```

### 2.2 Stream 类的定义

**文件**: `ROCm_keyDriver/rocm-systems/projects/clr/hipamd/src/hip_stream.hpp`

```cpp
namespace hip {

class Stream {
public:
    // 构造函数
    Stream(Device* device, unsigned int flags);
    
    // 析构函数
    ~Stream();
    
    // 初始化
    hipError_t initialize();
    
    // Kernel启动
    hipError_t launchKernel(hipFunction_t func,
                           const KernelParams& params);
    
    // 内存操作
    hipError_t memcpy(void* dst, const void* src, size_t size,
                     hipMemcpyKind kind);
    
    // 同步操作
    hipError_t synchronize();
    hipError_t query();
    
    // Event操作
    hipError_t recordEvent(hipEvent_t event);
    hipError_t waitEvent(hipEvent_t event);
    
    // 获取底层HSA Queue
    hsa_queue_t* getHsaQueue();
    
private:
    Device* device_;               // 所属设备
    unsigned int flags_;           // Stream标志
    bool is_default_;              // 是否是默认stream
    
    // 底层HSA Queue
    hsa_queue_t* hsa_queue_;       // HSA Queue指针
    bool queue_created_;           // Queue是否已创建
    
    // 同步机制
    std::vector<hipEvent_t> events_;  // 关联的events
    
    // 互斥锁
    std::mutex lock_;
    
    // 其他
    bool valid_;
};

} // namespace hip
```

### 2.3 Stream 初始化

**文件**: `ROCm_keyDriver/rocm-systems/projects/clr/hipamd/src/hip_stream.cpp`

```cpp
hipError_t Stream::initialize() {
    // 1. 检查设备是否有效
    if (device_ == nullptr) {
        return hipErrorInvalidDevice;
    }
    
    // 2. 如果是默认stream，可能延迟创建HSA Queue
    // 否则立即创建HSA Queue
    if (!is_default_) {
        hipError_t err = createHsaQueue();
        if (err != hipSuccess) {
            return err;
        }
    }
    
    // 3. 标记为有效
    valid_ = true;
    
    return hipSuccess;
}
```

### 2.4 创建 HSA Queue

**文件**: `ROCm_keyDriver/rocm-systems/projects/clr/hipamd/src/hip_stream.cpp`

```cpp
hipError_t Stream::createHsaQueue() {
    // 1. 检查是否已创建
    if (queue_created_) {
        return hipSuccess;
    }
    
    // 2. 调用 HSA Runtime 创建 Queue
    hsa_agent_t agent = device_->getHsaAgent();
    
    hsa_status_t status = hsa_queue_create(
        agent,                          // GPU agent
        1024,                          // Queue大小（1024个packet）
        HSA_QUEUE_TYPE_MULTI,          // Queue类型（多生产者）
        nullptr,                       // 回调函数
        nullptr,                       // 回调数据
        UINT32_MAX,                    // 私有段大小（使用默认）
        UINT32_MAX,                    // 组段大小（使用默认）
        &hsa_queue_                    // 输出Queue指针
    );
    
    if (status != HSA_STATUS_SUCCESS) {
        return hipErrorOutOfMemory;
    }
    
    // 3. 标记为已创建
    queue_created_ = true;
    
    return hipSuccess;
}
```

---

## 3️⃣ Stream 与 HSA Queue 的映射

### 3.1 一对一映射关系

```
HIP Stream (软件抽象)
    ↓ 1:1 映射
HSA Queue (用户空间队列)
    ↓ 硬件访问
AQL Queue (内存中的packet数组)
    ↓ Doorbell通知
GPU 调度器
    ├─ MES (Micro-Engine Scheduler) - 硬件调度器
    │  适用: CDNA3 (MI300A/X), CDNA2 (MI250X/MI210), RDNA3 (RX 7900)
    │
    └─ CPSCH (Compute Process Scheduler) - 软件调度器
       适用: MI308X (ALDEBARAN), MI100, Vega, RDNA2
    ↓
GPU执行
```

**关键理解**:
- ✅ 每个 HIP Stream 对应一个 HSA Queue
- ✅ HSA Queue 是实际的硬件队列
- ✅ 多个 Stream = 多个 HSA Queue = 可以并发执行
- ⚠️ 调度器类型取决于 GPU 架构（MES 或 CPSCH）

**调度器差异**:

| 特性 | MES 调度器 | CPSCH 调度器 |
|------|-----------|-------------|
| 类型 | 硬件调度器 | 软件调度器 |
| 队列访问 | 直接通过 Doorbell | 可能经过驱动层 |
| 延迟 | 更低 | 相对较高 |
| 适用架构 | 新架构（CDNA3+, RDNA3+） | 旧架构和特定型号 |
| 检查方式 | `cat /sys/module/amdgpu/parameters/mes` | 1=MES, 0=CPSCH |

### 3.2 默认 Stream 的特殊处理

**延迟创建**:
```cpp
// 默认stream在首次使用时才创建HSA Queue
Stream* default_stream = device->getDefaultStream();

// 第一次使用
kernel<<<grid, block>>>(data);  
// ↓ 触发 default_stream->launchKernel()
// ↓ 检测到 hsa_queue_ == nullptr
// ↓ 调用 createHsaQueue()
// ↓ 创建 HSA Queue
```

**默认 Stream 获取**:
```cpp
// 文件: hip_device.cpp
Stream* Device::getDefaultStream() {
    // 线程安全的单例模式
    if (default_stream_ == nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (default_stream_ == nullptr) {
            default_stream_ = new Stream(this, 
                                        hipStreamDefault | 
                                        hipStreamNonBlocking);
            default_stream_->is_default_ = true;
        }
    }
    return default_stream_;
}
```

### 3.3 Queue 大小和类型选择

**Queue 大小**:
```cpp
// 用户stream通常创建较小的queue
#define USER_STREAM_QUEUE_SIZE 1024    // 1024个packet

// 默认stream可能创建更大的queue
#define DEFAULT_STREAM_QUEUE_SIZE 4096 // 4096个packet
```

**Queue 类型**:
```cpp
enum hsa_queue_type {
    HSA_QUEUE_TYPE_MULTI,      // 多生产者队列（常用）
    HSA_QUEUE_TYPE_SINGLE,     // 单生产者队列（优化）
};

// 多个CPU线程可能同时提交到同一个stream
// 因此通常使用 MULTI 类型
```

### 3.4 多进程场景下的 Stream 到 Queue 映射问题 ⚠️

**重要发现**：基于实际研究（参考：`/mnt/md0/zhehan/code/rampup_doc/2PORC_streams/doc/0113_KFD_QUEUE_ANALYSIS.md`），在多进程场景下，发现了一个严重的性能问题。

#### 3.4.1 理想映射 vs 实际映射

**理想情况（预期的 1:1:1 映射）**:
```
进程1:
  HIP Stream 1 (0x11586c0) → HSA Queue → KFD Queue ID 0 (独立)
  HIP Stream 2 (0x1889540) → HSA Queue → KFD Queue ID 1 (独立)

进程2:
  HIP Stream 1 (0x22a16c0) → HSA Queue → KFD Queue ID 2 (独立)
  HIP Stream 2 (0x22c7620) → HSA Queue → KFD Queue ID 3 (独立)

→ 所有 Stream 并发执行 ✅
```

**实际情况（多个 Stream 映射到同一个 Queue）**:
```
进程1:
  HIP Stream 1 (0x11586c0) → HSA Queue → KFD Queue ID 0 (独立) ✅
  HIP Stream 2 (0x1889540) → HSA Queue → KFD Queue ID 1 (共享) ❌

进程2:
  HIP Stream 1 (0x22a16c0) → HSA Queue → KFD Queue ID 0 (独立) ✅
  HIP Stream 2 (0x22c7620) → HSA Queue → KFD Queue ID 1 (共享) ❌

→ Stream 2 串行执行！性能下降！❌
```

#### 3.4.2 实验数据

**测试环境**:
- 4 个进程，每个进程创建 2 个自定义 Stream + 1 个 Default Stream
- GPU: AMD MI308X (ALDEBARAN, CPSCH 模式)

**HIP Runtime 层面（用户空间）**:

| 进程 PID | Default Stream | Custom Stream 1 | Custom Stream 2 |
|---------|---------------|----------------|----------------|
| 6669    | NULL (0)      | 0x11586c0      | 0x1889540      |
| 6671    | NULL (0)      | 0x22a16c0      | 0x22c7620      |
| 6673    | NULL (0)      | 0x7bb6c0       | 0xe0f030       |
| 6675    | NULL (0)      | 0x246f6c0      | 0x2b6dce0      |

**观察**: ✅ 每个进程有独立的 Stream 对象（地址不同）

**KFD Queue 层面（内核空间）**:

| 进程 PID | Queue ID 0 | Queue ID 1 | Queue ID 2 |
|---------|-----------|-----------|-----------|
| 1991338 | ✅ 独立   | ⚠️ 共享   | ⚠️ 共享   |
| 1991342 | ✅ 独立   | ⚠️ 共享   | ⚠️ 共享   |
| 1991349 | ✅ 独立   | ⚠️ 共享   | ⚠️ 共享   |
| 1991353 | ✅ 独立   | ⚠️ 共享   | ⚠️ 共享   |

**发现**: ❌ Queue ID 1 和 2 被 4 个进程共享！

#### 3.4.3 性能影响机制

```
【串行化瓶颈】

进程1 的 Stream 2 提交 Job A → KFD Queue ID 1
进程2 的 Stream 2 提交 Job B → KFD Queue ID 1  ← 同一个 Queue！
进程3 的 Stream 2 提交 Job C → KFD Queue ID 1  ← 同一个 Queue！
进程4 的 Stream 2 提交 Job D → KFD Queue ID 1  ← 同一个 Queue！

硬件层面执行顺序:
  Job A (进程1) → Job B (进程2) → Job C (进程3) → Job D (进程4)
  
结果: 串行执行，无法并发！
```

**性能数据**:
- **单进程 QPS**: 107-116
- **4进程 QPS**: 59.0
- **性能损失**: 50%+

#### 3.4.4 根本原因分析

**可能的原因**:

1. **Queue ID 分配策略问题**
   - Queue ID 可能是进程内的局部索引（0, 1, 2, ...）
   - 而非全局唯一的 ID
   - 导致不同进程的相同索引映射到相同的硬件队列

2. **HSA Queue 创建时的资源复用**
   - HSA Runtime 可能为了节省资源，复用已存在的 Queue
   - 特别是对于某些特定类型的 Queue（如 Utility Queue）

3. **KFD Driver 的 Queue 管理策略**
   - KFD 在创建 Queue 时可能检查是否已有相同属性的 Queue
   - 如果存在，直接返回现有 Queue 的 ID

**文件位置**（需要进一步研究）:
- `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_device_queue_manager.c`
  - `pqm_create_queue()` - Queue 创建逻辑
  - Queue ID 的分配机制

#### 3.4.5 验证方法

**方法 1: KFD 驱动追踪**
```bash
# 在 KFD 驱动中添加 trace_printk
# 文件: kfd_chardev.c
trace_printk("CREATE_QUEUE: PID=%d, Queue_ID=%u, Queue_Type=%d\n",
             current->pid, args->queue_id, args->queue_type);

# 查看追踪
sudo cat /sys/kernel/debug/tracing/trace | grep CREATE_QUEUE
```

**方法 2: HIP Stream 追踪**
```bash
# 使用 LD_PRELOAD 拦截 HIP API
LD_PRELOAD=./libhip_stream_wrapper.so ./test_program

# 追踪 hipStreamCreate 调用
# 记录 Stream 指针和对应的 Queue ID
```

**方法 3: 对比测试**
```bash
# 测试 1: 使用 Default Stream（每进程独立）
./test_program --use-default-stream  # QPS: 107-116

# 测试 2: 使用自定义 Stream（可能共享）
./test_program --use-custom-streams  # QPS: 59.0

# 差异 → 验证 Queue 共享问题
```

#### 3.4.6 未解决的问题

**需要进一步研究的问题**:

1. ❓ **为什么 Queue ID 0 是独立的，而 Queue ID 1、2 是共享的？**
   - Queue ID 0 可能对应 Default Stream，有特殊处理
   - Queue ID 1、2 可能对应其他类型的 Queue（如 SDMA Queue）

2. ❓ **Queue ID 的分配逻辑在哪里？**
   - 需要深入分析 `pqm_create_queue()` 的实现
   - 需要理解 Queue ID 是局部还是全局

3. ❓ **如何确保每个进程的每个 Stream 映射到独立的 Queue？**
   - 是否需要修改 Queue 分配策略
   - 是否需要在 Queue ID 中加入进程标识

4. ❓ **这个问题是否与 CPSCH vs MES 有关？**
   - MI308X 使用 CPSCH（软件调度）
   - 新架构使用 MES（硬件调度）
   - 可能两种模式的 Queue 管理策略不同

#### 3.4.7 临时解决方案

**方案 1: 只使用 Default Stream**
```cpp
// 不创建自定义 Stream，全部使用 Default Stream
// 优点: Default Stream 是进程独立的
// 缺点: 无法使用多 Stream 并发优化

kernel1<<<grid, block>>>(data);  // 使用 Default Stream
kernel2<<<grid, block>>>(data);  // 使用 Default Stream
```

**方案 2: 使用不同的 Queue 类型**
```cpp
// 尝试创建不同类型或优先级的 Stream
int priority_high = -1;
int priority_low = 1;
hipStreamCreateWithPriority(&stream1, 0, priority_high);
hipStreamCreateWithPriority(&stream2, 0, priority_low);

// 可能会被分配到不同的 Queue
```

**方案 3: 进程级隔离**
```cpp
// 使用环境变量或其他机制
// 确保不同进程使用不同的 GPU 或不同的 Queue 池
export HIP_VISIBLE_DEVICES=0  # 进程1
export HIP_VISIBLE_DEVICES=1  # 进程2
```

#### 3.4.8 参考资料

- **研究文档**: `/mnt/md0/zhehan/code/rampup_doc/2PORC_streams/doc/0113_KFD_QUEUE_ANALYSIS.md`
- **测试日志**: `/mnt/md0/zhehan/code/rampup_doc/2PORC_streams/log/kfd_queue_test/`
- **相关代码**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_device_queue_manager.c`

**结论**: 多进程场景下的 Stream 到 Queue 映射问题是一个**实际存在的性能瓶颈**，需要深入研究 KFD 的 Queue 分配机制，并可能需要修改以确保每个进程的每个 Stream 都映射到独立的硬件队列。

#### 3.4.9 后续研究：Queue ID 分配优化的实施与结果 ⚠️⚠️

**重要更新**（基于后续深入研究）：

##### 🔧 已实施的优化

基于上述发现，研究团队实施了 Queue ID 分配优化：

**优化方案** (v5 版本):
```c
// 文件: kfd_process_queue_manager.c
static int find_available_queue_slot(struct process_queue_manager *pqm,
                    unsigned int *qid)
{
    unsigned long found;
    pid_t pid = pqm->process->lead_thread->pid;
    unsigned int process_index;
    unsigned int base_queue_id;
    unsigned int queues_per_process = 4; // 每个进程默认 4 个队列
    
    // 根据进程 PID 计算 Queue ID 范围
    process_index = pid % (KFD_MAX_NUM_OF_QUEUES_PER_PROCESS / queues_per_process);
    base_queue_id = process_index * queues_per_process;
    
    // 在该进程的 Queue ID 范围内分配
    for (found = base_queue_id; found < base_queue_id + queues_per_process; found++) {
        if (!test_bit(found, pqm->queue_slot_bitmap)) {
            set_bit(found, pqm->queue_slot_bitmap);
            *qid = found;
            return 0;
        }
    }
    
    // 如果范围已满，回退到全局搜索
    // ...
}
```

**优化效果**:
- ✅ 不同进程确实使用了不同的 Queue ID 范围
- ✅ 没有 Queue ID 共享问题
- ✅ 技术实现完全符合预期

##### 📊 性能测试结果

**但性能并没有显著提升**：

| 测试场景 | v4 (未优化) | v5 (Queue ID 优化) | 性能提升 |
|---------|-----------|------------------|---------|
| 2-PROC  | ~72.0 QPS | ~99.75 QPS      | +38.5%  |
| 4-PROC  | ~58.5 QPS | ~58.5 QPS       | 0%      |
| 6-PROC  | ~60.5 QPS | ~60.5 QPS       | 0%      |

**关键发现**:
- ⚠️ 2-PROC 有一定提升（38.5%），但仍远低于理想值（应接近 200%）
- ⚠️ 4-PROC 及以上几乎无提升
- ⚠️ 性能在 4-PROC 后趋于稳定（~60 QPS），说明存在更深层的瓶颈

##### 🔍 根本原因分析

**为什么 Queue ID 优化没有解决性能问题？**

1. **Queue ID 到硬件队列（ACE）的映射问题**:
   ```
   软件层面（已优化）:
   进程1: Queue ID 0-3   ✅ 独立
   进程2: Queue ID 4-7   ✅ 独立
   进程3: Queue ID 8-11  ✅ 独立
   
   硬件层面（仍有问题）:
   Queue ID 0-3  → ACE 0  ❌ 可能映射到同一个硬件队列
   Queue ID 4-7  → ACE 0  ❌ 可能映射到同一个硬件队列
   Queue ID 8-11 → ACE 0  ❌ 可能映射到同一个硬件队列
   ```
   - 即使 Queue ID 不同，它们可能映射到**同一个硬件队列（ACE）**
   - MI308X 有 32 个 ACE，但映射策略可能导致多个 Queue ID 共享同一个 ACE

2. **GPU 资源饱和**:
   ```
   单进程 CU 限制测试:
   - CU=80: 119.0 QPS (基准)
   - CU=70: 85.0 QPS (-28.3%)  ← 远超线性预期（-12.5%）
   - CU=60: 79.0 QPS (-33.3%)
   
   结论: 单进程 seq=500 时已经充分利用了 80 个 CUs
   ```
   - 测试负载可能不是"小"负载，GPU 已经接近饱和
   - 多进程竞争有限的 CU 资源，导致性能下降

3. **其他瓶颈**:
   - **Doorbell 层面**: 不同进程的 doorbell 可能映射到相同的硬件 doorbell
   - **内存带宽**: GPU 内存带宽成为瓶颈
   - **调度器串行化**: CPSCH（软件调度器）可能存在串行化问题

##### 🎯 关键洞察

**Queue ID 层面的优化是必要的，但不充分**:

```
问题层次：
┌─────────────────────────────────────┐
│ 应用层: HIP Stream                   │ ✅ 每个进程独立
├─────────────────────────────────────┤
│ KFD 层: Queue ID                    │ ✅ v5 已优化，每个进程独立
├─────────────────────────────────────┤
│ 硬件层: ACE 映射                     │ ❌ 可能仍然共享
├─────────────────────────────────────┤
│ 硬件层: CU 资源                      │ ❌ 已经饱和
├─────────────────────────────────────┤
│ 硬件层: 内存带宽                     │ ❌ 可能成为瓶颈
└─────────────────────────────────────┘
```

**解决一层问题不够，需要解决所有层的问题**。

##### 📚 后续尝试的优化方向

研究团队还尝试了多个其他优化方向：

| 优化方向 | 方法 | 结果 | 结论 |
|---------|------|------|------|
| **MES 调度器** | 启用 MES 硬件调度 | ❌ 失败 | MI308X 不支持 MES |
| **CU_MASK** | 为不同进程分配不同的 CU | ⚠️ 部分有效 | 对 6-PROC 有帮助，对 2-PROC 无效 |
| **active_runlist** | 移除串行化检查 | ❌ 无效 | 不是主要瓶颈 |
| **调度器性能监控** | 添加性能计数器 | ✅ 有效识别 | 调度器本身不是瓶颈 |

**关键发现**: 经过多次优化尝试，发现：
- ✅ 调度器本身不是瓶颈（无阻塞、无锁竞争）
- ✅ CU 竞争是部分瓶颈，但不是主要瓶颈
- ⚠️ 主要瓶颈在于 **GPU 资源本身已经饱和**

##### ⚠️ 重要警告和建议

**对于多进程应用开发者**:

1. **不要盲目假设多 Stream = 高性能**
   - 即使创建了独立的 Stream，底层可能仍然共享资源
   - 需要通过实际测试验证性能提升

2. **考虑工作负载大小**
   - 如果单进程已经充分利用 GPU，多进程不会带来性能提升
   - 只有在工作负载较小时，多进程才可能提升性能

3. **验证方法**
   ```bash
   # 1. 测试不同进程数的性能
   ./test_1proc  # 基准性能
   ./test_2proc  # 应接近 2× 单进程
   ./test_4proc  # 应接近 4× 单进程
   
   # 2. 如果多进程性能下降，考虑：
   #    - 减小工作负载大小
   #    - 使用 Default Stream（每个进程独立）
   #    - 避免创建过多自定义 Stream
   ```

4. **架构差异**
   - **CPSCH（MI308X）**: 软件调度，可能存在更多串行化问题
   - **MES（MI300A/X, MI250X）**: 硬件调度，理论上并行度更好
   - 在不同架构上测试可能得到不同结果

##### 📖 参考资料

详细的研究过程和数据参见：
- `/mnt/md0/zhehan/code/rampup_doc/2PORC_streams/doc/0113_V5_PERFORMANCE_ANALYSIS.md` - v5 性能分析
- `/mnt/md0/zhehan/code/rampup_doc/2PORC_streams/doc/0113_QUEUE_ID_ALLOCATION_OPTIMIZATION.md` - Queue ID 优化实现
- `/mnt/md0/zhehan/code/rampup_doc/2PORC_streams/doc/0116_OPTIMIZATION_SUMMARY.md` - 完整优化总结

---

## 4️⃣ Kernel 在 Stream 中的启动

### 4.1 hipLaunchKernel 中的 Stream 处理

**文件**: `ROCm_keyDriver/rocm-systems/projects/clr/hipamd/src/hip_module.cpp`

```cpp
hipError_t hipLaunchKernel(const void* hostFunction,
                           dim3 gridDim,
                           dim3 blockDim, 
                           void** args,
                           size_t sharedMemBytes,
                           hipStream_t stream) {
    // 1. 获取 stream 对象
    hip::Stream* hip_stream;
    
    if (stream == nullptr || stream == 0) {
        // 使用默认stream
        hip_stream = hip::getCurrentDevice()->getDefaultStream();
    } else {
        // 使用用户指定的stream
        hip_stream = reinterpret_cast<hip::Stream*>(stream);
        
        // 验证stream有效性
        if (!hip_stream->isValid()) {
            return hipErrorInvalidHandle;
        }
    }
    
    // 2. 从hostFunction获取kernel信息
    hipFunction_t func = hip::getFunc(hostFunction);
    if (func == nullptr) {
        return hipErrorInvalidDeviceFunction;
    }
    
    // 3. 准备kernel参数
    hip::KernelParams params;
    params.gridDim = gridDim;
    params.blockDim = blockDim;
    params.sharedMemBytes = sharedMemBytes;
    params.args = args;
    
    // 4. 调用stream的launchKernel方法
    return hip_stream->launchKernel(func, params);
}
```

### 4.2 Stream::launchKernel() 实现

**文件**: `ROCm_keyDriver/rocm-systems/projects/clr/hipamd/src/hip_stream.cpp`

```cpp
hipError_t Stream::launchKernel(hipFunction_t func, 
                               const KernelParams& params) {
    std::lock_guard<std::mutex> lock(lock_);
    
    // 1. 确保HSA Queue已创建
    if (hsa_queue_ == nullptr) {
        hipError_t err = createHsaQueue();
        if (err != hipSuccess) {
            return err;
        }
    }
    
    // 2. 准备 AQL Dispatch Packet
    hsa_kernel_dispatch_packet_t packet;
    memset(&packet, 0, sizeof(packet));
    
    // 3. 填充 packet
    prepareDispatchPacket(func, params, &packet);
    
    // 4. 提交 packet 到 HSA Queue
    return submitPacketToHsaQueue(hsa_queue_, &packet);
}
```

### 4.3 并发执行示例

**多 Stream 并发（理想情况）**:
```cpp
// 创建两个stream
hipStream_t stream1, stream2;
hipStreamCreate(&stream1);
hipStreamCreate(&stream2);

// 在stream1中启动kernel1
kernel1<<<grid, block, 0, stream1>>>(data1);
// ↓ 写入stream1的HSA Queue
// ↓ 写入stream1的doorbell

// 在stream2中启动kernel2
kernel2<<<grid, block, 0, stream2>>>(data2);
// ↓ 写入stream2的HSA Queue
// ↓ 写入stream2的doorbell

// 两个kernel可以并发执行！（前提是映射到不同的底层Queue）
// GPU调度器（MES或CPSCH）会从两个queue读取packet并调度
```

**⚠️ 实际情况警告**:

根据实际研究（详见 3.4 节），在多进程场景下，**多个 Stream 可能映射到同一个底层 KFD Queue**，导致：

```
【理想情况】
Stream 1 → Queue 1 ─┐
Stream 2 → Queue 2 ─┤ → 并发执行 ✅
Stream 3 → Queue 3 ─┘

【实际情况】
Stream 1 (进程1) → Queue 1 ─┐
Stream 2 (进程1) → Queue 2 ─┤ → 共享 Queue 2，串行执行 ❌
Stream 3 (进程2) → Queue 2 ─┘
```

**硬件层面的并发（理想情况）**:
```
MES/CPSCH 硬件调度器:
  ↓ 检测到 stream1 的 doorbell 更新
  ↓ 从 stream1 的 queue 读取 packet
  ↓ 调度 kernel1 到 CU0-CU7
  
  同时...
  
  ↓ 检测到 stream2 的 doorbell 更新
  ↓ 从 stream2 的 queue 读取 packet
  ↓ 调度 kernel2 到 CU8-CU15

两个kernel在不同的CU上并发执行！
```

**硬件层面的串行化（实际问题）**:
```
当多个 Stream 映射到同一个 Queue 时:

Queue 2 (被多个 Stream 共享):
  ↓ Job A (Stream 2, 进程1)
  ↓ Job B (Stream 3, 进程2)  ← 必须等待 Job A 完成
  ↓ Job C (Stream 4, 进程3)  ← 必须等待 Job B 完成

结果: 串行执行，性能下降 50%+
```

**验证建议**:
```cpp
// 在多进程场景下测试
// 检查是否真正实现了并发

#include <sys/types.h>
#include <unistd.h>

printf("Process PID: %d\n", getpid());
printf("Stream 1: %p\n", stream1);
printf("Stream 2: %p\n", stream2);

// 通过 KFD 追踪验证 Queue ID
// 确保不同进程的 Stream 映射到不同的 Queue
```

---

## 5️⃣ Stream 同步机制

### 5.1 hipStreamSynchronize() 实现

**文件**: `ROCm_keyDriver/rocm-systems/projects/clr/hipamd/src/hip_stream.cpp`

```cpp
hipError_t hipStreamSynchronize(hipStream_t stream) {
    HIP_INIT_API(hipStreamSynchronize, stream);
    
    // 1. 获取stream对象
    hip::Stream* hip_stream;
    if (stream == nullptr || stream == 0) {
        hip_stream = hip::getCurrentDevice()->getDefaultStream();
    } else {
        hip_stream = reinterpret_cast<hip::Stream*>(stream);
    }
    
    // 2. 调用stream的同步方法
    return hip_stream->synchronize();
}
```

**Stream::synchronize() 实现**:
```cpp
hipError_t Stream::synchronize() {
    // 1. 如果queue未创建，说明没有提交过任何操作
    if (hsa_queue_ == nullptr) {
        return hipSuccess;
    }
    
    // 2. 创建一个completion signal
    hsa_signal_t signal;
    hsa_status_t status = hsa_signal_create(1, 0, nullptr, &signal);
    if (status != HSA_STATUS_SUCCESS) {
        return hipErrorOutOfMemory;
    }
    
    // 3. 提交一个barrier packet
    // barrier packet会等待前面所有packet完成
    submitBarrierPacket(&signal);
    
    // 4. 等待signal变为0
    hsa_signal_value_t value = hsa_signal_wait_acquire(
        signal,
        HSA_SIGNAL_CONDITION_LT,  // 条件：小于
        1,                        // 比较值：1
        UINT64_MAX,               // 无限等待
        HSA_WAIT_STATE_BLOCKED    // 阻塞等待
    );
    
    // 5. 销毁signal
    hsa_signal_destroy(signal);
    
    return hipSuccess;
}
```

### 5.2 Barrier Packet

**文件**: `ROCm_keyDriver/rocm-systems/projects/clr/hipamd/src/hip_stream.cpp`

```cpp
hipError_t Stream::submitBarrierPacket(hsa_signal_t* completion_signal) {
    // 1. 准备 Barrier Packet
    hsa_barrier_and_packet_t barrier;
    memset(&barrier, 0, sizeof(barrier));
    
    // 2. 设置 header
    barrier.header = 
        (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
        (1 << HSA_PACKET_HEADER_BARRIER) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
    
    // 3. 设置 completion signal
    barrier.completion_signal = *completion_signal;
    
    // 4. 获取写指针
    uint64_t write_index = hsa_queue_add_write_index_relaxed(hsa_queue_, 1);
    
    // 5. 计算packet位置
    const uint32_t queueMask = hsa_queue_->size - 1;
    uint32_t packet_index = write_index & queueMask;
    
    // 6. 获取packet地址
    hsa_barrier_and_packet_t* queue_barrier = 
        &((hsa_barrier_and_packet_t*)hsa_queue_->base_address)[packet_index];
    
    // 7. 写入barrier packet
    memcpy((uint8_t*)queue_barrier + sizeof(barrier.header),
           (uint8_t*)&barrier + sizeof(barrier.header),
           sizeof(barrier) - sizeof(barrier.header));
    
    // 8. 内存屏障
    __atomic_thread_fence(__ATOMIC_RELEASE);
    
    // 9. 写入header（激活packet）
    __atomic_store_n(&queue_barrier->header, barrier.header, __ATOMIC_RELEASE);
    
    // 10. 写入doorbell
    hsa_signal_store_relaxed(hsa_queue_->doorbell_signal, write_index);
    
    return hipSuccess;
}
```

**Barrier Packet 的作用**:
```
Queue中的packet顺序:
  Packet 1: Kernel A
  Packet 2: Kernel B
  Packet 3: Memcpy
  Packet 4: Barrier  ← 等待前面所有packet完成
  Packet 5: Kernel C

当GPU调度器（MES或CPSCH）处理到Barrier时:
  - 等待Packet 1-3全部完成
  - 更新Barrier的completion_signal
  - 继续处理Packet 5
```

### 5.3 hipStreamQuery() 实现

**非阻塞查询**:
```cpp
hipError_t hipStreamQuery(hipStream_t stream) {
    hip::Stream* hip_stream = /* 获取stream */;
    
    if (hip_stream->hsa_queue_ == nullptr) {
        return hipSuccess;  // 没有操作，已完成
    }
    
    // 读取queue的read/write指针
    uint64_t read_index = hip_stream->hsa_queue_->read_dispatch_id;
    uint64_t write_index = hip_stream->hsa_queue_->write_dispatch_id;
    
    if (read_index == write_index) {
        return hipSuccess;  // 所有packet已处理完
    } else {
        return hipErrorNotReady;  // 还有packet未处理
    }
}
```

---

## 6️⃣ Stream 销毁

### 6.1 hipStreamDestroy() 实现

**文件**: `ROCm_keyDriver/rocm-systems/projects/clr/hipamd/src/hip_stream.cpp`

```cpp
hipError_t hipStreamDestroy(hipStream_t stream) {
    HIP_INIT_API(hipStreamDestroy, stream);
    
    if (stream == nullptr || stream == 0) {
        return hipErrorInvalidHandle;
    }
    
    // 1. 获取stream对象
    hip::Stream* hip_stream = reinterpret_cast<hip::Stream*>(stream);
    
    // 2. 不能销毁默认stream
    if (hip_stream->isDefault()) {
        return hipErrorInvalidHandle;
    }
    
    // 3. 同步stream（等待所有操作完成）
    hipError_t err = hip_stream->synchronize();
    if (err != hipSuccess) {
        return err;
    }
    
    // 4. 销毁HSA Queue
    if (hip_stream->hsa_queue_ != nullptr) {
        hsa_queue_destroy(hip_stream->hsa_queue_);
        hip_stream->hsa_queue_ = nullptr;
    }
    
    // 5. 删除stream对象
    delete hip_stream;
    
    return hipSuccess;
}
```

### 6.2 Stream 析构函数

```cpp
Stream::~Stream() {
    // 1. 销毁HSA Queue（如果还未销毁）
    if (hsa_queue_ != nullptr && queue_created_) {
        hsa_queue_destroy(hsa_queue_);
        hsa_queue_ = nullptr;
    }
    
    // 2. 清理events
    events_.clear();
    
    // 3. 标记为无效
    valid_ = false;
}
```

---

## 7️⃣ 高级特性

### 7.1 Stream Priority

**创建带优先级的 Stream**:
```cpp
hipError_t hipStreamCreateWithPriority(hipStream_t* stream,
                                       unsigned int flags,
                                       int priority) {
    // priority: 
    //   -1: 高优先级
    //    0: 正常优先级（默认）
    //   +1: 低优先级
    
    hip::Stream* hip_stream = new hip::Stream(device, flags);
    hip_stream->setPriority(priority);
    
    // 优先级会影响GPU调度器（MES或CPSCH）的调度决策
    // 高优先级的stream会更快地获得GPU资源
    
    *stream = reinterpret_cast<hipStream_t>(hip_stream);
    return hipSuccess;
}
```

### 7.2 Stream Callback

**在 Stream 中插入回调函数**:
```cpp
hipError_t hipStreamAddCallback(hipStream_t stream,
                                hipStreamCallback_t callback,
                                void* userData,
                                unsigned int flags) {
    // callback会在stream中所有之前提交的操作完成后执行
    
    hip::Stream* hip_stream = /* 获取stream */;
    
    // 1. 提交一个barrier packet
    // 2. 在barrier完成后，在CPU线程中调用callback
    
    return hip_stream->addCallback(callback, userData);
}
```

### 7.3 Stream Wait Event

**让 Stream 等待 Event**:
```cpp
hipError_t hipStreamWaitEvent(hipStream_t stream,
                              hipEvent_t event,
                              unsigned int flags) {
    // stream会等待event完成后才继续处理后续操作
    
    hip::Stream* hip_stream = /* 获取stream */;
    hip::Event* hip_event = /* 获取event */;
    
    // 在stream中插入一个wait操作
    return hip_stream->waitEvent(hip_event);
}
```

**实现原理**:
```cpp
hipError_t Stream::waitEvent(Event* event) {
    // 1. 获取event的signal
    hsa_signal_t event_signal = event->getSignal();
    
    // 2. 提交一个barrier packet，依赖于event的signal
    hsa_barrier_and_packet_t barrier;
    barrier.dep_signal[0] = event_signal;  // 依赖信号
    
    // 3. 提交barrier
    submitBarrierPacket(&barrier);
    
    return hipSuccess;
}
```

---

## 8️⃣ 性能考虑

### 8.1 Stream 数量的选择

**太少的 Stream**:
```cpp
// 只用默认stream
kernel1<<<grid, block>>>(data1);  // 串行
kernel2<<<grid, block>>>(data2);  // 等待kernel1

// 问题：无法利用GPU的并行能力
```

**太多的 Stream**:
```cpp
// 创建过多stream
hipStream_t streams[1000];
for (int i = 0; i < 1000; i++) {
    hipStreamCreate(&streams[i]);
    kernel<<<..., streams[i]>>>(data);
}

// 问题：
// 1. 每个stream对应一个HSA Queue，消耗内存
// 2. GPU调度器（MES或CPSCH）管理开销增加
// 3. 实际并发度受限于GPU硬件资源
```

**合理的 Stream 数量**:
```cpp
// 根据实际并发需求和硬件能力
// 通常2-8个stream就足够

int num_streams = 4;
hipStream_t streams[num_streams];

for (int i = 0; i < num_streams; i++) {
    hipStreamCreate(&streams[i]);
}

// 循环使用
for (int i = 0; i < N; i++) {
    kernel<<<..., streams[i % num_streams]>>>(data[i]);
}
```

### 8.2 Stream 同步的开销

**频繁同步的问题**:
```cpp
// 不好的做法
for (int i = 0; i < N; i++) {
    kernel<<<..., stream>>>(data[i]);
    hipStreamSynchronize(stream);  // 每次都同步！
}
// 问题：失去了异步执行的优势
```

**批量提交，减少同步**:
```cpp
// 好的做法
for (int i = 0; i < N; i++) {
    kernel<<<..., stream>>>(data[i]);
}
hipStreamSynchronize(stream);  // 最后同步一次
```

### 8.3 Default Stream 的注意事项

**Default Stream 的同步行为**:
```cpp
// 在某些HIP版本中，default stream是同步的
kernel1<<<grid, block>>>(data1);        // 默认stream
kernel2<<<grid, block, 0, stream1>>>(data2);  // 用户stream

// kernel2可能需要等待kernel1完成！
// 这是为了兼容CUDA的行为
```

**使用 hipStreamNonBlocking 避免**:
```cpp
hipStream_t stream;
hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);

// 这个stream不会与default stream同步
kernel1<<<grid, block>>>(data1);        // 默认stream
kernel2<<<grid, block, 0, stream>>>(data2);  // 可以并发
```

---

## 9️⃣ 调试和诊断

### 9.1 查看 Stream 信息

**使用环境变量**:
```bash
export AMD_LOG_LEVEL=5
export HIP_TRACE_API=1

./your_app
```

**日志示例**:
```
[HIP] hipStreamCreate(stream=0x7f8000001000)
[HIP] Created HSA queue: addr=0x7f8000002000, size=1024
[HIP] hipLaunchKernel(stream=0x7f8000001000)
[HIP] Submit to queue: wptr=5
[HIP] hipStreamSynchronize(stream=0x7f8000001000)
[HIP] Queue sync: rptr=5, wptr=5
```

### 9.2 检查 Stream 状态

**在代码中检查**:
```cpp
// 检查stream是否完成
hipError_t status = hipStreamQuery(stream);
if (status == hipSuccess) {
    printf("Stream completed\n");
} else if (status == hipErrorNotReady) {
    printf("Stream still busy\n");
}

// 获取stream的优先级
int priority;
hipStreamGetPriority(stream, &priority);
printf("Stream priority: %d\n", priority);
```

### 9.3 常见问题

**问题1: Stream 未并发执行**
```cpp
// 可能原因：
// 1. GPU资源不足（CU数量有限）
// 2. 使用了default stream
// 3. Kernel太小，启动开销大于执行时间
// 4. 内存带宽饱和
```

**问题2: Stream 同步卡住**
```cpp
// 可能原因：
// 1. Kernel出错（如非法内存访问）
// 2. Queue损坏
// 3. GPU调度器（MES或CPSCH）异常

// 调试方法：
// 1. 设置超时时间
// 2. 检查kernel错误
// 3. 使用ROCgdb调试
```

---

## 🔟 总结

### 10.1 Stream 层次结构

```
应用层：
  hipStream_t (handle)
    ↓
HIP Runtime层：
  hip::Stream (C++对象)
    ↓
HSA Runtime层：
  hsa_queue_t (HSA Queue)
    ↓
内核驱动层：
  kfd_process_device + queue
    ↓
硬件层：
  GPU调度器
    ├─ MES (Micro-Engine Scheduler) - 新架构
    └─ CPSCH (Compute Process Scheduler) - 旧架构/特定型号
```

### 10.2 关键代码位置

| 功能 | 文件路径 | 关键函数 |
|------|---------|---------|
| Stream创建 | `clr/hipamd/src/hip_stream.cpp` | `hipStreamCreate()` |
| Stream类定义 | `clr/hipamd/src/hip_stream.hpp` | `class Stream` |
| Kernel启动 | `clr/hipamd/src/hip_stream.cpp` | `Stream::launchKernel()` |
| Stream同步 | `clr/hipamd/src/hip_stream.cpp` | `Stream::synchronize()` |
| HSA Queue创建 | `clr/hipamd/src/hip_stream.cpp` | `Stream::createHsaQueue()` |
| Barrier提交 | `clr/hipamd/src/hip_stream.cpp` | `Stream::submitBarrierPacket()` |

### 10.3 最佳实践

1. **合理使用 Stream 数量**：根据实际并发需求，通常2-8个
2. **批量提交操作**：减少同步次数
3. **使用非阻塞 Stream**：避免与default stream同步
4. **注意资源清理**：及时销毁不用的stream
5. **考虑硬件限制**：并发度受GPU资源限制

---

## 相关文档

- [KERNEL_TRACE_01_APP_TO_HIP.md](./KERNEL_TRACE_01_APP_TO_HIP.md) - 应用层到HIP Runtime
- [KERNEL_TRACE_02_HSA_RUNTIME.md](./KERNEL_TRACE_02_HSA_RUNTIME.md) - HSA Runtime层
- [KERNEL_TRACE_INDEX.md](./KERNEL_TRACE_INDEX.md) - 总览文档


