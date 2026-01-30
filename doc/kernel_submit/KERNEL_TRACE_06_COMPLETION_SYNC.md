# Kernel提交流程追踪 (6/6) - Kernel完成同步机制

**范围**: Kernel执行完成后的向上同步流程  
**代码路径**: GPU Hardware → KFD → HSA Runtime → HIP Runtime → Application  
**关键机制**: HSA Signal、Event、同步原语

---

## 📋 本层概述

当 GPU 完成 Kernel 执行后，需要通知 CPU 端的应用程序。这个**向上同步**的流程与**向下提交**流程相反，涉及：

1. **GPU Hardware** - Kernel 执行完成，写入完成信号
2. **HSA Signal** - 硬件更新 Signal 值
3. **HSA Runtime** - 等待和检测 Signal 变化
4. **HIP Runtime** - 提供同步 API
5. **Application** - 获得完成通知

---

## 🔄 完整同步流程图

```
GPU Hardware Layer
  └─ Kernel 执行完成
       ↓
  └─ 写入 completion_signal (原子递减)
       ↓
  └─ Signal 内存更新 (CPU 可见)

HSA Runtime Layer
  └─ hsa_signal_wait_scacquire() 等待
       ↓
  └─ 检测 Signal 值变化
       ↓
  └─ 条件满足，返回

HIP Runtime Layer
  └─ hipDeviceSynchronize()
  └─ hipStreamSynchronize()
  └─ hipEventSynchronize()

Application Layer
  └─ 同步函数返回
       ↓
  └─ 可以安全访问结果
```

---

## 1️⃣ GPU 硬件层：完成信号写入

### 1.1 AQL Packet 中的 completion_signal

**文件**: HSA 标准定义

当 Kernel 提交时，AQL Packet 包含一个 `completion_signal` 字段：

```c
typedef struct hsa_kernel_dispatch_packet_s {
    // ... 其他字段 ...
    
    // [Byte 56-63] Completion signal
    hsa_signal_t completion_signal;  // Kernel 完成时写入此信号
    
} hsa_kernel_dispatch_packet_t;
```

**Signal 初始值**:
```c
// HSA Runtime 创建 signal 时设置初始值
hsa_signal_t signal;
hsa_signal_create(1, 0, NULL, &signal);  // 初始值为 1
```

### 1.2 GPU 硬件完成动作

**Kernel 执行完成时**，GPU 硬件会自动：

```
1. 检测到所有 Wave 完成
   ↓
2. 读取 AQL Packet 中的 completion_signal 地址
   ↓
3. 对 Signal 值执行原子递减操作
   signal.value--;  // 从 1 变为 0
   ↓
4. 更新 Signal 内存（CPU 可见）
```

**关键特性**:
- ✅ **硬件自动完成** - 无需驱动参与
- ✅ **原子操作** - 保证线程安全
- ✅ **CPU 可见** - Signal 内存在 CPU/GPU 共享空间

---

## 2️⃣ HSA Runtime 层：Signal 等待机制

### 2.1 hsa_signal_wait_scacquire()

**文件**: `ROCm_keyDriver/rocm-systems/projects/rocr-runtime/src/core/runtime/signal.cpp`

这是 HSA Runtime 提供的核心同步函数：

```c
hsa_signal_value_t hsa_signal_wait_scacquire(
    hsa_signal_t signal,           // 要等待的 signal
    hsa_signal_condition_t condition, // 等待条件 (LT, EQ, GTE等)
    hsa_signal_value_t compare_value, // 比较值 (通常是 0)
    uint64_t timeout_hint,         // 超时时间 (ns)
    hsa_wait_state_hint_t wait_hint   // 等待策略提示
)
{
    // 1. 快速检查：Signal 是否已经满足条件
    hsa_signal_value_t value = hsa_signal_load_relaxed(signal);
    if (signal_condition_met(value, condition, compare_value)) {
        return value;  // 已完成，立即返回
    }
    
    // 2. 等待策略
    switch (wait_hint) {
    case HSA_WAIT_STATE_BLOCKED:
        // 使用事件等待（更节能）
        return wait_blocked(signal, condition, compare_value, timeout_hint);
        
    case HSA_WAIT_STATE_ACTIVE:
        // 使用轮询等待（更低延迟）
        return wait_active(signal, condition, compare_value, timeout_hint);
    }
}
```

### 2.2 等待策略详解

#### 策略 1: Active Wait (活跃等待)

```c
static hsa_signal_value_t wait_active(
    hsa_signal_t signal,
    hsa_signal_condition_t condition,
    hsa_signal_value_t compare_value,
    uint64_t timeout_hint)
{
    uint64_t start_time = get_time_ns();
    
    // 持续轮询 Signal 值
    while (true) {
        // 1. 读取 Signal 当前值
        hsa_signal_value_t value = hsa_signal_load_acquire(signal);
        
        // 2. 检查是否满足条件
        if (signal_condition_met(value, condition, compare_value)) {
            return value;  // 条件满足，返回
        }
        
        // 3. 检查超时
        if (timeout_hint != UINT64_MAX) {
            uint64_t elapsed = get_time_ns() - start_time;
            if (elapsed >= timeout_hint) {
                return value;  // 超时，返回当前值
            }
        }
        
        // 4. 短暂休眠，避免 CPU 100% 占用
        cpu_relax();  // 或 _mm_pause() 在 x86
    }
}
```

**特点**:
- ✅ **低延迟** - 及时检测到完成
- ⚠️ **高 CPU 占用** - 持续轮询消耗 CPU
- 🎯 **适用场景** - 短时间等待（< 1ms）

#### 策略 2: Blocked Wait (阻塞等待)

```c
static hsa_signal_value_t wait_blocked(
    hsa_signal_t signal,
    hsa_signal_condition_t condition,
    hsa_signal_value_t compare_value,
    uint64_t timeout_hint)
{
    // 1. 注册事件
    struct signal_event event;
    event.signal = signal;
    event.condition = condition;
    event.compare_value = compare_value;
    
    // 2. 调用驱动层的事件等待（通过 ioctl）
    // 这会让 CPU 线程进入睡眠状态
    int ret = ioctl(kfd_fd, AMDKFD_IOC_WAIT_EVENTS, &event);
    
    // 3. 被唤醒后读取 Signal 值
    hsa_signal_value_t value = hsa_signal_load_acquire(signal);
    
    return value;
}
```

**特点**:
- ✅ **节能** - CPU 线程睡眠，不占用 CPU
- ⚠️ **高延迟** - 上下文切换开销（~几 μs）
- 🎯 **适用场景** - 长时间等待（> 1ms）

### 2.3 Signal 条件类型

```c
typedef enum {
    HSA_SIGNAL_CONDITION_EQ = 0,   // signal == compare_value
    HSA_SIGNAL_CONDITION_NE = 1,   // signal != compare_value
    HSA_SIGNAL_CONDITION_LT = 2,   // signal <  compare_value (最常用)
    HSA_SIGNAL_CONDITION_GTE = 3   // signal >= compare_value
} hsa_signal_condition_t;
```

**常见用法**:
```c
// 等待 signal 变为 0（表示完成）
hsa_signal_wait_scacquire(
    signal,
    HSA_SIGNAL_CONDITION_LT,  // signal < 1，即 signal == 0
    1,                        // compare_value
    UINT64_MAX,               // 无超时
    HSA_WAIT_STATE_BLOCKED    // 使用阻塞等待
);
```

### 2.4 实际追踪数据验证

从 rocprof 追踪可以看到：

```
hsa_signal_wait_scacquire: 6 次调用，总耗时 493 μs
平均每次: 82 μs
```

**分析**:
- 6 次调用对应不同的同步点（内存拷贝、kernel 执行等）
- 平均 82 μs 表明使用了较高效的等待策略
- 结合 `hsa_signal_load_relaxed` 29 次调用，说明有快速路径检查

---

## 3️⃣ HIP Runtime 层：同步 API

### 3.1 hipDeviceSynchronize()

**文件**: `ROCm_keyDriver/rocm-systems/projects/clr/hipamd/src/hip_device.cpp`

**作用**: 等待设备上的**所有** Stream 完成

```cpp
hipError_t hipDeviceSynchronize()
{
    // 1. 获取当前设备
    hip::Device* device = hip::getCurrentDevice();
    if (device == nullptr) {
        return hipErrorInvalidDevice;
    }
    
    // 2. 等待所有 Stream 完成
    for (auto& stream : device->streams()) {
        if (stream != nullptr) {
            // 对每个 Stream 调用同步
            stream->wait();
        }
    }
    
    return hipSuccess;
}
```

**内部实现**:
```cpp
void Stream::wait()
{
    // 获取 Stream 的最后一个 signal
    hsa_signal_t completion_signal = last_signal_;
    
    // 等待 signal 变为 0
    hsa_signal_wait_scacquire(
        completion_signal,
        HSA_SIGNAL_CONDITION_LT,
        1,
        UINT64_MAX,
        HSA_WAIT_STATE_BLOCKED
    );
}
```

**实际追踪数据**:
```
hipDeviceSynchronize: 1 次调用，29 μs
```

**分析**:
- 29 μs 非常快 → Kernel 可能已经完成，只是确认
- 如果 Kernel 还在执行，时间会更长

### 3.2 hipStreamSynchronize()

**文件**: `ROCm_keyDriver/rocm-systems/projects/clr/hipamd/src/hip_stream.cpp`

**作用**: 等待**指定** Stream 完成

```cpp
hipError_t hipStreamSynchronize(hipStream_t stream)
{
    // 1. 获取 Stream 对象
    hip::Stream* stream_obj = hip::getStream(stream);
    if (stream_obj == nullptr) {
        return hipErrorInvalidResourceHandle;
    }
    
    // 2. 等待该 Stream 完成
    stream_obj->wait();
    
    return hipSuccess;
}
```

**与 hipDeviceSynchronize 的区别**:
- `hipDeviceSynchronize`: 等待**所有** Stream
- `hipStreamSynchronize`: 只等待**一个** Stream
- `hipStreamSynchronize` 更精细，不会阻塞其他 Stream

### 3.3 hipEventSynchronize()

**文件**: `ROCm_keyDriver/rocm-systems/projects/clr/hipamd/src/hip_event.cpp`

**作用**: 等待**指定 Event** 完成

```cpp
hipError_t hipEventSynchronize(hipEvent_t event)
{
    // 1. 获取 Event 对象
    hip::Event* event_obj = hip::getEvent(event);
    if (event_obj == nullptr) {
        return hipErrorInvalidResourceHandle;
    }
    
    // 2. 检查 Event 状态
    if (event_obj->ready()) {
        return hipSuccess;  // 已完成
    }
    
    // 3. 等待 Event 的 signal
    hsa_signal_wait_scacquire(
        event_obj->signal(),
        HSA_SIGNAL_CONDITION_LT,
        1,
        UINT64_MAX,
        HSA_WAIT_STATE_BLOCKED
    );
    
    // 4. 标记 Event 为完成
    event_obj->set_ready(true);
    
    return hipSuccess;
}
```

**Event 的优势**:
- ✅ **更精细的同步** - 可以在 Stream 中插入多个 Event
- ✅ **测量时间** - `hipEventElapsedTime` 可以计算执行时间
- ✅ **跨 Stream 同步** - `hipStreamWaitEvent` 可以让一个 Stream 等待另一个 Stream 的 Event

### 3.4 同步 API 对比

| API | 等待范围 | 使用场景 | 性能影响 |
|-----|---------|---------|---------|
| `hipDeviceSynchronize` | 所有 Stream | 完整的设备同步 | 最大（阻塞所有） |
| `hipStreamSynchronize` | 一个 Stream | 单个 Stream 完成 | 中等 |
| `hipEventSynchronize` | 一个 Event | 精确的操作点 | 最小（最精细） |
| `hipStreamWaitEvent` | Event 点 | 跨 Stream 依赖 | 最小（异步） |

---

## 4️⃣ KFD Driver 层：事件等待支持

### 4.1 AMDKFD_IOC_WAIT_EVENTS ioctl

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_chardev.c`

当 HSA Runtime 使用 Blocked Wait 时，会调用这个 ioctl：

```c
static long kfd_ioctl_wait_events(
    struct file *filep,
    struct kfd_process *p,
    void __user *data)
{
    struct kfd_ioctl_wait_events_args args;
    int ret;
    
    // 1. 从用户空间拷贝参数
    if (copy_from_user(&args, data, sizeof(args))) {
        return -EFAULT;
    }
    
    // 2. 调用事件等待函数
    ret = kfd_wait_on_events(p, args.num_events,
                             (void __user *)args.events_ptr,
                             (args.wait_for_all != 0),
                             &args.timeout);
    
    // 3. 返回结果
    if (copy_to_user(data, &args, sizeof(args))) {
        return -EFAULT;
    }
    
    return ret;
}
```

### 4.2 kfd_wait_on_events() 实现

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_events.c`

```c
int kfd_wait_on_events(struct kfd_process *p,
                       uint32_t num_events,
                       void __user *data,
                       bool wait_all,
                       uint64_t *timeout)
{
    struct kfd_event *events[num_events];
    wait_queue_entry_t wait;
    int ret;
    
    // 1. 获取事件对象
    for (int i = 0; i < num_events; i++) {
        events[i] = lookup_event_by_id(p, event_ids[i]);
    }
    
    // 2. 快速检查：是否已经满足条件
    if (check_events_ready(events, num_events, wait_all)) {
        return 0;  // 已完成，立即返回
    }
    
    // 3. 注册等待队列
    init_wait_entry(&wait, current);
    add_wait_queue(&p->event_waitqueue, &wait);
    
    // 4. 进入睡眠等待
    while (true) {
        set_current_state(TASK_INTERRUPTIBLE);
        
        // 再次检查条件
        if (check_events_ready(events, num_events, wait_all)) {
            break;
        }
        
        // 检查超时
        if (*timeout == 0) {
            ret = -ETIMEDOUT;
            break;
        }
        
        // 睡眠（释放 CPU）
        if (schedule_timeout(*timeout) == 0) {
            ret = -ETIMEDOUT;
            break;
        }
    }
    
    // 5. 清理
    set_current_state(TASK_RUNNING);
    remove_wait_queue(&p->event_waitqueue, &wait);
    
    return ret;
}
```

**关键点**:
- ✅ CPU 线程进入 `TASK_INTERRUPTIBLE` 状态（睡眠）
- ✅ 不占用 CPU 资源
- ✅ 由内核调度器唤醒

### 4.3 事件唤醒机制

**当 GPU 完成 Kernel 时**，KFD 驱动会收到中断：

```c
// GPU 中断处理函数
static irqreturn_t kfd_interrupt_handler(int irq, void *data)
{
    struct kfd_dev *dev = data;
    
    // 1. 读取中断源
    uint32_t ih_ring_entry = read_interrupt_ring();
    
    // 2. 解析中断类型
    if (ih_ring_entry & IH_SIGNAL_COMPLETION) {
        // Signal 完成中断
        
        // 3. 更新 Signal 值（已由硬件完成）
        
        // 4. 唤醒等待的进程
        wake_up_all(&kfd_process->event_waitqueue);
    }
    
    return IRQ_HANDLED;
}
```

**流程**:
```
GPU 完成 Kernel
  ↓
硬件写入 Signal 值
  ↓
触发中断
  ↓
KFD 中断处理函数
  ↓
wake_up_all() 唤醒等待的线程
  ↓
CPU 线程从 schedule_timeout() 返回
  ↓
检查 Signal 值
  ↓
返回用户空间
```

---

## 5️⃣ 完整同步流程示例

### 5.1 代码示例

```cpp
// Application 代码
#include <hip/hip_runtime.h>

__global__ void myKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    float *d_data;
    const int N = 1024;
    
    // 1. 分配内存
    hipMalloc(&d_data, N * sizeof(float));
    
    // 2. 启动 Kernel（异步）
    hipLaunchKernelGGL(myKernel, dim3(4), dim3(256), 0, 0, d_data, N);
    // ↑ 此时返回，Kernel 可能还在执行
    
    // 3. 等待 Kernel 完成（同步点）
    hipDeviceSynchronize();
    // ↑ 阻塞直到 Kernel 完成
    
    // 4. 现在可以安全读取结果
    float h_data[N];
    hipMemcpy(h_data, d_data, N * sizeof(float), hipMemcpyDeviceToHost);
    
    hipFree(d_data);
    return 0;
}
```

### 5.2 详细流程分解

#### 步骤 1: Kernel 启动（异步）

```
Application: hipLaunchKernelGGL()
  ↓
HIP Runtime: 创建 AQL Packet
  ↓
  completion_signal = hsa_signal_create(1)  ← 初始值 1
  packet.completion_signal = completion_signal
  ↓
HSA Runtime: 写入 AQL Queue
  ↓
  *doorbell_ptr = write_index  ← 触发 GPU
  ↓
返回用户空间（Kernel 开始执行，但未完成）
```

#### 步骤 2: 应用继续执行

```
Application: 继续执行后续代码
  ↑
  此时 Kernel 在 GPU 上并行执行
```

#### 步骤 3: 同步调用

```
Application: hipDeviceSynchronize()
  ↓
HIP Runtime: stream->wait()
  ↓
HSA Runtime: hsa_signal_wait_scacquire(signal, LT, 1, ...)
  ↓
  if (signal.value < 1) {  ← 快速检查
      return;  // 已完成
  }
  ↓
  // 进入等待
  while (true) {
      value = hsa_signal_load_acquire(signal);
      if (value < 1) break;  // 检测到完成
      
      // 或者调用 ioctl 进入睡眠
      ioctl(kfd_fd, AMDKFD_IOC_WAIT_EVENTS, ...);
  }
  ↓
阻塞在此（CPU 不执行或睡眠）
```

#### 步骤 4: GPU 完成 Kernel

```
GPU Hardware:
  ↓
  所有 Wave 执行完成
  ↓
  读取 AQL Packet 的 completion_signal
  ↓
  原子操作: signal.value--  (1 → 0)
  ↓
  触发中断（如果使用 Blocked Wait）
```

#### 步骤 5: CPU 检测到完成

```
情况 A (Active Wait):
  HSA Runtime 轮询检测到 signal.value == 0
  ↓
  hsa_signal_wait_scacquire() 返回
  
情况 B (Blocked Wait):
  GPU 中断 → KFD 驱动
  ↓
  wake_up_all(&event_waitqueue)
  ↓
  CPU 线程被唤醒
  ↓
  从 ioctl 返回
  ↓
  检查 signal.value == 0
  ↓
  hsa_signal_wait_scacquire() 返回
```

#### 步骤 6: 返回应用

```
HSA Runtime: hsa_signal_wait_scacquire() 返回
  ↓
HIP Runtime: stream->wait() 返回
  ↓
HIP Runtime: hipDeviceSynchronize() 返回 hipSuccess
  ↓
Application: 继续执行
  ↓
  此时可以安全访问 GPU 计算结果
```

---

## 6️⃣ 实际追踪数据分析

从测试的 rocprof 追踪数据：

### 6.1 Signal 相关 API 调用

```
HSA API                          调用次数  总耗时     平均耗时
─────────────────────────────────────────────────────────────
hsa_signal_create                16      16.98 ms   1061 ns
hsa_amd_signal_create            64      9.82 ms    153 ns
hsa_signal_wait_scacquire        6       493 μs     82 μs    ← 等待
hsa_signal_store_screlease       2       38.7 μs    19 μs
hsa_signal_load_relaxed          29      4.85 μs    167 ns   ← 轮询
hsa_signal_silent_store_relaxed  4       555 ns     138 ns
hsa_signal_destroy               2       1.00 μs    500 ns
```

### 6.2 同步流程分析

**1. Signal 创建**:
- 16 次 `hsa_signal_create` - 创建主要的 Signal
- 64 次 `hsa_amd_signal_create` - AMD 扩展 Signal（用于内部）

**2. Signal 等待**:
- 6 次 `hsa_signal_wait_scacquire` 对应：
  - 3 次内存拷贝完成等待（Host→Device, Device→Host）
  - 1 次 Kernel 执行完成等待
  - 2 次其他操作等待

**3. Signal 轮询**:
- 29 次 `hsa_signal_load_relaxed` - 快速检查 Signal 状态
- 平均 167 ns - 非常快的内存读取

**4. HIP 同步**:
```
hipDeviceSynchronize: 1 次，29 μs
```
- 29 μs 包括：
  - HIP → HSA 的函数调用开销
  - Signal 状态检查
  - 可能的短暂等待

### 6.3 时间开销分析

```
总执行时间: ~158 ms

内存传输    136 ms  (86%)   ← 主要开销
HSA Setup    21 ms  (13%)   
同步等待    0.52 ms (<1%)   ← Signal wait
Kernel 执行  0.01 ms (<0.01%) ← GPU 很快
```

**关键发现**:
- ✅ 同步开销很小（< 1%）
- ✅ 大部分等待是在内存传输完成
- ✅ Kernel 执行本身很快（13.52 μs）

---

## 7️⃣ 高级同步模式

### 7.1 异步模式 - 使用 Event

```cpp
hipEvent_t start, stop;
hipEventCreate(&start);
hipEventCreate(&stop);

// 在 Stream 中插入 Event
hipEventRecord(start, stream);
hipLaunchKernelGGL(myKernel, ..., stream, ...);
hipEventRecord(stop, stream);

// CPU 可以继续执行其他工作
do_other_work();

// 只在需要时同步
hipEventSynchronize(stop);

// 获取执行时间
float milliseconds = 0;
hipEventElapsedTime(&milliseconds, start, stop);
printf("Kernel time: %f ms\n", milliseconds);
```

**优势**:
- ✅ CPU 和 GPU 并行工作
- ✅ 可以测量精确的执行时间
- ✅ 更灵活的同步控制

### 7.2 多 Stream 并行

```cpp
hipStream_t stream1, stream2;
hipStreamCreate(&stream1);
hipStreamCreate(&stream2);

// 两个 Kernel 在不同 Stream 中并行执行（理想情况）
hipLaunchKernelGGL(kernel1, ..., stream1, ...);
hipLaunchKernelGGL(kernel2, ..., stream2, ...);

// 可以选择等待特定 Stream
hipStreamSynchronize(stream1);  // 只等待 stream1

// 或等待所有
hipDeviceSynchronize();  // 等待 stream1 和 stream2
```

**⚠️ 多进程场景下的注意事项**:

根据实际研究（详见 [KERNEL_TRACE_STREAM_MANAGEMENT.md 第 3.4 节](./KERNEL_TRACE_STREAM_MANAGEMENT.md#34-多进程场景下的-stream-到-queue-映射问题-)），在多进程场景下，**多个进程的 Stream 可能映射到同一个底层 Queue**，这会影响同步行为：

```
【理想情况】
进程1 Stream 1 → Queue 1 → hipStreamSynchronize(stream1) 只等待进程1
进程2 Stream 1 → Queue 2 → hipStreamSynchronize(stream1) 只等待进程2

【实际情况（可能）】
进程1 Stream 1 → Queue 1 ─┐
进程2 Stream 1 → Queue 1 ─┤ → 共享 Queue
                           ↓
hipStreamSynchronize(stream1) 可能等待两个进程的任务！
```

**影响**:
- 同步时间可能比预期更长
- 一个进程的 Stream 同步可能被另一个进程的任务阻塞
- 性能测量结果可能不准确

### 7.3 跨 Stream 依赖

```cpp
hipEvent_t event;
hipEventCreate(&event);

// Stream 1: 执行 kernel1
hipLaunchKernelGGL(kernel1, ..., stream1, ...);
hipEventRecord(event, stream1);

// Stream 2: 等待 stream1 的 event，然后执行 kernel2
hipStreamWaitEvent(stream2, event, 0);
hipLaunchKernelGGL(kernel2, ..., stream2, ...);

// CPU 不阻塞，两个 Stream 自动同步
```

---

## 8️⃣ 性能优化建议

### 8.1 减少同步开销

**问题**: 频繁的 `hipDeviceSynchronize()` 降低性能

**解决方案**:
```cpp
// ❌ 不好：频繁同步
for (int i = 0; i < N; i++) {
    hipLaunchKernelGGL(kernel, ...);
    hipDeviceSynchronize();  // 每次都等待
}

// ✅ 好：批量执行后一次同步
for (int i = 0; i < N; i++) {
    hipLaunchKernelGGL(kernel, ...);
}
hipDeviceSynchronize();  // 一次等待全部完成
```

### 8.2 使用异步 API

```cpp
// ✅ 使用异步拷贝，让 CPU 和 GPU 并行
hipMemcpyAsync(d_data, h_data, size, hipMemcpyHostToDevice, stream);
hipLaunchKernelGGL(kernel, ..., stream, ...);
hipMemcpyAsync(h_result, d_result, size, hipMemcpyDeviceToHost, stream);

// CPU 可以做其他工作
do_cpu_work();

// 只在需要结果时同步
hipStreamSynchronize(stream);
```

### 8.3 使用 Event 而不是 Synchronize

```cpp
// ✅ 更精细的同步
hipEvent_t event;
hipEventCreate(&event);

hipLaunchKernelGGL(kernel, ..., stream, ...);
hipEventRecord(event, stream);

// CPU 继续工作
do_other_work();

// 只等待这个 Event
hipEventSynchronize(event);  // 比 hipDeviceSynchronize 更精细
```

---

## 9️⃣ 调试和追踪

### 9.1 使用 rocprof 追踪同步

```bash
# 追踪 HSA API（包括 signal 操作）
rocprof --hsa-trace ./your_program

# 查看 signal wait 调用
grep "hsa_signal_wait" trace_rocprof.hsa_stats.csv
```

### 9.2 使用 HIP 回调

```cpp
// 注册 Stream 完成回调
void HIPRT_CB myCallback(hipStream_t stream, hipError_t status, void* userData) {
    printf("Stream completed! Status: %d\n", status);
}

hipLaunchKernelGGL(kernel, ..., stream, ...);
hipStreamAddCallback(stream, myCallback, NULL, 0);
```

### 9.3 检查同步开销

```cpp
// 测量同步时间
auto start = std::chrono::high_resolution_clock::now();

hipDeviceSynchronize();

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
printf("Synchronize time: %ld us\n", duration.count());
```

---

## 🎯 关键要点总结

### 同步机制核心

1. **HSA Signal**
   - ✅ 硬件自动更新
   - ✅ CPU/GPU 共享内存
   - ✅ 原子操作保证线程安全

2. **等待策略**
   - ✅ Active Wait: 低延迟，高 CPU 占用
   - ✅ Blocked Wait: 节能，高延迟

3. **HIP API 层次**
   - `hipDeviceSynchronize`: 全局同步
   - `hipStreamSynchronize`: Stream 级同步
   - `hipEventSynchronize`: 事件级同步

4. **性能考虑**
   - ✅ 减少同步频率
   - ✅ 使用异步 API
   - ✅ 使用 Event 精细控制

### 与提交流程的对比

| 方向 | 路径 | 关键机制 | 延迟 |
|------|------|---------|------|
| **下行（提交）** | App → HIP → HSA → KFD → GPU | Doorbell 写入 | 极低（无系统调用） |
| **上行（同步）** | GPU → Signal → HSA → HIP → App | Signal 等待 | 可配置（Active/Blocked） |

---

## 📖 参考文档

- [KERNEL_TRACE_01_APP_TO_HIP.md](./KERNEL_TRACE_01_APP_TO_HIP.md) - 提交流程
- [KERNEL_TRACE_02_HSA_RUNTIME.md](./KERNEL_TRACE_02_HSA_RUNTIME.md) - HSA Queue 和 Signal
- [HSA 标准](https://www.hsafoundation.com/standards/) - Signal 规范

---

**下一步**: 查看完整的端到端流程图 [KERNEL_TRACE_INDEX.md](./KERNEL_TRACE_INDEX.md)

