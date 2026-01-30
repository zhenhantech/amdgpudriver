# Stream 优先级代码修复 - TODO List

**关键问题**: HSA Runtime 中 Queue 优先级被硬编码，导致优先级参数无法生效

**创建时间**: 2026-01-29

---

## 🚨 核心问题

### 问题描述

**位置**: `rocr-runtime/core/runtime/amd_aql_queue.cpp` Line 100

**当前代码**:
```cpp
AqlQueue::AqlQueue(core::SharedQueue* shared_queue, GpuAgent* agent, 
                   size_t req_size_pkts, HSAuint32 node_id, 
                   ScratchInfo& scratch, core::HsaEventCallback callback,
                   void* err_data, uint64_t flags)
    : Queue(shared_queue, flags, !agent->is_xgmi_cpu_gpu()),
      LocalSignal(0, false),
      DoorbellSignal(signal()),
      ring_buf_(nullptr),
      ring_buf_alloc_bytes_(0),
      queue_id_(HSA_QUEUEID(-1)),
      active_(false),
      agent_(agent),
      queue_scratch_(scratch),
      errors_callback_(callback),
      errors_data_(err_data),
      pm4_ib_buf_(nullptr),
      pm4_ib_size_b_(0x1000),
      dynamicScratchState(0),
      exceptionState(0),
      suspended_(false),
      priority_(HSA_QUEUE_PRIORITY_NORMAL),  // ⚠️⚠️⚠️ 问题在这里！
      exception_signal_(nullptr) {
    // ...
}
```

**问题**:
- `priority_` 被初始化为 `HSA_QUEUE_PRIORITY_NORMAL` (固定值)
- 构造函数没有接受 `priority` 参数
- **所有 Queue 都是 NORMAL 优先级，无论用户设置什么值！**

### 影响范围

```
用户调用:
  hipStreamCreateWithPriority(&stream_high, 0, -1)  // 期望 HIGH
  hipStreamCreateWithPriority(&stream_low, 0, 1)    // 期望 LOW

实际效果:
  stream_high → priority = NORMAL  ❌
  stream_low  → priority = NORMAL  ❌

最终 MQD:
  Queue-1: cp_hqd_pipe_priority = 1 (NORMAL)  ❌ 应该是 2 (HIGH)
  Queue-2: cp_hqd_pipe_priority = 1 (NORMAL)  ❌ 应该是 0 (LOW)

硬件行为:
  CP/MES 看到两个 Queue 都是相同优先级
  → 无法根据优先级调度！❌
```

---

## 🔧 修复方案

### Step 1: 修改 AqlQueue 构造函数

**文件**: `rocr-runtime/core/runtime/amd_aql_queue.cpp`

#### 1.1 添加 priority 参数

```cpp
// 修改前
AqlQueue::AqlQueue(core::SharedQueue* shared_queue, GpuAgent* agent, 
                   size_t req_size_pkts, HSAuint32 node_id, 
                   ScratchInfo& scratch, core::HsaEventCallback callback,
                   void* err_data, uint64_t flags)

// 修改后
AqlQueue::AqlQueue(core::SharedQueue* shared_queue, GpuAgent* agent, 
                   size_t req_size_pkts, HSAuint32 node_id, 
                   ScratchInfo& scratch, core::HsaEventCallback callback,
                   void* err_data, uint64_t flags,
                   HSA_QUEUE_PRIORITY priority = HSA_QUEUE_PRIORITY_NORMAL)  // ✅ 新增参数
```

#### 1.2 修改初始化列表

```cpp
// 修改前 (Line 100)
      priority_(HSA_QUEUE_PRIORITY_NORMAL),  // ❌ 写死了

// 修改后
      priority_(priority),  // ✅ 使用参数
```

### Step 2: 修改 AqlQueue 头文件

**文件**: `rocr-runtime/core/runtime/amd_aql_queue.h`

```cpp
// 在类定义中找到构造函数声明

// 修改前
AqlQueue(core::SharedQueue* shared_queue, GpuAgent* agent, 
         size_t req_size_pkts, HSAuint32 node_id, 
         ScratchInfo& scratch, core::HsaEventCallback callback,
         void* err_data, uint64_t flags);

// 修改后
AqlQueue(core::SharedQueue* shared_queue, GpuAgent* agent, 
         size_t req_size_pkts, HSAuint32 node_id, 
         ScratchInfo& scratch, core::HsaEventCallback callback,
         void* err_data, uint64_t flags,
         HSA_QUEUE_PRIORITY priority = HSA_QUEUE_PRIORITY_NORMAL);  // ✅ 新增参数
```

### Step 3: 修改 GpuAgent::QueueCreate

**文件**: `rocr-runtime/core/runtime/amd_gpu_agent.cpp`

找到创建 AqlQueue 的地方，传递 priority 参数：

```cpp
// 在 GpuAgent::QueueCreate() 函数中

// 修改前
queue = new AqlQueue(shared_queue, this, req_size_pkts, node_id, 
                     scratch, callback, user_data, flags);

// 修改后
queue = new AqlQueue(shared_queue, this, req_size_pkts, node_id, 
                     scratch, callback, user_data, flags,
                     priority);  // ✅ 传递 priority 参数
```

需要确保 `GpuAgent::QueueCreate()` 接受并正确传递 priority 参数。

### Step 4: 验证 CreateQueue 调用链

确保从 `hsa_queue_create()` 到 `AqlQueue` 构造函数的整个调用链都正确传递 priority：

```
hsa_queue_create(priority, ...)
  ↓
GpuAgent::QueueCreate(priority, ...)
  ↓
new AqlQueue(..., priority)  ✅ 需要添加
  ↓
CreateQueue() [KFD]
  ↓
init_mqd()
  ↓
set_priority()  ← 设置 cp_hqd_pipe_priority
```

---

## 📝 详细修改清单

### 文件 1: `amd_aql_queue.h`

**位置**: `rocr-runtime/core/runtime/amd_aql_queue.h`

```cpp
class AqlQueue : public core::Queue,
                 public core::LocalSignal,
                 public DoorbellSignal {
 public:
  // 修改构造函数声明
  AqlQueue(core::SharedQueue* shared_queue, 
           GpuAgent* agent,
           size_t req_size_pkts, 
           HSAuint32 node_id,
           ScratchInfo& scratch, 
           core::HsaEventCallback callback,
           void* err_data, 
           uint64_t flags,
           HSA_QUEUE_PRIORITY priority = HSA_QUEUE_PRIORITY_NORMAL);  // ✅ 添加
  
  // ... 其他成员函数 ...
};
```

### 文件 2: `amd_aql_queue.cpp`

**位置**: `rocr-runtime/core/runtime/amd_aql_queue.cpp`

#### 修改 1: 构造函数签名 (Line ~81)

```cpp
AqlQueue::AqlQueue(core::SharedQueue* shared_queue, 
                   GpuAgent* agent,
                   size_t req_size_pkts, 
                   HSAuint32 node_id, 
                   ScratchInfo& scratch,
                   core::HsaEventCallback callback, 
                   void* err_data, 
                   uint64_t flags,
                   HSA_QUEUE_PRIORITY priority)  // ✅ 添加参数
```

#### 修改 2: 初始化列表 (Line ~100)

```cpp
    : Queue(shared_queue, flags, !agent->is_xgmi_cpu_gpu()),
      LocalSignal(0, false),
      DoorbellSignal(signal()),
      ring_buf_(nullptr),
      ring_buf_alloc_bytes_(0),
      queue_id_(HSA_QUEUEID(-1)),
      active_(false),
      agent_(agent),
      queue_scratch_(scratch),
      errors_callback_(callback),
      errors_data_(err_data),
      pm4_ib_buf_(nullptr),
      pm4_ib_size_b_(0x1000),
      dynamicScratchState(0),
      exceptionState(0),
      suspended_(false),
      priority_(priority),  // ✅ 修改：使用参数而不是写死的 NORMAL
      exception_signal_(nullptr) {
```

### 文件 3: `amd_gpu_agent.cpp`

**位置**: `rocr-runtime/core/runtime/amd_gpu_agent.cpp`

需要找到 `GpuAgent::QueueCreate()` 函数：

```cpp
hsa_status_t GpuAgent::QueueCreate(size_t size, 
                                   hsa_queue_type32_t queue_type,
                                   core::HsaEventCallback event_callback, 
                                   void* data,
                                   uint32_t private_segment_size,
                                   uint32_t group_segment_size, 
                                   core::Queue** queue,
                                   HSA_QUEUE_PRIORITY priority) {  // ⭐ 确保有 priority 参数
  
  // ... 代码 ...
  
  // 创建 AqlQueue 时传递 priority
  *queue = new AqlQueue(shared_queue, this, size, node_id, 
                        scratch, event_callback, data, flags,
                        priority);  // ✅ 传递 priority
  
  // ... 代码 ...
}
```

### 文件 4: `hsa_ext_amd.cpp` 或类似文件

**可能需要修改的地方**: `hsa_queue_create` 的实现

确保从 HSA API 到 GpuAgent 的调用链正确传递 priority：

```cpp
hsa_status_t hsa_queue_create(...) {
  // ... 代码 ...
  
  // 调用 agent 的 QueueCreate 时传递 priority
  status = agent->QueueCreate(size, type, callback, data, 
                              priv_size, group_size, &queue,
                              priority);  // ✅ 确保传递
  
  // ... 代码 ...
}
```

---

## 🧪 测试验证

### 测试 1: 编译验证

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/rocm-systems
cd projects/rocr-runtime

# 重新编译
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# 安装（或手动复制 .so 文件）
sudo make install
# 或
sudo cp lib/libhsa-runtime64.so.1.* /opt/rocm/lib/
```

### 测试 2: 功能验证

```bash
# 运行测试程序
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/test_stream_priority
./test_concurrent

# 查看 dmesg
sudo dmesg | tail -50 | grep -E "priority|pipe_priority"

# ✅ 期望看到：
# Queue 1001: priority=11, pipe_priority=2 (HIGH)
# Queue 1002: priority=1,  pipe_priority=0 (LOW)

# ❌ 当前看到（修复前）：
# Queue 1001: priority=7, pipe_priority=1 (NORMAL)
# Queue 1002: priority=7, pipe_priority=1 (NORMAL)
```

### 测试 3: 详细日志验证

添加 debug 打印到 `amd_aql_queue.cpp`:

```cpp
AqlQueue::AqlQueue(..., HSA_QUEUE_PRIORITY priority)
    : ...,
      priority_(priority),
      ... {
  
  // ⭐ 添加 debug 打印
  printf("[HSA] AqlQueue::AqlQueue - priority parameter = %d, "
         "priority_ = %d\n", 
         priority, priority_);
  
  // ... 原有代码 ...
}
```

运行测试：

```bash
HIP_VISIBLE_DEVICES=0 ./test_concurrent

# 应该看到：
# [HSA] AqlQueue::AqlQueue - priority parameter = 15, priority_ = 15  (HIGH)
# [HSA] AqlQueue::AqlQueue - priority parameter = 1, priority_ = 1    (LOW)
```

### 测试 4: MQD 验证

如果有 debugfs 支持：

```bash
# 查看 MQD
sudo cat /sys/kernel/debug/kfd/mqds | grep -A 20 "Queue"

# ✅ 期望看到：
# Queue ID: 1001
#   cp_hqd_pipe_priority: 2  (HIGH)
# Queue ID: 1002
#   cp_hqd_pipe_priority: 0  (LOW)
```

---

## 📊 预期效果对比

### 修复前（当前状态）

```
用户代码:
  hipStreamCreateWithPriority(&s1, 0, -1)  // HIGH
  hipStreamCreateWithPriority(&s2, 0, 1)   // LOW

HSA Runtime:
  AqlQueue s1: priority_ = NORMAL (1)  ❌
  AqlQueue s2: priority_ = NORMAL (1)  ❌

KFD MQD:
  Queue 1001: cp_hqd_pipe_priority = 1  ❌
  Queue 1002: cp_hqd_pipe_priority = 1  ❌

硬件行为:
  两个 Queue 相同优先级，无法区分  ❌
```

### 修复后（目标状态）

```
用户代码:
  hipStreamCreateWithPriority(&s1, 0, -1)  // HIGH
  hipStreamCreateWithPriority(&s2, 0, 1)   // LOW

HSA Runtime:
  AqlQueue s1: priority_ = MAXIMUM (15)  ✅
  AqlQueue s2: priority_ = LOW (1)       ✅

KFD MQD:
  Queue 1001: cp_hqd_pipe_priority = 2 (HIGH)  ✅
  Queue 1002: cp_hqd_pipe_priority = 0 (LOW)   ✅

硬件行为:
  高优先级 Queue 优先调度  ✅
  低优先级 Queue 延后调度  ✅
```

---

## 🔍 调试技巧

### 1. 添加 HSA Runtime 日志

在 `amd_aql_queue.cpp` 的关键位置添加打印：

```cpp
// 构造函数
AqlQueue::AqlQueue(..., HSA_QUEUE_PRIORITY priority)
    : ...,
      priority_(priority),
      ... {
  
  fprintf(stderr, "[HSA_DEBUG] AqlQueue created with priority=%d\n", priority_);
}

// SetPriority 函数
hsa_status_t AqlQueue::SetPriority(HSA_QUEUE_PRIORITY priority) {
  fprintf(stderr, "[HSA_DEBUG] SetPriority called: old=%d, new=%d\n", 
          priority_, priority);
  
  priority_ = priority;
  
  // ... 调用 driver UpdateQueue ...
}
```

### 2. 添加 KFD Driver 日志

在 `kfd_mqd_manager_v11.c` 的 `set_priority()` 中添加：

```c
static void set_priority(struct v11_compute_mqd *m, struct queue_properties *q) {
    m->cp_hqd_pipe_priority = pipe_priority_map[q->priority];
    m->cp_hqd_queue_priority = q->priority;
    
    // ⭐ 添加 debug 打印
    pr_info("KFD: set_priority - queue_priority=%u, pipe_priority=%u\n",
            q->priority, m->cp_hqd_pipe_priority);
}
```

### 3. 完整日志链

修复后，运行测试应该看到完整的日志链：

```bash
sudo dmesg -C  # 清空 dmesg
./test_concurrent
sudo dmesg

# 预期日志顺序：
# [HSA_DEBUG] AqlQueue created with priority=15  (HIGH)
# [KFD] set_priority - queue_priority=15, pipe_priority=2
# [HSA_DEBUG] AqlQueue created with priority=1   (LOW)
# [KFD] set_priority - queue_priority=1, pipe_priority=0
```

---

## ✅ 验收标准

修复完成后，需要满足以下标准：

1. **编译通过** ✅
   - HSA Runtime 重新编译成功
   - 没有编译错误或警告

2. **参数传递正确** ✅
   - `hipStreamCreateWithPriority` 的 priority 参数正确传递到 `AqlQueue`
   - `AqlQueue::priority_` 不再固定为 NORMAL

3. **MQD 配置正确** ✅
   - 高优先级 Queue: `cp_hqd_pipe_priority = 2`
   - 低优先级 Queue: `cp_hqd_pipe_priority = 0`

4. **测试程序运行** ✅
   - `test_concurrent` 运行无错误
   - 日志显示不同的优先级值

5. **硬件行为验证** ✅（后续性能测试）
   - 高优先级 kernel 确实优先执行
   - 低优先级 kernel 被延迟执行

---

## 📅 实施计划

### Phase 1: 代码修改（预计 1 天）

- [ ] 修改 `amd_aql_queue.h` - 添加 priority 参数
- [ ] 修改 `amd_aql_queue.cpp` - 使用 priority 参数
- [ ] 修改 `amd_gpu_agent.cpp` - 传递 priority 参数
- [ ] 检查调用链 - 确保 priority 正确传递
- [ ] 添加 debug 日志

### Phase 2: 编译和测试（预计 0.5 天）

- [ ] 重新编译 HSA Runtime
- [ ] 运行基础测试 (`test_concurrent`)
- [ ] 检查 dmesg 日志
- [ ] 验证 MQD 配置

### Phase 3: 深度验证（预计 1 天）

- [ ] 性能测试 - 验证优先级调度效果
- [ ] 多进程测试 - 验证跨进程优先级
- [ ] 压力测试 - 长时间运行稳定性
- [ ] 文档更新 - 记录修复结果

### Phase 4: 不同优先级权限测试 ⭐ 新增

**日期**: 2026-01-30  
**问题来源**: `amd_aql_queue.cpp` Line 100 - queue权限被写死为 `HSA_QUEUE_PRIORITY_NORMAL`

- [ ] 创建测试用例 - 测试不同优先级（HIGH/NORMAL/LOW）的权限行为
  - [ ] 测试 HIGH priority queue 的权限配置
  - [ ] 测试 NORMAL priority queue 的权限配置
  - [ ] 测试 LOW priority queue 的权限配置
  - [ ] 对比不同优先级的 MQD 寄存器差异

- [ ] 验证优先级权限映射
  - [ ] 验证 `pipe_priority_map[]` 的映射关系
  - [ ] 确认 HIGH priority → pipe_priority = 2
  - [ ] 确认 NORMAL priority → pipe_priority = 1
  - [ ] 确认 LOW priority → pipe_priority = 0

- [ ] 硬件调度行为验证
  - [ ] 使用修改后的代码测试不同优先级的实际调度效果
  - [ ] 测量不同优先级 kernel 的执行延迟
  - [ ] 验证高优先级是否真的优先执行
  - [ ] 验证低优先级是否会被延迟

- [ ] 权限相关的性能基准测试
  - [ ] Baseline: 所有 queue 都是 NORMAL (当前状态)
  - [ ] Test 1: HIGH vs NORMAL priority queue
  - [ ] Test 2: HIGH vs LOW priority queue
  - [ ] Test 3: 多个不同优先级 queue 混合场景
  - [ ] 记录和对比各场景的性能数据

**备注**: 
- 这个测试必须在 Phase 1-3 完成后进行
- 需要确保代码修复正确后，再测试不同优先级的权限行为
- 测试结果将用于验证优先级调度系统的正确性

---

## 📚 参考文档

- [PRIORITY_TO_HARDWARE_DEEP_TRACE.md](./PRIORITY_TO_HARDWARE_DEEP_TRACE.md) - 优先级硬件追踪
- [PRIORITY_CPSCH_MODE_TRACE.md](./PRIORITY_CPSCH_MODE_TRACE.md) - CPSCH 模式说明
- [STREAM_PRIORITY_AND_QUEUE_MAPPING.md](./STREAM_PRIORITY_AND_QUEUE_MAPPING.md) - Stream 和 Queue 映射
- [test_stream_priority/README.md](./test_stream_priority/README.md) - 测试程序文档

---

**创建时间**: 2026-01-29  
**状态**: 📝 待实施  
**优先级**: 🔴 高 - 阻塞优先级功能测试  
**预计工作量**: 2.5 天
