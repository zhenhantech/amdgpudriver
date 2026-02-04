# 架构更新 v2.3: 纯软件调度设计

**日期**: 2026-01-30  
**更新版本**: v2.3 / v3.2  
**关键发现**: amd_aql_queue.cpp:100 priority 写死验证

---

## 🔍 关键发现

### 代码验证

```c
// 文件: rocr-runtime/core/runtime/amd_aql_queue.cpp:100
// 构造函数

AqlQueue::AqlQueue(...) {
  ...
  priority_(HSA_QUEUE_PRIORITY_NORMAL),  // ⚠️ 写死！
  ...
}
```

**这意味着什么？**

```
┌──────────────────────────────────────────────────────────────────┐
│ HIP 应用层                                                        │
│ • hipStreamCreateWithPriority(&stream, flags, priority)          │
│ • priority = 3 (训练) or 12 (推理)                               │
│ • HIP 层保留了 priority 信息 ✅                                  │
└─────────────────┬────────────────────────────────────────────────┘
                  ↓ 内部调用 hsa API
┌──────────────────────────────────────────────────────────────────┐
│ ROCm Runtime (AqlQueue) 层                                        │
│ • AqlQueue::AqlQueue(...) {                                      │
│     priority_(HSA_QUEUE_PRIORITY_NORMAL),  ⚠️ Line 100          │
│   }                                                               │
│ • 向 KFD 传递时，所有 queue 优先级都是 NORMAL                   │
└─────────────────┬────────────────────────────────────────────────┘
                  ↓ ioctl(AMDKFD_IOC_CREATE_QUEUE)
┌──────────────────────────────────────────────────────────────────┐
│ KFD 驱动层                                                        │
│ • struct kfd_ioctl_create_queue_args {                           │
│     .queue_priority = HSA_QUEUE_PRIORITY_NORMAL  ← 所有相同     │
│   }                                                               │
│ • struct queue {                                                  │
│     .properties.priority = NORMAL  ← 所有相同                    │
│   }                                                               │
└─────────────────┬────────────────────────────────────────────────┘
                  ↓ init_mqd() / load_mqd()
┌──────────────────────────────────────────────────────────────────┐
│ GPU 固件/硬件层                                                   │
│ • MQD.cp_hqd_queue_priority = NORMAL  ← 所有相同                │
│ • GPU 固件不区分 queue 优先级                                    │
│ • GPU 调度行为对所有 queue 一致                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🎯 架构调整

### 从硬件优先级 → 纯软件优先级调度

**原设计（v2.1/v3.1）:**
```
✗ 假设：利用 AMD 16 级硬件优先级
✗ queue.properties.priority = 0-15 (传递给 GPU)
✗ GPU 固件基于优先级做硬件调度
```

**新设计（v2.3/v3.2）:**
```
✓ 发现：所有 queue 的 hardware_priority = NORMAL
✓ 采用：纯软件优先级调度 ⭐⭐⭐
✓ 实现：
  • queue.logical_priority = HIP 设置 (软件使用)
  • queue.hardware_priority = NORMAL (硬件固定)
  • 软件层监控、检测、触发 CWSR
```

---

## 🏗️ 实现方案

### 方案 1: KFD 层调度（推荐）⭐⭐⭐

#### 步骤 1: 扩展 ioctl 参数传递 logical_priority

**修改 kfd_ioctl.h:**
```c
struct kfd_ioctl_create_queue_args {
    uint64_t ring_base_address;
    uint64_t write_pointer_address;
    uint64_t read_pointer_address;
    // ... 现有字段 ...
    
    // ⭐⭐⭐ v2.3 新增：逻辑优先级
    uint32_t logical_priority;  // HIP 设置的逻辑优先级
                                 // 0-15 或更多（软件定义）
};
```

#### 步骤 2: ROCm Runtime 传递 logical_priority

**修改 amd_aql_queue.cpp:**
```cpp
// 文件: rocr-runtime/core/runtime/amd_aql_queue.cpp

AqlQueue::AqlQueue(
    core::Agent* agent,
    size_t req_size_pkts,
    HSAuint32 node_id,
    ScratchInfo& scratch,
    core::HsaEventCallback callback,
    void* err_data,
    bool is_kv,
    int hip_stream_priority)  // ← 新增参数：从 HIP 传递的优先级
  : ...
    priority_(HSA_QUEUE_PRIORITY_NORMAL),  // ← 硬件优先级固定
    logical_priority_(hip_stream_priority), // ← 新增：逻辑优先级
    ... {
  
  // 创建 queue 时传递给 KFD
  struct kfd_ioctl_create_queue_args args = {
    ...
    .queue_priority = HSA_QUEUE_PRIORITY_NORMAL,  // ← 硬件优先级固定
    .logical_priority = logical_priority_,         // ← 新增：传递逻辑优先级
    ...
  };
  
  int ret = ioctl(kfd_fd, AMDKFD_IOC_CREATE_QUEUE, &args);
  ...
}
```

#### 步骤 3: KFD 存储和使用 logical_priority

**修改 kfd_priv.h:**
```c
struct queue {
    // ===== 现有字段 =====
    struct queue_properties properties;  // .priority = NORMAL (硬件)
    struct mqd_manager *mqd_mgr;
    void *mqd;
    // ...
    
    // ⭐⭐⭐ v2.3 新增：纯软件调度字段
    int logical_priority;     // HIP 设置的逻辑优先级
    int effective_priority;   // 动态优先级（基于 logical_priority 计算）
    
    // Ring Buffer 监控
    uint32_t hw_rptr;
    uint32_t hw_wptr;
    uint32_t pending_count;
    bool is_active;
    
    // 抢占状态
    enum queue_state state;
    bool preemption_pending;
    // ...
};
```

**修改 kfd_queue.c 创建逻辑:**
```c
int kfd_create_queue(..., struct kfd_ioctl_create_queue_args *args) {
    struct queue *q = kzalloc(sizeof(*q), GFP_KERNEL);
    
    // 设置硬件优先级（固定 NORMAL）
    q->properties.priority = HSA_QUEUE_PRIORITY_NORMAL;
    
    // ⭐⭐⭐ 设置逻辑优先级（从应用传递）
    q->logical_priority = args->logical_priority;
    q->effective_priority = args->logical_priority;
    
    // ... 其他初始化 ...
    
    // 注册到 GPREEMPT 调度器
    kfd_gpreempt_register_queue(dev, q);
    
    return 0;
}
```

#### 步骤 4: GPREEMPT 调度器使用 logical_priority

**kfd_gpreempt_scheduler.c:**
```c
// 优先级倒置检测
static bool gpreempt_detect_inversion(...) {
    // 遍历所有 queue
    list_for_each_entry(q, &sched->all_queues, gpreempt_list) {
        // ⭐ 使用 logical_priority 做软件调度决策
        if (q->is_active && q->pending_count > 0) {
            if (!highest_waiting ||
                q->logical_priority > highest_waiting->logical_priority) {
                highest_waiting = q;
            }
        }
    }
    
    // ... 检测倒置 ...
    
    if (highest_waiting && running_queue &&
        highest_waiting->logical_priority > running_queue->logical_priority) {
        // 触发抢占
        return true;
    }
    
    return false;
}

// 抢占触发（使用 CWSR）
static int gpreempt_preempt_queue(struct queue *q) {
    printk(KERN_INFO "Preempting Q%u (logical_prio=%d, hw_prio=NORMAL)\n",
           q->properties.queue_id,
           q->logical_priority);  // ← 显示逻辑优先级
    
    // 触发 CWSR（硬件机制）
    return q->mqd_mgr->destroy_mqd(
        q->mqd_mgr,
        q->mqd,
        KFD_PREEMPT_TYPE_WAVEFRONT_SAVE,
        0,  // timeout=0: 异步
        q->pipe,
        q->queue
    );
}
```

---

## ✅ 设计优势

### 1. 完全可控 ⭐⭐⭐

```
✓ 调度逻辑 100% 在软件层
✓ 不依赖 GPU 固件的优先级实现（可能不可预测）
✓ 行为清晰透明，易于调试和验证
✓ 可以通过日志完整追踪调度决策
```

### 2. 灵活扩展

```
✓ 不受硬件限制：
  • 可以支持 >16 级优先级
  • 可以实现任意优先级粒度
  
✓ 可以实现复杂调度算法：
  • HPF (Highest Priority First)
  • DRF (Dominant Resource Fairness)
  • Deadline Scheduling
  • 动态优先级（防止饥饿）
  • 多队列公平性保证
```

### 3. 硬件行为一致

```
✓ 所有 queue 在 GPU 固件眼里相同（priority = NORMAL）
✓ 简化硬件交互，避免固件 bug 或不一致问题
✓ 不依赖硬件特性，跨 GPU 版本兼容性好
```

### 4. 性能保证

```
✓ Doorbell 提交仍然 ~100ns（完全不变）
✓ CWSR 抢占仍然 1-10μs（硬件机制）
✓ 监控开销 <0.001% CPU（定期轮询）
✓ 无硬件优先级调度的潜在开销或不确定性
```

---

## 📊 对比分析

### vs 硬件优先级调度（假设 ROCm 支持）

| 维度 | 硬件优先级 | 纯软件调度（我们的方案）|
|------|-----------|------------------------|
| **实现方式** | 依赖 GPU 固件 | 软件检测 + CWSR |
| **行为可预测性** | ⚠️ 依赖固件实现 | ✅ 100% 可控 |
| **调试难度** | ⚠️ 硬件黑盒 | ✅ 软件透明 |
| **优先级级别** | ⚠️ 固定 16 级 | ✅ 任意级别 |
| **灵活性** | ⚠️ 有限 | ✅ 高度灵活 |
| **跨版本兼容** | ⚠️ 可能不一致 | ✅ 软件统一 |
| **性能** | ✅ 硬件速度 | ✅ CWSR 1-10μs |

### vs GPreempt (NVIDIA)

| 维度 | GPreempt | AMD 纯软件调度 |
|------|----------|---------------|
| **调度方式** | 软件技巧 | 软件调度 + 硬件 CWSR |
| **优先级支持** | 2 级（时间片模拟）| 任意级别（纯软件）|
| **抢占机制** | 清空 Ring Buffer ⚠️ | CWSR 保持不变 ✅ |
| **Resume 方式** | 重新提交 ⚠️ | 精确恢复 ✅ |
| **抢占延迟** | 10-100μs | 1-10μs ✅ |
| **实现位置** | 用户态 | 内核态 ✅ |

---

## 🚀 实施计划

### Phase 1: 接口扩展（1-2 周）

```
1.1 扩展 kfd_ioctl_create_queue_args
    • 添加 logical_priority 字段
    • 更新 ioctl 版本号
    
1.2 修改 ROCm Runtime (AqlQueue)
    • 添加 logical_priority_ 成员
    • 从 HIP 获取并传递给 KFD
    
1.3 修改 KFD 队列创建逻辑
    • 存储 logical_priority
    • 保持 hardware_priority = NORMAL
```

### Phase 2: GPREEMPT 调度器（2-3 周）

```
2.1 实现监控线程
    • 扫描 Ring Buffer 状态
    • 使用 logical_priority 检测倒置
    
2.2 实现抢占/恢复逻辑
    • 调用 CWSR API (v2.1 修正)
    • 管理 snapshot 数据
    
2.3 添加日志和统计
    • 显示 logical_priority vs hardware_priority
    • 追踪调度决策
```

### Phase 3: 测试验证（1-2 周）

```
3.1 单元测试
    • 验证 logical_priority 传递
    • 验证优先级倒置检测
    
3.2 双进程测试
    • 训练 + 推理场景
    • 验证抢占和恢复
    
3.3 性能测试
    • 延迟比 >3×
    • 抢占延迟 <10μs
    • CPU 开销 <0.001%
```

**总计**: 4-7 周

---

## 📝 文档更新

### 已更新文档

1. **ARCH_Design_02_AMD_GPREEMPT_redesign_v2.md**
   - 版本: v2.2 → v2.3
   - 更新: 纯软件调度架构说明
   - 添加: logical_priority vs hardware_priority 区分
   - 添加: 代码验证和调用链分析

2. **ARCH_Design_03_AMD_GPREEMPT_XSCHED.md**
   - 版本: v3.1 → v3.2
   - 更新: 方案 A 和方案 B 的数据结构
   - 更新: 优先级检测逻辑
   - 添加: 纯软件调度说明

3. **本文档（新增）**
   - UPDATE_v2.3_SOFTWARE_SCHEDULING.md
   - 总结关键发现和架构调整
   - 提供实施指南

---

## 🎓 核心理念

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⭐⭐⭐ 纯软件调度的哲学
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"不依赖不可控的硬件特性，而是利用硬件机制实现可控的软件逻辑"

✅ 利用的硬件机制：
   • CWSR (1-10μs 精确抢占和恢复)
   • Ring Buffer (状态监控)
   • Doorbell (~100ns 快速提交)
   
✅ 不依赖的硬件特性：
   • 硬件优先级调度（可能不可预测）
   • 固件调度策略（黑盒）
   
✅ 实现的软件逻辑：
   • 纯软件优先级调度（100% 可控）
   • 灵活的调度策略（易于扩展）
   • 透明的调试验证（开源代码）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**这是最优方案**：
- 充分利用 AMD CWSR 的硬件优势
- 避免依赖硬件优先级的不确定性
- 实现完全可控的纯软件调度
- 保留 Doorbell 的性能
- 超越 GPreempt 的软件技巧妥协

---

**下一步**: 开始 Phase 1 实施，扩展 ioctl 接口传递 logical_priority
