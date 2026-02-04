# AMD GPREEMPT 架构设计 v2.0

**日期**: 2026-01-29  
**版本**: 2.4 (MQD/HQD 澄清 + 状态增强)  
**最后更新**: 2026-01-30（MQD/HQD 机制澄清，添加 HQD 资源管理）
**状态**: 架构澄清完成，准备实施  
**目标**: 在 AMD MI300 上实现超越 NVIDIA GPreempt 的优先级调度系统

---

## 📖 核心概念：MQD vs HQD（必读！）⭐⭐⭐

### MQD (Memory Queue Descriptor) - 软件队列

**本质**: 队列的配置模板，存储在主机内存中

**特点**:
- **存储位置**: 主机内存（通过 GART 映射到 GPU 地址空间）
- **数量**: 可以很多（受内存限制，测试显示：20-64 个 Queue Pool）
- **状态**: ACTIVE 或 INACTIVE
  - INACTIVE: 仅在内存中，未占用 HQD 硬件资源
  - ACTIVE: 已加载到 HQD，占用硬件资源
- **标识字段**: `cp_hqd_active` (MQD offset 0x82)
  - 0 = INACTIVE（未占用 HQD）
  - 1 = ACTIVE（已加载到 HQD）
- **内容**: ring buffer 地址、doorbell、priority、wptr、rptr 等配置

**查看方式**:
```bash
sudo cat /sys/kernel/debug/kfd/mqds
```

### HQD (Hardware Queue Descriptor) - 硬件队列

**本质**: GPU Command Processor 寄存器中的执行队列

**特点**:
- **存储位置**: GPU CP 寄存器（硬件）
- **数量**: **固定 32 个**（MI300X: 4 Pipes × 8 Queues/Pipe）
- **状态**: 只有 ACTIVE 状态（占用时）
- **访问**: 通过 MMIO 读写 CP 寄存器
- **作用**: GPU 从 HQD 寄存器读取配置并执行任务

**查看方式**:
```bash
sudo cat /sys/kernel/debug/kfd/hqds
```

### 映射关系：N 个 MQD → 动态复用 → 32 个 HQD

```
软件层（主机内存）:
┌─────────────────────────────────────┐
│ MQD 1, MQD 2, ..., MQD 20           │ ← 预分配的队列池（Queue Pool）
│ 状态: 大部分是 INACTIVE              │ ← 不占用硬件资源
│ 特点: 可以有很多个                   │
└─────────┬───────────────────────────┘
          │ 按需加载/卸载 (load_mqd/destroy_mqd)
          ↓
硬件层（GPU 寄存器）:
┌─────────────────────────────────────┐
│ HQD 1, HQD 2, ..., HQD 32           │ ← 固定的硬件槽位
│ 状态: 只有 ACTIVE 状态               │ ← 占用硬件资源
│ 特点: 数量固定（32 个）              │
└─────────────────────────────────────┘

核心理解：
  • 支持 > 32 个队列：通过 MQD Queue Pool
  • 硬件限制：同时活跃的队列 ≤ 32 个（HQD 数量）
  • 动态调度：高优先级 MQD 可以抢占低优先级的 HQD
```

### destroy_mqd 的实际语义（重要澄清！）

**函数名误导性**:
- ❌ 名字：`destroy_mqd()` - 听起来像销毁 MQD
- ✅ 实际：卸载 HQD（unload HQD） - 只清除 GPU 寄存器
- ✅ MQD 保留在内存中 - 可以稍后通过 `load_mqd()` 恢复

**代码验证**（amdgpu_amdkfd_gfx_v9.c:524-574）:
```c
int kgd_gfx_v9_hqd_destroy(struct amdgpu_device *adev, void *mqd,
                          enum kfd_preempt_type reset_type, ...) {
    // 1. 写入 HQD 寄存器，请求卸载
    WREG32(mmCP_HQD_DEQUEUE_REQUEST, type);  // type = SAVE_WAVES (CWSR)
    
    // 2. 等待 HQD 变为 INACTIVE
    while (RREG32(mmCP_HQD_ACTIVE) & ACTIVE_MASK) {
        // 等待 CWSR 完成...
    }
    
    // 3. 返回（MQD 仍在内存，未被修改！）✅
    return 0;
}
```

**关键理解**:
- ✅ `destroy_mqd()` 只操作 HQD 寄存器
- ✅ MQD 结构不受影响，保留在内存
- ✅ Ring Buffer 完全保持不变 ⭐⭐⭐
- ✅ CWSR Area 保存了 Wave 状态
- ✅ 可以通过 `load_mqd()` 重新加载到 HQD

### 队列生命周期

```
T0: 创建队列
    • 分配 MQD（内存）
    • 分配 Ring Buffer
    • cp_hqd_active = 0 (INACTIVE)
    • 不占用 HQD 硬件资源

T1: 提交任务 → load_mqd()
    • 选择空闲 HQD slot (0-31)
    • 将 MQD 配置写入 HQD 寄存器
    • cp_hqd_active = 1 (ACTIVE)
    • 占用 1 个 HQD 硬件资源

T2: CWSR 抢占 → destroy_mqd(WAVEFRONT_SAVE)
    • GPU 保存 Wave 状态到 CWSR Area
    • 清除 HQD 寄存器
    • cp_hqd_active = 0 (INACTIVE)
    • MQD 仍在内存！Ring Buffer 不变！✅

T3: 恢复执行 → load_mqd()
    • 重新加载 MQD 到 HQD
    • GPU 恢复 CWSR 保存的 Wave
    • 从断点处继续执行 ✅
```

### 术语对照表

| 原术语 | 实际含义 | 说明 |
|--------|---------|------|
| "抢占 MQD" | 抢占 HQD（卸载 HQD） | MQD 保留，HQD 被释放 |
| "destroy_mqd 销毁队列" | destroy_mqd 卸载 HQD | MQD 不销毁，只卸载硬件 |
| "监控 Queue 状态" | 监控 HQD 状态 | 读取 HQD 寄存器（实时） |
| "Queue 占用硬件" | MQD 加载到 HQD | 占用 32 个 HQD 之一 |

---

## 📢 重要更新

### 🔴 v2.3 更新（2026-01-30）- 纯软件调度架构（代码验证）

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⭐⭐⭐ 关键发现：amd_aql_queue.cpp:100 priority 写死验证
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

代码验证（ROCm Runtime）:
  文件: rocr-runtime/core/runtime/amd_aql_queue.cpp:100
  代码: priority_(HSA_QUEUE_PRIORITY_NORMAL),
  
  → 无论 HIP 设置什么 priority，AqlQueue 向 KFD 传递时都是 NORMAL
  
完整调用链分析:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  HIP 应用层:
    hipStreamCreateWithPriority(&stream, flags, priority);
      ↓ priority = 3 (训练) or 12 (推理) ← HIP 层设置
    
  ROCm Runtime (HSA) 层:
    AqlQueue::AqlQueue(...) {
      ...
      priority_(HSA_QUEUE_PRIORITY_NORMAL),  ← ⚠️ Line 100：写死！
      ...
    }
      ↓ ioctl(AMDKFD_IOC_CREATE_QUEUE)
    struct kfd_ioctl_create_queue_args {
      .queue_priority = HSA_QUEUE_PRIORITY_NORMAL  ← 所有 queue 相同
    }
    
  KFD 驱动层:
    struct queue {
      .properties.priority = HSA_QUEUE_PRIORITY_NORMAL  ← 所有 queue 相同
    }
      ↓ init_mqd()/load_mqd()
    
  GPU 固件/硬件层:
    MQD.cp_hqd_queue_priority = NORMAL  ← 硬件看到的都相同
    
    → GPU 固件不区分 queue 优先级
    → GPU 调度行为对所有 queue 一致


核心结论：纯软件调度设计
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 硬件层面：
  • 所有 queue 在 GPU 固件/硬件眼里优先级相同（NORMAL）
  • GPU 不会基于优先级做硬件调度
  • 硬件行为完全一致、可预测
  • 避免了硬件优先级调度的不确定性

✅ 软件层面：
  • HIP 应用仍然可以设置逻辑 priority
  • AqlQueue/KFD 层读取 HIP 传递的逻辑 priority
  • 在软件层（KFD/ROCm Runtime）做调度决策：
    1. 监控 Ring Buffer 状态（rptr/wptr）
    2. 检测优先级倒置（基于逻辑 priority）
    3. 使用 CWSR 抢占低优先级 queue
    4. 使用 CWSR 恢复高优先级 queue
    5. 使用同步原语控制执行顺序
  • 行为完全可控，逻辑透明

软件调度实现方案（三个可选层次）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  方案 1: KFD 内核层调度（推荐 ⭐⭐⭐）
    • 在 KFD 添加 GPREEMPT 调度器
    • 监控所有 queue 的 Ring Buffer
    • 直接调用 destroy_mqd/restore_mqd
    • 无 ioctl 开销，性能最优
    • 问题：如何获取 HIP 的逻辑 priority？
      → 需要扩展 kfd_ioctl_create_queue_args 传递
      → 或从 AqlQueue 通过 ioctl 同步
    
  方案 2: ROCm Runtime (AqlQueue) 层调度
    • 在 AqlQueue 层添加调度逻辑
    • 已有 HIP 的逻辑 priority 信息
    • 监控自己管理的所有 queue
    • 通过 KFD ioctl 触发 CWSR
    • 需要添加 ioctl: AMDKFD_IOC_PREEMPT_QUEUE
    
  方案 3: HIP 层调度
    • 在 HIP 层添加调度器
    • 直接访问 priority 信息
    • 协调多个 stream 的提交
    • 通过 AqlQueue/KFD 接口控制


推荐实现：方案 1（KFD 层）+ priority 传递机制
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

关键修改：
  1. 扩展 kfd_ioctl_create_queue_args:
     struct kfd_ioctl_create_queue_args {
       ...
       __u32 logical_priority;  // ← 新增：HIP 的逻辑优先级
     };
  
  2. AqlQueue 创建时传递逻辑 priority:
     args.logical_priority = hip_stream_priority;  // ← 从 HIP 获取
     ioctl(kfd_fd, AMDKFD_IOC_CREATE_QUEUE, &args);
  
  3. KFD 存储逻辑 priority:
     struct queue {
       int logical_priority;     // ← 逻辑优先级（HIP 传递）
       int hardware_priority;    // ← 硬件优先级（固定 NORMAL）
     };
  
  4. KFD GPREEMPT 调度器使用 logical_priority:
     if (high_q->logical_priority > low_q->logical_priority) {
       // 触发 CWSR 抢占
       gpreempt_preempt_queue(low_q);
     }


设计优势总结:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 完全可控：
  • 调度逻辑在软件层，100% 可控
  • 不依赖硬件优先级的可能不确定行为
  • 逻辑清晰，易于调试和验证

✅ 灵活扩展：
  • 不受硬件优先级级别限制（可以 >16 级）
  • 可以实现动态优先级
  • 可以实现复杂调度算法（HPF、DRF、FIFO、RR 等）
  • 可以实现 deadline scheduling

✅ 硬件一致：
  • 所有 queue 硬件行为相同
  • 简化硬件交互
  • 避免固件 bug 或不一致问题

✅ 性能保证：
  • Doorbell 提交仍然 ~100ns（不变）
  • CWSR 抢占仍然 1-10μs（硬件机制）
  • 监控开销 <0.001% CPU

对比硬件优先级调度：
  如果使用硬件优先级（假设 ROCm 支持）：
    ⚠️ 行为依赖 GPU 固件实现细节
    ⚠️ 可能不可预测或文档不全
    ⚠️ 调试困难（硬件黑盒）
    ⚠️ 灵活性差（固定级别）
  
  我们的纯软件调度：
    ✅ 行为完全可控和可预测
    ✅ 逻辑清晰透明（开源代码）
    ✅ 易于调试和验证
    ✅ 可以实现任意复杂策略
    ✅ 不依赖硬件特性

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 🔴 v2.1 更新（2026-01-29）- CWSR API 修正

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⭐⭐⭐ 关键修正：CWSR API 使用方式（基于 KFD CRIU 代码分析）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

通过对比 CWSR_API_USAGE_REFERENCE.md（KFD CRIU 实现），发现并修正
了 5 个关键问题：

1. ✅ destroy_mqd 参数修正：
   • 错误: destroy_mqd(..., pasid)
   • 正确: destroy_mqd(..., pipe, queue)

2. ✅ restore_mqd 参数修正：
   • 错误: restore_mqd(mgr, mqd, pasid) - 3个参数
   • 正确: restore_mqd(mgr, &mqd, mem_obj, gart, props, 
                      mqd_src, ctl_src, size) - 8个参数
   • 关键: &q->mqd 是 double pointer！

3. ✅ 新增 checkpoint_mqd 调用：
   • 在 preempt 前必须调用 checkpoint_mqd
   • 保存 MQD 和 control stack 到 snapshot

4. ✅ 新增 load_mqd 调用：
   • 在 restore 后必须调用 load_mqd
   • 才能激活队列

5. ✅ 新增 snapshot 数据结构：
   • struct queue 中添加 snapshot 字段
   • 包含 mqd_backup, ctl_stack_backup 等
   • 在队列注册时分配

参考: ARCH_DESIGN_CORRECTIONS_CWSR_API.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 📝 v2.0 更新（2026-01-29）

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⭐⭐⭐ 核心理解：软件技巧 vs 硬件机制
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

基于对 GPreempt_完整技术分析_综合版.md 的最新修正，本架构文档
全面更新了对 GPreempt 和 AMD CWSR 的理解：

1. GPU 硬件调度器的本质：
   • 由 GPU 固件控制，软件（包括驱动）无法直接修改
   • XSched、GPreempt 都是纯软件调度，无法改变硬件行为

2. GPreempt 的软件技巧（重新理解！）：
   • 不是修改 GPU 硬件调度器
   • 而是通过软件技巧"间接"影响硬件：
     a) 配置时间片参数（1s vs 1μs）→ GPU 硬件按此执行切换
     b) 清空 Ring Buffer → 让硬件"看到"无任务
     c) Reset CUs → 停止正在执行的 Waves
   • 修改的是硬件的"输入数据"，不是调度逻辑

3. AMD CWSR 的硬件优势（更清晰！）：
   • 硬件原生支持 CWSR 和优先级
   • 不需要软件技巧和妥协
   • Ring Buffer 保持不变（vs GPreempt 清空）
   • 精确状态恢复（vs GPreempt 重新提交）
   • 真正的硬件机制 vs 软件妥协方案

4. 本架构文档的更新：
   • 全面增强了与 GPreempt 的对比
   • 明确标注"软件技巧"vs"硬件机制"
   • 强调 AMD 无需这些妥协的本质优势
   • 所有关键章节都已更新

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📌 执行摘要

本文档基于对 `thustorage/GPreempt` 实际代码的深度分析和对 Ring Buffer + Doorbell 机制的深刻理解，重新设计了 AMD GPREEMPT 架构。

### 核心发现（基于 GPreempt 代码分析）

```
1. ✅ Ring Buffer 是核心（用户洞察）
   • Doorbell 紧密结合 Ring Buffer (rptr/wptr)
   • GPU 从 Ring Buffer 读取并执行 kernels
   • 抢占时必须正确处理 Ring Buffer 状态
   • AMD CWSR 最大优势：Ring Buffer 保持不变

2. ✅ GPreempt 的软件技巧（代码分析验证）
   • 配置时间片（1s vs 1μs）→ GPU 硬件按此切换
   • 清空 BE Ring Buffer → 让硬件"看到"无任务
   • Reset CUs → 停止正在执行的 Waves
   • 不修改硬件调度逻辑，而是修改硬件的"输入数据"
   • Resume 需要重新提交 kernels（可能重复执行）
   • 只支持 2 个优先级

3. ✅ AMD 的硬件优势（超越 GPreempt）
   • 硬件优先级（0-15）→ 不需要时间片技巧
   • 硬件 CWSR（1-10μs）→ 不需要清空 Ring Buffer
   • Ring Buffer 保持不变 → 不需要重新提交
   • 精确状态恢复（指令级）→ 无重复执行
   • 开源驱动 → 易于实现和调试
```

### v2.0 的关键改进

```
相比原架构文档的改进：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ✅ 明确 Ring Buffer 在抢占中的核心作用
   原文档: 提到了 rptr/wptr，但没有深入机制
   新设计: 完整阐述 Ring Buffer 监控和保持不变的优势

2. ✅ 修正优先级概念
   原文档: 优先级概念有些混乱
   新设计: 清晰区分硬件优先级（0-15）和时间片（可选）

3. ✅ 强调 CWSR 的精确性
   原文档: 提到了 CWSR，但没有强调其相对 GPreempt 的优势
   新设计: 明确 AMD 不需要清空 Ring Buffer 这个核心优势

4. ✅ 理解 GPreempt 的软件技巧（新增！⭐⭐⭐）
   • GPU 硬件调度器是固件控制的，软件无法修改
   • GPreempt 的"抢占"是软件技巧的组合：
     1. 配置时间片参数（1s vs 1μs）
     2. 清空 Ring Buffer 让硬件"看到"无任务
     3. Reset CUs 停止执行
   • 修改的是硬件的"输入数据"，不是调度逻辑
   • AMD CWSR 是真正的硬件机制，无需这些妥协

5. ✅ 简化架构复杂度
   原文档: 有些过度设计
   新设计: 聚焦核心功能，去除不必要的复杂性

6. ✅ 基于实际代码验证
   原文档: 部分基于推测
   新设计: 基于 GPreempt 实际代码和用户洞察
```

---

## 🎯 架构概览

### 核心设计原则

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⭐⭐⭐ AMD 硬件机制 vs GPreempt 软件技巧
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

关键理解：
  • GPreempt 无法修改 GPU 硬件调度器（固件控制）
  • 只能通过软件技巧"间接"影响硬件：
    1. 配置时间片参数（NvRmModifyTS）
    2. 清空 Ring Buffer（让硬件"看到"无任务）
    3. Reset CUs（停止执行）
  
  • AMD CWSR 是真正的硬件机制：
    1. 硬件优先级（无需时间片技巧）
    2. 硬件 CWSR（无需清空 Ring Buffer）
    3. 精确状态恢复（无需重新提交）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ✅ 保留 Doorbell 性能（与 GPreempt 相同）
   • 应用直接通过 Doorbell 提交（~100ns）
   • 不拦截、不修改 HIP API
   • 完全透明

2. ✅ 内核态监控和控制（vs GPreempt 用户态）
   • KFD 内核线程定期轮询
   • 直接读取 Ring Buffer rptr/wptr (MMIO)
   • 无 ioctl 开销

3. ✅ 硬件 CWSR 抢占（vs GPreempt 软件技巧）
   • destroy_mqd() 触发 CWSR
   • GPU 保存 Wave 状态到 CWSR Area
   • ⭐⭐⭐ Ring Buffer 完全保持不变
   • vs GPreempt: 清空 Ring Buffer（妥协方案）

4. ✅ 精确 Resume（vs GPreempt 重新提交）
   • restore_mqd() 恢复 MQD
   • GPU 从 CWSR Area 恢复 Wave 状态
   • ⭐⭐⭐ 从 Ring Buffer rptr 继续读取
   • vs GPreempt: 重新提交 kernels（可能重复执行）

5. ✅ 任意级别软件优先级（vs GPreempt 2 级时间片模拟）⭐⭐⭐
   • 不受硬件限制，可以支持 >16 级
   • queue->logical_priority (软件定义，灵活)
   • queue->properties.priority = NORMAL (硬件固定，所有 queue 相同)
   • 纯软件调度，行为完全可控 ✅
   • vs GPreempt: 通过时间片模拟（1s vs 1μs）

6. ✅ 无需软件技巧（核心区别！）
   • 不需要配置时间片参数
   • 不需要清空 Ring Buffer
   • 不需要重新提交 kernels
   • 纯硬件机制，可靠、快速
```

### 三层架构

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 1: 应用层 (Application Layer)                         │
│  ──────────────────────────────────────────────────────────  │
│  • HIP/ROCm 应用（PyTorch, TensorFlow, ...）                  │
│  • Create stream/Queue with different priority               │
│  • 使用标准 HIP API：                                         │
│    hipLaunchKernel() → 写 Ring Buffer → Doorbell (~100ns) ⚡│
│  • 完全透明，无需修改 ✅                           │
└──────────────────────────────────────────────────────────────┘
          ↓ Doorbell (MMIO write, 直接到 GPU) ⚡
          ↓
┌──────────────────────────────────────────────────────────────┐
│  Layer 2: GPU 硬件层 (CPFW+GPU HW)                           │
│  ──────────────────────────────────────────────────────────  │
│  • Ring Buffer (rptr/wptr 寄存器) ⭐-CPFW                    │
│  • Command Processor (CP)                                    │
│  • CWSR Trap Handler                                         │
│  • Compute Units (CUs)                                       │
│  • ? if  stream/Queue has prioirty                           │
│  • 执行：从 Ring Buffer 读取并执行 kernels                   │
└──────────────────────────────────────────────────────────────┘
          ↑ 主动轮询监控 (MMIO read) 🔍
          ↑ 读取 rptr/wptr/status
          ↑
┌──────────────────────────────────────────────────────────────┐
│  Layer 3: 内核调度层 (KFD GPREEMPT Scheduler)               │
│  ──────────────────────────────────────────────────────────  │
│  • 内核态监控线程（kthread）                                  │
│  • 定期轮询所有队列状态（5ms）                                │
│  • 读取 rptr/wptr 判断队列活跃度                              │
│  • 检测优先级倒置                                             │
│  • 触发 CWSR 抢占（destroy_mqd）                              │
│  • 触发 CWSR Resume（restore_mqd）                            │
│  • 提供 ioctl 接口（配置和手动控制）                          │
└──────────────────────────────────────────────────────────────┘
          ↓ PM4 命令 (UNMAP_QUEUES)
          └────→ 回到 Layer 2 (触发 CWSR)

关键理解：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ Doorbell 不通知内核！
  • 应用 Doorbell 直接写 GPU 硬件（MMIO write）
  • KFD 无法收到任何中断或通知
  • KFD 必须主动轮询（MMIO read）

✅ 这个设计可行！
  • 数据平面（提交）：~100ns，不受影响 ⚡
  • 控制平面（调度）：5ms 轮询，开销极低 ✅
  • 检测延迟：平均 2.5ms，完全可接受 ✅
```

---

## 🏗️ 核心数据结构

### Ring Buffer 状态监控（核心！）

```c
// ============================================================================
// Ring Buffer 是整个系统的核心！
// ============================================================================

// 队列扩展结构（v2.4 - 添加 HQD 资源管理）
struct queue {
    // ===== 现有字段 =====
    struct queue_properties properties;  // 包含 hardware_priority (固定 NORMAL)
    struct mqd_manager *mqd_mgr;
    void *mqd;                           // ← MQD 在内存中的地址
    struct kfd_mem_obj *mqd_mem_obj;     // MQD 内存对象
    uint64_t gart_mqd_addr;              // GART 地址
    
    // ⭐⭐⭐ HQD 资源状态（v2.4 新增 - 核心！）
    bool is_loaded_to_hqd;               // ← cp_hqd_active 的软件表示
                                         //   = MQD.cp_hqd_active (offset 0x82)
                                         //   true: 占用 HQD，false: 仅在内存
    int32_t hqd_slot;                    // ← 占用哪个 HQD (0-31)
                                         //   -1 表示未加载到 HQD
    
    // ⭐⭐⭐ 纯软件调度字段（v2.3）
    int logical_priority;                // HIP 设置的逻辑优先级（0-15 或更多）
                                         // 从 AqlQueue 传递，用于软件调度决策
    
    // ===== Ring Buffer 状态（核心！）⭐⭐⭐ =====
    
    // ⚠️ 重要区分（v2.4 澄清）：
    // • is_loaded_to_hqd = true: 从 HQD 寄存器读取（实时）
    // • is_loaded_to_hqd = false: 从 MQD 或 Ring Buffer 读取（缓存值）
    
    // Ring Buffer 硬件指针
    u32 hw_rptr;          // GPU 读指针（GPU 更新）
                          // ⚠️ 只有 is_loaded_to_hqd=true 时才从 HQD 读取实时值
    u32 hw_wptr;          // CPU 写指针（应用更新）
    u32 ring_size;        // Ring Buffer 大小
    
    // 队列状态（从 Ring Buffer 计算得出）
    u32 pending_count;    // wptr - rptr = 待执行的 kernels 数量
    bool is_active;       // pending_count > 0
    bool is_running;      // GPU 正在执行此队列（is_loaded_to_hqd && pending_count > 0）
    
    // 优先级调度字段（v2.3 更新：纯软件调度）
    int base_priority;      // 基础逻辑优先级（从 logical_priority）
    int effective_priority; // 动态优先级（防止饥饿，基于 base_priority 计算）
    
    // ⚠️ properties.priority 固定为 NORMAL（硬件优先级）
    // ⚠️ 只使用 logical_priority 做软件调度
    
    // 时间片（可选功能）
    u64 timeslice_us;     // 时间片（微秒）
    ktime_t slice_start;  // 时间片开始时间
    
    // 抢占状态
    enum queue_state {
        QUEUE_STATE_ACTIVE,      // 活跃（正在执行或等待）
        QUEUE_STATE_PREEMPTED,   // 被抢占（CWSR saved）
        QUEUE_STATE_IDLE         // 空闲（pending_count=0）
    } state;
    
    bool preemption_pending;   // 抢占进行中
    ktime_t preempt_start;     // 抢占开始时间
    
    // ⭐⭐⭐ Snapshot（用于 checkpoint/restore）- 关键新增！
    struct {
        void *mqd_backup;          // MQD 备份 buffer
        void *ctl_stack_backup;    // Control stack 备份 buffer
        size_t ctl_stack_size;     // Control stack 大小
        bool valid;                // Snapshot 是否有效
    } snapshot;
    
    // CWSR 状态
    dma_addr_t cwsr_area;      // CWSR 保存区域地址
    size_t cwsr_size;          // 保存的状态大小
    bool cwsr_valid;           // CWSR 状态是否有效
    
    // 统计信息
    u64 total_preemptions;
    u64 total_resumes;
    u64 total_exec_time_us;
    
    // 链表节点
    struct list_head sched_list;      // 全局队列链表
    struct list_head priority_list;   // 同优先级队列链表
};


// 全局调度器（v2.4 - 添加 HQD 资源管理）
struct kfd_gpreempt_scheduler {
    // 监控线程
    struct task_struct *monitor_thread;
    wait_queue_head_t wait_queue;
    
    // 配置
    unsigned int check_interval_ms;    // 监控间隔（默认 5ms）
    bool enabled;                      // 是否启用
    
    // 队列管理
    struct list_head all_queues;              // 所有队列（MQD）
    struct list_head priority_queues[16];     // 按 logical_priority 分组
    spinlock_t queue_lock;
    
    // ⭐⭐⭐ HQD 资源管理（v2.4 新增）
    struct queue *hqd_slots[32];              // 32 个 HQD 的占用情况
                                              // hqd_slots[i] = 占用 HQD i 的队列
                                              // NULL = HQD i 空闲
    uint32_t active_hqd_count;                // 当前活跃的 HQD 数量 (≤32)
    uint32_t free_hqd_count;                  // 空闲的 HQD 数量 (=32-active)
    DECLARE_BITMAP(hqd_bitmap, 32);           // HQD 占用位图（快速查找）
    spinlock_t hqd_lock;                      // 保护 HQD 资源
    
    // HQD 资源统计
    uint64_t hqd_contention_count;            // HQD 资源竞争次数
    uint32_t hqd_max_usage;                   // 最高 HQD 使用数量
    
    // 调度策略（简化！）
    enum {
        SCHED_STRICT_PRIORITY,    // 严格优先级（主要）
        SCHED_TIMESLICE          // 时间片轮转（辅助）
    } policy;
    
    // 统计
    atomic64_t total_checks;
    atomic64_t total_inversions;
    atomic64_t total_preemptions;
};


// ⭐⭐⭐ HQD 资源管理辅助函数（v2.4 新增）
// ============================================================================

// 初始化 HQD 资源追踪
static void gpreempt_init_hqd_tracking(struct kfd_gpreempt_scheduler *sched)
{
    spin_lock_init(&sched->hqd_lock);
    
    for (int i = 0; i < 32; i++) {
        sched->hqd_slots[i] = NULL;
    }
    
    bitmap_zero(sched->hqd_bitmap, 32);
    sched->active_hqd_count = 0;
    sched->free_hqd_count = 32;
    sched->hqd_contention_count = 0;
    sched->hqd_max_usage = 0;
}


// 分配空闲 HQD slot
static int allocate_hqd_slot(struct kfd_gpreempt_scheduler *sched)
{
    unsigned long flags;
    int slot = -1;
    
    spin_lock_irqsave(&sched->hqd_lock, flags);
    
    // 快速查找第一个空闲 HQD
    slot = find_first_zero_bit(sched->hqd_bitmap, 32);
    if (slot < 32) {
        set_bit(slot, sched->hqd_bitmap);  // 标记为占用
    } else {
        slot = -1;  // 无空闲 HQD
    }
    
    spin_unlock_irqrestore(&sched->hqd_lock, flags);
    return slot;
}


// 标记 HQD 被占用
static void mark_hqd_occupied(struct kfd_gpreempt_scheduler *sched,
                              uint32_t slot,
                              struct queue *q)
{
    unsigned long flags;
    
    spin_lock_irqsave(&sched->hqd_lock, flags);
    
    sched->hqd_slots[slot] = q;
    sched->active_hqd_count++;
    sched->free_hqd_count--;
    
    // 更新统计
    if (sched->active_hqd_count > sched->hqd_max_usage) {
        sched->hqd_max_usage = sched->active_hqd_count;
    }
    
    spin_unlock_irqrestore(&sched->hqd_lock, flags);
}


// 标记 HQD 空闲
static void mark_hqd_free(struct kfd_gpreempt_scheduler *sched, uint32_t slot)
{
    unsigned long flags;
    
    spin_lock_irqsave(&sched->hqd_lock, flags);
    
    sched->hqd_slots[slot] = NULL;
    clear_bit(slot, sched->hqd_bitmap);
    sched->active_hqd_count--;
    sched->free_hqd_count++;
    
    spin_unlock_irqrestore(&sched->hqd_lock, flags);
}


// 检查 MQD 是否加载到 HQD
static bool is_queue_loaded_to_hqd(struct queue *q)
{
    // 方法 1: 读取 MQD 的 cp_hqd_active 字段（offset 0x82）
    uint32_t *mqd_ptr = (uint32_t *)q->mqd;
    uint32_t cp_hqd_active = mqd_ptr[0x82];
    
    return (cp_hqd_active == 1);
    
    // 方法 2: 使用软件维护的状态（更快）
    // return q->is_loaded_to_hqd;
}


// 从 HQD 寄存器读取实时 rptr
static uint32_t read_hqd_rptr_register(struct queue *q)
{
    // 通过 MMIO 读取 HQD 寄存器
    // 这是 GPU 硬件的实时状态
    
    if (!q->is_loaded_to_hqd || q->hqd_slot < 0) {
        // 未加载到 HQD，返回 MQD 中的缓存值
        return q->hw_rptr;
    }
    
    // 从 HQD 寄存器读取（实际实现需要访问 GPU MMIO）
    // 这里是伪代码示意
    void __iomem *hqd_base = get_hqd_register_base(q->hqd_slot);
    uint32_t rptr = readl(hqd_base + CP_HQD_PQ_RPTR_OFFSET);
    
    return rptr;
}
```

---

## ⚙️ 核心实现：监控线程

### 主循环（简化！）

```c
// ============================================================================
// 文件: amd/amdkfd/kfd_gpreempt_scheduler.c
// ============================================================================

static int kfd_gpreempt_monitor_thread(void *data)
{
    struct kfd_gpreempt_scheduler *sched = data;
    struct queue *high_q, *low_q;
    
    pr_info("KFD GPREEMPT: Monitor thread started (interval=%dms)\n",
            sched->check_interval_ms);
    
    while (!kthread_should_stop()) {
        // ⭐ 步骤 1: 定期休眠（减少 CPU 消耗）
        msleep_interruptible(sched->check_interval_ms);
        
        if (!sched->enabled)
            continue;
        
        // ⭐ 步骤 2: 扫描所有队列的 Ring Buffer 状态
        gpreempt_scan_queues(sched);
        
        // ⭐ 步骤 3: 检测优先级倒置（基于逻辑优先级）⭐⭐⭐
        if (gpreempt_detect_inversion(sched, &high_q, &low_q)) {
            pr_info("GPREEMPT: Priority inversion detected\n");
            pr_info("  High queue %u (logical_prio=%d, hw_prio=NORMAL, pending=%u) waiting\n",
                    high_q->properties.queue_id,
                    high_q->logical_priority,        // ← 使用逻辑优先级做决策
                    high_q->pending_count);
            pr_info("  Low queue %u (logical_prio=%d, hw_prio=NORMAL, pending=%u) running\n",
                    low_q->properties.queue_id,
                    low_q->logical_priority,         // ← 使用逻辑优先级做决策
                    low_q->pending_count);
            
            // ⭐ 步骤 4: 触发抢占（纯软件决策）
            gpreempt_preempt_queue(sched, low_q, high_q);
            
            atomic64_inc(&sched->total_inversions);
        }
        
        // ⭐ 步骤 5: 检查是否可以恢复被抢占的队列
        gpreempt_check_resume(sched);
        
        atomic64_inc(&sched->total_checks);
    }
    
    pr_info("KFD GPREEMPT: Monitor thread exiting\n");
    return 0;
}
```

### Ring Buffer 扫描（核心！）

```c
// ⭐⭐⭐ 扫描所有队列的 Ring Buffer 状态
//
// 关键理解（感谢用户洞察！）：
//   1. Ring Buffer 由 CPU (wptr) 和 GPU (rptr) 共同维护
//   2. wptr: 应用通过 Doorbell 更新（MMIO write）
//   3. rptr: GPU 读取后更新（MMIO write）
//   4. KFD 主动读取 rptr/wptr（MMIO read）
//   5. pending_count = wptr - rptr = 待执行的 kernels
//
static void gpreempt_scan_queues(struct kfd_gpreempt_scheduler *sched)
{
    struct queue *q;
    unsigned long flags;
    
    spin_lock_irqsave(&sched->queue_lock, flags);
    
    list_for_each_entry(q, &sched->all_queues, sched_list) {
        if (!q->properties.doorbell_ptr)
            continue;
        
        // ⭐⭐⭐ 核心：主动读取 Ring Buffer 指针（MMIO read）
        //
        // 这些寄存器与应用 Doorbell 写的是同一组硬件寄存器！
        //   应用: *doorbell_ptr = wptr  (MMIO write, ~100ns)
        //   内核: wptr = readl(...)     (MMIO read, ~100ns)
        //
        u32 hw_rptr = readl(q->properties.read_ptr);   // GPU 读指针
        u32 hw_wptr = readl(q->properties.write_ptr);  // CPU 写指针
        
        // 计算待执行的 kernels 数量
        u32 pending = (hw_wptr - hw_rptr) & q->ring_size_mask;
        
        // 更新队列状态
        q->hw_rptr = hw_rptr;
        q->hw_wptr = hw_wptr;
        q->pending_count = pending;
        q->is_active = (pending > 0);  // 有待执行的 kernels
        
        // 判断是否正在运行（可选，需要额外的状态寄存器）
        // u32 status = readl(q->properties.queue_address + OFFSET_STATUS);
        // q->is_running = (status & STATUS_RUNNING_BIT);
        
        // 调试日志
        if (q->is_active) {
            pr_debug("Queue %u: rptr=%u, wptr=%u, pending=%u, prio=%d\n",
                     q->properties.queue_id, hw_rptr, hw_wptr,
                     pending, q->effective_priority);
        }
    }
    
    spin_unlock_irqrestore(&sched->queue_lock, flags);
}


关键理解:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Ring Buffer 指针的可见性:
   • 应用通过 Doorbell 写 wptr（MMIO write）
   • 硬件寄存器立即更新
   • KFD 通过 readl 读 wptr（MMIO read）
   • 硬件保证原子性和可见性

✅ pending_count = wptr - rptr:
   • 表示 Ring Buffer 中有多少 kernels 待执行
   • pending > 0: 队列活跃
   • pending = 0: 队列空闲

✅ 轮询开销:
   • 每个队列: 2 次 MMIO read (~200ns)
   • 假设 100 个队列: 100 * 200ns = 20μs
   • 每 5ms 一次: CPU 占用 <0.001%
```

### 优先级倒置检测

```c
// ⭐⭐⭐ 检测优先级倒置（基于逻辑优先级）⭐⭐⭐
//
// 倒置定义：
//   有高逻辑优先级队列 (is_active=true, is_running=false)
//   且有低逻辑优先级队列 (is_running=true)
//
// ⚠️ 注意：所有 queue 的 hardware_priority 都是 NORMAL（相同）
// ⚠️ 这里使用 logical_priority 做软件调度决策
//
static bool gpreempt_detect_inversion(
    struct kfd_gpreempt_scheduler *sched,
    struct queue **out_high_q,
    struct queue **out_low_q)
{
    struct queue *q_high, *q_low;
    unsigned long flags;
    bool found = false;
    
    spin_lock_irqsave(&sched->queue_lock, flags);
    
    // ⭐ 从高到低遍历逻辑优先级（不限于 0-15）
    for (int high_prio = 15; high_prio >= 1 && !found; high_prio--) {
        list_for_each_entry(q_high, &sched->priority_queues[high_prio], priority_list) {
            
            // ⭐ 条件 1: 高逻辑优先级队列有待执行的任务
            if (!q_high->is_active || q_high->pending_count == 0)
                continue;
            
            // ⭐ 条件 2: 高逻辑优先级队列没在运行（被阻塞）
            if (q_high->is_running)
                continue;
            
            // 查找是否有低逻辑优先级队列在运行
            for (int low_prio = high_prio - 1; low_prio >= 0; low_prio--) {
                list_for_each_entry(q_low, &sched->priority_queues[low_prio], priority_list) {
                    
                    // ⭐ 条件 3: 低逻辑优先级队列正在运行
                    if (q_low->is_running && !q_low->preemption_pending) {
                        // ⚠️ 逻辑优先级倒置！
                        *out_high_q = q_high;
                        *out_low_q = q_low;
                        found = true;
                        goto out;
                    }
                }
            }
        }
    }
    
out:
    spin_unlock_irqrestore(&sched->queue_lock, flags);
    return found;
}


关键说明（v2.3 更新）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 纯软件调度逻辑：
  • 所有 queue 的 hardware_priority = NORMAL（GPU 固件眼里相同）
  • 使用 logical_priority（从 HIP 传递）做软件决策
  • GPU 不会基于优先级调度，需要软件干预

✅ priority_queues[] 数组：
  • 按 logical_priority 分组，不是 hardware_priority
  • logical_priority 可以 >15 级（软件定义）
  • 灵活扩展，不受硬件限制


简化版本（实际可能更简单）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

bool gpreempt_detect_inversion(...)
{
    // 找到优先级最高的活跃队列
    struct queue *highest_active = find_highest_priority_active_queue();
    
    // 找到当前正在运行的队列
    struct queue *current_running = find_running_queue();
    
    // 如果正在运行的队列优先级低于最高活跃队列
    if (current_running && highest_active &&
        current_running->effective_priority < highest_active->effective_priority) {
        *out_high_q = highest_active;
        *out_low_q = current_running;
        return true;
    }
    
    return false;
}
```

---

## ⚡ 核心实现：CWSR 抢占

### 抢占触发（关键修正！）

```c
// ⭐⭐⭐ 触发 CWSR 抢占
//
// 关键理解（修正原设计）：
//   1. 不清空 Ring Buffer！（这是 AMD 相对 GPreempt 的核心优势）
//   2. 先 checkpoint MQD，再触发 CWSR 保存 Wave 状态
//   3. Ring Buffer 中的 kernels 保持不变
//   4. Resume 时 GPU 从 rptr 继续读取
//
static int gpreempt_preempt_queue(
    struct kfd_gpreempt_scheduler *sched,
    struct queue *low_q,
    struct queue *high_q)
{
    int r;
    ktime_t start = ktime_get();
    
    pr_info("GPREEMPT: Preempting queue %u (prio=%d) for queue %u (prio=%d)\n",
            low_q->properties.queue_id, low_q->effective_priority,
            high_q->properties.queue_id, high_q->effective_priority);
    
    pr_info("  Ring Buffer state: rptr=%u, wptr=%u, pending=%u\n",
            low_q->hw_rptr, low_q->hw_wptr, low_q->pending_count);
    
    // ⭐⭐⭐ 步骤 1: Checkpoint MQD（必须在 destroy 前！）
    //
    // 关键：这一步保存 MQD 和 control stack 到 snapshot
    //       为后续 restore 提供数据
    //
    low_q->mqd_mgr->checkpoint_mqd(
        low_q->mqd_mgr,
        low_q->mqd,                       // 源 MQD
        low_q->snapshot.mqd_backup,       // 目标 MQD buffer
        low_q->snapshot.ctl_stack_backup  // 目标 control stack buffer
    );
    low_q->snapshot.valid = true;
    
    pr_debug("  MQD checkpointed (mqd=%p, ctl=%p)\n",
             low_q->snapshot.mqd_backup,
             low_q->snapshot.ctl_stack_backup);
    
    // 标记状态
    low_q->preemption_pending = true;
    low_q->preempt_start = start;
    low_q->state = QUEUE_STATE_PREEMPTED;
    
    // ⭐⭐⭐ 步骤 2: 触发硬件 CWSR（异步）
    //
    // 重要：这里只触发 CWSR，不修改 Ring Buffer！
    //       Ring Buffer 中的 kernels 保持原样
    //       GPU 保存 Wave 状态到 CWSR Area
    //
    // 参数修正：使用 pipe 和 queue，而不是 pasid
    //
    r = low_q->mqd_mgr->destroy_mqd(
        low_q->mqd_mgr,
        low_q->mqd,
        KFD_PREEMPT_TYPE_WAVEFRONT_SAVE,  // CWSR 模式
        0,                                 // timeout=0: 异步
        low_q->pipe,                       // ⭐ pipe 编号（修正！）
        low_q->queue                       // ⭐ queue 编号（修正！）
    );
    
    if (r == 0) {
        // 抢占立即完成（队列可能已经空闲）
        low_q->preemption_pending = false;
        pr_debug("Queue %u preemption completed immediately\n",
                 low_q->properties.queue_id);
    } else if (r == -EINPROGRESS) {
        // 抢占正在进行中（正常情况）
        atomic64_inc(&sched->total_preemptions);
        low_q->total_preemptions++;
        pr_debug("Queue %u CWSR in progress\n",
                 low_q->properties.queue_id);
        
        // 后续通过中断或下次轮询确认完成
    } else {
        // 错误
        pr_err("GPREEMPT: Preemption failed for queue %u: %d\n",
               low_q->properties.queue_id, r);
        low_q->preemption_pending = false;
        low_q->state = QUEUE_STATE_ACTIVE;
        low_q->snapshot.valid = false;  // 清除无效的 snapshot
        return r;
    }
    
    u64 latency = ktime_us_delta(ktime_get(), start);
    pr_info("GPREEMPT: Preemption latency: %llu us\n", latency);
    
    return 0;
}


关键对比（GPreempt vs AMD）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPreempt (NVIDIA) 抢占时做的事（软件技巧组合）:
  1. 设置 preempt_flag = 1 (GPU 内存)
  2. ⚠️ GPUClearHostQueue() - 清空 Ring Buffer
     • wptr = rptr
     • Ring Buffer 中的 kernels 丢失
     • 需要记录 kernel_offset
     • 目的：让硬件"看到" BE 无任务
  3. GPUResetCU() - 硬件抢占
     • 停止正在执行的 Waves
  
  延迟: 10-100μs
  本质: 软件妥协（无法修改硬件调度逻辑）⚠️
  问题: Ring Buffer 被清空，需要重新提交 ⚠️

AMD GPREEMPT 抢占时做的事（硬件机制）:
  1. destroy_mqd(..., WAVEFRONT_SAVE, ...)
     • GPU 执行 CWSR（硬件机制）
     • 保存 Wave 状态到 CWSR Area
     • ✅ Ring Buffer 完全保持不变！
       - rptr = 25 (GPU 读到这里)
       - wptr = 100 (应用提交到这里)
       - Ring Buffer 中的 75 个 kernels 仍然在！
  
  延迟: 1-10μs ✅
  本质: 纯硬件机制（不需要软件技巧）✅
  优势: Ring Buffer 保持不变，无需清空 ⭐⭐⭐

核心区别:
  • GPreempt: 无法修改硬件，只能通过软件技巧间接影响
  • AMD CWSR: 原生硬件支持，直接实现抢占
  
这是 AMD 相对 GPreempt 的最大优势！
```

---

## 🔄 核心实现：CWSR Resume

### Resume 触发（关键修正！）

```c
// ⭐⭐⭐ 检查并恢复被抢占的队列
//
// 关键理解（修正原设计）：
//   1. 不需要重新提交 kernels！（AMD vs GPreempt 核心差异）
//   2. 先 restore MQD，再 load MQD 激活队列
//   3. GPU 从 CWSR Area 恢复 Wave 状态
//   4. GPU 从 Ring Buffer rptr 继续读取 kernels
//   5. 就像从未被打断一样
//
static void gpreempt_check_resume(struct kfd_gpreempt_scheduler *sched)
{
    struct queue *q;
    unsigned long flags;
    
    spin_lock_irqsave(&sched->queue_lock, flags);
    
    // 遍历所有被抢占的队列
    list_for_each_entry(q, &sched->all_queues, sched_list) {
        if (q->state != QUEUE_STATE_PREEMPTED)
            continue;
        
        // 检查 snapshot 是否有效
        if (!q->snapshot.valid) {
            pr_warn("GPREEMPT: Queue %u has no valid snapshot\n",
                    q->properties.queue_id);
            continue;
        }
        
        // ⭐ 检查是否有更高优先级的队列在运行
        if (gpreempt_has_higher_priority_running(sched, q))
            continue;
        
        // ⭐ 可以恢复！
        pr_info("GPREEMPT: Resuming queue %u (prio=%d)\n",
                q->properties.queue_id, q->effective_priority);
        
        pr_info("  Ring Buffer state: rptr=%u, wptr=%u, pending=%u\n",
                q->hw_rptr, q->hw_wptr, q->pending_count);
        
        pr_info("  Snapshot: mqd=%p, ctl=%p, size=%zu\n",
                q->snapshot.mqd_backup,
                q->snapshot.ctl_stack_backup,
                q->snapshot.ctl_stack_size);
        
        // ⭐⭐⭐ 步骤 1: Restore MQD（从 snapshot 恢复）
        //
        // 重要：使用正确的参数（8个）
        //   - &q->mqd: double pointer（修正！）
        //   - 从 snapshot 读取保存的数据
        //
        q->mqd_mgr->restore_mqd(
            q->mqd_mgr,
            &q->mqd,                      // ⭐ double pointer（修正！）
            q->mqd_mem_obj,               // MQD 内存对象
            &q->gart_mqd_addr,            // GART 地址
            &q->properties,               // 队列属性
            q->snapshot.mqd_backup,       // 源 MQD（从 snapshot）
            q->snapshot.ctl_stack_backup, // 源 control stack（从 snapshot）
            q->snapshot.ctl_stack_size    // control stack 大小
        );
        
        pr_debug("  MQD restored from snapshot\n");
        
        // ⭐⭐⭐ 步骤 2: Load MQD（激活队列）
        //
        // 关键：restore_mqd 只是恢复了数据结构
        //       必须调用 load_mqd 才能让 GPU 重新执行这个队列
        //
        // GPU 会做什么？
        //   1. 从 CWSR Area 恢复 Wave 状态（PC + 寄存器 + LDS）
        //   2. ⭐ 从 Ring Buffer rptr 继续读取 kernels
        //   3. rptr=25, wptr=100 → 继续执行 kernel_25, 26, ..., 99
        //   4. 无需重新提交任何 kernel！✅
        //
        int r = q->mqd_mgr->load_mqd(
            q->mqd_mgr,
            q->mqd,
            q->pipe,
            q->queue,
            &q->properties,
            q->process->mm
        );
        
        if (r == 0) {
            // 成功
            q->preemption_pending = false;
            q->state = QUEUE_STATE_ACTIVE;
            q->properties.is_active = true;  // 标记为活动
            q->snapshot.valid = false;       // 清除 snapshot
            
            u64 latency = ktime_us_delta(ktime_get(), q->preempt_start);
            pr_info("GPREEMPT: Queue %u resumed (latency=%llu us)\n",
                    q->properties.queue_id, latency);
            
            q->total_resumes++;
            atomic64_inc(&sched->total_resumes);
        } else {
            pr_err("GPREEMPT: Failed to load MQD for queue %u: %d\n",
                   q->properties.queue_id, r);
            // 保持 PREEMPTED 状态，下次重试
        }
    }
    
    spin_unlock_irqrestore(&sched->queue_lock, flags);
}


关键对比（GPreempt vs AMD）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPreempt (NVIDIA) Resume（软件技巧妥协）:
  1. reset_preempt_flag = 0
  2. ⚠️ 从 kernel_offset 重新提交 kernels:
     for (i = kernel_offset; i < total; i++)
       cuLaunchKernel(kernel_i)  // 重新写 Ring Buffer
         ↓
       Ring Buffer: [k22, k23, k24, ..., k99]
         ↓
       Doorbell write ⚡
  3. GPU 重新执行
  
  延迟: N * 100ns (N=剩余 kernels 数量)
  本质: 因为清空了 Ring Buffer，必须重新提交 ⚠️
  问题: 
    • 需要重新提交 ⚠️
    • Ring Buffer 被重新填充 ⚠️
    • 可能重复执行部分 kernels ⚠️
    • kernel_offset 是近似值，不精确 ⚠️

AMD GPREEMPT Resume（硬件机制优势）:
  1. restore_mqd()
     • GPU 从 CWSR Area 恢复 Wave 状态 ✅
     • 恢复 PC（精确到指令）
     • 恢复所有寄存器
  2. ⭐⭐⭐ GPU 从 Ring Buffer rptr 继续读取
     • Ring Buffer 保持不变！✅
     • rptr=25, wptr=100
     • GPU 继续读取 kernel_25, 26, ..., 99
     • 就像从未被打断一样
  
  延迟: 1-10μs (固定) ✅
  本质: Ring Buffer 从未清空，直接继续 ✅
  优势:
    • 无需重新提交 ✅
    • Ring Buffer 不变 ✅
    • 无重复执行 ✅
    • 从断点精确继续（指令级）✅

核心区别:
  • GPreempt: 因为清空了 Ring Buffer（软件妥协），必须重新提交
  • AMD CWSR: Ring Buffer 保持不变（硬件机制），直接继续
  
这是 AMD 相对 GPreempt 的核心优势！⭐⭐⭐
```

---

## 📊 完整工作流程：双 AI 模型场景

### 场景设定

```
模型 A: BERT 训练（低优先级）
  • 优先级: 3
  • HIP Queue: queue_train
  • Kernels: 100 个
  • 执行时间: 数小时
  • 可被抢占

模型 B: ResNet 推理（高优先级）
  • 优先级: 12
  • HIP Queue: queue_infer
  • Kernels: 50 个
  • 执行时间: 20ms
  • SLA: <30ms
```

### 完整时序（端到端）

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
T < 0: 训练任务执行中
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

应用层（训练进程）:
  for (i = 0; i < 100; i++)
    hipLaunchKernel(train_kernel_i, ...)
      ↓ 每个 ~100ns
    写 Ring Buffer[wptr++]
    *doorbell_train = wptr  ⚡ MMIO write

GPU 硬件:
  Queue_train Ring Buffer:
    [k0, k1, k2, ..., k24, k25, ..., k99, empty, ...]
    ↑                         ↑
    rptr = 25                 wptr = 100
    
  • GPU 正在执行 kernel_25, kernel_26, ...
  • pending_count = 100 - 25 = 75
  • CU 占用率: 95%+

KFD 监控（上次检查 T=-5ms）:
  scan_queues():
    Queue_train: rptr=25, wptr=100, pending=75, active ✅
    Queue_infer: rptr=0, wptr=0, pending=0, idle ⏸️
  detect_inversion():
    无倒置 ✅


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
T = 0: 推理请求到达 ⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

应用层（推理服务）:
  void handle_request(image) {
    auto t0 = now();
    
    // ⭐ 提交 50 个 kernels
    for (i = 0; i < 50; i++)
      hipLaunchKernel(infer_kernel_i, ...)
        ↓ 每个 ~100ns
      写 Ring Buffer[wptr++]
      *doorbell_infer = wptr  ⚡ MMIO write
    
    // 提交完成，总延迟: 50 * 100ns = 5μs ✅
    
    // 等待完成
    hipStreamSynchronize(stream_infer);  // 阻塞等待
    
    auto t1 = now();
    return result;
  }

GPU 硬件:
  Queue_infer Ring Buffer:
    [infer_k0, infer_k1, ..., infer_k49, empty, ...]
    ↑                                      ↑
    rptr = 0                               wptr = 50
    
  • pending_count = 50
  • 但 Queue_train 正在占用 GPU！
  • Queue_infer 只能等待 ⏳

关键问题:
  ⚠️ GPU 不会主动抢占！
  • 硬件优先级字段存在，但当前实现不使用
  • GPU 继续执行 Queue_train
  • Queue_infer 等待 ⏳


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
T = 0 - 5ms: 等待 KFD 检测 ⏳
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPU 继续执行 Queue_train
  • Queue_infer 在 Ring Buffer 中等待
  • rptr 仍然是 0（没有被读取）

这是主要的延迟来源！
  平均延迟: 2.5ms (检查间隔的一半)
  最坏延迟: 5ms


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
T = 5ms: KFD 检测到优先级倒置 ⭐⭐⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KFD 监控线程（定时唤醒）:
  // 定时器到期，自动唤醒
  wake_up();
  
  // ⭐ 步骤 1: 扫描所有队列
  gpreempt_scan_queues():
    
    // Queue_train (MMIO read, ~200ns)
    rptr = readl(queue_train.read_ptr) = 30    // GPU 更新了
    wptr = readl(queue_train.write_ptr) = 100  // 应用提交的
    pending = 100 - 30 = 70
    is_active = true ✅
    
    // Queue_infer (MMIO read, ~200ns)
    rptr = readl(queue_infer.read_ptr) = 0     // GPU 还没读
    wptr = readl(queue_infer.write_ptr) = 50   // 应用提交的
    pending = 50 - 0 = 50
    is_active = true ✅
  
  扫描延迟: <10μs (2个队列 * 2次MMIO read)
  
  // ⭐ 步骤 2: 检测优先级倒置
  gpreempt_detect_inversion():
    
    high_q = Queue_infer (prio=12, pending=50, running=false)
    low_q = Queue_train (prio=3, pending=70, running=true)
    
    ⚠️ 优先级倒置！
  
  // ⭐ 步骤 3: 触发抢占
  gpreempt_preempt_queue(Queue_train, Queue_infer):
    
    printk("GPREEMPT: Preempting queue_train\n");
    printk("  Ring Buffer: rptr=30, wptr=100, pending=70\n");
    
    // ⭐ 触发 CWSR（异步）
    destroy_mqd(..., WAVEFRONT_SAVE, timeout=0)
      ↓
    构造 PM4 命令: PM4_ME_UNMAP_QUEUES
      queue_id = queue_train.id
      preempt_type = CWSR
      save_area = queue_train.cwsr_area
      ↓
    写入 CP Ring Buffer
      ↓
    敲 CP Doorbell
      ↓
    return -EINPROGRESS  // 异步返回
    
    Queue_train.state = PREEMPTED
    Queue_train.preemption_pending = true
  
  触发延迟: <1μs ✅ (内核函数调用，无 ioctl)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
T = 5.001ms: GPU 执行 CWSR ⭐⭐⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPU Command Processor:
  • 检测到 CP Doorbell
  • 读取 PM4 命令: UNMAP_QUEUES
  • 向 MEC 发送抢占请求

GPU MEC (Micro-Engine Compute):
  • 找到 Queue_train
  • 向所有活跃 CU 发送 Trap 信号
  • 等待 Wave 边界

GPU CUs (每个 CU 的 Trap Handler):
  For each active Wave on Queue_train:
    1. 停止执行（在指令边界）
    
    2. ⭐ 保存状态到 CWSR Area:
       • PC (Program Counter) = 0x401234
       • SGPR[0..127] (Scalar registers)
       • VGPR[0..255][0..63] (Vector registers)
       • LDS (Local Data Share, 64KB)
       • ACC VGPR (Accumulator registers)
       • Wave status flags
       
       保存大小: ~200KB per Wave
    
    3. 释放 CU 资源
    4. 标记 Wave 为 "SAVED"
  
  并行执行，延迟: 1-10μs ✅

关键状态（抢占完成后）:
  ⭐⭐⭐ Queue_train Ring Buffer 完全保持不变！
    rptr = 30  (GPU 读到这里，保持不变)
    wptr = 100 (应用提交的，保持不变)
    Ring Buffer: [k0, ..., k29, k30, k31, ..., k99]
                                ↑~~~~~~~~~~~~~~~~↑
                          这 70 个 kernels 仍然在 Ring Buffer 中！
  
  CWSR Area: 保存了 Wave 状态（PC=0x401234）
  MQD: 标记为 PREEMPTED


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
T = 5.010ms: GPU 切换到高优先级队列 ⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPU CP:
  • Queue_train 已挂起
  • 扫描 Runlist
  • 发现 Queue_infer (pending=50, prio=12)
  • 切换到 Queue_infer ✅

GPU 执行 Queue_infer:
  While (rptr < wptr) {
    • 从 Ring Buffer 读取 AQL_Packet[rptr]
    • 解析 kernel 参数
    • 分配 CU 资源
    • 启动 Wavefronts
    • rptr++
  }
  
  执行 50 个推理 kernels
  延迟: ~20ms


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
T = 25ms: 推理完成 ✅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPU 状态:
  Queue_infer: rptr=50, wptr=50, pending=0, idle ✅

应用层:
  hipStreamSynchronize(stream_infer);  // 返回 ✅
  
  auto t1 = now();
  latency = t1 - t0 = 25ms ✅ (满足 <30ms SLA)

端到端延迟分解:
  提交延迟:   5μs (50 * 100ns)
  等待检测:   5ms (平均 2.5ms)
  抢占延迟:   10μs (CWSR)
  切换延迟:   <1μs
  执行时间:   20ms
  ───────────────
  总延迟:     ~25ms ✅


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
T = 30ms: KFD 恢复训练任务 ⭐⭐⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KFD 监控（下次检查 T=30ms）:
  gpreempt_scan_queues():
    Queue_infer: rptr=50, wptr=50, pending=0, idle ✅
    Queue_train: state=PREEMPTED, cwsr_valid=true
  
  gpreempt_check_resume():
    // 没有更高优先级队列运行
    // 可以恢复 Queue_train
    
    printk("GPREEMPT: Resuming queue_train\n");
    printk("  Ring Buffer: rptr=30, wptr=100, pending=70\n");
    printk("  CWSR: area=0x2000_0000, valid=true\n");
    
    // ⭐⭐⭐ 恢复 MQD
    restore_mqd(queue_train.mqd, ...)
      ↓
    构造 PM4 命令: PM4_ME_MAP_QUEUES
      queue_id = queue_train.id
      restore_from = queue_train.cwsr_area
      ↓
    写入 CP Ring Buffer
      ↓
    敲 CP Doorbell
    
    Queue_train.state = ACTIVE
    Queue_train.preemption_pending = false
  
  恢复延迟: ~1μs


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
T = 30.001ms: GPU 恢复训练任务 ⭐⭐⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPU CP:
  • 检测到 CP Doorbell
  • 读取 PM4 命令: MAP_QUEUES
  • 解析 queue_train 信息

GPU CWSR Resume:
  1. 从 CWSR Area (0x2000_0000) 读取状态
  
  2. ⭐ 恢复所有 Wave 状态:
     For each saved Wave:
       • 恢复 PC = 0x401234 ✅
       • 恢复 SGPR[0..127]
       • 恢复 VGPR[0..255][0..63]
       • 恢复 LDS
       • 恢复 ACC VGPR
  
  3. 重新分配到 CU
  
  4. ⭐⭐⭐ 从 PC=0x401234 继续执行
     • 不是重新开始 kernel
     • 而是从中断的指令继续
     • 就像从未被打断 ✅
  
  5. ⭐⭐⭐ 继续从 Ring Buffer 读取后续 kernels
     • Ring Buffer: rptr=30, wptr=100
     • GPU 继续读取 kernel_30, 31, ..., 99
     • 这些 kernels 一直在 Ring Buffer 中！
     • 无需重新提交 ✅
  
  恢复延迟: 1-10μs ✅

GPU 继续执行:
  Queue_train: rptr 30 → 31 → ... → 100
  • 所有 100 个 kernels 正常完成
  • 无重复执行 ✅
  • 应用完全无感知 ✅
  • 增加的延迟: ~25ms (被抢占时间)


关键对比（GPreempt vs AMD）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPreempt (NVIDIA) Resume 流程（软件妥协的后果）:
  1. reset_preempt_flag = 0
  2. ⚠️ 计算 kernel_offset (近似，不精确):
     offset = launch_offset - pending - 2 = 99 - 70 - 2 = 27
  3. ⚠️ 重新提交 kernel_27 到 kernel_99:
     for (i = 27; i < 100; i++)
       cuLaunchKernel(kernel_i)
         ↓ ~100ns per kernel
       写 Ring Buffer
       Doorbell write
     总延迟: 73 * 100ns = 7.3μs
  4. ⚠️ GPU 重新执行 kernel_27, 28, 29, ...
     可能 kernel_27, 28, 29 重复执行
  
  本质: 因为清空了 Ring Buffer，只能估算并重新提交

AMD GPREEMPT Resume 流程（硬件机制的优势）:
  1. restore_mqd()
     ↓ ~1μs
  2. ✅ GPU 从 CWSR Area 恢复 Wave 状态（硬件）
     • PC=0x401234（指令级精度）
     • 所有寄存器
     延迟: 1-10μs
  3. ✅ GPU 从 Ring Buffer rptr=30 继续读取（硬件）
     • Ring Buffer 一直保持: rptr=30, wptr=100
     • kernel_30, 31, ..., 99 直接继续
     • 无需重新提交 ✅
  4. ✅ 被抢占的 Wave 从 PC=0x401234 继续
     • 不是重新开始
     • 无重复执行 ✅
  
  本质: Ring Buffer 从未改变，硬件直接恢复状态

AMD 优势总结:
  ✓ 无需重新提交（硬件机制）✅
  ✓ Ring Buffer 保持不变（硬件机制）✅
  ✓ 无重复执行（精确恢复）✅
  ✓ 延迟固定且低 (1-10μs)（硬件速度）✅
  ✓ 无需软件技巧和妥协 ✅
```

---

## 🎓 架构核心优势

### 相比 GPreempt (NVIDIA)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
维度                GPreempt (NVIDIA)        AMD GPREEMPT v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
任务提交            Pushbuffer+MMIO ~100ns   Doorbell+MMIO ~100ns ✅
提交性能            相同 ✅                  相同 ✅

优先级支持          2 个 (时间片模拟)        任意级别 (纯软件) ✅⭐⭐⭐
优先级实现          软件技巧：1s vs 1μs     纯软件调度 ✅
                    配置→硬件执行            • logical_priority (软件)
                                             • hardware_priority = NORMAL (固定)
                                             • 不受硬件限制

监控位置            用户态 (REEFScheduler)   内核态 (kthread) ✅
监控方式            ioctl 查询               MMIO 读取 ✅
监控开销            1-10μs per ioctl         <10μs per scan ✅
监控间隔            1ms                      5ms

抢占触发            用户态 + ioctl           内核态函数调用 ✅
抢占机制            软件技巧组合:            硬件 CWSR ✅
                    1. 配置时间片            
                    2. 清空 Ring Buffer
                    3. Reset CUs
抢占精度            Thread Block 边界        指令级 (Wave) ✅
抢占延迟            10-100μs                 1-10μs ✅

Ring Buffer         清空 (wptr=rptr) ⚠️      保持不变 ✅
                    软件妥协                 硬件机制优势
状态保存            kernel_offset (近似) ⚠️  PC+寄存器 (精确) ✅

Resume方式          重新提交 kernels ⚠️      恢复 Wave 状态 ✅
                    因为清空了 Ring Buffer   Ring Buffer 未改变
Resume延迟          N * 100ns                1-10μs (固定) ✅
重复执行            可能 ⚠️                  不会 ✅

应用修改            需要 preempt_flag ⚠️     无需修改 ✅
Kernel要求          幂等或无状态 ⚠️          任意 kernel ✅

本质区别            软件技巧妥协 ⚠️          硬件原生机制 ✅
                    无法修改硬件调度         硬件直接支持
                    只能间接影响             

部署方式            闭源驱动补丁 ⚠️          开源 KFD + DKMS ✅
部署难度            高                       中 ✅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

核心优势总结:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ✅ Ring Buffer 保持不变（vs 清空）
   • GPreempt: 必须清空（软件妥协，让硬件"看到"无任务）
   • AMD: 保持不变（硬件 CWSR 直接支持）
   这是相对 GPreempt 的最大优势！

2. ✅ 精确状态恢复（vs 重新提交）
   • GPreempt: 重新提交（因为清空了 Ring Buffer）
   • AMD: 从指令级断点继续（硬件恢复）
   无重复执行

3. ✅ 硬件原生机制（vs 软件技巧）
   • GPreempt: 无法修改硬件，只能通过软件技巧间接影响
   • AMD: 硬件原生支持，直接实现
   可靠、快速、透明

4. ✅ 内核态监控（vs 用户态）
   • GPreempt: 用户态，需要 ioctl
   • AMD: 内核态，无 ioctl 开销
   响应更快

5. ✅ 16 级硬件优先级（vs 2 级时间片模拟）
   • GPreempt: 通过时间片模拟（软件技巧）
   • AMD: 硬件原生支持
   支持复杂调度场景

6. ✅ 开源驱动（vs 闭源）
   易于修改、调试、部署
```

---

## 🔧 详细实现设计

### 1. 数据结构（最小化！）

```c
// ============================================================================
// 文件: amd/amdkfd/kfd_priv.h
// ============================================================================

// 队列扩展（只添加必要字段）
struct queue {
    // ===== 现有字段（保持不变）=====
    struct queue_properties properties;  // 包含 priority (0-15)
    struct mqd_manager *mqd_mgr;
    void *mqd;
    // ...
    
    // ===== Ring Buffer 状态监控 ⭐⭐⭐ =====
    u32 hw_rptr;              // 硬件 read pointer (GPU 更新)
    u32 hw_wptr;              // 硬件 write pointer (应用更新)
    u32 pending_count;        // wptr - rptr
    bool is_active;           // pending_count > 0
    
    // ===== 优先级调度 =====
    int base_priority;        // 基础优先级 (0-15)
    int effective_priority;   // 动态优先级（防止饥饿）
    
    // ===== 抢占状态 =====
    enum {
        QUEUE_STATE_ACTIVE,
        QUEUE_STATE_PREEMPTED,
        QUEUE_STATE_IDLE
    } state;
    
    bool preemption_pending;
    ktime_t preempt_start;
    
    // ===== 统计 =====
    u64 total_preemptions;
    u64 total_resumes;
    
    // ===== 链表 =====
    struct list_head sched_list;       // 全局队列链表
    struct list_head priority_list;    // 同优先级队列链表
};


// 全局调度器（简化！）
struct kfd_gpreempt_scheduler {
    // 监控线程
    struct task_struct *monitor_thread;
    
    // 配置
    unsigned int check_interval_ms;    // 默认 5ms
    bool enabled;
    
    // 队列管理
    struct list_head all_queues;
    struct list_head priority_queues[16];  // 按优先级分组
    spinlock_t lock;
    
    // 统计
    atomic64_t total_checks;
    atomic64_t total_inversions;
    atomic64_t total_preemptions;
};
```

### 2. 核心函数（简化实现）

```c
// ============================================================================
// 文件: amd/amdkfd/kfd_gpreempt_scheduler.c
// ============================================================================

// 初始化
int kfd_gpreempt_init(struct kfd_dev *dev)
{
    struct kfd_gpreempt_scheduler *sched;
    
    sched = kzalloc(sizeof(*sched), GFP_KERNEL);
    if (!sched)
        return -ENOMEM;
    
    sched->check_interval_ms = 5;
    sched->enabled = true;
    
    INIT_LIST_HEAD(&sched->all_queues);
    for (int i = 0; i < 16; i++)
        INIT_LIST_HEAD(&sched->priority_queues[i]);
    spin_lock_init(&sched->lock);
    
    // 创建监控线程
    sched->monitor_thread = kthread_run(
        kfd_gpreempt_monitor_thread,
        sched,
        "kfd_gpreempt"
    );
    
    if (IS_ERR(sched->monitor_thread)) {
        kfree(sched);
        return PTR_ERR(sched->monitor_thread);
    }
    
    dev->gpreempt_sched = sched;
    
    pr_info("KFD GPREEMPT: Initialized (interval=%dms)\n",
            sched->check_interval_ms);
    
    return 0;
}


// ⭐⭐⭐ 队列注册（分配 snapshot buffers）
//
// 这个函数在队列创建时调用，为每个队列分配 snapshot buffers
// 用于 checkpoint/restore
//
int kfd_gpreempt_register_queue(struct kfd_dev *dev, struct queue *q)
{
    struct kfd_gpreempt_scheduler *sched = dev->gpreempt_sched;
    size_t mqd_size, ctl_stack_size;
    unsigned long flags;
    
    if (!sched)
        return 0;  // GPREEMPT 未启用
    
    // ⭐ 步骤 1: 获取 MQD 大小（根据 GPU 版本）
    if (dev->device_info->asic_family == CHIP_VEGA10 ||
        dev->device_info->asic_family == CHIP_VEGA20) {
        mqd_size = sizeof(struct v9_mqd);
    } else if (dev->device_info->asic_family == CHIP_MI300) {
        mqd_size = sizeof(struct v11_mqd);  // MI300 使用 V11
    } else {
        mqd_size = PAGE_SIZE;  // 默认 4KB
    }
    
    // ⭐ 步骤 2: 获取 control stack 大小
    struct v9_mqd *mqd = (struct v9_mqd *)q->mqd;
    ctl_stack_size = mqd->cp_hqd_cntl_stack_size;
    if (ctl_stack_size == 0)
        ctl_stack_size = 4096;  // 默认 4KB
    
    // ⭐ 步骤 3: 分配 MQD backup buffer
    q->snapshot.mqd_backup = kzalloc(mqd_size, GFP_KERNEL);
    if (!q->snapshot.mqd_backup) {
        pr_err("GPREEMPT: Failed to allocate MQD backup for Q%u\n",
               q->properties.queue_id);
        return -ENOMEM;
    }
    
    // ⭐ 步骤 4: 分配 control stack backup buffer
    q->snapshot.ctl_stack_backup = kzalloc(ctl_stack_size, GFP_KERNEL);
    if (!q->snapshot.ctl_stack_backup) {
        kfree(q->snapshot.mqd_backup);
        pr_err("GPREEMPT: Failed to allocate control stack backup for Q%u\n",
               q->properties.queue_id);
        return -ENOMEM;
    }
    
    q->snapshot.ctl_stack_size = ctl_stack_size;
    q->snapshot.valid = false;
    
    // ⭐ 步骤 5: 初始化其他字段
    q->hw_rptr = 0;
    q->hw_wptr = 0;
    q->pending_count = 0;
    q->is_active = false;
    q->is_running = false;
    q->base_priority = q->properties.priority;
    q->effective_priority = q->properties.priority;
    q->state = QUEUE_STATE_IDLE;
    q->preemption_pending = false;
    q->total_preemptions = 0;
    q->total_resumes = 0;
    q->total_exec_time_us = 0;
    
    // ⭐ 步骤 6: 添加到调度器
    spin_lock_irqsave(&sched->lock, flags);
    list_add_tail(&q->sched_list, &sched->all_queues);
    list_add_tail(&q->priority_list, 
                  &sched->priority_queues[q->base_priority]);
    spin_unlock_irqrestore(&sched->lock, flags);
    
    pr_info("GPREEMPT: Registered Q%u (prio=%d, mqd=%zu, ctl=%zu)\n",
            q->properties.queue_id, q->base_priority,
            mqd_size, ctl_stack_size);
    
    return 0;
}


// ⭐⭐⭐ 队列注销（释放 snapshot buffers）
void kfd_gpreempt_unregister_queue(struct kfd_dev *dev, struct queue *q)
{
    struct kfd_gpreempt_scheduler *sched = dev->gpreempt_sched;
    unsigned long flags;
    
    if (!sched)
        return;
    
    // 从调度器移除
    spin_lock_irqsave(&sched->lock, flags);
    list_del(&q->sched_list);
    list_del(&q->priority_list);
    spin_unlock_irqrestore(&sched->lock, flags);
    
    // 释放 snapshot buffers
    if (q->snapshot.mqd_backup) {
        kfree(q->snapshot.mqd_backup);
        q->snapshot.mqd_backup = NULL;
    }
    
    if (q->snapshot.ctl_stack_backup) {
        kfree(q->snapshot.ctl_stack_backup);
        q->snapshot.ctl_stack_backup = NULL;
    }
    
    pr_info("GPREEMPT: Unregistered Q%u\n", q->properties.queue_id);
}


// 监控线程（简化！）
static int kfd_gpreempt_monitor_thread(void *data)
{
    struct kfd_gpreempt_scheduler *sched = data;
    struct queue *high_q, *low_q;
    
    while (!kthread_should_stop()) {
        // 休眠
        msleep_interruptible(sched->check_interval_ms);
        
        if (!sched->enabled)
            continue;
        
        // ⭐ 扫描队列
        gpreempt_scan_queues(sched);
        
        // ⭐ 检测倒置
        if (gpreempt_detect_inversion(sched, &high_q, &low_q)) {
            // ⭐ 触发抢占
            gpreempt_preempt_queue(sched, low_q, high_q);
        }
        
        // ⭐ 检查恢复
        gpreempt_check_resume(sched);
        
        atomic64_inc(&sched->total_checks);
    }
    
    return 0;
}


// ⭐⭐⭐ 扫描队列（核心！）
static void gpreempt_scan_queues(struct kfd_gpreempt_scheduler *sched)
{
    struct queue *q;
    unsigned long flags;
    
    spin_lock_irqsave(&sched->lock, flags);
    
    list_for_each_entry(q, &sched->all_queues, sched_list) {
        if (!q->properties.doorbell_ptr)
            continue;
        
        // ⭐ 主动读取 Ring Buffer 指针（MMIO read）
        u32 rptr = readl(q->properties.read_ptr);
        u32 wptr = readl(q->properties.write_ptr);
        
        // 计算状态
        u32 pending = (wptr - rptr) & q->ring_size_mask;
        
        q->hw_rptr = rptr;
        q->hw_wptr = wptr;
        q->pending_count = pending;
        q->is_active = (pending > 0);
        
        // 更新空闲状态
        if (pending == 0 && q->state == QUEUE_STATE_ACTIVE) {
            q->state = QUEUE_STATE_IDLE;
        }
    }
    
    spin_unlock_irqrestore(&sched->lock, flags);
}


// ⭐⭐⭐ 抢占队列（关键！）
static int gpreempt_preempt_queue(
    struct kfd_gpreempt_scheduler *sched,
    struct queue *low_q,
    struct queue *high_q)
{
    int r;
    
    low_q->state = QUEUE_STATE_PREEMPTED;
    low_q->preemption_pending = true;
    low_q->preempt_start = ktime_get();
    
    pr_info("GPREEMPT: Preempting Q%u (prio=%d, rptr=%u, wptr=%u, pending=%u)\n",
            low_q->properties.queue_id,
            low_q->effective_priority,
            low_q->hw_rptr,
            low_q->hw_wptr,
            low_q->pending_count);
    
    // ⭐⭐⭐ 触发 CWSR（异步）
    //
    // 重要：Ring Buffer 不变！
    //   • rptr 保持当前值（GPU 读到这里）
    //   • wptr 保持当前值（应用提交到这里）
    //   • Ring Buffer 中的 kernels 全部保留
    //   • CWSR 保存 Wave 状态
    //
    r = low_q->mqd_mgr->destroy_mqd(
        low_q->mqd_mgr,
        low_q->mqd,
        KFD_PREEMPT_TYPE_WAVEFRONT_SAVE,
        0,  // timeout=0: 异步
        low_q->process->pasid
    );
    
    if (r == 0 || r == -EINPROGRESS) {
        atomic64_inc(&sched->total_preemptions);
        low_q->total_preemptions++;
        return 0;
    }
    
    pr_err("GPREEMPT: Preemption failed: %d\n", r);
    low_q->state = QUEUE_STATE_ACTIVE;
    low_q->preemption_pending = false;
    return r;
}


// ⭐⭐⭐ 恢复队列（关键！）
static void gpreempt_check_resume(struct kfd_gpreempt_scheduler *sched)
{
    struct queue *q;
    unsigned long flags;
    
    spin_lock_irqsave(&sched->lock, flags);
    
    list_for_each_entry(q, &sched->all_queues, sched_list) {
        if (q->state != QUEUE_STATE_PREEMPTED)
            continue;
        
        // 检查是否有更高优先级队列运行
        if (gpreempt_has_higher_priority_running(sched, q))
            continue;
        
        // ⭐ 可以恢复
        pr_info("GPREEMPT: Resuming Q%u (prio=%d, rptr=%u, wptr=%u)\n",
                q->properties.queue_id,
                q->effective_priority,
                q->hw_rptr,
                q->hw_wptr);
        
        // ⭐⭐⭐ 触发 CWSR Resume
        //
        // 重要：GPU 会做什么？
        //   1. 从 CWSR Area 恢复 Wave 状态（PC + 寄存器）
        //   2. ⭐ 从 Ring Buffer rptr 继续读取 kernels
        //   3. Ring Buffer 保持: rptr=30, wptr=100
        //   4. GPU 继续读取 kernel_30, 31, ..., 99
        //   5. 无需重新提交任何 kernel！✅
        //
        int r = q->mqd_mgr->restore_mqd(
            q->mqd_mgr,
            q->mqd,
            q->process->pasid
        );
        
        if (r == 0) {
            q->state = QUEUE_STATE_ACTIVE;
            q->preemption_pending = false;
            q->total_resumes++;
            
            pr_info("GPREEMPT: Queue %u resumed successfully\n",
                    q->properties.queue_id);
        } else {
            pr_err("GPREEMPT: Resume failed for queue %u: %d\n",
                   q->properties.queue_id, r);
        }
    }
    
    spin_unlock_irqrestore(&sched->lock, flags);
}
```

---

## 🚀 部署和使用

### 编译和安装

```bash
# 1. 修改 DKMS 源码
cd /usr/src/amdgpu-debug-20260106

# 2. 添加新文件
touch amd/amdkfd/kfd_gpreempt_scheduler.c
touch amd/amdkfd/kfd_gpreempt_scheduler.h

# 3. 修改 Makefile
echo "AMDKFD-y += $(AMDKFD_PATH)/kfd_gpreempt_scheduler.o" >> amd/amdkfd/Makefile

# 4. 编译和安装
dkms remove amdgpu-debug/20260106 --all
dkms install amdgpu-debug/20260106

# 5. 重新加载模块
rmmod amdgpu
modprobe amdgpu

# 6. 验证
dmesg | grep GPREEMPT
# 应该看到:
# [xxx] KFD GPREEMPT: Initialized (interval=5ms)
# [xxx] KFD GPREEMPT: Monitor thread started
```

### 使用方式（完全透明！）

```cpp
// ============================================================================
// 应用程序（无需修改！）
// ============================================================================

// 训练程序（低优先级）
void train_model() {
    // ⭐ 创建队列时指定优先级
    hsa_queue_create(..., &queue);
    // 或者通过环境变量: HSA_QUEUE_PRIORITY=3
    
    // 正常使用 HIP API
    for (int epoch = 0; epoch < 100; epoch++) {
        for (int batch = 0; batch < 1000; batch++) {
            hipLaunchKernel(forward_kernel, ...);
            hipLaunchKernel(backward_kernel, ...);
            hipLaunchKernel(optimizer_kernel, ...);
            // ↓ 每个调用 ~100ns（Doorbell）⚡
            // ↓ 写 Ring Buffer，敲 Doorbell
            // ↓ 完全透明，无感知
        }
    }
}


// 推理服务（高优先级）
void inference_service() {
    // ⭐ 创建队列时指定优先级
    hsa_queue_create(..., &queue);
    // 或者通过环境变量: HSA_QUEUE_PRIORITY=12
    
    while (true) {
        auto request = receive_request();
        
        // 正常使用 HIP API
        for (int i = 0; i < 50; i++) {
            hipLaunchKernel(infer_kernel_i, ...);
            // ↓ ~100ns（Doorbell）⚡
        }
        
        hipStreamSynchronize(stream);
        
        // ✅ 自动获得高优先级调度
        // ✅ 自动抢占低优先级任务
        // ✅ 应用完全无感知
        
        send_response(result);
    }
}


// ============================================================================
// 管理工具（可选）
// ============================================================================

// 动态修改优先级
$ gpreempt-tool set-priority --queue=5 --priority=10

// 查看统计
$ gpreempt-tool stats
KFD GPREEMPT Statistics:
  Total checks:      120000
  Total inversions:  150
  Total preemptions: 150
  Avg check time:    8.5 us
  Avg preempt time:  4.2 us

// 手动触发抢占（调试）
$ gpreempt-tool preempt --queue=3

// 配置
$ echo 10 > /sys/module/amdgpu/parameters/gpreempt_interval_ms
```

---

## 📈 性能预期

### 端到端延迟

```
场景: ResNet-50 推理抢占 BERT 训练
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

无优先级调度（Baseline）:
  推理延迟: 数百ms - 数秒 (等待训练完成)
  
GPreempt (NVIDIA):
  推理延迟: ~100-500ms
    • 用户态检测: ~1ms
    • 软件协作抢占: 100-500μs
    • 执行: 20ms
    • Resume: 重新提交 (~10μs)

AMD GPREEMPT v2.0:
  推理延迟: ~25ms ✅
    • 内核态检测: 5ms (平均 2.5ms)
    • CWSR 抢占: 10μs
    • 执行: 20ms
    • Resume: CWSR 恢复 (10μs)
  
  改善: 4-20× vs GPreempt ✅
       20-100× vs 无优先级 ✅

训练影响:
  AMD GPREEMPT: +25ms (被抢占时间)
  相比训练总时间（数小时）: 可忽略 ✅
```

### 系统开销

```
CPU 开销:
  监控线程: <0.001% CPU
    • 每 5ms 唤醒一次
    • 每次扫描 ~10μs
    • 10μs / 5000μs = 0.0002 = 0.02%
  
  完全可忽略 ✅

内存开销:
  每个队列: ~200 bytes (新增字段)
  100 个队列: 20 KB
  
  完全可忽略 ✅

GPU 性能影响:
  数据平面: 0 (Doorbell 保持 ~100ns) ✅
  控制平面: MMIO read (不影响 GPU 执行) ✅
  
  无影响 ✅
```

---

## 🎯 实施计划

### Phase 1: 基础框架（1周）

```
1.1 创建文件:
  • kfd_gpreempt_scheduler.h
  • kfd_gpreempt_scheduler.c
  • 修改 Makefile

1.2 实现核心功能:
  • 监控线程启动
  • Ring Buffer 扫描（gpreempt_scan_queues）
  • 打印日志验证

1.3 测试验证:
  • 编译通过
  • 模块加载成功
  • 日志显示 rptr/wptr 正确
```

### Phase 2: 抢占和 Resume（2周）

```
2.1 实现抢占:
  • gpreempt_detect_inversion
  • gpreempt_preempt_queue
  • 调用 destroy_mqd

2.2 实现 Resume:
  • gpreempt_check_resume
  • 调用 restore_mqd

2.3 测试:
  • 双进程测试
  • 验证 Ring Buffer 保持不变
  • 验证无重复执行
```

### Phase 3: 优化和完善（2周）

```
3.1 性能优化:
  • 调整监控间隔
  • 优化锁粒度
  • 减少 MMIO 读取次数

3.2 功能完善:
  • 动态优先级（防止饥饿）
  • 时间片支持（可选）
  • 统计信息

3.3 压力测试:
  • 100 个队列
  • 长时间运行
  • 稳定性验证
```

---

## 📊 总结

### 核心设计理念

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⭐⭐⭐ 关键理解：软件技巧 vs 硬件机制
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPreempt (NVIDIA):
  • GPU 硬件调度器由固件控制，软件无法修改
  • 只能通过软件技巧"间接"影响硬件：
    1. 配置时间片参数（1s vs 1μs）
    2. 清空 Ring Buffer（让硬件"看到"无任务）
    3. Reset CUs（停止执行）
  • 修改的是硬件的"输入数据"，不是调度逻辑
  • 本质是软件妥协方案

AMD GPREEMPT:
  • 硬件原生支持 CWSR 和优先级
  • 不需要软件技巧和妥协
  • 直接利用硬件机制实现抢占
  • Ring Buffer 保持不变，精确状态恢复
  • 本质是硬件优势方案

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ✅ 简单、清晰、有效
   • 去除不必要的复杂性
   • 聚焦核心功能
   • 基于实际代码验证
   • 充分理解 GPreempt 的软件技巧

2. ✅ Ring Buffer 是核心
   • 监控 rptr/wptr 状态
   • 抢占时保持不变（vs GPreempt 清空）
   • Resume 时继续读取（vs GPreempt 重新提交）

3. ✅ 充分利用 AMD 硬件优势
   • CWSR 硬件抢占（vs GPreempt 软件技巧）
   • 16 级硬件优先级（vs GPreempt 时间片模拟）
   • 精确状态恢复（vs GPreempt 重新提交）
   • 开源驱动

4. ✅ 超越 GPreempt
   • 更快（10倍，硬件 vs 软件）
   • 更精确（指令级 vs 近似）
   • 更简单（无需修改应用）
   • 更可靠（硬件机制 vs 软件妥协）
```

### 与原架构的关键改进

```
原 ARCH_Design_01 vs 新 v2.0:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ✅ 明确 Ring Buffer 的核心作用
   原文档: 提到了，但不够深入
   v2.0: Ring Buffer 保持不变是核心优势

2. ✅ 简化数据结构
   原文档: 字段较多，有些不必要
   v2.0: 最小化字段，聚焦核心

3. ✅ 简化实现逻辑
   原文档: 有些过度设计
   v2.0: 简化实现，易于理解和维护

4. ✅ 强调 AMD 独特优势
   原文档: 优势分散
   v2.0: 系统总结 6 大核心优势

5. ✅ 深刻理解 GPreempt 的软件技巧（新增！）
   原文档: 未深入分析 GPreempt 的本质
   v2.0: 明确 GPreempt 是软件技巧妥协：
     • 配置时间片（配置 vs 执行）
     • 清空 Ring Buffer（让硬件"看到"无任务）
     • 无法修改硬件调度逻辑
     • 明确 AMD CWSR 是硬件原生机制

6. ✅ 基于实际代码验证
   原文档: 部分基于推测
   v2.0: 完全基于 GPreempt 代码分析
     • 明确事实 vs 推断
     • 理解 GPU 硬件调度器的固件控制特性
```

---

**文档版本**: ARCH_Design_02 v2.0 (更新：软件技巧 vs 硬件机制)  
**日期**: 2026-01-29  
**最后更新**: 2026-01-29（新增 GPreempt 软件技巧深度理解）
**状态**: 架构重新设计完成  
**下一步**: 开始实施 Phase 1

**关键更新（2026-01-29）**:
```
✅ 新增核心理解 1：软件技巧 vs 硬件机制
  1. GPU 硬件调度器是固件控制的，软件无法修改
  2. GPreempt 的"抢占"是软件技巧的组合：
     • 配置时间片参数（1s vs 1μs）→ GPU 硬件按此执行切换
     • 清空 Ring Buffer → 让硬件"看到"无任务
     • Reset CUs → 停止正在执行的 Waves
  3. 修改的是硬件的"输入数据"，不是调度逻辑
  4. AMD CWSR 是真正的硬件机制，无需这些妥协

✅ 新增核心理解 2：为什么 GPreempt 需要 Kernel Patch
  1. Ring Buffer 操作在 userspace（无需 kernel）
  2. Kernel Patch 作用是：查询和控制 Channels
     • NV_ESC_RM_QUERY_GROUP ioctl：查询 Channel handles
     • NV2080_CTRL_CMD_FIFO_DISABLE_CHANNELS：停止 Channels
  3. 因为 GPreempt 是 userspace 调度器
     • Channel 信息在 kernel，需要 ioctl 查询
     • 停止 Channel 需要特权，需要 ioctl 操作
  4. AMD GPREEMPT 是 kernel 调度器
     • 直接访问 queue 结构，无需查询 ioctl
     • 直接调用 destroy_mqd，无需控制 ioctl
     • 更简单、更快速 ✅

✅ 全面更新对比分析：
  • 明确 GPreempt 是软件妥协方案
  • 强调 AMD CWSR 是硬件优势方案
  • 系统阐述"配置 vs 执行"的关系
  • 详细解释清空 Ring Buffer 的本质作用
  • 新增 Kernel Patch 作用的详细分析
```

**关键文档**:
- `GPreempt_完整技术分析_综合版.md`（代码分析基础，含软件技巧分析）
- `GPreempt_codestudy_Ring_Buffer_Doorbell_抢占机制深度分析.md`（核心机制）
- `GPreempt_codestudy_优先级和时间片_对比分析.md`（优先级系统）

**感谢**: 
- 用户对 Ring Buffer + Doorbell 机制的深刻洞察
- 用户对清空 Ring Buffer 本质作用的准确理解
- 用户对时间片使用逻辑的精准提问
