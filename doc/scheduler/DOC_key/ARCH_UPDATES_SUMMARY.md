# 架构文档更新总结

**日期**: 2026-01-29  
**更新范围**: ARCH_Design_02 和 ARCH_Design_03  
**版本**: v2.1 / v3.1

---

## 📊 更新概述

基于 `CWSR_API_USAGE_REFERENCE.md`（KFD CRIU 代码分析），对两个架构文档进行了全面修正，修复了 CWSR API 使用中的 5 个关键问题。

---

## ✅ 更新内容详细列表

### 1. ARCH_Design_02_AMD_GPREEMPT_redesign_v2.md

#### 版本更新
- 版本号: 2.0 → **2.1**
- 状态: 架构重新设计 → **架构修正完成，准备实施**

#### 数据结构修正

**struct queue** 新增字段：
```c
// 新增现有必要字段
struct kfd_mem_obj *mqd_mem_obj;     // MQD 内存对象
uint64_t gart_mqd_addr;              // GART 地址

// ⭐⭐⭐ 新增 snapshot 结构
struct {
    void *mqd_backup;          // MQD 备份 buffer
    void *ctl_stack_backup;    // Control stack 备份 buffer
    size_t ctl_stack_size;     // Control stack 大小
    bool valid;                // Snapshot 是否有效
} snapshot;
```

#### 函数修正

**gpreempt_preempt_queue()** 修正：
```c
// ⭐ 新增步骤 1: Checkpoint MQD
q->mqd_mgr->checkpoint_mqd(
    q->mqd_mgr,
    q->mqd,
    q->snapshot.mqd_backup,
    q->snapshot.ctl_stack_backup
);
q->snapshot.valid = true;

// ⭐ 修正步骤 2: destroy_mqd 参数
r = q->mqd_mgr->destroy_mqd(
    q->mqd_mgr,
    q->mqd,
    KFD_PREEMPT_TYPE_WAVEFRONT_SAVE,
    0,
    q->pipe,    // ⭐ 修正：使用 pipe
    q->queue    // ⭐ 修正：使用 queue（不是 pasid）
);
```

**gpreempt_check_resume()** 修正：
```c
// ⭐ 新增 snapshot 有效性检查
if (!q->snapshot.valid) {
    pr_warn("Queue has no valid snapshot\n");
    continue;
}

// ⭐ 修正步骤 1: restore_mqd 参数（8个）
q->mqd_mgr->restore_mqd(
    q->mqd_mgr,
    &q->mqd,                      // ⭐ double pointer
    q->mqd_mem_obj,
    &q->gart_mqd_addr,
    &q->properties,
    q->snapshot.mqd_backup,       // 从 snapshot
    q->snapshot.ctl_stack_backup,
    q->snapshot.ctl_stack_size
);

// ⭐ 新增步骤 2: load_mqd（激活队列）
int r = q->mqd_mgr->load_mqd(
    q->mqd_mgr,
    q->mqd,
    q->pipe,
    q->queue,
    &q->properties,
    q->process->mm
);
```

#### 新增函数

**kfd_gpreempt_register_queue()** - 队列注册：
```c
// ⭐ 分配 MQD backup buffer
q->snapshot.mqd_backup = kzalloc(mqd_size, GFP_KERNEL);

// ⭐ 分配 control stack backup buffer
q->snapshot.ctl_stack_backup = kzalloc(ctl_stack_size, GFP_KERNEL);

q->snapshot.ctl_stack_size = ctl_stack_size;
q->snapshot.valid = false;
```

**kfd_gpreempt_unregister_queue()** - 队列注销：
```c
// 释放 snapshot buffers
if (q->snapshot.mqd_backup) {
    kfree(q->snapshot.mqd_backup);
}
if (q->snapshot.ctl_stack_backup) {
    kfree(q->snapshot.ctl_stack_backup);
}
```

---

### 2. ARCH_Design_03_AMD_GPREEMPT_XSCHED.md

#### 版本更新
- 版本号: 3.0 → **3.1**
- 新增更新说明章节

#### 同步修正内容

所有修正内容与 ARCH_Design_02 相同：

1. ✅ struct queue 添加 snapshot 字段
2. ✅ gpreempt_preempt_queue 修正
3. ✅ gpreempt_check_resume_queues 修正
4. ✅ kfd_gpreempt_register_queue 添加 snapshot 分配
5. ✅ 新增 kfd_gpreempt_unregister_queue

#### 特殊说明

文档中同时包含：
- **方案 A**: 纯内核态实现（已全部修正）
- **方案 B**: XSched Lv3 + KFD 混合（主要影响是 ioctl 接口定义）

---

## 🎯 修正要点总结

### 关键问题修正

| 问题 | 修正前 | 修正后 | 影响 |
|------|--------|--------|------|
| **destroy_mqd 参数** | `(..., pasid)` | `(..., pipe, queue)` | 🔴 无法触发 CWSR |
| **restore_mqd 参数** | `(mgr, mqd, pasid)` 3个 | `(mgr, &mqd, mem_obj, gart, props, mqd_src, ctl_src, size)` 8个 | 🔴 无法恢复 |
| **checkpoint_mqd** | 缺失 | 在 preempt 前调用 | 🔴 无数据恢复 |
| **load_mqd** | 缺失 | 在 restore 后调用 | 🔴 队列不激活 |
| **snapshot 字段** | 缺失 | 添加到 struct queue | 🔴 无处存储 |

### 正确的完整流程

#### Preempt（抢占）
```
1. checkpoint_mqd()      // 保存 MQD 到 snapshot
2. destroy_mqd(pipe, queue)  // 触发 CWSR
```

#### Resume（恢复）
```
1. restore_mqd(&mqd, ..., mqd_src, ctl_src, size)  // 8个参数，恢复 MQD
2. load_mqd()           // 激活队列
```

---

## 📚 参考文档

### 依据文档
- `CWSR_API_USAGE_REFERENCE.md` - KFD CRIU 代码分析
- `ARCH_DESIGN_CORRECTIONS_CWSR_API.md` - 详细修正说明

### KFD 源码参考
1. **CRIU Checkpoint**: `kfd_process_queue_manager.c:800-865`
2. **CRIU Restore**: `kfd_process_queue_manager.c:310-435`
3. **MQD Manager**: `kfd_mqd_manager_v9.c:436-474`

---

## ✅ 验证清单

更新后的文档已确保：

- [x] destroy_mqd 使用正确参数（pipe, queue）
- [x] restore_mqd 使用 8 个参数，&q->mqd 为 double pointer
- [x] preempt 前调用 checkpoint_mqd
- [x] restore 后调用 load_mqd
- [x] struct queue 包含 snapshot 字段
- [x] register_queue 分配 snapshot buffers
- [x] unregister_queue 释放 snapshot buffers
- [x] 所有代码示例已更新
- [x] 版本号已更新
- [x] 更新说明已添加

---

## 🔄 后续步骤

### 实施建议

1. **代码实现时**：
   - 严格按照更新后的文档实现
   - 特别注意 API 参数顺序和类型
   - 确保 snapshot 的分配和释放

2. **测试验证时**：
   - 验证 checkpoint 是否保存了数据
   - 验证 restore 是否正确恢复
   - 验证 load_mqd 是否激活队列
   - 检查内存泄漏（snapshot buffers）

3. **如有疑问**：
   - 参考 `ARCH_DESIGN_CORRECTIONS_CWSR_API.md`
   - 对照 KFD CRIU 源码
   - 查看 `CWSR_API_USAGE_REFERENCE.md`

---

## 📊 影响范围

### 直接影响
- ✅ 所有使用 CWSR API 的代码必须按新方式实现
- ✅ 数据结构定义必须包含新增字段
- ✅ 初始化流程必须分配 snapshot buffers

### 间接影响
- ⚠️ 文档中的所有代码示例已更新
- ⚠️ 时间线和流程图保持不变（CWSR 行为本质未变）
- ⚠️ 架构设计原则保持不变

---

## 🎉 更新完成状态

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ ARCH_Design_02_AMD_GPREEMPT_redesign_v2.md
   - 版本更新: v2.0 → v2.1
   - 数据结构: 已添加 snapshot 字段
   - preempt 函数: 已添加 checkpoint_mqd，修正参数
   - resume 函数: 已添加 load_mqd，修正参数
   - 新增函数: register/unregister queue
   - 更新说明: 已添加
   
✅ ARCH_Design_03_AMD_GPREEMPT_XSCHED.md
   - 版本更新: v3.0 → v3.1
   - 数据结构: 已添加 snapshot 字段
   - preempt 函数: 已添加 checkpoint_mqd，修正参数
   - resume 函数: 已添加 load_mqd，修正参数
   - 新增函数: register/unregister queue
   - 更新说明: 已添加

✅ ARCH_DESIGN_CORRECTIONS_CWSR_API.md
   - 详细修正说明
   - 错误 vs 正确对比
   - 完整代码示例

✅ 本总结文档
   - 更新内容汇总
   - 验证清单
   - 后续步骤指南
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

**更新完成日期**: 2026-01-29  
**更新人**: AI Assistant  
**状态**: ✅ 完成，可以开始实施
