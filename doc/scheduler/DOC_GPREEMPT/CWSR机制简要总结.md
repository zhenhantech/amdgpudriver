# CWSR机制简要总结

> KFD中的Compute Wave Save/Restore抢占机制核心要点  
> 2026-01-27

---

## 🎯 什么是CWSR？

**CWSR = Compute Wave Save/Restore**

AMD GPU的**硬件支持的wave级别抢占机制**，是实现GPU抢占式调度的基础。

```
CWSR = 微秒级GPU任务抢占 + 完整状态保存/恢复
```

---

## 🏗️ 三层架构

```
┌──────────────────────────────────────┐
│  GPREEMPT (User Layer)               │  ← TODO: from GPREEMPT
│  • Sched/preempt policy              │
│  • ioctl interface                   │
└──────────┬───────────────────────────┘
           │
┌──────────▼───────────────────────────┐
│  KFD CWSR APIs                       │   ← Ready: driver
│  • checkpoint_mqd()                  │
│  • restore_mqd()                     │
│  • destroy_mqd(WAVEFRONT_SAVE)       │
└──────────┬───────────────────────────┘
           │
┌──────────▼───────────────────────────┐
│ HW CWSR Mechanism                    │  ← Ready: GPU (MI308)
│  • Trap Handler (ASM)                │
│  • Register/LDS Save and Restore     │
└──────────────────────────────────────┘
```

---

## 💾 CWSR保存了什么？

```
每个Wave的完整状态:
├── 程序计数器 (PC)              ← 恢复到正确位置
├── 标量寄存器 (SGPRs)           ← 标量数据
├── 向量寄存器 (VGPRs)           ← 向量数据
├── 累加器寄存器 (ACC VGPRs)     ← GFX9.1+
├── Local Data Share (LDS)       ← 共享内存
└── 硬件状态寄存器               ← Wave状态
```

---

## ⚡ 三种抢占类型

| 类型 | 延迟 | 状态保存 | 用途 |
|------|------|----------|------|
| **WAVEFRONT_DRAIN** | 1-10ms | ❌ | 队列销毁 |
| **WAVEFRONT_RESET** | 10-50μs | ❌ | 错误恢复 |
| **WAVEFRONT_SAVE** (CWSR) | 1-10μs | ✅ | ⭐ 抢占调度 |

**GPREEMPT使用WAVEFRONT_SAVE！**

---

## 📊 内存开销

### MI300示例（304 CUs）

```
每个队列的CWSR内存:
├── Control Stack:    ~8.6 MB    (wave状态)
├── Workgroup Data:   ~177 MB    (LDS等)
└── Debug Memory:     ~0.3 MB    (调试信息)
    ──────────────────────────
    Total:            ~186 MB per queue

32个队列:              ~5.8 GB ⚠️ 巨大！
```

**这是CWSR的主要成本！**

---

## 🔄 工作流程

### 抢占（Save）

```
应用请求抢占
    ↓
GPREEMPT: kfd_queue_preempt_single()
    ↓
KFD: checkpoint_mqd() → 保存MQD和控制栈到备份
    ↓
KFD: destroy_mqd(WAVEFRONT_SAVE) → 触发硬件
    ↓
硬件: Trap Handler执行 → 保存所有状态到CWSR内存
    ↓
Wave挂起 ✅
```

### 恢复（Restore）

```
应用请求恢复
    ↓
GPREEMPT: kfd_queue_resume_single()
    ↓
KFD: restore_mqd() → 从备份恢复MQD和控制栈
    ↓
KFD: load_mqd() → 重新加载到GPU
    ↓
硬件: 从CWSR内存恢复所有状态
    ↓
Wave继续执行 ✅ （从断点处）
```

---

## 💻 关键代码位置

### 1. KFD接口

```c
// kfd_mqd_manager_v9.c
void checkpoint_mqd(struct mqd_manager *mm, void *mqd, 
                   void *mqd_dst, void *ctl_stack_dst);

void restore_mqd(struct mqd_manager *mm, void **mqd,
                struct kfd_mem_obj *mqd_mem_obj, ...);
```

### 2. 抢占类型选择

```c
// kfd_device_queue_manager.c:1026-1028
ret = mqd_mgr->destroy_mqd(mqd_mgr, q->mqd,
        (dqm->dev->kfd->cwsr_enabled ?          // ⭐ 检查CWSR
         KFD_PREEMPT_TYPE_WAVEFRONT_SAVE :      // ✅ 启用
         KFD_PREEMPT_TYPE_WAVEFRONT_DRAIN),     // ❌ 禁用
        timeout, q->pipe, q->queue);
```

### 3. Trap Handler

```bash
# 汇编源代码
/usr/src/amdgpu-6.12.12-2194681.el8/amd/amdkfd/
├── cwsr_trap_handler_gfx9.asm    # MI100/MI200/MI300
├── cwsr_trap_handler_gfx10.asm   # Navi
└── cwsr_trap_handler_gfx12.asm   # 最新
```

---

## ✅ Phase 1的关键发现

### 我们发现了什么？

```
1. ✅ AMD GPU硬件已支持CWSR
2. ✅ KFD驱动已完整实现CWSR
3. ✅ checkpoint/restore接口已存在
4. ✅ 只需要封装和暴露给用户空间
```

### 我们实现了什么？

```c
// Phase 1: 封装CWSR接口
int kfd_queue_preempt_single(struct queue *q,
                              enum kfd_preempt_type type,
                              unsigned int timeout)
{
    // ⭐ 使用现有的checkpoint_mqd
    mqd_mgr->checkpoint_mqd(...);
    
    // ⭐ 使用现有的destroy_mqd
    mqd_mgr->destroy_mqd(..., WAVEFRONT_SAVE, ...);
    
    return 0;
}

// 暴露到用户空间
ioctl(kfd_fd, AMDKFD_IOC_PREEMPT_QUEUE, &args);
```

---

## 🎊 总结

### CWSR的价值

```
✅ 微秒级抢占延迟      (vs 毫秒级等待)
✅ 完整状态保存         (vs 丢失进度)
✅ 硬件加速执行         (vs 纯软件模拟)
✅ 所有现代AMD GPU支持  (vs 特定型号)
```

### CWSR vs GPREEMPT

| 层面 | CWSR | GPREEMPT |
|------|------|----------|
| **定位** | 底层机制 | 上层策略 |
| **功能** | 状态保存/恢复 | 任务调度 |
| **实现者** | AMD硬件+KFD | 我们 |
| **状态** | ✅ 已完成 | 🚧 进行中 |

### 实施影响

```
因为CWSR已经完整实现:
→ GPREEMPT实施变得简单
→ 从"从零开发" 变为 "接口封装"
→ Phase 1: 8分钟完成 (vs 预计1-2天)
→ 风险大幅降低
```

---

## 🚀 下一步

### Phase 2: DKMS集成

```bash
# 编译驱动
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpu_DKMS/
   amdgpu-6.12.12-2194681.el8
make -j$(nproc)

# 安装测试
sudo make modules_install
sudo modprobe -r amdgpu
sudo modprobe amdgpu
```

### Phase 3: 测试验证

```c
// 测试CWSR抢占
1. 创建队列并提交kernel
2. 调用 ioctl(AMDKFD_IOC_PREEMPT_QUEUE)
3. 验证kernel被挂起
4. 调用 ioctl(AMDKFD_IOC_RESUME_QUEUE)
5. 验证kernel继续执行 ✅
```

---

**关键洞察**：

> *"CWSR是AMD GPU抢占调度的基石。理解了CWSR，就理解了为什么GPREEMPT的实施如此简单——因为底层已经完全就绪。"*

---

**详细文档**: 参见 `KFD_CWSR抢占机制深度分析.md`  
**代码实现**: `/mnt/md0/zhehan/code/coderampup/private_github/amdgpu_DKMS/`  
**状态**: ✅ Phase 1完成，准备Phase 2编译测试

