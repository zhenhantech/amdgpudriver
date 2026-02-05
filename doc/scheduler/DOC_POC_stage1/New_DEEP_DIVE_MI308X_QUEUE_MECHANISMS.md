# MI308X GPU队列管理机制深度分析

## 📌 核心结论（先读这个）⭐⭐⭐⭐⭐

```
✅ MI308X只使用CPSCH模式（enable_mes=0）
❌ MI308X不使用MES模式（MES用于更新GPU）
✅ 队列管理通过HWS + Runlist IB实现
✅ POC应基于CPSCH模式设计
```

**系统验证**：
```bash
$ cat /sys/module/amdgpu/parameters/mes
0  # ← 确认MI308X使用CPSCH
```

---

## 文档概述

本文档基于amdgpu-6.12.12-2194681.el8_preempt驱动代码和系统实测，深入分析MI308X GPU的队列管理机制，回答POC实现中的三个核心问题。

**⚠️ 重要说明**：文档中提到MES的部分仅供参考（MI308X不使用），用于理解代码架构。**MI308X实际只使用CPSCH模式。**

**分析目标**：实现Online-AI抢占Offline-AI的队列调度机制

**硬件背景**：MI308X (IP_VERSION 9.4.3, Aldebaran架构)，4 XCC，80个MQD，288个HQD

**驱动模式**：CPSCH (enable_mes=0)

---

## 问题1: MI308X调度器类型（CPSCH vs MES）⭐⭐⭐

### ✅ 核心结论（已验证）

**MI308X只使用CPSCH模式，不使用MES**

```bash
# 系统验证
$ cat /sys/module/amdgpu/parameters/mes
0  # ← MI308X上enable_mes=0，只用CPSCH

# 历史验证（参考DRIVER_47文档）
MES用于mes_v11_0/v12_0（RDNA3+/CDNA4+架构）
MI308X属于GFX 9.4.3（CDNA2/3），使用CPSCH
```

### 1.1 代码证据

#### 证据1：调度器选择逻辑（条件分支，但MI308X走CPSCH）

在`kfd_device_queue_manager.c`中，存在调度器选择代码：

```c
// 文件: kfd_device_queue_manager.c
// 行号: 1000, 1063, 1149, 1189, 1288, 1300, 1444, 1843, 1870, 1917, 1980, 1987, 1991

if (!dqm->dev->kfd->shared_resources.enable_mes) {
    // ✅ CPSCH路径 - MI308X走这里
    retval = execute_queues_cpsch(dqm, ...);
    // 或
    retval = map_queues_cpsch(dqm);
    // 或
    retval = unmap_queues_cpsch(dqm, ...);
} else {
    // ❌ MES路径 - MI308X不走这里（代码存在但不启用）
    retval = add_queue_mes(dqm, q, qpd);
    // 或
    retval = remove_queue_mes(dqm, q, qpd);
}
```

**关键发现**：
- `enable_mes`标志决定使用CPSCH还是MES
- ⚠️ **MI308X上enable_mes=0，只使用CPSCH模式**
- MES代码路径存在是为了支持更新的GPU（RDNA3+等）

#### 证据2：MES代码存在但MI308X不使用 ⚠️

```c
// 文件: kfd_device_queue_manager.c
// 行号: 221-306 (add_queue_mes函数)

static int add_queue_mes(struct device_queue_manager *dqm, struct queue *q,
                         struct qcm_process_device *qpd)
{
    struct amdgpu_device *adev = (struct amdgpu_device *)dqm->dev->adev;
    struct mes_add_queue_input queue_input;
    
    // ... 初始化queue_input ...
    queue_input.doorbell_offset = q->properties.doorbell_off;
    queue_input.mqd_addr = q->gart_mqd_addr;
    queue_input.wptr_addr = (uint64_t)q->properties.write_ptr;
    
    // 调用MES API
    r = adev->mes.funcs->add_hw_queue(&adev->mes, &queue_input);
    // ...
}
```

**⚠️ 重要说明**：
- 代码中有MES路径，但**MI308X不使用这个路径**
- MES用于更新的GPU（mes_v11_0/v12_0, RDNA3+/CDNA4+）
- 如果enable_mes=1，队列通过`mes.funcs->add_hw_queue`直接添加
- **MI308X enable_mes=0，不走这个路径**

#### 证据3：CPSCH路径的Runlist机制

```c
// 文件: kfd_device_queue_manager.c
// 行号: 2200-2221 (map_queues_cpsch函数)

static int map_queues_cpsch(struct device_queue_manager *dqm)
{
    // ...
    if (dqm->active_queue_count <= 0 || dqm->processes_count <= 0)
        return 0;
    if (dqm->active_runlist)
        return 0;
    
    retval = pm_send_runlist(&dqm->packet_mgr, &dqm->queues);
    // ...
    dqm->active_runlist = true;
    return retval;
}
```

**关键发现**：
- CPSCH模式下，通过`pm_send_runlist`发送runlist IB到HWS
- Runlist包含所有active队列的map信息

#### 证据4：MI308X的Packet Manager选择

```c
// 文件: kfd_packet_manager.c
// 行号: 295-299

if (KFD_GC_VERSION(dqm->dev) == IP_VERSION(9, 4, 2) ||
    KFD_GC_VERSION(dqm->dev) == IP_VERSION(9, 4, 3) ||
    KFD_GC_VERSION(dqm->dev) == IP_VERSION(9, 4, 4) ||
    KFD_GC_VERSION(dqm->dev) == IP_VERSION(9, 5, 0))
    pm->pmf = &kfd_aldebaran_pm_funcs;
```

**关键发现**：
- MI308X使用`kfd_aldebaran_pm_funcs`作为packet manager
- 支持Aldebaran特定的PM4 packet格式

### 1.2 结论 ⭐⭐⭐⭐⭐

**MI308X只使用CPSCH模式，不使用MES** ✅

---

#### 证据总结：

1. ✅ **系统验证**：`cat /sys/module/amdgpu/parameters/mes` 返回 `0`
2. ✅ **代码分析**：enable_mes=0时走CPSCH路径
3. ✅ **历史文档**：DRIVER_47明确MES用于mes_v12_0（更新架构）
4. ✅ **架构匹配**：MI308X是GFX 9.4.3（CDNA2/3），使用CPSCH

---

#### MI308X使用的调度器：**CPSCH模式** ✅

```
CPSCH (CP Scheduler with HWS):
  - 使用CP Firmware中的HWS（Hardware Scheduler）
  - 通过HIQ（Hardware Interface Queue）与HWS通信
  - 使用Runlist IB管理队列
  - ✅ MI308X (GFX 9.4.3)使用此模式
  - ✅ MI200系列也使用此模式
  
工作流程：
  1. KFD调用map_queues_cpsch()
  2. pm_send_runlist()创建runlist IB
  3. runlist IB发送到HIQ
  4. CP Firmware HWS解析runlist
  5. HWS将MQD加载到HQD
```

---

#### MES是什么？为什么代码里有但不用？⚠️

```
MES (Micro Engine Scheduler):
  - 新一代硬件调度器
  - 用于RDNA3+（gfx11/gfx12）和CDNA4+
  - 通过MES API直接管理队列，不需要HIQ
  - ❌ MI308X不支持MES硬件
  - ⚠️ 代码中的MES路径是为更新GPU准备的（向前兼容）
```

### 1.3 验证方法

```bash
# 方法1：检查enable_mes参数 ⭐推荐
cat /sys/module/amdgpu/parameters/mes
# MI308X输出: 0 （= 使用CPSCH）

# 方法2：检查dmesg日志
dmesg | grep -i "HWS\|enable_mes"
# 应该看到HWS相关日志，没有MES初始化

# 方法3：检查HIQ（CPSCH特有）
cat /sys/kernel/debug/kfd/hqds | grep -i "HIQ"
# 如果有HIQ输出 → 使用CPSCH
# 如果没有HIQ → 可能使用MES（但MI308X一定有HIQ）
```

---

## 问题2: Doorbell与MQD状态、Unmap后Ring-Buffer行为

### 2.1 MQD生命周期状态

#### 状态1：MQD分配（Allocated）

```c
// 文件: kfd_device_queue_manager.c
// 行号: 2050-2110 (create_queue_cpsch函数)

static int create_queue_cpsch(struct device_queue_manager *dqm, struct queue *q, ...)
{
    // 1. 分配doorbell
    retval = allocate_doorbell(qpd, q, ...);
    
    // 2. 分配MQD内存
    q->mqd_mem_obj = mqd_mgr->allocate_mqd(mqd_mgr->dev, &q->properties);
    
    // 3. 初始化MQD
    mqd_mgr->init_mqd(mqd_mgr, &q->mqd, q->mqd_mem_obj,
                      &q->gart_mqd_addr, &q->properties);
    
    // 4. 添加到队列列表
    list_add(&q->list, &qpd->queues_list);
    
    // 5. 如果is_active，执行映射
    if (q->properties.is_active) {
        if (!dqm->dev->kfd->shared_resources.enable_mes)
            retval = execute_queues_cpsch(dqm, ...);
        else
            retval = add_queue_mes(dqm, q, qpd);
    }
}
```

**MQD状态**：
- **Allocated**: MQD内存已分配，数据结构已初始化
- **Mapped**: MQD已加载到硬件（通过hqd_load或MES API）

#### 状态2：MQD加载到硬件（Mapped）

**CPSCH模式下的加载**：

```c
// 文件: kfd_mqd_manager_v9.c
// 行号: 278-288 (load_mqd函数)

static int load_mqd(struct mqd_manager *mm, void *mqd,
                    uint32_t pipe_id, uint32_t queue_id,
                    struct queue_properties *p, struct mm_struct *mms)
{
    uint32_t wptr_shift = (p->format == KFD_QUEUE_FORMAT_AQL ? 4 : 0);
    
    return mm->dev->kfd2kgd->hqd_load(mm->dev->adev, mqd, pipe_id, queue_id,
                                      (uint32_t __user *)p->write_ptr,
                                      wptr_shift, 0, mms, 0);
}
```

**关键发现**：
- `hqd_load`将MQD内容加载到硬件HQD寄存器
- 此时MQD处于**Mapped**状态，硬件可以处理该队列

**MES模式下的加载**：

```c
// 文件: kfd_device_queue_manager.c
// 行号: 221-306 (add_queue_mes函数)

static int add_queue_mes(struct device_queue_manager *dqm, struct queue *q, ...)
{
    queue_input.mqd_addr = q->gart_mqd_addr;  // MQD GPU地址
    queue_input.doorbell_offset = q->properties.doorbell_off;
    queue_input.wptr_addr = (uint64_t)q->properties.write_ptr;
    
    r = adev->mes.funcs->add_hw_queue(&adev->mes, &queue_input);
}
```

**关键发现**：
- MES模式下，MQD地址传递给MES硬件
- MES硬件直接读取MQD并管理队列

### 2.2 Doorbell敲响时的MQD状态

#### 代码证据：Doorbell配置

```c
// 文件: kfd_mqd_manager_v9.c
// 行号: 290-314 (update_mqd函数)

static void update_mqd(struct mqd_manager *mm, void *mqd,
                       struct queue_properties *q,
                       struct mqd_update_info *minfo)
{
    struct v9_mqd *m = get_mqd(mqd);
    
    // 配置ring buffer地址
    m->cp_hqd_pq_base_lo = lower_32_bits((uint64_t)q->queue_address >> 8);
    m->cp_hqd_pq_base_hi = upper_32_bits((uint64_t)q->queue_address >> 8);
    
    // 配置doorbell
    m->cp_hqd_pq_doorbell_control =
        q->doorbell_off << CP_HQD_PQ_DOORBELL_CONTROL__DOORBELL_OFFSET__SHIFT;
    
    // 配置wptr poll地址（AQL队列）
    m->cp_hqd_pq_wptr_poll_addr_lo = lower_32_bits((uint64_t)q->write_ptr);
    m->cp_hqd_pq_wptr_poll_addr_hi = upper_32_bits((uint64_t)q->write_ptr);
}
```

**关键发现**：
- MQD中配置了doorbell offset和wptr poll地址
- 当doorbell敲响时，硬件读取wptr并处理ring buffer

#### Doorbell敲响流程

```c
// 用户空间写入doorbell
// → GPU硬件检测doorbell写入
// → 硬件读取MQD中的wptr_poll_addr（对于AQL队列）
// → 或直接使用doorbell值作为wptr（对于PM4队列）
// → 硬件比较wptr和rptr，处理新任务
```

**结论**：当doorbell敲响时，MQD必须处于**Mapped**状态，即：
1. MQD已加载到硬件HQD寄存器（CPSCH模式）
2. （MI308X不适用，MES模式仅用于更新GPU）
3. MQD中的ring buffer地址、doorbell配置等已正确设置

### 2.3 Unmap后Ring-Buffer行为

#### 代码证据：Unmap流程

```c
// 文件: kfd_device_queue_manager.c
// 行号: 2353-2425 (unmap_queues_cpsch函数)

static int unmap_queues_cpsch(struct device_queue_manager *dqm,
                               enum kfd_unmap_queues_filter filter,
                               uint32_t filter_param,
                               uint32_t grace_period,
                               bool reset)
{
    // 1. 发送unmap packet
    retval = pm_send_unmap_queue(&dqm->packet_mgr, filter, filter_param, reset);
    
    // 2. 等待fence完成
    retval = amdkfd_fence_wait_timeout(dqm, KFD_FENCE_COMPLETED,
                                       queue_preemption_timeout_ms);
    
    // 3. 检查preemption是否成功
    if (mqd_mgr->check_preemption_failed(...)) {
        // 处理preemption失败
    }
    
    // 4. 释放runlist IB
    pm_release_ib(&dqm->packet_mgr);
    dqm->active_runlist = false;
}
```

**关键发现**：
- Unmap操作会等待preemption完成
- Preemption成功后，队列从runlist中移除
- **但MQD内存和ring buffer仍然存在**

#### Unmap后MQD状态

```c
// 文件: kfd_device_queue_manager.c
// 行号: 2537-2549 (destroy_queue_cpsch函数)

if (q->properties.is_active) {
    decrement_queue_count(dqm, qpd, q);
    q->properties.is_active = false;  // 标记为非active
    
    if (!dqm->dev->kfd->shared_resources.enable_mes) {
        retval = execute_queues_cpsch(dqm, ...);  // 从runlist移除
    } else {
        retval = remove_queue_mes(dqm, q, qpd);   // 从MES移除
    }
}
```

**关键发现**：
- Unmap后，`is_active = false`
- 队列从runlist/MES中移除
- **但MQD和ring buffer内存未释放**

#### Ring-Buffer行为分析

**推断**（基于代码逻辑）：

1. **Unmap后，ring buffer仍然可写**：
   - Ring buffer是用户空间分配的内存
   - Unmap只是从调度器中移除队列，不释放ring buffer
   - 用户空间仍可以写入ring buffer

2. **但硬件不会处理新任务**：
   - Unmap后，队列不在runlist中
   - 硬件调度器不会选择该队列执行
   - 即使doorbell敲响，硬件也不会处理

3. **验证方法**：
```c
// 需要测试：unmap后写入ring buffer并敲doorbell
// 预期：doorbell写入成功，但任务不执行
```

**结论**：
- **Unmap后，MQD处于Allocated但Unmapped状态**
- **Ring buffer可以继续写入，但硬件不会处理**
- **需要重新map队列才能恢复执行**

---

## 问题3: Runlist机制与抢占实现

### 3.1 Runlist与MQD/HQD的关系

#### Runlist的构建

```c
// 文件: kfd_packet_manager.c
// 行号: 136-277 (pm_create_runlist_ib函数)

static int pm_create_runlist_ib(struct packet_manager *pm,
                                 struct list_head *queues,
                                 uint64_t *rl_gpu_addr,
                                 size_t *rl_size_bytes)
{
    // 1. 遍历所有队列
    list_for_each_entry(cur, queues, list) {
        qpd = cur->qpd;
        
        // 2. 构建map_process packet
        retval = pm->pmf->map_process(pm, &rl_buffer[rl_wptr], qpd);
        
        // 3. 遍历该process的所有active队列
        list_for_each_entry(q, &qpd->queues_list, list) {
            if (!q->properties.is_active)  // ⭐ 关键：只包含active队列
                continue;
            
            // 4. 构建map_queues packet
            retval = pm->pmf->map_queues(pm, &rl_buffer[rl_wptr], q, ...);
        }
    }
}
```

**关键发现**：
- Runlist只包含`is_active = true`的队列
- Runlist包含map_process和map_queues两种packet
- Map_queues packet包含MQD地址等信息

#### Runlist与HQD的关系

```c
// 文件: kfd_packet_manager.c
// 行号: 223-242

list_for_each_entry(q, &qpd->queues_list, list) {
    if (!q->properties.is_active)
        continue;
    
    // map_queues packet包含：
    // - MQD地址 (q->gart_mqd_addr)
    // - Pipe/Queue ID
    // - 其他队列属性
    
    retval = pm->pmf->map_queues(pm, &rl_buffer[rl_wptr], q, ...);
}
```

**关键发现**：
- Runlist中的map_queues packet指向MQD地址
- CP Firmware解析runlist后，调用`hqd_load`将MQD加载到HQD寄存器
- **HQD是硬件寄存器，MQD是软件数据结构**

#### 关系图

```
Runlist IB (内存)
    ├── map_process packet (Process 1)
    │   ├── map_queues packet → MQD1地址 → HQD1寄存器
    │   ├── map_queues packet → MQD2地址 → HQD2寄存器
    │   └── ...
    ├── map_process packet (Process 2)
    │   └── ...
    └── runlist packet (指向runlist IB)
```

**关键理解**：
- **Runlist是队列映射的"快照"**，包含当前所有active队列
- **MQD是队列的元数据**，存储在系统内存中
- **HQD是硬件寄存器**，CP Firmware从MQD加载到HQD

### 3.2 抢占实现的本质

#### 代码证据：抢占流程

```c
// 文件: kfd_device_queue_manager.c
// 行号: 2442-2456 (execute_queues_cpsch函数)

static int execute_queues_cpsch(struct device_queue_manager *dqm,
                                 enum kfd_unmap_queues_filter filter,
                                 uint32_t filter_param,
                                 uint32_t grace_period)
{
    // 1. Unmap旧队列（从runlist移除）
    retval = unmap_queues_cpsch(dqm, filter, filter_param, grace_period, false);
    
    // 2. Map新队列（重新构建runlist）
    if (!retval)
        retval = map_queues_cpsch(dqm);
    
    return retval;
}
```

**关键发现**：
- 抢占 = Unmap旧队列 + Map新队列
- 通过重建runlist实现队列切换

#### 抢占的两种实现方式

**方式1：管理Runlist（CPSCH模式）**

```c
// 1. 设置队列is_active = false
q->properties.is_active = false;

// 2. 触发runlist重建
execute_queues_cpsch(dqm, KFD_UNMAP_QUEUES_FILTER_DYNAMIC_QUEUES, ...);
// → unmap_queues_cpsch: 发送unmap packet，等待preemption
// → map_queues_cpsch: 重新构建runlist（只包含is_active=true的队列）
```

**方式2：管理MQD/HQD的Active状态（MES模式）**

```c
// 1. 从MES移除队列
remove_queue_mes(dqm, q, qpd);
// → MES硬件停止调度该队列

// 2. 添加新队列到MES
add_queue_mes(dqm, new_q, new_qpd);
// → MES硬件开始调度新队列
```

### 3.3 POC实现建议

#### 方案1：基于Runlist管理（CPSCH模式）

**实现步骤**：

1. **标记Offline-AI队列为非active**：
```c
// 伪代码
for_each_offline_queue(q) {
    q->properties.is_active = false;
    decrement_queue_count(dqm, qpd, q);
}

// 触发runlist重建
execute_queues_cpsch(dqm, KFD_UNMAP_QUEUES_FILTER_DYNAMIC_QUEUES, 0, ...);
```

2. **确保Online-AI队列为active**：
```c
for_each_online_queue(q) {
    if (!q->properties.is_active) {
        q->properties.is_active = true;
        increment_queue_count(dqm, qpd, q);
    }
}

// 触发runlist重建（包含online队列）
map_queues_cpsch(dqm);
```

**优点**：
- 利用现有runlist机制
- 实现简单

**缺点**：
- 需要重建整个runlist
- 可能有延迟

#### 方案2：基于MES队列管理（MES模式）

**实现步骤**：

1. **从MES移除Offline-AI队列**：
```c
for_each_offline_queue(q) {
    remove_queue_mes(dqm, q, qpd);
    q->properties.is_active = false;
}
```

2. **添加Online-AI队列到MES**：
```c
for_each_online_queue(q) {
    if (!q->properties.is_active) {
        add_queue_mes(dqm, q, qpd);
        q->properties.is_active = true;
    }
}
```

**优点**：
- 粒度更细，可以单独管理每个队列
- 可能延迟更低

**缺点**：
- ❌ MI308X不支持（需要MES硬件）

### 3.4 结论

**抢占的本质**：

1. **CPSCH模式**：**管理Runlist**
   - Runlist是队列调度的"快照"
   - 通过重建runlist实现队列切换
   - `is_active`标志控制队列是否在runlist中

2. ❌ **MES模式**：MI308X不适用（仅更新GPU）
   - 直接通过MES API添加/移除队列
   - MES硬件管理调度
   - `is_active`标志与MES状态同步

**POC建议**：

- **MI308X只能使用CPSCH模式**：
  - 更细粒度的控制
  - 更低的延迟
  - 更适合动态抢占场景

- **如果只能使用CPSCH模式**：
  - 通过管理`is_active`标志
  - 调用`execute_queues_cpsch`重建runlist
  - 注意preemption的grace period

---

## POC实施建议（基于CPSCH模式）⭐⭐⭐⭐⭐

### 🎯 关键决策：MI308X只用CPSCH，POC基于CPSCH设计

```
✅ 使用CPSCH机制（HWS + Runlist）
✅ 操作is_active标志
✅ 通过execute_queues_cpsch触发重调度
❌ 不需要考虑MES相关功能
```

## POC实施路线图

### 4.1 调度器检测

```c
// 检测当前调度器模式
bool is_mes_mode = dqm->dev->kfd->shared_resources.enable_mes;

if (is_mes_mode) {
    // MES模式实现
    implement_preemption_mes(dqm, offline_queues, online_queues);
} else {
    // CPSCH模式实现
    implement_preemption_cpsch(dqm, offline_queues, online_queues);
}
```

### 4.2 抢占实现框架

```c
// 伪代码：Online-AI抢占Offline-AI
int preempt_offline_for_online(struct device_queue_manager *dqm,
                                struct list_head *offline_queues,
                                struct list_head *online_queues)
{
    int retval;
    
    // 1. 暂停Offline-AI队列
    list_for_each_entry(q, offline_queues, list) {
        if (q->properties.is_active) {
            q->properties.is_active = false;
            decrement_queue_count(dqm, qpd, q);
        }
    }
    
    // 2. 确保Online-AI队列active
    list_for_each_entry(q, online_queues, list) {
        if (!q->properties.is_active) {
            q->properties.is_active = true;
            increment_queue_count(dqm, qpd, q);
        }
    }
    
    // 3. 执行队列切换
    if (dqm->dev->kfd->shared_resources.enable_mes) {
        // MES模式
        list_for_each_entry(q, offline_queues, list) {
            if (q->properties.is_active == false)
                remove_queue_mes(dqm, q, qpd);
        }
        list_for_each_entry(q, online_queues, list) {
            if (q->properties.is_active == true)
                add_queue_mes(dqm, q, qpd);
        }
    } else {
        // CPSCH模式
        retval = execute_queues_cpsch(dqm,
                                      KFD_UNMAP_QUEUES_FILTER_DYNAMIC_QUEUES,
                                      0, USE_DEFAULT_GRACE_PERIOD);
    }
    
    return retval;
}
```

### 4.3 关键注意事项

1. **Preemption Grace Period**：
   - CPSCH模式下，unmap操作有grace period
   - 需要等待preemption完成才能继续

2. **队列状态同步**：
   - `is_active`标志必须与硬件状态同步
   - 使用`increment_queue_count`/`decrement_queue_count`维护计数

3. **多XCC支持**：
   - MI308X有4个XCC
   - 需要为每个XCC处理MQD

4. **Ring Buffer处理**：
   - Unmap后，ring buffer仍可写
   - 但硬件不会处理，需要重新map

---

## 验证方法和测试脚本

### 5.1 验证调度器模式

```bash
#!/bin/bash
# check_scheduler_mode.sh

echo "=== Checking MI308X Scheduler Mode ==="

# 方法1：检查内核日志
echo "1. Kernel log (enable_mes):"
dmesg | grep -i "enable_mes\|MES\|CPSCH" | tail -20

# 方法2：检查设备属性（如果支持）
if [ -f /sys/class/kfd/kfd/topology/nodes/0/properties ]; then
    echo "2. Device properties:"
    cat /sys/class/kfd/kfd/topology/nodes/0/properties | grep -i mes
fi

# 方法3：检查运行队列
echo "3. Active queues (if debugfs available):"
# 需要添加debugfs接口或使用rocprof
```

### 5.2 验证MQD状态

```c
// 测试代码：验证MQD状态
// test_mqd_state.c

#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>

// 伪代码：需要实际的KFD ioctl定义
void test_mqd_state(int queue_fd) {
    // 1. 创建队列（MQD allocated）
    // 2. 检查MQD是否在内存中
    
    // 3. Map队列（MQD mapped）
    // 4. 检查MQD是否加载到硬件
    
    // 5. 写入ring buffer并敲doorbell
    // 6. 验证任务是否执行
    
    // 7. Unmap队列
    // 8. 再次写入ring buffer并敲doorbell
    // 9. 验证任务不执行（但doorbell写入成功）
}
```

### 5.3 验证抢占机制

```c
// 测试代码：验证抢占
// test_preemption.c

void test_preemption() {
    // 1. 创建Offline-AI队列（80个）
    // 2. 启动Offline-AI任务
    // 3. 创建Online-AI队列（20个）
    // 4. 触发抢占（设置offline队列is_active=false）
    // 5. 验证Online-AI任务开始执行
    // 6. 验证Offline-AI任务暂停
    // 7. 恢复Offline-AI（设置is_active=true）
    // 8. 验证Offline-AI任务恢复
}
```

### 5.4 性能测试

```bash
#!/bin/bash
# benchmark_preemption.sh

echo "=== Preemption Latency Benchmark ==="

# 1. 测量unmap延迟
time_start=$(date +%s%N)
# 执行unmap操作
time_end=$(date +%s%N)
unmap_latency=$((($time_end - $time_start) / 1000000))
echo "Unmap latency: ${unmap_latency}ms"

# 2. 测量map延迟
time_start=$(date +%s%N)
# 执行map操作
time_end=$(date +%s%N)
map_latency=$((($time_end - $time_start) / 1000000))
echo "Map latency: ${map_latency}ms"

# 3. 总抢占延迟
total_latency=$(($unmap_latency + $map_latency))
echo "Total preemption latency: ${total_latency}ms"
```

---

## 总结

### 核心结论 ⭐⭐⭐⭐⭐

1. **MI308X只使用CPSCH调度器**，enable_mes=0（已验证）

2. **Doorbell敲响时，MQD必须处于Mapped状态**：
   - CPSCH模式：MQD已通过`hqd_load`加载到HQD寄存器
   - ❌ MES模式：MI308X不适用

3. **Unmap后，ring buffer仍可写，但硬件不会处理**：
   - MQD处于Allocated但Unmapped状态
   - 需要重新map才能恢复执行

4. **抢占的本质**：
   - **CPSCH模式**：管理Runlist（重建包含active队列的runlist）
   - ❌ **MES模式**：MI308X不使用

### POC实施路径

1. **检测调度器模式**（`enable_mes`标志）
2. **实现抢占逻辑**：
   - 设置`is_active`标志
   - 调用相应的map/unmap函数
   - 等待preemption完成
3. **验证和优化**：
   - 测量抢占延迟
   - 优化grace period
   - 处理多XCC场景

### 代码引用总结

| 功能 | 文件 | 行号 | 说明 |
|------|------|------|------|
| 调度器选择 | kfd_device_queue_manager.c | 1000, 1063, etc. | enable_mes标志检查 |
| MES添加队列 | kfd_device_queue_manager.c | 221-306 | add_queue_mes |
| CPSCH映射 | kfd_device_queue_manager.c | 2200-2221 | map_queues_cpsch |
| Runlist构建 | kfd_packet_manager.c | 136-277 | pm_create_runlist_ib |
| MQD加载 | kfd_mqd_manager_v9.c | 278-288 | load_mqd |
| Doorbell配置 | kfd_mqd_manager_v9.c | 290-314 | update_mqd |
| 抢占执行 | kfd_device_queue_manager.c | 2442-2456 | execute_queues_cpsch |

---

## 附录：关键数据结构

### Queue Properties

```c
struct queue_properties {
    // ...
    bool is_active;           // ⭐ 关键：控制队列是否在runlist中
    uint32_t doorbell_off;    // Doorbell偏移
    uint64_t queue_address;   // Ring buffer地址
    uint64_t write_ptr;       // WPTR地址
    // ...
};
```

### Device Queue Manager

```c
struct device_queue_manager {
    // ...
    bool active_runlist;       // ⭐ CPSCH模式：runlist是否active
    unsigned int active_queue_count;  // Active队列计数
    struct packet_manager packet_mgr;  // Packet manager
    // ...
};
```

---

**文档版本**: 1.0  
**最后更新**: 2026-02-04  
**基于代码**: amdgpu-6.12.12-2194681.el8_preempt
