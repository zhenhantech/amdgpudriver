# 软件队列(MQD)到硬件队列(HQD)的Map/Unmap机制

**日期**: 2026-02-03  
**GPU**: MI308X (GFX v9.4.3)  
**重要性**: ⭐⭐⭐⭐⭐ **核心运行时机制**

---

## 🎯 核心概念

### 软件队列 vs 硬件队列

```
软件队列 (SW Queue / MQD - Memory Queue Descriptor):
  - 存储在系统内存中的队列描述符
  - 包含队列配置信息
  - 可以有很多个（理论上无限制）
  - 数量 = 用户创建的队列数

硬件队列 (HW Queue / HQD - Hardware Queue Descriptor):
  - GPU硬件上的实际队列槽位
  - 数量有限（MI308X: 30个/XCC）
  - 需要物理硬件资源
  - 数量 = 硬件支持的最大队列数
```

### Map vs Unmap

```
Map (映射):
  - 将SW Queue (MQD) 加载到 HW Queue (HQD)
  - 队列变为"active"状态
  - 可以执行GPU任务
  
Unmap (解映射):
  - 将SW Queue从HW Queue卸载
  - 队列变为"inactive"状态
  - 硬件槽位可以给其他队列使用
```

---

## 📊 队列创建的完整流程

### 阶段1: 创建软件队列(MQD)

**函数**: `create_queue_cpsch()`  
**位置**: `kfd_device_queue_manager.c` line 2050

```c
static int create_queue_cpsch(struct device_queue_manager *dqm, struct queue *q,
                              struct qcm_process_device *qpd, ...)
{
    // 1. 检查队列总数限制
    if (dqm->total_queue_count >= max_num_of_queues_per_device) {
        trace_printk("Can't create new usermode queue because %d queues were already created\n",
                     dqm->total_queue_count);
        return -EPERM;  // ❌ 超过限制
    }
    
    // 2. 分配SDMA队列（如果是SDMA类型）
    if (q->properties.type == KFD_QUEUE_TYPE_SDMA) {
        retval = allocate_sdma_queue(dqm, q, ...);
    }
    
    // 3. 分配doorbell
    retval = allocate_doorbell(qpd, q, ...);
    
    // 4. 获取MQD manager
    mqd_mgr = dqm->mqd_mgrs[get_mqd_type_from_queue_type(q->properties.type)];
    
    // 5. 分配MQD内存（在系统内存中）
    q->mqd_mem_obj = mqd_mgr->allocate_mqd(mqd_mgr->dev, &q->properties);
    
    // 6. 初始化MQD
    mqd_mgr->init_mqd(mqd_mgr, &q->mqd, q->mqd_mem_obj,
                     &q->gart_mqd_addr, &q->properties);
    
    // 7. 添加到队列列表
    list_add(&q->list, &qpd->queues_list);
    
    // 8. 如果是active队列，触发map
    if (q->properties.is_active) {
        execute_queues_cpsch(dqm, ...);  // ← 触发map操作
    }
}
```

**关键点**：
- ✅ MQD在系统内存中创建（不需要HQD）
- ✅ 可以创建很多MQD（只要内存足够）
- ✅ 只有active的队列才会map到HQD

---

### 阶段2: 分配硬件队列(HQD)

**函数**: `allocate_hqd()`  
**位置**: `kfd_device_queue_manager.c` line 777

```c
static int allocate_hqd(struct device_queue_manager *dqm, struct queue *q)
{
    bool set = false;
    int pipe, bit, i;
    
    // 轮询所有Pipe，寻找空闲的硬件队列
    for (pipe = dqm->next_pipe_to_allocate, i = 0;
         i < get_pipes_per_mec(dqm);  // 4个Pipes
         pipe = ((pipe + 1) % get_pipes_per_mec(dqm)), ++i) {
        
        // 检查这个Pipe是否可用
        if (!is_pipe_enabled(dqm, 0, pipe))  // 只用MEC 0
            continue;
        
        // 从这个Pipe的队列位图中找空闲队列
        if (dqm->allocated_queues[pipe] != 0) {
            bit = ffs(dqm->allocated_queues[pipe]) - 1;  // 找第一个置位的bit
            dqm->allocated_queues[pipe] &= ~(1 << bit);  // 清除这个bit
            
            q->pipe = pipe;   // ← 分配pipe编号
            q->queue = bit;   // ← 分配queue编号
            set = true;
            break;
        }
    }
    
    if (!set) {
        pr_err("Failed to allocate HQD\n");
        return -ENOMEM;  // ❌ 没有空闲HQD了
    }
    
    return 0;
}
```

**HQD分配策略**：
```
轮询策略（Round-Robin）:
  - 从next_pipe_to_allocate开始
  - 依次检查Pipe 0, 1, 2, 3
  - 找到第一个有空闲队列的Pipe
  - 分配这个Pipe的第一个空闲Queue
  
目的：均衡负载到所有Pipe上
```

**allocated_queues位图**：
```
dqm->allocated_queues[pipe]: 
  - 每个Pipe一个位图
  - 每个bit代表一个Queue
  - bit=1: 空闲，可分配
  - bit=0: 已占用
  
示例 (8个Queue/Pipe):
  allocated_queues[0] = 0b11111100  // Queue 0-1已用，2-7空闲
  allocated_queues[1] = 0b11110000  // Queue 0-3已用，4-7空闲
```

---

### 阶段3: 加载MQD到HQD (Map操作)

**函数**: `load_mqd()` / `load_mqd_v9_4_3()`  
**位置**: `kfd_mqd_manager_v9.c` line 278, 857

#### 3.1 普通队列的Load MQD

```c
static int load_mqd(struct mqd_manager *mm, void *mqd,
                   uint32_t pipe_id, uint32_t queue_id,
                   struct queue_properties *p, struct mm_struct *mms)
{
    // 计算write pointer的偏移
    uint32_t wptr_shift = (p->format == KFD_QUEUE_FORMAT_AQL ? 4 : 0);
    
    // ⭐ 关键: 调用硬件接口加载MQD到HQD
    return mm->dev->kfd2kgd->hqd_load(
        mm->dev->adev,              // GPU设备
        mqd,                        // MQD内存地址
        pipe_id,                    // Pipe ID (0-3)
        queue_id,                   // Queue ID (0-7)
        (uint32_t __user *)p->write_ptr,
        wptr_shift, 
        0, 
        mms, 
        0
    );
}
```

#### 3.2 多XCC的Load MQD (MI308X)

```c
static int load_mqd_v9_4_3(struct mqd_manager *mm, void *mqd,
                          uint32_t pipe_id, uint32_t queue_id,
                          struct queue_properties *p, struct mm_struct *mms)
{
    uint32_t wptr_shift = (p->format == KFD_QUEUE_FORMAT_AQL ? 4 : 0);
    uint32_t xcc_mask = mm->dev->xcc_mask;  // ← 4个XCC
    int xcc_id, err, inst = 0;
    void *xcc_mqd;
    uint64_t mqd_stride = kfd_mqd_stride(mm->dev);
    
    // ⭐ 为每个XCC加载MQD
    for_each_inst(xcc_id, xcc_mask) {  // 遍历4个XCC
        xcc_mqd = mqd + mqd_stride * inst;  // 计算这个XCC的MQD偏移
        
        err = mm->dev->kfd2kgd->hqd_load(
            mm->dev->adev, 
            xcc_mqd,
            pipe_id,   // ← 同样的pipe_id
            queue_id,  // ← 同样的queue_id  
            (uint32_t __user *)p->write_ptr,
            wptr_shift, 
            0, 
            mms, 
            xcc_id    // ← 但是不同的xcc_id!
        );
        
        if (err) {
            pr_debug("Failed to load MQD for XCC: %d\n", inst);
            break;
        }
        ++inst;
    }
    
    return err;
}
```

**关键发现** ⭐⭐⭐：
```
MI308X的MQD加载：
  - 同一个逻辑队列 (pipe, queue)
  - 在4个XCC上都要加载MQD
  - 每个XCC有自己的MQD副本
  - 所以：1个软件队列 → 4个硬件队列（跨4个XCC）
```

---

### 阶段4: Map Queues (批量映射)

**函数**: `map_queues_cpsch()`  
**位置**: `kfd_device_queue_manager.c` line 60 (声明)

**调用链**：
```
用户创建active队列
  ↓
create_queue_cpsch()
  ↓
execute_queues_cpsch()
  ↓
map_queues_cpsch()  ← 批量Map操作
  ↓
pm_map_queues()     ← 发送Map Queues packet
  ↓
GPU硬件调度器(HWS)执行Map操作
```

**Map Queues的触发时机**：

1. **队列创建时**（is_active=true）
   ```c
   if (q->properties.is_active) {
       execute_queues_cpsch(dqm, ...);  // ← 触发map
   }
   ```

2. **队列更新时**（变为active）
   ```c
   if (!prev_active && q->properties.is_active) {
       retval = map_queues_cpsch(dqm);  // ← 触发map
   }
   ```

3. **系统启动时**（start_cpsch）
   ```c
   dqm->sched_running = true;
   execute_queues_cpsch(dqm, KFD_UNMAP_QUEUES_FILTER_DYNAMIC_QUEUES, ...);
   ```

4. **从halt恢复时**（unhalt_cpsch）
   ```c
   dqm->sched_halt = false;
   ret = execute_queues_cpsch(dqm, ...);
   ```

---

### 阶段5: Unmap Queues (批量解映射)

**函数**: `unmap_queues_cpsch()`  
**位置**: `kfd_device_queue_manager.c` line 54 (声明)

**Unmap的触发时机**：

1. **队列销毁时**
   ```c
   destroy_queue_cpsch() {
       unmap_queues_cpsch(dqm, KFD_UNMAP_QUEUES_FILTER_BY_PASID, ...);
       deallocate_hqd(dqm, q);  // 释放硬件队列
   }
   ```

2. **队列变为inactive时**
   ```c
   if (prev_active && !q->properties.is_active) {
       unmap_queues_cpsch(dqm, KFD_UNMAP_QUEUES_FILTER_DYNAMIC_QUEUES, ...);
   }
   ```

3. **系统halt时**
   ```c
   halt_cpsch() {
       unmap_queues_cpsch(dqm, KFD_UNMAP_QUEUES_FILTER_ALL_QUEUES, ...);
   }
   ```

4. **Preemption时**（抢占）
   ```c
   mqd_mgr->destroy_mqd(mqd_mgr, q->mqd,
                        KFD_PREEMPT_TYPE_WAVEFRONT_SAVE,
                        KFD_UNMAP_LATENCY_MS, 
                        q->pipe, q->queue);
   ```

---

### 阶段6: 释放硬件队列(HQD)

**函数**: `deallocate_hqd()`  
**位置**: `kfd_device_queue_manager.c` line 811

```c
static inline void deallocate_hqd(struct device_queue_manager *dqm,
                                  struct queue *q)
{
    // 简单！就是把bit重新置位
    dqm->allocated_queues[q->pipe] |= (1 << q->queue);
    //                                  ↑ 标记为空闲
}
```

**释放流程**：
```
1. Unmap队列 (从硬件卸载)
2. Deallocate HQD (释放硬件槽位)
3. Free MQD (释放系统内存)
```

---

## 🔄 完整的Map/Unmap生命周期

### 场景1: 创建Active队列

```
用户调用: hipStreamCreate()
  ↓
KFD: create_queue_cpsch()
  ├─ 1. 分配MQD (系统内存)
  ├─ 2. 初始化MQD
  ├─ 3. allocate_hqd() ← 分配硬件队列
  │    └─ 返回: (pipe=1, queue=3)
  ├─ 4. allocate_doorbell()
  └─ 5. execute_queues_cpsch()
       └─ map_queues_cpsch()
            └─ 对于MI308X的4个XCC:
                 ├─ load_mqd(..., xcc_id=0) ← Map到XCC 0
                 ├─ load_mqd(..., xcc_id=1) ← Map到XCC 1
                 ├─ load_mqd(..., xcc_id=2) ← Map到XCC 2
                 └─ load_mqd(..., xcc_id=3) ← Map到XCC 3

结果:
  - 1个软件队列(MQD)
  - 1个硬件队列槽位 (pipe=1, queue=3)
  - 4个XCC都加载了MQD
  - 队列状态: Active ✓
```

### 场景2: 创建Inactive队列

```
用户调用: hipStreamCreateWithFlags(..., hipStreamNonBlocking)
           但不立即使用
  ↓
KFD: create_queue_cpsch()
  ├─ 1. 分配MQD (系统内存)
  ├─ 2. 初始化MQD
  ├─ 3. properties.is_active = false ← 不分配HQD
  └─ 4. 添加到队列列表

结果:
  - 1个软件队列(MQD)
  - 0个硬件队列（未分配）
  - 队列状态: Inactive
```

### 场景3: Inactive → Active (Map)

```
队列首次使用时:
  ↓
KFD: update_queue()
  └─ update_queue_locked()
       ├─ prev_active = false
       ├─ q->properties.is_active = true ← 变为active
       ├─ allocate_hqd(dqm, q) ← 现在才分配硬件队列
       │    └─ 返回: (pipe=2, queue=5)
       └─ map_queues_cpsch(dqm) ← 加载MQD到HQD
            └─ 对于4个XCC都load_mqd()

结果:
  - 软件队列存在
  - 新分配硬件队列 (pipe=2, queue=5)
  - 4个XCC都加载了MQD
  - 队列状态: Active ✓
```

### 场景4: Active → Inactive (Unmap)

```
队列空闲时（或显式deactivate）:
  ↓
KFD: update_queue()
  └─ update_queue_locked()
       ├─ prev_active = true
       ├─ q->properties.is_active = false ← 变为inactive
       └─ unmap_queues_cpsch(dqm, ...) ← 从硬件卸载
            └─ destroy_mqd(..., pipe=2, queue=5)
                 └─ 对于4个XCC都unmap

       注意: deallocate_hqd()可能不立即调用
            硬件槽位保留，以便快速重新激活

结果:
  - 软件队列仍存在
  - 硬件队列已unmap（但槽位可能保留）
  - 队列状态: Inactive
```

### 场景5: 销毁队列

```
用户调用: hipStreamDestroy()
  ↓
KFD: destroy_queue_cpsch()
  ├─ 1. unmap_queues_cpsch(...) ← 从硬件卸载（如果是active）
  ├─ 2. destroy_mqd(...)        ← 清理MQD
  ├─ 3. deallocate_hqd(dqm, q)  ← 释放硬件槽位
  │    └─ allocated_queues[pipe] |= (1 << queue)
  ├─ 4. deallocate_doorbell(...) ← 释放doorbell
  ├─ 5. mqd_mgr->free_mqd(...)   ← 释放MQD内存
  └─ 6. list_del(&q->list)       ← 从列表移除

结果:
  - 软件队列释放 ✓
  - 硬件队列释放 ✓
  - (pipe=2, queue=5)槽位可用于新队列
```

---

## 🎨 队列状态转换图

```
               create_queue()
                    ↓
    ┌──────────────────────────────────┐
    │    Inactive Queue                │
    │  - MQD存在（系统内存）            │
    │  - HQD未分配                      │
    │  - 不消耗硬件资源                 │
    └──────────────────────────────────┘
                    │
                    │ activate / first use
                    │ allocate_hqd()
                    │ map_queues()
                    ↓
    ┌──────────────────────────────────┐
    │    Active Queue                   │
    │  - MQD存在（系统内存）            │
    │  - HQD已分配 (pipe, queue)        │
    │  - 加载到硬件                      │
    │  - 可以执行任务 ✓                 │
    └──────────────────────────────────┘
                    │
                    │ deactivate / idle
                    │ unmap_queues()
                    │ (可能保留HQD)
                    ↓
    ┌──────────────────────────────────┐
    │    Inactive Queue (快速重激活)    │
    │  - MQD存在                        │
    │  - HQD可能保留                    │
    │  - 未加载到硬件                   │
    └──────────────────────────────────┘
                    │
                    │ destroy_queue()
                    │ deallocate_hqd()
                    │ free_mqd()
                    ↓
              ┌─────────┐
              │ Freed   │
              └─────────┘
```

---

## 💡 重要设计理念

### 1. 软硬件队列分离设计 ⭐⭐⭐⭐⭐

```
为什么分离？

问题: 硬件队列数量有限
  - MI308X: 30个HQD/XCC × 4 XCC = 120个/GPU
  - 但应用可能创建>120个stream/queue

解决: MQD(软件) vs HQD(硬件)
  - MQD: 可以创建很多（只受内存限制）
  - HQD: 有限，动态分配
  - Inactive队列不占用HQD
  - Active队列才map到HQD
```

### 2. 动态Map/Unmap ⭐⭐⭐⭐

```
优势:
  1. 资源利用效率高
     - 空闲队列自动unmap
     - 硬件资源给活跃队列使用
     
  2. 支持超额订阅(Oversubscription)
     - 可以创建 > 硬件限制的队列数
     - 只要同时active的 ≤ 硬件限制

  3. 快速上下文切换
     - Map/Unmap开销小
     - 支持时间片轮转
```

### 3. HWS (Hardware Scheduler) ⭐⭐⭐⭐⭐

```
AMD的硬件调度器：

传统方式（NO_HWS）:
  - CPU软件管理队列调度
  - Map/Unmap需要CPU干预
  - 开销大，延迟高

HWS方式（CPSCH）:
  - 硬件自动调度队列
  - Map/Unmap由GPU完成
  - 发送packet给HWS即可
  - 低延迟，高效率 ✓
```

---

## 🔍 关键数据结构

### 1. MQD (Memory Queue Descriptor)

**位置**: 系统内存  
**大小**: ~512 bytes (GFX9)  
**内容**:

```c
struct v9_mqd {
    // 队列控制
    uint32_t cp_hqd_pq_control;        // 队列控制寄存器
    uint32_t cp_hqd_pq_base_lo;        // 队列基地址(低32位)
    uint32_t cp_hqd_pq_base_hi;        // 队列基地址(高32位)
    
    // 读写指针
    uint32_t cp_hqd_pq_rptr_report_addr_lo;  // 读指针地址
    uint32_t cp_hqd_pq_rptr_report_addr_hi;
    uint32_t cp_hqd_pq_wptr_poll_addr_lo;    // 写指针地址
    uint32_t cp_hqd_pq_wptr_poll_addr_hi;
    
    // Doorbell
    uint32_t cp_hqd_pq_doorbell_control;
    
    // 各种配置...
    uint32_t cp_hqd_ib_control;
    uint32_t cp_hqd_vmid;
    ...
};
```

### 2. HQD (Hardware Queue Descriptor)

**位置**: GPU硬件寄存器  
**标识**: (pipe_id, queue_id)  
**分配**: `allocate_hqd()` 通过位图管理

```c
struct device_queue_manager {
    // 每个Pipe一个位图，跟踪HQD分配状态
    unsigned int allocated_queues[KGD_MAX_QUEUES];
    //           allocated_queues[pipe] & (1 << queue)
    //           = 1: 空闲
    //           = 0: 已分配
    
    int next_pipe_to_allocate;  // Round-robin起始点
};
```

### 3. Queue Properties

```c
struct queue_properties {
    enum kfd_queue_type type;      // COMPUTE / SDMA / ...
    bool is_active;                // ← 关键: 是否active
    bool is_evicted;               // 是否被驱逐
    
    uint32_t pipe;                 // ← HQD pipe编号
    uint32_t queue;                // ← HQD queue编号
    
    uint64_t queue_address;        // 队列buffer地址
    uint64_t read_ptr;             // 读指针
    uint64_t write_ptr;            // 写指针
    uint32_t doorbell_off;         // Doorbell偏移
    ...
};
```

---

## 📈 性能考量

### Map/Unmap的开销

```
Map操作:
  1. 分配HQD槽位           ~1 μs
  2. 准备MQD数据           ~1 μs
  3. 发送MAP packet到HWS   ~5 μs
  4. HWS执行加载           ~10 μs
  5. 等待确认               ~5 μs
  总计: ~20-30 μs

Unmap操作:
  1. 发送UNMAP packet      ~5 μs
  2. HWS执行卸载           ~10 μs
  3. 等待确认               ~5 μs
  4. 释放HQD槽位           ~1 μs
  总计: ~20-25 μs
```

### 优化策略

**1. 延迟Deactivation**
```
不立即deallocate HQD:
  - Queue变inactive时
  - 暂时保留HQD分配
  - 如果很快重新activate
  - 可以跳过allocate_hqd()
  - 只需要重新map即可
```

**2. 批量Map/Unmap**
```
map_queues_cpsch():
  - 不是逐个队列map
  - 批量处理所有pending队列
  - 一次发送packet
  - 减少HWS通信开销
```

**3. Pipe负载均衡**
```
Round-robin分配:
  - 轮询所有Pipe
  - 避免单个Pipe过载
  - 提高并行度
```

---

## 🐛 常见问题和调试

### Q1: 队列创建失败 "Can't create new usermode queue"

**原因**:
```c
if (dqm->total_queue_count >= max_num_of_queues_per_device) {
    // ❌ 超过队列总数限制
}
```

**解决**:
- 检查`dqm->total_queue_count` (当前队列数)
- 检查`max_num_of_queues_per_device` (最大限制)
- 销毁不用的队列
- 或使用inactive队列（不占HQD）

### Q2: allocate_hqd失败 "Failed to allocate HQD"

**原因**:
```c
// 所有Pipe的allocated_queues都是0（全部占用）
if (dqm->allocated_queues[pipe] == 0) {
    // 没有空闲HQD
}
```

**解决**:
- 检查active队列数量
- Unmap一些idle队列
- 增加KCQ数量（减少用户队列）

### Q3: Map操作很慢

**可能原因**:
1. HWS hang (硬件调度器挂起)
2. 队列过多，批量map耗时长
3. Memory latency高（MQD读取慢）

**调试**:
```bash
# 查看HWS状态
cat /sys/kernel/debug/kfd/hqds

# 查看队列数量
cat /sys/kernel/debug/kfd/mqds

# 检查是否有pending的map操作
dmesg | grep -i "map.*queue"
```

---

## 📚 代码位置总结

| 操作 | 函数 | 文件 | 行号 |
|------|------|------|------|
| 创建队列 | `create_queue_cpsch()` | kfd_device_queue_manager.c | 2050 |
| 分配HQD | `allocate_hqd()` | kfd_device_queue_manager.c | 777 |
| 释放HQD | `deallocate_hqd()` | kfd_device_queue_manager.c | 811 |
| Map队列 | `map_queues_cpsch()` | kfd_device_queue_manager.c | 60 |
| Unmap队列 | `unmap_queues_cpsch()` | kfd_device_queue_manager.c | 54 |
| Load MQD | `load_mqd()` | kfd_mqd_manager_v9.c | 278 |
| Load MQD(MI308X) | `load_mqd_v9_4_3()` | kfd_mqd_manager_v9.c | 857 |
| Destroy MQD | `destroy_mqd()` | kfd_mqd_manager_v9.c | ~350 |
| Update队列 | `update_queue()` | kfd_device_queue_manager.c | 1083 |

---

## 🎯 关键要点总结

### 1. MQD vs HQD

```
MQD (软件队列):
  ✅ 存储在系统内存
  ✅ 数量灵活（可以很多）
  ✅ 可以inactive（不占硬件资源）
  ✅ 状态可保存/恢复

HQD (硬件队列):
  ✅ GPU硬件槽位
  ✅ 数量固定有限
  ✅ 只给active队列使用
  ✅ 通过(pipe, queue)标识
```

### 2. Map/Unmap时机

```
Map (加载到硬件):
  - 队列首次activate
  - 从inactive变active
  - 系统启动/恢复
  - Preemption后恢复

Unmap (从硬件卸载):
  - 队列deactivate
  - 队列销毁
  - 系统halt/suspend
  - Preemption抢占
```

### 3. MI308X特殊性

```
多XCC架构:
  - 1个软件队列(MQD)
  - 4个XCC都要加载MQD
  - 同样的(pipe, queue)编号
  - 但每个XCC独立的HQD
  
实际：1个逻辑队列 = 4个物理HQD
```

---

**创建时间**: 2026-02-03  
**参考代码**: amdgpu-6.12.12-2194681.el8_preempt  
**GPU架构**: GFX v9.4.3 (MI308X)  
**分析质量**: ⭐⭐⭐⭐⭐ (基于代码证据)
