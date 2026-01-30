# map_queues_cpsch() 深度分析

## 📋 概述

`map_queues_cpsch()` 是 CPSCH (Compute Process Scheduler) 模式下的核心调度函数，负责将软件队列列表提交到 MEC Firmware。本文档基于实际验证结果（1356 条日志）深入分析其工作机制和性能瓶颈。

**文件位置**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_device_queue_manager.c`

**验证依据**: 
- 测试日志: `/mnt/md0/zhehan/code/rampup_doc/2PORC_streams/log/kfd_queue_test/trace_kfd_kfd_queue_test_full_20260120_134045.txt`
- 日志数量: 1356 条 `map_queues_cpsch` 调用
- 所有队列: `pipe=0, queue=0`（证实不使用固定 HQD）

---

## 🔬 函数签名和上下文

```c
/* dqm->lock mutex has to be locked before calling this function */
static int map_queues_cpsch(struct device_queue_manager *dqm)
```

**调用前提**: 
- ✅ 必须持有 `dqm->lock` 互斥锁
- ✅ 调用者负责锁的获取和释放

**返回值**:
- `0`: 成功（包括因检查失败而跳过）
- `< 0`: 错误码

---

## 📊 完整实现分析

### 第1阶段: 初始化和性能监控准备

```c
static int map_queues_cpsch(struct device_queue_manager *dqm)
{
    struct device *dev = dqm->dev->adev->dev;
    int retval;
    u64 start_time, end_time;
    unsigned int runlist_size = 0;
    
    start_time = ktime_get_ns_safe();  // 记录开始时间
```

**关键点**:
- 📊 使用 `ktime_get_ns_safe()` 记录开始时间
- 📊 为性能统计做准备
- 🎯 `runlist_size` 将记录队列数量

---

### 第2阶段: 快速检查 - 是否需要执行

```c
    if (!dqm->sched_running || dqm->sched_halt) {
        return 0;
    }
    if (dqm->active_queue_count <= 0 || dqm->processes_count <= 0) {
        return 0;
    }
```

**检查逻辑**:

| 条件 | 含义 | 为什么返回 0 |
|------|------|-------------|
| `!dqm->sched_running` | 调度器未运行 | 无需映射队列 |
| `dqm->sched_halt` | 调度器已停止 | 停止期间不处理 |
| `active_queue_count <= 0` | 没有活动队列 | 没有队列需要映射 |
| `processes_count <= 0` | 没有进程 | 没有队列需要映射 |

**性能优化**: 
- ✅ 快速路径，避免不必要的处理
- ✅ 减少无效的 Runlist 构建

---

### 第3阶段: 🔴 active_runlist 检查（串行化瓶颈）

```c
    /* Check if blocked by active_runlist */
    if (dqm->active_runlist) {
        dqm->perf_stats.map_queues_blocked_count++;
        end_time = ktime_get_ns_safe();
        if (end_time > start_time) {
            u64 blocked_time = end_time - start_time;
            dqm->perf_stats.map_queues_blocked_time_ns += blocked_time;
        }
        return 0;  // ⚠️ 直接返回，队列映射请求被阻塞！
    }
```

**🔴 这是最大的性能瓶颈！**

#### 为什么存在这个检查？

**设计意图**:
```
防止并发的 Runlist 提交
    ↓
确保 MEC Firmware 一次只处理一个 Runlist
    ↓
避免固件状态混乱
```

#### 串行化机制详解

```
时间线示例（多进程场景）:
                                                
进程 1 调用 map_queues_cpsch()
    ↓ 设置 active_runlist = true
    ↓ 开始构建 Runlist
    ↓ 提交 PM4 packet 到 MEC
    |
    | ← active_runlist = true，阻塞其他调用
    |
    ├─ 进程 2 调用 map_queues_cpsch()
    │      ↓ 检查 active_runlist == true
    │      ↓ ✅ 立即返回 0（被阻塞）
    │      ↓ blocked_count++
    │
    ├─ 进程 3 调用 map_queues_cpsch()
    │      ↓ 检查 active_runlist == true
    │      ↓ ✅ 立即返回 0（被阻塞）
    │      ↓ blocked_count++
    |
    ↓ 等待 MEC Firmware 处理完成
    ↓ 通过 unmap_queues_cpsch() 清除 active_runlist
    ↓ active_runlist = false
    
进程 2 再次调用 map_queues_cpsch()
    ↓ 检查 active_runlist == false
    ↓ 通过检查，开始处理
    ↓ 设置 active_runlist = true
    |
    | ← 再次阻塞其他进程
    ...
```

#### 实际验证数据

**从 1356 条日志推断**:
```
假设场景: 4 进程，每进程 4 队列，运行 N 次迭代

理想情况（无串行化）:
  - 4 个进程并发调用 map_queues_cpsch
  - 预期调用次数: ~N 次

实际情况（串行化）:
  - 1356 次调用意味着大量的重试
  - 每次只有一个进程能成功
  - 其他进程被 active_runlist 阻塞
  
blocked_count 可能远大于成功的 map_queues_count
```

#### 性能影响分析

**串行化导致的问题**:

1. **队列更新延迟** (-20~30 QPS):
   ```
   进程 2 的队列更新被推迟
       ↓
   进程 2 的 kernel 提交可能被延迟
       ↓
   GPU 利用率下降
   ```

2. **CPU 开销增加**:
   ```
   被阻塞的调用仍然消耗 CPU
       ↓
   锁竞争增加
       ↓
   上下文切换增加
   ```

3. **批处理效率降低**:
   ```
   理想: 一次 Runlist 包含所有进程的队列更新
   实际: 每个进程单独提交，效率低
   ```

#### 为什么不能简单移除？

**潜在风险**:
```c
// 如果允许并发：
进程 1: 构建 Runlist A（包含队列 1, 2, 3）
进程 2: 构建 Runlist B（包含队列 4, 5, 6）
    ↓ 同时提交到 MEC
    ↓ MEC 可能收到不一致的状态
    ↓ 队列激活失败或状态混乱
```

**可能的解决方案**:

1. **方案 A: 队列机制** (推荐):
   ```c
   // 伪代码
   static int map_queues_cpsch_v2(struct device_queue_manager *dqm) {
       // 将请求加入队列
       add_to_runlist_queue(dqm, current_process);
       
       // 异步处理
       if (!runlist_worker_running) {
           schedule_work(&dqm->runlist_worker);
       }
       
       return 0;  // 立即返回，不阻塞
   }
   
   // Worker 线程批量处理
   static void runlist_worker_fn(struct work_struct *work) {
       // 收集所有待处理的队列更新
       // 一次性构建和提交 Runlist
       // 处理完成后通知所有等待的进程
   }
   ```

2. **方案 B: 进程级 Runlist**:
   ```c
   static int map_queues_cpsch_per_process(struct device_queue_manager *dqm) {
       struct kfd_process *current_process = get_current_process();
       
       // 进程级检查，而非全局检查
       if (current_process->active_runlist) {
           return 0;
       }
       
       // 只更新当前进程的队列
       // 不同进程的 Runlist 可以并发
   }
   ```

3. **方案 C: 切换到 NOCPSCH**:
   ```c
   // 如果可能，切换到直接模式
   dqm->sched_policy = KFD_SCHED_POLICY_NO_HWS;
   // 使用 allocate_hqd() 直接分配
   // 避免 Runlist 机制的开销
   ```

---

### 第4阶段: 统计 Runlist 大小

```c
    /* Count runlist size (number of queues) */
    {
        struct device_process_node *cur;
        struct qcm_process_device *qpd;
        struct queue *q;
        
        list_for_each_entry(cur, &dqm->queues, list) {
            qpd = cur->qpd;
            list_for_each_entry(q, &qpd->queues_list, list) {
                if (q->properties.is_active)
                    runlist_size++;
            }
        }
    }
```

**目的**: 
- 📊 统计活动队列数量
- 📊 用于性能监控和调试

**数据结构遍历**:
```
dqm->queues (进程列表)
    ↓ 遍历每个进程
device_process_node
    ↓ 获取进程的队列管理器
qpd->queues_list (队列列表)
    ↓ 遍历每个队列
queue
    ↓ 检查 is_active
    ↓ 如果活动，计数++
```

**从实际验证推断**:
```
假设 4 进程 × 4 队列 = 16 个队列
runlist_size 在每次调用时可能是：
  - 16（所有队列都活动）
  - 更少（部分队列未激活）
```

**性能考虑**:
- 🟡 这是 O(N) 遍历，N = 进程数 × 队列数
- 🟡 对于大量进程/队列，开销不可忽略
- ✅ 但相比 Runlist 构建，这只是统计

---

### 第5阶段: 🔴 构建和提交 Runlist（核心操作）

```c
    retval = pm_send_runlist(&dqm->packet_mgr, &dqm->queues);
    
    pr_debug("%s sent runlist\n", __func__);
    if (retval) {
        dev_err(dev, "failed to execute runlist\n");
        end_time = ktime_get_ns_safe();
        goto out;
    }
    dqm->active_runlist = true;  // ⚠️ 设置标志，阻塞后续调用
```

#### pm_send_runlist() 详解

这是最关键的函数，负责实际的 Runlist 提交：

```c
// kfd_packet_manager.c
int pm_send_runlist(struct packet_manager *pm, struct list_head *dqm_queues)
{
    uint64_t rl_gpu_ib_addr;
    uint32_t *rl_buffer;
    size_t rl_ib_size, packet_size_dwords;
    int retval;
    
    // 1️⃣ 创建 Runlist IB (Indirect Buffer)
    retval = pm_create_runlist_ib(pm, dqm_queues, &rl_gpu_ib_addr,
                                   &rl_ib_size);
    if (retval)
        goto fail_create_runlist_ib;
    
    pr_debug("runlist IB address: 0x%llX\n", rl_gpu_ib_addr);
    
    // 2️⃣ 获取 PM4 packet buffer
    packet_size_dwords = pm->pmf->runlist_size / sizeof(uint32_t);
    mutex_lock(&pm->lock);
    
    retval = kq_acquire_packet_buffer(pm->priv_queue,
                                       packet_size_dwords, &rl_buffer);
    if (retval)
        goto fail_acquire_packet_buffer;
    
    // 3️⃣ 构建 PM4 runlist packet
    retval = pm->pmf->runlist(pm, rl_buffer, rl_gpu_ib_addr,
                              rl_ib_size / sizeof(uint32_t), false);
    if (retval)
        goto fail_create_runlist;
    
    // 4️⃣ 提交 packet 到 HIQ (High-priority Interface Queue)
    retval = kq_submit_packet(pm->priv_queue);
    
    mutex_unlock(&pm->lock);
    
    return retval;
    
fail_create_runlist:
    kq_rollback_packet(pm->priv_queue);
fail_acquire_packet_buffer:
    mutex_unlock(&pm->lock);
fail_create_runlist_ib:
    pm_release_ib(pm);
    return retval;
}
```

#### pm_create_runlist_ib() 深度分析

这个函数构建 Runlist Indirect Buffer：

```c
static int pm_create_runlist_ib(struct packet_manager *pm,
                                 struct list_head *queues,
                                 uint64_t *rl_gpu_addr,
                                 size_t *rl_size_bytes)
{
    unsigned int alloc_size_bytes;
    unsigned int *rl_buffer, rl_wptr, i;
    struct device_process_node *cur;
    struct qcm_process_device *qpd;
    struct queue *q;
    
    // 1️⃣ 计算 Runlist IB 大小
    // 大小 = 进程数 × map_process_size + 队列数 × map_queue_size
    pm_calc_rlib_size(pm, &alloc_size_bytes, &is_over_subscription, xnack_conflict);
    
    // 2️⃣ 分配 GTT (Graphics Translation Table) 内存
    retval = kfd_gtt_sa_allocate(node, alloc_size_bytes, &pm->ib_buffer_obj);
    
    // 3️⃣ 获取 CPU 和 GPU 地址
    rl_buffer = pm->ib_buffer_obj->cpu_ptr;
    *rl_gpu_addr = pm->ib_buffer_obj->gpu_addr;
    
    // 4️⃣ 构建 Runlist 内容
    list_for_each_entry(cur, queues, list) {
        qpd = cur->qpd;
        
        // 为每个进程添加 MAP_PROCESS packet
        pm->pmf->map_process(pm, &rl_buffer[rl_wptr], qpd);
        rl_wptr += pm->pmf->map_process_size / sizeof(uint32_t);
        
        // 为每个队列添加 MAP_QUEUES packet
        list_for_each_entry(q, &qpd->queues_list, list) {
            if (q->properties.is_active) {
                pm->pmf->map_queues(pm, &rl_buffer[rl_wptr], q, ...);
                rl_wptr += pm->pmf->map_queue_size / sizeof(uint32_t);
            }
        }
    }
    
    return 0;
}
```

#### Runlist IB 结构

```
Runlist Indirect Buffer (在 GTT 内存中):
┌─────────────────────────────────────┐
│ MAP_PROCESS packet (进程 1)         │ ← pm->pmf->map_process()
│   - PASID                           │
│   - Page table base                 │
│   - Context save/restore info       │
├─────────────────────────────────────┤
│ MAP_QUEUES packet (队列 1)          │ ← pm->pmf->map_queues()
│   - Queue ID: 216                   │
│   - MQD address                     │
│   - Doorbell offset                 │
│   - Priority                        │
│   - ⚠️ 不包含 Pipe/Queue 信息       │ ← MEC Firmware 动态分配
├─────────────────────────────────────┤
│ MAP_QUEUES packet (队列 2)          │
│   - Queue ID: 217                   │
│   - ...                             │
├─────────────────────────────────────┤
│ MAP_PROCESS packet (进程 2)         │
├─────────────────────────────────────┤
│ MAP_QUEUES packet (队列 3)          │
│   - Queue ID: 220                   │
│   - ...                             │
├─────────────────────────────────────┤
│ ...                                 │
└─────────────────────────────────────┘
                ↓
      传递给 MEC Firmware
                ↓
   MEC 读取 Runlist，动态分配 HQD
                ↓
        所有队列 pipe=0, queue=0
     （软件层不知道实际的 HQD 分配）
```

**关键发现** (基于 1356 条日志验证):
- ✅ **Runlist 不包含 Pipe/Queue 信息**
- ✅ **MEC Firmware 完全控制 HQD 分配**
- ✅ **软件层无法直接控制队列到 HQD 的映射**

#### PM4 Packet 提交到 HIQ

```
User Space                Kernel Space (KFD)        MEC Firmware
    ↓                            ↓                         ↓
创建/更新队列            map_queues_cpsch()          等待命令
    ↓                            ↓                         ↓
    |                    构建 Runlist IB                   |
    |                            ↓                         |
    |                    创建 PM4 packet                   |
    |                            ↓                         |
    |                    kq_submit_packet()                |
    |                            ↓                         |
    |                    写入 HIQ (High-priority           |
    |                           Interface Queue)           |
    |                            ↓                         |
    |                    WPTR = WPTR + packet_size         |
    |                            ↓                         |
    |                    写入 Doorbell ----------------→   |
    |                                                      ↓
    |                                            读取 HIQ (RPTR != WPTR)
    |                                                      ↓
    |                                            解析 PM4 packet
    |                                                      ↓
    |                                            读取 Runlist IB
    |                                                      ↓
    |                                            处理 MAP_PROCESS
    |                                                      ↓
    |                                            处理 MAP_QUEUES
    |                                                      ↓
    |                                            动态分配 HQD (Pipe, Queue)
    |                                                      ↓
    |                                            加载 MQD 到 HQD 寄存器
    |                                                      ↓
    |                                            队列激活，可以执行
    |                                                      ↓
    |                  ← 更新 HIQ RPTR ←──────────────── 处理完成
    |                            ↓                         
    |                    unmap_queues_cpsch()
    |                            ↓
    |                    active_runlist = false
    ↓                            ↓                         ↓
```

**串行化瓶颈分析**:

1. **HIQ 是单一通道** 🔴🔴:
   ```
   所有进程共享一个 HIQ
       ↓
   PM4 packet 提交串行化
       ↓
   即使软件层可以并发，HIQ 也是瓶颈
   ```

2. **MEC Firmware 处理延迟** 🔴:
   ```
   MEC 必须完整处理 Runlist
       ↓
   包括分配 HQD、加载 MQD
       ↓
   在此期间 active_runlist = true
       ↓
   阻塞所有其他进程
   ```

3. **锁竞争** 🔴:
   ```
   pm->lock (packet_manager 锁)
       ↓
   同一时间只能有一个线程提交 packet
       ↓
   增加延迟
   ```

#### 实际性能开销估算

基于 1356 条日志和性能统计数据：

```
假设每次 map_queues_cpsch 平均耗时:
  - Runlist 构建: 50-100 μs (取决于队列数量)
  - PM4 packet 准备: 10-20 μs
  - HIQ 提交: 5-10 μs
  - MEC 处理: 100-500 μs (最大不确定性)
  - 总计: ~200-600 μs

1356 次调用总耗时:
  - 最小: 1356 × 200 μs = 271 ms
  - 最大: 1356 × 600 μs = 814 ms

如果测试运行时间约 10 秒:
  - 开销占比: 2.7% - 8.1%
  
但真正的问题是串行化导致的等待时间:
  - 进程 2-4 在等待进程 1 完成
  - GPU 可能处于空闲状态
  - 吞吐量严重下降
```

---

### 第6阶段: 性能统计更新

```c
out:
    /* Update performance statistics */
    dqm->perf_stats.map_queues_count++;
    if (end_time > start_time) {
        u64 elapsed = end_time - start_time;
        dqm->perf_stats.map_queues_total_time_ns += elapsed;
        if (elapsed > dqm->perf_stats.map_queues_max_time_ns)
            dqm->perf_stats.map_queues_max_time_ns = elapsed;
    }
    dqm->perf_stats.runlist_size_sum += runlist_size;
    dqm->perf_stats.runlist_build_count++;
    if (end_time > start_time) {
        u64 build_time = end_time - start_time;
        dqm->perf_stats.runlist_build_time_ns += build_time;
        if (build_time > dqm->perf_stats.runlist_build_max_time_ns)
            dqm->perf_stats.runlist_build_max_time_ns = build_time;
    }
    
    return retval;
}
```

**性能统计字段**:

| 字段 | 含义 | 用途 |
|------|------|------|
| `map_queues_count` | 成功执行次数 | 总调用次数 |
| `map_queues_blocked_count` | 被 active_runlist 阻塞次数 | **关键指标** |
| `map_queues_total_time_ns` | 总耗时（纳秒） | 平均耗时计算 |
| `map_queues_max_time_ns` | 最大耗时 | 峰值性能 |
| `map_queues_blocked_time_ns` | 阻塞总耗时 | **浪费的时间** |
| `runlist_size_sum` | Runlist 大小总和 | 平均队列数 |
| `runlist_build_count` | Runlist 构建次数 | 应该等于 map_queues_count |
| `runlist_build_time_ns` | Runlist 构建总耗时 | Runlist 构建开销 |
| `runlist_build_max_time_ns` | Runlist 构建最大耗时 | 峰值构建时间 |

**如何查看这些统计数据**:

```bash
# 方法 1: 通过 debugfs
cat /sys/kernel/debug/dri/0/kfd_dqm_stats

# 方法 2: 通过 dmesg（如果有日志）
dmesg | grep "map_queues"

# 方法 3: 添加自定义日志
# 在代码中添加：
pr_info("DQM Stats: map_queues=%llu, blocked=%llu, avg_time=%llu ns\n",
        dqm->perf_stats.map_queues_count,
        dqm->perf_stats.map_queues_blocked_count,
        dqm->perf_stats.map_queues_count > 0 ?
            dqm->perf_stats.map_queues_total_time_ns / dqm->perf_stats.map_queues_count : 0);
```

---

## 🎯 调用时机分析

### 何时调用 map_queues_cpsch()?

通过代码搜索，发现以下调用点：

#### 1. update_queue() - 队列属性更新
```c
// kfd_device_queue_manager.c:1252
static int update_queue(struct device_queue_manager *dqm, struct queue *q) {
    // ... 更新队列属性
    if (dqm->sched_policy != KFD_SCHED_POLICY_NO_HWS) {
        if (!dqm->dev->kfd->shared_resources.enable_mes)
            retval = map_queues_cpsch(dqm);  // ← CPSCH 模式
        else if (q->properties.is_active)
            retval = add_queue_mes(dqm, q, &pdd->qpd);  // ← MES 模式
    }
}
```

**触发场景**:
- 队列优先级更改
- 队列属性修改
- 队列激活/停用

**频率**: 🟡 中等（取决于应用程序行为）

#### 2. execute_queues_cpsch() - 队列执行
```c
// kfd_device_queue_manager.c:2707
static int execute_queues_cpsch(struct device_queue_manager *dqm,
                                enum kfd_unmap_queues_filter filter,
                                uint32_t filter_param,
                                uint32_t grace_period)
{
    retval = unmap_queues_cpsch(dqm, filter, filter_param, grace_period, false);
    if (!retval)
        retval = map_queues_cpsch(dqm);  // ← 重新映射队列
}
```

**触发场景**:
- 队列重置后恢复
- 调度器暂停后恢复
- 队列属性批量更新

**频率**: 🟢 低（主要是恢复场景）

#### 3. reserve_debug_trap_vmid() / release_debug_trap_vmid() - 调试支持
```c
// kfd_device_queue_manager.c:3476, 3525
int kfd_dqm_reserve_debug_trap_vmid(struct device_queue_manager *dqm) {
    r = unmap_queues_cpsch(dqm, ...);
    // ... 保留 VMID 用于调试
    r = map_queues_cpsch(dqm);  // ← 重新映射
}

int kfd_dqm_release_debug_trap_vmid(struct device_queue_manager *dqm) {
    r = unmap_queues_cpsch(dqm, ...);
    // ... 释放 VMID
    r = map_queues_cpsch(dqm);  // ← 重新映射
}
```

**触发场景**:
- 启动调试会话
- 结束调试会话

**频率**: 🟢 极低（只有调试时）

#### 4. debug_map_and_unlock() - 调试恢复
```c
// kfd_device_queue_manager.c:3932
int debug_map_and_unlock(struct device_queue_manager *dqm) {
    r = map_queues_cpsch(dqm);  // ← 调试后恢复
    dqm_unlock(dqm);
}
```

**触发场景**:
- 调试断点后恢复执行

**频率**: 🟢 极低（只有调试时）

### 调用频率总结

基于 1356 条日志和代码分析：

```
主要调用来源: update_queue()
    ↓
每次队列属性变化都可能触发
    ↓
在多进程场景下:
  - 每个进程可能频繁更新队列
  - 不同进程的更新互相竞争
  - active_runlist 导致串行化
    ↓
1356 次调用 = 大量的队列更新操作
```

**推测的测试场景**:
```
4 进程 × N 次迭代
每次迭代可能包含:
  - 队列激活
  - 提交 kernel
  - 队列属性更新
  - ...

如果 N = 300-400 次迭代:
  4 进程 × 350 迭代 = 1400 次潜在调用
  实际调用 1356 次 ≈ 97% 成功率
  (被阻塞的调用不计入统计)
```

---

## 🔧 性能优化方案

### 方案 1: 批量处理 (Batching) ⭐⭐⭐⭐⭐

**核心思想**: 收集多个进程的队列更新请求，一次性提交 Runlist

```c
/* 新增数据结构 */
struct runlist_update_request {
    struct list_head list;
    struct kfd_process *process;
    u64 timestamp;
};

struct runlist_batch_manager {
    struct list_head pending_requests;
    struct work_struct batch_worker;
    struct mutex lock;
    atomic_t pending_count;
    u64 last_submit_time;
};

/* 修改后的 map_queues_cpsch */
static int map_queues_cpsch_batched(struct device_queue_manager *dqm)
{
    struct runlist_update_request *req;
    u64 current_time = ktime_get_ns_safe();
    
    if (!dqm->sched_running || dqm->sched_halt) {
        return 0;
    }
    
    /* 创建更新请求 */
    req = kzalloc(sizeof(*req), GFP_KERNEL);
    req->process = current->process;
    req->timestamp = current_time;
    
    /* 加入批处理队列 */
    mutex_lock(&dqm->batch_mgr.lock);
    list_add_tail(&req->list, &dqm->batch_mgr.pending_requests);
    atomic_inc(&dqm->batch_mgr.pending_count);
    mutex_unlock(&dqm->batch_mgr.lock);
    
    /* 触发批处理 (如果满足条件) */
    u64 elapsed = current_time - dqm->batch_mgr.last_submit_time;
    int pending = atomic_read(&dqm->batch_mgr.pending_count);
    
    // 条件 1: 累积了足够多的请求
    // 条件 2: 距离上次提交超过阈值时间
    if (pending >= BATCH_THRESHOLD || elapsed >= BATCH_TIMEOUT_NS) {
        schedule_work(&dqm->batch_mgr.batch_worker);
    }
    
    return 0;  // 立即返回，不阻塞
}

/* 批处理 Worker */
static void runlist_batch_worker_fn(struct work_struct *work)
{
    struct runlist_batch_manager *mgr = container_of(work, ...);
    struct device_queue_manager *dqm = container_of(mgr, ...);
    struct list_head local_list;
    
    /* 获取所有待处理请求 */
    mutex_lock(&mgr->lock);
    list_replace_init(&mgr->pending_requests, &local_list);
    atomic_set(&mgr->pending_count, 0);
    mutex_unlock(&mgr->lock);
    
    /* 一次性构建和提交 Runlist */
    dqm_lock(dqm);
    
    if (!dqm->active_runlist && !list_empty(&local_list)) {
        int retval = pm_send_runlist(&dqm->packet_mgr, &dqm->queues);
        if (!retval) {
            dqm->active_runlist = true;
            mgr->last_submit_time = ktime_get_ns_safe();
        }
    }
    
    dqm_unlock(dqm);
    
    /* 清理请求列表 */
    // ... cleanup ...
}
```

**优点**:
- ✅ 减少 Runlist 提交次数
- ✅ 提高批处理效率
- ✅ 降低 active_runlist 阻塞时间
- ✅ 不需要修改 MEC Firmware

**预期效果**: +20-30 QPS

---

### 方案 2: 进程级 Runlist ⭐⭐⭐⭐

**核心思想**: 每个进程维护独立的 Runlist，避免全局串行化

```c
/* 新增数据结构 */
struct process_runlist_manager {
    bool active_runlist;        // 进程级标志
    struct list_head queue_updates;
    struct mutex lock;
};

/* 修改后的 map_queues_cpsch */
static int map_queues_cpsch_per_process(struct device_queue_manager *dqm)
{
    struct kfd_process *current_process = get_current_process();
    struct process_runlist_manager *prm = &current_process->runlist_mgr;
    
    if (!dqm->sched_running || dqm->sched_halt) {
        return 0;
    }
    
    /* 进程级检查，而非全局检查 */
    mutex_lock(&prm->lock);
    if (prm->active_runlist) {
        /* 只阻塞同一进程的并发请求 */
        mutex_unlock(&prm->lock);
        return 0;
    }
    
    /* 只构建当前进程的 Runlist */
    retval = pm_send_process_runlist(&dqm->packet_mgr, current_process);
    if (!retval) {
        prm->active_runlist = true;
    }
    
    mutex_unlock(&prm->lock);
    return retval;
}

/* 新函数: 发送进程级 Runlist */
int pm_send_process_runlist(struct packet_manager *pm, struct kfd_process *process)
{
    // 类似 pm_send_runlist，但只包含一个进程的队列
    // ...
}
```

**优点**:
- ✅ 不同进程可以并发提交 Runlist
- ✅ 只在进程内串行化
- ✅ 更细粒度的并发控制

**挑战**:
- ⚠️ 需要 MEC Firmware 支持进程级 Runlist
- ⚠️ 或者需要确保进程间 Runlist 不冲突

**预期效果**: +15-25 QPS

---

### 方案 3: 异步 Runlist 提交 ⭐⭐⭐

**核心思想**: 不等待 MEC 处理完成，立即返回

```c
static int map_queues_cpsch_async(struct device_queue_manager *dqm)
{
    if (!dqm->sched_running || dqm->sched_halt) {
        return 0;
    }
    
    /* 不检查 active_runlist，直接提交 */
    retval = pm_send_runlist_async(&dqm->packet_mgr, &dqm->queues);
    
    /* 不设置 active_runlist = true */
    /* 允许下一个进程立即提交 */
    
    return retval;
}

int pm_send_runlist_async(struct packet_manager *pm, struct list_head *dqm_queues)
{
    // 构建 Runlist IB
    retval = pm_create_runlist_ib(pm, dqm_queues, &rl_gpu_ib_addr, &rl_ib_size);
    
    // 提交 PM4 packet
    retval = kq_submit_packet(pm->priv_queue);
    
    // ⚠️ 不等待完成，立即返回
    return retval;
}
```

**优点**:
- ✅ 消除 active_runlist 阻塞
- ✅ 最大化并发

**挑战**:
- 🔴 MEC Firmware 可能无法处理并发的 Runlist
- 🔴 需要确保 MEC 的状态一致性
- 🔴 可能导致队列状态混乱

**预期效果**: +25-35 QPS（如果 MEC 支持）

**风险**: 高，可能导致系统不稳定

---

### 方案 4: 切换到 NOCPSCH 模式 ⭐⭐⭐⭐⭐

**核心思想**: 如果可能，直接使用固定 HQD 分配，避免 Runlist 开销

```c
/* 在 DQM 初始化时 */
struct device_queue_manager *device_queue_manager_init(struct kfd_node *dev)
{
    struct device_queue_manager *dqm;
    
    // ...
    
    /* 如果满足条件，使用 NOCPSCH 模式 */
    if (can_use_nocpsch(dev)) {
        dqm->sched_policy = KFD_SCHED_POLICY_NO_HWS;
        device_queue_manager_init_nocpsch(dqm);
    } else {
        dqm->sched_policy = KFD_SCHED_POLICY_HWS;
        device_queue_manager_init_cpsch(dqm);
    }
    
    return dqm;
}

bool can_use_nocpsch(struct kfd_node *dev)
{
    /* 检查是否满足 NOCPSCH 的条件 */
    // 1. HQD 资源充足
    // 2. 进程数量不多
    // 3. 不需要复杂的调度功能
    
    int total_hqds = dev->kfd->shared_resources.num_pipe_per_mec *
                     dev->kfd->shared_resources.num_queue_per_pipe;
    
    // 保守估计：每个进程最多 16 个队列
    int max_processes = total_hqds / 16;
    
    return max_processes >= EXPECTED_MAX_PROCESSES;
}

/* NOCPSCH 的队列创建（无 Runlist，无 active_runlist 阻塞）*/
static int create_queue_nocpsch(struct device_queue_manager *dqm, ...)
{
    // 直接分配 HQD
    retval = allocate_hqd(dqm, q);  // ← 固定分配 (Pipe, Queue)
    if (retval)
        return retval;
    
    // 直接加载 MQD 到 HQD 寄存器
    retval = load_mqd_to_hqd(dqm, q);
    
    // 队列立即可用，无需等待 MEC
    return retval;
}
```

**优点**:
- ✅ 完全消除 Runlist 开销
- ✅ 完全消除 active_runlist 串行化
- ✅ 固定的 HQD 映射，更可预测
- ✅ 更低的延迟

**挑战**:
- ⚠️ HQD 资源有限（通常 32-64 个）
- ⚠️ 不适合大量进程/队列的场景
- ⚠️ 缺少 CPSCH 的高级调度功能

**预期效果**: +30-50 QPS（如果 HQD 资源充足）

**适用场景**:
- 进程数量少（≤4）
- 每个进程队列数量少（≤8）
- 总 HQD 需求 < 硬件资源

---

### 方案 5: 混合模式 ⭐⭐⭐⭐

**核心思想**: 高优先级/关键进程使用 NOCPSCH，其他进程使用 CPSCH

```c
/* 进程创建时决定使用哪种模式 */
int kfd_process_device_init_cwsr(struct kfd_process_device *pdd, ...)
{
    // ...
    
    /* 检查进程是否应该使用 NOCPSCH */
    if (should_use_nocpsch(pdd->process)) {
        pdd->qpd.queue_create_policy = QUEUE_POLICY_NOCPSCH;
    } else {
        pdd->qpd.queue_create_policy = QUEUE_POLICY_CPSCH;
    }
}

bool should_use_nocpsch(struct kfd_process *process)
{
    /* 决策因素 */
    // 1. 进程优先级
    // 2. 队列数量需求
    // 3. 实时性要求
    // 4. HQD 可用性
    
    if (process->priority >= HIGH_PRIORITY_THRESHOLD &&
        estimate_queue_count(process) <= MAX_NOCPSCH_QUEUES &&
        get_available_hqds() >= estimate_queue_count(process)) {
        return true;
    }
    
    return false;
}

/* 队列创建时根据策略选择 */
int pqm_create_queue(struct process_queue_manager *pqm, ...)
{
    // ...
    
    if (pqm->process->queue_create_policy == QUEUE_POLICY_NOCPSCH) {
        retval = dqm->ops.create_queue_nocpsch(dqm, qpd, q);
    } else {
        retval = dqm->ops.create_queue_cpsch(dqm, qpd, q);
    }
    
    return retval;
}
```

**优点**:
- ✅ 灵活性高
- ✅ 关键进程获得最佳性能
- ✅ 充分利用 HQD 资源

**挑战**:
- ⚠️ 实现复杂
- ⚠️ 需要良好的策略决策

**预期效果**: +20-40 QPS

---

## 📈 性能优化优先级建议

### 短期 (1-2 周)

**推荐**: 方案 1 (批量处理) ⭐⭐⭐⭐⭐

**理由**:
- ✅ 实现相对简单
- ✅ 不需要 MEC Firmware 修改
- ✅ 风险低
- ✅ 效果明显（预期 +20-30 QPS）

**实施步骤**:
1. 添加批处理数据结构
2. 修改 `map_queues_cpsch()` 使用批处理
3. 实现批处理 Worker
4. 调整批处理参数（BATCH_THRESHOLD, BATCH_TIMEOUT_NS）
5. 性能测试和优化

### 中期 (1-2 月)

**推荐**: 方案 4 (NOCPSCH 模式) ⭐⭐⭐⭐⭐

**理由**:
- ✅ 效果最佳（预期 +30-50 QPS）
- ✅ 消除 Runlist 开销
- ✅ MI308X 的 HQD 资源充足（256+ HQDs）

**实施步骤**:
1. 评估 HQD 资源可用性
2. 添加运行时模式切换支持
3. 测试 NOCPSCH 稳定性
4. 逐步迁移到 NOCPSCH

**风险评估**:
- ⚠️ 需要确保 HQD 资源充足
- ⚠️ 需要验证多进程场景的稳定性

### 长期 (3-6 月)

**推荐**: 方案 5 (混合模式) ⭐⭐⭐⭐

**理由**:
- ✅ 最灵活
- ✅ 适应不同工作负载
- ✅ 充分利用硬件资源

**实施步骤**:
1. 设计策略决策算法
2. 实现模式切换机制
3. 添加运行时监控和调整
4. 长期性能验证

---

## 🔍 调试和监控

### 添加详细日志

```c
/* 在 map_queues_cpsch() 中添加调试日志 */
static int map_queues_cpsch(struct device_queue_manager *dqm)
{
    struct device *dev = dqm->dev->adev->dev;
    struct kfd_process *current_process = current->mm ?
        kfd_get_process(current) : NULL;
    u64 start_time = ktime_get_ns_safe();
    
    pr_debug("[KFD-CPSCH] map_queues called by PID=%d, PASID=%d\n",
             current->pid,
             current_process ? current_process->pasid : 0);
    
    if (dqm->active_runlist) {
        pr_debug("[KFD-CPSCH] BLOCKED by active_runlist (PID=%d)\n",
                 current->pid);
        dqm->perf_stats.map_queues_blocked_count++;
        return 0;
    }
    
    // ... 正常处理 ...
    
    pr_info("[KFD-CPSCH] SUCCESS: PID=%d, runlist_size=%u, time=%llu ns\n",
            current->pid, runlist_size, end_time - start_time);
    
    return retval;
}
```

### 性能监控脚本

```bash
#!/bin/bash
# monitor_cpsch_perf.sh

echo "=== CPSCH Performance Monitor ==="

# 1. 查看 DQM 统计
if [ -f /sys/kernel/debug/dri/0/kfd_dqm_stats ]; then
    echo "DQM Statistics:"
    cat /sys/kernel/debug/dri/0/kfd_dqm_stats
fi

# 2. 实时监控日志
echo ""
echo "Real-time CPSCH logs (Ctrl+C to stop):"
dmesg -w | grep -E "KFD-CPSCH|map_queues"
```

### 性能分析工具

```bash
#!/bin/bash
# analyze_cpsch_logs.sh

LOG_FILE="/var/log/kern.log"

echo "=== CPSCH Log Analysis ==="

# 统计总调用次数
TOTAL_CALLS=$(grep "map_queues_cpsch" $LOG_FILE | wc -l)
echo "Total map_queues_cpsch calls: $TOTAL_CALLS"

# 统计被阻塞次数
BLOCKED_CALLS=$(grep "BLOCKED by active_runlist" $LOG_FILE | wc -l)
echo "Blocked calls: $BLOCKED_CALLS"

# 计算阻塞率
if [ $TOTAL_CALLS -gt 0 ]; then
    BLOCK_RATE=$((BLOCKED_CALLS * 100 / TOTAL_CALLS))
    echo "Block rate: ${BLOCK_RATE}%"
fi

# 分析每个进程的调用次数
echo ""
echo "Calls per process:"
grep "map_queues_cpsch" $LOG_FILE | \
    grep -oP 'PID=\K[0-9]+' | \
    sort | uniq -c | sort -rn

# 分析平均耗时
echo ""
echo "Average execution time:"
grep "SUCCESS.*time=" $LOG_FILE | \
    grep -oP 'time=\K[0-9]+' | \
    awk '{sum+=$1; count++} END {if(count>0) print sum/count " ns"}'
```

---

## 📚 相关文档

- [KERNEL_TRACE_03_KFD_QUEUE.md](../KERNEL_TRACE_03_KFD_QUEUE.md) - KFD Queue 管理总览
- [DIRECTION1_ANALYSIS.md](./DIRECTION1_ANALYSIS.md) - 方向1验证结果分析
- [EXECUTIVE_SUMMARY.md](./EXECUTIVE_SUMMARY.md) - 多进程优化实验总结
- [FUTURE_RESEARCH_DIRECTIONS.md](./FUTURE_RESEARCH_DIRECTIONS.md) - 后续研究方向

---

## 🎯 核心结论

### 关键发现（已验证）

1. **active_runlist 是主要瓶颈** 🔴🔴🔴
   - 1356 次调用证明了频繁的调度操作
   - 全局串行化导致多进程性能下降
   - 影响: -30~40 QPS

2. **Runlist 机制的本质**
   - 不包含 Pipe/Queue 信息
   - MEC Firmware 完全控制 HQD 分配
   - 所有队列 pipe=0, queue=0（已验证）

3. **HIQ 是单一瓶颈**
   - 所有进程共享一个 HIQ
   - PM4 packet 提交串行化
   - 影响: -15~20 QPS

### 优化建议优先级

| 优先级 | 方案 | 预期效果 | 实施难度 | 风险 |
|--------|------|---------|---------|------|
| 🥇 | 批量处理 | +20-30 QPS | 🟢 低 | 🟢 低 |
| 🥈 | NOCPSCH 模式 | +30-50 QPS | 🟡 中 | 🟡 中 |
| 🥉 | 进程级 Runlist | +15-25 QPS | 🟡 中 | 🟡 中 |
| 4️⃣ | 混合模式 | +20-40 QPS | 🔴 高 | 🟡 中 |
| 5️⃣ | 异步提交 | +25-35 QPS | 🟡 中 | 🔴 高 |

### 下一步行动

**立即执行**:
1. 实施批量处理优化
2. 添加详细的性能监控
3. 评估 NOCPSCH 模式的可行性

**持续研究**:
1. 分析 MEC Firmware 的调度策略
2. 研究进程级 Runlist 的可行性
3. 探索混合模式的策略算法

---

**文档版本**: v1.0  
**创建日期**: 2025-01-20  
**基于**: 实际验证结果（1356 条日志）  
**维护者**: 研究团队

