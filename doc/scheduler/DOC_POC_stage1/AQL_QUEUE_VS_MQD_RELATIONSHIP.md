# AQL Queue与MQD的关系详解（基于代码分析）

**日期**: 2026-02-04  
**目的**: 回答"为什么需要MQD？只用AQL Queue + Ring-buffer + Doorbell不够吗？"

---

## 📌 核心问题

```
用户流程:
  Stream_A → Runtime创建AQL_Queue_A (ring-buffer + doorbell)
             → KFD创建MQD_A

疑问: 为什么需要MQD？
      只用AQL_Queue + ring-buffer + doorbell，直接敲门铃不就行了吗？
```

---

## ✅ 简短答案

**MQD是队列的"硬件配置文件"，不是数据存储。**

```
AQL Queue  = 数据通道（用户态ring-buffer存放命令）
MQD        = 元数据配置（告诉硬件这个队列在哪、怎么用）

比喻:
- AQL Queue = 快递箱（装货物）
- MQD       = 快递单（地址、收件人、规格）
- Doorbell  = 按门铃（通知有新快递）
```

**为什么不能只用AQL Queue？**
因为硬件需要知道：这个ring-buffer在哪？多大？doorbell在哪？优先级多少？CWSR上下文保存在哪？

---

## 🔍 代码证据1：queue_properties（软件视图）

**定义位置**: `/usr/src/amdgpu-6.12.12-2194681.el8_preempt/amd/amdkfd/kfd_priv.h:569`

```c
struct queue_properties {
    // ===== 队列基本信息 =====
    enum kfd_queue_type type;           // 队列类型（计算/SDMA等）
    enum kfd_queue_format format;       // 格式（AQL/PM4）
    unsigned int queue_id;              // 队列ID
    
    // ===== Ring-buffer配置（用户态） =====
    uint64_t queue_address;             // Ring-buffer的GPU地址 ⭐
    uint64_t queue_size;                // Ring-buffer大小 ⭐
    void __user *read_ptr;              // 读指针（用户态地址）⭐
    void __user *write_ptr;             // 写指针（用户态地址）⭐
    
    // ===== Doorbell配置 =====
    void __iomem *doorbell_ptr;         // Doorbell的虚拟地址 ⭐
    uint32_t doorbell_off;              // Doorbell在PCIe BAR的偏移 ⭐
    
    // ===== 队列状态 =====
    uint32_t priority;                  // 优先级（0-15）
    bool is_active;                     // 是否激活（影响map/unmap）⭐
    bool is_evicted;                    // 是否被驱逐
    bool is_suspended;                  // 是否被暂停
    
    // ===== CWSR上下文保存（抢占用） =====
    uint64_t ctx_save_restore_area_address;  // Wave状态保存区 ⭐⭐⭐
    uint32_t ctx_save_restore_area_size;     // 保存区大小
    uint32_t ctl_stack_size;                 // 控制栈大小
    uint64_t tba_addr;                       // Trap Handler地址
    uint64_t tma_addr;                       // Trap Memory地址
    
    // ===== BO管理（内核分配的内存对象） =====
    struct amdgpu_bo *wptr_bo;          // Write pointer BO
    struct amdgpu_bo *rptr_bo;          // Read pointer BO
    struct amdgpu_bo *ring_bo;          // Ring-buffer BO ⭐
    struct amdgpu_bo *cwsr_bo;          // CWSR保存区BO
};
```

**关键信息**:
- `queue_address`: AQL Queue的ring-buffer物理地址
- `read_ptr/write_ptr`: 用户态可见的读写指针
- `doorbell_off`: 硬件需要的doorbell偏移
- `ctx_save_restore_area_address`: **抢占时保存Wave状态的地方** ⭐⭐⭐

---

## 🔍 代码证据2：MQD如何使用queue_properties

**函数**: `update_mqd()` - 将软件配置写入MQD硬件描述符  
**位置**: `/usr/src/amdgpu-6.12.12-2194681.el8_preempt/amd/amdkfd/kfd_mqd_manager_v9.c:290`

```c
static void update_mqd(struct mqd_manager *mm, void *mqd,
                       struct queue_properties *q,
                       struct mqd_update_info *minfo)
{
    struct v9_mqd *m;
    m = get_mqd(mqd);

    // 1. 配置Ring-buffer地址和大小 ⭐
    m->cp_hqd_pq_control &= ~CP_HQD_PQ_CONTROL__QUEUE_SIZE_MASK;
    m->cp_hqd_pq_control |= order_base_2(q->queue_size / 4) - 1;
    
    m->cp_hqd_pq_base_lo = lower_32_bits((uint64_t)q->queue_address >> 8);
    m->cp_hqd_pq_base_hi = upper_32_bits((uint64_t)q->queue_address >> 8);
    //    ↑↑↑ 硬件通过这个地址找到AQL Queue的ring-buffer

    // 2. 配置Read/Write指针地址 ⭐
    m->cp_hqd_pq_rptr_report_addr_lo = lower_32_bits((uint64_t)q->read_ptr);
    m->cp_hqd_pq_rptr_report_addr_hi = upper_32_bits((uint64_t)q->read_ptr);
    m->cp_hqd_pq_wptr_poll_addr_lo = lower_32_bits((uint64_t)q->write_ptr);
    m->cp_hqd_pq_wptr_poll_addr_hi = upper_32_bits((uint64_t)q->write_ptr);
    //    ↑↑↑ 硬件通过这些地址读取当前读写位置

    // 3. 配置Doorbell偏移 ⭐
    m->cp_hqd_pq_doorbell_control =
        q->doorbell_off << CP_HQD_PQ_DOORBELL_CONTROL__DOORBELL_OFFSET__SHIFT;
    //    ↑↑↑ 硬件知道去哪里监听doorbell

    // 4. 配置CWSR上下文保存（抢占用）⭐⭐⭐
    if (mm->dev->kfd->cwsr_enabled && q->ctx_save_restore_area_address) {
        m->cp_hqd_persistent_state |=
            (1 << CP_HQD_PERSISTENT_STATE__QSWITCH_MODE__SHIFT);
        m->cp_hqd_ctx_save_base_addr_lo =
            lower_32_bits(q->ctx_save_restore_area_address);
        m->cp_hqd_ctx_save_base_addr_hi =
            upper_32_bits(q->ctx_save_restore_area_address);
        m->cp_hqd_ctx_save_size = q->ctx_save_restore_area_size;
        //    ↑↑↑ 硬件知道抢占时把Wave状态保存到哪里
    }
}
```

---

## 🎯 为什么需要MQD？关键原因

### 原因1: 硬件需要知道队列配置 ⭐⭐⭐⭐⭐

```
没有MQD，硬件怎么知道：
  ❓ Ring-buffer在哪？（cp_hqd_pq_base）
  ❓ Ring-buffer多大？（cp_hqd_pq_control）
  ❓ Read/Write指针在哪？（cp_hqd_pq_rptr_report_addr等）
  ❓ Doorbell在哪？（cp_hqd_pq_doorbell_control）
  ❓ 优先级是多少？（cp_hqd_pipe_priority）

用户只能：
  ✓ 写ring-buffer（填充命令）
  ✓ 敲doorbell（通知有新命令）
  
但不能直接配置硬件寄存器！
```

**解决方案**: MQD就是一个"配置模板"，KFD初始化好，硬件scheduler加载到HQD寄存器。

---

### 原因2: 抢占与上下文切换 ⭐⭐⭐⭐⭐

**CWSR (Compute Wave Save/Restore)** 是抢占的核心机制：

```
场景: Online-AI抢占Offline-AI

步骤1: Offline队列正在执行
  - 1000个Wave在GPU上运行
  - 每个Wave有寄存器状态、LDS数据、PC等

步骤2: Online队列需要执行，发起抢占
  KFD → unmap Offline队列 → HWS接收命令

步骤3: HWS触发CWSR ⭐⭐⭐
  硬件自动：
    1. 暂停所有Wave
    2. 读取MQD中的ctx_save_restore_area_address
    3. 把1000个Wave的状态保存到这个地址
    4. 标记队列为"已保存"
    
步骤4: Online队列map并执行

步骤5: Online完成，Offline resume
  KFD → map Offline队列 → HWS加载MQD
  硬件自动：
    1. 读取MQD中的ctx_save_restore_area_address
    2. 从这个地址恢复1000个Wave状态
    3. 继续执行
```

**关键**: 如果没有MQD记录`ctx_save_restore_area_address`，硬件不知道把Wave状态保存到哪里！

**代码证据**: `kfd_mqd_manager_v9.c:254-265`

```c
if (mm->dev->kfd->cwsr_enabled && q->ctx_save_restore_area_address) {
    m->cp_hqd_persistent_state |=
        (1 << CP_HQD_PERSISTENT_STATE__QSWITCH_MODE__SHIFT);
    m->cp_hqd_ctx_save_base_addr_lo =
        lower_32_bits(q->ctx_save_restore_area_address);
    m->cp_hqd_ctx_save_base_addr_hi =
        upper_32_bits(q->ctx_save_restore_area_address);
    m->cp_hqd_ctx_save_size = q->ctx_save_restore_area_size;
    //    ↑↑↑ MQD告诉硬件：抢占时保存到这里
}
```

---

### 原因3: 多队列管理（CPSCH模式）⭐⭐⭐⭐

**Runlist机制**: HWS通过runlist批量管理多个队列

```
场景: 系统有80个MQD（10个/GPU * 8个GPU）

HWS（硬件调度器）需要：
  1. 遍历runlist IB（Indirect Buffer）
  2. 对于每个MQD：
     - 读取MQD的queue_address → 知道ring-buffer在哪
     - 读取MQD的doorbell_off → 监听这个doorbell
     - 读取MQD的priority → 决定调度优先级
     - 读取MQD的is_active标志 → 决定是否map到HQD
  3. 动态map/unmap队列到有限的HQD资源

如果没有MQD：
  ❌ 硬件无法批量管理多个队列
  ❌ 无法实现超额订阅（80个MQD → 32个HQD/XCC）
  ❌ 无法动态调度
```

**代码证据**: Runlist发送 - `kfd_packet_manager.c:359`

```c
int pm_send_runlist(struct packet_manager *pm, struct list_head *dqm_queues)
{
    // 遍历所有队列，收集MQD指针
    list_for_each_entry(kq, dqm_queues, list) {
        // 把每个队列的MQD地址写入runlist IB
        packet->map_queues.mqd_addr_lo = lower_32_bits(kq->mqd_gpu_addr);
        packet->map_queues.mqd_addr_hi = upper_32_bits(kq->mqd_gpu_addr);
    }
    
    // 发送runlist给HIQ → HWS加载所有MQD
    pm_send_command(pm, packet, ...);
}
```

---

### 原因4: 状态持久化与恢复 ⭐⭐⭐

**场景**: 队列被unmap后，状态如何保持？

```
时刻T0: 队列A mapped，HQD寄存器配置好
  HQD.cp_hqd_pq_base = 0x1000_0000  (ring-buffer地址)
  HQD.cp_hqd_pq_rptr = 100          (读到100个包)
  HQD.cp_hqd_pq_wptr = 150          (写了150个包)
  HQD.cp_hqd_doorbell = 0x5000      (doorbell地址)

时刻T1: 队列A被unmap（让给高优先级队列）
  问题: HQD寄存器被清空或分配给其他队列，状态丢失？

时刻T2: 队列A重新map
  问题: 如何恢复之前的配置？
  
答案: MQD保存了完整配置！
  ✓ MQD.cp_hqd_pq_base 始终是 0x1000_0000
  ✓ MQD.cp_hqd_pq_rptr/wptr 更新为当前值
  ✓ MQD.cp_hqd_doorbell 始终是 0x5000
  
  当队列重新map时，HWS从MQD恢复所有配置到HQD！
```

**代码证据**: Load MQD到HQD - `kfd_mqd_manager_v9.c:278`

```c
static int load_mqd(struct mqd_manager *mm, void *mqd,
                    uint32_t pipe_id, uint32_t queue_id,
                    struct queue_properties *p, struct mm_struct *mms)
{
    // 将MQD加载到HQD寄存器
    return mm->dev->kfd2kgd->hqd_load(mm->dev->adev, mqd, pipe_id, queue_id,
                                      (uint32_t __user *)p->write_ptr,
                                      wptr_shift, 0, mms, 0);
    //            ↑↑↑ 硬件从MQD读取所有配置，写入HQD寄存器
}
```

---

## 📊 完整流程图

### 1. 队列创建流程

```
用户态:
  hipStreamCreate(stream_A)
    ↓
  Runtime创建AQL Queue:
    - 分配ring-buffer (queue_address)
    - 分配read/write指针
    - mmap doorbell (doorbell_ptr)
    ↓
  hsa_queue_create() → ioctl(KFD_IOC_CREATE_QUEUE)
    ↓
────────────────────────────────────────
内核态KFD:
  create_queue_cpsch()
    ↓
  1. 分配queue_properties结构
     - 记录queue_address（用户传入）
     - 记录doorbell_off（KFD分配）
     - 分配ctx_save_restore_area（CWSR用）⭐
    ↓
  2. 创建MQD
     mqd_mgr->init_mqd(mqd, &queue_properties)
       ↓
       update_mqd():
         - m->cp_hqd_pq_base = queue_address
         - m->cp_hqd_doorbell = doorbell_off
         - m->cp_hqd_ctx_save_addr = cwsr_area ⭐
    ↓
  3. 添加到runlist（如果is_active=true）
     map_queues_cpsch() → pm_send_runlist()
       ↓
       HIQ发送runlist → HWS加载MQD到HQD
```

**关键**: 用户创建AQL Queue时，只提供ring-buffer地址，KFD负责：
- 创建MQD并填充配置
- 分配CWSR区域
- 管理doorbell分配
- 通过HIQ通知HWS

---

### 2. 提交Kernel流程（用户视角）

```
用户态:
  1. 用户写PM4/AQL命令到ring-buffer
     memcpy(queue_address + write_ptr, packet, size);
     
  2. 更新write_ptr
     write_ptr += size;
     
  3. 敲doorbell
     *doorbell_ptr = write_ptr;  // 写入doorbell寄存器
       ↓
────────────────────────────────────────
硬件:
  4. Doorbell控制器检测到写入
     → 根据MQD.cp_hqd_pq_doorbell_control找到对应队列
     
  5. HWS检查队列状态
     → 读取MQD.cp_hqd_pq_base（ring-buffer地址）
     → 读取MQD.cp_hqd_pq_wptr（新的write_ptr）
     
  6. CP Firmware从ring-buffer取命令
     addr = MQD.cp_hqd_pq_base + read_ptr;
     fetch_packet(addr);
     
  7. 提交给GPU执行
```

**关键**: 
- 用户只操作ring-buffer和doorbell（用户态内存）
- 硬件通过MQD找到ring-buffer位置
- MQD是硬件和软件的"桥梁"

---

### 3. 抢占流程（CWSR）⭐⭐⭐

```
时刻T0: Offline队列执行中
  HQD已加载MQD配置
  1000个Wave在GPU运行
  
时刻T1: Online队列到达，发起抢占
  用户态: ioctl(KFD_IOC_DBG_TRAP_SUSPEND_QUEUES, offline_queue_id)
    ↓
  KFD: unmap_queues_cpsch()
    ↓
    pm_send_unmap_queue() → HIQ发送UNMAP包
      ↓
────────────────────────────────────────
时刻T2: HWS收到UNMAP命令
  1. 暂停队列执行（停止fetch新packet）
  
  2. 触发CWSR保存 ⭐⭐⭐
     for each Wave:
       addr = MQD.cp_hqd_ctx_save_base_addr + wave_id * wave_size;
       save_wave_state(addr);
       //   ↑↑↑ 硬件自动保存到MQD指定的地址
  
  3. 更新MQD状态
     MQD.cp_hqd_pq_rptr = current_rptr;  (保存当前读位置)
     MQD.is_active = false;
  
  4. 释放HQD资源
     HQD寄存器清空，可分配给其他队列
────────────────────────────────────────
时刻T3: Online队列map并执行

时刻T4: Online完成，Offline resume
  KFD: map_queues_cpsch() → HIQ发送MAP包
    ↓
  HWS加载MQD:
    1. 恢复HQD配置
       HQD.cp_hqd_pq_base = MQD.cp_hqd_pq_base;
       HQD.cp_hqd_pq_rptr = MQD.cp_hqd_pq_rptr;
       HQD.cp_hqd_doorbell = MQD.cp_hqd_doorbell;
    
    2. 触发CWSR恢复 ⭐⭐⭐
       for each Wave:
         addr = MQD.cp_hqd_ctx_save_base_addr + wave_id * wave_size;
         restore_wave_state(addr);
         //   ↑↑↑ 硬件自动从MQD指定的地址恢复
    
    3. 继续执行
       从MQD.cp_hqd_pq_rptr位置继续读取命令
```

**关键**: CWSR完全依赖MQD中的`ctx_save_restore_area_address`！

---

## 🎯 总结：AQL Queue vs MQD

| 维度 | AQL Queue | MQD |
|------|-----------|-----|
| **本质** | 数据通道 | 元数据配置 |
| **位置** | 用户态可见（mmap） | 内核态管理 |
| **内容** | PM4/AQL命令包 | 队列配置参数 |
| **大小** | 可变（通常64KB-1MB） | 固定（~4KB） |
| **谁写入** | 用户态Runtime | KFD驱动 |
| **谁读取** | GPU CP Firmware | GPU HWS + CP |
| **生命周期** | 队列销毁时释放 | 队列存在期间持久 |
| **作用** | 存放待执行的命令 | 告诉硬件如何处理命令 |

### 核心逻辑关系

```
Stream_A (HIP)
  ↓
AQL_Queue_A (Runtime创建)
  ├── ring_buffer (存放命令) ← 用户写入
  ├── read_ptr    (GPU更新)
  ├── write_ptr   (用户更新)
  └── doorbell    (用户敲响)
  
  关联↓
  
MQD_A (KFD创建)
  ├── cp_hqd_pq_base         = &ring_buffer    ← 告诉硬件ring在哪
  ├── cp_hqd_pq_rptr_addr    = &read_ptr       ← 告诉硬件rptr在哪
  ├── cp_hqd_pq_wptr_addr    = &write_ptr      ← 告诉硬件wptr在哪
  ├── cp_hqd_doorbell        = doorbell_off    ← 告诉硬件doorbell在哪
  ├── cp_hqd_ctx_save_addr   = cwsr_area       ← ⭐抢占时Wave保存在哪
  ├── cp_hqd_priority        = priority        ← 告诉HWS优先级
  └── ... (其他50+个配置字段)
  
  加载到↓
  
HQD_X (硬件寄存器)
  ← HWS从MQD加载配置
  ← 硬件根据这些配置执行队列
```

---

## 🔑 回答原问题

### Q: 为什么不能只用AQL_Queue + ring-buffer + doorbell？

**A: 因为缺少以下关键能力**:

1. ❌ **硬件配置**: 硬件不知道ring-buffer在哪、多大、doorbell在哪
2. ❌ **抢占支持**: 没地方记录CWSR上下文保存区地址
3. ❌ **状态持久化**: unmap后配置丢失，无法恢复
4. ❌ **多队列管理**: HWS无法批量管理和调度多个队列
5. ❌ **优先级调度**: 没地方记录队列优先级信息

### Q: MQD存在哪里？

**A: 系统内存（GTT或VRAM）**，通过GPU地址访问：

```c
// MQD分配（内核态）
struct amdgpu_bo *mqd_bo = kfd_gtt_sa_allocate(mqd_size);
uint64_t mqd_gpu_addr = amdgpu_bo_gpu_offset(mqd_bo);

// 发送给HWS
packet->map_queues.mqd_addr = mqd_gpu_addr;
pm_send_to_hiq(packet);  // 通过HIQ发送

// HWS读取MQD
hws_load_mqd(mqd_gpu_addr);  // 硬件从这个地址读取MQD
```

### Q: 用户能直接访问MQD吗？

**A: 不能！MQD是内核态数据结构**:

```
用户态可访问:
  ✓ AQL Queue ring-buffer (mmap)
  ✓ read/write指针 (mmap)
  ✓ doorbell (mmap)

用户态不可访问:
  ❌ MQD (内核专属，通过sysfs debugfs只读查看)
  ❌ HQD寄存器 (硬件专属)
  ❌ HIQ (KFD专用队列)
```

---

## 📚 相关文档

- `New_DEEP_DIVE_MI308X_QUEUE_MECHANISMS.md` - 队列管理机制
- `New_MAP_UNMAP_DETAILED_PROCESS.md` - Map/Unmap详细流程
- `MI308X_HARDWARE_INFO.md` - 硬件配置

---

## 🔗 代码参考

**关键文件**:
- `kfd_priv.h:569` - queue_properties定义
- `kfd_mqd_manager_v9.c:290` - update_mqd()实现
- `kfd_mqd_manager_v9.c:254` - CWSR配置
- `kfd_device_queue_manager.c` - map/unmap队列
- `kfd_packet_manager.c:359` - runlist发送

**验证MQD内容**:
```bash
# 查看所有MQD
sudo cat /sys/kernel/debug/kfd/mqds

# 查看HQD状态（加载了哪些MQD）
sudo cat /sys/kernel/debug/kfd/hqds
```

---

**最后更新**: 2026-02-04  
**验证状态**: ✅ 基于代码分析  
**适用平台**: MI308X (CPSCH模式)
