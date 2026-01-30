# Kernel提交流程追踪 (3/5) - KFD驱动层Queue管理

**范围**: KFD驱动层的Queue创建和管理  
**代码路径**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/`  
**关键操作**: 处理CREATE_QUEUE ioctl、Device Queue Manager、MES Queue添加

---

## 🔴 重要更新（2026-01-20）

> **关键发现**: CPSCH 模式下，**Queue ID 不直接映射到固定的 HQD (Hardware Queue Descriptor)**！
> 
> 这与直观理解和 NOCPSCH 模式有本质差异：
> - **CPSCH 模式**: Queue ID → Runlist → MEC Firmware 动态分配 HQD（不可预测）
> - **NOCPSCH 模式**: Queue ID → `allocate_hqd()` → 固定的 HQD (Pipe, Queue)
> 
> 这意味着：
> 1. ❌ 基于 Queue ID 的优化在 CPSCH 模式下可能无效
> 2. ✅ HQD 分配由 MEC Firmware 控制，软件层无法直接控制
> 3. ⚠️ MI308X 使用 CPSCH 模式（不支持 MES）
> 
> **详见**: [8.4 CPSCH Queue 管理机制详解](#84-cpsch-queue-管理机制详解-)

---

## 📋 本层概述

KFD (Kernel Fusion Driver) 是AMD GPU的内核驱动，负责：
1. 处理来自HSA Runtime的ioctl请求
2. 管理GPU Queue的创建和销毁
3. 与MES调度器交互
4. 管理进程的GPU资源（Context、Entity等）

---

## 1️⃣ KFD设备文件打开

### 1.1 /dev/kfd设备节点

```bash
# 查看KFD设备文件
$ ls -l /dev/kfd
crw-rw-rw- 1 root root 511, 0 Jan 16 10:00 /dev/kfd

# 主设备号：511
# 次设备号：0
```

### 1.2 kfd_open() 处理

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_chardev.c`

```c
static int kfd_open(struct inode *inode, struct file *filep)
{
    struct kfd_process *process;
    bool is_32bit_user_mode;
    
    // 1. 检查用户模式（32位还是64位）
    is_32bit_user_mode = in_compat_syscall();
    
    if (is_32bit_user_mode) {
        dev_warn(kfd_device,
                "Process %d (32-bit) failed to open /dev/kfd\n",
                current->pid);
        return -EPERM;  // 不支持32位
    }
    
    // 2. 创建或获取KFD进程对象
    // 每个应用进程对应一个kfd_process
    process = kfd_create_process(current);
    if (IS_ERR(process)) {
        return PTR_ERR(process);
    }
    
    // 3. 增加引用计数
    kref_get(&process->ref);
    
    // 4. 保存到文件私有数据
    filep->private_data = process;
    
    dev_dbg(kfd_device, "Process %d opened /dev/kfd\n", 
            current->pid);
    
    return 0;
}
```

**关键发现**:
- ✅ 每个进程打开 `/dev/kfd` 时创建一个 `kfd_process` 对象
- ✅ `kfd_process` 代表应用进程在KFD驱动中的抽象
- ✅ 一个进程对应一个 `kfd_process`

---

## 2️⃣ CREATE_QUEUE ioctl处理

### 2.1 ioctl入口

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_chardev.c`

```c
static long kfd_ioctl(struct file *filep, unsigned int cmd, unsigned long arg)
{
    struct kfd_process *process;
    long err = -EINVAL;
    
    // 1. 获取进程对象
    process = filep->private_data;
    
    // 2. 根据ioctl命令分发
    switch (cmd) {
    case AMDKFD_IOC_GET_VERSION:
        err = kfd_ioctl_get_version(filep, process, (void __user *)arg);
        break;
        
    case AMDKFD_IOC_CREATE_QUEUE:
        // 这是我们关注的！
        err = kfd_ioctl_create_queue(filep, process, (void __user *)arg);
        break;
        
    case AMDKFD_IOC_DESTROY_QUEUE:
        err = kfd_ioctl_destroy_queue(filep, process, (void __user *)arg);
        break;
        
    // ... 其他ioctl命令
    
    default:
        dev_err(kfd_device, "Unknown ioctl command: 0x%x\n", cmd);
        err = -ENOIOCTLCMD;
    }
    
    return err;
}
```

### 2.2 kfd_ioctl_create_queue() 实现

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_chardev.c`

```c
static int kfd_ioctl_create_queue(struct file *filep,
                                  struct kfd_process *p,
                                  void __user *data)
{
    struct kfd_ioctl_create_queue_args args;
    struct kfd_node *dev;
    int err = 0;
    struct queue_properties q_properties;
    
    // 1. 从用户空间拷贝参数
    if (copy_from_user(&args, data, sizeof(args)))
        return -EFAULT;
    
    pr_debug("Creating queue for process %d\n", p->pasid);
    
    // 2. 验证参数
    if (args.queue_type > KFD_IOC_QUEUE_TYPE_LAST) {
        pr_err("Invalid queue type: %d\n", args.queue_type);
        return -EINVAL;
    }
    
    // 3. 根据gpu_id获取设备对象
    dev = kfd_device_by_id(args.gpu_id);
    if (!dev) {
        pr_err("Invalid GPU ID: %d\n", args.gpu_id);
        return -EINVAL;
    }
    
    // 4. 初始化queue属性
    memset(&q_properties, 0, sizeof(struct queue_properties));
    
    // 5. 从用户参数设置queue属性
    // 这是关键步骤！
    err = set_queue_properties_from_user(&q_properties, &args);
    if (err) {
        return err;
    }
    
    // 6. 设置进程相关信息
    q_properties.process = p;
    q_properties.cu_mask = NULL;
    
    // 7. 调用进程队列管理器创建queue
    // 这是核心函数！
    err = pqm_create_queue(p, dev, filep, &q_properties, &args.queue_id);
    if (err != 0) {
        pr_err("Failed to create queue: %d\n", err);
        goto err_create_queue;
    }
    
    // 8. 将结果拷贝回用户空间
    // 返回: queue_id, doorbell_offset等
    if (copy_to_user(data, &args, sizeof(args))) {
        err = -EFAULT;
        goto err_copy_to_user;
    }
    
    return 0;
    
err_copy_to_user:
    pqm_destroy_queue(p, args.queue_id);
err_create_queue:
    return err;
}
```

### 2.3 set_queue_properties_from_user() - 设置Queue属性

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_chardev.c`

```c
static int set_queue_properties_from_user(struct queue_properties *q_properties,
                                          struct kfd_ioctl_create_queue_args *args)
{
    // 1. 设置queue类型
    q_properties->type = args->queue_type;
    
    // 2. 设置queue地址和大小
    q_properties->queue_address = args->ring_base_address;
    q_properties->queue_size = args->ring_size;
    
    // 3. 设置读写指针地址
    q_properties->read_ptr = (uint32_t *) args->read_pointer_address;
    q_properties->write_ptr = (uint32_t *) args->write_pointer_address;
    
    // 4. 设置优先级
    q_properties->priority = args->queue_priority;
    q_properties->queue_percent = args->queue_percentage;
    
    // 5. 设置EOP (End Of Pipe) buffer
    if (args->eop_buffer_address && args->eop_buffer_size) {
        q_properties->eop_ring_buffer_address = args->eop_buffer_address;
        q_properties->eop_ring_buffer_size = args->eop_buffer_size;
    }
    
    // 6. 设置Context保存恢复地址
    if (args->ctx_save_restore_address) {
        q_properties->ctx_save_restore_area_address =
            args->ctx_save_restore_address;
        q_properties->ctx_save_restore_area_size =
            args->ctx_save_restore_size;
    }
    
    // 7. 设置控制栈
    if (args->ctl_stack_size) {
        q_properties->ctl_stack_size = args->ctl_stack_size;
    }
    
    return 0;
}
```

---

## 3️⃣ Process Queue Manager (PQM)

### 3.1 pqm_create_queue() 实现

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_process_queue_manager.c`

```c
int pqm_create_queue(struct kfd_process *p,
                    struct kfd_node *dev,
                    struct file *f,
                    struct queue_properties *properties,
                    unsigned int *qid)
{
    int retval;
    struct kfd_process_device *pdd;
    struct process_queue_node *pqn;
    struct queue *q;
    
    // 1. 获取或创建process_device对象
    // 每个进程在每个GPU上有一个pdd
    pdd = kfd_get_process_device_data(dev, p);
    if (!pdd) {
        pr_err("Process device doesn't exist\n");
        return -EINVAL;
    }
    
    // 2. 分配process_queue_node
    pqn = kzalloc(sizeof(*pqn), GFP_KERNEL);
    if (!pqn) {
        return -ENOMEM;
    }
    
    // 3. 根据queue类型调用不同的创建函数
    switch (properties->type) {
    case KFD_QUEUE_TYPE_COMPUTE:
    case KFD_QUEUE_TYPE_COMPUTE_AQL:
        // Compute AQL Queue - 我们的情况！
        retval = create_cp_queue(pqm, dev, &q, properties, f, *qid);
        break;
        
    case KFD_QUEUE_TYPE_SDMA:
    case KFD_QUEUE_TYPE_SDMA_XGMI:
        // SDMA Queue
        retval = create_sdma_queue(pqm, dev, &q, properties, f);
        break;
        
    default:
        pr_err("Invalid queue type: %d\n", properties->type);
        retval = -EINVAL;
    }
    
    if (retval != 0) {
        kfree(pqn);
        return retval;
    }
    
    // 4. 设置pqn属性
    pqn->q = q;
    pqn->kq = NULL;
    
    // 5. 生成queue ID
    retval = ida_simple_get(&p->pqm.queue_slot_index,
                           1, MAX_PROCESS_QUEUES, GFP_KERNEL);
    if (retval < 0) {
        pr_err("Failed to allocate queue ID\n");
        goto err_allocate_qid;
    }
    *qid = retval;
    pqn->q_id = *qid;
    
    // 6. 添加到进程的queue列表
    list_add(&pqn->process_queue_list, &p->pqm.queues);
    
    pr_debug("Created queue %d for process %d\n", *qid, p->pasid);
    
    return 0;
    
err_allocate_qid:
    // 错误处理
    destroy_queue(q);
    kfree(pqn);
    return retval;
}
```

### 3.2 create_cp_queue() - 创建Compute Queue

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_process_queue_manager.c`

```c
static int create_cp_queue(struct process_queue_manager *pqm,
                          struct kfd_node *dev,
                          struct queue **q,
                          struct queue_properties *q_properties,
                          struct file *f,
                          unsigned int qid)
{
    int retval;
    
    // 1. 调用设备队列管理器创建queue
    // DQM (Device Queue Manager) 负责硬件层面的queue管理
    retval = dev->dqm->ops.create_queue(dev->dqm,
                                       q,
                                       q_properties,
                                       &pdd->qpd);
    
    if (retval != 0) {
        pr_err("DQM create queue failed: %d\n", retval);
        return retval;
    }
    
    // 2. 设置doorbell地址
    // doorbell_offset会返回给用户空间
    q_properties->doorbell_off =
        kfd_get_doorbell_dw_offset_in_bar(dev, pdd, *q);
    
    // 3. 返回doorbell offset给用户空间
    // 用户空间用这个offset进行mmap
    
    return 0;
}
```

---

## 4️⃣ Device Queue Manager (DQM)

### 4.1 DQM初始化

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_device_queue_manager.c`

```c
struct device_queue_manager *device_queue_manager_init(struct kfd_node *dev)
{
    struct device_queue_manager *dqm;
    
    // 分配DQM结构
    dqm = kzalloc(sizeof(*dqm), GFP_KERNEL);
    if (!dqm)
        return NULL;
    
    dqm->dev = dev;
    
    // 根据GPU架构选择不同的DQM实现
    switch (dev->adev->asic_type) {
    case CHIP_ARCTURUS:      // MI100 (CDNA1)
    case CHIP_ALDEBARAN:     // MI250X, MI210 (CDNA2)
    case CHIP_AQUA_VANJARAM: // MI300A/X (CDNA3) 或 MI308X (ALDEBARAN)
        // 选择 v12 版本的 DQM 实现
        // 注意：并非所有这些芯片都支持 MES！
        // - CHIP_AQUA_VANJARAM (MI300A/X): 支持 MES (GC IP 12.0.x)
        // - CHIP_AQUA_VANJARAM (MI308X): 不支持 MES (GC IP 9.4.2/3) ← 使用 CPSCH
        // - CHIP_ALDEBARAN (MI250X): 支持 MES (GC IP 9.4.1)
        // - CHIP_ARCTURUS (MI100): 不支持 MES (GC IP 9.4.0) ← 使用 CPSCH
        // DQM v12 内部会根据 enable_mes 标志选择 MES 或 CPSCH 路径
        device_queue_manager_init_v12(dqm);
        break;
        
    case CHIP_VEGA10:        // Vega 10
    case CHIP_VEGA20:        // Vega 20, MI50, MI60
        // 使用CPSCH调度器（旧架构）
        device_queue_manager_init_v9(dqm);
        break;
        
    default:
        pr_err("Unsupported ASIC type: %d\n", dev->adev->asic_type);
        kfree(dqm);
        return NULL;
    }
    
    return dqm;
}
```

### 4.2 DQM create_queue() - MES模式

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_device_queue_manager_v12.c`

```c
static int create_queue_mes(struct device_queue_manager *dqm,
                           struct queue **q,
                           struct queue_properties *properties,
                           struct qcm_process_device *qpd)
{
    struct queue *new_q;
    struct mes_add_queue_input queue_input;
    int retval;
    
    // 1. 分配queue结构
    new_q = kzalloc(sizeof(*new_q), GFP_KERNEL);
    if (!new_q)
        return -ENOMEM;
    
    // 2. 初始化queue属性
    new_q->properties = *properties;
    new_q->device = dqm->dev;
    new_q->process = properties->process;
    
    // 3. 分配MQD (Memory Queue Descriptor)
    // MQD是queue的硬件描述符
    retval = allocate_mqd(dqm->dev, &new_q->mqd, &new_q->mqd_mem_obj);
    if (retval != 0) {
        goto err_allocate_mqd;
    }
    
    // 4. 初始化MQD
    dqm->ops.init_mqd(dqm, &new_q->mqd, &new_q->mqd_mem_obj,
                     &new_q->gart_mqd_addr, &new_q->properties);
    
    // 5. 准备MES add_queue参数
    memset(&queue_input, 0, sizeof(struct mes_add_queue_input));
    queue_input.process_id = qpd->pqm->process->pasid;
    queue_input.page_table_base_addr = qpd->page_table_base;
    queue_input.process_va_start = 0;
    queue_input.process_va_end = (dqm->dev->adev->vm_manager.max_pfn - 1) << PAGE_SHIFT;
    queue_input.process_quantum = 10000;  // 10ms
    queue_input.process_context_addr = qpd->proc_ctx_bo;
    queue_input.gang_context_addr = new_q->gang_ctx_gpu_addr;
    queue_input.queue_type = convert_to_mes_queue_type(properties->type);
    queue_input.mqd_addr = new_q->gart_mqd_addr;
    queue_input.wptr_addr = (uint64_t)properties->write_ptr;
    queue_input.queue_size = properties->queue_size;
    queue_input.doorbell_offset = properties->doorbell_off;
    queue_input.page_table_base_addr = qpd->page_table_base;
    
    // 6. 调用MES添加硬件queue
    // 这是关键！会调用到amdgpu driver的MES接口
    retval = dqm->dev->adev->mes.funcs->add_hw_queue(&dqm->dev->adev->mes,
                                                      &queue_input);
    if (retval) {
        pr_err("Failed to add queue to MES: %d\n", retval);
        goto err_add_queue_mes;
    }
    
    // 7. 保存MES返回的queue ID
    new_q->doorbell_id = queue_input.doorbell_offset;
    
    *q = new_q;
    
    pr_debug("Created MES queue: pasid=%d, queue_type=%d, doorbell=0x%llx\n",
             qpd->pqm->process->pasid,
             properties->type,
             properties->doorbell_off);
    
    return 0;
    
err_add_queue_mes:
    deallocate_mqd(dqm->dev, new_q->mqd, new_q->mqd_mem_obj);
err_allocate_mqd:
    kfree(new_q);
    return retval;
}
```

### 4.3 DQM create_queue() - CPSCH模式 🔴

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_device_queue_manager.c`

> **⚠️ 重要**: CPSCH 模式与 MES 模式有本质差异 - **不直接分配 HQD**！

```c
static int create_queue_cpsch(struct device_queue_manager *dqm,
                              struct queue **q,
                              struct queue_properties *properties,
                              struct qcm_process_device *qpd)
{
    struct queue *new_q;
    int retval;
    
    // 1. 分配queue结构
    new_q = kzalloc(sizeof(*new_q), GFP_KERNEL);
    if (!new_q)
        return -ENOMEM;
    
    // 2. 初始化queue属性
    new_q->properties = *properties;
    new_q->device = dqm->dev;
    new_q->process = properties->process;
    
    // 3. 分配MQD (Memory Queue Descriptor)
    retval = allocate_mqd(dqm->dev, &new_q->mqd, &new_q->mqd_mem_obj);
    if (retval != 0) {
        goto err_allocate_mqd;
    }
    
    // 4. 初始化MQD（仅在内存中）
    dqm->ops.init_mqd(dqm, &new_q->mqd, &new_q->mqd_mem_obj,
                     &new_q->gart_mqd_addr, &new_q->properties);
    
    // ⚠️ 关键差异：CPSCH 不调用 allocate_hqd()！
    // ❌ 不分配 HQD (Pipe, Queue)
    // ❌ 不写入硬件寄存器
    
    // 5. 添加到进程队列列表
    list_add(&new_q->list, &qpd->queues_list);
    qpd->queue_count++;
    
    // 6. 标记为需要更新 runlist
    dqm->sched_running = false;
    
    // 7. 队列状态：未激活
    new_q->properties.is_active = 0;
    
    *q = new_q;
    
    pr_debug("Created CPSCH queue: pasid=%d, queue_id=%u, is_active=0\n",
             qpd->pqm->process->pasid,
             properties->queue_id);
    
    // ⚠️ 注意：队列创建完成，但还未激活
    // HQD 将在后续通过 Runlist 机制由 MEC Firmware 动态分配
    
    return 0;
    
err_allocate_mqd:
    kfree(new_q);
    return retval;
}
```

**CPSCH 的关键特点**:

1. **不调用 `allocate_hqd()`**:
   ```c
   // MES 模式:
   add_hw_queue() → 立即分配硬件资源
   
   // CPSCH 模式:
   create_queue_cpsch() → 只创建 MQD
   // HQD 在后续由 MEC Firmware 动态分配
   ```

2. **队列激活延迟**:
   ```c
   // 创建时: is_active = 0
   // 首次使用时通过 map_queues_cpsch() 激活
   ```

3. **Runlist 机制**:
   ```c
   // 队列通过 Runlist 提交到 MEC Firmware
   map_queues_cpsch() {
       // 构建包含所有队列的 Runlist
       // 通过 PM4 packet 发送到 MEC
       pm_send_set_resources();
   }
   ```

4. **HQD 动态分配**:
   ```c
   // MEC Firmware 接收 Runlist 后
   // 根据调度策略动态分配 HQD
   // 可能每次分配都不同
   // 软件层无法直接控制
   ```

**与 MES 模式的对比**:

| 特性 | MES 模式 | CPSCH 模式 |
|------|---------|-----------|
| HQD 分配 | `add_hw_queue()` 立即分配 | MEC Firmware 动态分配 |
| 分配时机 | Queue 创建时 | Runlist 提交时 |
| 固定映射 | ✅ 固定 | ❌ 动态、可变 |
| 软件控制 | ✅ 完全控制 | ❌ 有限控制（通过 Runlist） |
| 调用函数 | `create_queue_mes()` | `create_queue_cpsch()` |
| 关键函数 | `allocate_hqd()` ✅ 调用 | `allocate_hqd()` ❌ 不调用 |

**性能影响**:
- ✅ **优点**: 灵活调度，更好的资源利用
- ❌ **缺点**: 可能导致队列共享 HQD，串行化
- ⚠️ **瓶颈**: `active_runlist` 全局串行化

**详见**: [8.4 CPSCH Queue 管理机制详解](#84-cpsch-queue-管理机制详解-)

---

## 5️⃣ 关键数据结构

### 5.1 kfd_process - 进程对象

```c
// 表示一个应用进程在KFD驱动中的抽象
struct kfd_process {
    struct kref ref;                    // 引用计数
    struct work_struct release_work;
    
    struct mutex mutex;
    
    // 进程标识
    struct mm_struct *mm;               // 内存管理结构
    struct pid *lead_thread;            // 主线程PID
    uint32_t pasid;                     // Process Address Space ID
    
    // Queue管理
    struct process_queue_manager pqm;   // Process Queue Manager
    
    // 设备列表（多GPU支持）
    struct list_head per_device_data;   // kfd_process_device列表
    
    // 调试和统计
    struct kfd_event_waiter event_waiter;
    bool debug_trap_enabled;
    
    // ... 其他字段
};
```

### 5.2 queue_properties - Queue属性

```c
struct queue_properties {
    enum kfd_queue_type type;           // Queue类型
    enum kfd_queue_format format;       // Queue格式
    
    // Queue内存
    uint64_t queue_address;             // Queue基地址（用户空间）
    uint64_t queue_size;                // Queue大小
    
    uint32_t *read_ptr;                 // 读指针地址
    uint32_t *write_ptr;                // 写指针地址
    
    // Doorbell
    uint32_t doorbell_off;              // Doorbell偏移
    
    // EOP buffer
    uint64_t eop_ring_buffer_address;
    uint32_t eop_ring_buffer_size;
    
    // Context保存恢复
    uint64_t ctx_save_restore_area_address;
    uint32_t ctx_save_restore_area_size;
    uint32_t ctl_stack_size;
    
    // 优先级
    enum kfd_queue_priority priority;
    unsigned int queue_percent;
    
    // 进程信息
    struct kfd_process *process;
    
    // ... 其他字段
};
```

### 5.3 queue - Queue对象

```c
struct queue {
    struct list_head list;              // 链表节点
    void *mqd;                          // Memory Queue Descriptor
    struct kfd_mem_obj *mqd_mem_obj;    // MQD内存对象
    uint64_t gart_mqd_addr;             // MQD的GART地址
    struct queue_properties properties; // Queue属性
    struct kfd_node *device;            // 设备对象
    struct kfd_process *process;        // 所属进程
    
    // Doorbell
    uint32_t doorbell_id;               // Doorbell ID
    
    // Gang调度（MI300等新架构）
    uint64_t gang_ctx_gpu_addr;         // Gang context GPU地址
    void *gang_ctx_cpu_ptr;             // Gang context CPU指针
    
    // ... 其他字段
};
```

---

## 6️⃣ 流程图

### 6.1 MES 模式流程

```
用户空间: HSA Runtime
  │ ioctl(AMDKFD_IOC_CREATE_QUEUE)
  ↓
────────────────────────────────────────────────
内核空间: KFD Driver

kfd_ioctl()  [kfd_chardev.c]
  │
  │ 1. 获取kfd_process
  │ 2. 分发到具体ioctl处理函数
  ↓
kfd_ioctl_create_queue()  [kfd_chardev.c]
  │
  │ 1. copy_from_user(args)
  │ 2. 验证参数
  │ 3. 查找GPU设备
  │ 4. set_queue_properties_from_user()
  ↓
pqm_create_queue()  [kfd_process_queue_manager.c]
  │
  │ 1. 获取process_device_data
  │ 2. 分配process_queue_node
  │ 3. 根据queue类型分发
  ↓
create_cp_queue()  [kfd_process_queue_manager.c]
  │
  │ 1. 调用DQM->ops.create_queue
  │ 2. 获取doorbell_offset
  ↓
DQM: create_queue_mes()  [kfd_device_queue_manager_v12.c]
  │
  │ 1. 分配queue结构
  │ 2. 分配和初始化MQD
  │ 3. 准备mes_add_queue_input
  │ 4. 调用MES add_hw_queue()  ← 立即分配 HQD
  ↓
────────────────────────────────────────────────
转到AMDGPU Driver的MES接口
（见下一章）
  ↓
MES硬件调度器注册Queue
  ↓
────────────────────────────────────────────────
返回路径：

返回queue_id和doorbell_offset
  ↓
copy_to_user(args)
  ↓
返回用户空间
```

### 6.2 CPSCH 模式流程 🔴

```
用户空间: HSA Runtime
  │ ioctl(AMDKFD_IOC_CREATE_QUEUE)
  ↓
────────────────────────────────────────────────
内核空间: KFD Driver

kfd_ioctl()  [kfd_chardev.c]
  │
  │ 1. 获取kfd_process
  │ 2. 分发到具体ioctl处理函数
  ↓
kfd_ioctl_create_queue()  [kfd_chardev.c]
  │
  │ 1. copy_from_user(args)
  │ 2. 验证参数
  │ 3. 查找GPU设备
  │ 4. set_queue_properties_from_user()
  ↓
pqm_create_queue()  [kfd_process_queue_manager.c]
  │
  │ 1. 获取process_device_data
  │ 2. 分配process_queue_node
  │ 3. 根据queue类型分发
  ↓
create_cp_queue()  [kfd_process_queue_manager.c]
  │
  │ 1. 调用DQM->ops.create_queue
  │ 2. 获取doorbell_offset
  ↓
DQM: create_queue_cpsch()  [kfd_device_queue_manager.c]
  │
  │ 1. 分配queue结构
  │ 2. 分配和初始化MQD（仅在内存）
  │ 3. 添加到进程队列列表
  │ 4. is_active = 0（未激活）
  │ ⚠️ 不调用 allocate_hqd()！
  │ ⚠️ 不分配 HQD！
  ↓
────────────────────────────────────────────────
返回路径（Queue 创建完成，但未激活）：

返回queue_id和doorbell_offset
  ↓
copy_to_user(args)
  ↓
返回用户空间
  ↓
────────────────────────────────────────────────
用户提交 Kernel（首次使用 Queue）：
  ↓
写入 Doorbell
  ↓
触发 KFD 的 Queue 激活流程
  ↓
────────────────────────────────────────────────
内核空间: KFD Driver

map_queues_cpsch()  [kfd_device_queue_manager.c]
  │
  │ 1. 检查 active_runlist（⚠️ 串行化点）
  │ 2. 构建 Runlist（包含所有队列）
  │ 3. pm_send_set_resources()
  ↓
PM4 Packet 通过 HIQ → MEC Firmware
  ↓
────────────────────────────────────────────────
MEC Firmware:

接收 Runlist
  ↓
根据调度策略动态分配 HQD  ⚠️ 动态、不固定
  ↓
为每个队列分配 (Pipe, Queue)
  ↓
加载 MQD 到 HQD 寄存器
  ↓
Queue 激活，可以执行
```

**关键差异**:
- **MES**: Queue 创建时立即分配 HQD，固定映射
- **CPSCH**: Queue 创建时不分配 HQD，通过 Runlist 动态分配

---

## 7️⃣ 关键代码位置总结

| 功能 | 文件路径 | 关键函数 |
|------|---------|---------|
| 设备文件打开 | `amdkfd/kfd_chardev.c` | `kfd_open()` |
| ioctl入口 | `amdkfd/kfd_chardev.c` | `kfd_ioctl()` |
| CREATE_QUEUE处理 | `amdkfd/kfd_chardev.c` | `kfd_ioctl_create_queue()` |
| Queue属性设置 | `amdkfd/kfd_chardev.c` | `set_queue_properties_from_user()` |
| PQM创建queue | `amdkfd/kfd_process_queue_manager.c` | `pqm_create_queue()` |
| Compute queue创建 | `amdkfd/kfd_process_queue_manager.c` | `create_cp_queue()` |
| DQM初始化 | `amdkfd/kfd_device_queue_manager.c` | `device_queue_manager_init()` |
| **MES模式创建queue** | `amdkfd/kfd_device_queue_manager_v12.c` | `create_queue_mes()` |
| **CPSCH模式创建queue** 🔴 | `amdkfd/kfd_device_queue_manager.c` | `create_queue_cpsch()` |
| **CPSCH Runlist映射** 🔴 | `amdkfd/kfd_device_queue_manager.c` | `map_queues_cpsch()` |
| **CPSCH PM4提交** 🔴 | `amdkfd/kfd_packet_manager.c` | `pm_send_set_resources()` |
| **HQD分配（仅NOCPSCH）** | `amdkfd/kfd_device_queue_manager.c` | `allocate_hqd()` |
| MQD分配 | `amdkfd/kfd_mqd_manager.c` | `allocate_mqd()` |

**注意**: 
- 🔴 标记的是 CPSCH 模式特有的关键函数
- **CPSCH 模式不调用 `allocate_hqd()`**，HQD 由 MEC Firmware 动态分配

---

## 8️⃣ 关键发现

### 8.1 进程、设备、Queue的关系

```
kfd_process (进程)
    ↓ 1:N
kfd_process_device (进程在每个GPU上的抽象)
    ↓ 1:N
process_queue_node (Queue节点)
    ↓ 1:1
queue (实际的Queue对象)
```

### 8.2 MES vs CPSCH

在 `create_queue` 时，根据 `enable_mes` 标志选择：

```c
// 在 device_queue_manager.c 中
if (!dqm->dev->kfd->shared_resources.enable_mes) {
    // 旧架构：CPSCH模式
    retval = create_queue_cpsch(dqm, q, properties, qpd);
} else {
    // 新架构：MES模式
    retval = create_queue_mes(dqm, q, properties, qpd);
}
```

#### MES 模式（硬件调度器）

**支持的 GPU**:
- ✅ **CDNA3**: MI300A/X (IP_VERSION 12.0.x)
- ✅ **CDNA2**: MI250X, MI210 (IP_VERSION 9.4.1)
- ✅ **RDNA3**: RX 7900 XT/XTX (IP_VERSION 11.0.x)

**特点**:
- ✅ Queue创建时调用MES的 `add_hw_queue()`
- ✅ Kernel提交通过doorbell，**不经过KFD驱动层**
- ✅ 更低的延迟（用户空间直接写 doorbell）
- ✅ 硬件调度，无需驱动干预

**与 CPSCH 的关键差异**:
- ✅ **MES**: 创建时立即分配硬件资源
- ✅ **MES**: 不使用 Runlist 机制
- ✅ **MES**: 软件层对 Queue 有更多控制
- ✅ **MES**: 没有 `active_runlist` 串行化问题

#### CPSCH 模式（软件调度器）

**使用的 GPU**:
- ⚠️ **CDNA2**: **MI308X (Aqua Vanjaram)** (IP_VERSION 9.4.2/3) ← 特殊情况
- ⚠️ **CDNA1**: MI100 (IP_VERSION 9.4.0)
- ⚠️ **Vega 20**: MI50, MI60 (IP_VERSION 9.0.x)
- ⚠️ **RDNA2**: RX 6000 系列 (IP_VERSION 10.3.x)

**特点**:
- ✅ Queue创建时通过CPSCH管理器
- ✅ 使用 Runlist 机制管理队列
- ✅ 软件调度，驱动参与调度过程
- ⚠️ 相对较高的延迟（需要驱动参与）

> **⚠️ 重要提示**：MI308X 虽然命名类似 MI300 系列，但实际使用 **ALDEBARAN 架构**，**不支持 MES**，使用 CPSCH 调度器。这是基于实际硬件验证的结果。

> **🔴 关键发现（2026-01-20）**: CPSCH 模式下，**Queue ID 不直接映射到固定的 HQD (Hardware Queue Descriptor)**！
> 
> - **Queue 创建**: `create_queue_cpsch()` 只在内存中创建 MQD，**不调用** `allocate_hqd()`
> - **Runlist 机制**: 队列通过 Runlist 提交到 MEC Firmware
> - **动态分配**: HQD (Pipe, Queue) 由 **MEC Firmware 动态分配**，可能每次都不同
> - **软件层控制有限**: KFD 驱动层无法直接控制 HQD 分配
> 
> 详见：[multiple_doorbellQueue/DIRECTION1_ANALYSIS.md](./multiple_doorbellQueue/DIRECTION1_ANALYSIS.md)

#### 如何检查您的系统使用哪种模式

```bash
# 检查 enable_mes 参数
cat /sys/module/amdgpu/parameters/mes
# 输出: 1 = MES 模式, 0 = CPSCH 模式

# 查看 dmesg 日志
dmesg | grep -i "mes\|cpsch"
```

### 8.3 MQD (Memory Queue Descriptor)

MQD是Queue的硬件描述符，包含：
- Queue的物理地址
- Queue的大小和类型
- Doorbell信息
- Context保存恢复信息

**MQD的作用**:
- ✅ 硬件通过MQD了解Queue的所有信息
- ✅ MQD在GPU内存中，硬件可以直接访问
- ✅ 不同架构的MQD格式不同（v9, v10, v11, v12等）

### 8.4 CPSCH Queue 管理机制详解 🔴

> **重要更新（2026-01-20）**: 基于实际验证，CPSCH 模式的 Queue 管理机制与 MES 模式有本质差异。

#### 8.4.1 CPSCH vs NOCPSCH 的架构差异

**NOCPSCH (直接模式)**:
```c
create_queue_nocpsch()
    ↓
allocate_hqd()  // ✅ 直接分配 (Pipe, Queue)
    ↓ 固定映射
HQD (Pipe 0, Queue 0)  // 软件层完全控制
    ↓
load_mqd_to_hqd()  // 直接写入硬件寄存器
    ↓
HQD 立即可用，固定不变
```

**CPSCH (调度器模式)**:
```c
create_queue_cpsch()  // ❌ 不调用 allocate_hqd()
    ↓
创建 MQD（仅在内存）
    ↓
add to process queue list
    ↓
map_queues_cpsch()  // 构建 Runlist
    ↓
pm_send_set_resources()  // PM4 packet → HIQ → MEC
    ↓
MEC Firmware 接收 Runlist
    ↓
MEC Firmware 动态分配 HQD  // ⚠️ 在固件层完成，软件层不可见
    ↓
HQD (Pipe, Queue) - 动态、可变、软件层无法直接控制
```

#### 8.4.2 Queue ID vs HQD 的关系

**NOCPSCH 模式（固定映射）**:
```
Queue ID 0 → HQD (Pipe 0, Queue 0) - 固定
Queue ID 1 → HQD (Pipe 1, Queue 0) - 固定
Queue ID 2 → HQD (Pipe 2, Queue 0) - 固定
...

关系: 1:1 固定映射
软件控制: ✅ 完全控制
```

**CPSCH 模式（动态映射）**:
```
Queue ID 216 → Runlist Entry #0
Queue ID 217 → Runlist Entry #1
Queue ID 220 → Runlist Entry #2
    ↓
MEC Firmware 根据调度策略动态分配
    ↓
可能的结果（每次可能不同）:
  Entry #0 → HQD (Pipe 0, Queue 0)
  Entry #1 → HQD (Pipe 1, Queue 0)
  Entry #2 → HQD (Pipe 0, Queue 1)
或
  Entry #0 → HQD (Pipe 2, Queue 0)
  Entry #1 → HQD (Pipe 0, Queue 0)  ← 可能复用！
  Entry #2 → HQD (Pipe 1, Queue 0)

关系: N:M 动态映射，软件层不可预测
软件控制: ❌ 有限控制（仅通过 Runlist）
```

#### 8.4.3 Runlist 机制

**什么是 Runlist**:
- Runlist 是 CPSCH 模式下管理队列的核心机制
- 它是一个队列列表，包含所有需要调度的队列信息
- 通过 PM4 packet 发送到 MEC Firmware

**Runlist 的内容**:
```c
struct runlist_entry {
    uint32_t queue_id;          // 软件 Queue ID
    uint64_t mqd_addr;          // MQD 内存地址
    uint32_t queue_type;        // 队列类型（COMPUTE, SDMA等）
    uint32_t priority;          // 优先级
    // ⚠️ 不包含 Pipe/Queue 信息（由 MEC 决定）
};
```

**Runlist 的生命周期**:
```
1. 创建队列: create_queue_cpsch()
   - 队列加入进程队列列表
   - is_active = 0 (未激活)

2. 构建 Runlist: map_queues_cpsch()
   - 遍历所有进程的队列
   - 构建 Runlist
   - 发送 PM4 packet 到 MEC

3. MEC 处理:
   - 接收 Runlist
   - 动态分配 HQD
   - 加载 MQD 到 HQD

4. 队列激活: update_queue_locked()
   - is_active = 1
   - load_mqd() 被调用
```

#### 8.4.4 为什么 CPSCH 不使用固定 HQD？

**设计原因**:
1. **灵活调度**: MEC Firmware 可以根据负载动态调整
2. **资源优化**: 同一个 HQD 可以在不同时间服务不同队列
3. **优先级管理**: 高优先级队列可以抢占 HQD
4. **多进程支持**: 更好地支持多进程并发

**代价**:
1. **软件控制有限**: KFD 驱动层无法精确控制 HQD 分配
2. **调试困难**: HQD 分配是动态的，难以追踪
3. **性能不可预测**: 队列可能共享 HQD，导致串行化
4. **优化受限**: 基于 Queue ID 的优化可能无效

#### 8.4.5 CPSCH 的性能瓶颈

**已知瓶颈**:
1. 🔴🔴🔴 **`active_runlist` 串行化**:
   ```c
   // kfd_device_queue_manager.c
   static int map_queues_cpsch(struct device_queue_manager *dqm) {
       if (dqm->active_runlist)  // ⚠️ 全局串行化
           return 0;
       // ...
   }
   ```
   - 同一时间只能有一个 active runlist
   - 多进程时，runlist 更新被串行化
   - **影响**: -30~40 QPS

2. 🔴🔴 **HIQ (High-priority Interface Queue) 瓶颈**:
   - 所有 PM4 packet 通过单一的 HIQ 发送
   - HIQ 是串行通道
   - **影响**: -15~20 QPS

3. 🔴 **MEC Firmware 调度策略**:
   - 固件层的调度算法是黑盒
   - 可能有内部串行化
   - 可能基于属性复用 HQD
   - **影响**: 未知

#### 8.4.6 优化方向

**软件层优化（可行）**:
```c
// v6: 移除或放宽 active_runlist 限制
static int map_queues_cpsch_v6(...) {
    // 不检查 active_runlist
    // 使用队列机制管理多个 runlist
    add_to_runlist_queue(dqm, ...);
    schedule_work(&dqm->runlist_worker);
    return 0;
}

// v7: 进程级 Runlist
static int map_queues_per_process(...) {
    if (process->active_runlist)
        return 0;  // 进程级检查
    // 只更新该进程的队列
    // ...
}
```

**固件层优化（困难）**:
- MEC Firmware 是闭源的
- 调度策略由 AMD 控制
- 需要 AMD 官方支持

**架构级优化（最佳）**:
- 迁移到 MES 模式（如果硬件支持）
- MES 模式使用硬件调度，没有 Runlist 串行化问题
- 但 MI308X 不支持 MES

#### 8.4.7 实验验证

**验证方法**:
```bash
# 1. 在 map_queues_cpsch() 中添加日志
# 查看 runlist 构建和提交过程

# 2. 在 pm_send_set_resources() 中添加日志
# 查看 PM4 packet 的内容和数量

# 3. 对比 NOCPSCH 模式（如果可能）
# 强制使用 create_queue_nocpsch()
# 验证 allocate_hqd() 的行为
```

**参考文档**:
- [DIRECTION1_ANALYSIS.md](./multiple_doorbellQueue/DIRECTION1_ANALYSIS.md) - 方向1验证结果分析
- [FUTURE_RESEARCH_DIRECTIONS.md](./multiple_doorbellQueue/FUTURE_RESEARCH_DIRECTIONS.md) - 后续研究方向

---

## 9️⃣ 下一步

### 针对 MES 模式

在下一层（MES调度器和硬件层），我们将看到：
- AMDGPU Driver如何实现MES接口
- MES如何通过MES Ring提交ADD_QUEUE命令
- MES硬件调度器如何工作

继续阅读: [KERNEL_TRACE_04_MES_HARDWARE.md](./KERNEL_TRACE_04_MES_HARDWARE.md)

### 针对 CPSCH 模式 🔴

对于 MI308X 等使用 CPSCH 模式的 GPU：
- CPSCH 不使用 MES 硬件调度器
- Queue 管理通过 Runlist 和 MEC Firmware 完成
- HQD 分配是动态的，由固件层控制

**相关文档**:
- [KERNEL_TRACE_CPSCH_MECHANISM.md](./KERNEL_TRACE_CPSCH_MECHANISM.md) - CPSCH 调度器机制详解
- [multiple_doorbellQueue/DIRECTION1_ANALYSIS.md](./multiple_doorbellQueue/DIRECTION1_ANALYSIS.md) - CPSCH Queue 管理验证结果
- [multiple_doorbellQueue/EXECUTIVE_SUMMARY.md](./multiple_doorbellQueue/EXECUTIVE_SUMMARY.md) - 多进程 Queue 优化实验总结

**重要发现**:
- ⚠️ CPSCH 模式下，Queue ID 不直接映射到固定的 HQD
- ⚠️ HQD 由 MEC Firmware 动态分配，软件层无法直接控制
- ⚠️ 基于 Queue ID 的优化在 CPSCH 模式下可能无效
- ⚠️ 主要瓶颈在 Runlist 管理层（`active_runlist` 串行化）


