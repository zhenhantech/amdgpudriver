# Kernel提交流程追踪 (5/5) - 关键数据结构详解

**范围**: 完整的数据结构定义和说明  
**代码路径**: 跨越多个模块  
**目的**: 作为数据结构参考手册

---

## 📋 文档说明

本文档详细列出kernel提交流程中涉及的所有关键数据结构，包括：
1. AQL Packet格式
2. Queue相关结构
3. Process和Device结构
4. MES相关结构
5. Context和Entity结构

---

## 1️⃣ AQL (Architected Queuing Language) 数据结构

### 1.1 AQL Dispatch Packet (64字节)

**规范**: HSA 1.2标准  
**用途**: Kernel启动命令

```c
typedef struct hsa_kernel_dispatch_packet_s {
    // [Byte 0-1] Header
    uint16_t header;
    /*
     * Packet类型 (bits 0-7):
     *   0: Invalid
     *   1: Kernel Dispatch  ← 最常用
     *   2: Barrier-AND
     *   3: Barrier-OR
     *   4: Agent Dispatch
     * 
     * Barrier bit (bit 8):
     *   0: 不等待前面的packet
     *   1: 等待前面的packet完成
     * 
     * Acquire fence scope (bits 9-10):
     *   0: No fence
     *   1: Agent scope
     *   2: System scope  ← 常用
     * 
     * Release fence scope (bits 11-12):
     *   0: No fence
     *   1: Agent scope
     *   2: System scope  ← 常用
     */
    
    // [Byte 2-3] Setup
    uint16_t setup;
    /*
     * 低16位: workgroup_size_x维度 (bits 0-15)
     * 高16位: workgroup_size_y维度 (需要左移16位)
     */
    
    // [Byte 4-9] Workgroup size
    uint16_t workgroup_size_x;      // Workgroup X维度
    uint16_t workgroup_size_y;      // Workgroup Y维度
    uint16_t workgroup_size_z;      // Workgroup Z维度
    
    // [Byte 10-11] Reserved
    uint16_t reserved0;
    
    // [Byte 12-23] Grid size (全局大小)
    uint32_t grid_size_x;           // Grid X维度 = blocks_x * workgroup_size_x
    uint32_t grid_size_y;           // Grid Y维度
    uint32_t grid_size_z;           // Grid Z维度
    
    // [Byte 24-27] Private segment size
    uint32_t private_segment_size;  // 每个work-item的私有内存大小
    
    // [Byte 28-31] Group segment size
    uint32_t group_segment_size;    // 每个workgroup的共享内存大小（LDS）
    
    // [Byte 32-39] Kernel object address
    uint64_t kernel_object;         // Kernel代码的GPU地址
    
    // [Byte 40-47] Kernarg address
    uint64_t kernarg_address;       // Kernel参数buffer的GPU地址
    
    // [Byte 48-55] Reserved
    uint64_t reserved1;
    
    // [Byte 56-63] Completion signal
    hsa_signal_t completion_signal; // 完成信号（用于同步）
    
} hsa_kernel_dispatch_packet_t;
```

**Header字段详解**:
```c
// Header的常见值：0x1402
#define HSA_PACKET_TYPE_KERNEL_DISPATCH 1

uint16_t header = 
    (HSA_PACKET_TYPE_KERNEL_DISPATCH << 0) |   // bits 0-7: type=1
    (1 << 8) |                                  // bit 8: barrier=1
    (HSA_FENCE_SCOPE_SYSTEM << 9) |            // bits 9-10: acquire=2
    (HSA_FENCE_SCOPE_SYSTEM << 11);            // bits 11-12: release=2

// 结果: 0x1402
// 二进制: 0001 0100 0000 0010
//         ^^^^ ^^   ^^   ^^^^
//         |    |    |    type (1)
//         |    |    acquire (2)
//         |    release (2)
//         barrier (1)
```

**示例: 启动一个256x1x1的kernel, 每个block 64个thread**:
```c
hsa_kernel_dispatch_packet_t packet = {
    .header = 0x1402,                  // Type=1, Barrier, System fence
    .setup = 64,                       // workgroup_size_x = 64
    .workgroup_size_x = 64,
    .workgroup_size_y = 1,
    .workgroup_size_z = 1,
    .grid_size_x = 16384,              // 256 blocks * 64 threads = 16384
    .grid_size_y = 1,
    .grid_size_z = 1,
    .private_segment_size = 0,         // 无私有内存
    .group_segment_size = 4096,        // 4KB LDS (shared memory)
    .kernel_object = 0x7f8000040000,   // Kernel代码地址
    .kernarg_address = 0x7f8000050000, // Kernel参数地址
    .completion_signal = {.handle = 0x7f8000060000},
};
```

### 1.2 AQL Queue结构

**文件**: `ROCm_keyDriver/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/inc/hsa.h`

```c
typedef struct hsa_queue_s {
    // 标准HSA字段
    hsa_queue_type32_t type;           // Queue类型
    uint32_t features;                  // 特性标志
    
    // Doorbell signal
    hsa_signal_t doorbell_signal;       // Doorbell signal handle
    
    // Queue大小
    uint32_t size;                      // Queue中packet的数量（2的幂）
    uint32_t reserved1;
    
    // Queue ID
    uint64_t id;                        // Queue唯一标识
    
} hsa_queue_t;
```

**AMD扩展的Queue结构**:
```c
typedef struct amd_queue_s {
    // 继承HSA标准字段
    hsa_queue_t hsa_queue;
    
    // 读写指针（在用户空间）
    volatile uint64_t write_dispatch_id;   // 写指针（软件更新）
    volatile uint64_t read_dispatch_id;    // 读指针（硬件更新）
    
    // Queue内存
    uint64_t base_address;              // Packet数组基地址
    
    // 扩展属性
    volatile uint32_t* queue_properties;
    uint64_t reserved[2];
    
} amd_queue_t;
```

### 1.3 AQL Signal

```c
typedef struct hsa_signal_s {
    uint64_t handle;                    // Signal的内存地址
} hsa_signal_t;

// Signal值类型
typedef int64_t hsa_signal_value_t;

// Signal操作
hsa_signal_value_t hsa_signal_load_relaxed(hsa_signal_t signal);
void hsa_signal_store_relaxed(hsa_signal_t signal, hsa_signal_value_t value);
hsa_signal_value_t hsa_signal_wait_acquire(hsa_signal_t signal,
                                            hsa_signal_condition_t condition,
                                            hsa_signal_value_t compare_value,
                                            uint64_t timeout_hint,
                                            hsa_wait_state_t wait_state_hint);
```

---

## 2️⃣ KFD (Kernel Fusion Driver) 数据结构

### 2.1 kfd_ioctl_create_queue_args

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/include/uapi/linux/kfd_ioctl.h`

```c
// 用户空间传递给KFD的queue创建参数
struct kfd_ioctl_create_queue_args {
    // Queue内存地址
    uint64_t ring_base_address;        // Queue基地址（用户空间分配）
    
    // 读写指针地址
    uint64_t write_pointer_address;    // 写指针地址
    uint64_t read_pointer_address;     // 读指针地址
    
    // Doorbell信息
    uint64_t doorbell_offset;          // OUT: KFD返回的doorbell偏移
    
    // Queue大小和类型
    uint32_t ring_size;                // Queue大小（字节）
    uint32_t gpu_id;                   // GPU ID
    uint32_t queue_type;               // Queue类型（见下面的枚举）
    uint32_t queue_percentage;         // Queue优先级百分比（0-100）
    uint32_t queue_priority;           // 优先级级别
    
    // EOP (End Of Pipe) buffer
    uint64_t eop_buffer_address;       // EOP buffer地址
    uint64_t eop_buffer_size;          // EOP buffer大小
    
    // Context保存恢复
    uint64_t ctx_save_restore_address; // Context保存恢复区域地址
    uint32_t ctx_save_restore_size;    // Context保存恢复区域大小
    uint32_t ctl_stack_size;           // 控制栈大小
    
    // Queue ID
    uint32_t queue_id;                 // OUT: KFD返回的queue ID
    
    // CU masking
    uint32_t num_cu_mask;              // CU mask数量
    uint64_t cu_mask_ptr;              // CU mask指针
};
```

**Queue类型枚举**:
```c
enum kfd_queue_type {
    KFD_IOC_QUEUE_TYPE_COMPUTE = 0,         // Compute queue (旧式)
    KFD_IOC_QUEUE_TYPE_SDMA,                // SDMA queue (内存拷贝)
    KFD_IOC_QUEUE_TYPE_COMPUTE_AQL,         // Compute AQL queue ← 常用
    KFD_IOC_QUEUE_TYPE_SDMA_XGMI,           // SDMA XGMI queue
};
```

### 2.2 kfd_process - 进程对象

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_priv.h`

```c
struct kfd_process {
    // 引用计数和生命周期
    struct kref ref;
    struct work_struct release_work;
    
    // 进程标识
    struct mm_struct *mm;              // Linux内存管理结构
    struct pid *lead_thread;           // 主线程PID
    uint32_t pasid;                    // Process Address Space ID
    
    // 同步
    struct mutex mutex;
    
    // Queue管理
    struct process_queue_manager pqm;  // Process Queue Manager
    
    // 设备列表（多GPU支持）
    struct list_head per_device_data;  // kfd_process_device列表
    size_t n_pdds;                     // 设备数量
    
    // 内存管理
    struct kfd_process_device *pdds[MAX_GPU_INSTANCE];
    
    // 调试和事件
    bool debug_trap_enabled;
    struct kfd_event_waiter event_waiter;
    
    // 统计信息
    bool signal_event_limit_reached;
    
    // ... 其他字段
};
```

### 2.3 queue - KFD内核Queue对象

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_priv.h`

```c
struct queue {
    // 链表节点
    struct list_head list;
    
    // MQD (Memory Queue Descriptor)
    void *mqd;                         // MQD CPU指针
    struct kfd_mem_obj *mqd_mem_obj;   // MQD内存对象
    uint64_t gart_mqd_addr;            // MQD的GART地址
    
    // Queue属性
    struct queue_properties properties;
    
    // 所属对象
    struct kfd_node *device;           // GPU设备
    struct kfd_process *process;       // 所属进程
    
    // Doorbell
    uint32_t doorbell_id;              // Doorbell ID
    
    // Gang调度（新架构）
    uint64_t gang_ctx_gpu_addr;        // Gang context GPU地址
    void *gang_ctx_cpu_ptr;            // Gang context CPU指针
    
    // 其他
    uint64_t tma_addr;                 // TMA地址
    
    // ... 其他字段
};
```

### 2.4 queue_properties - Queue属性

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_priv.h`

```c
struct queue_properties {
    // Queue类型和格式
    enum kfd_queue_type type;          // Queue类型
    enum kfd_queue_format format;      // Queue格式
    
    // Queue内存
    uint64_t queue_address;            // Queue基地址
    uint64_t queue_size;               // Queue大小
    uint32_t queue_id;                 // Queue ID
    
    // 读写指针
    uint32_t *read_ptr;                // 读指针地址
    uint32_t *write_ptr;               // 写指针地址
    
    // Doorbell
    uint32_t doorbell_off;             // Doorbell偏移
    void __iomem *doorbell_ptr;        // Doorbell指针（内核空间）
    
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
    struct kfd_node *dev;
    
    // CU masking
    uint32_t *cu_mask;
    
    // 其他
    bool is_interop;
    bool is_gws;
    bool is_active;
    
    // ... 其他字段
};
```

---

## 3️⃣ MES (Micro-Engine Scheduler) 数据结构

### 3.1 mes_add_queue_input

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/amdgpu_mes.h`

```c
struct mes_add_queue_input {
    // Process信息
    uint32_t process_id;               // Process ID (PASID)
    uint64_t page_table_base_addr;     // 页表基地址
    uint64_t process_va_start;         // 进程虚拟地址起始
    uint64_t process_va_end;           // 进程虚拟地址结束
    uint64_t process_quantum;          // 进程时间片（纳秒）
    uint64_t process_context_addr;     // 进程context地址
    
    // Gang调度信息
    uint64_t gang_context_addr;        // Gang context地址
    uint32_t inprocess_gang_priority;  // Gang内部优先级
    uint32_t gang_global_priority_level; // Gang全局优先级
    
    // Queue信息
    uint32_t queue_type;               // Queue类型
    uint64_t mqd_addr;                 // MQD GPU地址
    uint64_t wptr_addr;                // 写指针地址
    uint64_t rptr_addr;                // 读指针地址
    uint32_t queue_size;               // Queue大小
    uint64_t doorbell_offset;          // Doorbell偏移
    uint64_t page_table_base_va;       // 页表基虚拟地址
    
    // GDS (Global Data Share)
    uint32_t gds_base;
    uint32_t gds_size;
    uint32_t gws_base;
    uint32_t gws_size;
    uint32_t oa_mask;
    
    // Trap handler
    uint64_t tba_addr;                 // Trap Base Address
    uint64_t tma_addr;                 // Trap Memory Address
    
    // 标志
    bool is_kfd_process;
    bool is_aql_queue;
    bool skip_process_ctx_clear;
    bool is_tmz_queue;
};
```

### 3.2 MESAPI__ADD_QUEUE Packet

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/amdgpu_mes.h`

```c
// MES API packet结构（提交给MES硬件）
union MESAPI__ADD_QUEUE {
    struct {
        // Header
        union MES_API_HEADER header;   // 4 DWords
        
        // Process信息（与mes_add_queue_input对应）
        uint32_t process_id;
        uint64_t page_table_base_addr;
        uint64_t process_va_start;
        uint64_t process_va_end;
        uint64_t process_quantum;
        uint64_t process_context_addr;
        
        // Gang信息
        uint64_t gang_context_addr;
        uint32_t inprocess_gang_priority;
        uint32_t gang_global_priority_level;
        
        // Queue信息
        uint32_t queue_type;
        uint64_t mqd_addr;
        uint64_t wptr_addr;
        uint32_t queue_size;
        uint64_t doorbell_offset;
        
        // GDS
        uint32_t gds_base;
        uint32_t gds_size;
        uint32_t gws_base;
        uint32_t gws_size;
        uint32_t oa_mask;
        
        // Trap
        uint64_t trap_handler_addr;
        uint64_t tma_addr;
        
        // 标志
        uint32_t is_kfd_process;
        uint32_t is_aql_queue;
        uint32_t is_tmz_queue;
        
        // Reserved
        uint32_t reserved[10];
        
        // API status（MES填充返回）
        struct MES_API_STATUS api_status;
    };
    
    // 确保packet大小
    uint32_t max_dwords[API_FRAME_SIZE_IN_DWORDS];
};
```

### 3.3 MES Queue类型

```c
enum mes_queue_type {
    MES_QUEUE_TYPE_GFX,
    MES_QUEUE_TYPE_COMPUTE,
    MES_QUEUE_TYPE_COMPUTE_AQL,        // Compute AQL
    MES_QUEUE_TYPE_SDMA,
    MES_QUEUE_TYPE_SDMA_XGMI,
};
```

---

## 4️⃣ AMDGPU Driver数据结构

### 4.1 amdgpu_ring - Ring结构

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/amdgpu_ring.h`

```c
struct amdgpu_ring {
    struct amdgpu_device *adev;        // 设备对象
    
    // Ring类型
    enum amdgpu_ring_type type;        // Ring类型
    char name[16];                     // Ring名称
    
    // Ring内存
    struct amdgpu_bo *ring_obj;        // Ring buffer对象
    volatile uint32_t *ring;           // Ring buffer CPU地址
    uint64_t gpu_addr;                 // Ring buffer GPU地址
    uint32_t *ring_ptr_mask;           // Ring指针掩码
    
    // Ring指针
    uint32_t wptr;                     // 写指针
    uint32_t wptr_old;                 // 旧的写指针
    unsigned wptr_offs;                // 写指针偏移
    
    // Ring大小
    u64 ring_size;                     // Ring大小（字节）
    u32 buf_mask;                      // Buffer掩码
    
    // Doorbell
    bool use_doorbell;                 // 是否使用doorbell
    unsigned doorbell_index;           // Doorbell索引
    
    // 函数指针
    const struct amdgpu_ring_funcs *funcs;
    
    // 调度器
    struct drm_gpu_scheduler sched;    // GPU调度器
    
    // 其他
    bool ready;
    atomic_t fence_drv_seq;
    
    // ... 其他字段
};
```

### 4.2 amdgpu_ring_funcs - Ring函数指针

```c
struct amdgpu_ring_funcs {
    enum amdgpu_ring_type type;        // Ring类型
    uint32_t align_mask;
    u32 nop;                           // NOP命令
    
    // 指针操作
    uint64_t (*get_rptr)(struct amdgpu_ring *ring);
    uint64_t (*get_wptr)(struct amdgpu_ring *ring);
    void (*set_wptr)(struct amdgpu_ring *ring);
    
    // Packet发送
    void (*emit_ib)(struct amdgpu_ring *ring,
                   struct amdgpu_ib *ib,
                   unsigned vmid,
                   bool ctx_switch);
    void (*emit_fence)(struct amdgpu_ring *ring,
                      uint64_t addr,
                      uint64_t seq,
                      unsigned flags);
    
    // 测试
    int (*test_ring)(struct amdgpu_ring *ring);
    int (*test_ib)(struct amdgpu_ring *ring, long timeout);
    
    // 其他
    void (*insert_nop)(struct amdgpu_ring *ring, uint32_t count);
    void (*emit_wreg)(struct amdgpu_ring *ring, uint32_t reg, uint32_t val);
    
    // ... 其他函数指针
};
```

### 4.3 Ring类型枚举

```c
enum amdgpu_ring_type {
    AMDGPU_RING_TYPE_GFX = 0,         // Graphics ring
    AMDGPU_RING_TYPE_COMPUTE,         // Compute ring
    AMDGPU_RING_TYPE_SDMA,            // SDMA ring
    AMDGPU_RING_TYPE_UVD,             // Video decode
    AMDGPU_RING_TYPE_VCE,             // Video encode
    AMDGPU_RING_TYPE_KIQ,             // Kernel interface queue
    AMDGPU_RING_TYPE_MES,             // MES ring ← MES管理命令
    AMDGPU_RING_TYPE_VCN_DEC,
    AMDGPU_RING_TYPE_VCN_ENC,
    AMDGPU_RING_TYPE_VCN_JPEG,
    // ... 其他类型
};
```

---

## 5️⃣ Context和Entity数据结构

### 5.1 amdgpu_ctx - Context结构

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/amdgpu_ctx.h`

```c
struct amdgpu_ctx {
    struct kref refcount;
    struct amdgpu_device *adev;
    struct amdgpu_ctx_mgr *mgr;
    unsigned reset_counter;
    unsigned reset_counter_query;
    uint32_t vram_lost_counter;
    spinlock_t ring_lock;
    
    // Entity数组
    // [hw_ip][ring] = entity
    struct amdgpu_ctx_entity **entities;
    
    // 优先级
    int32_t init_priority;
    int32_t override_priority;
    
    // 其他
    atomic_t guilty;
    unsigned long ras_counter_ce;
    unsigned long ras_counter_ue;
    uint32_t stable_pstate;
};
```

### 5.2 amdgpu_ctx_entity - Entity结构

```c
struct amdgpu_ctx_entity {
    // 调度entity
    struct drm_sched_entity entity;    // DRM调度器entity
    
    // 序列号
    uint64_t sequence;
    
    // Fence管理
    struct dma_fence **fences;
    struct drm_sched_entity *entity_ptr;
    
    // hw_ip类型（保存用于调试）
    uint32_t hw_ip;
};
```

### 5.3 Entity数量配置

```c
// 每个Context可以有的Entity数量
const unsigned int amdgpu_ctx_num_entities[AMDGPU_HW_IP_NUM] = {
    [AMDGPU_HW_IP_GFX]     = 1,        // Graphics: 1个entity
    [AMDGPU_HW_IP_COMPUTE] = 4,        // Compute: 4个entity
    [AMDGPU_HW_IP_DMA]     = 2,        // SDMA: 2个entity
    [AMDGPU_HW_IP_UVD]     = 1,
    [AMDGPU_HW_IP_VCE]     = 1,
    [AMDGPU_HW_IP_UVD_ENC] = 1,
    [AMDGPU_HW_IP_VCN_DEC] = 1,
    [AMDGPU_HW_IP_VCN_ENC] = 1,
    [AMDGPU_HW_IP_VCN_JPEG] = 1,
};
```

### 5.4 drm_sched_entity - DRM调度Entity

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/scheduler/gpu_scheduler.h`

```c
struct drm_sched_entity {
    // 关联的调度器列表
    struct drm_gpu_scheduler **sched_list;
    unsigned int num_sched_list;
    
    // 当前使用的调度器
    struct drm_sched_rq *rq;           // Run queue
    
    // Job队列
    struct spsc_queue job_queue;
    atomic_t fence_seq;
    uint64_t fence_context;
    
    // Guilty标志
    atomic_t *guilty;
    
    // 优先级
    enum drm_sched_priority priority;
    
    // 其他
    struct dma_fence_cb cb;
};
```

---

## 6️⃣ MQD (Memory Queue Descriptor)

### 6.1 MQD结构（以v12为例）

**文件**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_mqd_manager_v12.c`

```c
struct v12_compute_mqd {
    // Queue控制
    uint32_t compute_pipelinestat_enable;
    uint32_t compute_static_thread_mgmt_se0;
    uint32_t compute_static_thread_mgmt_se1;
    uint32_t compute_static_thread_mgmt_se2;
    uint32_t compute_static_thread_mgmt_se3;
    
    // Queue地址和大小
    uint32_t cp_hqd_pq_base_lo;        // Queue基地址低32位
    uint32_t cp_hqd_pq_base_hi;        // Queue基地址高32位
    uint32_t cp_hqd_pq_rptr;           // 读指针
    uint32_t cp_hqd_pq_wptr_lo;        // 写指针低32位
    uint32_t cp_hqd_pq_wptr_hi;        // 写指针高32位
    uint32_t cp_hqd_pq_control;        // Queue控制
    
    // Doorbell
    uint32_t cp_hqd_pq_doorbell_control; // Doorbell控制
    uint32_t cp_hqd_eop_base_addr_lo;  // EOP基地址低
    uint32_t cp_hqd_eop_base_addr_hi;  // EOP基地址高
    uint32_t cp_hqd_eop_control;       // EOP控制
    
    // VM (Virtual Memory)
    uint32_t cp_hqd_vmid;              // VMID
    
    // Active状态
    uint32_t cp_hqd_active;            // Queue是否active
    
    // Queue优先级
    uint32_t cp_hqd_queue_priority;
    uint32_t cp_hqd_quantum;           // 时间片
    
    // ... 其他字段（硬件寄存器映射）
};
```

---

## 7️⃣ 总结：数据流

```
用户空间:
  hsa_kernel_dispatch_packet_t (64字节)
    ↓
  写入 amd_queue_t.base_address
    ↓
  更新 amd_queue_t.write_dispatch_id
    ↓
  写入 amd_queue_t.doorbell_signal (doorbell)

────────────────────────────────────

内核空间 (Queue创建时):
  kfd_ioctl_create_queue_args
    ↓
  queue_properties
    ↓
  queue
    ↓
  mes_add_queue_input
    ↓
  MESAPI__ADD_QUEUE (MES packet)
    ↓
  通过 amdgpu_ring (MES Ring) 提交
    ↓
  MES硬件调度器注册queue

────────────────────────────────────

硬件层 (Kernel执行时):
  检测doorbell更新
    ↓
  读取 v12_compute_mqd (MQD)
    ↓
  获取queue信息
    ↓
  从queue读取 hsa_kernel_dispatch_packet_t
    ↓
  解析packet，调度执行
```

---

## 8️⃣ 关键大小和对齐

| 结构 | 大小 | 对齐要求 | 说明 |
|------|------|---------|------|
| hsa_kernel_dispatch_packet_t | 64字节 | 64字节 | AQL packet固定大小 |
| amd_queue_t | 变长 | 页对齐 | Queue结构 |
| Queue buffer | N*64字节 | 页对齐 | N是queue大小（2的幂） |
| MQD | ~256字节 | 256字节 | 取决于GPU架构 |
| Doorbell | 8字节 | 8字节 | 一个uint64_t |
| hsa_signal_t | 8字节 | 8字节 | Signal handle |

---

## 9️⃣ 常用常量

```c
// Queue大小限制
#define MIN_AQL_QUEUE_SIZE 32          // 最小32个packet
#define MAX_AQL_QUEUE_SIZE 131072      // 最大128K个packet

// Packet类型
#define HSA_PACKET_TYPE_VENDOR_SPECIFIC 0
#define HSA_PACKET_TYPE_INVALID         1
#define HSA_PACKET_TYPE_KERNEL_DISPATCH 2
#define HSA_PACKET_TYPE_BARRIER_AND     3
#define HSA_PACKET_TYPE_AGENT_DISPATCH  4
#define HSA_PACKET_TYPE_BARRIER_OR      5

// Fence scope
#define HSA_FENCE_SCOPE_NONE    0
#define HSA_FENCE_SCOPE_AGENT   1
#define HSA_FENCE_SCOPE_SYSTEM  2

// Entity限制
#define MAX_COMPUTE_ENTITIES    4      // 每个Context最多4个Compute Entity
#define MAX_SDMA_ENTITIES       2      // 每个Context最多2个SDMA Entity
```

---

## 总结

本文档提供了完整的数据结构参考。配合前面4个文档，您可以：

1. **理解数据流**: 看到数据如何从用户空间传递到硬件
2. **调试问题**: 知道每个字段的含义和作用
3. **扩展功能**: 了解结构后可以添加新功能
4. **性能优化**: 理解数据布局，优化内存访问

**完整流程回顾**:
- [第1部分: 应用层到HIP Runtime](./KERNEL_TRACE_01_APP_TO_HIP.md)
- [第2部分: HSA Runtime层](./KERNEL_TRACE_02_HSA_RUNTIME.md)
- [第3部分: KFD驱动层](./KERNEL_TRACE_03_KFD_QUEUE.md)
- [第4部分: MES调度器与硬件层](./KERNEL_TRACE_04_MES_HARDWARE.md)
- [第5部分: 关键数据结构（本文档）](./KERNEL_TRACE_05_DATA_STRUCTURES.md)
- [总览文档](./KERNEL_TRACE_INDEX.md)


