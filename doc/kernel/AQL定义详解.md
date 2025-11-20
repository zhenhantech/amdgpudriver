# AQL (Architected Queuing Language) 定义详解

## 📍 AQL定义位置

AQL的完整定义在 **HSA Runtime** 的标准头文件中：

### 主要位置

```
rocr-runtime/runtime/hsa-runtime/inc/hsa.h
  - 第2803行开始：AQL section
  - 第2810-2843行：Packet类型定义
  - 第2845-2931行：Packet header定义
  - 第2933-3070行：Kernel dispatch packet
  - 第3075-3124行：Agent dispatch packet
  - 第3129-3164行：Barrier-AND packet
  - 第3169-3204行：Barrier-OR packet
```

### 扩展定义

```
rocr-runtime/runtime/hsa-runtime/inc/hsa_ext_amd.h
  - AMD特定的AQL扩展
  - Vendor-specific packet格式
```

---

## 📚 AQL定义详解

### 1. Packet类型枚举

**位置**：`hsa.h` 第2810-2843行

```c
/**
 * @brief Packet type.
 */
typedef enum {
  /**
   * Vendor-specific packet.
   */
  HSA_PACKET_TYPE_VENDOR_SPECIFIC = 0,
  
  /**
   * The packet has been processed in the past, but has not been 
   * reassigned to the packet processor. A packet processor must 
   * not process a packet of this type. All queues support this 
   * packet type.
   */
  HSA_PACKET_TYPE_INVALID = 1,
  
  /**
   * Packet used by agents for dispatching jobs to kernel agents. 
   * Not all queues support packets of this type.
   */
  HSA_PACKET_TYPE_KERNEL_DISPATCH = 2,
  
  /**
   * Packet used by agents to delay processing of subsequent packets, 
   * and to express complex dependencies between multiple packets. 
   * All queues support this packet type.
   */
  HSA_PACKET_TYPE_BARRIER_AND = 3,
  
  /**
   * Packet used by agents for dispatching jobs to agents. Not all
   * queues support packets of this type.
   */
  HSA_PACKET_TYPE_AGENT_DISPATCH = 4,
  
  /**
   * Packet used by agents to delay processing of subsequent packets, 
   * and to express complex dependencies between multiple packets. 
   * All queues support this packet type.
   */
  HSA_PACKET_TYPE_BARRIER_OR = 5
} hsa_packet_type_t;
```

---

### 2. Fence Scope

**位置**：`hsa.h` 第2845-2863行

```c
/**
 * @brief Scope of the memory fence operation associated with a packet.
 */
typedef enum {
  /**
   * No scope (no fence is applied). The packet relies on external 
   * fences to ensure visibility of memory updates.
   */
  HSA_FENCE_SCOPE_NONE = 0,
  
  /**
   * The fence is applied with agent scope for the global segment.
   */
  HSA_FENCE_SCOPE_AGENT = 1,
  
  /**
   * The fence is applied across both agent and system scope for 
   * the global segment.
   */
  HSA_FENCE_SCOPE_SYSTEM = 2
} hsa_fence_scope_t;
```

---

### 3. Packet Header

**位置**：`hsa.h` 第2865-2931行

#### Header字段枚举

```c
/**
 * @brief Sub-fields of the header field that is present in any AQL
 * packet. The offset (with respect to the address of header) of a 
 * sub-field is identical to its enumeration constant.
 */
typedef enum {
  /**
   * Packet type. The value of this sub-field must be one of
   * hsa_packet_type_t.
   */
  HSA_PACKET_HEADER_TYPE = 0,
  
  /**
   * Barrier bit. If the barrier bit is set, the processing of the 
   * current packet only launches when all preceding packets (within 
   * the same queue) are complete.
   */
  HSA_PACKET_HEADER_BARRIER = 8,
  
  /**
   * Acquire fence scope. The value of this sub-field determines the 
   * scope and type of the memory fence operation applied before the 
   * packet enters the active phase.
   */
  HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE = 9,
  
  /**
   * Release fence scope. The value of this sub-field determines the 
   * scope and type of the memory fence operation applied after kernel 
   * completion but before the packet is completed.
   */
  HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE = 11
} hsa_packet_header_t;
```

#### Header字段宽度

```c
/**
 * @brief Width (in bits) of the sub-fields in hsa_packet_header_t.
 */
typedef enum {
  HSA_PACKET_HEADER_WIDTH_TYPE = 8,                      // 8位
  HSA_PACKET_HEADER_WIDTH_BARRIER = 1,                   // 1位
  HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE = 2,    // 2位
  HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE = 2     // 2位
} hsa_packet_header_width_t;
```

#### Header位布局

```
位15-13: Reserved
位12-11: Release Fence Scope (2位)
位10-9:  Acquire Fence Scope (2位)
位8:     Barrier (1位)
位7-0:   Packet Type (8位)
```

---

### 4. Kernel Dispatch Packet

**位置**：`hsa.h` 第2957-3070行

这是最重要的packet类型，用于提交compute kernel。

#### Setup字段

```c
/**
 * @brief Sub-fields of the kernel dispatch packet setup field.
 */
typedef enum {
  /**
   * Number of dimensions of the grid. Valid values are 1, 2, or 3.
   */
  HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS = 0
} hsa_kernel_dispatch_packet_setup_t;

typedef enum {
  HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS = 2
} hsa_kernel_dispatch_packet_setup_width_t;
```

#### 完整结构定义（64字节）

```c
/**
 * @brief AQL kernel dispatch packet
 */
typedef struct hsa_kernel_dispatch_packet_s {
  union {
    struct {
      /**
       * Packet header. Used to configure multiple packet parameters 
       * such as the packet type. The parameters are described by 
       * hsa_packet_header_t.
       */
      uint16_t header;      // 偏移0-1

      /**
       * Dispatch setup parameters. Used to configure kernel dispatch 
       * parameters such as the number of dimensions in the grid.
       */
      uint16_t setup;       // 偏移2-3
    };
    uint32_t full_header;
  };

  /**
   * X dimension of work-group, in work-items. Must be greater than 0.
   */
  uint16_t workgroup_size_x;  // 偏移4-5

  /**
   * Y dimension of work-group, in work-items. Must be greater than 0.
   * If the grid has 1 dimension, the only valid value is 1.
   */
  uint16_t workgroup_size_y;  // 偏移6-7

  /**
   * Z dimension of work-group, in work-items. Must be greater than 0.
   * If the grid has 1 or 2 dimensions, the only valid value is 1.
   */
  uint16_t workgroup_size_z;  // 偏移8-9

  /**
   * Reserved. Must be 0.
   */
  uint16_t reserved0;         // 偏移10-11

  /**
   * X dimension of grid, in work-items. Must be greater than 0. 
   * Must not be smaller than workgroup_size_x.
   */
  uint32_t grid_size_x;       // 偏移12-15

  /**
   * Y dimension of grid, in work-items. Must be greater than 0. 
   * If the grid has 1 dimension, the only valid value is 1. 
   * Must not be smaller than workgroup_size_y.
   */
  uint32_t grid_size_y;       // 偏移16-19

  /**
   * Z dimension of grid, in work-items. Must be greater than 0. 
   * If the grid has 1 or 2 dimensions, the only valid value is 1. 
   * Must not be smaller than workgroup_size_z.
   */
  uint32_t grid_size_z;       // 偏移20-23

  /**
   * Size in bytes of private memory allocation request (per work-item).
   */
  uint32_t private_segment_size;  // 偏移24-27

  /**
   * Size in bytes of group memory allocation request (per work-group). 
   * Must not be less than the sum of the group memory used by the 
   * kernel and the dynamically allocated group segment variables.
   */
  uint32_t group_segment_size;    // 偏移28-31

  /**
   * Opaque handle to a code object that includes an implementation-
   * defined executable code for the kernel.
   */
  uint64_t kernel_object;         // 偏移32-39

  /**
   * Pointer to a buffer containing the kernel arguments. May be NULL.
   * The buffer must be allocated using hsa_memory_allocate, and must 
   * not be modified once the kernel dispatch packet is enqueued until 
   * the dispatch has completed execution.
   */
  void* kernarg_address;          // 偏移40-47

  /**
   * Reserved. Must be 0.
   */
  uint32_t reserved1;             // 偏移48-51 (小端模式)

  /**
   * Reserved. Must be 0.
   */
  uint64_t reserved2;             // 偏移48-55

  /**
   * Signal used to indicate completion of the job. The application 
   * can use the special signal handle 0 to indicate that no signal 
   * is used.
   */
  hsa_signal_t completion_signal; // 偏移56-63

} hsa_kernel_dispatch_packet_t;
```

#### 内存布局图

```
偏移   大小   字段名                  说明
0-1    2字节  header                 packet类型、barrier、fence
2-3    2字节  setup                  维度信息
4-5    2字节  workgroup_size_x      Block X
6-7    2字节  workgroup_size_y      Block Y
8-9    2字节  workgroup_size_z      Block Z
10-11  2字节  reserved0             保留
12-15  4字节  grid_size_x           Grid X
16-19  4字节  grid_size_y           Grid Y
20-23  4字节  grid_size_z           Grid Z
24-27  4字节  private_segment_size  私有内存（寄存器溢出）
28-31  4字节  group_segment_size    共享内存（LDS）
32-39  8字节  kernel_object         GPU代码地址
40-47  8字节  kernarg_address       参数缓冲区地址
48-55  8字节  reserved2             保留
56-63  8字节  completion_signal     完成信号
─────────────────────────────────────────────
总计   64字节
```

---

### 5. Agent Dispatch Packet

**位置**：`hsa.h` 第3075-3124行

用于CPU端的任务分发。

```c
/**
 * @brief Agent dispatch packet.
 */
typedef struct hsa_agent_dispatch_packet_s {
  /**
   * Packet header.
   */
  uint16_t header;

  /**
   * Application-defined function to be performed by the 
   * destination agent.
   */
  uint16_t type;

  /**
   * Reserved. Must be 0.
   */
  uint32_t reserved0;

  /**
   * Address where to store the function return values, if any.
   */
  void* return_address;

  /**
   * Function arguments.
   */
  uint64_t arg[4];

  /**
   * Reserved. Must be 0.
   */
  uint64_t reserved2;

  /**
   * Signal used to indicate completion of the job.
   */
  hsa_signal_t completion_signal;

} hsa_agent_dispatch_packet_t;
```

---

### 6. Barrier-AND Packet

**位置**：`hsa.h` 第3129-3164行

用于等待多个依赖信号（所有信号都满足才继续）。

```c
/**
 * @brief Barrier-AND packet.
 */
typedef struct hsa_barrier_and_packet_s {
  /**
   * Packet header.
   */
  uint16_t header;

  /**
   * Reserved. Must be 0.
   */
  uint16_t reserved0;

  /**
   * Reserved. Must be 0.
   */
  uint32_t reserved1;

  /**
   * Array of dependent signal objects. Signals with a handle 
   * value of 0 are allowed and are interpreted by the packet 
   * processor as satisfied dependencies.
   */
  hsa_signal_t dep_signal[5];

  /**
   * Reserved. Must be 0.
   */
  uint64_t reserved2;

  /**
   * Signal used to indicate completion of the job.
   */
  hsa_signal_t completion_signal;

} hsa_barrier_and_packet_t;
```

---

### 7. Barrier-OR Packet

**位置**：`hsa.h` 第3169-3204行

用于等待多个依赖信号（任意一个信号满足就继续）。

```c
/**
 * @brief Barrier-OR packet.
 */
typedef struct hsa_barrier_or_packet_s {
  /**
   * Packet header.
   */
  uint16_t header;

  /**
   * Reserved. Must be 0.
   */
  uint16_t reserved0;

  /**
   * Reserved. Must be 0.
   */
  uint32_t reserved1;

  /**
   * Array of dependent signal objects. Signals with a handle 
   * value of 0 are allowed and are interpreted by the packet 
   * processor as dependencies not satisfied.
   */
  hsa_signal_t dep_signal[5];

  /**
   * Reserved. Must be 0.
   */
  uint64_t reserved2;

  /**
   * Signal used to indicate completion of the job.
   */
  hsa_signal_t completion_signal;

} hsa_barrier_or_packet_t;
```

---

## 🔍 相关实现文件

### 1. AQL Queue实现

```
rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp
rocr-runtime/runtime/hsa-runtime/core/inc/amd_aql_queue.h
  - AqlQueue类实现
  - Doorbell操作
  - Write/Read index管理
```

### 2. Packet构建

```
clr/rocclr/device/rocm/rocvirtual.cpp
  - VirtualGPU::submitKernelInternal()
  - 填充hsa_kernel_dispatch_packet_t
```

### 3. Packet处理（GPU端）

```
kfd/amdkfd/kfd_device_queue_manager.c
  - GPU Command Processor读取packet
  - 硬件解析和执行
```

---

## 📖 HSA标准文档

AQL是 **HSA (Heterogeneous System Architecture)** 标准的一部分：

### 官方文档

1. **HSA Programmer's Reference Manual**
   - 完整的AQL规范
   - Packet格式详细说明
   - 内存模型和同步

2. **HSA Runtime Specification**
   - HSA Runtime API定义
   - Queue操作语义
   - Signal机制

3. **在线资源**
   - HSA Foundation: http://www.hsafoundation.com/
   - Specifications: http://www.hsafoundation.com/standards/

---

## 🎯 关键概念总结

### AQL设计原则

1. **固定大小**：所有packet都是64字节
   - 方便硬件解析
   - 简化队列管理

2. **类型明确**：Header中包含类型信息
   - KERNEL_DISPATCH：GPU计算
   - BARRIER：同步
   - AGENT_DISPATCH：CPU任务

3. **硬件直接理解**
   - GPU CP直接读取解析
   - 无需软件翻译
   - 低延迟启动

4. **内存fence控制**
   - Acquire fence：进入前同步
   - Release fence：完成后同步
   - Agent/System scope

### Packet生命周期

```
1. 初始化packet（header=INVALID）
   ↓
2. 填充packet body
   ↓
3. 设置completion_signal
   ↓
4. 内存屏障（std::atomic_thread_fence）
   ↓
5. 原子写header（使packet生效）
   ↓
6. Ring doorbell（通知GPU）
   ↓
7. GPU读取并执行
   ↓
8. GPU写completion_signal
   ↓
9. 更新read_index
```

### 与其他GPU架构对比

| 特性 | HSA AQL | NVIDIA CUDA |
|-----|---------|-------------|
| **标准化** | ✅ 开放标准 | ❌ 专有格式 |
| **Packet大小** | 64字节固定 | 可变 |
| **类型** | 5种标准类型 | 专有类型 |
| **可读性** | ✅ 文档完整 | ❌ 未公开 |
| **跨厂商** | ✅ 理论支持 | ❌ NVIDIA专用 |

---

## 🛠️ 使用示例

### 构建Kernel Dispatch Packet

```cpp
void build_dispatch_packet(
    hsa_kernel_dispatch_packet_t* pkt,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t grid_z,
    uint16_t block_x,
    uint16_t block_y,
    uint16_t block_z,
    uint64_t kernel_addr,
    void* args,
    hsa_signal_t signal)
{
  // 1. 初始化header为INVALID
  pkt->header = HSA_PACKET_TYPE_INVALID;
  
  // 2. 设置setup（维度）
  pkt->setup = 3;  // 3D
  
  // 3. 设置workgroup尺寸（block）
  pkt->workgroup_size_x = block_x;
  pkt->workgroup_size_y = block_y;
  pkt->workgroup_size_z = block_z;
  
  // 4. 设置grid尺寸
  pkt->grid_size_x = grid_x;
  pkt->grid_size_y = grid_y;
  pkt->grid_size_z = grid_z;
  
  // 5. 设置内存段大小
  pkt->private_segment_size = 0;    // 寄存器溢出
  pkt->group_segment_size = 0;      // 共享内存
  
  // 6. 设置kernel和参数
  pkt->kernel_object = kernel_addr;
  pkt->kernarg_address = args;
  
  // 7. 设置completion signal
  pkt->completion_signal = signal;
  
  // 8. 保留字段
  pkt->reserved0 = 0;
  pkt->reserved1 = 0;
  pkt->reserved2 = 0;
  
  // 9. 内存屏障
  std::atomic_thread_fence(std::memory_order_release);
  
  // 10. 原子写header（使packet生效）
  uint16_t header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << 0) |
                    (0 << 8) |  // barrier=0
                    (HSA_FENCE_SCOPE_AGENT << 9) |   // acquire
                    (HSA_FENCE_SCOPE_AGENT << 11);   // release
  
  atomic::Store(&pkt->header, header, std::memory_order_release);
}
```

### 构建Barrier Packet

```cpp
void build_barrier_packet(
    hsa_barrier_and_packet_t* pkt,
    hsa_signal_t dep_signals[5],
    hsa_signal_t completion)
{
  // 1. 初始化header
  pkt->header = HSA_PACKET_TYPE_INVALID;
  
  // 2. 设置依赖信号
  for (int i = 0; i < 5; i++) {
    pkt->dep_signal[i] = dep_signals[i];
  }
  
  // 3. 设置completion signal
  pkt->completion_signal = completion;
  
  // 4. 保留字段
  pkt->reserved0 = 0;
  pkt->reserved1 = 0;
  pkt->reserved2 = 0;
  
  // 5. 内存屏障
  std::atomic_thread_fence(std::memory_order_release);
  
  // 6. 激活packet
  uint16_t header = (HSA_PACKET_TYPE_BARRIER_AND << 0) |
                    (1 << 8) |  // barrier=1（必须等待）
                    (HSA_FENCE_SCOPE_SYSTEM << 9) |
                    (HSA_FENCE_SCOPE_SYSTEM << 11);
  
  atomic::Store(&pkt->header, header, std::memory_order_release);
}
```

---

## 📊 性能考虑

### Packet大小

- **64字节** = 1个缓存行（典型）
- ✅ 一次内存访问读取完整packet
- ✅ 避免false sharing

### Header原子写

- **为什么最后写header？**
  - GPU轮询header判断packet是否有效
  - 先写body，最后写header保证原子性
  - 避免GPU读到部分写入的packet

### Reserved字段

- **为什么有reserved字段？**
  - 保持64字节对齐
  - 为未来扩展预留空间
  - 硬件可能用于内部状态

---

## 🔗 参考链接

1. **本地源码**
   - `rocr-runtime/runtime/hsa-runtime/inc/hsa.h`
   - `rocr-runtime/runtime/hsa-runtime/inc/hsa_ext_amd.h`

2. **相关文档**
   - `ROCm_Kernel_Dispatch流程详解.md` - Kernel提交流程
   - `ROCm内存管理分层详解.md` - 内存管理
   - `AMD_ROCM_架构分析.md` - 整体架构

3. **在线资源**
   - HSA Foundation官网
   - ROCm Documentation
   - AMD Developer Resources

---

**文档版本**：1.0  
**创建日期**：2024年11月  
**作者**：基于ROCm源码分析

