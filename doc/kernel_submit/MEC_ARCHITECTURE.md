# MEC (Micro-Engine Compute) 架构详解

## 📋 文档概览

本文档深入解析 AMD GPU 中的 **MEC (Micro-Engine Compute)** 架构和 **XCD/XCP** 分区机制，这是理解 GPU 计算队列管理的基础。

**关键词**: MEC, Micro-Engine Compute, Command Processor, Pipe, Queue, HQD, XCD, XCC, XCP, Partition Mode

**重点内容**:
- ⭐ MI308X 架构：**8 个 XCC (逻辑核心，系统可见)**
- ⭐ XCP (XCC Partition) 软件抽象和分区模式（管理 XCC）
- ⭐ 为什么系统中有 127 个 DRI 设备（8 GPU × 8 XCC × 2 节点）

**⚠️ 重要概念区分**:
- **XCD (Die)**: 物理计算芯片（数量未在代码/日志中明确）
- **XCC (Core)**: 逻辑计算核心，MI308X 每个有 **8 个**（✅ 系统确认：8 个 render 节点）
- **XCP (Partition)**: 驱动层软件抽象，对应 XCC

**📊 系统观察到的证据**:
- ✅ 每个 GPU 有 8 个 DRI render 节点（renderD128-135 等）
- ✅ 每个 GPU 有 80 个 Compute Units（从 rocminfo 确认）
- ℹ️ XCD 物理数量：推测为 4 个（80 CU ÷ 20 CU/XCD）或 8 个（80 CU ÷ 10 CU/XCD）
- ⚠️ 注意：代码中没有直接证据显示 XCD 确切数量

---

## 1. MEC 基础概念

### 1.1 什么是 MEC？

**MEC = Micro-Engine Compute（计算微引擎）**

```
定义: GPU 中专门负责处理 Compute 工作负载的硬件单元
作用: 管理和调度计算队列（Compute Queues）
本质: 一个独立的 Command Processor (CP) 实例
固件: 运行 MEC 微代码固件来控制队列调度和执行
```

**与传统概念的对应**:
- **MEC** ≈ CPU 的核心（Core）
- **Pipe** ≈ CPU 的执行单元
- **Queue** ≈ CPU 的硬件线程

### 1.2 MEC 的历史演进

| GPU 代 | 架构 | MEC 数量 | 说明 |
|--------|------|---------|------|
| **GCN 1-2** | GFX 6-7 | 1 个 | 最初引入 MEC |
| **GCN 3-5** | GFX 8-9 | 1-2 个 | 开始支持双 MEC |
| **CDNA 1** | GFX 9.0.8 (MI100) | 1 个 | 数据中心 GPU |
| **CDNA 2** | GFX 9.0.a (MI250X) | 2 个 | 增强计算能力 |
| **CDNA 2** | GFX 9.4.2 (MI308X) | 2 个 | 本文档重点 |
| **CDNA 3** | GFX 9.4.3 (MI300) | 2 个 | 最新架构 |
| **RDNA 2-3** | GFX 10-11 | 1-2 个 | 游戏/消费级 GPU |

### 1.3 MEC 与其他组件的关系

```
GPU 硬件架构
│
├─ GFX Block (Graphics & Compute)
│  ├─ Graphics Engine (ME/PFP/CE)
│  │  └─ 负责图形渲染管线
│  │
│  └─ Compute Engine (MEC) ⭐
│     ├─ MEC 0 (Primary Compute Engine)
│     │  └─ 处理 Compute/OpenCL/HIP Kernels
│     │
│     └─ MEC 1 (Secondary Compute Engine)
│        └─ 扩展计算能力（如果存在）
│
├─ SDMA (System DMA Engine)
│  └─ 负责内存拷贝
│
└─ Display Controller
   └─ 负责显示输出
```

---

## 2. MEC 架构层次详解

### 2.1 三层架构：MEC → Pipe → Queue

```
MEC（Micro-Engine Compute）
  │
  ├─ Pipe 0（管道 0）
  │  ├─ Queue 0 ──┐
  │  ├─ Queue 1   │
  │  ├─ Queue 2   ├─ HQD (Hardware Queue Descriptors)
  │  ├─ ...       │
  │  └─ Queue 7 ──┘
  │
  ├─ Pipe 1（管道 1）
  │  └─ Queue 0-7
  │
  ├─ Pipe 2（管道 2）
  │  └─ Queue 0-7
  │
  └─ Pipe 3（管道 3）
     └─ Queue 0-7
```

### 2.2 各层含义

#### 2.2.1 MEC 层（Micro-Engine）

| 属性 | 说明 |
|------|------|
| **定义** | 独立的计算微引擎硬件单元 |
| **功能** | 管理和调度计算队列，执行 AQL packets |
| **固件** | 运行独立的 MEC 微代码（MEC firmware） |
| **寄存器** | 独立的寄存器空间（CP_MEC_*） |
| **独立性** | 每个 MEC 可以独立工作 |

#### 2.2.2 Pipe 层（管道）

| 属性 | 说明 |
|------|------|
| **定义** | MEC 内部的队列组 |
| **功能** | 负载均衡，将队列分组管理 |
| **并行性** | 不同 Pipe 可以并行处理队列 |
| **资源隔离** | 一定程度的资源隔离 |

#### 2.2.3 Queue 层（队列）

| 属性 | 说明 |
|------|------|
| **定义** | 硬件队列槽位（HQD） |
| **功能** | 存储队列的元数据和状态 |
| **对应** | 每个 Queue 对应一组 CP_HQD_* 寄存器 |
| **用户可见** | 对应用户空间的 HSA Queue |

### 2.3 MI308X 的完整架构

```
MI308X (gfx942, IP 9.4.2)
│
├─ MEC 0（用于 KFD Compute）⭐
│  │
│  ├─ Pipe 0
│  │  ├─ Queue 0 (HQD #0)  ← CP_HQD 寄存器组 0
│  │  ├─ Queue 1 (HQD #1)  ← CP_HQD 寄存器组 1
│  │  ├─ Queue 2 (HQD #2)
│  │  ├─ Queue 3 (HQD #3)
│  │  ├─ Queue 4 (HQD #4)
│  │  ├─ Queue 5 (HQD #5)
│  │  ├─ Queue 6 (HQD #6)
│  │  └─ Queue 7 (HQD #7)
│  │
│  ├─ Pipe 1
│  │  └─ Queue 0-7 (HQD #8-15)
│  │
│  ├─ Pipe 2
│  │  └─ Queue 0-7 (HQD #16-23)
│  │
│  └─ Pipe 3
│     └─ Queue 0-7 (HQD #24-31)
│
│  └─ 小计: 4 pipes × 8 queues = 32 个 HQD
│
└─ MEC 1（预留，不用于 KFD）
   └─ Pipe 0-3
      └─ 各 8 个 queues
      └─ 小计: 32 个 HQD

总硬件队列: 2 MECs × 32 HQDs = 64 个 HQD（理论）
KFD 可用: 1 MEC × 32 HQDs = 32 个 HQD（实际）
```

---

## 3. 硬件配置和代码证据

### 3.1 MI308X 的 MEC 配置

**代码位置**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/gfx_v9_0.c`

**注意**: MI308X 是 **gfx942 (IP 9.4.2)**，其配置在 gfx9 系列通用文件中。

```c
// 行 2220-2233: 根据 IP 版本设置 MEC 数量

static int gfx_v9_0_sw_init(void *handle)
{
    struct amdgpu_device *adev = (struct amdgpu_device *)handle;
    
    // ... 其他初始化 ...
    
    switch (amdgpu_ip_version(adev, GC_HWIP, 0)) {
    case IP_VERSION(9, 0, 1):
    case IP_VERSION(9, 2, 1):
    case IP_VERSION(9, 4, 0):
    case IP_VERSION(9, 2, 2):
    case IP_VERSION(9, 1, 0):
    case IP_VERSION(9, 4, 1):
    case IP_VERSION(9, 3, 0):
    case IP_VERSION(9, 4, 2):  // ⭐ MI308X (gfx942)
        adev->gfx.mec.num_mec = 2;  // ⭐ 2 个 MECs
        break;
    default:
        adev->gfx.mec.num_mec = 1;
        break;
    }
    
    // 行 2272-2273: 通用配置（适用于所有 gfx9 系列）
    adev->gfx.mec.num_pipe_per_mec = 4;     // ⭐ 每个 MEC 4 个 pipes
    adev->gfx.mec.num_queue_per_pipe = 8;   // ⭐ 每个 pipe 8 个 queues
    
    // ... 其他配置 ...
    
    return 0;
}
```

### 3.2 MEC 数据结构定义

**代码位置**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/amdgpu_gfx.h`

```c
// 行 102-114

struct amdgpu_mec {
    // MEC 固件相关
    struct amdgpu_bo    *hpd_eop_obj;       // HPD (High Priority Doorbell) EOP buffer
    u64                 hpd_eop_gpu_addr;
    struct amdgpu_bo    *mec_fw_obj;        // MEC 固件对象
    u64                 mec_fw_gpu_addr;
    struct amdgpu_bo    *mec_fw_data_obj;
    u64                 mec_fw_data_gpu_addr;
    
    // MEC 拓扑配置
    u32 num_mec;                            // ⭐ MEC 数量
    u32 num_pipe_per_mec;                   // ⭐ 每个 MEC 的 Pipe 数量
    u32 num_queue_per_pipe;                 // ⭐ 每个 Pipe 的 Queue 数量
    
    // MQD (Memory Queue Descriptor) 备份
    void *mqd_backup[AMDGPU_MAX_COMPUTE_RINGS * AMDGPU_MAX_GC_INSTANCES];
};
```

### 3.3 不同 GPU 的 MEC 配置对比

| GPU 型号 | GFX 版本 | num_mec | num_pipe_per_mec | num_queue_per_pipe | 总 HQD 数（per MEC） |
|---------|----------|---------|------------------|-------------------|-------------------|
| **MI308X** | gfx942 | 2 | 4 | 8 | 32 |
| **MI300A/X** | gfx940/941 | 2 | 4 | 8 | 32 |
| **MI250X** | gfx90a | 2 | 4 | 8 | 32 |
| **MI100** | gfx908 | 1 | 4 | 8 | 32 |
| **Vega 20** | gfx906 | 2 | 4 | 8 | 32 |
| **RX 6900 XT** | gfx1030 | 2 | 4 | 8 | 32 |
| **RX 7900 XTX** | gfx1100 | 2 | 4 | 8 | 32 |

**观察**:
- ✅ Pipe 和 Queue 数量在不同 GPU 间**非常一致**（4 pipes × 8 queues = 32 HQDs）
- ✅ 这是 AMD GPU 架构的标准配置
- ✅ 主要区别在 MEC 数量（1 个或 2 个）

---

## 4. XCD/XCP 架构与分区模式 ⭐ **MI308X 特性**

### 4.1 什么是 XCD 和 XCC？

**关键概念**:
- **XCD = eXtended Compute Die（扩展计算芯片）** - 物理硬件单元
- **XCC = eXtended Compute Core（扩展计算核心）** - 逻辑软件单元

MI308X 采用 **chiplet（芯片小片）架构**：

```
MI308X 架构（系统观察）
│
├─ 8 个 XCC (eXtended Compute Core) ⭐ 软件可见
│  ├─ XCC 0: 逻辑计算核心（对应 renderD128）
│  ├─ XCC 1: 逻辑计算核心（对应 renderD129）
│  ├─ ...
│  └─ XCC 7: 逻辑计算核心（对应 renderD135）
│
├─ XCD 物理数量: **未在代码中明确** ⚠️
│  └─ 推测: 可能是 4 个或 8 个（取决于每 XCD 的 CU 数量）
│
└─ 统一的内存和互连系统
```

**系统可观察到的证据**:
```bash
# 每个 GPU 有 8 个 render 节点（对应 8 个 XCC）
$ ls /dev/dri/renderD{128..135}
renderD128 renderD129 ... renderD135  # GPU 1 的 8 个 XCC

# 每个 GPU 有 80 个 Compute Units
$ rocminfo | grep -A2 "Name.*gfx942"
Compute Unit:            80  # 每个 MI308X 芯片
```

**重要说明** ⚠️:
- **XCC (8 个)**: 可以通过 render 节点直接观察，代码中有 `xcc_mask` 管理
- **XCD 数量**: 在驱动代码和日志中**没有直接证据**，只能通过 CU 数量推测
  - 假设 1: 4 个 XCD × 20 CU/XCD = 80 CU
  - 假设 2: 8 个 XCD × 10 CU/XCD = 80 CU
- 本文档主要讨论**软件可见的 XCC**，而非物理 XCD

**每个 XCC 包含**:
- 约 10-20 个计算单元（CUs）
- L1/L2 缓存
- 独立的调度能力

### 4.2 XCP (XCC Partition) 软件抽象

**XCP = XCC Partition（XCC 分区）**

驱动层使用 **XCP** 作为软件抽象来管理逻辑的 **XCC**（不是物理的 XCD）：

```c
// ROCm_keyDriver/.../amd/amdgpu/amdgpu_xcp.h

#define MAX_XCP 8  // 最多支持 8 个 XCP

struct amdgpu_xcp_mgr {
    struct amdgpu_device *adev;
    struct amdgpu_xcp xcp[MAX_XCP];  // 8 个独立的 XCP
    uint8_t num_xcps;                 // 当前激活的 XCP 数量
    int8_t mode;                      // 分区模式
    struct mutex xcp_lock;
    // ...
};

struct amdgpu_xcp {
    struct amdgpu_xcp_ip ip[AMDGPU_XCP_MAX_BLOCKS];  // IP 块（GFX, SDMA, VCN）
    uint8_t id;                      // XCP ID (0-7)
    uint8_t mem_id;                  // 内存分区 ID
    struct drm_device *ddev;         // 独立的 DRM 设备（Primary node）
    struct drm_device *rdev;         // 独立的 DRM 设备（Render node）
    atomic_t ref_cnt;
    // ...
};
```

**代码位置**: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/amdgpu_xcp.h`

### 4.3 支持的分区模式 (Partition Modes)

MI308X 支持多种分区模式，允许灵活配置 **8 个 XCC**（逻辑核心）：

```c
// ROCm_keyDriver/.../amd/amdgpu/amdgpu_gfx.h: 63-72

enum amdgpu_gfx_partition {
    AMDGPU_SPX_PARTITION_MODE = 0,  // Single Partition
    AMDGPU_DPX_PARTITION_MODE = 1,  // Dual Partition
    AMDGPU_TPX_PARTITION_MODE = 2,  // Triple Partition
    AMDGPU_QPX_PARTITION_MODE = 3,  // Quad Partition
    AMDGPU_CPX_PARTITION_MODE = 4,  // Custom Partition
    AMDGPU_UNKNOWN_COMPUTE_PARTITION_MODE = -1,
    AMDGPU_AUTO_COMPUTE_PARTITION_MODE = -2,
};
```

| 模式 | 分区数 | 每分区 XCC 数 | XCC 分配 | 典型用途 |
|------|--------|--------------|---------|---------|
| **SPX** | 1 | 8 | 1×8 | 单进程最大性能 |
| **DPX** | 2 | 4 | 2×4 | 双进程严格隔离 |
| **TPX** | 3 | 不均匀 | 3×2 + 1×2 | 三进程场景 |
| **QPX** | 4 | 2 | 4×2 | 四进程细粒度隔离 |
| **CPX** | 可配置 | 灵活 | 自定义 | 特殊需求 |

**分区模式切换**（动态重配置）:

```c
// ROCm_keyDriver/.../amd/amdgpu/amdgpu_xcp.c

// 切换分区模式
int amdgpu_xcp_switch_partition_mode(struct amdgpu_xcp_mgr *xcp_mgr, int mode);

// 查询当前分区模式
int amdgpu_xcp_query_partition_mode(struct amdgpu_xcp_mgr *xcp_mgr, u32 flags);

// 获取指定 XCP 的资源
int amdgpu_xcp_get_partition(struct amdgpu_xcp_mgr *xcp_mgr,
                             enum AMDGPU_XCP_IP_BLOCK ip, int instance);
```

### 4.4 XCP 与 DRI 设备的映射关系 🔍

**这就是为什么系统中有 127 个 DRI 设备！**

```
DRI 设备分布 (以 GPU 1 为例，0000:0a:00.0)
│
├─ Primary Nodes (DRI 1-8)
│  ├─ DRI 1: XCC 0 - 显示和特权操作
│  ├─ DRI 2: XCC 1
│  ├─ DRI 3: XCC 2
│  ├─ DRI 4: XCC 3
│  ├─ DRI 5: XCC 4
│  ├─ DRI 6: XCC 5
│  ├─ DRI 7: XCC 6
│  └─ DRI 8: XCC 7
│
└─ Render Nodes (DRI 128-135) ⭐ 系统实际观察
   ├─ DRI 128: XCC 0 - 计算和渲染（无需特权）
   ├─ DRI 129: XCC 1
   ├─ ...
   └─ DRI 135: XCC 7

软件架构（可验证）:
- 8 个 XCC (逻辑核心，对应 8 个 render 节点) ✅
- 8 个 XCP (软件分区，对应 8 个 XCC) ✅
- 80 个 Compute Units（每个 GPU）✅

物理架构（未确认）:
- XCD 数量: 代码/日志中无直接证据 ⚠️

DRI 设备总数: 8 GPU × 8 XCC × 2 节点类型 + 1 集成显卡 = 129 个 DRI 设备
```

**验证命令**:

```bash
# 查看 DRI 设备的实际映射
$ for i in {1..8} {128..135}; do 
    echo -n "DRI $i: "; 
    sudo cat /sys/kernel/debug/dri/$i/name; 
done

# 输出示例:
# DRI 1: amdgpu dev=0000:0a:00.0 unique=0000:0a:00.0  (XCC 0)
# DRI 2: amdgpu dev=0000:0a:00.0 unique=0000:0a:00.0  (XCC 1)
# DRI 3: amdgpu dev=0000:0a:00.0 unique=0000:0a:00.0  (XCC 2)
# ...
# DRI 128: amdgpu dev=0000:0a:00.0 unique=0000:0a:00.0  (XCC 0 render)
# DRI 129: amdgpu dev=0000:0a:00.0 unique=0000:0a:00.0  (XCC 1 render)
```

### 4.5 XCP 的独立控制能力

驱动层面对每个 XCP 的精细控制：

```c
// ROCm_keyDriver/.../amd/amdgpu/amdgpu_xcp.h

// 遍历所有 XCP
#define for_each_xcp(xcp_mgr, xcp, i) \
    for (i = 0, xcp = amdgpu_get_next_xcp(xcp_mgr, &i); xcp; \
         ++i, xcp = amdgpu_get_next_xcp(xcp_mgr, &i))

// 单个 XCP 的电源管理
int amdgpu_xcp_prepare_suspend(struct amdgpu_xcp_mgr *xcp_mgr, int xcp_id);
int amdgpu_xcp_suspend(struct amdgpu_xcp_mgr *xcp_mgr, int xcp_id);
int amdgpu_xcp_prepare_resume(struct amdgpu_xcp_mgr *xcp_mgr, int xcp_id);
int amdgpu_xcp_resume(struct amdgpu_xcp_mgr *xcp_mgr, int xcp_id);

// 为 Ring 分配特定的 XCP
static void aqua_vanjaram_set_xcp_id(struct amdgpu_device *adev,
                     uint32_t inst_idx, struct amdgpu_ring *ring)
{
    int xcp_id = amdgpu_xcp_get_partition(adev->xcp_mgr, ip_blk, inst_mask);
    ring->xcp_id = xcp_id;  // Ring 绑定到特定 XCP
}
```

**XCP 独立控制能力总结**:

| 能力 | 是否支持 | 说明 |
|------|---------|------|
| **独立访问** | ✅ 是 | 每个 XCC 有独立的 DRI 节点 |
| **动态分区** | ✅ 是 | 运行时切换分区模式 |
| **资源隔离** | ✅ 是 | VMID、内存、调度器完全隔离 |
| **单独挂起/恢复** | ✅ 是 | 可以单独操作某个 XCP (XCC) |
| **独立调度** | ✅ 是 | 每个 XCP 有独立的 GPU scheduler |
| **选择特定 XCC** | ✅ 是 | 通过 `xcp_id` 参数精确指定 |

### 4.6 用户空间如何使用 XCP？

**方式 1: 环境变量选择**

```bash
# 选择特定的 GPU 分区
export CUDA_VISIBLE_DEVICES=0,1  # 选择前 2 个分区

# ROCm 环境变量
export ROCR_VISIBLE_DEVICES=0    # 只使用第一个 XCP
export HIP_VISIBLE_DEVICES=0,2   # 使用 XCP 0 和 XCP 2
```

**方式 2: ROCm Runtime API**

```c
// HSA API: 枚举所有 agent（对应不同的 XCP）
hsa_status_t hsa_iterate_agents(
    hsa_status_t (*callback)(hsa_agent_t agent, void* data),
    void* data
);

// 在特定 agent (XCP) 上创建队列
hsa_status_t hsa_queue_create(
    hsa_agent_t agent,      // 指定 XCP
    uint32_t size,
    hsa_queue_type_t type,
    // ...
    hsa_queue_t** queue
);
```

**方式 3: 直接访问 DRI 节点**

```bash
# 直接打开特定的 render node
int fd = open("/dev/dri/renderD128", O_RDWR);  // XCP 0
int fd = open("/dev/dri/renderD129", O_RDWR);  // XCP 1
```

### 4.7 验证和调试 XCP ⭐ **实战验证**

#### 4.7.1 通过 sysfs 查看分区模式（推荐）

MI308X 提供了**官方 sysfs 接口**来查看和配置分区模式：

```bash
# 1. 查看当前的计算分区模式（Compute Partition）
$ cat /sys/class/drm/card*/device/current_compute_partition
SPX

# 2. 查看可用的计算分区模式
$ cat /sys/class/drm/card*/device/available_compute_partition
SPX, DPX, CPX

# 3. 查看当前的内存分区模式（Memory Partition）
$ cat /sys/class/drm/card*/device/current_memory_partition
NPS1

# 4. 查看可用的内存分区模式
$ cat /sys/class/drm/card*/device/available_memory_partition
NPS1, NPS4

# 5. 切换分区模式（需要 root 权限）⚠️
$ echo "DPX" | sudo tee /sys/class/drm/card1/device/current_compute_partition
# 注意: 切换模式可能需要重新加载应用程序
```

**代码实现**: `ROCm_keyDriver/.../amd/amdgpu/amdgpu_gfx.c`

```c
// Line 1346-1362: 读取当前分区模式
static ssize_t amdgpu_gfx_get_current_compute_partition(struct device *dev,
                        struct device_attribute *addr, char *buf)
{
    struct amdgpu_device *adev = drm_to_adev(dev_get_drvdata(dev));
    int mode = amdgpu_xcp_query_partition_mode(adev->xcp_mgr, 
                                              AMDGPU_XCP_FL_NONE);
    return sysfs_emit(buf, "%s\n", amdgpu_gfx_compute_mode_desc(mode));
}

// Line 1364-1413: 设置分区模式
static ssize_t amdgpu_gfx_set_compute_partition(struct device *dev,
                        struct device_attribute *addr,
                        const char *buf, size_t count)
{
    // 解析模式字符串
    if (!strncasecmp("SPX", buf, strlen("SPX"))) {
        mode = AMDGPU_SPX_PARTITION_MODE;
    } else if (!strncasecmp("DPX", buf, strlen("DPX"))) {
        // DPX 要求 XCC 数量是 4 的倍数
        if (num_xcc % 4)
            return -EINVAL;
        mode = AMDGPU_DPX_PARTITION_MODE;
    } else if (!strncasecmp("QPX", buf, strlen("QPX"))) {
        // QPX 要求 XCC 数量是 8
        if (num_xcc != 8)
            return -EINVAL;
        mode = AMDGPU_QPX_PARTITION_MODE;
    }
    // ...
    ret = amdgpu_xcp_switch_partition_mode(adev->xcp_mgr, mode);
    return ret ? ret : count;
}
```

# SR 26
[root@hjbog-srdc-26 device]# cat /sys/class/drm/card*/device/current_compute_partition
SPX
SPX
SPX
SPX
SPX
SPX
SPX
SPX
[root@hjbog-srdc-26 device]# cat /sys/class/drm/card*/device/available_compute_partition
SPX, DPX, CPX
SPX, DPX, CPX
SPX, DPX, CPX
SPX, DPX, CPX
SPX, DPX, CPX
SPX, DPX, CPX
SPX, DPX, CPX
SPX, DPX, CPX

[root@hjbog-srdc-26 device]# cat /sys/class/drm/card*/device/current_memory_partition
NPS1
NPS1
NPS1
NPS1
NPS1
NPS1
NPS1
NPS1

[root@hjbog-srdc-26 device]# cat /sys/class/drm/card*/device/available_memory_partition
NPS1, NPS4
NPS1, NPS4
NPS1, NPS4
NPS1, NPS4
NPS1, NPS4
NPS1, NPS4
NPS1, NPS4
NPS1, NPS4


**⚠️ 分区切换限制**:
- 不能在 GPU reset 期间切换
- QPX 模式要求 8 个 XCC
- DPX 模式要求 XCC 数量是 4 的倍数
- TPX 模式要求 6 个 XCC

**📊 sysfs 接口总结（基础接口）**:

| 文件路径 | 权限 | 说明 | 示例值 |
|---------|------|------|--------|
| `/sys/class/drm/card*/device/current_compute_partition` | rw | 当前计算分区模式 | `SPX` |
| `/sys/class/drm/card*/device/available_compute_partition` | r | 可用的计算分区模式 | `SPX, DPX, CPX` |
| `/sys/class/drm/card*/device/current_memory_partition` | r | 当前内存分区模式 | `NPS1` |
| `/sys/class/drm/card*/device/available_memory_partition` | r | 可用的内存分区模式 | `NPS1, NPS4` |

**📊 sysfs 高级配置接口（需要 root 权限）**:

```bash
# 查看详细分区配置（在 compute_partition_config/ 目录下）
$ sudo cat /sys/class/drm/card1/device/compute_partition_config/xcp_config
SPX

$ sudo cat /sys/class/drm/card1/device/compute_partition_config/supported_xcp_configs
SPX, DPX, CPX

$ sudo cat /sys/class/drm/card1/device/compute_partition_config/supported_nps_configs
NPS1

# 查看硬件单元实例数（num_inst = 实例数，num_shared = 共享数）
$ sudo cat /sys/class/drm/card1/device/compute_partition_config/xcc/num_inst
4

$ sudo cat /sys/class/drm/card1/device/compute_partition_config/dma/num_inst
8  # 8 个 SDMA 引擎

$ sudo cat /sys/class/drm/card1/device/compute_partition_config/dec/num_inst
4  # 4 个视频解码引擎

$ sudo cat /sys/class/drm/card1/device/compute_partition_config/jpeg/num_inst
32  # 32 个 JPEG 解码器
```

**💡 NPS (NUMA Per Socket) 内存分区说明**:
- **NPS1**: 单个 NUMA 节点（默认），所有 XCC 共享内存
- **NPS4**: 4 个 NUMA 节点，每个 XCC 组有独立内存域
- 内存分区与计算分区独立配置

**⚠️ 注意**: `xcc/num_inst = 4` 可能表示物理 XCC 组数量，与逻辑 XCC (8 个) 不同

#### 4.7.2 通过 DRI 设备验证

```bash
# 列出所有 render nodes
$ ls /dev/dri/renderD*
/dev/dri/renderD128  # GPU 1, XCC 0
/dev/dri/renderD129  # GPU 1, XCC 1
...
/dev/dri/renderD135  # GPU 1, XCC 7

# 每 8 个 render 节点对应一个 GPU 的 8 个 XCC
```

#### 4.7.3 完整验证脚本 🎯

已提供完整的分区模式验证脚本：**`scripts/partition_info.sh`**

**脚本位置**: `/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/scripts/partition_info.sh`

**使用方法**:

```bash
# 方法 1: 直接运行（推荐）
$ cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit
$ bash scripts/partition_info.sh

# 方法 2: 用 sudo 运行以获取更多信息
$ sudo bash scripts/partition_info.sh

# 方法 3: 添加到 PATH
$ export PATH=$PATH:/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/scripts
$ partition_info.sh
```

**脚本功能**:
- ✅ 自动扫描所有 GPU 卡
- ✅ 显示计算分区和内存分区配置
- ✅ 显示可用的分区模式
- ✅ 显示硬件单元数量（XCC, SDMA, DEC, JPEG）
- ✅ 显示 render 节点分布
- ✅ 统计总的 render 节点数量

**输出示例**:

```bash
$ bash scripts/partition_info.sh
=== card1 ===
Compute Partition: SPX
Available Compute: SPX, DPX, CPX
Memory Partition:  NPS1
Available Memory:  NPS1, NPS4
XCC Instances:     4
SDMA Engines:      8

=== Render Nodes Distribution ===
GPU 0: renderD128 - renderD135
GPU 1: renderD136 - renderD143
...
```

**使用 rocm-smi 查看**:

```bash
$ rocm-smi --showid

# 输出会显示 8 个 GPU（逻辑上），每个对应一个物理 MI308X
# 每个 MI308X 有 8 个 XCC（逻辑核心），80 个 CU
GPU[0]  : gfx942  (8 XCC, 80 CU)
GPU[1]  : gfx942  (8 XCC, 80 CU)
...
GPU[7]  : gfx942  (8 XCC, 80 CU)

# 总计: 8 GPU × 8 XCC = 64 个逻辑 XCC
#      8 GPU × 80 CU = 640 个 Compute Units
```

**代码验证**:

```c
// ROCm_keyDriver/.../amd/amdkfd/kfd_topology.c

// 枚举 XCC（对应 XCD）
int num_xcc = NUM_XCC(knode->xcc_mask);  // 获取 XCC 数量
int start = ffs(knode->xcc_mask) - 1;    // 第一个 XCC 的 ID
int end = start + num_xcc;                // 最后一个 XCC 的 ID

// 遍历所有 XCC
for (xcc = start; xcc < end; xcc++) {
    // 对每个 XCC 进行操作
}
```

### 4.8 XCP 的应用场景

| 场景 | 推荐模式 | 说明 |
|------|---------|------|
| **单进程最大性能** | SPX (1×8) | AI 训练、大模型推理 |
| **多进程严格隔离** | DPX/QPX | 云环境、容器化部署 |
| **资源共享** | CPX | 灵活配置，按需分配 |
| **开发调试** | SPX | 简单直接，全部资源可用 |

**云环境示例**:

```yaml
# Kubernetes Pod 配置
resources:
  limits:
    amd.com/gpu: 2  # 分配 2 个 XCP (QPX 模式下)
```

---

## 5. MEC 0 vs MEC 1：两个 MEC 的不同用途

### 5.1 双 MEC 的使用策略 ⚠️ **重要更新**

MI308X 有 2 个 MEC，**两个都在使用，但用途不同**：

| MEC | 别名 | 用途 | VMID | 队列类型 |
|-----|------|------|------|---------|
| **MEC 0** | ME 1 | **用户态 Compute 队列** | 8-15 | AQL 队列 (ROCm/HIP) |
| **MEC 1** | ME 2 | **内核态特权队列** | 0 | KIQ/HIQ (PM4 命令) |

**KFD 用户态队列只使用 MEC 0**：

```c
// ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_device_queue_manager.c
// 行 965-997

static int allocate_hqd(struct device_queue_manager *dqm, struct queue *q)
{
    bool set;
    int pipe, bit, i;
    
    set = false;
    
    // 轮询所有 pipes
    for (pipe = dqm->next_pipe_to_allocate, i = 0;
         i < get_pipes_per_mec(dqm);
         pipe = ((pipe + 1) % get_pipes_per_mec(dqm)), ++i) {
        
        // ⭐ 固定检查 MEC 0（第一个参数 = 0）
        if (!is_pipe_enabled(dqm, 0, pipe))
            continue;
        
        // 在 MEC 0 的 pipe 中分配 queue
        if (dqm->allocated_queues[pipe] != 0) {
            bit = ffs(dqm->allocated_queues[pipe]) - 1;
            dqm->allocated_queues[pipe] &= ~(1 << bit);
            q->pipe = pipe;
            q->queue = bit;
            set = true;
            break;
        }
    }
    
    if (!set)
        return -EBUSY;
    
    pr_debug("hqd slot - pipe %d, queue %d\n", q->pipe, q->queue);
    
    // 更新下一个分配的 pipe（水平分配）
    dqm->next_pipe_to_allocate = (pipe + 1) % get_pipes_per_mec(dqm);
    
    return 0;
}
```

### 5.2 MEC 1 的特权队列用途

**MEC 1 (ME 2) 专门用于内核态的特权队列**：

| 队列类型 | 用途 | 示例 |
|---------|------|------|
| **KIQ** (Kernel Interface Queue) | 内核与 GPU 通信 | 寄存器访问、队列管理、固件命令 |
| **HIQ** (Hardware Interface Queue) | 硬件调度器通信 | MES调度命令（如果启用MES） |
| **系统队列** | 驱动内部操作 | SDMA 队列、维护任务等 |

**实际示例**（来自 `umr -cpc` 输出）：

```
ME 1 Pipe 0: INSTR_PTR 0x47a  INT_STAT_DEBUG 0x4000000
Pipe 0  Queue 2  VMID 8   ← 用户态 Compute 队列 (MEC 0)
  PQ BASE 0x7f6c61920000  RPTR 0x10  WPTR 0x10
  MQD 0xa02800  AQL_CONTROL 0x1   ← AQL 队列标记

ME 2 Pipe 0: INSTR_PTR 0x47a  INT_STAT_DEBUG 0x0
Pipe 0  Queue 0  VMID 0   ← 内核态特权队列 (MEC 1)
  PQ BASE 0xa00000  RPTR 0x94  WPTR 0x94
  MQD 0x10847dd1000  AQL_CONTROL 0x0   ← 非 AQL，是 PM4 命令队列
```

### 5.3 为什么用户队列不用 MEC 1？

| 原因 | 说明 |
|------|------|
| **功能隔离** | 用户态和内核态队列分离，提高安全性和稳定性 |
| **资源预留** | MEC 1 专门服务内核态，保证系统操作的响应速度 |
| **历史兼容** | 早期 GPU 只有 1 个 MEC，KFD 设计基于单 MEC |
| **容量充足** | 32 个用户队列（MEC 0）通常足够，大多数应用不需要超过 32 个 |
| **简化管理** | 避免跨 MEC 调度复杂性 |

## 6. MEC 固件（MEC Firmware）

### 6.1 MEC 固件的作用

```
MEC 固件 = MEC 微引擎运行的微代码程序

功能:
├─ 解析和执行 AQL packets
├─ 管理队列状态（active, suspended, reset）
├─ 处理 doorbell 信号
├─ 调度 CU（Compute Unit）执行 kernel
├─ 处理队列抢占和恢复
└─ 报告队列事件和错误
```

### 6.2 MEC 固件版本查看

**方法 1: dmesg 查看**

```bash
$ dmesg | grep -i "mec.*fw"

# 输出示例:
[    2.345678] [drm] MEC firmware version: 51 feature version: 51
[    2.345679] [drm] MEC 2 is disabled
```

**方法 2: sysfs 查看**

```bash
$ cat /sys/kernel/debug/dri/0/amdgpu_firmware_info | grep -A3 "MEC"

# 输出示例:
MEC feature version: 51, firmware version: 0x00000033
MEC2 feature version: 0, firmware version: 0x00000000
```

**方法 3: 代码查看**

```c
// ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_topology.c
// 行 2002-2006

switch (KFD_GC_VERSION(knode)) {
case IP_VERSION(9, 4, 2):  // MI308X
    firmware_supported = dev->gpu->kfd->mec_fw_version >= 51;
    break;
case IP_VERSION(9, 4, 3):  // MI300
    firmware_supported = dev->gpu->kfd->mec_fw_version >= 60;
    break;
// ...
}
```

### 6.3 MEC 固件文件位置

```bash
# MEC 固件文件通常位于
/lib/firmware/amdgpu/

# MI308X (gfx942) 的固件文件
gc_9_4_3_mec.bin       # MEC firmware
gc_9_4_3_rlc.bin       # RLC firmware
# ...

# 查看固件文件
$ ls -lh /lib/firmware/amdgpu/ | grep mec

-rw-r--r-- 1 root root  28K Jan  1 2024 gc_9_4_3_mec.bin
```

---

## 7. MEC 寄存器访问

### 7.1 CP_HQD 寄存器组

每个 (MEC, Pipe, Queue) 组合对应一组独立的 CP_HQD_* 寄存器：

```c
// ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/amdgpu_amdkfd_gfx_v9.c
// 行 222-299

int kgd_gfx_v9_hqd_load(struct amdgpu_device *adev, void *mqd,
                        uint32_t pipe_id, uint32_t queue_id,
                        uint32_t __user *wptr, uint32_t wptr_shift,
                        uint32_t wptr_mask, struct mm_struct *mm,
                        uint32_t inst)
{
    struct v9_mqd *m;
    uint32_t *mqd_hqd;
    uint32_t reg, hqd_base, data;
    
    m = get_mqd(mqd);
    
    // 1. 获取访问权限（锁定 SRBM）
    kgd_gfx_v9_acquire_queue(adev, pipe_id, queue_id, inst);
    
    // 2. 写入所有 CP_HQD_* 寄存器
    mqd_hqd = &m->cp_mqd_base_addr_lo;
    hqd_base = SOC15_REG_OFFSET(GC, GET_INST(GC, inst), mmCP_MQD_BASE_ADDR);
    
    for (reg = hqd_base;
         reg <= SOC15_REG_OFFSET(GC, GET_INST(GC, inst), mmCP_HQD_PQ_WPTR_HI);
         reg++)
        WREG32_XCC(reg, mqd_hqd[reg - hqd_base], inst);
    
    // 3. 激活 Doorbell
    data = REG_SET_FIELD(m->cp_hqd_pq_doorbell_control,
                         CP_HQD_PQ_DOORBELL_CONTROL, DOORBELL_EN, 1);
    WREG32_SOC15_RLC(GC, GET_INST(GC, inst), mmCP_HQD_PQ_DOORBELL_CONTROL, data);
    
    // 4. 激活 HQD
    data = REG_SET_FIELD(m->cp_hqd_active, CP_HQD_ACTIVE, ACTIVE, 1);
    WREG32_SOC15_RLC(GC, GET_INST(GC, inst), mmCP_HQD_ACTIVE, data);
    
    // 5. 释放访问权限
    kgd_gfx_v9_release_queue(adev, inst);
    
    return 0;
}
```

### 7.2 主要的 CP_HQD 寄存器

| 寄存器名称 | 作用 |
|-----------|------|
| `CP_MQD_BASE_ADDR` | MQD 基地址 |
| `CP_HQD_PQ_BASE` | Packet Queue 基地址 |
| `CP_HQD_PQ_RPTR` | Read Pointer |
| `CP_HQD_PQ_WPTR` | Write Pointer |
| `CP_HQD_PQ_DOORBELL_CONTROL` | Doorbell 控制 |
| `CP_HQD_ACTIVE` | 队列激活状态 |
| `CP_HQD_VMID` | VMID（虚拟内存 ID） |
| `CP_HQD_EOP_BASE_ADDR` | End-Of-Packet 基地址 |
| `CP_HQD_EOP_RPTR` | EOP Read Pointer |
| `CP_HQD_EOP_WPTR` | EOP Write Pointer |

### 7.3 SRBM (System Register Bus Manager) 锁机制

```c
// ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/amdgpu_amdkfd_gfx_v9.c
// 行 63-84

void kgd_gfx_v9_acquire_queue(struct amdgpu_device *adev, 
                               uint32_t pipe_id,
                               uint32_t queue_id, 
                               uint32_t inst)
{
    uint32_t mec = (pipe_id / adev->gfx.mec.num_pipe_per_mec) + 1;
    uint32_t pipe = (pipe_id % adev->gfx.mec.num_pipe_per_mec);
    
    // 锁定 SRBM，指定要访问的 (mec, pipe, queue)
    kgd_gfx_v9_lock_srbm(adev, mec, pipe, queue_id, 0, inst);
}

void kgd_gfx_v9_release_queue(struct amdgpu_device *adev, uint32_t inst)
{
    // 释放 SRBM 锁
    kgd_gfx_v9_unlock_srbm(adev, inst);
}
```

**SRBM 的作用**:
- 控制寄存器访问的路由
- 将寄存器读写操作定向到特定的 (mec, pipe, queue)
- 防止并发访问冲突
- 确保寄存器操作的原子性

---

## 8. 验证和调试方法

### 8.1 查看 MEC 配置

**方法 1: dmesg 查看初始化日志**

```bash
$ sudo dmesg | grep -i "mec\|pipe"

# 预期输出:
[    2.123456] [drm] amdgpu: num_mec=2
[    2.123457] [drm] amdgpu: num_pipe_per_mec=4
[    2.123458] [drm] amdgpu: num_queue_per_pipe=8
[    2.234567] [drm] kfd: num of pipes: 4
```

**方法 2: 通过 KFD 日志**

```bash
# 启用 KFD debug 日志
echo 'module kfd +p' | sudo tee /sys/kernel/debug/dynamic_debug/control

# 运行测试程序
./your_hip_program

# 查看日志
sudo dmesg | grep "num of pipes"
# 输出: [drm] kfd: num of pipes: 4
```

**方法 3: 查看源码配置**

```bash
# MI308X (gfx942, IP 9.4.2) 的配置在 gfx_v9_0.c 中
$ grep -n "num_pipe_per_mec\|num_queue_per_pipe" \
    /usr/src/amdgpu-*/amd/amdgpu/gfx_v9_0.c

# 输出:
2272:   adev->gfx.mec.num_pipe_per_mec = 4;
2273:   adev->gfx.mec.num_queue_per_pipe = 8;

# 还可以查看 MEC 数量配置（IP 9.4.2 对应第 2227 行）
$ grep -n "IP_VERSION(9, 4, 2)" /usr/src/amdgpu-*/amd/amdgpu/gfx_v9_0.c
2227:   case IP_VERSION(9, 4, 2):
2228:       adev->gfx.mec.num_mec = 2;
```

### 8.2 查看 MEC 固件版本

```bash
# 方法1: dmesg
$ dmesg | grep -i "mec.*firmware"

[    2.345678] [drm] MEC firmware version: 51 feature version: 51

# 方法2: debugfs
$ sudo cat /sys/kernel/debug/dri/0/amdgpu_firmware_info | grep -A2 "MEC"

MEC feature version: 51, firmware version: 0x00000033
MEC2 feature version: 0, firmware version: 0x00000000
```

### 8.3 验证双 MEC 使用情况 ⭐ **实战验证**

使用 UMR (User-Mode Register Debugger) 的 `-cpc` 选项可以直接查看两个 MEC 的实际使用情况：

```bash
# 安装 umr（如果没有）
# Ubuntu/Debian: apt install umr
# 或从源码编译: https://gitlab.freedesktop.org/tomstdenis/umr

# 查看 CPC (Command Processor Compute) 状态
$ sudo umr -cpc

# 输出示例:
ME 1 Pipe 0: INSTR_PTR 0x47a  INT_STAT_DEBUG 0x4000000
Pipe 0  Queue 2  VMID 8   ← 用户态 Compute 队列 (MEC 0)
  PQ BASE 0x7f6c61920000  RPTR 0x10  WPTR 0x10  RPTR_ADDR 0x7f6c61a04080
  EOP BASE 0x7f6c619be000  RPTR 0x40000000  WPTR 0x3f70000
  MQD 0xa02800  DEQ_REQ 0x0  IQ_TIMER 0x0  AQL_CONTROL 0x1   ← AQL 队列标记
  SAVE BASE 0x0  SIZE 0x0  STACK OFFSET 0x0  SIZE 0x0

ME 1 Pipe 1: INSTR_PTR 0x47a  INT_STAT_DEBUG 0x4000000
Pipe 1  Queue 2  VMID 8
  PQ BASE 0x7f6c619c0000  RPTR 0x20  WPTR 0x20
  EOP BASE 0x7f6c61a23000  RPTR 0x40000010  WPTR 0x3ff8010
  MQD 0xa01e00  AQL_CONTROL 0x1

ME 2 Pipe 0: INSTR_PTR 0x47a  INT_STAT_DEBUG 0x0
Pipe 0  Queue 0  VMID 0   ← 内核态特权队列 (MEC 1)
  PQ BASE 0xa00000  RPTR 0x94  WPTR 0x94  RPTR_ADDR 0xa01800
  EOP BASE 0xa00800  RPTR 0x40000000  WPTR 0x3ff8000
  MQD 0x10847dd1000  AQL_CONTROL 0x0   ← 非 AQL，是 PM4 命令队列
```

**关键观察点**：
| 字段 | MEC 0 (ME 1) | MEC 1 (ME 2) | 说明 |
|------|-------------|-------------|------|
| **VMID** | 8-15 | 0 | MEC 0 用于用户态，MEC 1 用于内核态 |
| **AQL_CONTROL** | 0x1 | 0x0 | MEC 0 运行 AQL 队列，MEC 1 运行 PM4 队列 |
| **MQD 地址** | 小地址 (VRAM) | 大地址 (系统内存) | 不同的内存分配策略 |
| **队列类型** | Compute 队列 | KIQ/HIQ 特权队列 | 功能完全隔离 |

### 8.4 验证 HQD 分配 ⭐

**重要**: HQD 分配日志默认关闭，需要通过 Dynamic Debug 启用。

#### 启用调试日志

```bash
# 方法 1: 使用提供的脚本（推荐）
cd scripts
sudo bash enable_kfd_debug.sh

# 方法 2: 手动启用
sudo su -c 'echo "file kfd_device_queue_manager.c line 992 +p" > /sys/kernel/debug/dynamic_debug/control'

# 验证是否启用
sudo grep "allocate_hqd" /sys/kernel/debug/dynamic_debug/control
# 应该看到 "=p" 标志（已启用）
```

#### 为什么看不到日志？ ⚠️

**关键原因**: **HIP/ROCm 在进程启动时创建 Queue 池，后续只是复用，不会触发新的 HQD 分配**

要看到 HQD 分配日志，需要在**首次创建 Queue 时**观察：

**方法 1: 系统启动后首次运行（最可靠）**

```bash
# 系统重启后
sudo dmesg -C
cd tests
./test_queue_creation
sudo dmesg | grep "hqd slot"
```

**方法 2: 重新加载 amdgpu 模块**

```bash
# ⚠️ 会中断所有 GPU 进程
sudo pkill -9 -f rocm
sudo dmesg -C
sudo modprobe -r amdgpu && sudo modprobe amdgpu
./test_queue_creation
sudo dmesg | grep "hqd slot"
```

**输出示例**:

```bash
$ sudo dmesg | grep "hqd slot"
[  123.456] kfd: hqd slot - pipe 0, queue 0
[  123.457] kfd: hqd slot - pipe 1, queue 0
[  123.458] kfd: hqd slot - pipe 2, queue 0
[  123.459] kfd: hqd slot - pipe 3, queue 0
```

#### 调试工具

```bash
# 完整的诊断流程
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit

# 1. 启用调试日志
sudo bash scripts/enable_kfd_debug.sh

# 2. 运行测试程序
cd tests
./test_queue_creation

# 3. 查看日志（可能为空，见上述原因）
sudo dmesg | grep "hqd slot"

# 4. 禁用调试（可选）
cd ..
sudo bash scripts/disable_kfd_debug.sh
```

### 8.5 计算总队列数

```python
#!/usr/bin/env python3
# calculate_hqd_count.py

def calculate_hqd_count(num_mec, num_pipe_per_mec, num_queue_per_pipe):
    """计算 HQD 总数"""
    hqd_per_mec = num_pipe_per_mec * num_queue_per_pipe
    total_hqd = num_mec * hqd_per_mec
    
    print(f"MEC 配置:")
    print(f"  num_mec = {num_mec}")
    print(f"  num_pipe_per_mec = {num_pipe_per_mec}")
    print(f"  num_queue_per_pipe = {num_queue_per_pipe}")
    print(f"\n计算结果:")
    print(f"  每个 MEC 的 HQD 数 = {hqd_per_mec}")
    print(f"  总 HQD 数（理论）= {total_hqd}")
    print(f"  KFD 可用 HQD 数 = {hqd_per_mec} (只使用 MEC 0)")

# MI308X 配置
calculate_hqd_count(num_mec=2, num_pipe_per_mec=4, num_queue_per_pipe=8)

# 输出:
# MEC 配置:
#   num_mec = 2
#   num_pipe_per_mec = 4
#   num_queue_per_pipe = 8
# 
# 计算结果:
#   每个 MEC 的 HQD 数 = 32
#   总 HQD 数（理论）= 64
#   KFD 可用 HQD 数 = 32 (只使用 MEC 0)
```

---

## 9. MEC 相关的常见问题

### 9.1 为什么用户态只能创建 32 个 Compute Queue？

**答**: 
- 硬件上有 2 个 MEC，每个 32 个 HQD，共 64 个
- **MEC 0** (32 HQDs) 专门用于**用户态 Compute 队列**（VMID 8-15，AQL 队列）
- **MEC 1** (32 HQDs) 专门用于**内核态特权队列**（VMID 0，KIQ/HIQ）
- 这是功能隔离的设计决策，不是硬件限制

### 9.2 MEC 和 CP 是什么关系？

**答**:
```
CP (Command Processor) = 泛指 GPU 的命令处理器
  ├─ Graphics Engine (ME/PFP/CE) - 图形命令处理
  └─ Compute Engine (MEC) - 计算命令处理 ⭐

MEC 是 CP 的一种，专门用于 Compute 工作负载
```

### 9.3 Pipe 和 CU (Compute Unit) 的关系？

**答**:
- **Pipe** 是队列管理层面的概念（软件/固件层）
- **CU** 是实际执行单元（硬件层）
- 没有直接的 1:1 映射关系
- 所有 Pipe 的 Queue 都共享相同的 CU 资源

```
Pipe（队列管理）     CU（执行单元）
├─ Pipe 0           ┌─ CU 0
├─ Pipe 1           ├─ CU 1
├─ Pipe 2     →→→   ├─ ...
└─ Pipe 3           └─ CU 79
    (管理层)            (执行层)
                      全部共享
```

### 9.4 如何知道某个 Queue 在哪个 MEC/Pipe/Queue？

**方法 1: 通过 KFD 日志**

```bash
sudo dmesg | grep "hqd slot"
# 输出: kfd: hqd slot - pipe 2, queue 3
```

**方法 2: 通过数据结构**

```c
struct queue {
    uint32_t mec;    // MEC ID (总是 0)
    uint32_t pipe;   // Pipe ID (0-3)
    uint32_t queue;  // Queue ID (0-7)
    // ...
};
```

**方法 3: 通过 procfs (如果可用)**

```bash
cat /sys/kernel/debug/kfd/proc/*/queues
```

### 9.5 MEC 1 完全不用吗？

**答**: 不完全是。
- KFD Compute Queues **不使用** MEC 1
- 但其他组件可能使用 MEC 1：
  - Graphics 队列
  - System 队列
  - 特殊用途队列
- 具体使用情况取决于驱动实现

---

## 10. MEC 在整个 Kernel 提交流程中的位置

```
【完整的 Kernel 提交流程】

应用层:
  hipLaunchKernel()
    ↓
HIP Runtime:
  准备 AQL packet
    ↓
HSA Runtime:
  写 packet 到 Queue
  写 Doorbell
    ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
硬件层:
    ↓
  Doorbell 通知
    ↓
  MEC 固件检测 ⭐
    │
    ├─ 从 Queue 读取 AQL packet
    ├─ 解析 packet 内容
    ├─ 分配 CU 资源
    └─ 调度 kernel 执行
         ↓
      CU 执行 kernel
         ↓
      写入 Completion Signal
         ↓
      触发中断（可选）
```

**MEC 的关键作用**:
1. ✅ 检测 Doorbell 信号
2. ✅ 从用户空间 Queue 读取 AQL packet
3. ✅ 解析和验证 packet
4. ✅ 分配和调度 CU 资源
5. ✅ 管理 kernel 执行
6. ✅ 处理完成信号

---

## 11. 总结

### 11.1 关键要点

| 概念 | 说明 |
|------|------|
| **MEC** | GPU 中的计算微引擎，负责管理计算队列 |
| **双 MEC** | MI308X 有 2 个 MEC，各司其职 |
| **MEC 0** | 用于用户态 Compute 队列（32个 HQDs，VMID 8-15） |
| **MEC 1** | 用于内核态特权队列（32个 HQDs，VMID 0，KIQ/HIQ） |
| **XCD** | eXtended Compute Die（物理芯片），数量未在代码中确认 ⚠️ |
| **XCC** | eXtended Compute Core，MI308X 每个芯片有 **8 个逻辑 XCC** ✅ |
| **XCP** | XCC Partition，驱动层的软件抽象，管理 XCC |
| **分区模式** | SPX/DPX/TPX/QPX/CPX，灵活配置 8 个 XCC 的使用方式 |
| **DRI 设备** | 8 GPU × 8 XCC × 2 节点类型 = 128 个 DRI 设备（+1 显卡）✅ |
| **Compute Units** | 每个 MI308X 有 80 个 CU（系统观察）✅ |
| **4 Pipes** | 每个 MEC 有 4 个 Pipe，用于负载均衡 |
| **8 Queues** | 每个 Pipe 有 8 个 Queue 槽位 |
| **固件驱动** | MEC 运行独立的微代码固件 |
| **寄存器访问** | 通过 SRBM 锁定后访问 CP_HQD_* 寄存器 |

### 11.2 MEC 架构优势

```
✅ 专用硬件: 专门优化的计算命令处理
✅ 并行调度: 多个 Pipe 可并行处理
✅ 硬件隔离: 每个 Queue 独立的寄存器组
✅ 低延迟: 固件直接控制，无需 OS 干预
✅ 高吞吐: 可同时管理 32 个活跃队列
```

### 11.3 理解 MEC 的重要性

理解 MEC 架构对以下工作至关重要：

1. **性能优化**: 了解队列分配策略
2. **问题诊断**: 理解队列状态和调度
3. **驱动开发**: 正确配置和管理队列
4. **架构研究**: 深入理解 GPU 计算架构

---

## 📚 参考资料

### 代码位置

**MEC 相关**:
- MEC 结构定义: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/amdgpu_gfx.h:102`
- MEC 配置 (MI308X/gfx942): `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/gfx_v9_0.c:2227-2273`
  - `IP_VERSION(9, 4, 2)` 在第 2227 行
  - `num_pipe_per_mec = 4` 在第 2272 行
  - `num_queue_per_pipe = 8` 在第 2273 行
- HQD 分配: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_device_queue_manager.c:965`
- HQD 加载: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/amdgpu_amdkfd_gfx_v9.c:222`

**XCD/XCP 相关**:
- XCP 数据结构: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/amdgpu_xcp.h`
  - `struct amdgpu_xcp_mgr` 定义在第 113 行
  - `struct amdgpu_xcp` 定义在第 98 行
  - `#define MAX_XCP 8` 在第 32 行
- XCP 管理函数: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/amdgpu_xcp.c`
- 分区模式定义: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/amdgpu_gfx.h:63-72`
- Aqua Vanjaram (MI308X) XCP 实现: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdgpu/aqua_vanjaram.c`
- KFD XCD 支持: `ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_device.c:710-740`

### 相关文档

- [KERNEL_TRACE_CPSCH_MECHANISM.md](./KERNEL_TRACE_CPSCH_MECHANISM.md) - CPSCH 调度器机制
- [KERNEL_TRACE_03_KFD_QUEUE.md](./KERNEL_TRACE_03_KFD_QUEUE.md) - KFD Queue 管理
- [KERNEL_TRACE_04_MES_HARDWARE.md](./KERNEL_TRACE_04_MES_HARDWARE.md) - MES vs CPSCH 对比
- [AQL定义详解.md](./AQL定义详解.md) - AQL Packet 格式

### AMD 官方文档

- AMD GPU 架构白皮书
- ROCm 文档: https://rocm.docs.amd.com/
- HSA Runtime 规范

---

**文档版本**: v2.0  
**最后更新**: 2026-01-19  
**适用ROCm版本**: 6.x  
**主要更新**: 新增 XCD/XCP 架构章节，解释 MI308X 的 127 个 DRI 设备  
**测试硬件**: MI308X (gfx942, IP 9.4.2)

