# MI308X硬件信息（来自sysfs验证）

**日期**: 2026-02-04  
**来源**: `/sys/class/kfd/kfd/topology/nodes/2/properties`

---

## 核心配置 ⭐⭐⭐

```bash
# 调度器模式
enable_mes = 0              # ← MI308X使用CPSCH，不使用MES

# GPU架构
gfx_target_version = 90402  # GFX 9.4.2/9.4.3 (MI300系列)
num_xcc = 4                 # 4个XCC（eXtreme Compute Core）
device_id = 29858           # MI308X设备ID

# 计算资源
simd_count = 320            # 320个SIMD单元
array_count = 16            # 16个计算数组
cu_per_simd_array = 10      # 每个SIMD数组10个CU
simd_per_cu = 4             # 每个CU 4个SIMD
max_waves_per_simd = 8      # 每SIMD最大8个wave

# 队列配置
num_cp_queues = 120         # 总计120个CP队列 (30/XCC * 4)
num_sdma_engines = 2        # 2个标准SDMA引擎
num_sdma_xgmi_engines = 6   # 6个XGMI SDMA引擎
num_sdma_queues_per_engine = 8  # 每引擎8个队列

# 内存
lds_size_in_kb = 64         # 每CU 64KB LDS
num_gws = 64                # 64个GWS（Global Wave Sync）
wave_front_size = 64        # 64-wide wavefront
```

---

## 队列管理机制

### CPSCH模式（MI308X使用）✅

```
调度器: HWS (Hardware Scheduler) in CP Firmware
通信机制: HIQ (Hardware Interface Queue)
队列管理: Runlist IB
```

**验证命令**:
```bash
# 确认CPSCH模式
$ cat /sys/module/amdgpu/parameters/mes
0  # ← 0表示CPSCH，1表示MES

# 查看HIQ（CPSCH特有）
$ sudo cat /sys/kernel/debug/kfd/hqds | grep -i HIQ
# 应该有输出 → 确认使用CPSCH
```

### MES模式（MI308X不使用）❌

MI308X硬件不支持MES，代码中的MES路径用于更新GPU（RDNA3+/CDNA4+）。

---

## 计算资源分析

### 总计算能力

```
总SIMD: 320
总CU: 160 (16 array * 10 CU/array)
总Wave Slots: 2560 (320 SIMD * 8 wave/SIMD)
理论并行Wave: 2560个
```

### 每XCC资源

```
SIMD: 80 (320/4)
CU: 40 (160/4)
CP队列: 30 (120/4)
Wave Slots: 640 (2560/4)
```

---

## POC关键数据

### 队列配置

```
系统总队列数: 120个CP队列
  └─ 每GPU（8个物理GPU）: 15个队列
      └─ 每XCC: ~3-4个队列

实际测试观察（基于之前日志）:
  - 峰值MQD: 80个（10/GPU * 8GPU）
  - 峰值HQD: 288个（80 MQD * 4 XCC - 一些系统队列）
  
建议配置:
  - Online-AI: 预留2-4个队列/GPU
  - Offline-AI: 使用剩余队列
```

### SDMA配置

```
标准SDMA:
  - 2个引擎
  - 每引擎8个队列
  - 总计: 16个队列

XGMI SDMA:
  - 6个引擎（用于GPU间通信）
  - 每引擎8个队列
  - 总计: 48个队列

SDMA总计: 64个队列
```

---

## 性能参数

```
最大计算频率: 3800 MHz (max_engine_clk_ccompute)
GWS数量: 64 (用于跨Wave同步)
LDS: 64KB/CU
Wavefront宽度: 64 (CDNA2/3标准)
```

---

## POC设计建议

### 基于硬件配置的策略

1. **队列分配**
   ```
   每GPU 15个队列 (120/8):
     - Online: 预留2个 (13%)
     - Offline: 13个 (87%)
   ```

2. **XCC考虑**
   ```
   4个XCC共享队列资源
   需要考虑跨XCC调度均衡
   ```

3. **SDMA使用**
   ```
   数据传输使用SDMA队列
   与CP队列分离
   ```

---

## 验证信息准确性

```bash
# 方法1: 查看完整属性
cat /sys/class/kfd/kfd/topology/nodes/2/properties

# 方法2: 验证XCC数量
cat /sys/class/kfd/kfd/topology/nodes/*/properties | grep num_xcc

# 方法3: 确认CPSCH模式
cat /sys/module/amdgpu/parameters/mes

# 方法4: 查看所有GPU的队列数
for i in /sys/class/kfd/kfd/topology/nodes/*/properties; do 
    echo "$(dirname $i): $(grep num_cp_queues $i)"
done
```

---

## 相关文档

- `New_DEEP_DIVE_MI308X_QUEUE_MECHANISMS.md` - 队列管理机制深度分析
- `README_START_HERE.md` - POC快速入门
- `ARCH_Design_01_POC_Stage1_实施方案.md` - 实施方案

---

**最后更新**: 2026-02-04  
**验证状态**: ✅ 已通过sysfs验证  
**CPSCH模式**: ✅ 已确认（enable_mes=0）
