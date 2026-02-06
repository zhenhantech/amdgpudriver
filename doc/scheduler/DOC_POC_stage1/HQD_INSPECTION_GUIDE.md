# KFD中查看HQD信息和状态指南

**日期**: 2026-02-05  
**问题**: 在KFD中可以看到HQD的信息和状态吗？  
**答案**: ✅ 可以！有多种方法

---

## 📋 HQD vs MQD

### 概念区分

| 概念 | 全称 | 位置 | 作用 |
|------|------|------|------|
| **MQD** | Memory Queue Descriptor | 内存（软件） | 软件维护的Queue描述符 |
| **HQD** | Hardware Queue Descriptor | GPU寄存器（硬件） | 硬件执行的Queue状态 |

### 关系

```
用户空间 (HIP/ROCr)
    ↓
┌─────────────────┐
│  MQD (内存)     │  ← KFD维护，软件可读
│  - Queue配置    │
│  - Doorbell地址 │
│  - Ring Buffer  │
└─────────────────┘
    ↓ (写入)
┌─────────────────┐
│  HQD (GPU寄存器)│  ← 硬件执行，通过MMIO读取
│  - Read Pointer │
│  - Write Pointer│
│  - Queue状态    │
└─────────────────┘
```

---

## 🔍 查看HQD的方法

### 方法1: sysfs/debugfs (`/sys/kernel/debug/kfd/hqds`) ⭐⭐⭐⭐⭐

**最直接、最全面的方法**

#### 查看所有HQD

```bash
sudo cat /sys/kernel/debug/kfd/hqds
```

**输出示例**:
```
Node 0, GPU 0, Queue 0
  CP_HQD_VMID = 0x00000001
  CP_HQD_PQ_BASE = 0x00000001a97c0000
  CP_HQD_PQ_BASE_HI = 0x00007f3d
  CP_HQD_PQ_RPTR = 0x000007d0
  CP_HQD_PQ_WPTR = 0x000007d0
  CP_HQD_PQ_CONTROL = 0x02040001
  CP_HQD_IB_CONTROL = 0xffc10008
  CP_HQD_ACTIVE = 0x00000001        ← Queue是否活跃
  CP_HQD_QUANTUM = 0x00000200
  ...
```

#### 解析关键字段

| 字段 | 说明 | 重要性 |
|------|------|--------|
| `CP_HQD_ACTIVE` | Queue是否活跃 (1=运行中) | ⭐⭐⭐⭐⭐ |
| `CP_HQD_PQ_RPTR` | Read Pointer（GPU读到哪里）| ⭐⭐⭐⭐⭐ |
| `CP_HQD_PQ_WPTR` | Write Pointer（CPU写到哪里）| ⭐⭐⭐⭐⭐ |
| `CP_HQD_VMID` | 虚拟内存ID | ⭐⭐⭐ |
| `CP_HQD_QUANTUM` | 时间片配额 | ⭐⭐⭐⭐ |

#### 判断Queue状态

```bash
# 检查Queue是否活跃
sudo cat /sys/kernel/debug/kfd/hqds | grep "CP_HQD_ACTIVE"

# 查看Read/Write Pointer差异（判断是否有积压）
sudo cat /sys/kernel/debug/kfd/hqds | grep -E "CP_HQD_PQ_(RPTR|WPTR)"

# 示例输出:
# CP_HQD_PQ_RPTR = 0x000007d0  ← GPU已读取到这里
# CP_HQD_PQ_WPTR = 0x000007d5  ← CPU已写到这里
# 差值 = 5 个命令还在Queue中
```

---

### 方法2: AMD日志（`AMD_LOG_LEVEL=5`）⭐⭐⭐⭐

**从ROCr运行时日志中提取HQD信息**

```bash
export AMD_LOG_LEVEL=5
python3 your_test.py 2>&1 | tee test.log
```

**日志中的HQD信息**:

```
:4:rocvirtual.cpp :1177: 175228597956 us: [pid:157801 tid: 0x7fb0621f8480] 
SWq=0x7faf945b8000,    ← Software Queue地址
HWq=0x7fad66c00000,    ← Hardware Queue地址 ⭐⭐⭐⭐⭐
id=1,                  ← Queue ID
Dispatch Header = 0xb02 (type=2, barrier=1, acquire=1, release=1), 
setup=3, 
grid=[20480, 1, 1], 
workgroup=[256, 1, 1], 
private_seg_size=0, 
group_seg_size=30528, 
kernel_obj=0x7f8d40c960c0, 
kernarg_address=0x7fad66600000, 
completion_signal=0x0, 
correlation_id=0, 
rptr=255297,           ← Read Pointer ⭐⭐⭐⭐⭐
wptr=255297            ← Write Pointer ⭐⭐⭐⭐⭐
```

**提取HQD地址和ID**:

```bash
# 提取所有HQD地址
grep 'HWq=' test.log | grep -o 'HWq=0x[0-9a-f]*' | sort -u

# 提取Queue ID
grep 'HWq=.*id=' test.log | grep -o 'id=[0-9]*' | sort -u

# 提取Read/Write Pointer
grep 'rptr=.*wptr=' test.log | sed 's/.*rptr=\([0-9]*\).*wptr=\([0-9]*\).*/rptr=\1, wptr=\2/'
```

---

### 方法3: KFD Debug API (IOCTLs) ⭐⭐⭐⭐

**通过KFD Debug Trap接口获取Queue快照**

#### 使用已有工具

```bash
# 使用我们的queue_monitor工具
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

sudo ./queue_monitor <PID>
```

**输出**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPU Queue 监控 (PID: 12345)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Queue ID    Size      Read Ptr   Write Ptr   Ring WPTR  Ring Size   Priority
      1     1024      000007d0   000007d5    000007d5      4096      High
      2     2048      00001234   00001234    00001234      8192      Normal
```

#### C API示例

```c
#include <linux/kfd_ioctl.h>

// 获取Queue快照
struct kfd_ioctl_dbg_trap_get_queue_snapshot_args args = {0};
args.exception_mask = KFD_EC_MASK(KFD_EC_QUEUE_NEW);
args.max_queues = 64;

kfd_queue_snapshot_entry *queue_entries = malloc(
    args.max_queues * sizeof(kfd_queue_snapshot_entry)
);
args.queue_entries_ptr = (__u64)queue_entries;

if (ioctl(kfd_fd, KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT, &args) == 0) {
    for (int i = 0; i < args.num_queues; i++) {
        printf("Queue ID: %u\n", queue_entries[i].queue_id);
        printf("  Read Ptr:  0x%llx\n", queue_entries[i].read_pointer);
        printf("  Write Ptr: 0x%llx\n", queue_entries[i].write_pointer);
        printf("  Size:      %u\n", queue_entries[i].queue_size);
        printf("  Priority:  %u\n", queue_entries[i].ctx_priority);
    }
}
```

---

### 方法4: rocm-smi ⭐⭐⭐

**虽然不显示HQD寄存器，但可以看到Queue使用情况**

```bash
# 查看进程的Queue使用
rocm-smi --showpids

# 示例输出:
# GPU[0]: PID 12345
#   Name: python3
#   Compute queues: 2    ← 使用了2个Compute Queue
#   DMA queues: 1        ← 使用了1个DMA Queue
```

---

## 📊 HQD状态解析

### HQD寄存器字段详解

#### CP_HQD_ACTIVE

```
值: 0x00000001
含义: 
  - Bit 0 = 1: Queue活跃（正在执行）
  - Bit 0 = 0: Queue空闲

判断:
  如果ACTIVE=1，说明Queue正在处理命令
```

#### CP_HQD_PQ_RPTR / CP_HQD_PQ_WPTR

```
CP_HQD_PQ_RPTR = 0x000007d0  (2000)   ← GPU已读
CP_HQD_PQ_WPTR = 0x000007d5  (2005)   ← CPU已写

积压 = WPTR - RPTR = 5 个命令

状态判断:
  - WPTR == RPTR: Queue空闲（没有待处理命令）
  - WPTR > RPTR: Queue忙碌（有命令在队列中）
  - WPTR - RPTR 很大: Queue积压严重
```

#### CP_HQD_QUANTUM

```
值: 0x00000200 (512)
含义: 时间片配额（以时钟周期计）

判断:
  - 大值: 长时间片（适合长任务）
  - 小值: 短时间片（适合交互式任务）
```

---

## 🎯 实战示例

### 示例1: 监控Queue是否运行

```bash
#!/bin/bash
# monitor_hqd_activity.sh

while true; do
    echo "=== $(date) ==="
    
    # 提取ACTIVE状态
    sudo cat /sys/kernel/debug/kfd/hqds | grep -A 20 "Queue 0" | grep "CP_HQD_ACTIVE"
    
    # 提取Read/Write Pointer
    sudo cat /sys/kernel/debug/kfd/hqds | grep -A 20 "Queue 0" | grep -E "CP_HQD_PQ_(RPTR|WPTR)"
    
    echo ""
    sleep 1
done
```

**输出**:
```
=== Wed Feb  5 14:30:00 CST 2026 ===
  CP_HQD_ACTIVE = 0x00000001
  CP_HQD_PQ_RPTR = 0x000007d0
  CP_HQD_PQ_WPTR = 0x000007d5

=== Wed Feb  5 14:30:01 CST 2026 ===
  CP_HQD_ACTIVE = 0x00000001
  CP_HQD_PQ_RPTR = 0x000007d8    ← RPTR增加了
  CP_HQD_PQ_WPTR = 0x000007dd    ← WPTR也增加了
  → Queue正在处理命令！
```

### 示例2: 计算Queue积压

```bash
#!/bin/bash
# calculate_queue_backlog.sh

RPTR=$(sudo cat /sys/kernel/debug/kfd/hqds | grep "CP_HQD_PQ_RPTR" | head -1 | awk '{print $3}')
WPTR=$(sudo cat /sys/kernel/debug/kfd/hqds | grep "CP_HQD_PQ_WPTR" | head -1 | awk '{print $3}')

# 转换为十进制
RPTR_DEC=$((RPTR))
WPTR_DEC=$((WPTR))

BACKLOG=$((WPTR_DEC - RPTR_DEC))

echo "Read Pointer:  $RPTR ($RPTR_DEC)"
echo "Write Pointer: $WPTR ($WPTR_DEC)"
echo "积压命令数:     $BACKLOG"

if [ $BACKLOG -eq 0 ]; then
    echo "状态: Queue空闲"
elif [ $BACKLOG -lt 100 ]; then
    echo "状态: Queue轻度负载"
elif [ $BACKLOG -lt 500 ]; then
    echo "状态: Queue中度负载"
else
    echo "状态: Queue重度负载（可能积压）"
fi
```

### 示例3: 比较两个Case的HQD使用

```bash
# 1. 运行Case-A并记录HQD
./run_case_comparison.sh zhen_vllm_dsv3 60

# 2. 提取HQD信息
echo "=== Case-A HQD ==="
grep 'HWq=0x' log/case_comparison_*/case_a_cnn.log | head -10

echo "=== Case-B HQD ==="
grep 'HWq=0x' log/case_comparison_*/case_b_transformer.log | head -10

# 3. 对比Queue ID
echo "=== Queue ID对比 ==="
echo "Case-A:"
grep 'HWq=.*id=' log/case_comparison_*/case_a_cnn.log | grep -o 'id=[0-9]*' | sort -u

echo "Case-B:"
grep 'HWq=.*id=' log/case_comparison_*/case_b_transformer.log | grep -o 'id=[0-9]*' | sort -u
```

---

## 💡 常见问题

### Q1: HQD和MQD有什么区别？

**答**:
- **MQD**: 软件维护，在内存中，随时可读
- **HQD**: 硬件执行，在GPU寄存器中，通过MMIO读取

**关系**: MQD → (KFD写入) → HQD → (GPU执行)

### Q2: 为什么`lsof /dev/kfd`看不到但HQD存在？

**答**: ROCm 7.x可能使用了新的HSA/DRM接口，不再通过传统`/dev/kfd`设备文件，但HQD仍然存在并可以通过`debugfs`查看。

### Q3: HQD信息更新频率？

**答**: 
- **sysfs/debugfs**: 实时（每次读取时刷新）
- **AMD日志**: 每次Kernel提交时记录
- **KFD API**: 调用时快照

### Q4: 如何判断Queue是否被抢占了？

**答**:
1. **ACTIVE状态突然变为0**: Queue被暂停
2. **RPTR不再增加**: 没有新命令被处理
3. **Quantum超时**: 时间片用完被切换

---

## 📚 相关文档

- **AMD官方文档**: https://docs.kernel.org/gpu/amdgpu/driver-core.html
- **KFD IOCTL**: `/usr/src/amdgpu-6.12.12-2194681.el8_preempt/include/uapi/linux/kfd_ioctl.h`
- **HQD寄存器定义**: AMD GPU架构手册

---

## ✅ 总结

| 方法 | 可见性 | 实时性 | 易用性 | 推荐度 |
|------|--------|--------|--------|--------|
| `sysfs/debugfs` | 完整寄存器 | 实时 | 简单 | ⭐⭐⭐⭐⭐ |
| `AMD_LOG_LEVEL` | HQD地址+指针 | 事件触发 | 中等 | ⭐⭐⭐⭐ |
| `KFD Debug API` | Queue快照 | 按需 | 复杂 | ⭐⭐⭐⭐ |
| `rocm-smi` | Queue数量 | 实时 | 简单 | ⭐⭐⭐ |

**推荐组合**:
1. 日常监控: `sysfs/debugfs`
2. 详细调试: `AMD_LOG_LEVEL=5`
3. 程序化: `KFD Debug API`

---

**维护者**: AI Assistant  
**日期**: 2026-02-05
