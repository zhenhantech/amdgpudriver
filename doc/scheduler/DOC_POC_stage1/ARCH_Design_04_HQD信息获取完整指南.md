# HQD (Hardware Queue Descriptor) 信息获取完整指南

**日期**: 2026-02-03  
**问题**: 在 KFD 中可以看到 HQD 的信息和状态吗？  
**答案**: ✅ **可以！通过 debugfs 接口**

---

## 🎯 快速回答

### 是的，KFD 提供了两个 debugfs 接口：

| 接口 | 路径 | 内容 | 粒度 |
|------|------|------|------|
| **MQD** | `/sys/kernel/debug/kfd/mqds` | 软件队列描述符 | 进程级 |
| **HQD** | `/sys/kernel/debug/kfd/hqds` | 硬件队列寄存器 | 硬件级 |

---

## 📊 MQD vs HQD 对比

### MQD (Memory Queue Descriptor) - 软件层

**定义**: KFD 内核驱动维护的软件队列描述符

**信息**:
```bash
sudo cat /sys/kernel/debug/kfd/mqds

# 输出示例:
Compute queue on device 0001:01:00.0
    Queue ID: 1 (0x1)
    Address: 0x7f8c00000000
    Process: pid 15234 pasid 0x8001
    is active: yes         ← 软件层的 active 状态
    priority: 7
    queue count: 1
```

**包含字段**:
- ✅ Queue ID (用户态队列 ID)
- ✅ Process PID (进程 ID)
- ✅ Priority (优先级)
- ✅ is active (软件认为队列是否活跃)
- ✅ Queue address (队列地址)

**用途**: 
- 查找进程的队列
- 获取 Queue ID 用于抢占
- 检查软件层状态

---

### HQD (Hardware Queue Descriptor) - 硬件层

**定义**: GPU 硬件队列寄存器的快照

**信息**:
```bash
sudo cat /sys/kernel/debug/kfd/hqds

# 输出示例:
 Inst 0,  CP Pipe 0, Queue 1
    0000c914: 006ed000 00000000 60004032 00000303 ...
                                ^^^^^^^^ 
                                CP_HQD_ACTIVE (0x1247)
                                bit[0]=0 → 队列非活跃
                                bit[0]=1 → 队列活跃 ✅
```

**包含内容**:
- ✅ 56 个硬件寄存器的值
- ✅ **CP_HQD_ACTIVE** (0x1247) - 活跃状态寄存器
- ✅ CP_HQD_VMID - 虚拟内存 ID
- ✅ CP_HQD_PQ_RPTR/WPTR - Ring Buffer 读写指针
- ✅ CP_HQD_IB_BASE_ADDR - Indirect Buffer 地址
- ✅ 其他控制寄存器

**用途**: 
- 检查硬件层真实状态
- 验证队列是否真正在 GPU 上运行
- 调试硬件问题

---

## 🔑 关键寄存器：CP_HQD_ACTIVE

### 寄存器位置

**地址**: `0x1247` (mmCP_HQD_ACTIVE)  
**在 HQD dump 中的位置**: 第 1 行，第 3 个寄存器 (index=2)

### Bit 定义

```c
// gc_9_0_sh_mask.h
#define CP_HQD_ACTIVE__ACTIVE__SHIFT    0x0
#define CP_HQD_ACTIVE__ACTIVE_MASK      0x00000001L
#define CP_HQD_ACTIVE__BUSY_GATE__SHIFT 0x1
#define CP_HQD_ACTIVE__BUSY_GATE_MASK   0x00000002L
```

**判断方法**:
- **bit[0] = 1** → 队列活跃 ✅
- **bit[0] = 0** → 队列非活跃 ❌

**示例值**:
```
0x60004032 → bit[0]=0 → 非活跃
0x00000001 → bit[0]=1 → 活跃 ✅
0x6000402a → bit[0]=0 → 非活跃
```

---

## 📐 完整的 HQD 信息读取方法

### 方法 1: Shell 脚本读取 (⭐⭐⭐⭐⭐ 最简单)

```bash
#!/bin/bash
# count_active_hqd.sh

HQD_FILE="/sys/kernel/debug/kfd/hqds"

active_count=0
total_count=0

while IFS= read -r line; do
    # 检测队列标识行
    if [[ $line =~ "CP Pipe" ]]; then
        ((total_count++))
        
        # 读取下一行（第一行HQD数据）
        read -r hqd_line
        
        # 提取第3个十六进制数字（CP_HQD_ACTIVE）
        # 格式: "    0000c914: 006ed000 00000000 60004032 ..."
        #                                     ^^^^^^^^ 第3个
        hqd_active=$(echo "$hqd_line" | awk '{print $4}')
        
        # 检查 bit[0]
        if [ -n "$hqd_active" ]; then
            # 转换为十进制并检查最低位
            dec_value=$((16#$hqd_active))
            if [ $((dec_value & 0x1)) -eq 1 ]; then
                ((active_count++))
                echo "✅ Active: Inst $(echo $line | grep -oP 'Inst \K\d+'), Pipe $(echo $line | grep -oP 'Pipe \K\d+'), Queue $(echo $line | grep -oP 'Queue \K\d+')"
            fi
        fi
    fi
done < "$HQD_FILE"

echo ""
echo "📊 统计结果:"
echo "  总 HQD 数:   $total_count"
echo "  活跃 HQD:    $active_count"
echo "  非活跃 HQD:  $((total_count - active_count))"
echo "  活跃率:      $((active_count * 100 / total_count))%"
```

---

### 方法 2: Python 读取 (⭐⭐⭐⭐ 适合集成)

```python
#!/usr/bin/env python3
# hqd_reader.py

import re
from dataclasses import dataclass
from typing import List

@dataclass
class HQDInfo:
    inst: int          # GPU instance
    pipe: int          # CP Pipe
    queue: int         # Queue slot
    cp_hqd_active: int # CP_HQD_ACTIVE 寄存器值
    is_active: bool    # bit[0] 是否为 1
    all_regs: List[int] # 所有56个寄存器

def parse_hqds(hqd_path="/sys/kernel/debug/kfd/hqds") -> List[HQDInfo]:
    """解析 HQD debugfs 文件"""
    
    hqds = []
    
    with open(hqd_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 检测队列标识行: " Inst 0,  CP Pipe 0, Queue 1"
        m = re.match(r'\s*Inst\s+(\d+),\s+CP Pipe\s+(\d+),\s+Queue\s+(\d+)', line)
        if m:
            inst = int(m.group(1))
            pipe = int(m.group(2))
            queue = int(m.group(3))
            
            # 读取寄存器行（通常是7行，每行8个寄存器）
            regs = []
            i += 1
            while i < len(lines) and not re.match(r'\s*Inst\s+\d+', lines[i]):
                reg_line = lines[i].strip()
                if ':' in reg_line:
                    # 解析十六进制值
                    hex_values = reg_line.split(':')[1].strip().split()
                    for hex_val in hex_values:
                        try:
                            regs.append(int(hex_val, 16))
                        except ValueError:
                            pass
                i += 1
            
            # CP_HQD_ACTIVE 是第3个寄存器 (index=2)
            cp_hqd_active = regs[2] if len(regs) > 2 else 0
            is_active = (cp_hqd_active & 0x1) == 1
            
            hqd = HQDInfo(
                inst=inst,
                pipe=pipe,
                queue=queue,
                cp_hqd_active=cp_hqd_active,
                is_active=is_active,
                all_regs=regs
            )
            hqds.append(hqd)
        else:
            i += 1
    
    return hqds


def count_active_hqds(hqds: List[HQDInfo]) -> dict:
    """统计活跃 HQD"""
    
    total = len(hqds)
    active = sum(1 for h in hqds if h.is_active)
    inactive = total - active
    
    # 按 Inst 分组统计
    by_inst = {}
    for h in hqds:
        if h.inst not in by_inst:
            by_inst[h.inst] = {'total': 0, 'active': 0}
        by_inst[h.inst]['total'] += 1
        if h.is_active:
            by_inst[h.inst]['active'] += 1
    
    return {
        'total': total,
        'active': active,
        'inactive': inactive,
        'by_inst': by_inst
    }


# 使用示例
if __name__ == '__main__':
    import sys
    
    print("🔍 读取 HQD 信息...")
    
    try:
        hqds = parse_hqds()
    except PermissionError:
        print("❌ 权限不足，请使用 sudo 运行")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ /sys/kernel/debug/kfd/hqds 不存在")
        print("   请确认 KFD debugfs 已挂载")
        sys.exit(1)
    
    # 统计
    stats = count_active_hqds(hqds)
    
    print(f"\n📊 HQD 统计:")
    print(f"  总 HQD:      {stats['total']}")
    print(f"  活跃 HQD:    {stats['active']} ({stats['active']*100//stats['total']}%)")
    print(f"  非活跃 HQD:  {stats['inactive']}")
    
    print(f"\n📊 按 GPU Instance 分组:")
    for inst, data in sorted(stats['by_inst'].items()):
        print(f"  Inst {inst}: {data['active']}/{data['total']} active")
    
    # 列出活跃的 HQD
    print(f"\n✅ 活跃的 HQD 列表:")
    for h in hqds:
        if h.is_active:
            print(f"  Inst {h.inst}, Pipe {h.pipe}, Queue {h.queue}: "
                  f"CP_HQD_ACTIVE=0x{h.cp_hqd_active:08x}")
```

---

### 方法 3: C 代码读取 (⭐⭐⭐⭐⭐ 性能最优)

```c
// hqd_monitor.c
// 集成到 libgpreempt_poc.so

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define HQD_FILE "/sys/kernel/debug/kfd/hqds"
#define CP_HQD_ACTIVE_INDEX 2  // 第3个寄存器 (0-based)

typedef struct {
    int inst;
    int pipe;
    int queue;
    uint32_t cp_hqd_active;
    int is_active;
} hqd_info_t;

int gpreempt_read_hqd_status(hqd_info_t **hqds_out, int *count_out) {
    FILE *fp = fopen(HQD_FILE, "r");
    if (!fp) {
        perror("Failed to open hqds");
        return -1;
    }
    
    hqd_info_t *hqds = NULL;
    int capacity = 1024;
    int count = 0;
    
    hqds = malloc(capacity * sizeof(hqd_info_t));
    if (!hqds) {
        fclose(fp);
        return -1;
    }
    
    char line[1024];
    int inst, pipe, queue;
    uint32_t regs[8];
    
    while (fgets(line, sizeof(line), fp)) {
        // 检测队列标识行: " Inst 0,  CP Pipe 0, Queue 1"
        if (sscanf(line, " Inst %d, CP Pipe %d, Queue %d", 
                   &inst, &pipe, &queue) == 3) {
            
            // 读取下一行（第一行HQD数据）
            if (fgets(line, sizeof(line), fp)) {
                // 解析8个十六进制寄存器
                if (sscanf(line, " %*x: %x %x %x %x %x %x %x %x",
                          &regs[0], &regs[1], &regs[2], &regs[3],
                          &regs[4], &regs[5], &regs[6], &regs[7]) >= 3) {
                    
                    // CP_HQD_ACTIVE 是第3个 (index=2)
                    uint32_t cp_hqd_active = regs[CP_HQD_ACTIVE_INDEX];
                    int is_active = (cp_hqd_active & 0x1) ? 1 : 0;
                    
                    // 扩展数组
                    if (count >= capacity) {
                        capacity *= 2;
                        hqds = realloc(hqds, capacity * sizeof(hqd_info_t));
                    }
                    
                    // 保存信息
                    hqds[count].inst = inst;
                    hqds[count].pipe = pipe;
                    hqds[count].queue = queue;
                    hqds[count].cp_hqd_active = cp_hqd_active;
                    hqds[count].is_active = is_active;
                    count++;
                }
            }
        }
    }
    
    fclose(fp);
    
    *hqds_out = hqds;
    *count_out = count;
    return 0;
}

int gpreempt_count_active_hqds(int *active_out, int *total_out) {
    hqd_info_t *hqds;
    int count;
    
    if (gpreempt_read_hqd_status(&hqds, &count) < 0) {
        return -1;
    }
    
    int active = 0;
    for (int i = 0; i < count; i++) {
        if (hqds[i].is_active) {
            active++;
        }
    }
    
    *active_out = active;
    *total_out = count;
    
    free(hqds);
    return 0;
}

// 使用示例
int main() {
    int active, total;
    
    if (gpreempt_count_active_hqds(&active, &total) < 0) {
        return 1;
    }
    
    printf("📊 HQD 统计:\n");
    printf("  总 HQD:   %d\n", total);
    printf("  活跃 HQD: %d (%d%%)\n", active, active * 100 / total);
    
    return 0;
}
```

---

## 🔬 HQD 状态的深入理解

### MQD active ≠ HQD active

**关键发现** (参考 `HARDWARE_QUEUE_DISTRIBUTION_ANALYSIS.md`):

| 时间 | MQD Active | HQD Active | 差异 |
|------|-----------|-----------|------|
| 13:50:44 | 80 | 63 | -17 (-21%) |

**原因**:
1. **MQD active**: 软件层认为队列已分配并可用
2. **HQD active**: 硬件层队列真正在 GPU 上激活

**状态转换**:
```
MQD 创建 → MQD active=true (软件层)
    ↓ allocate_hqd()
    ↓ load_mqd()
    ↓ map_queues_cpsch()  发送 PM4 MAP_QUEUES
    ↓ CP Scheduler 处理
HQD 激活 → HQD active=true (硬件层，CP_HQD_ACTIVE bit[0]=1)
```

**中间状态**: 
- MQD 已创建，但 HQD 还未完全激活
- CP Scheduler 还在处理 MAP_QUEUES packet
- 或者队列在 Runlist 中等待调度

---

## 🎯 POC Stage 1 中如何使用

### 场景: 识别 Online/Offline 队列

**步骤 1: 启动 AI 模型**

```bash
# 终端 1: Offline 训练
python3 offline_training.py &
OFFLINE_PID=$!

# 终端 2: Online 推理
python3 online_inference.py &
ONLINE_PID=$!
```

**步骤 2: 通过 MQD 获取 Queue ID**

```python
# 在调度器中
from mqd_parser import find_queue_by_pid

# 获取 Offline 队列
offline_queues = find_queue_by_pid(OFFLINE_PID)
offline_queue_ids = [q.queue_id for q in offline_queues if q.is_active]

# 获取 Online 队列
online_queues = find_queue_by_pid(ONLINE_PID)
online_queue_ids = [q.queue_id for q in online_queues if q.is_active]

print(f"Offline Queue IDs: {offline_queue_ids}")
print(f"Online Queue IDs: {online_queue_ids}")
```

**步骤 3: 验证 HQD 活跃状态**

```python
from hqd_reader import parse_hqds, count_active_hqds

# 读取 HQD 状态
hqds = parse_hqds()

# 验证队列确实在硬件上运行
for qid in offline_queue_ids:
    # 需要通过某种方式将 Queue ID 映射到 (inst, pipe, queue)
    # 这个映射关系在 MQD 中可能没有直接提供
    # 需要进一步研究
    pass
```

---

## ⚠️ 当前限制和待解决问题

### 限制 1: MQD Queue ID → HQD (inst, pipe, queue) 映射

**问题**: 
- MQD 显示的 Queue ID (0, 1, 2, ...) 是用户态队列 ID
- HQD 显示的是 (Inst, Pipe, Queue) 硬件坐标
- 两者之间没有直接映射关系公开

**解决方案**:

**方案 A: 通过时间相关性** (简单但不精确)
```python
# 启动模型前后对比 HQD
hqds_before = parse_hqds()
# 启动模型
time.sleep(2)
hqds_after = parse_hqds()

# 新增的活跃 HQD 就是该模型使用的
new_hqds = [h for h in hqds_after if h.is_active 
            and h not in hqds_before]
```

**方案 B: 通过 KFD 内核代码获取** (精确但需要修改)
```c
// 在 struct queue 中添加字段
struct queue {
    ...
    int hardware_inst;  // GPU instance
    int hardware_pipe;  // CP Pipe
    int hardware_queue; // Queue slot
};

// 在 allocate_hqd() 中记录
q->hardware_inst = ...;
q->hardware_pipe = pipe;
q->hardware_queue = bit;

// 在 MQD debugfs 中导出
seq_printf(m, "    hardware: inst=%d pipe=%d queue=%d\n",
          q->hardware_inst, q->hardware_pipe, q->hardware_queue);
```

**方案 C: 只使用 MQD Queue ID** (推荐用于 POC Stage 1)
```python
# POC Stage 1 不需要精确的 HQD 映射
# 只需要 MQD Queue ID 就可以调用 suspend_queues

offline_queue_ids = [1, 2, 3]  # 从 MQD 获取
suspend_queues(offline_queue_ids)  # 直接使用 ✅
```

---

### 限制 2: HQD 信息更新频率

**问题**: `/sys/kernel/debug/kfd/hqds` 是快照，不是实时的

**影响**: 
- 读取时可能已经过时
- 队列状态可能在读取瞬间变化

**解决方案**: 
- 对于 POC，快照足够
- 对于生产，考虑内核态实时监控

---

## 📚 HQD 寄存器完整列表

### 关键寄存器 (前 16 个)

| Index | 地址 | 名称 | 用途 |
|-------|------|------|------|
| 0 | 0x1245 | mmCP_MQD_BASE_ADDR | MQD 基地址 |
| 1 | 0x1246 | mmCP_MQD_BASE_ADDR_HI | MQD 基地址高位 |
| 2 | 0x1247 | **mmCP_HQD_ACTIVE** | **活跃状态 ⭐** |
| 3 | 0x1248 | mmCP_HQD_VMID | 虚拟内存 ID |
| 4 | 0x1249 | mmCP_HQD_PERSISTENT_STATE | 持久化状态 |
| 5 | 0x124a | mmCP_HQD_PIPE_PRIORITY | Pipe 优先级 |
| 6 | 0x124b | mmCP_HQD_QUEUE_PRIORITY | Queue 优先级 |
| 7 | 0x124c | mmCP_HQD_QUANTUM | 时间片 |
| 8 | 0x124d | mmCP_HQD_PQ_BASE | Ring Buffer 基地址 |
| 9 | 0x124e | mmCP_HQD_PQ_BASE_HI | Ring Buffer 基地址高位 |
| 10 | 0x124f | mmCP_HQD_PQ_RPTR | **Ring Buffer 读指针 ⭐** |
| 11 | 0x1250 | mmCP_HQD_PQ_RPTR_REPORT_ADDR | rptr 报告地址 |
| 12 | 0x1251 | mmCP_HQD_PQ_RPTR_REPORT_ADDR_HI | rptr 报告地址高位 |
| 13 | 0x1252 | mmCP_HQD_PQ_WPTR_POLL_ADDR | wptr 轮询地址 |
| 14 | 0x1253 | mmCP_HQD_PQ_WPTR_POLL_ADDR_HI | wptr 轮询地址高位 |
| 15 | 0x1254 | mmCP_HQD_PQ_DOORBELL_CONTROL | **Doorbell 控制 ⭐** |

### 如何判断队列是否在运行？

**方法 1: CP_HQD_ACTIVE bit[0]** (是否活跃)
```python
is_active = (cp_hqd_active & 0x1) == 1
```

**方法 2: Ring Buffer 指针变化** (是否有工作)
```python
rptr = hqd.all_regs[10]  # mmCP_HQD_PQ_RPTR
wptr = read_wptr_from_doorbell_memory()  # 从 Doorbell 内存读取

has_pending_work = (rptr != wptr)
```

**方法 3: 组合判断** (最精确)
```python
def is_queue_truly_running(hqd):
    # 1. 硬件层必须活跃
    if not (hqd.cp_hqd_active & 0x1):
        return False
    
    # 2. Ring Buffer 有待处理的工作
    rptr = hqd.all_regs[10]
    wptr = read_wptr()
    if rptr == wptr:
        return False  # 没有待处理的 packet
    
    # 3. 或者轮询 rptr 是否变化
    time.sleep(0.001)
    rptr_new = read_hqd_rptr()
    if rptr_new != rptr:
        return True  # rptr 在移动，队列在运行
    
    return False
```

---

## 🎯 POC Stage 1 中的应用

### 场景 1: 识别活跃队列

```python
# 在 GPreemptScheduler 中

def update_queue_status(self):
    """更新所有队列的状态"""
    
    # 1. 读取 MQD (软件层)
    mqds = parse_mqd_debugfs()
    
    # 2. 读取 HQD (硬件层)
    hqds = parse_hqds()
    active_hqd_count = sum(1 for h in hqds if h.is_active)
    
    # 3. 对比 MQD 和 HQD
    mqd_active = sum(1 for m in mqds if m.is_active)
    
    print(f"MQD active: {mqd_active}")
    print(f"HQD active: {active_hqd_count}")
    print(f"差异: {mqd_active - active_hqd_count}")
    
    # 4. 只操作 MQD active 的队列（足够了）
    self.offline_queues = [m for m in mqds 
                          if m.priority <= 5 and m.is_active]
```

---

### 场景 2: 验证抢占是否生效

```python
# 抢占前
hqds_before = parse_hqds()
active_before = sum(1 for h in hqds_before if h.is_active)

# 执行抢占
suspend_queues(offline_queue_ids)

# 抢占后
time.sleep(0.1)
hqds_after = parse_hqds()
active_after = sum(1 for h in hqds_after if h.is_active)

# 验证
if active_after < active_before:
    print(f"✅ 抢占成功！HQD active 从 {active_before} 降到 {active_after}")
else:
    print(f"❌ 抢占失败或未生效")
```

---

## 📊 实际数据示例

根据之前的测试数据：

### MQD 快照示例

```
Total queues in MQD: 80
Active queues:       80  (100%)
  - Priority 2:      76 queues (Offline AI)
  - Priority 7:      4 queues  (System)
```

### HQD 快照示例

```
Total HQD slots:     960  (32 XCC × 30 slots/XCC)
Active HQD:          63   (6.6%)
  - CP Queues:       63
  - HIQ:             8    (系统队列)
```

**观察**: 
- 80 个 MQD active，但只有 63 个 HQD active
- 差异 21% (-17 个队列)
- 说明有些 MQD 虽然软件层认为 active，但硬件层还未完全激活

---

## 🛠️ 实用工具脚本

### 快速查看 HQD 活跃数

```bash
#!/bin/bash
# quick_hqd_count.sh

sudo cat /sys/kernel/debug/kfd/hqds | \
  awk '/CP Pipe/ {
      getline;  # 读取寄存器行
      split($0, regs, " ");
      # 第4个字段是 CP_HQD_ACTIVE (regs[4])
      if (regs[4] != "") {
          cmd = "printf \"%d\" $((" regs[4] " & 0x1))";
          cmd | getline active;
          close(cmd);
          total++;
          if (active == 1) {
              active_count++;
          }
      }
  }
  END {
      printf "Total: %d, Active: %d (%.1f%%)\n", total, active_count, active_count*100.0/total
  }'
```

---

### 监控 HQD 变化

```bash
#!/bin/bash
# monitor_hqd_changes.sh

while true; do
    count=$(sudo bash quick_hqd_count.sh 2>/dev/null | grep -oP 'Active: \K\d+')
    echo "$(date '+%H:%M:%S') - Active HQDs: $count"
    sleep 1
done
```

---

## 📖 相关文档

### 详细技术分析

- `HARDWARE_QUEUE_DISTRIBUTION_ANALYSIS.md` - HQD 分布分析
- `MQD_HQD_MAPPING_ANALYSIS.md` - MQD↔HQD 映射关系
- `PRECISE_HQD_COUNTING_METHOD.md` - HQD 精确统计方法

### POC Stage 1

- `ARCH_Design_01_POC_Stage1_实施方案.md` - 整体方案
- `ARCH_Design_03_QueueID获取与环境配置.md` - Queue ID 获取

---

## ✅ 总结

### 问题：在 KFD 中可以看到 HQD 的信息和状态吗？

**答案**: ✅ **可以！**

**方法**:
1. **HQD 寄存器快照**: `/sys/kernel/debug/kfd/hqds`
2. **关键寄存器**: `CP_HQD_ACTIVE` (0x1247) bit[0]
3. **解析方法**: Shell/Python/C 脚本读取和解析

**用途**:
- ✅ 统计活跃 HQD 数量
- ✅ 验证队列是否真正在 GPU 上运行
- ✅ 调试 MQD ↔ HQD 映射问题
- ✅ 监控队列状态变化

**限制**:
- ⚠️ 是快照，不是实时
- ⚠️ MQD Queue ID 到 HQD (inst, pipe, queue) 的映射不直接提供
- ⚠️ 需要 root 权限

**POC Stage 1 建议**:
- ✅ **使用 MQD Queue ID 就足够**（用于 suspend_queues）
- ✅ HQD 信息用于验证和调试
- ✅ 不需要精确的 MQD↔HQD 映射

---

**最后更新**: 2026-02-03  
**维护者**: Zhehan

**下一步**: 实施 `libgpreempt_poc.so` 中的 `hqd_monitor.c` 模块 ✅
