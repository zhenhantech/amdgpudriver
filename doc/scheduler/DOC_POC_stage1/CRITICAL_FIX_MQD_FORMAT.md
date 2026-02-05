# 关键修正：MQD输出格式理解错误

**发现时间**: 2026-02-04  
**严重性**: 🔴🔴🔴 **严重**（整个实验设计基于错误假设）

---

## 🚨 问题发现

### 我的错误假设

我假设 `/sys/kernel/debug/kfd/mqds` 输出的是人类可读的文本格式，包含：
```
Queue ID: 0
pid 12345
priority: 0
is active: true
type: COMPUTE
```

### 实际输出格式

**实际上是二进制内存dump**:
```
Process 1616740 PASID 1616740:
  Compute queue on device f7bc
    00000000: c0310800 00004c24 00020000 00000001 00000001 00000000 00000000 00000000
    00000020: 00000100 00000001 00000001 00000001 00000000 3ced6407 0000007f 000012f2
    00000040: 00000000 00000000 00000000 00af0249 001c83da 00000000 00000000 ffffffff
    ...
```

**格式说明**:
- 每行前面是偏移量（00000000, 00000020, ...）
- 后面是32位十六进制值
- 这是MQD结构体的原始内存dump

---

## 🔍 正确的MQD结构理解

### MQD结构体定义

根据KFD代码，MQD是一个C结构体，例如 `v9_mqd` (GFX9系列):

```c
struct v9_mqd {
    uint32_t cp_hqd_pq_base;        // 0x00
    uint32_t cp_hqd_pq_base_hi;     // 0x04
    uint32_t cp_hqd_pq_rptr;        // 0x08
    uint32_t cp_hqd_pq_wptr;        // 0x0C
    uint32_t cp_hqd_pq_control;     // 0x10
    uint32_t cp_hqd_pq_doorbell;    // 0x14
    // ... 更多字段
};
```

### 你看到的dump解析

```
Process 1616740 PASID 1616740:
  ├─ Process ID: 1616740
  ├─ PASID: 1616740
  └─ 有一个Compute queue

  Compute queue on device f7bc:
    ├─ 设备ID: 0xf7bc
    └─ MQD内存dump（256+ 字节）
```

**关键发现**:
- ✅ 可以看到Process ID
- ✅ 可以看到队列类型（Compute queue）
- ✅ 可以看到设备ID
- ❌ **没有明确的"Queue ID"字段**
- ❌ **没有human-readable的队列属性**

---

## 💡 正确的队列识别方法

### 方法1: 通过Process计数 ⭐推荐

```bash
# 统计某个进程有多少个队列
grep -A 1 "Process $PID" /sys/kernel/debug/kfd/mqds | grep "queue on device" | wc -l
```

**示例**:
```bash
$ grep -A 10 "Process 1616740" /sys/kernel/debug/kfd/mqds
Process 1616740 PASID 1616740:
  Compute queue on device f7bc
    00000000: c0310800 00004c24 ...
    
# 输出说明这个进程有1个Compute queue
```

---

### 方法2: 解析MQD二进制数据（复杂）⚠️

需要了解MQD结构体的exact layout：

```python
import struct

def parse_mqd_dump(mqd_hex_lines):
    """解析MQD十六进制dump"""
    # 提取所有十六进制值
    values = []
    for line in mqd_hex_lines:
        if ':' in line:
            hex_values = line.split(':')[1].strip().split()
            for hv in hex_values:
                values.append(int(hv, 16))
    
    # 根据v9_mqd结构解析
    # 例如：offset 0x14是doorbell
    doorbell_offset = 0x14 // 4  # 转换为索引
    if len(values) > doorbell_offset:
        doorbell = values[doorbell_offset]
        print(f"Doorbell: 0x{doorbell:08x}")
```

**问题**: 需要exact的结构体定义，不同GPU可能不同

---

### 方法3: 使用其他debugfs文件

#### Option A: 查看进程的队列目录

```bash
# KFD可能有其他接口
ls -la /sys/kernel/debug/kfd/
```

可能的文件：
- `proc_info` - 进程信息
- `queue_info` - 队列信息
- `topology` - 拓扑信息

#### Option B: 通过HQD反推

```bash
# HQD输出更清晰，可以看到活跃的队列
sudo cat /sys/kernel/debug/kfd/hqds | grep -A 58 "HQD.*active"
```

---

## 🛠️ 修正后的实验方法

### 新实验设计：基于Process和Queue计数

```bash
#!/bin/bash
# exp01_queue_monitor_v2.sh

# 1. 获取测试进程PID
TEST_PID=$(docker exec zhenaiter ps aux | grep test_model | grep -v grep | awk '{print $2}')

# 2. 统计该进程的队列数量
count_queues() {
    local pid=$1
    # 方法1: 计算"queue on device"出现次数
    grep -A 1 "Process $pid" /sys/kernel/debug/kfd/mqds | \
        grep -c "queue on device"
}

# 3. 提取队列类型
get_queue_types() {
    local pid=$1
    grep -A 100 "Process $pid" /sys/kernel/debug/kfd/mqds | \
        grep "queue on device" | \
        awk '{print $1, $2}'  # 例如: "Compute queue"
}

# 4. 持续监控
for i in {1..10}; do
    echo "采样 $i:"
    
    NUM_QUEUES=$(count_queues $TEST_PID)
    echo "  队列数量: $NUM_QUEUES"
    
    QUEUE_TYPES=$(get_queue_types $TEST_PID)
    echo "  队列类型: $QUEUE_TYPES"
    
    echo ""
    sleep 10
done
```

---

### 新分析方法：基于MQD块计数

```python
#!/usr/bin/env python3
# analyze_mqd_v2.py

import re
from collections import defaultdict

def parse_mqd_file(filepath):
    """解析MQD文件（新方法）"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # 按Process分割
    process_blocks = re.split(r'Process \d+ PASID \d+:', content)
    
    queue_info = {}
    
    for block in process_blocks[1:]:  # 跳过第一个空块
        # 提取PID
        pid_match = re.search(r'Process (\d+) PASID', content)
        if not pid_match:
            continue
        
        pid = int(pid_match.group(1))
        
        # 统计队列数量（通过"queue on device"）
        queue_count = len(re.findall(r'(\w+) queue on device', block))
        
        # 提取队列类型
        queue_types = re.findall(r'(\w+) queue on device (\w+)', block)
        
        queue_info[pid] = {
            'count': queue_count,
            'types': queue_types
        }
    
    return queue_info

def analyze_process_queues(results_dir, target_pid):
    """分析特定进程的队列使用"""
    import glob
    
    mqd_files = sorted(glob.glob(f"{results_dir}/snapshot_mqd_*.txt"))
    
    print(f"🎯 目标进程PID: {target_pid}")
    print("")
    
    queue_counts = []
    
    for mqd_file in mqd_files:
        info = parse_mqd_file(mqd_file)
        
        if target_pid in info:
            count = info[target_pid]['count']
            types = info[target_pid]['types']
            queue_counts.append(count)
            
            print(f"采样: {count} 个队列")
            for qtype, device in types:
                print(f"  - {qtype} queue on device {device}")
        else:
            queue_counts.append(0)
            print(f"采样: 0 个队列（进程可能未初始化）")
    
    print("")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("总结:")
    print(f"  平均队列数: {sum(queue_counts)/len(queue_counts):.1f}")
    print(f"  最小: {min(queue_counts)}")
    print(f"  最大: {max(queue_counts)}")
    
    if min(queue_counts) == max(queue_counts):
        print("  ✅ 队列数量稳定")
    else:
        print("  ⚠️ 队列数量有变化")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("用法: python3 analyze_mqd_v2.py <results_dir> <pid>")
        sys.exit(1)
    
    analyze_process_queues(sys.argv[1], int(sys.argv[2]))
```

---

## 🎯 Queue ID的真相

### 关键发现：Queue ID可能不存在于MQD ⚠️

从你的输出看：
- MQD dump中**没有明确的Queue ID字段**
- 只有Process信息和队列内存dump

### Queue ID在哪里？

**推测**:
1. **在KFD内部**: Queue ID是内核内部的标识符
2. **在HQD中**: HQD可能有更多信息
3. **在其他debugfs**: 可能有专门的queue_info文件

### 验证方法

```bash
# 1. 检查所有KFD debugfs文件
ls -la /sys/kernel/debug/kfd/

# 2. 检查是否有queue相关的文件
find /sys/kernel/debug/kfd/ -name "*queue*"

# 3. 检查HQD输出
sudo cat /sys/kernel/debug/kfd/hqds | head -100
```

---

## 📊 对实验设计的影响

### 原计划 ❌

```
1. 从MQD提取Queue ID
2. 使用Queue ID调用suspend_queues(queue_id)
3. 验证抢占效果
```

### 新计划 ✅

```
1. 统计进程的队列数量（通过MQD dump中的"queue on device"）
2. 通过HQD找到活跃队列的硬件坐标
3. 或者使用process-level的IOCTL（如果存在）
4. 或者直接用debug IOCTL操作整个进程的队列
```

---

## 🔍 需要进一步调查

### 问题1: Debug IOCTL如何使用？

```c
// KFD_IOC_DBG_TRAP_SUSPEND_QUEUES
// 输入参数是什么？Queue ID还是其他？
```

需要查看：
- `kfd_ioctl.h` 中的结构体定义
- 已有的测试代码如何调用

### 问题2: 是否有其他方式获取Queue ID？

可能的来源：
- `/sys/class/kfd/` 下的文件
- procfs
- 内核日志（dmesg）

### 问题3: HQD格式是什么？

```bash
sudo cat /sys/kernel/debug/kfd/hqds | head -200
```

需要确认HQD是否也是二进制dump，还是文本格式

---

## 🚀 立即行动

### Step 1: 验证MQD格式理解

```bash
# 运行一个简单的GPU程序
docker exec zhenaiter python3 -c "import torch; x=torch.randn(100,100,device='cuda'); torch.cuda.synchronize(); import time; time.sleep(30)" &

# 获取PID
PID=$(docker exec zhenaiter ps aux | grep python3 | grep -v grep | awk '{print $2}')

# 查看该进程的MQD
sudo cat /sys/kernel/debug/kfd/mqds | grep -A 50 "Process $PID"
```

### Step 2: 检查HQD格式

```bash
sudo cat /sys/kernel/debug/kfd/hqds | head -100
```

### Step 3: 查找Queue ID来源

```bash
# 列出所有KFD debugfs文件
ls -la /sys/kernel/debug/kfd/

# 查找queue相关
find /sys/kernel/debug/kfd/ -type f -exec echo "=== {} ===" \; -exec head -20 {} \;
```

---

## 📚 需要查阅的代码

1. **MQD结构定义**:
   ```
   /usr/src/amdgpu-.../amd/amdkfd/kfd_mqd_manager*.c
   ```

2. **Debug IOCTL实现**:
   ```
   /usr/src/amdgpu-.../amd/amdkfd/kfd_debug.c
   ```

3. **Debugfs实现**:
   ```
   /usr/src/amdgpu-.../amd/amdkfd/kfd_debugfs.c
   ```

---

## 💡 关键教训

1. **不要假设输出格式**: 应该先验证实际输出
2. **二进制dump需要结构体定义**: 需要对应的C结构体才能解析
3. **Debug接口可能不完整**: debugfs可能不是为用户态使用设计的
4. **需要多种数据源**: 结合MQD, HQD, sysfs等多个来源

---

**状态**: 🔴 实验设计需要重新评估  
**下一步**: 调查正确的队列识别和操作方法  
**优先级**: 🔥🔥🔥 最高

这个发现改变了整个实验的基础！需要重新设计。
