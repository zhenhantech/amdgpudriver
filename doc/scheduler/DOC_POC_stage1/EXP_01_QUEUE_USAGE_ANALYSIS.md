# 实验1: AI 模型队列使用分析实验

**日期**: 2026-02-04  
**目标**: 确定单个AI模型使用了哪些队列（MQD和HQD）  
**重要性**: ⭐⭐⭐⭐⭐ POC Stage 1 的核心前置实验  
**基于**: Map/Unmap机制研究成果

---

## 🎯 实验目标

### 核心问题

1. **一个AI模型会创建多少个队列？**
2. **这些队列在MQD中如何体现？**
3. **这些MQD映射到哪些HQD？**
4. **队列数量是否稳定？**（多次运行是否一致）
5. **不同模型的队列数量是否不同？**

### 实验意义

```
为POC Stage 1提供关键数据：
  ├─ 确定抢占粒度（按模型 vs 按队列）
  ├─ 设计队列识别策略
  ├─ 验证批量操作的可行性
  └─ 为性能预测提供参数
```

---

## 📐 实验设计

### 实验1.1: 单模型队列分析（基线）

**目标**: 确定单个模型的队列使用情况

#### 实验流程

```
┌─────────────────────────────────────────────────────────────┐
│ 步骤1: 系统基线                                              │
│ ════════════════════════════════════════════════════════════ │
│                                                              │
│ 1. 重启系统或卸载所有GPU进程                                 │
│ 2. 记录初始状态：                                            │
│    sudo cat /sys/kernel/debug/kfd/mqds > baseline_mqd.txt   │
│    sudo cat /sys/kernel/debug/kfd/hqds > baseline_hqd.txt   │
│ 3. 验证: 应该看到0个用户队列（只有系统队列）                 │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤2: 启动测试模型                                          │
│ ════════════════════════════════════════════════════════════ │
│                                                              │
│ 1. 启动一个标准的PyTorch测试程序                             │
│    - 模型: ResNet50 或简单的矩阵乘法                         │
│    - 长时间运行（2分钟）保证队列稳定                         │
│                                                              │
│ 2. 等待初始化（20秒）                                        │
│    - PyTorch需要时间初始化                                   │
│    - 队列创建需要时间                                        │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤3: 持续监控队列状态                                      │
│ ════════════════════════════════════════════════════════════ │
│                                                              │
│ 每10秒采样一次，共采样10次（100秒）：                        │
│                                                              │
│ 1. MQD快照：                                                 │
│    sudo cat /sys/kernel/debug/kfd/mqds > snapshot_mqd_$i.txt│
│                                                              │
│ 2. HQD快照：                                                 │
│    sudo cat /sys/kernel/debug/kfd/hqds > snapshot_hqd_$i.txt│
│                                                              │
│ 3. 进程信息：                                                │
│    ps aux | grep python > snapshot_ps_$i.txt                │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤4: 数据分析                                              │
│ ════════════════════════════════════════════════════════════ │
│                                                              │
│ 1. 提取该进程的所有MQD：                                     │
│    - Queue ID列表                                            │
│    - 每个Queue的属性（priority, type, active）               │
│                                                              │
│ 2. 提取对应的HQD：                                           │
│    - (Inst, Pipe, Queue)坐标                                 │
│    - HQD_ACTIVE状态                                          │
│                                                              │
│ 3. 验证MQD ↔ HQD映射关系：                                   │
│    - 1个MQD → 4个HQD (MI308X)                                │
│    - 坐标一致性                                              │
└─────────────────────────────────────────────────────────────┘
```

---

### 实验1.2: 队列一致性测试

**目标**: 验证队列分配的可预测性

#### 测试方案

```
测试A: 同一模型多次运行
  - 运行同一个模型5次
  - 每次记录队列数量和Queue ID
  - 验证一致性
  
  预期结果：
    ✅ 队列数量一致（例如都是2个）
    ⚠️ Queue ID可能不同（动态分配）
    ✅ HQD (pipe, queue)可能不同（Round-Robin）

测试B: 不同模型对比
  - 运行3种不同的模型：
    1. 简单矩阵乘法（单GPU）
    2. PyTorch ResNet50（单GPU）
    3. 多GPU训练（如果系统支持）
  
  预期结果：
    ✅ 不同模型可能使用不同数量的队列
    ✅ 单GPU模型通常使用较少队列
    ✅ 多GPU模型可能创建更多队列

测试C: 并发模型测试
  - 同时运行2个相同的模型
  - 验证队列是否重叠
  
  预期结果：
    ✅ 每个模型有独立的队列
    ❌ Queue ID不重叠
    ✅ 总队列数 = 模型1队列数 + 模型2队列数
```

---

### 实验1.3: 队列生命周期追踪

**目标**: 理解队列的创建和销毁时机

#### 追踪方案

```
阶段1: 模型启动前（T-10s）
  └─ 记录MQD/HQD基线

阶段2: 模型启动后（T+0s ~ T+30s）
  ├─ T+5s:  首次采样
  ├─ T+10s: 第二次采样
  ├─ T+20s: 第三次采样
  └─ T+30s: 稳定状态采样
  
  观察: 队列何时创建？逐步创建还是一次性？

阶段3: 模型运行中（T+30s ~ T+90s）
  └─ 每10秒采样
  
  观察: 队列数量是否变化？是否有inactive队列？

阶段4: 模型结束后（T+100s ~ T+110s）
  ├─ T+100s: 发送SIGTERM
  ├─ T+105s: 采样（进程结束中）
  └─ T+110s: 采样（进程已结束）
  
  观察: 队列何时销毁？是否立即销毁？
```

---

## 🛠️ 实验脚本

### 脚本1: 自动化队列监控脚本

```bash
#!/bin/bash
# exp01_queue_monitor.sh
# 在宿主机运行

set -e

CONTAINER="zhenaiter"
OUTPUT_DIR="./exp01_results"
DURATION=100  # 总监控时长（秒）
INTERVAL=10   # 采样间隔（秒）

echo "╔════════════════════════════════════════════════════════╗"
echo "║  实验1: AI模型队列使用分析                              ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# ========== 准备工作 ==========
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "步骤1: 记录系统基线"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

sudo cat /sys/kernel/debug/kfd/mqds > baseline_mqd.txt
sudo cat /sys/kernel/debug/kfd/hqds > baseline_hqd.txt

BASELINE_QUEUES=$(grep -c "Queue ID" baseline_mqd.txt || echo "0")
echo "✅ 基线队列数: $BASELINE_QUEUES"
echo ""

# ========== 启动测试模型 ==========
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "步骤2: 启动测试模型"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 创建测试脚本
cat > test_model.py << 'PYEOF'
#!/usr/bin/env python3
import torch
import time
import os

print(f"[{time.strftime('%H:%M:%S')}] 测试模型启动")
print(f"  PID: {os.getpid()}")
print(f"  CUDA可用: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("ERROR: CUDA不可用！")
    exit(1)

print(f"  GPU数量: {torch.cuda.device_count()}")
print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
print("")

# 创建GPU张量并执行计算
print(f"[{time.strftime('%H:%M:%S')}] 创建GPU数据...")
x = torch.randn(2000, 2000, device='cuda')
y = torch.randn(2000, 2000, device='cuda')
torch.cuda.synchronize()
print("✅ GPU数据创建完成")
print("")

# 持续计算
print(f"[{time.strftime('%H:%M:%S')}] 开始计算（2分钟）...")
start = time.time()
iteration = 0

while time.time() - start < 120:
    z = torch.mm(x, y)
    torch.cuda.synchronize()
    
    iteration += 1
    if iteration % 100 == 0:
        elapsed = time.time() - start
        print(f"  [{elapsed:6.1f}s] Iteration {iteration}")
    
    time.sleep(0.02)  # 20ms间隔

print("")
print(f"[{time.strftime('%H:%M:%S')}] 计算完成")
print(f"  总迭代: {iteration}")
PYEOF

# 在容器中启动测试模型
docker exec $CONTAINER bash -c "
export MAMBA_EXE='/root/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/root/micromamba'
eval \"\$(/root/.local/bin/micromamba shell hook --shell=bash)\"
micromamba activate flashinfer-rocm
python3 /data/dockercode/gpreempt_test/test_model.py
" > model_output.log 2>&1 &

MODEL_PID=$!
echo "✅ 测试模型已启动（后台）"
echo "   宿主机进程PID: $MODEL_PID"
echo ""

# 等待初始化
echo "⏳ 等待模型初始化（20秒）..."
for i in {1..20}; do
    echo -n "."
    sleep 1
done
echo ""
echo ""

# 查找容器内的Python进程
CONTAINER_PID=$(docker exec $CONTAINER ps aux | grep test_model.py | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$CONTAINER_PID" ]; then
    echo "⚠️ 未找到容器内的进程"
    echo "   查看模型输出:"
    head -20 model_output.log
    exit 1
fi

echo "✅ 找到容器内进程"
echo "   容器内PID: $CONTAINER_PID"
echo ""

# ========== 持续监控 ==========
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "步骤3: 持续监控队列（每${INTERVAL}秒）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

SAMPLES=$((DURATION / INTERVAL))

for i in $(seq 1 $SAMPLES); do
    TIMESTAMP=$(date +%s)
    HUMAN_TIME=$(date +%H:%M:%S)
    
    echo "采样 $i/$SAMPLES ($HUMAN_TIME)"
    
    # MQD快照
    sudo cat /sys/kernel/debug/kfd/mqds > "snapshot_mqd_${i}_${TIMESTAMP}.txt"
    
    # HQD快照
    sudo cat /sys/kernel/debug/kfd/hqds > "snapshot_hqd_${i}_${TIMESTAMP}.txt"
    
    # 进程信息
    docker exec $CONTAINER ps aux > "snapshot_ps_${i}_${TIMESTAMP}.txt"
    
    # 提取该进程的队列信息
    QUEUE_INFO=$(sudo cat /sys/kernel/debug/kfd/mqds | grep -B 2 -A 5 "pid $CONTAINER_PID" || echo "")
    
    if [ -n "$QUEUE_INFO" ]; then
        QUEUE_COUNT=$(echo "$QUEUE_INFO" | grep -c "Queue ID" || echo "0")
        echo "  ✅ 找到 $QUEUE_COUNT 个队列"
        
        # 保存到单独文件
        echo "$QUEUE_INFO" > "queue_info_${i}_${TIMESTAMP}.txt"
        
        # 显示Queue IDs
        QUEUE_IDS=$(echo "$QUEUE_INFO" | grep "Queue ID" | awk '{print $3}' | tr '\n' ',' | sed 's/,$//')
        echo "     Queue IDs: $QUEUE_IDS"
    else
        echo "  ⚠️ 未找到队列信息"
    fi
    
    echo ""
    
    if [ "$i" -lt "$SAMPLES" ]; then
        sleep $INTERVAL
    fi
done

# ========== 等待模型完成 ==========
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "步骤4: 等待模型完成"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if ps -p $MODEL_PID > /dev/null 2>&1; then
    echo "等待测试模型完成..."
    wait $MODEL_PID 2>/dev/null || true
    echo "✅ 模型已完成"
else
    echo "模型已经结束"
fi

# 记录结束后状态
echo ""
echo "记录模型结束后状态..."
sudo cat /sys/kernel/debug/kfd/mqds > final_mqd.txt
sudo cat /sys/kernel/debug/kfd/hqds > final_hqd.txt

FINAL_QUEUES=$(grep -c "Queue ID" final_mqd.txt || echo "0")
echo "✅ 最终队列数: $FINAL_QUEUES"

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  数据收集完成                                           ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "结果保存在: $OUTPUT_DIR"
echo ""
echo "下一步: 运行分析脚本"
echo "  python3 ../analyze_queue_usage.py $OUTPUT_DIR"
```

---

### 脚本2: 队列数据分析脚本

```python
#!/usr/bin/env python3
# analyze_queue_usage.py
# 分析队列监控数据

import os
import sys
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Set

@dataclass
class QueueSnapshot:
    timestamp: int
    queue_id: int
    pid: int
    priority: int
    is_active: bool
    queue_type: str

@dataclass
class HQDSnapshot:
    timestamp: int
    inst: int
    pipe: int
    queue: int
    is_active: bool

def parse_mqd_file(filepath):
    """解析MQD快照文件"""
    queues = []
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # 按队列分割
    queue_blocks = re.split(r'\n\s*\n', content)
    
    for block in queue_blocks:
        if 'Queue ID' not in block:
            continue
        
        # 提取信息
        queue_id_match = re.search(r'Queue ID:\s+(\d+)', block)
        pid_match = re.search(r'pid\s+(\d+)', block)
        priority_match = re.search(r'priority:\s+(\d+)', block)
        active_match = re.search(r'is active:\s+(\w+)', block)
        type_match = re.search(r'type:\s+(\w+)', block)
        
        if queue_id_match and pid_match:
            queue = QueueSnapshot(
                timestamp=0,  # 从文件名提取
                queue_id=int(queue_id_match.group(1)),
                pid=int(pid_match.group(1)),
                priority=int(priority_match.group(1)) if priority_match else 0,
                is_active=active_match.group(1).lower() == 'true' if active_match else False,
                queue_type=type_match.group(1) if type_match else 'unknown'
            )
            queues.append(queue)
    
    return queues

def parse_hqd_file(filepath):
    """解析HQD快照文件"""
    hqds = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # 每个HQD有58行
    for i in range(0, len(lines), 58):
        if i + 2 >= len(lines):
            break
        
        # 第1行: inst, pipe, queue
        header_match = re.search(r'Inst (\d+), Pipe (\d+), Queue (\d+)', lines[i])
        if not header_match:
            continue
        
        # 第3行: HQD_ACTIVE
        active_match = re.search(r'0x([0-9a-fA-F]+)', lines[i+2])
        
        hqd = HQDSnapshot(
            timestamp=0,
            inst=int(header_match.group(1)),
            pipe=int(header_match.group(2)),
            queue=int(header_match.group(3)),
            is_active=(int(active_match.group(1), 16) & 1) == 1 if active_match else False
        )
        hqds.append(hqd)
    
    return hqds

def analyze_queue_usage(results_dir):
    """分析队列使用情况"""
    
    print("╔════════════════════════════════════════════════════════╗")
    print("║  队列使用情况分析                                       ║")
    print("╚════════════════════════════════════════════════════════╝")
    print("")
    
    # 收集所有快照文件
    mqd_files = sorted([f for f in os.listdir(results_dir) if f.startswith('snapshot_mqd_')])
    
    if not mqd_files:
        print("❌ 未找到快照文件")
        return
    
    print(f"📊 找到 {len(mqd_files)} 个MQD快照")
    print("")
    
    # 分析每个快照
    all_queues_by_sample = []
    target_pid = None
    
    for mqd_file in mqd_files:
        filepath = os.path.join(results_dir, mqd_file)
        queues = parse_mqd_file(filepath)
        
        # 找到目标PID（假设是最新的非0进程）
        if target_pid is None and queues:
            pids = [q.pid for q in queues if q.pid > 0]
            if pids:
                target_pid = max(pids)
        
        all_queues_by_sample.append(queues)
    
    if target_pid is None:
        print("❌ 未找到目标进程")
        return
    
    print(f"🎯 目标进程PID: {target_pid}")
    print("")
    
    # ========== 分析1: 队列数量变化 ==========
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("分析1: 队列数量变化")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("")
    
    queue_counts = []
    for i, queues in enumerate(all_queues_by_sample):
        target_queues = [q for q in queues if q.pid == target_pid]
        queue_counts.append(len(target_queues))
        print(f"  采样 {i+1:2d}: {len(target_queues)} 个队列")
    
    print("")
    print(f"  平均队列数: {sum(queue_counts) / len(queue_counts):.1f}")
    print(f"  最小队列数: {min(queue_counts)}")
    print(f"  最大队列数: {max(queue_counts)}")
    
    if min(queue_counts) == max(queue_counts):
        print("  ✅ 队列数量稳定（一致）")
    else:
        print("  ⚠️ 队列数量有变化")
    
    print("")
    
    # ========== 分析2: Queue ID分布 ==========
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("分析2: Queue ID分布")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("")
    
    all_queue_ids = set()
    queue_id_by_sample = []
    
    for i, queues in enumerate(all_queues_by_sample):
        target_queues = [q for q in queues if q.pid == target_pid]
        queue_ids = {q.queue_id for q in target_queues}
        queue_id_by_sample.append(queue_ids)
        all_queue_ids.update(queue_ids)
        
        print(f"  采样 {i+1:2d}: {sorted(queue_ids)}")
    
    print("")
    print(f"  所有出现的Queue IDs: {sorted(all_queue_ids)}")
    print(f"  唯一Queue ID数量: {len(all_queue_ids)}")
    
    # 检查一致性
    if len(all_queue_ids) == queue_counts[0] and all(len(ids) == queue_counts[0] for ids in queue_id_by_sample):
        print("  ✅ Queue ID在所有采样中一致")
    else:
        print("  ⚠️ Queue ID有变化")
    
    print("")
    
    # ========== 分析3: 队列属性 ==========
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("分析3: 队列属性分析")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("")
    
    # 使用最后一个快照
    final_queues = [q for q in all_queues_by_sample[-1] if q.pid == target_pid]
    
    for q in final_queues:
        print(f"  Queue ID {q.queue_id}:")
        print(f"    Priority: {q.priority}")
        print(f"    Type: {q.queue_type}")
        print(f"    Active: {q.is_active}")
        print("")
    
    # ========== 分析4: MQD → HQD映射 ==========
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("分析4: MQD → HQD映射验证")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("")
    
    # 分析HQD（使用第一个快照）
    hqd_file = mqd_files[0].replace('mqd', 'hqd')
    if os.path.exists(os.path.join(results_dir, hqd_file)):
        hqds = parse_hqd_file(os.path.join(results_dir, hqd_file))
        active_hqds = [h for h in hqds if h.is_active]
        
        print(f"  总HQD数: {len(hqds)}")
        print(f"  Active HQD数: {len(active_hqds)}")
        print("")
        
        # 按XCC统计
        hqds_by_xcc = defaultdict(list)
        for h in active_hqds:
            hqds_by_xcc[h.inst].append(h)
        
        print(f"  Active HQD分布（按XCC）:")
        for xcc in sorted(hqds_by_xcc.keys()):
            print(f"    XCC {xcc}: {len(hqds_by_xcc[xcc])} 个")
        print("")
        
        # 验证: MQD数量 vs HQD数量
        num_mqds = len(final_queues)
        num_active_hqds = len(active_hqds)
        
        print(f"  MQD数量: {num_mqds}")
        print(f"  Active HQD数量: {num_active_hqds}")
        
        expected_hqds = num_mqds * 4  # MI308X: 4个XCC
        print(f"  期望HQD数量: {expected_hqds} (MQD × 4)")
        
        if num_active_hqds == expected_hqds:
            print("  ✅ 映射关系正确: 1 MQD → 4 HQD")
        else:
            print(f"  ⚠️ 映射关系不匹配（差异: {num_active_hqds - expected_hqds}）")
    
    print("")
    
    # ========== 总结 ==========
    print("╔════════════════════════════════════════════════════════╗")
    print("║  实验总结                                               ║")
    print("╚════════════════════════════════════════════════════════╝")
    print("")
    
    print(f"✅ 该模型使用 {queue_counts[0]} 个队列（MQD）")
    print(f"✅ Queue IDs: {sorted(all_queue_ids)}")
    print(f"✅ 队列数量稳定性: {'一致' if min(queue_counts) == max(queue_counts) else '有变化'}")
    print("")
    
    print("💡 对POC Stage 1的意义:")
    print(f"   - 抢占粒度: {queue_counts[0]} 个队列")
    print(f"   - 批量操作可行: ✅ (可以一次操作{queue_counts[0]}个队列)")
    print(f"   - 识别策略: 使用PID过滤，Queue ID: {sorted(all_queue_ids)}")
    print("")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 analyze_queue_usage.py <results_dir>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    if not os.path.exists(results_dir):
        print(f"❌ 目录不存在: {results_dir}")
        sys.exit(1)
    
    analyze_queue_usage(results_dir)
```

---

## 📊 预期结果

### 场景1: 单GPU PyTorch程序

```
预期队列数: 1-2个
  - 1个compute queue (用于kernel执行)
  - 可能1个SDMA queue (用于数据传输)

MQD → HQD映射:
  - 1个MQD → 4个HQD（跨4个XCC）
  - 或 2个MQD → 8个HQD

Queue ID稳定性:
  ✅ 队列数量一致
  ⚠️ Queue ID可能不同（动态分配）
  ⚠️ HQD坐标可能不同（Round-Robin）
```

### 场景2: 多GPU训练

```
预期队列数: 2-4个/GPU
  - 每个GPU至少1个compute queue
  - 可能有额外的SDMA队列用于GPU间通信

总队列数 = 队列数/GPU × GPU数量
```

---

## 🎯 成功标准

### 必须达成 ✅

1. **能够识别模型的队列**
   - 通过PID过滤出模型的所有MQD
   - 提取Queue ID列表

2. **验证MQD → HQD映射**
   - 确认1个MQD映射到4个XCC的HQD
   - 验证HQD_ACTIVE状态

3. **队列数量稳定**
   - 同一模型多次运行，队列数量一致
   - ±1个队列的差异可接受

### 希望达成 ⭐

4. **Queue ID可预测**
   - 如果Queue ID每次一致，说明分配是确定性的
   - 这会简化POC实现

5. **快速识别**
   - 模型启动后5秒内能识别队列
   - 为实时抢占提供基础

---

## 🚀 立即执行

### 快速开始

```bash
# 1. 准备脚本
cd /mnt/md0/zhehan/code/flashinfer/dockercode/gpreempt_test

# 2. 复制实验脚本
# (将上面的exp01_queue_monitor.sh保存)

chmod +x exp01_queue_monitor.sh

# 3. 运行实验
./exp01_queue_monitor.sh

# 4. 分析结果
python3 analyze_queue_usage.py ./exp01_results
```

### 预计时间

- 准备: 5分钟
- 执行: 10分钟（自动）
- 分析: 5分钟
- **总计**: ~20分钟

---

## 📝 实验记录模板

### 实验日志

```markdown
# 实验1执行记录

**日期**: 2026-02-04  
**执行人**: Zhehan  
**系统**: MI308X, 8 GPUs, RHEL 8

## 实验配置

- 容器: zhenaiter
- 测试模型: PyTorch矩阵乘法
- 监控时长: 100秒
- 采样间隔: 10秒

## 实验结果

### 队列数量
- 稳定队列数: X个
- Queue IDs: [...]

### MQD → HQD映射
- MQD数: X
- Active HQD数: X
- 验证结果: ✅/❌

### 一致性测试
- 5次运行的队列数: [X, X, X, X, X]
- 一致性: ✅/❌

## 关键发现

1. ...
2. ...
3. ...

## 对POC的影响

- 抢占粒度: X个队列
- 识别策略: ...
- 实施建议: ...

## 附件

- 原始数据: ./exp01_results/
- 分析脚本: analyze_queue_usage.py
- 日志文件: exp01_execution.log
```

---

## 🔗 后续实验

### 实验2: 不同模型对比

```bash
# 测试3种模型:
1. simple_matmul.py   (简单矩阵乘法)
2. resnet50_train.py  (ResNet50训练)
3. bert_inference.py  (BERT推理)

# 对比队列使用情况
```

### 实验3: 并发模型测试

```bash
# 同时运行2个模型
# 验证队列是否独立、不重叠
```

### 实验4: 队列生命周期详细追踪

```bash
# 1秒采样间隔
# 精确追踪队列创建和销毁时机
```

---

**创建时间**: 2026-02-04  
**重要性**: ⭐⭐⭐⭐⭐  
**执行优先级**: 最高  
**预计收益**: 为POC Stage 1提供关键数据

**立即行动**: 运行 `exp01_queue_monitor.sh` 开始实验！
