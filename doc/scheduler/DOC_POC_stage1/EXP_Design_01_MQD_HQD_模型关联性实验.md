# MQD/HQD 与 AI 模型关联性实验设计

**日期**: 2026-02-03  
**目标**: 验证 AI 模型与 Queue (MQD/HQD) 的映射关系和一致性  
**重要性**: ⭐⭐⭐⭐⭐ 这是 POC Stage 1 的关键前置实验

---

## 🎯 实验目标

### 核心问题

1. **同一模型多次运行，Queue ID 是否一致？**
   - 如果一致 → 可以预先映射 Queue ID
   - 如果不一致 → 需要动态发现机制

2. **不同模型使用的 Queue 是否不同？**
   - 如果不同 → 可以区分模型
   - 如果相同 → 无法基于 Queue 区分

3. **多模型并发时，Queue 是否重合？**
   - 如果不重合 → 可以独立控制
   - 如果重合 → 需要更复杂的策略

---

## 📊 实验设计

### 实验 1: 单模型多次运行 (模型 A)

**目标**: 验证 Queue ID 的一致性

```bash
# 重复 5 次
for i in {1..5}; do
    echo "=== Run $i ==="
    
    # 运行前记录
    snapshot_before
    
    # 运行模型 A
    run_model_A
    
    # 运行中记录 (模型正在运行)
    snapshot_during
    
    # 结束后记录
    snapshot_after
    
    # 对比
    compare_snapshots
    
    sleep 5  # 等待队列完全释放
done
```

**预期结果**:
- **情况 A**: Queue ID 每次都相同 (例如总是 0, 1)
  - → 说明 KFD 按顺序分配
  - → 可预测性高
- **情况 B**: Queue ID 每次不同
  - → 需要动态发现
  - → 不可预测

---

### 实验 2: 单模型多次运行 (模型 B)

**目标**: 与模型 A 对比，验证不同模型的模式

```bash
# 重复 5 次
for i in {1..5}; do
    echo "=== Run $i (Model B) ==="
    
    snapshot_before
    run_model_B
    snapshot_during
    snapshot_after
    compare_snapshots
    
    sleep 5
done
```

**预期结果**:
- 与模型 A 的 Queue ID 是否有规律性
- 是否使用不同的 Queue ID 范围

---

### 实验 3: 双模型并发运行

**目标**: 验证 Queue 是否重合

```bash
echo "=== Concurrent Run: Model A + Model B ==="

# 清空 dmesg
sudo dmesg -c > /dev/null

# 运行前记录
snapshot_before

# 启动模型 A (后台)
run_model_A &
PID_A=$!
sleep 2  # 等待启动

# 记录模型 A 的 Queue
snapshot_model_A

# 启动模型 B (后台)
run_model_B &
PID_B=$!
sleep 2  # 等待启动

# 记录模型 A + B 的 Queue
snapshot_model_AB

# 等待完成
wait $PID_A
wait $PID_B

# 运行后记录
snapshot_after

# 分析
analyze_queue_overlap
```

**预期结果**:
- **情况 A**: Queue 不重合
  - 模型 A 用 Queue 0, 1
  - 模型 B 用 Queue 2, 3
  - → 可以独立控制
- **情况 B**: Queue 重合
  - 两个模型共享某些 Queue
  - → 需要更细粒度的控制

---

## 🛠️ 实验工具脚本

### 工具 1: snapshot_mqd_hqd.sh

```bash
#!/bin/bash
# snapshot_mqd_hqd.sh
# 记录当前的 MQD 和 HQD 状态

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$1"
TAG="$2"  # before/during/after

mkdir -p "$OUTPUT_DIR"

echo "📸 Snapshot at $TIMESTAMP ($TAG)"

# 记录 MQD
echo "=== MQD Snapshot ===" > "$OUTPUT_DIR/mqd_${TAG}_${TIMESTAMP}.txt"
sudo cat /sys/kernel/debug/kfd/mqds >> "$OUTPUT_DIR/mqd_${TAG}_${TIMESTAMP}.txt" 2>&1

# 提取关键信息
echo ""
echo "MQD Summary:" | tee "$OUTPUT_DIR/mqd_${TAG}_${TIMESTAMP}_summary.txt"
sudo cat /sys/kernel/debug/kfd/mqds | grep -E "Queue ID:|Process:|is active:|priority:" | tee -a "$OUTPUT_DIR/mqd_${TAG}_${TIMESTAMP}_summary.txt"

# 记录 HQD
echo "=== HQD Snapshot ===" > "$OUTPUT_DIR/hqd_${TAG}_${TIMESTAMP}.txt"
sudo cat /sys/kernel/debug/kfd/hqds >> "$OUTPUT_DIR/hqd_${TAG}_${TIMESTAMP}.txt" 2>&1

# 统计 HQD 活跃数
echo ""
echo "HQD Summary:" | tee "$OUTPUT_DIR/hqd_${TAG}_${TIMESTAMP}_summary.txt"
python3 - <<'EOF' | tee -a "$OUTPUT_DIR/hqd_${TAG}_${TIMESTAMP}_summary.txt"
import sys
import re

try:
    with open('/sys/kernel/debug/kfd/hqds', 'r') as f:
        content = f.read()
    
    # 统计 HQD
    lines = content.split('\n')
    total_hqd = 0
    active_hqd = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        if 'CP Pipe' in line:
            total_hqd += 1
            # 读取下一行寄存器
            if i+1 < len(lines):
                reg_line = lines[i+1]
                parts = reg_line.split()
                if len(parts) >= 4:
                    # 第3个寄存器是 CP_HQD_ACTIVE
                    try:
                        cp_hqd_active = int(parts[3], 16)
                        if cp_hqd_active & 0x1:
                            active_hqd += 1
                    except:
                        pass
        i += 1
    
    print(f"Total HQD: {total_hqd}")
    print(f"Active HQD: {active_hqd}")
    print(f"Active Rate: {active_hqd*100//total_hqd if total_hqd>0 else 0}%")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
EOF

echo "✅ Snapshot saved to $OUTPUT_DIR"
```

---

### 工具 2: extract_queue_info.py

```python
#!/usr/bin/env python3
# extract_queue_info.py
# 从 MQD snapshot 中提取特定进程的 Queue 信息

import sys
import re
from dataclasses import dataclass
from typing import List

@dataclass
class QueueInfo:
    queue_id: int
    pid: int
    pasid: int
    is_active: bool
    priority: int
    device: str

def parse_mqd_file(mqd_file: str) -> List[QueueInfo]:
    """解析 MQD snapshot 文件"""
    
    queues = []
    
    with open(mqd_file, 'r') as f:
        content = f.read()
    
    # 按 "Compute queue on device" 分割
    blocks = re.split(r'Compute queue on device', content)
    
    for block in blocks[1:]:
        lines = block.strip().split('\n')
        device = lines[0].strip()
        
        info = {}
        for line in lines[1:]:
            # Queue ID
            m = re.search(r'Queue ID:\s+(\d+)', line)
            if m:
                info['queue_id'] = int(m.group(1))
            
            # Process
            m = re.search(r'Process:\s+pid\s+(\d+)\s+pasid\s+(0x[0-9a-fA-F]+)', line)
            if m:
                info['pid'] = int(m.group(1))
                info['pasid'] = int(m.group(2), 16)
            
            # is active
            m = re.search(r'is active:\s+(yes|no)', line)
            if m:
                info['is_active'] = (m.group(1) == 'yes')
            
            # priority
            m = re.search(r'priority:\s+(\d+)', line)
            if m:
                info['priority'] = int(m.group(1))
        
        if 'queue_id' in info:
            q = QueueInfo(
                queue_id=info['queue_id'],
                pid=info.get('pid', 0),
                pasid=info.get('pasid', 0),
                is_active=info.get('is_active', False),
                priority=info.get('priority', 0),
                device=device
            )
            queues.append(q)
    
    return queues


def filter_by_pid(queues: List[QueueInfo], pid: int) -> List[QueueInfo]:
    """筛选特定 PID 的队列"""
    return [q for q in queues if q.pid == pid]


def compare_queue_lists(queues1: List[QueueInfo], queues2: List[QueueInfo]) -> dict:
    """对比两次运行的 Queue 列表"""
    
    qids1 = set(q.queue_id for q in queues1)
    qids2 = set(q.queue_id for q in queues2)
    
    common = qids1 & qids2
    only_in_1 = qids1 - qids2
    only_in_2 = qids2 - qids1
    
    return {
        'common': common,
        'only_in_1': only_in_1,
        'only_in_2': only_in_2,
        'total_1': len(qids1),
        'total_2': len(qids2),
        'overlap_rate': len(common) / max(len(qids1), len(qids2)) if qids1 or qids2 else 0
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract Queue Info from MQD snapshot')
    parser.add_argument('mqd_file', help='MQD snapshot file')
    parser.add_argument('--pid', type=int, help='Filter by PID')
    parser.add_argument('--compare', help='Compare with another MQD file')
    
    args = parser.parse_args()
    
    # 解析第一个文件
    queues1 = parse_mqd_file(args.mqd_file)
    
    if args.pid:
        queues1 = filter_by_pid(queues1, args.pid)
    
    print(f"📊 MQD File: {args.mqd_file}")
    print(f"Total Queues: {len(queues1)}")
    
    if args.pid:
        print(f"Filtered by PID: {args.pid}")
    
    print(f"\n📋 Queue List:")
    for q in queues1:
        active_str = "✅" if q.is_active else "❌"
        print(f"  Queue ID: {q.queue_id}, PID: {q.pid}, Priority: {q.priority}, Active: {active_str}")
    
    # 对比模式
    if args.compare:
        queues2 = parse_mqd_file(args.compare)
        
        if args.pid:
            queues2 = filter_by_pid(queues2, args.pid)
        
        comp = compare_queue_lists(queues1, queues2)
        
        print(f"\n🔍 Comparison:")
        print(f"  File 1: {len(comp['total_1'])} queues")
        print(f"  File 2: {len(comp['total_2'])} queues")
        print(f"  Common: {len(comp['common'])} queues - {list(comp['common'])}")
        print(f"  Only in File 1: {len(comp['only_in_1'])} queues - {list(comp['only_in_1'])}")
        print(f"  Only in File 2: {len(comp['only_in_2'])} queues - {list(comp['only_in_2'])}")
        print(f"  Overlap Rate: {comp['overlap_rate']*100:.1f}%")
        
        if comp['overlap_rate'] > 0.8:
            print(f"\n✅ 高度一致！Queue ID 具有可预测性")
        elif comp['overlap_rate'] > 0.5:
            print(f"\n⚠️ 部分一致，需要进一步分析")
        else:
            print(f"\n❌ 低一致性，Queue ID 不可预测")
```

---

### 工具 3: run_experiment.sh

```bash
#!/bin/bash
# run_experiment.sh
# 完整的 MQD/HQD 实验脚本

set -e

# 配置
EXPERIMENT_DIR="./experiment_results"
MODEL_A_CMD="python3 simple_model_a.py"
MODEL_B_CMD="python3 simple_model_b.py"

mkdir -p "$EXPERIMENT_DIR"

echo "╔════════════════════════════════════════════════════════╗"
echo "║  MQD/HQD 模型关联性实验                                 ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# ========== 实验 1: 模型 A 单次运行 ==========
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 实验 1: 模型 A 多次运行（5次）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

for i in {1..5}; do
    echo "=== Run $i/5 ==="
    
    RUN_DIR="$EXPERIMENT_DIR/model_a_run_$i"
    mkdir -p "$RUN_DIR"
    
    # Before
    ./snapshot_mqd_hqd.sh "$RUN_DIR" "before"
    
    # 运行模型 A
    echo "🚀 启动模型 A..."
    $MODEL_A_CMD &
    PID_A=$!
    echo "  PID: $PID_A"
    echo "$PID_A" > "$RUN_DIR/model_a_pid.txt"
    
    # 等待模型启动
    sleep 3
    
    # During
    ./snapshot_mqd_hqd.sh "$RUN_DIR" "during"
    
    # 提取模型 A 的 Queue
    python3 extract_queue_info.py \
        "$RUN_DIR/mqd_during_"*.txt \
        --pid $PID_A \
        > "$RUN_DIR/model_a_queues.txt"
    
    echo "  模型 A 使用的 Queue:"
    cat "$RUN_DIR/model_a_queues.txt" | grep "Queue ID"
    
    # 等待完成
    wait $PID_A
    
    # After
    ./snapshot_mqd_hqd.sh "$RUN_DIR" "after"
    
    echo "✅ Run $i 完成"
    echo ""
    
    sleep 5  # 等待队列完全释放
done

echo "✅ 实验 1 完成！"
echo ""

# ========== 实验 2: 模型 B 单次运行 ==========
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 实验 2: 模型 B 多次运行（5次）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

for i in {1..5}; do
    echo "=== Run $i/5 ==="
    
    RUN_DIR="$EXPERIMENT_DIR/model_b_run_$i"
    mkdir -p "$RUN_DIR"
    
    ./snapshot_mqd_hqd.sh "$RUN_DIR" "before"
    
    echo "🚀 启动模型 B..."
    $MODEL_B_CMD &
    PID_B=$!
    echo "  PID: $PID_B"
    echo "$PID_B" > "$RUN_DIR/model_b_pid.txt"
    
    sleep 3
    
    ./snapshot_mqd_hqd.sh "$RUN_DIR" "during"
    
    python3 extract_queue_info.py \
        "$RUN_DIR/mqd_during_"*.txt \
        --pid $PID_B \
        > "$RUN_DIR/model_b_queues.txt"
    
    echo "  模型 B 使用的 Queue:"
    cat "$RUN_DIR/model_b_queues.txt" | grep "Queue ID"
    
    wait $PID_B
    
    ./snapshot_mqd_hqd.sh "$RUN_DIR" "after"
    
    echo "✅ Run $i 完成"
    echo ""
    
    sleep 5
done

echo "✅ 实验 2 完成！"
echo ""

# ========== 实验 3: 双模型并发 ==========
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 实验 3: 模型 A + B 并发运行（3次）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

for i in {1..3}; do
    echo "=== Run $i/3 ==="
    
    RUN_DIR="$EXPERIMENT_DIR/concurrent_run_$i"
    mkdir -p "$RUN_DIR"
    
    ./snapshot_mqd_hqd.sh "$RUN_DIR" "before"
    
    # 启动模型 A
    echo "🚀 启动模型 A..."
    $MODEL_A_CMD &
    PID_A=$!
    echo "  PID_A: $PID_A"
    echo "$PID_A" > "$RUN_DIR/model_a_pid.txt"
    
    sleep 3
    
    # 记录模型 A
    ./snapshot_mqd_hqd.sh "$RUN_DIR" "model_a_only"
    python3 extract_queue_info.py \
        "$RUN_DIR/mqd_model_a_only_"*.txt \
        --pid $PID_A \
        > "$RUN_DIR/model_a_queues.txt"
    
    # 启动模型 B
    echo "🚀 启动模型 B..."
    $MODEL_B_CMD &
    PID_B=$!
    echo "  PID_B: $PID_B"
    echo "$PID_B" > "$RUN_DIR/model_b_pid.txt"
    
    sleep 3
    
    # 记录模型 A + B
    ./snapshot_mqd_hqd.sh "$RUN_DIR" "both"
    python3 extract_queue_info.py \
        "$RUN_DIR/mqd_both_"*.txt \
        --pid $PID_A \
        > "$RUN_DIR/model_a_queues_concurrent.txt"
    python3 extract_queue_info.py \
        "$RUN_DIR/mqd_both_"*.txt \
        --pid $PID_B \
        > "$RUN_DIR/model_b_queues_concurrent.txt"
    
    echo "  模型 A 的 Queue:"
    cat "$RUN_DIR/model_a_queues_concurrent.txt" | grep "Queue ID"
    echo "  模型 B 的 Queue:"
    cat "$RUN_DIR/model_b_queues_concurrent.txt" | grep "Queue ID"
    
    # 等待完成
    wait $PID_A $PID_B
    
    ./snapshot_mqd_hqd.sh "$RUN_DIR" "after"
    
    echo "✅ Run $i 完成"
    echo ""
    
    sleep 5
done

echo "✅ 实验 3 完成！"
echo ""

# ========== 分析结果 ==========
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 开始分析结果..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 analyze_experiment_results.py "$EXPERIMENT_DIR"

echo ""
echo "✅ 所有实验完成！结果保存在: $EXPERIMENT_DIR"
```

---

### 工具 4: analyze_experiment_results.py

```python
#!/usr/bin/env python3
# analyze_experiment_results.py
# 分析实验结果

import os
import sys
import re
from pathlib import Path
from collections import defaultdict

def extract_queue_ids_from_file(filepath):
    """从 queue 文件中提取 Queue ID"""
    queue_ids = []
    
    if not os.path.exists(filepath):
        return queue_ids
    
    with open(filepath, 'r') as f:
        for line in f:
            m = re.search(r'Queue ID:\s+(\d+)', line)
            if m:
                queue_ids.append(int(m.group(1)))
    
    return queue_ids


def analyze_experiment_dir(exp_dir):
    """分析实验目录"""
    
    print("╔════════════════════════════════════════════════════════╗")
    print("║  实验结果分析报告                                        ║")
    print("╚════════════════════════════════════════════════════════╝")
    print("")
    
    # ========== 分析实验 1: 模型 A 一致性 ==========
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📊 实验 1: 模型 A 的 Queue ID 一致性")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("")
    
    model_a_runs = []
    for i in range(1, 6):
        run_dir = os.path.join(exp_dir, f"model_a_run_{i}")
        queue_file = os.path.join(run_dir, "model_a_queues.txt")
        
        if os.path.exists(queue_file):
            queue_ids = extract_queue_ids_from_file(queue_file)
            model_a_runs.append(queue_ids)
            print(f"  Run {i}: Queue IDs = {queue_ids}")
    
    # 分析一致性
    if len(model_a_runs) > 1:
        all_same = all(set(run) == set(model_a_runs[0]) for run in model_a_runs[1:])
        
        print(f"\n结论:")
        if all_same:
            print(f"  ✅ 模型 A 的 Queue ID 完全一致！")
            print(f"  ✅ Queue IDs: {model_a_runs[0]}")
            print(f"  ✅ 可预测性: 高")
        else:
            print(f"  ⚠️ 模型 A 的 Queue ID 不一致")
            print(f"  ⚠️ 可预测性: 低")
            print(f"  ⚠️ 需要动态发现机制")
    
    print("")
    
    # ========== 分析实验 2: 模型 B 一致性 ==========
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📊 实验 2: 模型 B 的 Queue ID 一致性")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("")
    
    model_b_runs = []
    for i in range(1, 6):
        run_dir = os.path.join(exp_dir, f"model_b_run_{i}")
        queue_file = os.path.join(run_dir, "model_b_queues.txt")
        
        if os.path.exists(queue_file):
            queue_ids = extract_queue_ids_from_file(queue_file)
            model_b_runs.append(queue_ids)
            print(f"  Run {i}: Queue IDs = {queue_ids}")
    
    # 分析一致性
    if len(model_b_runs) > 1:
        all_same = all(set(run) == set(model_b_runs[0]) for run in model_b_runs[1:])
        
        print(f"\n结论:")
        if all_same:
            print(f"  ✅ 模型 B 的 Queue ID 完全一致！")
            print(f"  ✅ Queue IDs: {model_b_runs[0]}")
        else:
            print(f"  ⚠️ 模型 B 的 Queue ID 不一致")
    
    print("")
    
    # ========== 分析实验 3: 双模型并发 ==========
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📊 实验 3: 双模型并发时的 Queue 重合度")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("")
    
    for i in range(1, 4):
        run_dir = os.path.join(exp_dir, f"concurrent_run_{i}")
        
        qa_file = os.path.join(run_dir, "model_a_queues_concurrent.txt")
        qb_file = os.path.join(run_dir, "model_b_queues_concurrent.txt")
        
        if os.path.exists(qa_file) and os.path.exists(qb_file):
            qa_ids = set(extract_queue_ids_from_file(qa_file))
            qb_ids = set(extract_queue_ids_from_file(qb_file))
            
            overlap = qa_ids & qb_ids
            
            print(f"  Run {i}:")
            print(f"    模型 A Queue IDs: {sorted(qa_ids)}")
            print(f"    模型 B Queue IDs: {sorted(qb_ids)}")
            print(f"    重合的 Queue IDs: {sorted(overlap)}")
            
            if len(overlap) == 0:
                print(f"    ✅ 无重合，可以独立控制")
            else:
                print(f"    ⚠️ 有重合，需要更细粒度控制")
            print("")
    
    # ========== 总结 ==========
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🎯 实验总结")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("")
    
    # 模型 A 一致性
    if model_a_runs:
        a_consistent = all(set(run) == set(model_a_runs[0]) for run in model_a_runs[1:])
        print(f"1. 模型 A Queue ID 一致性: {'✅ 一致' if a_consistent else '❌ 不一致'}")
        if a_consistent:
            print(f"   固定 Queue IDs: {model_a_runs[0]}")
    
    # 模型 B 一致性
    if model_b_runs:
        b_consistent = all(set(run) == set(model_b_runs[0]) for run in model_b_runs[1:])
        print(f"2. 模型 B Queue ID 一致性: {'✅ 一致' if b_consistent else '❌ 不一致'}")
        if b_consistent:
            print(f"   固定 Queue IDs: {model_b_runs[0]}")
    
    # POC Stage 1 建议
    print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"💡 POC Stage 1 实施建议")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"")
    
    if model_a_runs and model_b_runs:
        a_consistent = all(set(run) == set(model_a_runs[0]) for run in model_a_runs[1:])
        b_consistent = all(set(run) == set(model_b_runs[0]) for run in model_b_runs[1:])
        
        if a_consistent and b_consistent:
            print(f"✅ Queue ID 高度可预测")
            print(f"")
            print(f"推荐方案:")
            print(f"  1. 预先配置 Queue ID 映射")
            print(f"     - Online-AI (模型 A) → Queue IDs: {model_a_runs[0]}")
            print(f"     - Offline-AI (模型 B) → Queue IDs: {model_b_runs[0]}")
            print(f"")
            print(f"  2. Test Framework 直接使用这些 Queue IDs")
            print(f"     - 无需动态发现")
            print(f"     - 简化代码逻辑")
        else:
            print(f"⚠️ Queue ID 不可预测")
            print(f"")
            print(f"推荐方案:")
            print(f"  1. 实现动态发现机制")
            print(f"     - 解析 /sys/kernel/debug/kfd/mqds")
            print(f"     - 根据 PID 查找 Queue ID")
            print(f"")
            print(f"  2. 模型启动时打印 Queue ID")
            print(f"     - 修改模型代码")
            print(f"     - 保存到文件供调度器读取")
    
    print("")
    print("✅ 所有分析完成！")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_experiment_results.py <experiment_dir>")
        sys.exit(1)
    
    exp_dir = sys.argv[1]
    
    if not os.path.exists(exp_dir):
        print(f"❌ 实验目录不存在: {exp_dir}")
        sys.exit(1)
    
    analyze_experiment_dir(exp_dir)
```

---

## 🧪 测试模型准备

### 模型 A: simple_model_a.py (轻量级推理)

```python
#!/usr/bin/env python3
# simple_model_a.py
# 模拟 Online-AI (推理)

import torch
import torch.nn as nn
import time
import os
import sys

class SimpleModelA(nn.Module):
    """轻量级推理模型"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    print("╔════════════════════════════════════════╗")
    print("║  模型 A: Online-AI (推理)               ║")
    print("╚════════════════════════════════════════╝")
    print(f"PID: {os.getpid()}")
    print("")
    
    # 创建模型
    model = SimpleModelA().cuda()
    model.eval()
    
    # 等待队列创建
    time.sleep(1)
    
    print("🚀 开始推理循环...")
    print("   (运行 30 秒)")
    print("")
    
    # 推理循环
    start_time = time.time()
    iteration = 0
    
    while time.time() - start_time < 30:  # 运行 30 秒
        x = torch.randn(32, 512).cuda()
        
        with torch.no_grad():
            y = model(x)
        
        iteration += 1
        if iteration % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  Iteration {iteration}, Elapsed: {elapsed:.1f}s")
        
        time.sleep(0.01)  # 10ms 间隔
    
    print("")
    print(f"✅ 完成！总迭代: {iteration}")

if __name__ == '__main__':
    main()
```

---

### 模型 B: simple_model_b.py (重量级训练)

```python
#!/usr/bin/env python3
# simple_model_b.py
# 模拟 Offline-AI (训练)

import torch
import torch.nn as nn
import time
import os

class SimpleModelB(nn.Module):
    """重量级训练模型"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 1024)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    print("╔════════════════════════════════════════╗")
    print("║  模型 B: Offline-AI (训练)              ║")
    print("╚════════════════════════════════════════╝")
    print(f"PID: {os.getpid()}")
    print("")
    
    # 创建模型
    model = SimpleModelB().cuda()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # 等待队列创建
    time.sleep(1)
    
    print("🚀 开始训练循环...")
    print("   (运行 30 秒)")
    print("")
    
    # 训练循环
    start_time = time.time()
    epoch = 0
    
    while time.time() - start_time < 30:  # 运行 30 秒
        x = torch.randn(64, 1024).cuda()
        
        # Forward
        y = model(x)
        loss = y.sum()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch += 1
        if epoch % 20 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch}, Loss: {loss.item():.4f}, Elapsed: {elapsed:.1f}s")
        
        time.sleep(0.05)  # 50ms 间隔
    
    print("")
    print(f"✅ 完成！总 Epoch: {epoch}")

if __name__ == '__main__':
    main()
```

---

## 🚀 实验执行步骤

### 准备阶段

```bash
# 1. 进入 Docker 容器
docker exec -it zhenaiter /bin/bash

# 2. 激活环境
export MAMBA_EXE='/root/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/root/micromamba'
eval "$(/root/.local/bin/micromamba shell hook --shell=bash)"
micromamba activate flashinfer-rocm

# 3. 创建实验目录
cd /data/dockercode
mkdir -p poc_stage1_experiment
cd poc_stage1_experiment

# 4. 复制脚本（假设已经创建好）
# 从宿主机复制：
# docker cp ./tools/ zhenaiter:/data/dockercode/poc_stage1_experiment/

# 或直接在容器内创建
cat > snapshot_mqd_hqd.sh << 'EOF'
# (粘贴上面的脚本内容)
EOF
chmod +x snapshot_mqd_hqd.sh

# 创建其他脚本...
```

---

### 执行实验

```bash
# 方式 1: 完全自动化
./run_experiment.sh

# 方式 2: 手动执行（更灵活）
# 实验 1
for i in {1..5}; do
    echo "Run $i..."
    mkdir -p exp1_run_$i
    
    ./snapshot_mqd_hqd.sh exp1_run_$i before
    python3 simple_model_a.py &
    PID=$!
    sleep 3
    ./snapshot_mqd_hqd.sh exp1_run_$i during
    python3 extract_queue_info.py exp1_run_$i/mqd_during_*.txt --pid $PID
    wait $PID
    ./snapshot_mqd_hqd.sh exp1_run_$i after
    
    sleep 5
done
```

---

## 📊 预期结果和分析

### 场景 A: Queue ID 高度可预测 (理想情况)

**实验结果**:
```
实验 1 - 模型 A:
  Run 1: Queue IDs = [0, 1]
  Run 2: Queue IDs = [0, 1]  ← 一致
  Run 3: Queue IDs = [0, 1]  ← 一致
  Run 4: Queue IDs = [0, 1]  ← 一致
  Run 5: Queue IDs = [0, 1]  ← 一致

实验 2 - 模型 B:
  Run 1: Queue IDs = [0, 1]
  Run 2: Queue IDs = [0, 1]  ← 一致
  Run 3: Queue IDs = [0, 1]  ← 一致
  ...

实验 3 - 并发:
  模型 A Queue IDs: [0, 1]
  模型 B Queue IDs: [2, 3]  ← 不重合！
```

**对 POC Stage 1 的影响**:
- ✅ **极其有利！**
- ✅ 可以预配置 Queue ID
- ✅ Test Framework 可以硬编码
- ✅ 无需复杂的动态发现

**实施方案**:
```python
# gpreempt_scheduler.py (简化版)

# 硬编码 Queue IDs
ONLINE_AI_QUEUES = [0, 1]   # 模型 A
OFFLINE_AI_QUEUES = [2, 3]  # 模型 B

def handle_online_request():
    # 直接暂停 Offline 队列
    suspend_queues(OFFLINE_AI_QUEUES)
    
    # 等待 Online 完成
    time.sleep(0.05)
    
    # 恢复 Offline 队列
    resume_queues(OFFLINE_AI_QUEUES)
```

---

### 场景 B: Queue ID 部分可预测

**实验结果**:
```
实验 1 - 模型 A:
  Run 1: Queue IDs = [0, 1]
  Run 2: Queue IDs = [0, 1]  ← 一致
  Run 3: Queue IDs = [1, 2]  ← 不一致！
  Run 4: Queue IDs = [0, 1]
  Run 5: Queue IDs = [2, 3]  ← 不一致！
```

**对 POC Stage 1 的影响**:
- ⚠️ 需要动态发现
- ⚠️ 不能硬编码
- ⚠️ 但仍然可行

**实施方案**:
```python
# gpreempt_scheduler.py (动态版)

def discover_queues_by_priority():
    """动态发现队列"""
    mqds = parse_mqd_debugfs()
    
    online_queues = [q for q in mqds if q.priority >= 10]
    offline_queues = [q for q in mqds if q.priority <= 5]
    
    return online_queues, offline_queues

def handle_online_request():
    # 动态发现
    online_qs, offline_qs = discover_queues_by_priority()
    
    # 暂停 Offline
    offline_qids = [q.queue_id for q in offline_qs if q.is_active]
    suspend_queues(offline_qids)
    
    time.sleep(0.05)
    
    resume_queues(offline_qids)
```

---

### 场景 C: Queue 完全不可预测 (最差情况)

**实验结果**:
```
实验 1 - 模型 A:
  Run 1: Queue IDs = [5, 17, 23]  ← 随机
  Run 2: Queue IDs = [1, 9, 14]   ← 完全不同
  Run 3: Queue IDs = [3, 7, 21]   ← 无规律
```

**对 POC Stage 1 的影响**:
- ❌ 严重影响可行性
- ❌ 需要复杂的运行时发现
- ❌ 或修改模型代码打印 Queue ID

**实施方案**:
```python
# 方案 1: 通过 PID 查找 (仍然可行)
def find_model_queues_by_pid(model_pid):
    mqds = parse_mqd_debugfs()
    return [q for q in mqds if q.pid == model_pid and q.is_active]

# 方案 2: 修改模型打印 Queue ID
# 在模型代码中添加:
import ctypes
lib = ctypes.CDLL('./libgpreempt_poc.so')
lib.gpreempt_print_my_queues()  # 打印到日志
```

---

## 📁 实验输出结构

```
experiment_results/
├── model_a_run_1/
│   ├── mqd_before_20260203_120000.txt
│   ├── mqd_during_20260203_120005.txt
│   ├── mqd_after_20260203_120035.txt
│   ├── hqd_before_20260203_120000.txt
│   ├── hqd_during_20260203_120005.txt
│   ├── hqd_after_20260203_120035.txt
│   ├── model_a_queues.txt           ← 提取的 Queue IDs
│   └── model_a_pid.txt              ← PID
├── model_a_run_2/
│   └── ...
├── model_b_run_1/
│   └── ...
├── concurrent_run_1/
│   ├── mqd_before_*.txt
│   ├── mqd_model_a_only_*.txt       ← 只有 A 运行
│   ├── mqd_both_*.txt               ← A + B 都运行
│   ├── model_a_queues.txt
│   ├── model_b_queues.txt
│   └── ...
└── analysis_report.txt              ← 最终分析报告
```

---

## 🎯 实验时间表

```
准备阶段:        30 分钟
  - 创建脚本
  - 准备模型
  - 测试工具

实验 1 (模型 A):  15 分钟 (5 次 × 3 分钟)
实验 2 (模型 B):  15 分钟
实验 3 (并发):    10 分钟 (3 次 × 3 分钟)

分析结果:        15 分钟
───────────────────────────
总计:           ~1.5 小时
```

---

## 🔧 Docker 环境选择

### 选项 1: zhenaiter 容器 (⭐⭐⭐⭐⭐ 推荐)

**优点**:
- ✅ 已验证 CWSR/GPREEMPT 测试
- ✅ 有 PyTorch + ROCm 环境
- ✅ 可以访问 /dev/kfd 和 debugfs
- ✅ 环境稳定

**环境**:
```
容器: zhenaiter
ROCm: 6.4
PyTorch: 2.9.1+rocm6.4
GPU: 8× AMD Instinct MI308X
```

**测试目录**:
```bash
/data/dockercode/poc_stage1_experiment/
```

---

### 选项 2: XSched 容器

**优点**:
- ✅ 已有 BERT 等 AI 模型测试
- ✅ 可以复用 AI 模型脚本

**缺点**:
- ⚠️ 可能需要关闭 XSched 的 LD_PRELOAD
- ⚠️ 担心 XSched 影响队列行为

---

### 推荐: 使用 zhenaiter + 简单模型

**原因**:
1. 环境纯净，无 XSched 干扰
2. 简单模型更容易控制和理解
3. 可以复用 CWSR 测试的经验

**决策**:
- 先在 zhenaiter 用简单模型做实验
- 如果成功，再考虑集成 XSched 的 BERT 模型

---

## ✅ 实验检查清单

**准备阶段**:
- [ ] zhenaiter 容器可以访问
- [ ] PyTorch + ROCm 环境正常
- [ ] 可以读取 `/sys/kernel/debug/kfd/mqds`
- [ ] 可以读取 `/sys/kernel/debug/kfd/hqds`
- [ ] 脚本已创建并测试

**实验 1**:
- [ ] 模型 A 脚本可运行
- [ ] 5 次运行全部完成
- [ ] MQD/HQD 快照已保存
- [ ] Queue IDs 已提取

**实验 2**:
- [ ] 模型 B 脚本可运行
- [ ] 5 次运行全部完成
- [ ] MQD/HQD 快照已保存
- [ ] Queue IDs 已提取

**实验 3**:
- [ ] 双模型并发运行成功
- [ ] 3 次运行全部完成
- [ ] Queue 重合度已分析

**分析阶段**:
- [ ] 一致性分析完成
- [ ] POC Stage 1 策略已确定
- [ ] 实验报告已生成

---

## 📊 数据收集表格

### 模型 A Queue ID 一致性

| Run | Queue IDs | 一致性 | 备注 |
|-----|-----------|--------|------|
| 1 | | | |
| 2 | | ✅ / ❌ | |
| 3 | | ✅ / ❌ | |
| 4 | | ✅ / ❌ | |
| 5 | | ✅ / ❌ | |

**一致性**: ___% (5 次中有 ___ 次一致)

---

### 模型 B Queue ID 一致性

| Run | Queue IDs | 一致性 | 备注 |
|-----|-----------|--------|------|
| 1 | | | |
| 2 | | ✅ / ❌ | |
| 3 | | ✅ / ❌ | |
| 4 | | ✅ / ❌ | |
| 5 | | ✅ / ❌ | |

**一致性**: ___% (5 次中有 ___ 次一致)

---

### 双模型并发 Queue 重合度

| Run | 模型 A Queue IDs | 模型 B Queue IDs | 重合 Queue IDs | 备注 |
|-----|-----------------|-----------------|---------------|------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

**重合率**: ___% 

---

## 🎯 根据实验结果的 POC Stage 1 策略

### 如果 Queue ID 一致且不重合 (最理想)

**策略**: 硬编码 Queue ID

```python
# 配置文件
QUEUE_MAPPING = {
    'online_ai': [0, 1],   # 固定的 Queue IDs
    'offline_ai': [2, 3]
}

# 调度器
class GPreemptScheduler:
    def __init__(self):
        self.offline_queues = QUEUE_MAPPING['offline_ai']
    
    def handle_online_request(self):
        # 直接使用
        suspend_queues(self.offline_queues)
        # ...
        resume_queues(self.offline_queues)
```

**优点**:
- ✅ 极简单
- ✅ 无运行时开销
- ✅ 1-2 天完成

---

### 如果 Queue ID 一致但重合 (需要优先级)

**策略**: 基于优先级发现

```python
def discover_offline_queues():
    """查找低优先级队列"""
    mqds = parse_mqd_debugfs()
    return [q.queue_id for q in mqds if q.priority <= 5 and q.is_active]

class GPreemptScheduler:
    def handle_online_request(self):
        # 动态发现低优先级队列
        offline_qids = discover_offline_queues()
        
        if offline_qids:
            suspend_queues(offline_qids)
            # ...
            resume_queues(offline_qids)
```

**优点**:
- ✅ 灵活
- ✅ 支持多种场景
- ⚠️ 需要解析 debugfs

---

### 如果 Queue ID 不一致 (最复杂)

**策略**: 基于 PID 发现

```python
class GPreemptScheduler:
    def __init__(self):
        self.model_pids = {
            'online': None,
            'offline': None
        }
    
    def register_model(self, model_type, pid):
        """模型启动时注册 PID"""
        self.model_pids[model_type] = pid
    
    def find_queues_by_pid(self, pid):
        """根据 PID 查找队列"""
        mqds = parse_mqd_debugfs()
        return [q.queue_id for q in mqds if q.pid == pid and q.is_active]
    
    def handle_online_request(self):
        # 查找 Offline 模型的队列
        offline_pid = self.model_pids['offline']
        if offline_pid:
            offline_qids = self.find_queues_by_pid(offline_pid)
            
            if offline_qids:
                suspend_queues(offline_qids)
                # ...
                resume_queues(offline_qids)
```

**优点**:
- ✅ 支持任意场景
- ✅ 最通用

**缺点**:
- ⚠️ 需要 PID 注册机制
- ⚠️ 需要解析 debugfs

---

## 🔍 深入分析：为什么需要这个实验？

### POC Stage 1 的核心挑战

使用 `KFD_IOC_DBG_TRAP_SUSPEND_QUEUES` API 需要提供 **Queue ID 数组**:

```c
int gpreempt_suspend_queues(uint32_t *queue_ids,  // ← 需要知道!
                           uint32_t num_queues,
                           uint32_t grace_period_us);
```

**问题**: 如何知道 Offline-AI 使用的 Queue IDs？

**可能方案**:
1. **硬编码** - 如果 Queue ID 可预测
2. **动态发现** - 解析 debugfs
3. **模型注册** - 模型启动时注册 PID
4. **暴力枚举** - 尝试所有可能的 Queue ID

**本实验目的**: 确定哪种方案最可行

---

## 📖 参考文档

### 已有的 Queue 研究

- `QUEUE_ID_SOLUTION.md` - Queue ID 获取历史经验
- `HARDWARE_QUEUE_DISTRIBUTION_ANALYSIS.md` - HQD 分布分析
- `MQD_HQD_MAPPING_ANALYSIS.md` - MQD↔HQD 映射关系

### MQD/HQD debugfs 说明

- MQD: `/sys/kernel/debug/kfd/mqds`
  - Queue ID (用户态)
  - Process PID
  - Priority
  - is active

- HQD: `/sys/kernel/debug/kfd/hqds`
  - Hardware 坐标 (Inst, Pipe, Queue)
  - CP_HQD_ACTIVE bit[0]
  - 56 个硬件寄存器

---

## 🚀 立即开始

### 快速测试 (5 分钟)

```bash
# 进入容器
docker exec -it zhenaiter /bin/bash

# 进入测试目录
cd /data/dockercode/gpreempt_test

# 手动测试 1 次
echo "=== Before ==="
sudo cat /sys/kernel/debug/kfd/mqds | grep "Queue ID"

echo "=== Running ==="
HIP_DEVICE=0 ./test_hip_preempt 10000 5000 0 &
PID=$!
echo "PID: $PID"
sleep 2

echo "=== During ==="
sudo cat /sys/kernel/debug/kfd/mqds | grep -A 3 "pid $PID"

wait $PID

echo "=== After ==="
sudo cat /sys/kernel/debug/kfd/mqds | grep "Queue ID"
```

**观察**: 
- 运行中出现了哪些新的 Queue ID？
- 这些 Queue ID 是否是小整数 (0, 1, 2...)?

---

### 完整实验 (1.5 小时)

```bash
# 1. 准备环境
docker exec -it zhenaiter /bin/bash
cd /data/dockercode
mkdir -p poc_stage1_experiment
cd poc_stage1_experiment

# 2. 创建所有脚本
# (复制上面的脚本内容)

# 3. 创建测试模型
# (复制 simple_model_a.py 和 simple_model_b.py)

# 4. 运行完整实验
./run_experiment.sh

# 5. 查看结果
cat analysis_report.txt
```

---

## 💡 实验成功的标志

**最理想**:
```
✅ 模型 A Queue IDs: [0, 1] (5/5 一致)
✅ 模型 B Queue IDs: [2, 3] (5/5 一致)
✅ 并发时不重合
→ 结论: 可以硬编码，POC Stage 1 极简单
```

**可接受**:
```
⚠️ 模型 A Queue IDs: 不完全一致 (3/5 一致)
⚠️ 模型 B Queue IDs: 不完全一致
→ 结论: 需要动态发现，但仍可行
```

**需要调整**:
```
❌ Queue IDs 完全随机
→ 结论: 需要修改模型代码或使用更复杂的发现机制
```

---

## ➡️ 下一步

### 如果实验成功

1. 根据实验结果选择 POC Stage 1 策略
2. 实施 libgpreempt_poc.so
3. 编写 Test Framework
4. 开始功能测试

### 如果遇到问题

1. 尝试不同的模型 (HIP kernel vs PyTorch)
2. 检查是否需要修改模型代码
3. 考虑使用 XSched 的 BERT 模型

---

**最后更新**: 2026-02-03  
**维护者**: Zhehan

**状态**: 📋 实验方案已完成，可以开始执行！

**重要性**: ⭐⭐⭐⭐⭐ **这个实验的结果将直接决定 POC Stage 1 的实施策略！**

---

## 🎓 附录：为什么这个实验如此重要？

### 问题背景

POC Stage 1 的核心是使用 `suspend_queues(queue_ids)` 暂停 Offline-AI，但我们不知道：

1. **如何获取 Queue ID？**
   - 暴力枚举？
   - 解析 debugfs？
   - 修改模型代码？

2. **Queue ID 是否稳定？**
   - 如果稳定 → 简单
   - 如果不稳定 → 复杂

3. **多模型如何区分？**
   - 如果不重合 → 容易
   - 如果重合 → 困难

### 实验价值

**1 小时的实验可以节省 1 周的开发时间！**

- 如果 Queue ID 可预测 → POC Stage 1 只需 3-5 天
- 如果 Queue ID 不可预测 → POC Stage 1 可能需要 7-10 天

**立即开始实验！** 🚀
