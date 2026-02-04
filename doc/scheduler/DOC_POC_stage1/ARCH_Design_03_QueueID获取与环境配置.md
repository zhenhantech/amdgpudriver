# POC Stage 1: Queue ID 获取与测试环境配置

**日期**: 2026-02-03  
**目的**: 解决 POC Stage 1 的核心问题 - 如何获取运行中 AI 模型的 Queue ID

---

## 🎯 核心问题

POC Stage 1 使用 `KFD_IOC_DBG_TRAP_SUSPEND_QUEUES` API 需要提供 **Queue ID**，但如何在 AI 模型运行时获取其使用的 Queue ID？

---

## 📚 历史经验：已有的解决方案

根据之前 GPREEMPT 测试经验（参考 `DOC_GPREEMPT/MI300_Testing/QUEUE_ID_SOLUTION.md`）：

### 关键发现

**Queue ID 的特点**:
- ✅ 是进程内的用户态队列 ID
- ✅ 由 KFD 在创建队列时分配
- ✅ **通常是小的整数**: 0, 1, 2, 3...
- ❌ 不是全局硬件队列 ID
- ❌ 不跨进程

**典型分布**:

| Queue ID | 用途 | 可能性 |
|----------|------|--------|
| **0** | 第一个 compute queue | ⭐⭐⭐⭐⭐ 非常高 |
| **1** | 第二个 queue / transfer queue | ⭐⭐⭐⭐ 高 |
| **2-3** | 额外的 compute/transfer queue | ⭐⭐⭐ 中等 |
| **4-10** | 罕见（多 stream 程序） | ⭐ 低 |
| **> 10** | 非常罕见 | ⭐ 很低 |

---

## ✅ 推荐方案（按难度排序）

### 方案 A: 暴力枚举 Queue ID (⭐⭐⭐⭐⭐ 推荐)

**优点**:
- ✅ 最简单，无需修改代码
- ✅ 1-2 分钟就能找到
- ✅ 适合 POC 快速验证

**实施**:

```python
# poc_stage1/tools/find_queue_id.py

import subprocess
import time

def find_active_queue_id(max_attempts=20):
    """暴力枚举查找活跃的 Queue ID"""
    
    print("🔍 开始查找活跃的 Queue ID...")
    print("请确保目标 AI 模型正在运行！\n")
    
    for qid in range(max_attempts):
        print(f"尝试 Queue ID {qid}...", end=' ')
        
        # 调用测试程序（不真正抢占，只是查询）
        ret = subprocess.call([
            'sudo', './test_queue_exists', str(qid)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if ret == 0:
            print("✅ 找到！")
            return qid
        else:
            print("❌")
    
    print(f"\n⚠️ 未在 0-{max_attempts-1} 范围内找到活跃队列")
    return None

if __name__ == '__main__':
    qid = find_active_queue_id()
    if qid is not None:
        print(f"\n✅ 活跃的 Queue ID: {qid}")
        print(f"\n下一步: 使用此 ID 进行抢占测试")
        print(f"  gpreempt_suspend_queues(&qid, 1, 1000);")
```

**对应的 C 测试工具**:

```c
// test_queue_exists.c
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/kfd_ioctl.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <queue_id>\n", argv[0]);
        return 1;
    }
    
    uint32_t qid = atoi(argv[1]);
    int fd = open("/dev/kfd", O_RDWR);
    if (fd < 0) {
        return 1;
    }
    
    // 使用 suspend_queues 测试队列是否存在
    // grace_period=0 表示立即检查，不真正抢占
    struct kfd_ioctl_dbg_trap_args args = {0};
    args.op = KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT;  // 用查询代替抢占
    
    // 或者直接检查 MQD debugfs
    char path[256];
    snprintf(path, sizeof(path), 
             "/sys/kernel/debug/kfd/mqds");
    FILE *fp = fopen(path, "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            int id;
            if (sscanf(line, "    Queue ID: %d", &id) == 1) {
                if (id == qid) {
                    fclose(fp);
                    close(fd);
                    return 0;  // 找到
                }
            }
        }
        fclose(fp);
    }
    
    close(fd);
    return 1;  // 未找到
}
```

---

### 方案 B: 解析 MQD debugfs (⭐⭐⭐⭐ 推荐用于生产)

**优点**:
- ✅ 精确可靠
- ✅ 可以获取队列详细信息（优先级、进程 PID 等）
- ✅ 适合自动化

**位置**: `/sys/kernel/debug/kfd/mqds`

**格式示例**:

```
Compute queue on device 0001:01:00.0
    Queue ID: 1 (0x1)
    Address: 0x7f8c00000000
    Process: pid 15234 pasid 0x8001
    is active: yes
    priority: 7
    queue count: 1

Compute queue on device 0001:01:00.0
    Queue ID: 2 (0x2)
    ...
```

**解析代码**:

```python
# poc_stage1/libgpreempt_poc/mqd_parser.py

import re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class QueueInfo:
    queue_id: int
    pid: int
    pasid: int
    is_active: bool
    priority: int
    address: int
    device: str

def parse_mqd_debugfs(mqd_path="/sys/kernel/debug/kfd/mqds") -> List[QueueInfo]:
    """解析 MQD debugfs 文件"""
    
    queues = []
    
    with open(mqd_path, 'r') as f:
        content = f.read()
    
    # 按 "Compute queue on device" 分割
    queue_blocks = re.split(r'Compute queue on device', content)
    
    for block in queue_blocks[1:]:  # 跳过第一个空块
        lines = block.strip().split('\n')
        
        # 提取设备名
        device = lines[0].strip()
        
        # 解析字段
        queue_info = {
            'device': device
        }
        
        for line in lines[1:]:
            line = line.strip()
            
            # Queue ID
            m = re.search(r'Queue ID:\s+(\d+)', line)
            if m:
                queue_info['queue_id'] = int(m.group(1))
            
            # Process info
            m = re.search(r'Process:\s+pid\s+(\d+)\s+pasid\s+(0x[0-9a-fA-F]+)', line)
            if m:
                queue_info['pid'] = int(m.group(1))
                queue_info['pasid'] = int(m.group(2), 16)
            
            # is active
            m = re.search(r'is active:\s+(yes|no)', line)
            if m:
                queue_info['is_active'] = (m.group(1) == 'yes')
            
            # priority
            m = re.search(r'priority:\s+(\d+)', line)
            if m:
                queue_info['priority'] = int(m.group(1))
            
            # address
            m = re.search(r'Address:\s+(0x[0-9a-fA-F]+)', line)
            if m:
                queue_info['address'] = int(m.group(1), 16)
        
        # 构建对象
        if 'queue_id' in queue_info:
            q = QueueInfo(
                queue_id=queue_info['queue_id'],
                pid=queue_info.get('pid', 0),
                pasid=queue_info.get('pasid', 0),
                is_active=queue_info.get('is_active', False),
                priority=queue_info.get('priority', 0),
                address=queue_info.get('address', 0),
                device=queue_info.get('device', '')
            )
            queues.append(q)
    
    return queues


def find_queue_by_pid(target_pid: int) -> List[QueueInfo]:
    """根据进程 PID 查找队列"""
    all_queues = parse_mqd_debugfs()
    return [q for q in all_queues if q.pid == target_pid]


def find_queue_by_priority(min_prio: int, max_prio: int) -> List[QueueInfo]:
    """根据优先级范围查找队列"""
    all_queues = parse_mqd_debugfs()
    return [q for q in all_queues 
            if min_prio <= q.priority <= max_prio and q.is_active]


# 使用示例
if __name__ == '__main__':
    import os
    
    # 查找当前进程的队列
    my_pid = os.getpid()
    my_queues = find_queue_by_pid(my_pid)
    
    print(f"进程 {my_pid} 的队列:")
    for q in my_queues:
        print(f"  Queue ID: {q.queue_id}, Priority: {q.priority}, Active: {q.is_active}")
    
    # 查找所有高优先级队列（Online-AI）
    online_queues = find_queue_by_priority(10, 15)
    print(f"\n高优先级队列 (Online-AI):")
    for q in online_queues:
        print(f"  Queue ID: {q.queue_id}, PID: {q.pid}, Priority: {q.priority}")
    
    # 查找所有低优先级队列（Offline-AI）
    offline_queues = find_queue_by_priority(0, 5)
    print(f"\n低优先级队列 (Offline-AI):")
    for q in offline_queues:
        print(f"  Queue ID: {q.queue_id}, PID: {q.pid}, Priority: {q.priority}")
```

---

### 方案 C: 修改 HIP 程序打印 Queue ID (⭐⭐⭐ 最精确)

**适用场景**: 需要完全确定的情况

**实施**: 修改 AI 模型的启动脚本，添加 Queue ID 打印

```python
# ai_model_with_qid_print.py

import torch
import os
import ctypes

# 加载 libgpreempt_poc.so
lib = ctypes.CDLL('./libgpreempt_poc.so')

# HIP 模型初始化
model = YourAIModel().cuda()

# 等待队列创建
time.sleep(0.5)

# 获取当前进程的队列
queues = get_process_queues(os.getpid())
print(f"✅ 模型使用的 Queue IDs: {[q.queue_id for q in queues]}")

# 保存到文件，供外部读取
with open('/tmp/model_queue_ids.txt', 'w') as f:
    f.write(','.join(str(q.queue_id) for q in queues))

# 开始推理/训练
model.inference(...)
```

---

## 🐳 Docker 环境配置

根据之前的测试经验，我们有两套 Docker 环境：

### Docker 1: zhenaiter (CWSR + GPREEMPT 测试) ⭐⭐⭐⭐⭐

**环境信息**:
```bash
容器名:     zhenaiter
ROCm:      6.4
PyTorch:   2.9.1+rocm6.4
GPU:       8× AMD Instinct MI308X
Conda:     flashinfer-rocm (micromamba)
```

**已测试功能**:
- ✅ GPREEMPT IOCTL 接口
- ✅ CWSR 抢占/恢复
- ✅ Queue ID 暴力枚举方法

**推荐用于**: POC Stage 1 初步测试

**启动方式**:
```bash
# 进入容器
docker exec -it zhenaiter /bin/bash

# 激活环境
export MAMBA_EXE='/root/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/root/micromamba'
eval "$(/root/.local/bin/micromamba shell hook --shell=bash)"
micromamba activate flashinfer-rocm

# 测试目录
cd /data/dockercode/gpreempt_test  # GPREEMPT 测试
# 或
cd /data/dockercode/xsched          # XSched 测试（如果需要）
```

---

### Docker 2: XSched 专用容器 (Paper #2 测试)

**环境信息**:
```bash
容器名:     待确认
ROCm:      6.4
XSched:    /workspace/xsched/output/lib/libshimhip.so
```

**已测试功能**:
- ✅ XSched LD_PRELOAD 拦截
- ✅ BERT 多优先级调度
- ✅ 双模型并发测试

**推荐用于**: XSched 功能测试（不含 GPREEMPT）

---

## 🎯 POC Stage 1 推荐策略

### 策略 1: 使用 zhenaiter 容器 (⭐⭐⭐⭐⭐ 强烈推荐)

**原因**:
1. ✅ 已经验证了 GPREEMPT IOCTL 工作
2. ✅ 已经有 Queue ID 枚举经验
3. ✅ 环境稳定，不会破坏 XSched 测试

**实施计划**:
```
第1步: 在 zhenaiter 容器内准备 POC Stage 1 代码
  ├─ libgpreempt_poc.so (C 库)
  ├─ mqd_parser.py (Queue ID 解析)
  └─ test_priority_scheduling.py (测试框架)

第2步: 使用简单的 AI 模型测试
  ├─ 简单的 HIP kernel（如之前的 test_hip_preempt）
  ├─ 或轻量级的 PyTorch 模型
  └─ 暴力枚举获取 Queue ID

第3步: 进行 Online/Offline 场景测试
  ├─ Offline: 长时间训练循环
  ├─ Online: 间歇推理请求
  └─ 验证抢占和恢复
```

---

### 策略 2: 不用 XSched，单独测试 GPREEMPT (⭐⭐⭐⭐ 推荐)

**原因**:
- POC Stage 1 的目标是验证 **Queue-level 抢占**
- 不需要 XSched 的应用层调度
- XSched 是 Paper #2，可以在 Stage 2/3 集成

**测试流程**:

```python
# 测试场景：不使用 XSched
# 只测试 KFD_IOC_DBG_TRAP_SUSPEND_QUEUES 的抢占功能

# Step 1: 启动 Offline-AI（低优先级训练）
offline_script.py  # 持续训练，不停止

# Step 2: 获取 Offline 队列 ID
offline_queue_ids = parse_mqd_debugfs()  # 找到 priority=2 的队列

# Step 3: 间歇启动 Online-AI（高优先级推理）
for i in range(10):
    # 暂停 Offline
    suspend_queues(offline_queue_ids)
    
    # Online 推理
    result = online_model.inference(data)
    
    # 恢复 Offline
    resume_queues(offline_queue_ids)
    
    time.sleep(0.5)
```

**优势**: 
- ✅ 简单清晰
- ✅ 专注于核心功能
- ✅ 易于调试

---

## 📐 详细实施方案

### Phase 1: 环境准备 (半天)

**在 zhenaiter 容器内**:

```bash
# 1. 进入容器
docker exec -it zhenaiter /bin/bash

# 2. 创建工作目录
mkdir -p /data/dockercode/poc_stage1
cd /data/dockercode/poc_stage1

# 3. 复制代码（从宿主机）
# 宿主机执行:
docker cp \
  /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/ \
  zhenaiter:/data/dockercode/poc_stage1/

# 4. 编译 C 库
cd /data/dockercode/poc_stage1/libgpreempt_poc
make

# 5. 测试库是否可用
./test_api_availability
```

---

### Phase 2: Queue ID 获取测试 (半天)

**测试 1: 暴力枚举法**

```bash
cd /data/dockercode/poc_stage1

# 终端 1: 启动简单的 HIP kernel
./simple_hip_kernel &

# 终端 2: 查找 Queue ID
python3 tools/find_queue_id.py
# 应该输出: "✅ 活跃的 Queue ID: 0" (或 1, 2...)
```

**测试 2: MQD debugfs 解析**

```bash
cd /data/dockercode/poc_stage1

# 启动 HIP kernel
./simple_hip_kernel &
KERNEL_PID=$!

# 使用解析脚本
python3 -c "
from mqd_parser import find_queue_by_pid
queues = find_queue_by_pid($KERNEL_PID)
for q in queues:
    print(f'Queue ID: {q.queue_id}, Priority: {q.priority}')
"
```

---

### Phase 3: AI 模型集成 (1天)

**Offline-AI 模型** (低优先级训练):

```python
# offline_training.py

import torch
import torch.nn as nn
import time
import os
import sys

sys.path.append('/data/dockercode/poc_stage1')
from mqd_parser import find_queue_by_pid

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024, 1024)
    
    def forward(self, x):
        return self.fc(x)

# 创建模型
model = SimpleModel().cuda()
model.train()

# 打印队列信息
time.sleep(0.5)  # 等待队列创建
queues = find_queue_by_pid(os.getpid())
print(f"✅ Offline 模型使用的 Queue IDs: {[q.queue_id for q in queues]}")
with open('/tmp/offline_queue_ids.txt', 'w') as f:
    f.write(','.join(str(q.queue_id) for q in queues))

# 持续训练
print("🚀 开始持续训练...")
for epoch in range(1000):
    x = torch.randn(128, 1024).cuda()
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    time.sleep(0.1)  # 模拟训练延迟
```

**Online-AI 模型** (高优先级推理):

```python
# online_inference.py

import torch
import time
import os
import sys

sys.path.append('/data/dockercode/poc_stage1')
from mqd_parser import find_queue_by_pid

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 512)
    
    def forward(self, x):
        return self.fc(x)

# 创建模型
model = SimpleModel().cuda()
model.eval()

# 打印队列信息
time.sleep(0.5)
queues = find_queue_by_pid(os.getpid())
print(f"✅ Online 模型使用的 Queue IDs: {[q.queue_id for q in queues]}")
with open('/tmp/online_queue_ids.txt', 'w') as f:
    f.write(','.join(str(q.queue_id) for q in queues))

# 间歇推理
print("🚀 开始推理循环...")
for i in range(100):
    start = time.time()
    
    x = torch.randn(32, 512).cuda()
    with torch.no_grad():
        y = model(x)
    
    latency = (time.time() - start) * 1000
    print(f"Request {i}: {latency:.2f} ms")
    
    time.sleep(0.5)  # 每 500ms 一个请求
```

---

### Phase 4: 调度器测试 (1天)

```python
# test_online_offline_scheduling.py

import subprocess
import time
import ctypes

# 加载库
lib = ctypes.CDLL('/data/dockercode/poc_stage1/libgpreempt_poc/libgpreempt_poc.so')
lib.gpreempt_poc_init()

# Step 1: 启动 Offline 模型（后台）
print("🚀 启动 Offline-AI 模型...")
offline_proc = subprocess.Popen(['python3', 'offline_training.py'])

# 等待队列创建
time.sleep(2)

# 读取 Offline 队列 ID
with open('/tmp/offline_queue_ids.txt', 'r') as f:
    offline_qids = [int(x) for x in f.read().strip().split(',')]
print(f"✅ Offline Queue IDs: {offline_qids}")

# Step 2: 启动 Online 模型（后台）
print("🚀 启动 Online-AI 模型...")
online_proc = subprocess.Popen(['python3', 'online_inference.py'])

# 等待队列创建
time.sleep(2)

# 读取 Online 队列 ID
with open('/tmp/online_queue_ids.txt', 'r') as f:
    online_qids = [int(x) for x in f.read().strip().split(',')]
print(f"✅ Online Queue IDs: {online_qids}")

# Step 3: 模拟 Online 任务高峰，触发抢占
print("\n📊 开始调度测试...")
for i in range(20):
    print(f"\n=== 测试轮次 {i+1} ===")
    
    # 暂停 Offline
    print(f"⏸️  暂停 Offline 队列: {offline_qids}")
    offline_qids_array = (ctypes.c_uint32 * len(offline_qids))(*offline_qids)
    ret = lib.gpreempt_suspend_queues(offline_qids_array, len(offline_qids), 1000)
    
    if ret == 0:
        print("✅ Offline 队列已暂停")
    else:
        print(f"❌ 暂停失败: {ret}")
    
    # 等待 Online 任务完成（模拟）
    time.sleep(0.05)  # 50ms
    
    # 恢复 Offline
    print(f"▶️  恢复 Offline 队列: {offline_qids}")
    ret = lib.gpreempt_resume_queues(offline_qids_array, len(offline_qids))
    
    if ret == 0:
        print("✅ Offline 队列已恢复")
    else:
        print(f"❌ 恢复失败: {ret}")
    
    # 间隔
    time.sleep(0.5)

# 清理
lib.gpreempt_poc_cleanup()
offline_proc.terminate()
online_proc.terminate()

print("\n🎉 测试完成!")
```

---

## 📊 对比：CWSR 测试 vs XSched 测试

### CWSR/GPREEMPT 测试 (zhenaiter 容器)

**Docker**: zhenaiter  
**目录**: `/data/dockercode/gpreempt_test/`  
**测试内容**:
- CWSR 抢占/恢复
- Queue ID 枚举
- 手动触发抢占

**AI 模型**: 
- 简单的 HIP kernel (`test_hip_preempt`)
- 或简单的 PyTorch 模型

**Queue ID 获取方式**: 暴力枚举 0-10

---

### XSched 测试

**Docker**: zhenaiter (同一容器，不同目录)  
**目录**: `/data/dockercode/xsched/`  
**测试内容**:
- XSched LD_PRELOAD
- BERT 多优先级调度
- 应用层调度策略

**AI 模型**: 
- BERT (transformers)
- 双模型并发（test_phase4_dual_model_CORRECT.py）

**Queue ID 获取方式**: 不需要（XSched 在应用层拦截）

---

## 🎯 POC Stage 1 推荐配置

### 推荐方案: 使用 zhenaiter + 简单模型

**环境**: zhenaiter 容器  
**AI 模型**: 简单的 PyTorch 模型（如上面的 SimpleModel）  
**Queue ID**: 暴力枚举 + MQD debugfs 解析  

**优势**:
1. ✅ 环境已验证
2. ✅ 无需 XSched 复杂性
3. ✅ 快速迭代
4. ✅ 易于调试

**测试流程**:
```
Day 1: 
  - 准备 C 库和 Python 框架
  - 在 zhenaiter 容器内编译

Day 2:
  - 测试 Queue ID 获取
  - 验证 suspend/resume 可用

Day 3-4:
  - 编写 Online/Offline 模型
  - 运行完整测试

Day 5:
  - 性能测试和报告
```

---

## 🔧 快速开始指南

### 立即可执行的测试

**测试目标**: 验证能否获取 Queue ID

```bash
# 1. 进入容器
docker exec -it zhenaiter /bin/bash

# 2. 激活环境
export MAMBA_EXE='/root/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/root/micromamba'
eval "$(/root/.local/bin/micromamba shell hook --shell=bash)"
micromamba activate flashinfer-rocm

# 3. 启动测试 kernel
cd /data/dockercode/gpreempt_test
HIP_DEVICE=0 ./test_hip_preempt 100000 20000 0 &

# 4. 查看 Queue ID
sleep 2
cat /sys/kernel/debug/kfd/mqds | grep -A 5 "Queue ID"

# 5. 或者使用之前的枚举方法
for i in {0..5}; do
    echo "Testing Queue ID: $i"
    # 这里可以调用测试程序
done
```

---

## 📚 可复用的代码和脚本

### 从 XSched 可以复用的

**AI 模型**:
- `test_bert_with_xsched_api.py` - BERT 推理框架
- 可以去掉 XSched 部分，只保留模型加载和推理

**测试工具**:
- `quick_env_check.py` - 环境检查
- `simple_test_runner.py` - 简单的测试 runner

### 从 CWSR 测试可以复用的

**Queue ID 方法**:
- 暴力枚举脚本（手动版）
- QUEUE_ID_SOLUTION.md 中的方法

**测试工具**:
- `test_hip_preempt` - 简单的长时间 kernel
- `preempt_queue_manual` - 手动抢占工具

---

## 🎯 最终建议

### 推荐配置

**Docker**: zhenaiter 容器  
**目录**: `/data/dockercode/poc_stage1/` (新建)  
**AI 模型**: 简单的 PyTorch 模型 (不用 BERT，太重)  
**Queue ID**: MQD debugfs 解析 + 暴力枚举备用  

### 实施步骤

```bash
# Week 1: 基础框架
Day 1-2: 编写 libgpreempt_poc.so + mqd_parser.py
Day 3:   编写简单的 PyTorch Online/Offline 模型
Day 4:   编写 GPreemptScheduler 调度器
Day 5:   基本功能测试

# Week 2: 性能测试
Day 6-7: 延迟测试
Day 8-9: 吞吐量测试
Day 10:  报告和文档
```

---

## ✅ 检查清单

**环境准备**:
- [ ] zhenaiter 容器可以访问
- [ ] ROCm 和 PyTorch 可用
- [ ] /dev/kfd 可访问
- [ ] MQD debugfs 可读取

**代码准备**:
- [ ] libgpreempt_poc.so 编译成功
- [ ] mqd_parser.py 可以解析 debugfs
- [ ] test_queue_exists 工具可用
- [ ] find_queue_id.py 脚本工作

**模型准备**:
- [ ] offline_training.py (简单模型)
- [ ] online_inference.py (简单模型)
- [ ] 能打印 Queue ID

**测试准备**:
- [ ] test_priority_scheduling.py
- [ ] GPreemptScheduler 类
- [ ] 统计收集代码

---

## 📖 参考文档

### DOC_GPREEMPT (CWSR 测试)

- `MI300_Testing/QUEUE_ID_SOLUTION.md` - Queue ID 获取方案
- `MI300_Testing/Docker容器内端到端测试_快速开始.md` - Docker 测试指南
- `MI300_Testing/GPREEMPT_XSched_EndToEnd_Test.md` - 完整测试方案

### XSCHED (应用层调度)

- `tests/RUN_IN_DOCKER.md` - XSched Docker 运行指南
- `tests/test_bert_with_xsched_api.py` - BERT 测试脚本
- `tests/test_phase4_dual_model_CORRECT.py` - 双模型测试

### POC_Stage1 (当前)

- `ARCH_Design_01_POC_Stage1_实施方案.md` - 整体方案
- `ARCH_Design_02_三种API技术对比.md` - API 对比
- `POC_Stage1_TODOLIST.md` - 任务清单

---

## ➡️ 下一步

1. **立即可做**: 在 zhenaiter 容器内测试 Queue ID 获取
   ```bash
   docker exec -it zhenaiter bash
   cat /sys/kernel/debug/kfd/mqds | head -30
   ```

2. **本周目标**: 完成 Phase 1-2（API 验证 + Queue ID 机制）

3. **下周目标**: 完成 Phase 3-4（Test Framework + 完整测试）

---

**最后更新**: 2026-02-03  
**维护者**: Zhehan

**结论**: 使用 zhenaiter 容器 + MQD debugfs 解析是 POC Stage 1 最优方案，环境已验证，可以立即开始实施。✅
