

# GPU Queue抢占机制设计

**日期**: 2026-02-05  
**目标**: 设计Case-A抢占Case-B的机制  
**场景**: 高优先级任务（Case-A）需要抢占低优先级任务（Case-B）

---

## 📋 目录

1. [背景与目标](#背景与目标)
2. [Queue使用对比分析](#queue使用对比分析)
3. [抢占机制设计](#抢占机制设计)
4. [实现方案](#实现方案)
5. [测试计划](#测试计划)

---

## 背景与目标

### 测试案例

| Case | 类型 | 特点 | 优先级 |
|------|------|------|--------|
| **Case-A** | CNN卷积网络 | 卷积、池化、批归一化等多种操作 | 高（在线推理） |
| **Case-B** | Transformer | MatMul密集、注意力机制 | 低（离线训练） |

### 目标

1. **分析Queue使用差异**
   - Case-A和Case-B分别使用了哪些Queue？
   - Queue数量、类型、使用模式的差异？

2. **设计抢占机制**
   - Case-A如何抢占Case-B？
   - 如何确保Case-A优先执行？
   - Case-B被抢占后如何恢复？

---

## Queue使用对比分析

### 预期Queue使用模式

#### Case-A (CNN)

**操作类型**:
- 卷积 (Conv2d): Compute Queue
- 池化 (MaxPool): Compute Queue
- 批归一化 (BatchNorm): Compute + Reduction Queue
- 数据传输: 可能使用DMA Queue

**预期特点**:
```
Queue类型: 多样（Compute + DMA）
Queue数量: 可能 2-4 个
使用模式: 操作类型交替，Queue切换频繁
Kernel大小: 中等
```

#### Case-B (Transformer)

**操作类型**:
- Multi-head Attention: 大量MatMul
- Feedforward: Linear (MatMul)
- LayerNorm: Reduction操作
- Softmax: Element-wise操作

**预期特点**:
```
Queue类型: 主要Compute Queue
Queue数量: 可能 1-2 个
使用模式: MatMul密集，单一Queue高频使用
Kernel大小: 大（注意力矩阵）
```

### 实际对比（从AMD日志提取）

**运行测试后填写**:

```bash
# 提取Queue ID
grep 'HWq=.*id=' case_a_cnn.log | grep -o 'id=[0-9]*' | sort -u
grep 'HWq=.*id=' case_b_transformer.log | grep -o 'id=[0-9]*' | sort -u

# 统计Queue使用次数
grep -c 'HWq=' case_a_cnn.log
grep -c 'HWq=' case_b_transformer.log

# 提取Kernel类型
grep 'ShaderName' case_a_cnn.log | sed 's/.*ShaderName : //' | sort | uniq -c
grep 'ShaderName' case_b_transformer.log | sed 's/.*ShaderName : //' | sort | uniq -c
```

**结果示例**:

| Metric | Case-A (CNN) | Case-B (Transformer) |
|--------|--------------|----------------------|
| Queue IDs | 1, 2 | 1 |
| 主要Queue | Compute + DMA | Compute |
| Kernel提交次数 | ~1000 | ~800 |
| 主要Kernel | Conv2d, Pool | MatMul, Softmax |

---

## 抢占机制设计

### 设计目标

1. **优先级保证**: Case-A（高优先级）总是优先执行
2. **低延迟**: Case-A启动后，Case-B应该在毫秒级内暂停
3. **公平性**: Case-B在Case-A空闲时应该能恢复执行
4. **开销低**: 抢占机制本身不应消耗过多资源

### 抢占策略

#### 策略1: Queue优先级抢占（推荐）⭐⭐⭐⭐⭐

**原理**:
- 为Case-A分配高优先级Queue
- 为Case-B分配低优先级Queue
- GPU硬件自动调度：高优先级Queue优先

**优点**:
- ✅ 硬件支持，延迟低
- ✅ 实现简单
- ✅ 开销小

**缺点**:
- ❌ 依赖硬件Queue优先级支持
- ❌ 粒度可能较粗

**伪代码**:
```python
# Case-A (高优先级)
stream_a = torch.cuda.Stream(priority=-1)  # 高优先级
with torch.cuda.stream(stream_a):
    output_a = model_a(input_a)

# Case-B (低优先级)
stream_b = torch.cuda.Stream(priority=0)  # 普通优先级
with torch.cuda.stream(stream_b):
    output_b = model_b(input_b)
```

---

#### 策略2: 显式Suspend/Resume（精确控制）⭐⭐⭐⭐

**原理**:
- 监控Case-A启动
- 使用KFD IOCTLs暂停Case-B的Queue
- Case-A完成后恢复Case-B

**优点**:
- ✅ 精确控制
- ✅ 可以完全暂停Case-B
- ✅ 适用于严格的优先级场景

**缺点**:
- ❌ 需要内核支持（KFD Debug Trap）
- ❌ ROCm 7.x可能不支持
- ❌ 实现复杂

**伪代码**:
```python
import kfd_debug_api

# 监控线程
def monitor():
    while True:
        if case_a_ready():
            # 暂停Case-B
            kfd_debug_api.suspend_queues(pid_b)
            wait_for_case_a_complete()
            # 恢复Case-B
            kfd_debug_api.resume_queues(pid_b)
```

---

#### 策略3: 时间片轮转（公平调度）⭐⭐⭐

**原理**:
- Case-A和Case-B轮流使用GPU
- Case-A时间片更长（例如 80% vs 20%）

**优点**:
- ✅ 两个任务都能执行
- ✅ 可调节比例
- ✅ 不需要特殊硬件支持

**缺点**:
- ❌ Case-A延迟增加
- ❌ 实现复杂（需要调度器）
- ❌ Context切换开销

**伪代码**:
```python
def time_slice_scheduler():
    while True:
        # Case-A: 80ms
        with timeout(80):
            run_case_a()
        
        # Case-B: 20ms
        with timeout(20):
            run_case_b()
```

---

#### 策略4: Event-based同步（协作式）⭐⭐

**原理**:
- Case-B主动检查Case-A的Event
- 如果Case-A启动，Case-B暂停

**优点**:
- ✅ 不需要外部调度器
- ✅ 纯PyTorch实现

**缺点**:
- ❌ Case-B需要修改代码
- ❌ 延迟较高（轮询）
- ❌ 不适合不可控的workload

**伪代码**:
```python
# Case-B
event_a = torch.cuda.Event()
while True:
    if not event_a.query():  # Case-A在运行
        time.sleep(0.001)  # 暂停
        continue
    
    # Case-A空闲，执行Case-B
    output_b = model_b(input_b)
```

---

## 实现方案

### 推荐方案：Queue优先级 + 监控

结合**策略1（Queue优先级）**和**策略2（监控）**：

```
┌─────────────────────────────────────────┐
│  调度器 (Scheduler)                      │
│  - 监控Case-A和Case-B                   │
│  - 动态调整优先级                        │
└─────────────────────────────────────────┘
         │                    │
         ↓                    ↓
    ┌─────────┐          ┌─────────┐
    │ Case-A  │          │ Case-B  │
    │ High Pri│          │ Low Pri │
    └─────────┘          └─────────┘
         │                    │
         ↓                    ↓
    ┌─────────────────────────────┐
    │  GPU Hardware Queue         │
    │  - 优先调度高优先级Queue     │
    └─────────────────────────────┘
```

### 实现步骤

#### 步骤1: 基础实现（纯PyTorch）

```python
#!/usr/bin/env python3
"""
基础抢占测试：使用PyTorch Stream优先级
"""

import torch
import time

# Case-A: 高优先级
stream_high = torch.cuda.Stream(priority=-1)  # 最高优先级
model_a = SimpleCNN().cuda().eval()
input_a = torch.randn(16, 3, 256, 256, device='cuda')

# Case-B: 低优先级
stream_low = torch.cuda.Stream(priority=0)  # 普通优先级
model_b = SimpleTransformer().cuda().eval()
input_b = torch.randn(32, 128, 512, device='cuda')

def run_case_a():
    with torch.cuda.stream(stream_high):
        return model_a(input_a)

def run_case_b():
    with torch.cuda.stream(stream_low):
        return model_b(input_b)

# 并发执行
import threading

def worker_a():
    for i in range(100):
        run_case_a()
        torch.cuda.synchronize()

def worker_b():
    for i in range(100):
        run_case_b()
        torch.cuda.synchronize()

# 启动
thread_a = threading.Thread(target=worker_a)
thread_b = threading.Thread(target=worker_b)

thread_b.start()  # 先启动Case-B
time.sleep(1)     # 等待Case-B运行
thread_a.start()  # 启动Case-A（应该抢占Case-B）

thread_a.join()
thread_b.join()
```

#### 步骤2: 增强监控

```python
#!/usr/bin/env python3
"""
增强监控：记录执行时间，验证抢占效果
"""

import torch
import time
import threading
from collections import defaultdict

# 记录每次执行时间
timings = defaultdict(list)
lock = threading.Lock()

def run_with_timing(name, func):
    start = time.perf_counter()
    result = func()
    elapsed = (time.perf_counter() - start) * 1000  # ms
    
    with lock:
        timings[name].append(elapsed)
    
    return result

def worker_a():
    for i in range(50):
        run_with_timing('Case-A', run_case_a)
        torch.cuda.synchronize()

def worker_b():
    for i in range(50):
        run_with_timing('Case-B', run_case_b)
        torch.cuda.synchronize()

# ... 运行并分析 ...

# 分析结果
import numpy as np

print("Case-A延迟统计:")
print(f"  平均: {np.mean(timings['Case-A']):.2f}ms")
print(f"  中位数: {np.median(timings['Case-A']):.2f}ms")
print(f"  P95: {np.percentile(timings['Case-A'], 95):.2f}ms")

print("\nCase-B延迟统计:")
print(f"  平均: {np.mean(timings['Case-B']):.2f}ms")
print(f"  中位数: {np.median(timings['Case-B']):.2f}ms")
print(f"  P95: {np.percentile(timings['Case-B'], 95):.2f}ms")

# 验证：Case-A应该延迟更低且更稳定
```

#### 步骤3: 完整调度器（如果需要更精确控制）

```python
#!/usr/bin/env python3
"""
完整GPU调度器：支持优先级、时间片、资源限制
"""

class GPUScheduler:
    def __init__(self):
        self.tasks = []
        self.running = False
    
    def register_task(self, name, func, priority, time_slice_ms=None):
        self.tasks.append({
            'name': name,
            'func': func,
            'priority': priority,
            'time_slice': time_slice_ms,
            'stream': torch.cuda.Stream(priority=priority)
        })
    
    def run(self):
        self.running = True
        
        # 按优先级排序
        self.tasks.sort(key=lambda x: x['priority'], reverse=True)
        
        threads = []
        for task in self.tasks:
            t = threading.Thread(target=self._run_task, args=(task,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
    
    def _run_task(self, task):
        with torch.cuda.stream(task['stream']):
            while self.running:
                task['func']()
                torch.cuda.synchronize()

# 使用示例
scheduler = GPUScheduler()
scheduler.register_task('Case-A', run_case_a, priority=-1)
scheduler.register_task('Case-B', run_case_b, priority=0)
scheduler.run()
```

---

## 测试计划

### 测试1: 基础抢占验证

**目标**: 验证高优先级Queue是否优先执行

**步骤**:
1. 启动Case-B（低优先级）
2. 等待1秒
3. 启动Case-A（高优先级）
4. 测量Case-A和Case-B的延迟

**预期结果**:
- Case-A延迟应该低于Case-B
- Case-A延迟应该稳定（方差小）
- Case-B延迟可能增加（被抢占）

### 测试2: 延迟对比

**目标**: 量化抢占对延迟的影响

**度量指标**:
| Metric | Case-A | Case-B |
|--------|--------|--------|
| 平均延迟 | ? ms | ? ms |
| P50延迟 | ? ms | ? ms |
| P95延迟 | ? ms | ? ms |
| P99延迟 | ? ms | ? ms |
| 抖动(std) | ? ms | ? ms |

### 测试3: 吞吐量对比

**目标**: 验证抢占是否影响总吞吐量

**度量指标**:
| Scenario | 总吞吐量 | Case-A吞吐 | Case-B吞吐 |
|----------|----------|-----------|-----------|
| 无抢占（顺序） | ? | ? | ? |
| 有抢占（并发） | ? | ? | ? |

### 测试4: 资源利用率

**目标**: 验证GPU是否充分利用

**度量指标**:
- GPU利用率（rocm-smi）
- 内存利用率
- Queue占用率

---

## 关键问题与解决方案

### Q1: ROCm 7.x中如何实现Queue优先级？

**答**:
```python
# PyTorch支持
stream = torch.cuda.Stream(priority=-1)  # -1是最高优先级

# 验证是否生效
# 方法1: 测量延迟（间接）
# 方法2: AMD_LOG_LEVEL=5查看Queue属性
```

### Q2: 如何监控抢占是否生效？

**答**:
1. **测量延迟**: Case-A延迟应该低且稳定
2. **AMD日志**: 查看Queue调度顺序
3. **rocm-smi**: 监控GPU利用率变化
4. **时间戳分析**: 记录每次Kernel提交和完成时间

### Q3: 如果PyTorch Stream优先级不生效怎么办？

**答**:
- **Plan B**: 使用时间片轮转
- **Plan C**: 修改KFD驱动（如果可能）
- **Plan D**: 使用XSched等用户态调度器

---

## 附录: 相关API和工具

### PyTorch Stream API

```python
# 创建Stream
stream = torch.cuda.Stream(priority=-1, device=0)

# 使用Stream
with torch.cuda.stream(stream):
    output = model(input)

# 同步
stream.synchronize()
stream.wait_stream(another_stream)
```

### ROCm监控工具

```bash
# 查看GPU使用
rocm-smi --showuse

# 查看进程
rocm-smi --showpids

# 持续监控
watch -n 1 'rocm-smi --showuse --showpids'
```

### KFD Debug API（如果可用）

```c
#include <linux/kfd_ioctl.h>

// 暂停Queue
kfd_ioctl_dbg_trap_suspend_queues_args suspend_args = {
    .num_queues = 1,
    .queue_ids = {queue_id},
};
ioctl(kfd_fd, KFD_IOC_DBG_TRAP_SUSPEND_QUEUES, &suspend_args);

// 恢复Queue
kfd_ioctl_dbg_trap_resume_queues_args resume_args = {
    .num_queues = 1,
    .queue_ids = {queue_id},
};
ioctl(kfd_fd, KFD_IOC_DBG_TRAP_RESUME_QUEUES, &resume_args);
```

---

**下一步**:
1. 运行Case-A和Case-B对比测试
2. 分析Queue使用差异
3. 实现基础抢占机制
4. 测试验证抢占效果

---

**维护者**: AI Assistant  
**日期**: 2026-02-05  
**状态**: 设计完成，待实现和测试
