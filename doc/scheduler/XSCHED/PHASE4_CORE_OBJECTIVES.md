# Phase 4: 多模型优先级调度核心目标

**日期**: 2026-01-28  
**核心目标**: 验证 XSched 在多 AI 模型场景下的优先级调度和 latency 保证

---

## 🎯 Phase 4 核心目标

### 主要验证点

```
1. 多个 AI 模型并发运行
   ├─ 高优先级模型（前台任务）
   ├─ 低优先级模型（后台任务）
   └─ 同时运行，竞争 GPU 资源

2. 优先级调度正确性
   ├─ 高优先级任务优先执行
   ├─ 低优先级任务不被饿死
   └─ 调度策略生效

3. Latency 保证
   ├─ 高优先级任务 P99 延迟低
   ├─ 优于 Native scheduler
   └─ 接近 standalone 性能

4. 抢占功能验证
   ├─ 高优先级任务到达时
   ├─ 能够抢占低优先级任务
   └─ 抢占延迟可接受
```

---

## 📊 测试场景设计

### 场景 1: 推理服务场景（核心）

**模拟生产环境的推理服务**

```
┌─────────────────────────────────────────────────────┐
│  高优先级任务（在线推理服务）                         │
│  - 模型: ResNet-18 (轻量级)                          │
│  - 请求频率: 10 reqs/sec                             │
│  - SLA 要求: P99 延迟 < 50ms                         │
│  - 优先级: HIGH (2)                                  │
└─────────────────────────────────────────────────────┘
                    ↓ 同时运行 ↓
┌─────────────────────────────────────────────────────┐
│  低优先级任务（离线训练 / 批处理）                    │
│  - 模型: ResNet-50 (更重)                            │
│  - 请求频率: 连续推理 (100% GPU)                     │
│  - 要求: 尽可能高吞吐量                              │
│  - 优先级: LOW (1)                                   │
└─────────────────────────────────────────────────────┘
```

**预期结果**:
- ✅ 高优先级 P99 延迟 < 50ms (接近 standalone)
- ✅ 低优先级仍能获得 GPU 时间（吞吐量 > 0）
- ✅ 总 GPU 利用率接近 100%

---

### 场景 2: 多租户场景

**多个用户/租户共享 GPU**

```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Tenant A        │  │  Tenant B        │  │  Tenant C        │
│  (Production)    │  │  (Development)   │  │  (Batch)         │
│                  │  │                  │  │                  │
│  Priority: 3     │  │  Priority: 2     │  │  Priority: 1     │
│  Model: ViT      │  │  Model: MobileNet│  │  Model: ResNet50 │
│  Latency: <100ms │  │  Latency: <500ms │  │  Best effort     │
└──────────────────┘  └──────────────────┘  └──────────────────┘
          ↓                    ↓                    ↓
    ┌─────────────────────────────────────────────────┐
    │              XSched 优先级调度器                  │
    │    - Tenant A 优先                               │
    │    - Tenant B 次之                               │
    │    - Tenant C 最低（但仍能执行）                 │
    └─────────────────────────────────────────────────┘
```

**预期结果**:
- ✅ Tenant A P99 < 100ms
- ✅ Tenant B P99 < 500ms
- ✅ Tenant C 吞吐量 > Standalone 的 20%

---

### 场景 3: 视频会议场景（实时 + 批处理）

**实时任务 + 后台任务**

```
┌─────────────────────────────────────────────────────┐
│  实时任务（视频会议背景虚化）                         │
│  - 模型: DeepLabV3 (轻量)                            │
│  - 帧率: 30 FPS (33ms/frame)                         │
│  - SLA: P99 < 40ms (不掉帧)                          │
│  - 优先级: HIGH (3)                                  │
└─────────────────────────────────────────────────────┘
                    ↓ 同时运行 ↓
┌─────────────────────────────────────────────────────┐
│  批处理任务（语音转文字）                             │
│  - 模型: Whisper-base                                │
│  - 周期: 每 3 秒处理一次                              │
│  - 要求: 3 秒内完成即可                              │
│  - 优先级: LOW (1)                                   │
└─────────────────────────────────────────────────────┘
```

**预期结果**:
- ✅ 视频处理 P99 < 40ms (不掉帧)
- ✅ 语音转文字在 3 秒内完成（不丢失内容）
- ✅ 两个任务都能正常工作

---

## 🔬 测试用例设计

### Test 4.1: 双模型优先级测试（基础）

**目标**: 验证基本的优先级调度

```python
# test_phase4_1_dual_model.py

import torch
import torchvision.models as models
import multiprocessing as mp
import time
import numpy as np

def high_priority_task(duration=60):
    """
    高优先级任务: ResNet-18 在线推理
    模拟: 10 reqs/sec，每个请求需要 ~20ms
    """
    # TODO: 设置优先级
    # XHintPriority(2)
    
    model = models.resnet18(weights=None).cuda()
    model.eval()
    
    x = torch.randn(1, 3, 224, 224).cuda()
    
    latencies = []
    start_time = time.time()
    
    while (time.time() - start_time) < duration:
        req_start = time.time()
        
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        
        latency = (time.time() - req_start) * 1000  # ms
        latencies.append(latency)
        
        # 10 reqs/sec = 100ms 间隔
        time.sleep(0.1)
    
    # 统计
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    avg = np.mean(latencies)
    
    print(f"[HIGH PRIORITY]")
    print(f"  Requests: {len(latencies)}")
    print(f"  Avg latency: {avg:.2f} ms")
    print(f"  P50 latency: {p50:.2f} ms")
    print(f"  P99 latency: {p99:.2f} ms")
    
    return p99

def low_priority_task(duration=60):
    """
    低优先级任务: ResNet-50 连续推理
    模拟: 批处理任务，尽可能高吞吐
    """
    # TODO: 设置优先级
    # XHintPriority(1)
    
    model = models.resnet50(weights=None).cuda()
    model.eval()
    
    x = torch.randn(8, 3, 224, 224).cuda()  # 更大批量
    
    count = 0
    start_time = time.time()
    
    while (time.time() - start_time) < duration:
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        count += 1
    
    elapsed = time.time() - start_time
    throughput = count / elapsed
    
    print(f"[LOW PRIORITY]")
    print(f"  Iterations: {count}")
    print(f"  Throughput: {throughput:.2f} iter/s")
    
    return throughput

if __name__ == '__main__':
    print("=" * 60)
    print("Phase 4 Test 4.1: Dual Model Priority Test")
    print("=" * 60)
    
    # 启动两个进程
    with mp.Pool(2) as pool:
        results = pool.starmap(
            lambda f, d: f(d),
            [(high_priority_task, 60), (low_priority_task, 60)]
        )
    
    high_p99, low_throughput = results
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  High Priority P99: {high_p99:.2f} ms")
    print(f"  Low Priority Throughput: {low_throughput:.2f} iter/s")
    print("=" * 60)
    
    # 判断
    # TODO: 需要 baseline 数据进行对比
    # 目前只是功能性测试
    if high_p99 < 100:  # 宽松标准
        print("✅ PASS: High priority latency acceptable")
    else:
        print("❌ FAIL: High priority latency too high")
```

**运行**:
```bash
# 1. Baseline (无 XSched)
unset LD_PRELOAD
python test_phase4_1_dual_model.py > baseline.txt

# 2. With XSched
export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so
python test_phase4_1_dual_model.py > xsched.txt

# 3. 对比
python compare_results.py baseline.txt xsched.txt
```

---

### Test 4.2: 三模型优先级测试（多租户）

**目标**: 验证多优先级层次

```python
# test_phase4_2_multi_tenant.py

def tenant_a_task():  # Priority 3 (Highest)
    model = models.vit_b_16(weights=None).cuda()
    # 测量 latency

def tenant_b_task():  # Priority 2 (Medium)
    model = models.mobilenet_v2(weights=None).cuda()
    # 测量 latency

def tenant_c_task():  # Priority 1 (Lowest)
    model = models.resnet50(weights=None).cuda()
    # 测量 throughput

# 同时运行 3 个租户
```

**预期**:
- ✅ Tenant A P99 < 100ms (最高优先级)
- ✅ Tenant B P99 < 500ms (中等优先级)
- ✅ Tenant C 吞吐量 > 0 (最低优先级不饿死)

---

### Test 4.3: 实时 + 批处理测试

**目标**: 验证实时任务的 latency 保证

```python
# test_phase4_3_realtime_batch.py

def realtime_task():
    """
    实时任务: 30 FPS 视频处理
    要求: P99 < 40ms
    """
    model = create_lightweight_model().cuda()
    
    for frame in range(30 * 60):  # 60 秒
        start = time.time()
        process_frame(model, frame)
        latency = (time.time() - start) * 1000
        
        # 记录延迟
        latencies.append(latency)
        
        # 30 FPS = 33.3ms 间隔
        time.sleep(0.033)

def batch_task():
    """
    批处理任务: 每 3 秒一次
    要求: 3 秒内完成
    """
    model = create_heavy_model().cuda()
    
    while True:
        start = time.time()
        process_batch(model)
        duration = time.time() - start
        
        if duration > 3.0:
            print("⚠️  Batch task exceeded deadline!")
        
        time.sleep(3.0)
```

---

## 📊 关键指标

### 1. Latency (延迟)

```
高优先级任务延迟:
  - P50, P99, P999
  - 与 standalone 对比
  - 与 Native scheduler 对比
```

### 2. Throughput (吞吐量)

```
低优先级任务吞吐量:
  - 绝对值 (iter/s)
  - 与 standalone 的比例
  - 是否被饿死
```

### 3. GPU Utilization (GPU 利用率)

```
整体 GPU 使用率:
  - 应接近 100%
  - 无空闲浪费
```

### 4. Fairness (公平性)

```
资源分配:
  - 是否符合优先级设置
  - 低优先级是否仍能获得资源
```

---

## 🎯 成功标准

### Phase 4 Test 4.1 (双模型)

| 指标 | Baseline | Native | XSched 目标 |
|------|---------|--------|------------|
| 高优先级 P99 | 20ms | 80ms | < 30ms |
| 低优先级吞吐 | 100% | 50% | > 30% |
| 总 GPU 利用率 | 50% | 90% | > 90% |

### Phase 4 Test 4.2 (多租户)

| Tenant | Priority | Latency 目标 | Throughput 目标 |
|--------|----------|-------------|----------------|
| A | 3 | P99 < 100ms | - |
| B | 2 | P99 < 500ms | - |
| C | 1 | - | > 20% standalone |

### Phase 4 Test 4.3 (实时+批处理)

| 任务 | 指标 | 目标 |
|------|------|------|
| 实时任务 | P99 延迟 | < 40ms (不掉帧) |
| 批处理任务 | 完成时间 | < 3s (不丢内容) |

---

## 🚀 实施计划

### Week 1: 基础双模型测试

```bash
Day 1-2: 
  - 验证已有 XSched 环境
  - 创建 Test 4.1 脚本
  - 收集 baseline 数据

Day 3-4:
  - 运行 XSched 测试
  - 对比分析
  - 调试问题（如果有）

Day 5:
  - 文档整理
  - 准备 Test 4.2
```

### Week 2: 多租户和实时场景

```bash
Day 1-3:
  - 实现 Test 4.2 (三租户)
  - 实现 Test 4.3 (实时+批处理)

Day 4-5:
  - 数据分析
  - 性能优化（如果需要）
  - 最终报告
```

---

## 📝 利用已有成果

### Phase 2-3 的模型（已验证可用）

**Phase 3 测试结果**: 13/14 测试通过 (92.9%) ✅  
**详细报告**: [PHASE3_TEST_RESULTS.md](PHASE3_TEST_RESULTS.md)

```python
✅ 已测试成功的模型（推理）:
  - ResNet-18, ResNet-50              ✅
  - MobileNetV2                        ✅
  - EfficientNet-B0                    ✅
  - Vision Transformer (ViT-B/16)      ✅
  - DenseNet-121                       ✅
  - VGG-16, AlexNet, SqueezeNet       ✅

✅ 已测试成功（训练）:
  - ResNet-18 Training                 ✅
  - MobileNetV2 Training               ✅

✅ 已测试成功（批处理）:
  - ResNet-50 Batch=32                 ✅
  - EfficientNet Batch=16              ✅

✅ 已有的测试框架:
  - TEST_REAL_MODELS.sh
  - BENCHMARK.sh
  - 性能测量工具
```

### 已有的 XSched 环境

```bash
✅ XSched 路径:
  /data/dockercode/xsched-official
  /data/dockercode/xsched-build
  /data/dockercode/xsched-build/output

✅ 关键修复:
  - Symbol Versioning (hip_version.map)
  - PyTorch 集成

✅ 环境设置:
  export LD_LIBRARY_PATH=.../output/lib:$LD_LIBRARY_PATH
  export LD_PRELOAD=.../output/lib/libshimhip.so
```

---

## 🎉 Phase 4 的价值

### 技术价值

1. **验证 XSched 在 AI 场景的有效性**
   - 多模型并发
   - 优先级调度
   - Latency 保证

2. **补充论文未涉及的内容**
   - PyTorch + XSched
   - 真实 AI 模型（不只是 micro-benchmark）
   - MI308X 平台数据

3. **生产环境参考**
   - 推理服务场景
   - 多租户场景
   - 实时应用场景

### 学术价值

1. 可以发表技术报告
2. 补充论文实验数据
3. 为社区提供 AI + XSched 的案例

---

## 🚀 立即开始

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED

# Step 1: 验证环境
./run_phase4_test1.sh

# Step 2: 创建双模型测试（明天）
# 编写 test_phase4_1_dual_model.py
```

---

**Phase 4 核心**: 多 AI 模型 + 优先级调度 + Latency 保证 🎯
