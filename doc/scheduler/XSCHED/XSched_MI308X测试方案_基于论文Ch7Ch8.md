# XSched on AMD MI308X: 测试方案设计（基于论文Chapter 7 & 8）

> **📌 文档说明**  
> 本测试方案严格基于XSched论文Chapter 7（Experimental Evaluation）和Chapter 8（Case Studies）的实验设计，针对AMD MI308X GPU进行适配和扩展。
> - **论文版本**：XSched: Preemptive Scheduling for Diverse XPUs (OSDI 2025)
> - **目标硬件**：AMD Instinct MI308X (gfx942)
> - **测试环境**：Docker容器 `zhenaiter`，ROCm 6.4+
> - **创建日期**：2026-01-27

---

## 📋 目录

1. [测试目标与分层设计](#1-测试目标与分层设计)
2. [Chapter 7 测试用例](#2-chapter-7-测试用例)
3. [Chapter 8 测试用例](#3-chapter-8-测试用例)
4. [AMD特有测试](#4-amd特有测试)
5. [实施计划](#5-实施计划)

---

## 1. 测试目标与分层设计

### 1.1 论文验证的三大核心特性

基于论文Chapter 7的组织结构：

```
┌─────────────────────────────────────────────────────────────┐
│  7.1 Portability（可移植性）                                 │
│  - 验证XSched在AMD MI308X上的成功适配                        │
│  - 代码量：841 LoC (HIP平台，论文Table 3)                    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────────┐
│  7.2 Uniformity（统一性）                                      │
│  - Fixed Priority Policy（固定优先级）                         │
│  - Bandwidth Partition Policy（带宽分区）                      │
│  - Heterogeneous XPU Coordination（异构协同，未来）            │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────────┐
│  7.3 Evolvability（可演进性）                                  │
│  - Lv1 基准测试（AMD MI308X当前支持）                         │
│  - Lv3 扩展测试（AMD CWSR，MI308X硬件支持）                   │
│  - 抢占延迟分析（不同命令执行时间）                           │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────────┐
│  7.4 Scheduling Overhead（调度开销）                          │
│  - Runtime overhead（运行时开销 < 3.4%）                      │
│  - CPU overhead（单核CPU使用率增加 < 5%）                     │
│  - In-flight command threshold调优                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 测试层次设计

| 测试层次 | 论文章节 | 测试目标 | MI308X硬件级别 | 预期结果 |
|---------|---------|---------|---------------|---------|
| **L1: 基础功能** | 7.1 | 编译、运行、基本抢占 | Lv1 | 成功适配 |
| **L2: 调度策略** | 7.2 | 多优先级、带宽分区 | Lv1 | P99延迟 < 1.30× |
| **L3: 性能评估** | 7.3 | 抢占延迟、硬件级别 | Lv1 → Lv3 | 开销 < 3.4% |
| **L4: 实际应用** | 8.1-8.3 | 推理服务、多租户 | Lv1/Lv3 | 生产可用 |

---

## 2. Chapter 7 测试用例

### 2.1 Portability Tests（可移植性测试）

#### Test 7.1.1: XSched on AMD MI308X - 基础适配验证

**目标**：验证XSched在AMD MI308X上的编译和基本运行能力

**参考**：论文Table 3 - AMD GPUs行

| 指标 | 论文值 | MI308X目标值 |
|-----|-------|------------|
| XShim LoC | 316 | 验证 |
| Lv1 LoC | 841 | 验证 |
| Lv2 支持 | ❌（论文未实现） | ✅（潜在，基于flushing） |
| Lv3 支持 | ❌（论文未实现） | ✅（CWSR，本项目重点） |

**测试步骤**：

```bash
# 1. 编译XSched HIP平台支持
cd /workspace/xsched
export CXXFLAGS='-Wno-error=maybe-uninitialized'
make hip

# 2. 验证基础example运行
cd examples/Linux/1_transparent_sched
make hip
export LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib:/opt/rocm/lib:$LD_LIBRARY_PATH
./app

# 3. 收集输出
# - Task execution time
# - XSched overhead (应接近论文MI50的1.7%)
```

**成功标准**：
- ✅ 编译无错误
- ✅ 运行无崩溃
- ✅ Runtime overhead < 3.4%（论文Lv1上限）

---

### 2.2 Uniformity Tests（统一性测试）

#### Test 7.2.1: Fixed Priority Policy - 前台/后台任务调度

**目标**：复现论文Fig. 9 (top)的固定优先级实验

**参考**：
- 论文实验设置：
  - 前台任务：周期性提交（20% peak throughput）
  - 后台任务：连续提交（100% peak throughput）
  - 前台高优先级，后台低优先级

**Workload（论文7.2节）**：
- GPU/NPU：ResNet-152推理
- AMD MI308X：ResNet-152推理（PyTorch + HIP backend）

**测试代码**：基于`examples/Linux/3_intra_process_sched/app_concurrent.hip`

```bash
# 运行多优先级抢占测试
cd /workspace/xsched/examples/Linux/3_intra_process_sched
make hip
./app_concurrent

# 收集指标：
# - 前台任务P99延迟
# - 后台任务吞吐量
```

**性能指标对比**：

| 指标 | 论文MI50 | MI308X目标值 |
|-----|---------|------------|
| 前台P99延迟（Standalone） | 基准 | 基准 |
| 前台P99延迟（Native） | 1.60× ~ 2.19× | < 2.0× |
| 前台P99延迟（XSched） | 1.02× ~ 1.30× | < 1.30× |
| Runtime overhead | 1.7% | < 3.4% |

**成功标准**：
- ✅ 前台P99延迟 < 1.30× standalone
- ✅ 后台任务仍能获得执行机会
- ✅ 优于Native hardware scheduler

---

#### Test 7.2.2: Bandwidth Partition Policy - 带宽分区测试

**目标**：复现论文Fig. 9 (bottom)的带宽分区实验

**参考**：
- 论文实验设置：
  - 前台进程：连续提交任务（max frequency）
  - 后台进程：连续提交任务（max frequency）
  - XSched分配：75% XPU利用率给前台，25%给后台

**测试步骤**：

1. **修改XSched调度策略**（需要实现或修改现有代码）：

```cpp
// 在XScheduler中设置带宽分区
XHintSetScheduler(kSchedulerLocal, kPolicyBandwidthPartition);
XHintSetBandwidthRatio(fg_xqueue, 0.75); // 前台75%
XHintSetBandwidthRatio(bg_xqueue, 0.25); // 后台25%
```

2. **运行测试**：

```bash
# 编写新的测试程序：test_bandwidth_partition.hip
# 两个进程同时运行ResNet-152推理
# 测量：
# - 前台吞吐量 (normalized by standalone)
# - 后台吞吐量 (normalized by standalone)
# - 总吞吐量 vs. Native scheduler
```

**性能指标对比**：

| 配置 | 前台吞吐量 | 后台吞吐量 | 总吞吐量 | 分配比例 |
|-----|----------|----------|---------|---------|
| Standalone (Fg) | 1.0 | - | 1.0 | 100% |
| Standalone (Bg) | - | 1.0 | 1.0 | 100% |
| Native | ~0.50 | ~0.50 | ~1.0 | 50%/50% |
| XSched (目标) | ~0.75 | ~0.25 | ~1.0 | 75%/25% |

**成功标准**：
- ✅ 吞吐量分配比例 ≈ 75:25
- ✅ 总开销 < 1.5%（论文平均值）
- ✅ 总吞吐量 ≈ Standalone

---

### 2.3 Evolvability Tests（可演进性测试）

#### Test 7.3.1: Preemption Latency - 不同硬件级别对比

**目标**：复现论文Fig. 11 (a)，对比Lv1和Lv3的抢占延迟

**参考**：
- 论文实验设置：
  - 被抢占任务：持续启动执行时间为T=0.5ms的命令
  - In-flight command threshold = 8
  - 测量P99抢占延迟

**理论预期（基于论文）**：

| 硬件级别 | 论文GV100 | 论文K40m | 论文NPU3720 | MI308X预期 |
|---------|----------|---------|------------|-----------|
| Lv1 | ~4ms (8T) | ~4ms (8T) | ~4ms (8T) | ~4ms (8T) |
| Lv2 | - | ~0.5ms (1T) | ~0.5ms (1T) | 待实现 |
| Lv3 | 32μs | - | - | **< 100μs** (CWSR) |

**测试步骤**：

1. **Lv1基准测试**（当前XSched实现）：

```cpp
// 测试程序：test_preemption_latency_lv1.hip
// 1. 低优先级任务持续提交0.5ms的kernel
// 2. 高优先级任务周期性到达
// 3. 测量抢占延迟（高优先级任务实际开始执行时间 - 预期开始时间）

__global__ void delay_kernel(int iterations) {
    // 精确控制执行时间为0.5ms
    clock_t start = clock64();
    while ((clock64() - start) < iterations) {}
}

void run_lv1_test() {
    // 设置in-flight threshold = 8
    XQueueSetLaunchConfig(lp_xqueue, 8, 4);
    
    // 测量P99抢占延迟
    std::vector<double> preemption_latencies;
    // ... 收集数据 ...
}
```

2. **Lv3扩展测试**（未来，基于CWSR）：

```cpp
// 测试程序：test_preemption_latency_lv3.hip
// 使用CWSR的interrupt()接口实现Lv3抢占

#include <linux/kfd_ioctl.h>

void interrupt_lv3(uint32_t queue_id) {
    int kfd_fd = open("/dev/kfd", O_RDWR);
    struct kfd_ioctl_preempt_queue_args args = {
        .queue_id = queue_id,
        .preempt_type = KFD_PREEMPT_TYPE_WAVEFRONT_SAVE, // Lv3
        .timeout_ms = 1000
    };
    ioctl(kfd_fd, AMDKFD_IOC_PREEMPT_QUEUE, &args);
    close(kfd_fd);
}

void run_lv3_test() {
    // 测量P99抢占延迟（目标 < 100μs）
    std::vector<double> preemption_latencies;
    // ... 使用CWSR接口 ...
}
```

**成功标准**：
- ✅ Lv1 P99延迟 ≈ 8T (约4ms，T=0.5ms)
- ✅ Lv3 P99延迟 < 100μs（AMD CWSR硬件能力）
- ✅ Lv3延迟独立于命令执行时间T（论文Fig. 11b特性）

---

#### Test 7.3.2: Command Execution Time Impact - 不同执行时间的抢占延迟

**目标**：复现论文Fig. 11 (b)，测试不同命令执行时间对抢占延迟的影响

**参考**：
- 论文测试范围：命令执行时间从0.01ms到2ms
- Lv1：延迟随T线性增长
- Lv3：延迟保持恒定（~32μs）

**测试步骤**：

```cpp
// 测试程序：test_exec_time_impact.hip

void test_different_exec_times() {
    std::vector<double> exec_times = {0.01, 0.1, 0.5, 1.0, 2.0}; // ms
    
    for (auto T : exec_times) {
        // 1. 配置kernel执行时间为T
        int iterations = calculate_iterations_for_time(T);
        
        // 2. 运行抢占测试
        auto latency = measure_preemption_latency(iterations);
        
        // 3. 记录结果
        printf("T=%.2f ms, P99 Latency=%.2f ms\n", T, latency);
    }
}
```

**预期结果**：

| 命令执行时间 T | Lv1 P99延迟 | Lv3 P99延迟 |
|--------------|------------|-----------|
| 0.01 ms | ~0.08 ms (8T) | < 0.1 ms |
| 0.1 ms | ~0.8 ms (8T) | < 0.1 ms |
| 0.5 ms | ~4 ms (8T) | < 0.1 ms |
| 1.0 ms | ~8 ms (8T) | < 0.1 ms |
| 2.0 ms | ~16 ms (8T) | < 0.1 ms |

**成功标准**：
- ✅ Lv1延迟 ≈ 8T（验证progressive launching机制）
- ✅ Lv3延迟恒定（验证CWSR硬件抢占能力）

---

#### Test 7.3.3: In-flight Command Threshold - 阈值调优

**目标**：复现论文Fig. 11 (c)，分析in-flight command threshold对开销的影响

**参考**：
- 论文实验：threshold从1到10
- 命令执行时间：0.01ms, 0.1ms, 1ms
- 开销目标：< 3%

**测试步骤**：

```cpp
// 测试程序：test_threshold_tuning.hip

void test_threshold_impact() {
    std::vector<int> thresholds = {1, 2, 4, 6, 8, 10};
    std::vector<double> exec_times = {0.01, 0.1, 1.0}; // ms
    
    for (auto threshold : thresholds) {
        for (auto T : exec_times) {
            // 1. 设置threshold
            XQueueSetLaunchConfig(xqueue, threshold, 4);
            
            // 2. 运行任务，测量runtime overhead
            auto overhead = measure_runtime_overhead();
            
            // 3. 记录结果
            printf("Threshold=%d, T=%.2f ms, Overhead=%.2f%%\n",
                   threshold, T, overhead);
        }
    }
}
```

**预期结果（论文Fig. 11c）**：

| Threshold | T=0.01ms开销 | T=0.1ms开销 | T=1ms开销 |
|-----------|-------------|-----------|----------|
| 1 | 30% | 10% | 2% |
| 2 | 20% | 6% | 1.5% |
| 4 | 10% | 3% | 1% |
| 6 | 5% | 2% | 0.8% |
| 8 | 3% | 1.5% | 0.7% |
| 10 | < 1% | < 1% | < 1% |

**成功标准**：
- ✅ Threshold ≥ 10时，开销 < 1%
- ✅ 找到最佳threshold：最小开销同时保证抢占延迟可接受

---

### 2.4 Scheduling Overhead Tests（调度开销测试）

#### Test 7.4.1: Runtime Overhead - 运行时开销测量

**目标**：复现论文Fig. 12 (a)，测量XSched的运行时开销

**参考**：
- 论文MI50：Lv1开销 = 1.7%
- 论文上限：Lv1开销 < 3.4%

**测试步骤**：

```bash
# 1. Baseline: 不使用XSched运行任务
cd /workspace/xsched/examples/Linux/1_transparent_sched
# 修改Makefile，使用native HIP API
./app_native  # 记录执行时间 T_native

# 2. XSched: 使用XSched运行同样任务
./app  # 记录执行时间 T_xsched

# 3. 计算开销
Runtime_Overhead = (T_xsched - T_native) / T_native * 100%
```

**测试工作负载**：
- 论文7.2节workload：ResNet-152推理
- MI308X适配：使用PyTorch + HIP backend

**成功标准**：
- ✅ Lv1 Runtime overhead < 3.4%
- ✅ 接近论文MI50的1.7%（目标）

---

#### Test 7.4.2: CPU Overhead - CPU使用率测量

**目标**：复现论文Fig. 12 (b)，测量XSched增加的CPU使用率

**参考**：
- 论文MI50：单核CPU使用率增加3.6%
- 论文上限：< 5%（大多数情况）

**测试步骤**：

```bash
# 1. 使用top/htop监控CPU使用率
# Baseline: 不使用XSched
top -p $(pgrep app_native) -d 1

# 2. XSched: 使用XSched
top -p $(pgrep app) -d 1

# 3. 计算增加的CPU使用率
CPU_Overhead = CPU_xsched - CPU_native
```

**成功标准**：
- ✅ 单核CPU使用率增加 < 5%
- ✅ 无spinning行为（AMD驱动问题，论文910b/PVA有18.3%/11.9%）

---

## 3. Chapter 8 测试用例

### 3.1 Case Study 1: GPU Harvesting on Multi-Tenant Server

#### Test 8.1.1: Production + Opportunistic Jobs - DL训练共存

**目标**：复现论文Fig. 13左侧，生产任务和机会任务共存

**参考**：
- 论文实验：
  - Production job (Pjob)：DL训练（严格性能要求）
  - Opportunistic job (Ojob)：DL训练（尽力而为）
  - 对比系统：Native, vCUDA, TGS, XSched

**Workload**：
- 两个Docker容器运行PyTorch训练任务
- Pjob：ResNet-50训练（高优先级）
- Ojob：ResNet-50训练（低优先级）

**测试步骤**：

1. **环境准备**：

```bash
# 1. 创建两个Docker容器或两个进程
# Container 1: Production job
docker run --name pjob --gpus all -d pytorch/pytorch:rocm python train_resnet50.py

# Container 2: Opportunistic job
docker run --name ojob --gpus all -d pytorch/pytorch:rocm python train_resnet50.py
```

2. **性能测量**：

```python
# train_resnet50.py - 修改版
import torch
import time
from xsched import XQueue, XHintPriority  # 假设Python binding

def train_with_priority(priority='high'):
    model = torchvision.models.resnet50()
    # ... 训练循环 ...
    
    if priority == 'high':
        XHintPriority(2)  # Production job
    else:
        XHintPriority(1)  # Opportunistic job
    
    # 测量吞吐量和训练时间
    start_time = time.time()
    for epoch in range(10):
        train_one_epoch(model)
    duration = time.time() - start_time
    print(f"Duration: {duration}s")
```

**性能指标对比（论文Fig. 13）**：

| 系统 | Pjob性能 | Ojob性能 | 总利用率 |
|------|---------|---------|---------|
| Native | 0.50 | 0.50 | 1.0 |
| vCUDA | 0.85 | 0.15 | 1.0（需预配置quota） |
| TGS | 0.93 | 0.07 | 1.0 |
| **XSched (目标)** | **0.99** | **0.20** | **1.0** |

**成功标准**：
- ✅ Pjob性能 > 0.95（接近Standalone）
- ✅ Ojob仍能获得GPU资源（> 10%）
- ✅ 优于TGS的资源利用

---

#### Test 8.1.2: Production + Opportunistic Jobs - 金融算法 + 科学计算

**目标**：复现论文Fig. 13右侧，异构工作负载共存

**Workload**：
- Pjob：Financial algorithms（Black-Scholes期权定价）
- Ojob：Scientific computing（CFD流体力学仿真）

**测试步骤**：

1. **Black-Scholes实现**（AMD HIP版本）：

```cpp
// black_scholes.hip
__global__ void BlackScholesGPU(float *d_Call, float *d_Put,
                                 float *d_S, float *d_X, float *d_T,
                                 float R, float V, int optN) {
    const int opt = blockDim.x * blockIdx.x + threadIdx.x;
    if (opt < optN) {
        float S = d_S[opt];
        float X = d_X[opt];
        float T = d_T[opt];
        // ... Black-Scholes公式计算 ...
        d_Call[opt] = call_value;
        d_Put[opt] = put_value;
    }
}
```

2. **CFD实现**（使用Rodinia benchmark suite）：

```bash
# 下载Rodinia for HIP
git clone https://github.com/AMDComputeLibraries/Rodinia_HIP.git
cd Rodinia_HIP/opencl/cfd
make
```

3. **共存测试**：

```bash
# Terminal 1: Production job (高优先级)
./black_scholes --priority high --requests 1000

# Terminal 2: Opportunistic job (低优先级)
./cfd --priority low --continuous
```

**性能指标对比（论文Fig. 13）**：

| 系统 | Pjob延迟 | Ojob吞吐量 |
|------|---------|-----------|
| Native | 1.0x | 0.5 |
| vCUDA | 1.80x | 0.15 |
| TGS | 1.70x | 0.0 (失败) |
| **XSched (目标)** | **1.01x** | **0.20** |

**成功标准**：
- ✅ Pjob延迟 < 1.05× Standalone
- ✅ Ojob仍能获得GPU资源
- ✅ 不受工作负载类型限制（优于TGS）

---

### 3.2 Case Study 2: Video Conferencing on AI PC

#### Test 8.2.1: Fake-Background + Speech-to-Text - 实时视频会议

**目标**：复现论文Fig. 14，实时视频会议场景的帧延迟优化

**参考**：
- 论文实验（Intel NPU3720）：
  - LFBW（Fake-background）：25 FPS，延迟敏感
  - whisper.cpp（Speech-to-text）：每3秒，周期性
  - Native：P99帧延迟 = 880ms（20.12× standalone）
  - XSched (laxity-based)：P99帧延迟 = 95ms（9.26× improvement）

**AMD MI308X适配**：
- 使用GPU替代NPU
- Fake-background：使用DeepLabV3+进行背景分割
- Speech-to-text：使用Whisper模型（HIP backend）

**测试步骤**：

1. **Fake-Background实现**：

```python
# fake_background.py
import torch
import cv2
import time
from xsched import XHintPriority, XHintSetScheduler, kPolicyLaxityBased

def run_fake_background():
    model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet50', pretrained=True)
    model = model.to('cuda')
    
    XHintPriority(2)  # 高优先级
    XHintSetScheduler(kPolicyLaxityBased)
    
    cap = cv2.VideoCapture(0)
    frame_latencies = []
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        
        # 背景分割
        output = model(preprocess(frame))
        blurred_frame = apply_blur(frame, output)
        
        # 显示
        cv2.imshow('Fake Background', blurred_frame)
        
        # 记录帧延迟
        frame_latency = time.time() - start_time
        frame_latencies.append(frame_latency)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 输出P99延迟
    print(f"P99 Frame Latency: {np.percentile(frame_latencies, 99) * 1000:.2f} ms")
```

2. **Speech-to-Text实现**：

```python
# speech_to_text.py
import whisper
import pyaudio
import time
from xsched import XHintPriority

def run_speech_to_text():
    model = whisper.load_model("base").to('cuda')
    
    XHintPriority(1)  # 低优先级，但有deadline
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paFloat32, channels=1, rate=16000, input=True)
    
    while True:
        # 每3秒录音
        audio_chunk = stream.read(16000 * 3)
        
        # 转录（有3秒deadline）
        start_time = time.time()
        result = model.transcribe(audio_chunk)
        duration = time.time() - start_time
        
        print(f"Transcription: {result['text']}, Time: {duration:.2f}s")
        
        if duration > 3.0:
            print("WARNING: Missed deadline!")
```

3. **性能测试**：

```bash
# Terminal 1: Fake-Background (25 FPS)
python fake_background.py

# Terminal 2: Speech-to-Text (每3秒)
python speech_to_text.py

# 测量：
# - Fake-Background的P99帧延迟
# - Speech-to-Text的完成时间（应 < 3秒）
```

**性能指标对比（论文Fig. 14）**：

| 调度策略 | LFBW P99延迟 | Whisper完成时间 |
|---------|-------------|---------------|
| Native | 880 ms | < 3s（但LFBW掉帧） |
| XSched (Fixed Priority) | 40 ms | > 3s（丢失内容） |
| **XSched (Laxity-based)** | **95 ms** | **< 3s** |

**成功标准**：
- ✅ LFBW P99延迟 < 100ms（保证25 FPS）
- ✅ Whisper完成时间 < 3s（无内容丢失）
- ✅ 优于Native scheduler的9× improvement

**注意**：此测试需要实现Laxity-based policy（论文104 LoC），或使用deadline-aware scheduling。

---

### 3.3 Case Study 3: Multi-Model Inference Serving

#### Test 8.3.1: Triton Integration - 多模型推理服务

**目标**：复现论文Fig. 15 (a)，集成XSched到Triton Inference Server

**参考**：
- 论文实验：
  - 两个Bert-large模型
  - 高优先级客户端：10 reqs/sec
  - 低优先级客户端：连续发送请求
  - Vanilla Triton：P99延迟 = 1.53× standalone
  - T+XSched：P99延迟 = 1.07× standalone

**AMD MI308X适配**：
- 使用Triton的PyTorch Backend（支持ROCm）
- 或使用ONNX Runtime（支持ROCm）

**测试步骤**：

1. **Triton Server配置**（AMD GPU版本）：

```bash
# 1. 启动Triton Server (ROCm版本)
docker pull nvcr.io/nvidia/tritonserver:23.10-py3  # 需要找到ROCm版本
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/model_repository:/models \
    tritonserver --model-repository=/models

# 2. 模型配置：model_repository/bert_large/config.pbtxt
name: "bert_large"
platform: "pytorch_libtorch"
max_batch_size: 8
instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

2. **XSched集成**（论文仅需10 LoC）：

```python
# 修改Triton的Backend代码（伪代码）
# triton-inference-server/src/backends/backend/triton/api.cc

#include "xsched/xsched.h"

TRITONSERVER_Error* ModelInstanceExecute(...) {
    // 原始代码
    // ...
    
    // 新增：提交调度hint到XSched
    auto priority = model_config.GetPriority();  // 从模型配置读取优先级
    XHintPriority(xqueue, priority);
    
    // 继续执行推理
    // ...
}
```

3. **客户端测试**：

```python
# client_high_priority.py
import tritonclient.http as httpclient
import time

client = httpclient.InferenceServerClient(url="localhost:8000")

# 高优先级客户端：10 reqs/sec
for i in range(100):
    start_time = time.time()
    result = client.infer(model_name="bert_large_high", inputs=...)
    latency = time.time() - start_time
    print(f"Request {i}: {latency * 1000:.2f} ms")
    time.sleep(0.1)  # 10 reqs/sec
```

```python
# client_low_priority.py
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8000")

# 低优先级客户端：连续发送
while True:
    result = client.infer(model_name="bert_large_low", inputs=...)
```

**性能指标对比（论文Fig. 15a）**：

| 配置 | 高优先级P99延迟 | 与Standalone对比 |
|------|---------------|----------------|
| Standalone | 基准 | 1.0× |
| Vanilla Triton | +53% | 1.53× |
| T+Priority | +51% | 1.51×（Triton优先级无效） |
| **T+XSched (目标)** | **+7%** | **1.07×** |

**成功标准**：
- ✅ 高优先级P99延迟 < 1.10× standalone
- ✅ 优于Vanilla Triton的30% improvement
- ✅ 低优先级任务仍能执行

---

#### Test 8.3.2: Paella Comparison - 高吞吐量推理服务

**目标**：复现论文Fig. 15 (b)，与Paella系统对比吞吐量-延迟曲线

**参考**：
- 论文实验：
  - 工作负载：log-normal分布（σ=2.0）
  - 调度策略：K-EDF (K-earliest deadline first, K=16)
  - 吞吐量范围：100 ~ 1200 reqs/sec
  - XSched在1000 reqs/sec时优于Paella 1.3×

**测试步骤**：

1. **K-EDF策略实现**（论文200 LoC）：

```cpp
// k_edf_scheduler.cpp
class KEDFScheduler {
public:
    void SubmitRequest(XQueue* xq, Request req, double deadline) {
        requests_.push_back({xq, req, deadline});
        
        // 按deadline排序
        std::sort(requests_.begin(), requests_.end(),
                  [](const auto& a, const auto& b) {
                      return a.deadline < b.deadline;
                  });
        
        // 只执行前K个
        for (int i = 0; i < std::min(K, requests_.size()); i++) {
            XQueueLaunch(requests_[i].xq, requests_[i].req);
        }
    }
    
private:
    static constexpr int K = 16;
    std::vector<RequestInfo> requests_;
};
```

2. **负载生成器**（log-normal分布）：

```python
# load_generator.py
import numpy as np
import tritonclient.http as httpclient
import time

def generate_lognormal_load(mean_rps, sigma=2.0, duration=60):
    client = httpclient.InferenceServerClient(url="localhost:8000")
    
    inter_arrival_times = np.random.lognormal(mean=np.log(1.0/mean_rps), sigma=sigma, size=1000)
    
    latencies = []
    start_time = time.time()
    
    for interval in inter_arrival_times:
        if time.time() - start_time > duration:
            break
        
        time.sleep(interval)
        
        req_start = time.time()
        result = client.infer(model_name="bert_large", inputs=...)
        latency = time.time() - req_start
        latencies.append(latency)
    
    p99_latency = np.percentile(latencies, 99)
    throughput = len(latencies) / duration
    print(f"Throughput: {throughput:.2f} reqs/sec, P99 Latency: {p99_latency*1000:.2f} ms")
```

3. **吞吐量-延迟曲线测试**：

```bash
# 测试不同吞吐量
for rps in 100 200 400 600 800 1000 1200; do
    echo "Testing throughput: $rps reqs/sec"
    python load_generator.py --rps $rps --duration 60
done
```

**性能指标对比（论文Fig. 15b）**：

| 吞吐量 (reqs/sec) | Paella P99延迟 | XSched P99延迟 | 改进 |
|------------------|--------------|--------------|------|
| 100 | 50 ms | 48 ms | 1.04× |
| 400 | 120 ms | 110 ms | 1.09× |
| 600 | 200 ms | 180 ms | 1.11× |
| **1000** | **400 ms** | **300 ms** | **1.3×** |
| 1200 | 600 ms | 550 ms | 1.09× |

**成功标准**：
- ✅ 在1000 reqs/sec时，P99延迟优于Paella 1.3×
- ✅ 整体吞吐量-延迟曲线优于或接近Paella

---

## 4. AMD特有测试

### 4.1 CWSR Lv3 Integration - AMD硬件加速

#### Test 4.1.1: CWSR Lv3 vs XSched Lv1 - 抢占延迟对比

**目标**：验证AMD CWSR硬件能力对XSched的增强

**参考**：
- 论文6.3节：Queue-based preemption (Lv3)
- AMD CWSR：`AMDKFD_IOC_PREEMPT_QUEUE` ioctl

**测试步骤**：

1. **实现Lv3接口**（修改XSched的HIP平台实现）：

```cpp
// platforms/hip/hal/src/hip_queue.cpp

#include <linux/kfd_ioctl.h>

class HipQueue : public Queue {
public:
    // 实现Lv3接口
    Status Interrupt() override {
        if (kfd_fd_ < 0) {
            kfd_fd_ = open("/dev/kfd", O_RDWR);
        }
        
        struct kfd_ioctl_preempt_queue_args args = {
            .queue_id = queue_id_,
            .preempt_type = KFD_PREEMPT_TYPE_WAVEFRONT_SAVE,
            .timeout_ms = 1000
        };
        
        int ret = ioctl(kfd_fd_, AMDKFD_IOC_PREEMPT_QUEUE, &args);
        return (ret == 0) ? Status::OK : Status::ERROR;
    }
    
    Status Restore() override {
        struct kfd_ioctl_resume_queue_args args = {
            .queue_id = queue_id_
        };
        
        int ret = ioctl(kfd_fd_, AMDKFD_IOC_RESUME_QUEUE, &args);
        return (ret == 0) ? Status::OK : Status::ERROR;
    }
    
private:
    int kfd_fd_ = -1;
    uint32_t queue_id_;
};
```

2. **对比测试**：

```bash
# 1. Lv1测试
./test_preemption_latency_lv1
# 记录P99延迟：预期 ~4ms

# 2. Lv3测试（CWSR）
./test_preemption_latency_lv3
# 记录P99延迟：预期 < 100μs
```

**预期结果**：

| 指标 | Lv1 (论文MI50) | Lv3 (MI308X+CWSR) | 改进 |
|------|---------------|------------------|------|
| P99抢占延迟 | ~4 ms | < 100 μs | 40× |
| 独立于T | ❌ | ✅ | - |
| 硬件支持 | ❌ | ✅（CWSR） | - |

**成功标准**：
- ✅ Lv3 P99延迟 < 100μs
- ✅ 优于Lv1的40×以上
- ✅ 接近NVIDIA GV100 Lv3的32μs（考虑硬件差异）

---

### 4.2 ROCm Platform Validation - ROCm生态兼容性

#### Test 4.2.1: PyTorch + HIP Backend - 深度学习框架兼容性

**目标**：验证XSched与PyTorch ROCm版本的兼容性

**测试步骤**：

```python
# test_pytorch_compatibility.py
import torch
from xsched import XQueue, XHintPriority

def test_pytorch_with_xsched():
    # 1. 创建PyTorch模型
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1024)
    ).to('cuda')
    
    # 2. 设置XSched优先级
    XHintPriority(2)
    
    # 3. 运行推理
    input_data = torch.randn(32, 1024).to('cuda')
    output = model(input_data)
    
    print(f"Output shape: {output.shape}")
    print("PyTorch + XSched integration: SUCCESS")

if __name__ == '__main__':
    test_pytorch_with_xsched()
```

**成功标准**：
- ✅ PyTorch模型正常运行
- ✅ XSched优先级生效
- ✅ 无性能退化

---

#### Test 4.2.2: MIOpen + hipBLAS - AMD计算库兼容性

**目标**：验证XSched与AMD核心库的兼容性

**测试步骤**：

```cpp
// test_miopen_compatibility.cpp
#include <miopen/miopen.h>
#include <hipblas.h>
#include "xsched/xsched.h"

void test_miopen_with_xsched() {
    // 1. 初始化MIOpen
    miopenHandle_t miopen_handle;
    miopenCreate(&miopen_handle);
    
    // 2. 创建XQueue
    XQueue* xq = XQueueCreate();
    XHintPriority(xq, 2);
    
    // 3. 运行卷积操作
    miopenConvolutionForward(...);
    
    // 4. 清理
    XQueueDestroy(xq);
    miopenDestroy(miopen_handle);
}

void test_hipblas_with_xsched() {
    // 1. 初始化hipBLAS
    hipblasHandle_t hipblas_handle;
    hipblasCreate(&hipblas_handle);
    
    // 2. 创建XQueue
    XQueue* xq = XQueueCreate();
    XHintPriority(xq, 2);
    
    // 3. 运行矩阵乘法
    hipblasSgemm(hipblas_handle, ...);
    
    // 4. 清理
    XQueueDestroy(xq);
    hipblasDestroy(hipblas_handle);
}
```

**成功标准**：
- ✅ MIOpen操作正常运行
- ✅ hipBLAS操作正常运行
- ✅ XSched调度生效

---

## 5. 实施计划

### 5.1 测试阶段划分

```
Phase 1: 基础验证（Week 1-2）
├── Test 7.1.1: XSched编译和基本运行
├── Test 7.4.1: Runtime overhead测量
└── Test 7.4.2: CPU overhead测量

Phase 2: 调度策略（Week 3-4）
├── Test 7.2.1: Fixed priority policy
├── Test 7.2.2: Bandwidth partition policy
└── Test 8.1.1: Multi-tenant GPU harvesting

Phase 3: 性能优化（Week 5-6）
├── Test 7.3.1: Preemption latency (Lv1)
├── Test 7.3.2: Command execution time impact
├── Test 7.3.3: Threshold tuning
└── Test 8.2.1: Video conferencing (Laxity-based)

Phase 4: Lv3扩展（Week 7-8）
├── Test 4.1.1: CWSR Lv3 implementation
├── Test 7.3.1 (Lv3): Preemption latency with CWSR
└── Test 8.3.1: Triton integration (Lv3优化)

Phase 5: 生产验证（Week 9-10）
├── Test 8.3.2: Paella comparison
├── Test 4.2.1: PyTorch compatibility
└── Test 4.2.2: AMD libraries compatibility
```

### 5.2 预期成果

#### 5.2.1 论文复现指标对照表

| 论文指标 | 论文值（MI50/GV100） | MI308X目标值 | 验证状态 |
|---------|---------------------|-------------|---------|
| Runtime overhead (Lv1) | 1.7% / 0.7% | < 3.4% | ⏳ |
| CPU overhead | 3.6% / 2.8% | < 5% | ⏳ |
| Fixed priority P99延迟 | 1.30× | < 1.30× | ⏳ |
| Bandwidth partition准确性 | 75:25 | 75:25 ± 5% | ⏳ |
| Lv1 P99抢占延迟 (T=0.5ms) | ~4 ms | ~4 ms | ⏳ |
| **Lv3 P99抢占延迟** | **N/A** | **< 100 μs** | ⏳ |
| Multi-tenant Pjob性能 | 0.99 (GV100) | > 0.95 | ⏳ |
| Triton高优先级P99延迟 | 1.07× | < 1.10× | ⏳ |

#### 5.2.2 重大创新点（超越论文）

```
┌──────────────────────────────────────────────────────┐
│  AMD MI308X + CWSR + XSched = Lv3硬件加速            │
│                                                      │
│  论文MI50：仅Lv1支持                                 │
│  本项目：Lv1 + Lv3（CWSR）                           │
│                                                      │
│  预期抢占延迟：                                      │
│  - Lv1: ~4 ms (8T, T=0.5ms)                         │
│  - Lv3: < 100 μs (40× improvement)                  │
│                                                      │
│  意义：                                              │
│  - AMD GPU首次达到NVIDIA GV100级别的抢占性能        │
│  - 证明CWSR是XSched Lv3的理想实现                    │
│  - 为AMD GPU在AI推理服务中的应用提供技术基础         │
└──────────────────────────────────────────────────────┘
```

---

## 6. 测试工具和脚本

### 6.1 自动化测试脚本

```bash
#!/bin/bash
# run_all_tests.sh - 自动运行所有测试

set -e

echo "=== XSched on AMD MI308X: Automated Test Suite ==="
echo "Based on Paper Chapter 7 & 8"
echo ""

# Phase 1: Basic Validation
echo "[Phase 1] Basic Validation"
./tests/test_7.1.1_portability.sh
./tests/test_7.4.1_runtime_overhead.sh
./tests/test_7.4.2_cpu_overhead.sh

# Phase 2: Scheduling Policies
echo "[Phase 2] Scheduling Policies"
./tests/test_7.2.1_fixed_priority.sh
./tests/test_7.2.2_bandwidth_partition.sh
./tests/test_8.1.1_multi_tenant.sh

# Phase 3: Performance Optimization
echo "[Phase 3] Performance Optimization"
./tests/test_7.3.1_preemption_latency.sh
./tests/test_7.3.2_exec_time_impact.sh
./tests/test_7.3.3_threshold_tuning.sh
./tests/test_8.2.1_video_conferencing.sh

# Phase 4: Lv3 Extension (if CWSR implemented)
if [ -f "./tests/test_4.1.1_cwsr_lv3.sh" ]; then
    echo "[Phase 4] Lv3 Extension (CWSR)"
    ./tests/test_4.1.1_cwsr_lv3.sh
fi

# Phase 5: Production Validation
echo "[Phase 5] Production Validation"
./tests/test_8.3.1_triton.sh
./tests/test_8.3.2_paella.sh
./tests/test_4.2.1_pytorch_compat.sh

echo ""
echo "=== All tests completed ==="
echo "See detailed results in ./test_results/"
```

### 6.2 结果收集脚本

```python
#!/usr/bin/env python3
# collect_results.py - 收集测试结果并生成报告

import json
import pandas as pd
import matplotlib.pyplot as plt

def collect_results():
    results = {
        "portability": {},
        "uniformity": {},
        "evolvability": {},
        "overhead": {},
        "case_studies": {}
    }
    
    # 读取各测试的JSON结果
    with open('test_results/7.1.1_portability.json') as f:
        results['portability'] = json.load(f)
    
    # ... 读取其他结果 ...
    
    return results

def generate_report(results):
    """生成Markdown报告"""
    
    report = f"""
# XSched on AMD MI308X: Test Results

## Summary

| Test Category | Pass Rate | Average Performance |
|--------------|-----------|---------------------|
| Portability | {results['portability']['pass_rate']} | - |
| Uniformity | {results['uniformity']['pass_rate']} | {results['uniformity']['avg_perf']} |
| Evolvability | {results['evolvability']['pass_rate']} | {results['evolvability']['avg_perf']} |
| Overhead | {results['overhead']['pass_rate']} | {results['overhead']['avg_overhead']}% |
| Case Studies | {results['case_studies']['pass_rate']} | - |

## Detailed Results

### 7.1 Portability
- XSched compilation: {'✅ PASS' if results['portability']['compilation'] else '❌ FAIL'}
- Basic execution: {'✅ PASS' if results['portability']['execution'] else '❌ FAIL'}
- Runtime overhead: {results['portability']['runtime_overhead']}%

### 7.2 Uniformity
- Fixed priority P99 latency: {results['uniformity']['fixed_priority_p99']}× standalone
- Bandwidth partition ratio: {results['uniformity']['bandwidth_ratio']}
- ...

### 7.3 Evolvability
- Lv1 P99 preemption latency: {results['evolvability']['lv1_p99_latency']} ms
- Lv3 P99 preemption latency: {results['evolvability']['lv3_p99_latency']} μs (if implemented)
- ...

### 7.4 Overhead
- Runtime overhead: {results['overhead']['runtime']}%
- CPU overhead: {results['overhead']['cpu']}%
- ...

### 8. Case Studies
- Case 1 (Multi-tenant): Pjob={results['case_studies']['case1_pjob_perf']}, Ojob={results['case_studies']['case1_ojob_perf']}
- Case 2 (Video conferencing): P99 frame latency={results['case_studies']['case2_p99_latency']} ms
- Case 3 (Inference serving): P99 latency={results['case_studies']['case3_p99_latency']}× standalone
"""
    
    with open('test_results/REPORT.md', 'w') as f:
        f.write(report)
    
    print("Report generated: test_results/REPORT.md")

def plot_results(results):
    """绘制结果图表（对应论文Fig. 9, 11, 12, 13, 14, 15）"""
    
    # Fig. 9对应图：Fixed priority latency CDF
    # ...
    
    # Fig. 11对应图：Preemption latency vs. hardware level
    # ...
    
    plt.savefig('test_results/figures.pdf')
    print("Figures saved: test_results/figures.pdf")

if __name__ == '__main__':
    results = collect_results()
    generate_report(results)
    plot_results(results)
```

---

## 7. 参考资料

### 7.1 论文关键章节

- **Chapter 7.1 (Portability)**：Page 11, Table 3
- **Chapter 7.2 (Uniformity)**：Page 11-12, Fig. 9
- **Chapter 7.3 (Evolvability)**：Page 12-13, Fig. 11
- **Chapter 7.4 (Overhead)**：Page 13, Fig. 12
- **Chapter 8.1 (GPU Harvesting)**：Page 13, Fig. 13
- **Chapter 8.2 (Video Conferencing)**：Page 13-14, Fig. 14
- **Chapter 8.3 (Inference Serving)**：Page 14, Fig. 15

### 7.2 相关代码和工具

- **XSched GitHub**：https://github.com/XpuOS/xsched
- **XSched Artifacts**：https://github.com/XpuOS/xsched-artifacts
- **Rodinia Benchmark (HIP)**：https://github.com/AMDComputeLibraries/Rodinia_HIP
- **PyTorch ROCm**：https://pytorch.org/
- **Triton Inference Server**：https://github.com/triton-inference-server
- **Paella Artifact**：https://github.com/eniac/paella

### 7.3 AMD技术文档

- **CWSR机制**：`/mnt/md0/zhehan/code/rampup_doc/GPREEMPT_MI300_Testing/CWSR机制简要总结.md`
- **KFD ioctl接口**：`/usr/include/linux/kfd_ioctl.h`
- **ROCm文档**：https://rocm.docs.amd.com/
- **MI300系列规格**：https://www.amd.com/en/products/accelerators/instinct/mi300

---

## 附录：测试数据记录模板

```json
{
  "test_id": "7.2.1",
  "test_name": "Fixed Priority Policy",
  "date": "2026-01-27",
  "hardware": "AMD MI308X (gfx942)",
  "rocm_version": "6.4.0",
  "xsched_version": "1.0",
  "results": {
    "foreground_p99_latency_standalone": 15.2,
    "foreground_p99_latency_native": 30.5,
    "foreground_p99_latency_xsched": 18.1,
    "background_throughput_standalone": 1.0,
    "background_throughput_native": 0.48,
    "background_throughput_xsched": 0.25,
    "runtime_overhead_percent": 2.3,
    "cpu_overhead_percent": 3.8
  },
  "pass": true,
  "notes": "P99 latency within 1.20x of standalone, better than 2.0x of native scheduler."
}
```

---

**文档维护**：
- 创建日期：2026-01-27
- 最后更新：2026-01-27
- 维护者：AI Assistant
- 状态：📋 测试计划中










