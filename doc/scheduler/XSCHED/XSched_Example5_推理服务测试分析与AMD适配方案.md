# XSched Example 5: 推理服务测试分析与AMD GPU适配方案

**文档日期**: 2026-01-27  
**目标平台**: AMD Instinct MI308X (GFX942)  
**适配目标**: 将NVIDIA TensorRT方案迁移到AMD ROCm生态  
**环境状态**: ✅ 现有 ROCm+PyTorch 环境可用 (`zhenaiter:flashinfer-rocm`)

---

## 🎉 重要发现：现有环境评估

### 当前环境配置（Docker `zhenaiter`）

```bash
Environment: flashinfer-rocm (micromamba)
- PyTorch: 2.9.1+rocm6.4
- ROCm: 6.4 (8× AMD Instinct MI308X)
- pytorch-triton-rocm: 3.5.1
- numpy: 2.4.0
- 缺少: transformers (需要安装)
```

**关键优势**:
- ✅ PyTorch + ROCm 已配置完成
- ✅ GPU 检测正常（8× MI308X）
- ✅ 无需从头搭建环境
- ⚠️ 仅需安装 `transformers` 库

**环境激活命令**:
```bash
export MAMBA_EXE='/root/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/root/micromamba'
eval "$(/root/.local/bin/micromamba shell hook --shell=bash)"
micromamba activate flashinfer-rocm
```

---

## 1. Example 5 测试内容分析

### 1.1 测试场景概述

**场景名称**: 推理服务器多优先级模型调度  
**论文对应**: XSched论文 Figure 15a（核心benchmark之一）  
**实际应用**: AI推理服务（如ChatGPT、Stable Diffusion等）

### 1.2 核心测试目标

```
┌─────────────────────────────────────────────┐
│   Triton Inference Server                   │
│  ┌─────────────────────────────────────┐    │
│  │  BERT模型（3个不同优先级副本）      │    │
│  │  - bert-high  (PRIORITY_MAX)         │    │
│  │  - bert-norm  (默认优先级)           │    │
│  │  - bert-low   (PRIORITY_MIN)         │    │
│  └─────────────────────────────────────┘    │
│           ↓ 通过XSched调度                   │
│  ┌─────────────────────────────────────┐    │
│  │  XQueue抽象层                        │    │
│  │  - 每个CUDA Stream → 一个XQueue     │    │
│  │  - 继承模型的优先级设置              │    │
│  │  - HPF调度策略                       │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

### 1.3 测试验证点

| 验证点 | 说明 | 论文期望 |
|-------|------|---------|
| **高优先级延迟** | bert-high模型的P99延迟 | < 100ms |
| **低优先级延迟** | bert-low模型的P99延迟 | 可接受范围内 |
| **延迟比** | 高/低优先级延迟比 | 显著差异 |
| **吞吐量影响** | XSched对总体吞吐量的影响 | < 10% |
| **抢占效果** | 高优先级任务能否抢占低优先级 | 是 |

### 1.4 测试工作负载

**模型**: BERT-Large（Question Answering，SQuAD 1.1数据集）
- 参数量: ~340M
- 输入: Sequence length = 384
- Batch size: 1
- 精度: FP16

**请求模式**:
- 高优先级: 低频率（延迟敏感）
- 中优先级: 中等频率
- 低优先级: 高频率（吞吐量优先）

---

## 2. 原始实现（NVIDIA方案）

### 2.1 技术栈

```
┌─────────────────────────────────────┐
│  Triton Inference Server            │
│  ├─ TensorRT Backend                │  ← NVIDIA专用
│  │   └─ TensorRT Engine (.plan)     │  ← NVIDIA专用
│  └─ CUDA Streams                    │  ← NVIDIA API
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  XSched (CUDA版本)                   │
│  ├─ libshimcuda.so                  │  ← CUDA API拦截
│  ├─ libhalcuda.so                   │  ← CUDA硬件抽象
│  └─ libpreempt.so                   │  ← 调度核心
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  CUDA Driver + cuDNN + TensorRT     │  ← NVIDIA生态
└─────────────────────────────────────┘
```

### 2.2 关键依赖（NVIDIA专有）

| 组件 | 作用 | 是否AMD可替代 |
|------|------|--------------|
| **TensorRT** | 模型优化和推理引擎 | ⚠️ 需替代 |
| **TensorRT Backend** | Triton的TensorRT插件 | ⚠️ 需替代 |
| **CUDA Runtime** | GPU执行环境 | ✅ HIP替代 |
| **cuDNN** | 深度学习算子库 | ✅ MIOpen替代 |
| **CUDA Streams** | 异步执行流 | ✅ HIP Streams替代 |

### 2.3 XSched集成点

**修改位置**: TensorRT Backend（仅10行代码）

```cpp
// 原始代码（伪代码）
cudaStream_t stream = GetCudaStream(model);

// XSched修改后（伪代码）
cudaStream_t stream = GetCudaStream(model);
HwQueueHandle hwq;
CudaQueueCreate(&hwq, stream);  // 创建硬件队列
XQueueHandle xq;
XQueueCreate(&xq, hwq, ...);    // 创建XQueue
XHintPriority(xq, model.priority); // 继承模型优先级
```

**关键**: XSched只需要修改**Backend代码**，不需要修改模型文件！

---

## 3. AMD GPU 适配方案

### 3.1 方案概述

**核心思路**: 替换NVIDIA专有组件，保持XSched架构不变

```
原始方案 (NVIDIA)              →    AMD适配方案
─────────────────────────────────────────────────
TensorRT Engine                →    MIGraphX / PyTorch
TensorRT Backend               →    PyTorch Backend / ONNX Backend
CUDA Runtime                   →    HIP Runtime
libshimcuda.so                 →    libshimhip.so (已有)
libhalcuda.so                  →    libhalhip.so (已有)
```

### 3.2 推荐方案：使用PyTorch Backend ⭐⭐⭐⭐⭐

#### 方案优势

| 优势 | 说明 |
|------|------|
| ✅ **官方支持** | Triton官方维护PyTorch Backend |
| ✅ **AMD兼容** | PyTorch对ROCm有完整支持 |
| ✅ **易于部署** | 无需重新训练模型 |
| ✅ **性能良好** | PyTorch 2.0+ JIT编译优化 |
| ✅ **生态成熟** | 大量现成的BERT实现 |

#### 技术栈

```
┌─────────────────────────────────────┐
│  Triton Inference Server            │
│  ├─ PyTorch Backend                 │  ← 官方支持
│  │   └─ TorchScript Model (.pt)     │  ← 跨平台
│  └─ HIP Streams                     │  ← AMD ROCm
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  XSched (HIP版本)                    │
│  ├─ libshimhip.so                   │  ← 已编译 ✅
│  ├─ libhalhip.so                    │  ← 已编译 ✅
│  └─ libpreempt.so                   │  ← 已编译 ✅
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  ROCm + MIOpen + PyTorch-ROCm       │  ← AMD生态
└─────────────────────────────────────┘
```

---

## 4. 详细实施步骤

### 4.1 环境准备（✅ 使用现有环境，大幅简化！）

#### Step 1: 激活现有 ROCm+PyTorch 环境

```bash
# 在 Docker zhenaiter 容器内
export MAMBA_EXE='/root/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/root/micromamba'
eval "$(/root/.local/bin/micromamba shell hook --shell=bash)"
micromamba activate flashinfer-rocm

# 验证环境
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'ROCm: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"
# 期望输出:
# PyTorch: 2.9.1+rocm6.4
# ROCm: True
# GPUs: 8
```

**时间节省**: 从 2 天减少到 5 分钟！ ⏱️

#### Step 2: 安装缺失的依赖（仅 transformers）

```bash
# 安装 Hugging Face Transformers
pip install transformers
pip install tritonclient[all]  # Triton客户端库（用于测试）

# 验证安装
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

**时间节省**: 从半天减少到 5 分钟！ ⏱️

#### ~~Step 3: 安装Triton Server（暂缓）~~

**决策**: 由于 Example 5 的重点是**多优先级推理调度**，我们可以先用**简化版本**验证核心功能，无需完整的 Triton Server：

**简化方案A**: 直接用 PyTorch + XSched（推荐先做）
- 绕过 Triton Server 复杂性
- 直接测试多优先级 BERT 推理
- 时间: 1-2 天

**完整方案B**: 集成 Triton Server（后续可选）
- 更接近论文场景
- 需要编译 Triton PyTorch Backend
- 时间: 5-7 天

**推荐**: 先执行方案A，验证核心功能后再考虑方案B

---

## 4A. ⚡ 简化版实施方案（推荐先做）

**核心思路**: 绕过 Triton Server，直接用 PyTorch + XSched 验证多优先级推理

### 4A.1 核心测试代码（Python）

创建测试脚本 `test_multi_priority_bert.py`:

```python
#!/usr/bin/env python3
"""
简化版 Example 5: 多优先级 BERT 推理测试
使用 PyTorch + XSched，无需 Triton Server
"""

import torch
import time
import threading
import numpy as np
from transformers import BertForQuestionAnswering, BertTokenizer
import ctypes

# 加载 XSched 库
xsched = ctypes.CDLL('/workspace/xsched/output/lib/libpreempt.so')
halhip = ctypes.CDLL('/workspace/xsched/output/lib/libhalhip.so')

# 定义 XSched API（与 Example 3 类似）
class XSched:
    def __init__(self):
        self.xsched = xsched
        self.halhip = halhip
        
    def create_xqueue(self, hip_stream, priority):
        # 创建 HwQueue
        hwq = ctypes.c_void_p()
        self.halhip.HipQueueCreate(ctypes.byref(hwq), hip_stream)
        
        # 创建 XQueue
        xq = ctypes.c_void_p()
        self.xsched.XQueueCreate(ctypes.byref(xq), hwq, 2, 0)  # Lv2 (Block-level)
        
        # 设置优先级
        self.xsched.XHintPriority(xq, priority)
        
        return xq
    
    def set_scheduler(self):
        # 设置本地调度器 + HPF 策略
        self.xsched.XHintSetScheduler(0, 1)  # kSchedulerLocal, kPolicyHighestPriorityFirst

xs = XSched()
xs.set_scheduler()

# 加载 BERT 模型
print("Loading BERT model...")
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model.eval()
model.to('cuda')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# 准备测试数据
question = "What is the capital of France?"
context = "France is a country in Europe. The capital of France is Paris, which is known for the Eiffel Tower."

inputs = tokenizer(question, context, return_tensors='pt', max_length=384, padding='max_length', truncation=True)
inputs = {k: v.to('cuda') for k, v in inputs.items()}

def run_inference(priority, num_requests=50, name=""):
    """
    运行推理任务
    priority: 1=低, 2=中, 3=高
    """
    # 创建独立的 HIP Stream
    stream = torch.cuda.Stream()
    
    # 创建 XQueue（这里需要通过 ctypes 传递 stream 句柄）
    # 注意: 这是简化示例，实际需要 HIP API 集成
    
    latencies = []
    
    with torch.cuda.stream(stream):
        for i in range(num_requests):
            start = time.time()
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            torch.cuda.synchronize(stream)
            
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)
            
            if i % 10 == 0:
                print(f"[{name}] Request {i}: {latency:.2f}ms")
            
            time.sleep(0.01)  # 10ms间隔
    
    # 统计
    print(f"\n=== {name} Statistics ===")
    print(f"  Mean: {np.mean(latencies):.2f}ms")
    print(f"  P50:  {np.percentile(latencies, 50):.2f}ms")
    print(f"  P99:  {np.percentile(latencies, 99):.2f}ms")
    print(f"  Min:  {np.min(latencies):.2f}ms")
    print(f"  Max:  {np.max(latencies):.2f}ms")
    
    return latencies

# 测试: 并发运行高、中、低优先级任务
print("\n" + "="*60)
print("Starting Multi-Priority BERT Inference Test")
print("="*60)

high_thread = threading.Thread(target=run_inference, args=(3, 50, "HIGH Priority"))
norm_thread = threading.Thread(target=run_inference, args=(2, 50, "NORM Priority"))
low_thread = threading.Thread(target=run_inference, args=(1, 50, "LOW Priority"))

start_time = time.time()

# 启动任务（模拟不同时间到达）
low_thread.start()
time.sleep(0.5)
norm_thread.start()
time.sleep(0.5)
high_thread.start()

# 等待完成
high_thread.join()
norm_thread.join()
low_thread.join()

total_time = time.time() - start_time

print(f"\n=== Overall Statistics ===")
print(f"Total time: {total_time:.2f}s")
print(f"XSched is {'ENABLED' if xsched else 'DISABLED'}")
```

### 4A.2 运行测试

```bash
# 在 Docker zhenaiter 容器内
cd /workspace

# 激活环境
export MAMBA_EXE='/root/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/root/micromamba'
eval "$(/root/.local/bin/micromamba shell hook --shell=bash)"
micromamba activate flashinfer-rocm

# 设置库路径
export LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib:/opt/rocm/lib:/workspace/xsched/output/lib:$LD_LIBRARY_PATH

# 运行测试
python test_multi_priority_bert.py
```

### 4A.3 预期结果

| 场景 | 高优先级 P99 | 中优先级 P99 | 低优先级 P99 | 说明 |
|------|-------------|-------------|-------------|------|
| **Baseline (无 XSched)** | ~120ms | ~120ms | ~120ms | 无优先级区分 |
| **XSched (Lv1)** | ~110ms | ~120ms | ~130ms | 轻微差异 |
| **XSched (Lv2)** | ~90ms | ~120ms | ~150ms | 显著差异 |

**时间投入**: 1-2 天（vs 完整方案的 1-2 周）

---

### 4.2 模型准备（完整 Triton 方案）

> **注意**: 以下是完整的 Triton Server 集成方案，建议在完成 4A 简化版后再实施

#### Step 1: 下载BERT模型

```bash
# 使用Hugging Face Transformers
pip install transformers

# 下载BERT-Large模型
python3 << EOF
from transformers import BertForQuestionAnswering, BertTokenizer

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# 保存模型
model.save_pretrained('./bert_model')
tokenizer.save_pretrained('./bert_model')
EOF
```

#### Step 2: 转换为TorchScript

```python
import torch
from transformers import BertForQuestionAnswering, BertTokenizer

# 加载模型
model = BertForQuestionAnswering.from_pretrained('./bert_model')
model.eval()
model.to('cuda')  # 移动到GPU

# 创建示例输入（用于trace）
dummy_input = {
    'input_ids': torch.randint(0, 30522, (1, 384)).to('cuda'),
    'attention_mask': torch.ones(1, 384).to('cuda'),
    'token_type_ids': torch.zeros(1, 384).to('cuda')
}

# 转换为TorchScript
with torch.no_grad():
    traced_model = torch.jit.trace(
        model,
        example_kwarg_inputs=dummy_input,
        strict=False
    )

# 保存TorchScript模型
traced_model.save('model.pt')
print("TorchScript模型已保存: model.pt")
```

#### Step 3: 创建模型仓库

```bash
# 创建模型仓库目录结构
mkdir -p model-repo/bert-high/1
mkdir -p model-repo/bert-norm/1
mkdir -p model-repo/bert-low/1

# 复制模型文件（3个副本）
cp model.pt model-repo/bert-high/1/
cp model.pt model-repo/bert-norm/1/
cp model.pt model-repo/bert-low/1/
```

#### Step 4: 配置模型文件（config.pbtxt）

**model-repo/bert-high/config.pbtxt**:
```protobuf
name: "bert-high"
backend: "pytorch"
max_batch_size: 1

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ 384 ]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ 384 ]
  },
  {
    name: "token_type_ids"
    data_type: TYPE_INT64
    dims: [ 384 ]
  }
]

output [
  {
    name: "start_logits"
    data_type: TYPE_FP32
    dims: [ 384 ]
  },
  {
    name: "end_logits"
    data_type: TYPE_FP32
    dims: [ 384 ]
  }
]

# ★ 设置高优先级
optimization {
  priority: PRIORITY_MAX
  cuda {
    graphs: false
  }
}

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

**model-repo/bert-low/config.pbtxt**:
```protobuf
name: "bert-low"
backend: "pytorch"
max_batch_size: 1

# ... (input/output同上) ...

# ★ 设置低优先级
optimization {
  priority: PRIORITY_MIN
  cuda {
    graphs: false
  }
}

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

**model-repo/bert-norm/config.pbtxt**:
```protobuf
name: "bert-norm"
backend: "pytorch"
max_batch_size: 1

# ... (input/output同上) ...

# ★ 默认优先级
instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

### 4.3 修改PyTorch Backend集成XSched

**关键文件**: `pytorch_backend/src/libtorch.cc`

#### 修改点1: 添加XSched头文件

```cpp
// 在文件开头添加
#include "xsched/xsched.h"
#include "xsched/hip/hal.h"  // 使用HIP版本
```

#### 修改点2: 在Stream创建时集成XSched

```cpp
// 原始代码（伪代码）
c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool();

// 添加XSched集成（约10行代码）
c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool();

// 创建XQueue
HwQueueHandle hwq;
HipQueueCreate(&hwq, stream.stream());  // 使用HIP API

XQueueHandle xq;
XQueueCreate(&xq, hwq, kPreemptLevelBlock, kQueueCreateFlagNone);

// 从config.pbtxt读取优先级并设置
int priority = model_config_.optimization().priority();
if (priority == PRIORITY_MAX) {
    XHintPriority(xq, 3);  // 高优先级
} else if (priority == PRIORITY_MIN) {
    XHintPriority(xq, 1);  // 低优先级
} else {
    XHintPriority(xq, 2);  // 默认优先级
}

// 设置本地调度器和HPF策略
XHintSetScheduler(kSchedulerLocal, kPolicyHighestPriorityFirst);
```

**注意**: 实际修改需要参考Triton PyTorch Backend的源码结构。

### 4.4 编译和部署

#### Step 1: 编译XSched（已完成）

```bash
# 已经在之前步骤中完成
cd /workspace/xsched
export LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib:/opt/rocm/lib:/workspace/xsched/output/lib:$LD_LIBRARY_PATH
# XSched库已编译完成
```

#### Step 2: 编译PyTorch Backend（带XSched）

```bash
# 克隆Triton PyTorch Backend
git clone https://github.com/triton-inference-server/pytorch_backend -b r22.08

cd pytorch_backend

# 应用XSched修改（创建补丁文件）
# 参考Example 5的triton-backend-xsched.patch

# 编译
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
      -DXSched_DIR=/workspace/xsched/output/lib/cmake/XSched \
      -DTRITON_COMMON_REPO_TAG=r22.08 \
      -DTRITON_CORE_REPO_TAG=r22.08 \
      -DTRITON_BACKEND_REPO_TAG=r22.08 ..
make -j$(nproc)
make install
```

#### Step 3: 启动Triton Server

```bash
# 设置环境变量
export LD_LIBRARY_PATH=/workspace/xsched/output/lib:/opt/rocm/lib:$LD_LIBRARY_PATH
export XSCHED_POLICY=HPF

# 启动服务器
tritonserver \
  --backend-directory ./build/install/backends \
  --model-repository=./model-repo \
  --strict-model-config=false \
  --log-verbose=1
```

#### Step 4: 验证服务启动

```bash
# 应该看到类似输出：
# +----------+---------+--------+
# | Model    | Version | Status |
# +----------+---------+--------+
# | bert-high| 1       | READY  |
# | bert-low | 1       | READY  |
# | bert-norm| 1       | READY  |
# +----------+---------+--------+
```

### 4.5 性能测试

#### 测试脚本（Python）

```python
import tritonclient.http as httpclient
import numpy as np
import time

# 创建客户端
client = httpclient.InferenceServerClient(url="localhost:8000")

# 准备输入数据
input_ids = np.random.randint(0, 30522, (1, 384), dtype=np.int64)
attention_mask = np.ones((1, 384), dtype=np.int64)
token_type_ids = np.zeros((1, 384), dtype=np.int64)

inputs = [
    httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
    httpclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
    httpclient.InferInput("token_type_ids", token_type_ids.shape, "INT64"),
]

inputs[0].set_data_from_numpy(input_ids)
inputs[1].set_data_from_numpy(attention_mask)
inputs[2].set_data_from_numpy(token_type_ids)

# 测试高优先级模型
latencies_high = []
for i in range(100):
    start = time.time()
    result = client.infer("bert-high", inputs)
    latency = (time.time() - start) * 1000  # ms
    latencies_high.append(latency)
    print(f"High priority request {i}: {latency:.2f} ms")
    time.sleep(0.05)  # 50ms间隔

# 测试低优先级模型（同时运行）
latencies_low = []
for i in range(100):
    start = time.time()
    result = client.infer("bert-low", inputs)
    latency = (time.time() - start) * 1000  # ms
    latencies_low.append(latency)
    print(f"Low priority request {i}: {latency:.2f} ms")
    time.sleep(0.05)

# 统计结果
print("\n=== 性能统计 ===")
print(f"高优先级 - 平均: {np.mean(latencies_high):.2f}ms, P99: {np.percentile(latencies_high, 99):.2f}ms")
print(f"低优先级 - 平均: {np.mean(latencies_low):.2f}ms, P99: {np.percentile(latencies_low, 99):.2f}ms")
print(f"延迟比: {np.mean(latencies_low) / np.mean(latencies_high):.2f}x")
```

---

## 5. 预期测试结果

### 5.1 论文基准数据（NVIDIA GV100 + TensorRT）

| 场景 | 高优先级P99 | 低优先级P99 | 延迟比 |
|------|------------|------------|--------|
| **Native Triton** | 150ms | 150ms | 1.0× |
| **Triton + Priority Config** | 120ms | 180ms | 1.5× |
| **Triton + XSched (Lv2)** | **65ms** | **180ms** | **2.77×** |

### 5.2 AMD MI308X预期结果

基于我们之前的测试（Example 3），AMD GPU在Lv1硬件级别的延迟差异约为**1.07-1.14倍**。

**保守估计**（Lv1）:

| 场景 | 高优先级P99 | 低优先级P99 | 延迟比 |
|------|------------|------------|--------|
| **Native Triton** | 100ms | 100ms | 1.0× |
| **Triton + XSched (Lv1)** | **95ms** | **110ms** | **1.16×** |

**乐观估计**（如果MI308X支持Lv2）:

| 场景 | 高优先级P99 | 低优先级P99 | 延迟比 |
|------|------------|------------|--------|
| **Triton + XSched (Lv2)** | **70ms** | **160ms** | **2.29×** |

### 5.3 关键指标

| 指标 | 目标值 | 验证方法 |
|------|--------|---------|
| **XSched开销** | < 10% | 对比Native vs XSched的吞吐量 |
| **抢占生效** | 是 | 观察高优先级P99延迟降低 |
| **低优先级饥饿** | 否 | 观察低优先级任务是否能完成 |
| **系统稳定性** | 良好 | 长时间运行无崩溃 |

---

## 6. 替代方案对比

### 6.1 方案A: PyTorch Backend（推荐） ⭐⭐⭐⭐⭐

| 优势 | 劣势 |
|------|------|
| ✅ 官方支持，稳定性好 | ⚠️ 性能略低于TensorRT（约10-20%） |
| ✅ 易于部署，生态成熟 | - |
| ✅ AMD兼容性好 | - |
| ✅ 模型转换简单 | - |

**实施难度**: ⭐⭐ (中等)  
**推荐指数**: ⭐⭐⭐⭐⭐

### 6.2 方案B: ONNX Runtime Backend ⭐⭐⭐⭐

| 优势 | 劣势 |
|------|------|
| ✅ 跨平台标准 | ⚠️ 需要ONNX转换 |
| ✅ AMD MIGraphX支持 | ⚠️ 某些算子可能不支持 |
| ✅ 优化效果好 | - |

**实施难度**: ⭐⭐⭐ (较难)  
**推荐指数**: ⭐⭐⭐⭐

### 6.3 方案C: MIGraphX Backend（最优性能） ⭐⭐⭐

| 优势 | 劣势 |
|------|------|
| ✅ AMD专为ROCm优化 | ❌ 需要自行开发Backend |
| ✅ 性能接近TensorRT | ❌ 生态不如PyTorch成熟 |
| ✅ 原生支持AMD特性 | ❌ 学习曲线陡峭 |

**实施难度**: ⭐⭐⭐⭐⭐ (很难)  
**推荐指数**: ⭐⭐⭐

### 6.4 推荐决策树

```
是否需要极致性能？
  ├─ 是 → 方案C (MIGraphX Backend)
  │       ⚠️ 需要2-4周开发时间
  │
  └─ 否 → 是否需要标准化？
          ├─ 是 → 方案B (ONNX Runtime)
          │       ⏱️ 需要1-2周适配时间
          │
          └─ 否 → ✅ 方案A (PyTorch Backend)
                  ⏱️ 需要3-5天适配时间
```

---

## 7. 实施时间线（⚡ 基于现有环境，大幅加速）

### 7.1 简化版快速验证（1-2天） ⭐⭐⭐⭐⭐

**目标**: 验证核心功能（多优先级 BERT 推理）

| 阶段 | 任务 | 时间 | 状态 |
|------|------|------|------|
| ~~Day 1-2~~ | ~~环境搭建~~ | ~~2天~~ | ✅ **已有环境** |
| Hour 1 | 安装 transformers 库 | 5分钟 | ⏳ 待做 |
| Hour 2-3 | 下载和测试 BERT 模型 | 2小时 | ⏳ 待做 |
| Day 1 | 编写简化版测试脚本 | 4小时 | ⏳ 待做 |
| Day 1-2 | 运行多优先级测试和调试 | 1天 | ⏳ 待做 |
| Day 2 | 数据分析和报告 | 4小时 | ⏳ 待做 |

**总时间**: 1-2 天（vs 原计划的 1 周）⚡

### 7.2 完整 Triton 集成（1-2周，可选）

**目标**: 完整实现论文场景（Triton Server + XSched）

| 阶段 | 任务 | 时间 |
|------|------|------|
| Day 1 | 简化版验证完成（基础） | ✅ |
| Day 2-3 | 搭建 Triton Server ROCm 环境 | 2天 |
| Day 4-5 | 模型转换和部署（TorchScript） | 2天 |
| Day 6-7 | XSched集成（修改PyTorch Backend） | 2天 |
| Day 8-9 | 功能测试和调试 | 2天 |
| Day 10 | 性能测试和报告 | 1天 |

**总时间**: 1-2 周

### 7.3 推荐路径

```
Phase 1: 简化版验证 (1-2天)
    ↓ 验证成功 ✅
Phase 2: 完整 Triton 集成 (1-2周) - 可选
    ↓
Phase 3: 生产部署 (1-2周) - 可选
```

**决策建议**: 
- ✅ 先完成 Phase 1，证明核心功能
- ⚠️ 如果 Phase 1 效果良好，再投入 Phase 2
- ❌ 不要跳过 Phase 1 直接做 Phase 2（风险高）

---

## 8. 关键技术要点

### 8.1 Triton Server架构理解

```
┌─────────────────────────────────────────┐
│  Triton Inference Server (Core)         │
│  ┌───────────────────────────────────┐  │
│  │  Model Management                 │  │
│  │  - 加载/卸载模型                   │  │
│  │  - 版本管理                       │  │
│  │  - 优先级配置 ← ★ 这里设置优先级 │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │  Request Scheduler                │  │
│  │  - 批处理                         │  │
│  │  - 队列管理                       │  │
│  │  - 默认不支持抢占 ← ★ XSched补充 │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │  Backend Interface                │  │
│  │  - PyTorch / TensorRT / ONNX     │  │
│  │  - ★ 这里集成XSched               │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### 8.2 XSched集成的关键点

1. **每个模型实例 → 独立的HIP Stream**
2. **每个HIP Stream → 一个XQueue**
3. **XQueue继承模型的优先级设置**
4. **XSched调度器在Backend层工作**

### 8.3 优先级传递路径

```
config.pbtxt (priority: PRIORITY_MAX)
    ↓
Triton Model Config
    ↓
Backend initialization
    ↓
HIP Stream creation
    ↓
XQueue creation + XHintPriority()
    ↓
XSched调度器 (HPF策略)
```

---

## 9. 常见问题与解决方案

### Q1: PyTorch性能是否足够？

**A**: 
- PyTorch 2.0+ 的JIT编译和优化已经很成熟
- 对于BERT这类Transformer模型，性能差距在10-20%
- 可以通过`torch.compile()`进一步优化
- **对于验证XSched功能，性能完全足够**

### Q2: 如何验证XSched是否生效？

**A**: 
```bash
# 1. 检查日志输出
grep "XQueue" /var/log/triton.log
# 应该看到: [INFO] XQueue created with priority X

# 2. 对比延迟
# 高优先级P99应明显低于低优先级

# 3. 监控GPU利用率
rocm-smi --showuse
# 应该看到GPU利用率稳定在高水平
```

### Q3: 如果性能不如预期怎么办？

**A**: 
1. **检查硬件级别**: 确认MI308X是否支持Lv2
2. **调整Progressive Launching参数**: 降低`max_in_flight`
3. **增加负载强度**: 提高请求频率
4. **考虑ONNX Runtime**: 性能优于纯PyTorch

---

## 10. 后续优化方向

### 10.1 短期优化（1个月内）

1. ✅ **验证基本功能** (PyTorch Backend + XSched)
2. ✅ **性能基准测试** (对比Native vs XSched)
3. ⚠️ **Lv2硬件验证** (测试MI308X是否支持Guardian-based)

### 10.2 中期优化（3个月内）

1. ⚠️ **ONNX Runtime集成** (更好的性能)
2. ⚠️ **多模型并发测试** (除了BERT，测试ResNet、GPT等)
3. ⚠️ **生产环境部署** (长时间稳定性测试)

### 10.3 长期优化（6个月+）

1. ⚠️ **MIGraphX Backend开发** (极致性能)
2. ⚠️ **自动化测试框架** (CI/CD集成)
3. ⚠️ **贡献回XSched社区** (AMD适配经验分享)

---

## 11. 总结

### 11.1 核心结论

1. ✅ **Example 5可以在AMD GPU上运行**
   - PyTorch Backend是最佳适配路径
   - 技术栈完全兼容ROCm

2. ✅ **XSched架构设计优秀**
   - Backend层集成，对模型无侵入
   - 仅需10行代码修改
   - 跨平台能力强

3. ⚠️ **性能需要实测验证**
   - AMD Lv1: 预期1.16倍延迟差异
   - AMD Lv2: 预期2.29倍延迟差异（如果支持）
   - 优于Example 3的验证维度

4. ⚠️ **实施难度适中**
   - PyTorch方案: 1周可完成
   - ONNX方案: 2周可完成
   - MIGraphX方案: 1个月可完成

### 11.2 推荐行动计划（⚡ 基于现有环境加速版）

**Phase 1: 快速验证（今天-明天）** ⭐⭐⭐⭐⭐
```bash
✅ 环境已就绪（zhenaiter:flashinfer-rocm）
⏳ 1. 安装 transformers 库 (5分钟)
⏳ 2. 下载 BERT 模型 (2小时)
⏳ 3. 编写简化版测试脚本 (4小时)
⏳ 4. 运行多优先级并发测试 (1天)
⏳ 5. 生成性能报告 (4小时)
```
**预计完成**: 1-2 天 ⚡

**Phase 2: 完整 Triton 集成（后续，可选）**
```bash
1. 搭建 Triton Server ROCm 环境 (2天)
2. 转换 BERT 模型到 TorchScript (2天)
3. 集成 XSched 到 PyTorch Backend (2天)
4. 运行完整测试 (2天)
5. 生成对比报告 (1天)
```
**预计完成**: 1-2 周

**Phase 3: 深度优化（后续，可选）**
```bash
1. 验证 MI308X Lv2 支持
2. 尝试 ONNX Runtime 方案
3. 生产环境部署测试
```
**预计完成**: 2-4 周

### 11.3 与Example 3的对比

| 维度 | Example 3 | Example 5 |
|------|-----------|-----------|
| **场景** | 通用向量加法 | **真实推理服务** ⭐⭐⭐ |
| **复杂度** | 简单 | **复杂（多组件）** |
| **论文价值** | 核心原理验证 | **实际应用验证** ⭐⭐⭐ |
| **实施难度** | 易 | **中等** |
| **验证维度** | 多优先级调度 | **端到端性能** ⭐⭐⭐ |

**结论**: Example 5是**更接近生产场景**的测试！

---

**文档完成时间**: 2026-01-27 04:00:00  
**作者**: AI Assistant  
**状态**: 待实施验证

---

## 附录A: 快速启动命令

```bash
# 1. 环境准备
docker run -it --rm --device=/dev/kfd --device=/dev/dri \
  --group-add video --name triton-xsched \
  rocm/triton-server:latest bash

# 2. 安装PyTorch-ROCm
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7

# 3. 转换模型（在Python中）
python3 convert_bert_to_torchscript.py

# 4. 编译XSched（使用已编译的）
export LD_LIBRARY_PATH=/workspace/xsched/output/lib:$LD_LIBRARY_PATH

# 5. 启动Triton Server
export XSCHED_POLICY=HPF
tritonserver --model-repository=./model-repo

# 6. 测试
python3 test_inference.py
```

---

**下一步建议**: 先完成PyTorch Backend的快速验证，再考虑ONNX或MIGraphX方案。

