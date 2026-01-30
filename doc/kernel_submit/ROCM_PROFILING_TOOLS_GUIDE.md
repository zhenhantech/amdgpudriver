# ROCm Profiling Tools 使用指南

**文档目的**: ROCm 性能分析工具的选择和使用指南  
**推荐工具**: ROCprofiler-SDK (rocprofv3)  
**适用场景**: Kernel 提交流程的学习和验证  
**参考**: [ROCm 官方对比文档](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/conceptual/comparing-with-legacy-tools.html)

---

## 📋 目录

1. [工具对比](#1-工具对比)
2. [为什么选择 ROCprofiler-SDK](#2-为什么选择-rocprofiler-sdk)
3. [ROCprofiler-SDK 基础使用](#3-rocprofiler-sdk-基础使用)
4. [验证 Kernel 提交流程](#4-验证-kernel-提交流程)
5. [高级追踪技巧](#5-高级追踪技巧)
6. [输出格式和可视化](#6-输出格式和可视化)
7. [常见问题](#7-常见问题)

---

## 1️⃣ 工具对比

### 1.1 ROCm Profiling Tools 演进

```
ROCm 5.x 及之前:
  rocprof (rocprofv1)
    ↓
ROCm 6.x:
  rocprofv2
    ↓
ROCm 6.4+ (推荐):
  ROCprofiler-SDK (rocprofv3)  ← 我们使用这个！
```

### 1.2 主要区别对比

| 特性 | rocprofv1/v2 | ROCprofiler-SDK (v3) | 对学习的帮助 |
|------|-------------|---------------------|-------------|
| **Context 机制** | ❌ 无 | ✅ 有，更好的资源管理 | 理解 Context 概念 |
| **细粒度追踪** | ❌ 粗粒度 | ✅ 可分离 HIP/HSA API | 精确追踪调用链 |
| **时间精度** | ⚠️ 约 20% 误差 | ✅ 更准确 | 验证 doorbell 延迟 |
| **线程安全** | ⚠️ 一般 | ✅ 改进 | 减少对程序的干扰 |
| **Memory Trace** | ❌ 混在 API trace 中 | ✅ 独立选项 | 追踪内存操作 |
| **Scratch Memory** | ❌ 不支持 | ✅ 支持 | 理解 scratch 分配 |
| **PC Sampling** | ❌ 不支持 | ✅ Beta 支持 | 性能热点分析 |
| **输出格式** | CSV, JSON | CSV, JSON, Perfetto, OTF2 | 更好的可视化 |

### 1.3 命令对比速查

| 功能 | rocprofv2 | rocprofv3 |
|------|-----------|-----------|
| HIP 追踪 | `--hip-trace` | `--hip-runtime-trace` 或 `--hip-trace` |
| HSA 追踪 | `--hsa-trace` | `--hsa-core-trace` 或 `--hsa-trace` |
| Kernel 追踪 | `--kernel-trace` | `--kernel-trace` |
| ROCTx 标记 | `--roctx-trace` | `--marker-trace` |
| 内存拷贝 | 包含在 `--hip-trace` | `--memory-copy-trace` |
| 默认行为 | Kernel trace | Agent 信息（需显式指定） |

---

## 2️⃣ 为什么选择 ROCprofiler-SDK？

### 2.1 更适合学习 Kernel 提交流程

**ROCprofiler-SDK 的 Context 机制**与我们研究的概念完美对应：

```
我们的研究层次:
┌─────────────────────┐
│ Application         │
│  ↓ hipLaunchKernel  │
├─────────────────────┤
│ HIP Stream          │  ← ROCprofiler Context 可以精确追踪
│  ↓ launchKernel     │
├─────────────────────┤
│ HSA Queue           │  ← 可以看到 hsa_queue_create
│  ↓ write packet     │
├─────────────────────┤
│ KFD Context         │  ← 可以追踪 ioctl 调用
│  ↓ ioctl            │
├─────────────────────┤
│ MES Scheduler       │  ← 通过 kernel trace 观察
└─────────────────────┘
```

### 2.2 关键改进

#### 改进 1: 细粒度追踪

**旧版本 (rocprofv2)**:
```bash
# 粗粒度，难以分离
rocprofv2 --hip-trace --hsa-trace app

# 输出混在一起，包含：
# - HIP Runtime API
# - HIP Compiler 生成的代码
# - HSA Core API
# - HSA AMD Extension
# - 内存操作
```

**新版本 (rocprofv3)**:
```bash
# 可以精确选择需要的追踪
rocprofv3 \
  --hip-runtime-trace \    # 只追踪 HIP Runtime (如 hipLaunchKernel)
  --hsa-core-trace \       # 只追踪 HSA Core (如 hsa_queue_create)
  --kernel-trace \         # 只追踪 kernel 执行
  app
```

#### 改进 2: 时间精度

根据 [AMD 官方文档](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/conceptual/comparing-with-legacy-tools.html)：

> rocprofv3 has improved the accuracy of timing information by reducing the tool overhead. The result is a reduction in variance of kernel times and more accurate timing. There can be substantial (20%) differences in execution time reported by v1/v2 vs v3 for a single kernel execution.

**对学习的意义**:
- 更准确地测量 doorbell 写入的延迟
- 更好地理解异步执行的时间线
- 减少对被测程序的干扰

#### 改进 3: Context 机制

**旧版本问题**:
```cpp
// roctracer_init() 必须准备所有可能的服务
// 即使工具只需要 kernel trace，也要：
roctracer_init();  // 准备所有服务
  ↓
- 为所有 HIP API 安装 wrapper
- 为所有 HSA API 安装 wrapper
- 为所有 ROCTX 安装 hook
  ↓
大量不必要的 overhead！
```

**新版本设计**:
```cpp
// 只初始化需要的服务
rocprofiler_context_t context;
rocprofiler_create_context(&context);

// 只启用 kernel trace
rocprofiler_configure_kernel_trace_service(context, ...);

// 不启用的服务没有任何 overhead！
```

---

## 3️⃣ ROCprofiler-SDK 基础使用

### 3.1 安装

```bash
# ROCm 6.4+ 已包含
which rocprofv3
# /opt/rocm/bin/rocprofv3

# 查看版本
rocprofv3 --version
```

### 3.2 基础命令

**最简单的使用**:
```bash
# 默认：输出 agent 信息
rocprofv3 ./your_app

# 追踪 kernel 执行
rocprofv3 --kernel-trace ./your_app
```

**查看可用选项**:
```bash
# 查看所有 trace 选项
rocprofv3 --help | grep trace

# 查看输出格式
rocprofv3 --help | grep output-format
```

### 3.3 常用追踪组合

#### 组合 1: 基础 Kernel 追踪
```bash
rocprofv3 \
  --kernel-trace \
  --output-format csv \
  --output-directory ./results \
  ./your_app
```

**输出文件**:
```
results/
├── kernel_trace.csv         # Kernel 执行信息
└── metadata.json            # 运行元数据
```

#### 组合 2: API 调用链追踪
```bash
rocprofv3 \
  --hip-runtime-trace \      # HIP API 调用
  --hsa-core-trace \         # HSA API 调用
  --kernel-trace \           # Kernel 执行
  --output-format perfetto \ # Perfetto 格式
  --output-directory ./trace \
  ./your_app
```

#### 组合 3: 完整流程追踪
```bash
rocprofv3 \
  --hip-trace \              # 所有 HIP 相关
  --hsa-trace \              # 所有 HSA 相关
  --kernel-trace \           # Kernel 执行
  --memory-copy-trace \      # 内存操作
  --marker-trace \           # ROCTx markers
  --output-format csv \
  ./your_app
```

---

## 4️⃣ 验证 Kernel 提交流程

### 4.1 验证 Stream 创建和 Queue 映射

**目标**: 验证 HIP Stream → HSA Queue 的 1:1 映射

**测试程序** (`test_stream.cpp`):
```cpp
#include <hip/hip_runtime.h>
#include <roctx.h>

int main() {
    // 创建两个 stream
    roctxMark("Before stream creation");
    
    hipStream_t stream1, stream2;
    hipStreamCreate(&stream1);
    hipStreamCreate(&stream2);
    
    roctxMark("After stream creation");
    
    // 在不同 stream 中启动 kernel
    dim3 grid(256), block(64);
    
    roctxRangePush("Launch kernel1");
    myKernel<<<grid, block, 0, stream1>>>(data1);
    roctxRangePop();
    
    roctxRangePush("Launch kernel2");
    myKernel<<<grid, block, 0, stream2>>>(data2);
    roctxRangePop();
    
    hipStreamSynchronize(stream1);
    hipStreamSynchronize(stream2);
    
    hipStreamDestroy(stream1);
    hipStreamDestroy(stream2);
    
    return 0;
}
```

**追踪命令**:
```bash
rocprofv3 \
  --hip-runtime-trace \
  --hsa-core-trace \
  --kernel-trace \
  --marker-trace \
  --output-format csv \
  --output-directory ./stream_trace \
  ./test_stream
```

**分析输出**:
```bash
# 查看 HIP API 调用
cat stream_trace/hip_api_trace.csv | grep -E "hipStreamCreate|hipLaunchKernel"

# 输出示例：
# Time(ns)  | Function           | Stream    | Details
# 1000000   | hipStreamCreate    | 0x7f8001  | 
# 1050000   | hipStreamCreate    | 0x7f8002  |
# 2000000   | hipLaunchKernel    | 0x7f8001  | grid=[256,1,1]
# 2100000   | hipLaunchKernel    | 0x7f8002  | grid=[256,1,1]

# 查看 HSA Queue 创建
cat stream_trace/hsa_api_trace.csv | grep "hsa_queue_create"

# 输出示例：
# Time(ns)  | Function           | Queue     | Size
# 1010000   | hsa_queue_create   | 0x7f9001  | 1024
# 1060000   | hsa_queue_create   | 0x7f9002  | 1024

# 验证：每个 hipStreamCreate 对应一个 hsa_queue_create
```

### 4.2 验证 Doorbell 机制的低延迟

**目标**: 测量从 `hipLaunchKernel` 到 kernel 实际开始执行的延迟

**追踪命令**:
```bash
rocprofv3 \
  --hip-runtime-trace \
  --kernel-trace \
  --output-format csv \
  ./kernel_latency_test
```

**分析输出**:
```bash
# 提取关键时间戳
cat results/hip_api_trace.csv | grep "hipLaunchKernel" > launch_times.txt
cat results/kernel_trace.csv | grep "myKernel" > kernel_times.txt

# Python 分析脚本
python3 << 'EOF'
import csv

# 读取 launch times
with open('launch_times.txt') as f:
    reader = csv.DictReader(f)
    for row in reader:
        launch_end = int(row['EndTime(ns)'])
        print(f"hipLaunchKernel 返回: {launch_end} ns")

# 读取 kernel start times
with open('kernel_times.txt') as f:
    reader = csv.DictReader(f)
    for row in reader:
        kernel_start = int(row['BeginTime(ns)'])
        print(f"Kernel 开始执行: {kernel_start} ns")
        
        # 计算延迟
        latency = kernel_start - launch_end
        print(f"Doorbell 延迟: {latency} ns = {latency/1000:.2f} us")
        
        # 应该非常小（通常 < 10 us）
EOF
```

**预期结果**:
```
hipLaunchKernel 返回: 1000000 ns
Kernel 开始执行: 1000005 ns
Doorbell 延迟: 5000 ns = 5.00 us  ← 非常小！

这证明了 doorbell 机制的低延迟特性
```

### 4.3 验证 AQL Packet 的写入

**目标**: 观察 HSA API 调用序列

**追踪命令**:
```bash
rocprofv3 \
  --hsa-core-trace \
  --hsa-amd-trace \
  --output-format csv \
  ./packet_test
```

**关键 HSA API 调用序列**:
```bash
cat results/hsa_api_trace.csv | grep -E "hsa_queue|hsa_signal"

# 预期看到的调用序列：
# 1. hsa_queue_create()           ← 创建 Queue
# 2. hsa_signal_create()           ← 创建 completion signal
# 3. hsa_queue_add_write_index()   ← 获取写指针
# 4. [写入 AQL packet 到内存]     ← 用户空间操作，不可见
# 5. hsa_signal_store()            ← 写入 doorbell
# 6. hsa_signal_wait()             ← 等待完成
```

### 4.4 对比 MES vs CPSCH

**目标**: 验证 MES 模式下 kernel 不经过驱动层 Ring

**检查 MES 状态**:
```bash
# 检查是否启用 MES
cat /sys/module/amdgpu/parameters/mes
# 输出: 1 表示启用，0 表示未启用
```

**追踪 ftrace 事件**:
```bash
# Terminal 1: 启用 ftrace
sudo su
echo 1 > /sys/kernel/debug/tracing/events/drm/drm_run_job/enable
echo 1 > /sys/kernel/debug/tracing/events/drm/drm_sched_job/enable
cat /sys/kernel/debug/tracing/trace_pipe > ftrace.log

# Terminal 2: 运行程序（使用 rocprofv3）
rocprofv3 --kernel-trace ./compute_kernel_test

# Terminal 3: 检查 ftrace 日志
cat ftrace.log | grep drm_run_job

# 预期结果：
# - 如果使用 MES：只看到 sdma ring，没有 compute ring
# - 如果使用 CPSCH：会看到 compute ring
```

---

## 5️⃣ 高级追踪技巧

### 5.1 使用配置文件

**创建配置文件** (`trace_config.json`):
```json
{
  "rocprofiler": {
    "services": {
      "hip_runtime_trace": {
        "enabled": true
      },
      "hsa_core_trace": {
        "enabled": true
      },
      "kernel_trace": {
        "enabled": true,
        "iteration_range": [0, 10]
      }
    },
    "output": {
      "format": "perfetto",
      "directory": "./trace_output"
    }
  }
}
```

**使用配置文件**:
```bash
rocprofv3 --config trace_config.json ./your_app
```

### 5.2 过滤特定 Kernel

**只追踪特定 kernel**:
```bash
rocprofv3 \
  --kernel-trace \
  --kernel-include-regex "myKernel.*" \
  ./your_app
```

**排除某些 kernel**:
```bash
rocprofv3 \
  --kernel-trace \
  --kernel-exclude-regex "small_kernel.*" \
  ./your_app
```

**追踪 kernel 的特定迭代**:
```bash
# 只追踪第 100-200 次迭代
rocprofv3 \
  --kernel-trace \
  --kernel-iteration-range 100:200 \
  ./your_app
```

### 5.3 收集性能计数器

**查看可用计数器**:
```bash
rocprofv3-avail --metric
```

**收集特定计数器**:
```bash
rocprofv3 \
  --pmc SQ_WAVES,SQ_INSTS_VALU \
  --kernel-trace \
  ./your_app
```

**使用自定义 metrics 文件**:
```bash
# 创建 metrics.txt
cat > metrics.txt << 'EOF'
# Wave occupancy
pmc: SQ_WAVES
pmc: SQ_WAVE_CYCLES

# Memory bandwidth
pmc: TCC_EA_RDREQ_sum
pmc: TCC_EA_WRREQ_sum
EOF

rocprofv3 -E metrics.txt --kernel-trace ./your_app
```

### 5.4 PC Sampling (Beta)

**启用 PC sampling**:
```bash
rocprofv3 \
  --pc-sampling-beta-enabled \
  --kernel-trace \
  ./your_app
```

**作用**: 采样 kernel 中的 PC (Program Counter)，找出热点代码

---

## 6️⃣ 输出格式和可视化

### 6.1 CSV 格式（脚本处理）

**优点**: 易于用脚本处理和分析

```bash
rocprofv3 \
  --kernel-trace \
  --output-format csv \
  --output-directory ./csv_output \
  ./your_app

# 生成的文件
ls csv_output/
# kernel_trace.csv
# metadata.json
```

**CSV 分析示例**:
```python
import pandas as pd

# 读取 kernel trace
df = pd.read_csv('csv_output/kernel_trace.csv')

# 统计每个 kernel 的平均执行时间
kernel_stats = df.groupby('KernelName').agg({
    'Duration(ns)': ['mean', 'std', 'min', 'max', 'count']
})

print(kernel_stats)

# 计算 kernel 启动频率
df['StartTime'] = df['BeginTime(ns)']
df = df.sort_values('StartTime')
df['TimeDiff'] = df['StartTime'].diff()

print(f"平均 kernel 间隔: {df['TimeDiff'].mean()/1e6:.2f} ms")
```

### 6.2 Perfetto 格式（可视化）

**优点**: 强大的可视化界面，支持大规模 trace

```bash
rocprofv3 \
  --hip-runtime-trace \
  --hsa-core-trace \
  --kernel-trace \
  --marker-trace \
  --output-format perfetto \
  --output-directory ./perfetto_output \
  ./your_app
```

**可视化**:
```bash
# 方法1: 在线查看
# 1. 打开 https://ui.perfetto.dev/
# 2. 点击 "Open trace file"
# 3. 选择 perfetto_output/*.pftrace

# 方法2: 本地 Perfetto UI
# git clone https://github.com/google/perfetto.git
# cd perfetto
# ./tools/install-build-deps --ui
# ./tools/ninja -C out/ui ui
# python3 -m http.server --directory out/ui
# 打开 http://localhost:8000
```

**Perfetto UI 中可以看到**:
- 时间线视图：HIP API → HSA API → Kernel 执行
- 嵌套的 ROCTx ranges
- Stream 之间的并发关系
- 内存操作的时间线

### 6.3 OTF2 格式（大规模分析）

**优点**: 适合超大规模 trace，支持 MPI 程序

```bash
rocprofv3 \
  --kernel-trace \
  --output-format otf2 \
  --output-directory ./otf2_output \
  ./your_app
```

**使用 Vampir 查看**:
```bash
# 需要安装 Vampir (商业软件) 或 Vampir Web
vampir otf2_output/trace.otf2
```

---

## 7️⃣ 常见问题

### 7.1 找不到 rocprofv3 命令

**问题**:
```bash
$ rocprofv3 --version
-bash: rocprofv3: command not found
```

**解决**:
```bash
# 检查 ROCm 版本
cat /opt/rocm/.info/version
# 需要 >= 6.4

# 添加到 PATH
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# 或永久添加
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### 7.2 默认不输出 kernel trace

**问题**:
```bash
# 运行 rocprofv3，但没有 kernel trace
rocprofv3 ./app
```

**解决**: rocprofv3 默认只输出 agent 信息，需要显式指定
```bash
rocprofv3 --kernel-trace ./app
```

### 7.3 输出文件太大

**问题**: Perfetto 文件过大，无法在浏览器中打开

**解决方案1**: 限制追踪时间
```bash
rocprofv3 \
  --kernel-trace \
  --collection-period 0:5s:0 \  # 只收集 5 秒
  ./app
```

**解决方案2**: 使用 OTF2 格式
```bash
rocprofv3 \
  --kernel-trace \
  --output-format otf2 \  # OTF2 处理大文件更好
  ./app
```

**解决方案3**: 使用过滤
```bash
rocprofv3 \
  --kernel-trace \
  --kernel-include-regex "important_kernel.*" \  # 只追踪重要的
  --kernel-iteration-range 0:100 \               # 只追踪前100次
  ./app
```

### 7.4 时间戳不对齐

**问题**: CSV 中的时间戳难以对齐分析

**解决**: 使用 Perfetto 格式，它会自动对齐所有事件
```bash
rocprofv3 \
  --hip-runtime-trace \
  --kernel-trace \
  --output-format perfetto \
  ./app
```

### 7.5 与旧版本结果不一致

**问题**: rocprofv3 的时间与 rocprofv2 差异很大

**解释**: 这是正常的！根据 [AMD 文档](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/conceptual/comparing-with-legacy-tools.html)：
- rocprofv3 的时间更准确（减少了约 20% 的误差）
- rocprofv3 降低了工具的 overhead
- 对于大量样本，平均时间差异在个位数百分比

---

## 8️⃣ 完整示例：追踪多 Stream 程序

### 8.1 测试程序

**文件**: `multi_stream_test.cpp`
```cpp
#include <hip/hip_runtime.h>
#include <roctx.h>
#include <stdio.h>

__global__ void compute_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < 1000; i++) {
            sum += data[idx] * 0.1f;
        }
        data[idx] = sum;
    }
}

int main() {
    const int N = 1024 * 1024;
    const int num_streams = 4;
    
    // 分配内存
    float *d_data[num_streams];
    for (int i = 0; i < num_streams; i++) {
        hipMalloc(&d_data[i], N * sizeof(float));
    }
    
    // 创建 streams
    roctxMark("Creating streams");
    hipStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        hipStreamCreate(&streams[i]);
    }
    
    // 在不同 stream 中启动 kernel
    dim3 grid(N / 256);
    dim3 block(256);
    
    for (int i = 0; i < num_streams; i++) {
        char range_name[64];
        snprintf(range_name, sizeof(range_name), "Stream %d", i);
        roctxRangePush(range_name);
        
        compute_kernel<<<grid, block, 0, streams[i]>>>(d_data[i], N);
        
        roctxRangePop();
    }
    
    // 同步所有 streams
    roctxMark("Synchronizing streams");
    for (int i = 0; i < num_streams; i++) {
        hipStreamSynchronize(streams[i]);
    }
    
    // 清理
    for (int i = 0; i < num_streams; i++) {
        hipStreamDestroy(streams[i]);
        hipFree(d_data[i]);
    }
    
    return 0;
}
```

**编译**:
```bash
hipcc multi_stream_test.cpp -o multi_stream_test -lroctx64
```

### 8.2 追踪命令

```bash
rocprofv3 \
  --hip-runtime-trace \
  --hsa-core-trace \
  --kernel-trace \
  --marker-trace \
  --memory-copy-trace \
  --output-format perfetto \
  --output-format csv \
  --output-directory ./multi_stream_trace \
  --log-level info \
  ./multi_stream_test
```

### 8.3 分析结果

**查看 CSV 输出**:
```bash
# 查看 stream 创建
cat multi_stream_trace/hip_api_trace.csv | grep "hipStreamCreate"

# 查看对应的 queue 创建
cat multi_stream_trace/hsa_api_trace.csv | grep "hsa_queue_create"

# 查看 kernel 执行
cat multi_stream_trace/kernel_trace.csv | sort -t, -k2 -n

# Python 分析
python3 << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt

# 读取 kernel trace
df = pd.read_csv('multi_stream_trace/kernel_trace.csv')

# 转换时间为毫秒
df['StartTime_ms'] = df['BeginTime(ns)'] / 1e6
df['Duration_ms'] = df['Duration(ns)'] / 1e6

# 绘制时间线
plt.figure(figsize=(12, 6))
for idx, row in df.iterrows():
    plt.barh(row['QueueId'], row['Duration_ms'], 
             left=row['StartTime_ms'], height=0.5)

plt.xlabel('Time (ms)')
plt.ylabel('Queue ID')
plt.title('Multi-Stream Kernel Execution Timeline')
plt.savefig('timeline.png')
print("Timeline saved to timeline.png")

# 检查并发执行
print("\n并发执行分析:")
print(f"总共 {len(df)} 个 kernel")
print(f"时间跨度: {df['StartTime_ms'].max() - df['StartTime_ms'].min():.2f} ms")
print(f"如果串行执行需要: {df['Duration_ms'].sum():.2f} ms")
print(f"并发加速比: {df['Duration_ms'].sum() / (df['StartTime_ms'].max() - df['StartTime_ms'].min()):.2f}x")
EOF
```

**在 Perfetto 中查看**:
1. 打开 https://ui.perfetto.dev/
2. 加载 `multi_stream_trace/*.pftrace`
3. 观察：
   - 4 个 Stream 的 kernel 并发执行
   - ROCTx markers 显示的范围
   - HIP API 调用与 kernel 执行的对应关系

---

## 9️⃣ 与文档其他部分的集成

### 9.1 验证文档中的流程

**KERNEL_TRACE_01_APP_TO_HIP.md** 中的流程可以这样验证：

```bash
# 追踪 hipLaunchKernel 到 AQL packet 的流程
rocprofv3 \
  --hip-runtime-trace \   # 验证 hipLaunchKernel 调用
  --hsa-core-trace \      # 验证 hsa_queue_create 和 signal 操作
  --kernel-trace \        # 验证 kernel 执行
  --output-format perfetto \
  ./test_app

# 在 Perfetto 中观察调用链
```

**KERNEL_TRACE_02_HSA_RUNTIME.md** 中的 doorbell 机制：

```bash
# 测量 doorbell 延迟
rocprofv3 \
  --hip-runtime-trace \
  --kernel-trace \
  --output-format csv \
  ./doorbell_test

# 分析 hipLaunchKernel 返回时间和 kernel 开始时间的差异
```

**KERNEL_TRACE_STREAM_MANAGEMENT.md** 中的 Stream 管理：

```bash
# 验证 Stream 到 Queue 的映射
rocprofv3 \
  --hip-runtime-trace \
  --hsa-core-trace \
  --kernel-trace \
  --marker-trace \
  --output-format perfetto \
  ./stream_management_test
```

---

## 🔟 总结

### 10.1 推荐的学习路径

```
第1步：基础追踪
  rocprofv3 --kernel-trace ./app
  → 熟悉工具基本用法

第2步：API 追踪
  rocprofv3 --hip-runtime-trace --hsa-core-trace ./app
  → 验证文档中的 API 调用链

第3步：完整流程
  rocprofv3 --hip-trace --hsa-trace --kernel-trace ./app
  → 观察完整的 kernel 提交流程

第4步：可视化分析
  rocprofv3 --output-format perfetto ...
  → 使用 Perfetto 可视化时间线

第5步：性能分析
  rocprofv3 --pmc <counters> ...
  → 收集性能计数器
```

### 10.2 常用命令速查

```bash
# 最常用的追踪组合
alias trace-kernel='rocprofv3 --kernel-trace --output-format csv'
alias trace-api='rocprofv3 --hip-runtime-trace --hsa-core-trace --output-format perfetto'
alias trace-full='rocprofv3 --hip-trace --hsa-trace --kernel-trace --marker-trace --output-format perfetto'

# 使用
trace-kernel ./your_app
trace-api ./your_app
trace-full ./your_app
```

### 10.3 关键资源

- [ROCprofiler-SDK 官方文档](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/)
- [工具对比](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/conceptual/comparing-with-legacy-tools.html)
- [Perfetto UI](https://ui.perfetto.dev/)
- 本系列文档：
  - [Kernel 提交流程索引](./KERNEL_TRACE_INDEX.md)
  - [应用层到 HIP](./KERNEL_TRACE_01_APP_TO_HIP.md)
  - [HSA Runtime](./KERNEL_TRACE_02_HSA_RUNTIME.md)
  - [KFD 驱动层](./KERNEL_TRACE_03_KFD_QUEUE.md)
  - [MES 调度器](./KERNEL_TRACE_04_MES_HARDWARE.md)
  - [数据结构](./KERNEL_TRACE_05_DATA_STRUCTURES.md)
  - [Stream 管理](./KERNEL_TRACE_STREAM_MANAGEMENT.md)

---

**最后建议**: ROCprofiler-SDK 是学习 ROCm 内部机制的强大工具，结合本系列文档使用，可以深入理解从应用层到硬件层的完整流程！


