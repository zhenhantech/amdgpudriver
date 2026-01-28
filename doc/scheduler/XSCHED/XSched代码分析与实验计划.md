# XSched代码分析与实验计划

> **文档状态**：基于XSched GitHub仓库和OSDI 2025论文  
> **创建时间**：2026-01-26  
> **代码路径**：`/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/code/xsched`  
> **目标环境**：Docker zhenaiter容器（AMD GPU）

---

## 📋 目录

- [1. XSched代码架构分析](#1-xsched代码架构分析)
- [2. AMD GPU (HIP) 支持现状](#2-amd-gpu-hip-支持现状)
- [3. 论文测试用例清单](#3-论文测试用例清单)
- [4. 详细实验TODO列表](#4-详细实验todo列表)
- [5. Docker环境配置](#5-docker环境配置)
- [6. 对比实验方案](#6-对比实验方案)

---

## 1. XSched代码架构分析

### 1.1 核心模块结构

```
xsched/
├── preempt/              # XPreempt Lib - XQueue实现
│   ├── src/
│   │   ├── queue/        # XQueue核心逻辑
│   │   ├── sched/        # Agent（监控XQueue状态，IPC通信）
│   │   └── hint/         # Hint API
│   └── include/
│
├── sched/                # 调度策略实现
│   ├── policy/           # 各种调度策略
│   │   ├── hpf.cpp       # Highest Priority First
│   │   ├── fp.cpp        # Fixed Priority
│   │   ├── bp.cpp        # Bandwidth Partition
│   │   └── ...
│   └── README.md
│
├── service/              # XScheduler守护进程
│   ├── server/           # xserver - 全局调度器
│   └── cli/              # xcli - 命令行工具
│
├── platforms/            # 平台适配层
│   ├── cuda/             # NVIDIA CUDA支持
│   │   ├── hal/          # XAL Lib (HwQueue, HwCommand)
│   │   └── shim/         # XShim Lib (API拦截)
│   ├── hip/              # ✅ AMD HIP支持（我们需要的）
│   │   ├── hal/          # HIP HAL实现
│   │   └── shim/         # HIP Shim实现
│   ├── levelzero/        # Intel GPU/NPU
│   ├── opencl/           # OpenCL通用
│   ├── ascend/           # 华为昇腾NPU
│   └── ...
│
├── examples/             # 示例和测试用例
│   ├── Linux/
│   │   ├── 1_transparent_sched/     # ✅ 基础透明调度测试
│   │   ├── 2_give_hints/            # Hint API使用示例
│   │   ├── 3_intra_process_sched/   # 进程内调度
│   │   ├── 4_manual_sched/          # 手动调度
│   │   ├── 5_infer_serving/         # ✅ 论文Figure 15a实验
│   │   └── ...
│   └── Windows/
│
└── integration/          # 集成案例
    ├── llama.cpp/        # LLM推理集成
    └── triton/           # NVIDIA Triton集成
```

### 1.2 XSched工作流程

```
应用程序
    │
    │ (拦截Driver API)
    ↓
┌─────────────────────────────────────────────────────────┐
│ XShim Lib (libshimhip.so / libshimcuda.so)             │
│ - 拦截hipLaunchKernel/cuLaunchKernel                    │
│ - 封装为HwCommand                                        │
│ - 提交到XQueue                                           │
└─────────────────────────────────────────────────────────┘
    │
    ↓
┌─────────────────────────────────────────────────────────┐
│ XPreempt Lib (libpreempt.so)                            │
│ - XQueue管理（Ready/Idle状态）                           │
│ - Progressive Launching (Level-1)                        │
│ - Queue Deactivation/Reactivation (Level-2)             │
│ - Running Command Preemption (Level-3)                   │
│                                                          │
│ - Agent：监控XQueue状态 → IPC通知XScheduler              │
└─────────────────────────────────────────────────────────┘
    │                                   ↑
    │ (调度事件)                        │ (调度操作)
    ↓                                   │
┌─────────────────────────────────────────────────────────┐
│ XScheduler Daemon (xserver)                             │
│ - 全局XQueue状态监控                                     │
│ - 调用调度策略（HPF/FP/BP/...）                          │
│ - 发送调度决策（suspend/resume）                         │
└─────────────────────────────────────────────────────────┘
    │
    ↓
┌─────────────────────────────────────────────────────────┐
│ XAL Lib (libhal.so)                                     │
│ - 调用Driver API执行调度操作                             │
│ - hipStreamSynchronize() / hipDeviceSynchronize()        │
└─────────────────────────────────────────────────────────┘
    │
    ↓
GPU硬件
```

---

## 2. AMD GPU (HIP) 支持现状

### 2.1 HIP平台支持矩阵

基于README.md的支持矩阵：

| 平台 | XPU | Shim | Level-1 | Level-2 | Level-3 |
|------|-----|------|---------|---------|---------|
| **HIP** | **AMD GPUs** | ✅ | ✅ | 🔘 (未实现) | 🔘 (未实现) |

**关键发现**：
- ✅ **Shim支持**：可以拦截HIP API
- ✅ **Level-1支持**：Progressive Launching已实现
- ⚠️ **Level-2未实现**：Queue Deactivation/Reactivation
- ⚠️ **Level-3未实现**：Running Command Preemption

### 2.2 抢占级别详解

#### Level-1: Progressive Launching (已支持 ✅)

```python
# 概念
原本：应用提交100个kernel → GPU队列立即全部接收
XSched：应用提交100个kernel → XQueue缓存 → 逐步释放（如每次8个）

# 优点
- 高优先级任务到来时，可以快速插入
- 低优先级任务的in-flight kernel少，抢占快

# 缺点
- Threshold太小 → overhead高（频繁轮询）
- Threshold太大 → 抢占慢（等待in-flight完成）

# 性能（论文数据）
- Threshold=16, Batch=8: overhead < 3.4%
- 抢占延迟: ~200μs（等待in-flight完成）
```

#### Level-2: Queue Deactivation (HIP未实现 🔘)

```python
# 概念
暂停（Deactivate）低优先级队列 → GPU不再调度该队列的kernel
高优先级完成后 → 恢复（Reactivate）低优先级队列

# 实现方式（NVIDIA Volta可用）
- GPU微控制器支持queue enable/disable
- 或通过flushing机制模拟

# 性能（论文数据 - NVIDIA Volta）
- 抢占延迟: ~50-100μs
- Overhead: <1%

# AMD GPU现状
- ❓ 需要验证MI300是否有类似机制
- ❓ 可能需要通过KFD驱动实现
```

#### Level-3: Running Command Preemption (HIP未实现 🔘)

```python
# 概念
中断正在运行的kernel → 保存上下文 → 运行高优先级 → 恢复

# 实现方式
- 硬件支持：类似GPREEMPT的方式
- 需要GPU硬件级别的context switch能力

# 性能（论文数据）
- 抢占延迟: ~40-80μs
- 但实现复杂度最高

# AMD GPU现状
- ❌ 需要硬件MES支持或驱动层面改动
- 短期内无法实现
```

### 2.3 HIP平台代码结构

```
platforms/hip/
├── hal/                      # XAL实现
│   ├── include/
│   │   └── xsched/hip/hal/
│   │       ├── command.h     # HwCommand (hipLaunchKernel封装)
│   │       └── queue.h       # HwQueue (hipStream封装)
│   └── src/
│       ├── command.cpp       # Level-1实现
│       └── queue.cpp         # Level-1实现
│
└── shim/                     # XShim实现
    ├── include/
    │   └── hip_stub.h        # HIP API声明
    ├── src/
    │   ├── intercept.cpp     # 自动生成的拦截代码
    │   └── shim.cpp          # 拦截逻辑实现
    └── CMakeLists.txt
```

---

## 3. 论文测试用例清单

### 3.1 XSched Artifacts仓库

**GitHub地址**：https://github.com/XpuOS/xsched-artifacts  
**Zenodo DOI**：https://doi.org/10.5281/zenodo.15327992

**包含的实验**：
- Figure 4：XSched在不同Threshold下的overhead
- Figure 5-8：不同调度策略的性能对比
- Figure 13a (Example 5)：Triton推理服务多模型调度
- Figure 13b：LLaMA.cpp推理服务
- Figure 14：AI PC场景（Intel NPU）
- 等等...

### 3.2 适合我们的测试用例

#### 🎯 推荐测试用例1：透明调度基础测试（Example 1）

**路径**：`examples/Linux/1_transparent_sched/`

**描述**：
- 最简单的测试用例
- 两个进程同时运行vector addition
- 验证XSched的基本功能

**支持平台**：✅ CUDA, ✅ HIP, ✅ LevelZero, ✅ OpenCL

**测试内容**：
```bash
# 场景1：无XSched，两个进程公平竞争GPU
./app  # 进程1
./app  # 进程2
# 结果：两个进程延迟都增加2倍（公平但低效）

# 场景2：有XSched，高优先级进程优先
export XSCHED_AUTO_XQUEUE_PRIORITY=1  # 高优先级
./app  # 进程1 - 延迟接近单独运行
export XSCHED_AUTO_XQUEUE_PRIORITY=0  # 低优先级
./app  # 进程2 - 延迟显著增加
# 结果：高优先级进程延迟低，低优先级被抢占
```

**预期结果**：
- GPU利用率：~80%（Level-1限制）
- 高优先级延迟：+5-10%（相对单独运行）
- 低优先级延迟：+100-200%（被抢占）

**优点**：
- ✅ 快速验证XSched基本功能
- ✅ 支持HIP（AMD GPU）
- ✅ 编译和运行简单

---

#### 🎯 推荐测试用例2：Hint API和进程内调度（Example 2-3）

**路径**：
- `examples/Linux/2_give_hints/` - Hint API使用
- `examples/Linux/3_intra_process_sched/` - 进程内多队列调度

**描述**：
- 演示如何使用Hint API动态设置优先级
- 演示同一进程内多个XQueue的调度

**支持平台**：✅ CUDA, ✅ HIP

**测试内容**：
```cpp
// 创建XQueue并设置优先级
XQueue* xq_high = xqueue_create(stream_high, 1 /*priority*/);
XQueue* xq_low = xqueue_create(stream_low, 0 /*priority*/);

// 动态修改优先级
xqueue_set_priority(xq_low, 2);  // 提升低优先级队列
```

**预期结果**：
- 验证Hint API功能
- 验证Local Scheduler（进程内调度器）
- 为后续复杂场景打基础

---

#### 🎯 推荐测试用例3：多模型推理调度（Example 5）

**路径**：`examples/Linux/5_infer_serving/`

**描述**：
- **这是论文Figure 15a的实际实验！**
- Triton Inference Server集成
- 3个BERT模型（high/norm/low优先级）
- 模拟真实推理场景

**支持平台**：✅ CUDA（⚠️ 仅CUDA，需要改造支持HIP）

**测试内容**：
```bash
# 场景对比
1. Standalone: 单个模型独占GPU（基准性能）
2. Triton: 多模型共享GPU（无优先级）
3. Triton+Priority Config: Triton的优先级配置
4. XSched: XSched调度（论文方案）

# 评估指标
- P50/P99延迟
- 吞吐量
- GPU利用率
```

**论文结果（Figure 15a - NVIDIA Volta）**：
```
High-Priority模型延迟：
- Triton: ~120ms
- XSched: ~35ms（减少71%！）

Low-Priority模型延迟：
- Triton: ~120ms
- XSched: ~180ms（增加50%，但高优先级获益巨大）

GPU利用率：
- Triton: ~85%
- XSched: ~82%（略有下降，但延迟显著改善）
```

**挑战**：
- ⚠️ 需要CUDA（原始实现）
- ⚠️ 需要Docker环境（Triton Server容器）
- ⚠️ 需要下载BERT模型（~2GB）

**改造方案（支持AMD GPU）**：
1. 短期：先在NVIDIA GPU上验证（如果有的话）
2. 中期：改造为HIP版本（需要修改Triton Backend）
3. 长期：使用AMD版本的推理框架（如vLLM + ROCm）

---

### 3.3 测试用例对比矩阵

| 测试用例 | 平台支持 | 复杂度 | 论文对应 | 推荐优先级 | 预计时间 |
|---------|---------|-------|---------|----------|---------|
| **Example 1: 透明调度** | ✅ HIP | ⭐ 简单 | - | 🔴 P0 | 1小时 |
| **Example 2: Hint API** | ✅ HIP | ⭐⭐ 中等 | - | 🟠 P1 | 2小时 |
| **Example 3: 进程内调度** | ✅ HIP | ⭐⭐ 中等 | - | 🟠 P1 | 2小时 |
| **Example 4: 手动调度** | ✅ HIP | ⭐⭐ 中等 | - | 🟡 P2 | 2小时 |
| **Example 5: Triton推理** | ⚠️ CUDA | ⭐⭐⭐⭐ 复杂 | Figure 15a | 🟢 P2 | 1-2天 |
| **Example 7: 终端窗口** | ✅ HIP | ⭐⭐⭐ 较复杂 | Figure 14 | 🟢 P2 | 4小时 |

**建议测试顺序**：
1. 🔴 P0: Example 1（验证基本功能）
2. 🟠 P1: Example 2-3（验证高级功能）
3. 🟡 P2: Example 4（可选）
4. 🟢 P2: Example 5或7（论文对比）

---

## 4. 详细实验TODO列表

### 阶段1：环境准备（预计1-2小时）

#### ✅ Task 1.1: 代码已下载
- [x] XSched代码克隆到目标目录
- [x] Git submodule初始化

#### ⏳ Task 1.2: 检查Docker环境

**目标**：验证Docker zhenaiter容器配置

**操作**：
```bash
# 进入Docker容器
docker exec -it zhenaiter bash

# 检查GPU
rocm-smi
# 或
rocminfo | grep "Name:"

# 检查ROCm版本
apt list --installed | grep rocm
# 或
/opt/rocm/bin/rocminfo --version

# 检查HIP环境
hipconfig
which hipcc

# 检查编译工具
cmake --version  # 需要 >= 3.18
g++ --version    # 需要支持C++17
```

**预期结果**：
- GPU型号：MI300系列
- ROCm版本：>= 5.4
- HIP编译器：可用
- CMake版本：>= 3.18

**如果缺少**：
```bash
# 安装编译工具
apt update
apt install -y cmake build-essential

# 安装ROCm开发包（如果缺少）
apt install -y rocm-dev rocm-libs hip-dev
```

---

#### ⏳ Task 1.3: 检查XSched依赖

**目标**：确保第三方库完整

**操作**：
```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/code/xsched

# 检查submodule
ls -la 3rdparty/
# 应该看到：CLI11, cpp-httplib, cuxtra, ftxui, ipc, jsoncpp

# 如果缺少，重新初始化
git submodule update --init --recursive
```

**预期结果**：
- ✅ 所有第三方库已下载
- ✅ 无错误信息

---

### 阶段2：编译XSched（预计30分钟-1小时）

#### ⏳ Task 2.1: 编译HIP平台支持

**目标**：编译XSched的HIP支持库

**操作**：
```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/code/xsched

# 清理之前的构建（如果有）
make clean

# 编译HIP平台
make hip

# 或指定安装路径
make hip INSTALL_PATH=/mnt/md0/zhehan/xsched-install
```

**预期输出**：
```
[ 10%] Building CXX object ...
[ 20%] Building CXX object ...
...
[100%] Built target xsched
```

**生成的文件**：
```
output/  (或 INSTALL_PATH)
├── bin/
│   ├── xserver        # XScheduler守护进程
│   └── xcli           # 命令行工具
├── lib/
│   ├── libpreempt.so  # XPreempt Lib
│   ├── libhal.so      # XAL Lib (HIP HAL)
│   ├── libshimhip.so  # XShim Lib (HIP拦截)
│   └── cmake/
└── include/
    └── xsched/
```

**如果编译失败**：
```bash
# 查看详细错误
make hip VERBOSE=ON

# 常见问题：
# 1. 找不到HIP头文件
export HIP_PATH=/opt/rocm/hip
export ROCM_PATH=/opt/rocm

# 2. CMake版本太低
# 需要升级CMake到3.18+

# 3. C++编译器不支持C++17
# 需要使用g++7+或clang6+
```

**验证编译结果**：
```bash
# 检查生成的库
ls -lh output/lib/
ldd output/lib/libshimhip.so  # 检查依赖

# 检查可执行文件
output/bin/xserver --help
output/bin/xcli --help
```

---

### 阶段3：运行基础测试（Example 1）（预计1-2小时）

#### ⏳ Task 3.1: 编译测试应用

**目标**：编译Example 1的HIP版本

**操作**：
```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/code/xsched/examples/Linux/1_transparent_sched

# 查看Makefile
cat Makefile

# 编译HIP版本
make hip

# 或手动编译
hipcc -o app app.hip -std=c++11
```

**预期输出**：
```
生成可执行文件：./app
```

**验证**：
```bash
# 单独运行测试
./app

# 预期输出（无XSched）：
Task 0 completed in 66 ms
Task 1 completed in 66 ms
...
```

---

#### ⏳ Task 3.2: 无XSched基准测试

**目标**：测量无XSched时的性能（基准）

**操作**：
```bash
# 在一个终端运行
time ./app > /tmp/app1_baseline.log 2>&1

# 在另一个终端同时运行
time ./app > /tmp/app2_baseline.log 2>&1
```

**记录数据**：
```
进程1平均延迟：______ ms
进程2平均延迟：______ ms
GPU利用率：______%（通过rocm-smi查看）
```

---

#### ⏳ Task 3.3: 启动XScheduler

**目标**：启动XSched守护进程

**操作**：
```bash
# 新开一个终端
export LD_LIBRARY_PATH=/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/code/xsched/output/lib:$LD_LIBRARY_PATH

# 启动xserver
/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/code/xsched/output/bin/xserver HPF 50000

# HPF: Highest Priority First调度策略
# 50000: 监听端口（用于xcli连接）
```

**预期输出**：
```
[INFO] XScheduler started
[INFO] Policy: HPF (Highest Priority First)
[INFO] Listening on port 50000
```

**验证**：
```bash
# 另一个终端
/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/code/xsched/output/bin/xcli top -f 10

# 应该看到XQueue列表（开始为空）
```

---

#### ⏳ Task 3.4: 有XSched测试（高优先级 vs 低优先级）

**目标**：测试XSched的优先级调度效果

**操作**：

**终端1（高优先级进程）**：
```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/code/xsched/examples/Linux/1_transparent_sched

# 设置环境变量
export XSCHED_SCHEDULER=GLB                    # 使用全局调度器
export XSCHED_AUTO_XQUEUE=ON                   # 自动创建XQueue
export XSCHED_AUTO_XQUEUE_PRIORITY=1           # 高优先级
export XSCHED_AUTO_XQUEUE_LEVEL=1              # Level-1抢占
export XSCHED_AUTO_XQUEUE_THRESHOLD=16         # Threshold=16
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=8         # Batch=8
export LD_LIBRARY_PATH=/path/to/xsched/output/lib:$LD_LIBRARY_PATH

# 运行应用
time ./app > /tmp/app1_xsched_high.log 2>&1
```

**终端2（低优先级进程）**：
```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/code/xsched/examples/Linux/1_transparent_sched

# 设置环境变量
export XSCHED_SCHEDULER=GLB
export XSCHED_AUTO_XQUEUE=ON
export XSCHED_AUTO_XQUEUE_PRIORITY=0           # 低优先级（唯一不同）
export XSCHED_AUTO_XQUEUE_LEVEL=1
export XSCHED_AUTO_XQUEUE_THRESHOLD=4          # 更小的threshold
export XSCHED_AUTO_XQUEUE_BATCH_SIZE=2
export LD_LIBRARY_PATH=/path/to/xsched/output/lib:$LD_LIBRARY_PATH

# 运行应用
time ./app > /tmp/app2_xsched_low.log 2>&1
```

**终端3（监控）**：
```bash
# 实时监控XQueue状态
/path/to/xsched/output/bin/xcli top -f 10

# 应该看到：
# - 两个XQueue
# - 优先级分别为1和0
# - 高优先级XQueue的任务更频繁执行
```

**记录数据**：
```
高优先级进程平均延迟：______ ms
低优先级进程平均延迟：______ ms
GPU利用率：______%
```

---

#### ⏳ Task 3.5: 数据分析和对比

**目标**：分析XSched的效果

**对比表格**：

| 场景 | 高优先级延迟 | 低优先级延迟 | GPU利用率 |
|------|------------|------------|---------|
| **无XSched（基准）** | ~130ms | ~130ms | ~100% |
| **有XSched** | ~70ms (预期) | ~200ms (预期) | ~80% (预期) |
| **改善** | -46% ✅ | +54% ⚠️ | -20% ⚠️ |

**论文对比（NVIDIA Volta，Level-2）**：
- 高优先级延迟：-30% ~ -50%
- 低优先级延迟：+50% ~ +100%
- GPU利用率：82% ~ 85%

**我们的预期（AMD MI300，Level-1）**：
- 高优先级延迟：-30% ~ -40%（略差于Level-2）
- 低优先级延迟：+50% ~ +150%
- GPU利用率：75% ~ 80%（Level-1的overhead更高）

---

### 阶段4：高级测试（Example 2-3）（预计2-4小时）

#### ⏳ Task 4.1: Hint API测试（Example 2）

**目标**：验证动态优先级调整

**操作**：
```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/code/xsched/examples/Linux/2_give_hints

# 编译
make hip

# 运行
export LD_LIBRARY_PATH=/path/to/xsched/output/lib:$LD_LIBRARY_PATH
./app
```

**验证内容**：
- XQueue API是否正常
- 动态修改优先级是否生效
- xcli hint命令是否工作

---

#### ⏳ Task 4.2: 进程内调度测试（Example 3）

**目标**：验证Local Scheduler（进程内多队列调度）

**操作**：
```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/code/xsched/examples/Linux/3_intra_process_sched

# 编译
make hip

# 运行
export LD_LIBRARY_PATH=/path/to/xsched/output/lib:$LD_LIBRARY_PATH
./app
```

**验证内容**：
- 同一进程内多个XQueue
- Local Scheduler是否正确调度
- 优先级是否生效

---

### 阶段5：性能分析和对比（预计2-4小时）

#### ⏳ Task 5.1: 详细性能Profiling

**目标**：深入分析XSched的性能特征

**工具**：
- `rocprof`：AMD GPU profiler
- `rocm-smi`：GPU监控
- XSched自带的性能统计

**测试场景**：
1. **Overhead测试**：不同Threshold的overhead
2. **抢占延迟测试**：测量实际抢占延迟
3. **GPU利用率测试**：在不同负载下的利用率

**Threshold实验**：
```bash
# 测试不同的Threshold设置
for threshold in 4 8 16 32 64; do
    export XSCHED_AUTO_XQUEUE_THRESHOLD=$threshold
    export XSCHED_AUTO_XQUEUE_BATCH_SIZE=$((threshold/2))
    ./app > /tmp/result_threshold_${threshold}.log
done

# 分析overhead和延迟的trade-off
```

**预期发现**（基于论文Figure 4）：
- Threshold=4: overhead ~10%, 抢占快
- Threshold=16: overhead ~3%, 抢占中等
- Threshold=64: overhead ~1%, 抢占慢

---

#### ⏳ Task 5.2: 对比GPREEMPT

**目标**：定性对比XSched (Level-1) 和 GPREEMPT

**对比维度**：

| 维度 | GPREEMPT | XSched (Level-1) | 差距 |
|------|---------|----------------|------|
| **抢占延迟** | ~40μs | ~200μs | 5倍慢 ⚠️ |
| **GPU利用率** | 88.6% | ~80% (预期) | -8% ⚠️ |
| **实施复杂度** | 高（需改驱动） | 低（纯用户空间） | ✅ |
| **通用性** | 中（需硬件支持） | 高（任何GPU） | ✅ |
| **Level-1 overhead** | N/A | 3-10% | ⚠️ |

**关键问题**：
1. 能否在MI300上实现Level-2？（需要调研KFD能力）
2. Level-1的overhead能否接受？
3. 对实际应用（如vLLM）的影响？

---

### 阶段6：评估MI300适用性（预计1-2天）

#### ⏳ Task 6.1: MI300硬件能力调研

**目标**：确定MI300是否支持Level-2/3

**调研内容**：

1. **KFD Queue Management能力**：
   - MI300的MES是否支持queue enable/disable？
   - KFD驱动是否暴露相关接口？

2. **查看ROCm文档和代码**：
```bash
# 查找KFD相关API
cd /opt/rocm
grep -r "kfd_ioctl" .
grep -r "queue.*suspend\|queue.*resume" .

# 查找HIP Stream相关
grep -r "hipStream" ./include
```

3. **实验验证**：
```bash
# 测试hipStreamQuery的行为
# 测试hipDeviceSynchronize的延迟
```

**可能的发现**：
- ✅ 如果MI300 MES支持queue控制 → 可以实现Level-2
- ⚠️ 如果不支持 → 只能使用Level-1
- ❌ 如果需要改驱动 → 短期无法实现

---

#### ⏳ Task 6.2: Level-2实现可行性评估

**如果MI300支持queue控制**：

**需要实现的接口**（在`platforms/hip/hal/src/queue.cpp`）：
```cpp
// Level-2接口
class HipQueue : public HwQueue {
public:
    // Level-1（已实现）
    void Launch(HwCommand* cmd) override;
    void Synchronize() override;
    
    // Level-2（需要实现）✨
    void Deactivate() override {
        // 方案1：使用KFD ioctl暂停queue（如果支持）
        // 方案2：使用hipStreamSynchronize() + 手动暂停提交
    }
    
    void Reactivate() override {
        // 恢复queue的执行
    }
};
```

**实现难度评估**：
- ⭐⭐ 如果KFD直接支持：中等难度
- ⭐⭐⭐⭐ 如果需要改ROCm Runtime：较高难度
- ⭐⭐⭐⭐⭐ 如果需要改内核驱动：很高难度

---

#### ⏳ Task 6.3: 编写评估报告

**报告内容**：
1. XSched在MI300上的实际性能（Level-1）
2. 与GPREEMPT的对比
3. Level-2/3的实现可行性
4. 对vLLM等应用的影响
5. 推荐方案

---

## 5. Docker环境配置

### 5.1 检查清单

```bash
# 在docker zhenaiter中执行

# 1. GPU可见性
rocm-smi
# 应该看到MI300 GPU

# 2. ROCm版本
rocminfo --version
apt list --installed | grep rocm-dev

# 3. HIP编译环境
hipconfig
which hipcc
hipcc --version

# 4. 编译工具
cmake --version  # >= 3.18
g++ --version    # >= 7.0
make --version

# 5. 运行时库
ldconfig -p | grep hip
ldconfig -p | grep rocm
```

### 5.2 缺少依赖时的安装

```bash
# 如果缺少ROCm开发包
apt update
apt install -y rocm-dev rocm-libs hip-dev

# 如果缺少编译工具
apt install -y cmake build-essential git

# 如果需要性能分析工具
apt install -y rocm-profiler rocprofiler-dev

# 设置环境变量（添加到~/.bashrc）
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm/hip
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
```

---

## 6. 对比实验方案

### 6.1 实验目标

**核心问题**：
1. XSched (Level-1) 在MI300上的实际性能如何？
2. 相比GPREEMPT的差距有多大？
3. 是否值得投入资源实现Level-2？

### 6.2 实验矩阵

| 实验编号 | 场景描述 | 对比方案 | 评估指标 |
|---------|---------|---------|---------|
| **Exp-1** | 单进程基准 | 无调度 | 基准性能 |
| **Exp-2** | 双进程公平竞争 | 无调度 | 基准延迟和利用率 |
| **Exp-3** | 双进程优先级 | XSched Level-1 | 优先级效果 |
| **Exp-4** | Threshold影响 | XSched (不同Threshold) | Overhead vs 抢占延迟 |
| **Exp-5** | 多进程混合 | XSched vs 无调度 | 复杂场景表现 |
| **Exp-6** | 实际应用集成 | vLLM + XSched (可选) | 实用性验证 |

### 6.3 评估指标

**性能指标**：
- **任务延迟**（Latency）：P50, P90, P99
- **任务吞吐**（Throughput）：tasks/second
- **GPU利用率**：平均利用率，通过rocm-smi测量
- **抢占延迟**：高优先级任务的响应时间

**Overhead指标**：
- **Progressive Launching Overhead**：相对无XSched的slowdown
- **Context Switch Overhead**：切换延迟

**可用性指标**：
- **易用性**：环境变量配置 vs 代码修改
- **透明性**：是否需要修改应用代码
- **通用性**：是否适用于各种workload

### 6.4 预期结果

**基于论文数据和Level-1限制的推测**：

| 指标 | GPREEMPT (A100, 硬件) | XSched (MI300, Level-1) | 差距 |
|------|---------------------|----------------------|------|
| GPU利用率 | 88.6% | ~80% (预期) | -8% |
| 高优先级延迟 | +20% (vs LCO) | +30-40% (预期) | +10-20% |
| 低优先级延迟 | +100% (vs NP) | +150-200% (预期) | +50-100% |
| 抢占延迟 | ~40μs | ~200μs (预期) | 5倍慢 |
| Overhead | 很低 | 3-10% (Threshold相关) | 较高 |

**关键洞察**：
- ✅ XSched (Level-1) 可以工作，但性能不如GPREEMPT
- ⚠️ 如果能实现Level-2，性能会显著接近GPREEMPT
- ❓ 对于vLLM等应用，Level-1是否足够？（需要实际测试）

---

## 7. 常见问题和解决方案

### Q1: 编译XSched时找不到HIP

**错误**：
```
CMake Error: Could not find HIP
```

**解决**：
```bash
export HIP_PATH=/opt/rocm/hip
export ROCM_PATH=/opt/rocm
export CMAKE_PREFIX_PATH=/opt/rocm
```

### Q2: 运行时找不到libshimhip.so

**错误**：
```
error while loading shared libraries: libshimhip.so: cannot open shared object file
```

**解决**：
```bash
export LD_LIBRARY_PATH=/path/to/xsched/output/lib:$LD_LIBRARY_PATH
# 或将其添加到ldconfig
```

### Q3: xserver无法启动

**错误**：
```
[ERROR] Failed to bind to port 50000
```

**解决**：
```bash
# 检查端口是否被占用
netstat -tulpn | grep 50000

# 使用其他端口
xserver HPF 50001
```

### Q4: 应用运行但没有使用XSched

**症状**：设置了环境变量，但应用行为没有变化

**排查**：
```bash
# 1. 检查xserver是否运行
ps aux | grep xserver

# 2. 检查环境变量
echo $XSCHED_SCHEDULER
echo $LD_LIBRARY_PATH

# 3. 检查应用是否加载了shim库
ldd ./app | grep shim

# 4. 启用调试日志
export XSCHED_LOG_LEVEL=DEBUG
./app
```

### Q5: rocm-smi显示GPU利用率为0

**原因**：MI300可能使用不同的监控方式

**解决**：
```bash
# 使用rocprof
rocprof --stats ./app

# 或使用rocm-bandwidth-test
rocm-bandwidth-test

# 检查GPU活动
watch -n 1 rocm-smi
```

---

## 8. 下一步行动

### 立即可做（1-2天）：

1. **✅ 已完成**：下载XSched代码
2. **⏳ 进行中**：检查Docker环境
3. **📋 待办**：编译XSched HIP支持
4. **📋 待办**：运行Example 1基础测试
5. **📋 待办**：生成初步性能报告

### 短期计划（1-2周）：

1. 完成Example 1-3的测试
2. 分析性能数据
3. 评估MI300的Level-2可行性
4. 编写详细评估报告

### 中期计划（1-2月）：

1. 如果Level-2可行，实现Level-2支持
2. 集成到实际应用（如vLLM）
3. 与GPREEMPT进行全面对比

---

## 9. 参考资料

### 论文和文档

- **XSched论文**：`docs/xsched-osdi25.pdf`
- **XSched博客**：`docs/xsched-intro-2025-en.md`
- **Artifacts仓库**：https://github.com/XpuOS/xsched-artifacts
- **GPREEMPT论文**：已下载在`papers/`目录

### 代码仓库

- **XSched主仓库**：https://github.com/XpuOS/xsched
- **GPREEMPT代码**：https://github.com/thustorage/GPreempt

### ROCm文档

- **HIP Programming Guide**：https://rocm.docs.amd.com/projects/HIP/
- **ROCm System Management**：https://rocm.docs.amd.com/en/latest/
- **KFD Documentation**：Linux Kernel Documentation

---

## 10. 实验日志模板

### 实验日志格式

```markdown
## 实验日期：2026-01-XX

### 实验环境
- GPU型号：
- ROCm版本：
- XSched版本：
- Docker镜像：

### 实验内容
- 测试用例：
- 配置参数：

### 实验结果
- 性能数据：
- 观察到的现象：
- 遇到的问题：

### 结论和下一步
- 主要发现：
- 待解决问题：
- 下一步计划：
```

---

## 总结

**XSched的优势**：
- ✅ 纯用户空间实现，易于部署
- ✅ 支持AMD GPU (HIP)
- ✅ 已有论文验证的效果
- ✅ 开源，可以修改和扩展

**XSched的限制**：
- ⚠️ Level-1性能低于GPREEMPT的硬件方案
- ⚠️ Progressive Launching有一定overhead
- ⚠️ Level-2/3在AMD GPU上需要验证

**我们的行动**：
1. 🔴 **P0**: 快速验证XSched Level-1的基本功能（Example 1）
2. 🟠 **P1**: 测量实际性能并与GPREEMPT对比
3. 🟡 **P2**: 评估Level-2在MI300上的可行性
4. 🟢 **P3**: 如果性能可接受，集成到实际应用

**预期时间线**：
- Day 1-2: 环境准备和基础测试
- Day 3-5: 性能评估和分析
- Day 6-7: Level-2可行性调研
- Week 2: 编写详细报告和下一步建议

---

**文档结束 - 准备开始实验！** 🚀

