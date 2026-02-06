# 简单测试脚本使用指南

**路径**: `/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code`  
**更新**: 2026-02-05  
**用途**: Queue监控调试

---

## 📋 概述

这两个测试脚本专门为Queue监控调试设计，相比DSV3.2更加简单、可控：

| 测试 | 文件 | 运行时长 | 特点 |
|------|------|---------|------|
| **GEMM测试** | `test_simple_gemm_3min.py` | 3分钟 | 纯矩阵乘法，单一操作类型 |
| **PyTorch测试** | `test_simple_pytorch_3min.py` | 3分钟 | 多种操作（Linear, ReLU, MatMul等） |

### 为什么选择这些测试？

✅ **简单可控**
- 不依赖XSched或复杂框架
- 运行时长固定（3分钟）
- 输出清晰，易于观察

✅ **适合调试**
- 队列使用稳定，不会频繁创建/销毁
- 可预测的GPU行为
- 便于重复测试

✅ **相比DSV3.2**
- 不需要加载大模型（省时间）
- 内存占用小
- 不会因模型问题导致测试失败

---

## 🚀 快速开始

### 方式1: 使用启动脚本（推荐）⭐⭐⭐

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 运行GEMM测试
./run_simple_tests.sh gemm

# 运行PyTorch测试
./run_simple_tests.sh pytorch

# 依次运行两个测试
./run_simple_tests.sh both

# 交互式选择
./run_simple_tests.sh
```

### 方式2: 直接运行Python脚本

```bash
# GEMM测试
python3 test_simple_gemm_3min.py

# PyTorch测试
python3 test_simple_pytorch_3min.py
```

---

## 💡 配合Queue监控使用

### 完整工作流（推荐）

#### 步骤1: 启动Queue监控

在**终端1**运行：

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 方式A: 自动监控（等待新进程）⭐⭐⭐
sudo ./watch_new_gpu.sh

# 方式B: 实时监控（需要先知道PID）
sudo ./watch_queue_live.sh <PID> 3

# 方式C: API持续监控
sudo ./queue_monitor <PID> 200 10
```

#### 步骤2: 运行测试

在**终端2**运行：

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 运行测试
./run_simple_tests.sh gemm
```

#### 步骤3: 观察结果

监控工具会自动显示：
- ✅ 检测到新的GPU进程
- ✅ Queue数量和ID
- ✅ 队列状态变化
- ✅ HQD信息（如果使用sysfs监控）

---

## 📊 测试详情

### Test 1: GEMM测试

**文件**: `test_simple_gemm_3min.py`

**特点**:
- 纯矩阵乘法运算 (2048x2048)
- 持续提交相同类型的kernel
- 队列使用非常稳定
- 适合验证基本的Queue监控功能

**输出示例**:
```
━━━ 开始GEMM测试 ━━━
  测试时长:       180 秒 (3.0 分钟)
  矩阵大小:       2048 x 2048
  数据类型:       float32

  时间(s) | 迭代次数 | 平均GFLOPS | 已用内存(GB) | 剩余时间(s)
  -----------------------------------------------------------------
     10.0 |      523 |     2847.32 |        0.032 |        170.0
     20.0 |     1046 |     2851.18 |        0.032 |        160.0
     30.0 |     1569 |     2849.95 |        0.032 |        150.0
...
```

**预期Queue行为**:
- 初始化时创建队列
- 队列数量保持稳定（通常24个，对应8个GPU）
- 队列ID不变
- HQD稳定映射（1:4比例，MI308X）

---

### Test 2: PyTorch测试

**文件**: `test_simple_pytorch_3min.py`

**特点**:
- 简单的3层神经网络
- 多种操作类型（Linear, ReLU, MatMul, Reduction等）
- 更接近真实AI workload
- 可以观察到不同kernel对Queue的使用

**输出示例**:
```
━━━ 开始PyTorch测试 ━━━
  测试时长:       180 秒 (3.0 分钟)
  批次大小:       64
  输入维度:       1024

  时间(s) | 迭代次数 | 平均延迟(ms) | 已用内存(GB) | 剩余时间(s)
  --------------------------------------------------------------------
     10.0 |      412 |         24.27 |        0.156 |        170.0
     20.0 |      824 |         24.31 |        0.156 |        160.0
     30.0 |     1236 |         24.28 |        0.156 |        150.0
...
```

**包含的操作**:
1. **Linear层** (全连接层)
2. **ReLU** (激活函数)
3. **Element-wise加法**
4. **矩阵乘法**
5. **Reduction** (求平均)

**预期Queue行为**:
- 初始化时创建队列
- 队列数量稳定
- 可能看到不同类型的Queue（Compute、DMA等）
- Queue活跃度可能比GEMM略高（更多操作类型）

---

## 🔍 对比不同测试的Queue特征

### 运行对比实验

```bash
# 1. 运行GEMM测试，保存日志
sudo ./queue_monitor <PID_GEMM> 180 10 > gemm_queues.log 2>&1

# 2. 运行PyTorch测试，保存日志
sudo ./queue_monitor <PID_PYTORCH> 180 10 > pytorch_queues.log 2>&1

# 3. 对比队列使用
diff gemm_queues.log pytorch_queues.log
```

### 预期差异

| 特征 | GEMM测试 | PyTorch测试 |
|------|---------|------------|
| **Queue数量** | 24（稳定） | 24（稳定） |
| **Queue类型** | 主要Compute | Compute + 可能有DMA |
| **活跃度** | 高（持续GEMM） | 中高（多种操作） |
| **内存使用** | 低（~32MB） | 中（~156MB） |
| **可重复性** | 非常高 | 高 |

---

## 🐛 调试场景

### 场景1: 验证Queue监控工具

**目标**: 确认监控工具能正确检测和显示Queue

**步骤**:
1. 先运行简单的GEMM测试
2. 使用`watch_new_gpu.sh`自动检测
3. 验证Queue数量、ID正确显示

**预期结果**:
- ✅ 自动检测到进程
- ✅ 显示24个Queue
- ✅ Queue ID稳定不变

---

### 场景2: 测试不同监控方式

**目标**: 对比API和sysfs两种监控方式

**步骤**:
```bash
# 终端1: 运行测试
python3 test_simple_gemm_3min.py &
PID=$!

# 终端2: API监控
sudo ./queue_monitor $PID 180 10 > api_result.log &

# 终端3: sysfs监控
sudo ./monitor_queue_sysfs.sh $PID 180 10 > sysfs_result.log &

# 等待完成后对比
sudo ./compare_api_vs_sysfs.sh $PID
```

**预期结果**:
- ✅ 两种方式Queue数量一致
- ✅ MQD:HQD = 1:4 (MI308X)

---

### 场景3: 队列稳定性测试

**目标**: 验证Queue在长时间运行中是否稳定

**步骤**:
```bash
# 使用持续采样监控
sudo ./monitor_queue_sysfs.sh <PID> 180 10

# 查看分析报告
cat queue_monitor_*/analysis.txt
```

**预期结果**:
```
监控结果分析:
  队列数量完全稳定
  所有采样都是 24 个队列
```

---

## 📝 常见问题

### Q1: 测试运行太快就结束了

**A**: 确认是否：
- Python环境正确
- CUDA可用
- 脚本有执行权限

可以先运行环境检查：
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Q2: 看不到Queue

**A**: 可能原因：
1. 监控启动太晚（程序已初始化完成）
   - **解决**: 使用`watch_new_gpu.sh`提前启动监控
2. 权限不足
   - **解决**: 使用`sudo`
3. PID不正确
   - **解决**: 使用`lsof /dev/kfd`查找正确的PID

### Q3: GEMM和PyTorch的Queue数量一样吗？

**A**: 通常是一样的。
- Queue数量主要取决于GPU数量和配置
- 不同的是Queue的**使用模式**
- GEMM: 持续高负载
- PyTorch: 多种操作混合

### Q4: 如何在Docker容器内运行？

**A**: 两种方式：

**方式1: 从宿主机监控Docker进程**
```bash
# 终端1: 监控
sudo ./watch_docker_gpu.sh <container_name> 300 5

# 终端2: 在容器内运行测试
docker exec -it <container_name> bash
cd /data/code/.../code
./run_simple_tests.sh gemm
```

**方式2: 在容器内运行所有工具**
```bash
docker exec -it <container_name> bash
cd /data/code/.../code

# 终端1（容器内）: 监控
sudo ./watch_gpu_in_docker.sh

# 终端2（容器内）: 测试
./run_simple_tests.sh gemm
```

---

## 🎯 最佳实践

### 1. 首次测试建议流程

```bash
# Step 1: 环境检查
./run_simple_tests.sh  # 会自动检查环境

# Step 2: 运行最简单的测试
./run_simple_tests.sh gemm

# Step 3: 如果成功，再测试PyTorch
./run_simple_tests.sh pytorch
```

### 2. 调试Queue监控工具

```bash
# 使用GEMM测试（最简单、最稳定）
# 终端1
sudo ./watch_new_gpu.sh

# 终端2
./run_simple_tests.sh gemm
```

### 3. 对比实验

```bash
# 依次运行两个测试，记录Queue差异
./run_simple_tests.sh both

# 在监控日志中查找差异
grep "队列" monitor.log
```

---

## 📚 相关文档

- **MONITORING_GUIDE_DSV3.md** - 完整的监控使用指南
- **README.md** - 所有工具的总览
- **DOCKER_INSIDE_GUIDE.md** - Docker内监控指南
- **QUICK_DOCKER_REFERENCE.md** - Docker快速参考

---

## 🔗 脚本文件

| 文件 | 说明 | 用法 |
|------|------|------|
| `test_simple_gemm_3min.py` | GEMM测试脚本 | `python3 test_simple_gemm_3min.py` |
| `test_simple_pytorch_3min.py` | PyTorch测试脚本 | `python3 test_simple_pytorch_3min.py` |
| `run_simple_tests.sh` | 启动脚本 | `./run_simple_tests.sh [gemm\|pytorch\|both]` |

---

## 总结

这两个简单测试脚本是Queue监控调试的**理想选择**：

✅ **简单**: 无需复杂配置，开箱即用  
✅ **可控**: 固定3分钟运行时长，行为可预测  
✅ **稳定**: 队列使用稳定，便于验证监控工具  
✅ **轻量**: 不需要加载大模型，启动快速  
✅ **实用**: 相比DSV3.2更适合调试阶段

**下一步**:
1. 使用GEMM测试验证基本监控功能
2. 使用PyTorch测试观察多种操作的Queue行为
3. 成功后再进行DSV3.2等复杂测试

---

**最后更新**: 2026-02-05  
**维护者**: AI Assistant  
**测试平台**: MI308X + ROCm 7.x
