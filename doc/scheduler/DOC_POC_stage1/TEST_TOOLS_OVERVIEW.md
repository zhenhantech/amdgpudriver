# POC Stage 1 测试工具总览

本目录包含了POC Stage 1的所有测试工具和脚本。

---

## 📂 目录结构

```
code/
├── 🔧 基础测试工具
│   ├── test_simple_gemm_3min.py          # 简单GEMM测试（3分钟）
│   ├── test_simple_pytorch_3min.py       # 简单PyTorch测试（3分钟）
│   ├── run_simple_tests.sh               # 运行简单测试的wrapper
│   └── test_gemm_mini.py                 # 迷你GEMM测试（10秒，用于详细trace）
│
├── 🎯 Case对比测试
│   ├── case_a_cnn.py                     # Case-A: CNN模型（Online-AI）
│   ├── case_b_transformer.py             # Case-B: Transformer模型（Offline-AI）
│   └── run_case_comparison.sh            # 运行Case-A和Case-B对比测试
│
├── 📊 高级测试（新增）
│   └── run_deepseek_with_ftrace.sh       # DeepSeek 3.2 + ftrace测试
│
├── 🔍 分析工具
│   ├── get_docker_pid_mapping.sh         # Docker PID映射查询工具
│   └── run_gemm_with_ftrace.sh           # GEMM + ftrace同步测试
│
├── 🛠️ POC实现
│   └── poc_implementation/
│       ├── queue_finder.c                # Queue查找工具（C）
│       ├── Makefile                      # 编译Queue Finder
│       ├── test_queue_finder.sh          # 测试Queue Finder
│       └── README.md                     # Queue Finder文档
│
├── 📚 文档
│   ├── DOCKER_PID_SOLUTION.md            # Docker PID映射方案
│   ├── FTRACE_ANALYSIS_GUIDE.md          # ftrace分析指南
│   └── DEEPSEEK_TEST_GUIDE.md            # DeepSeek测试指南（新增）
│
└── 📁 日志目录
    └── log/
        ├── gemm_ftrace_<timestamp>/      # GEMM + ftrace日志
        ├── case_comparison_<timestamp>/  # Case对比日志
        └── deepseek_ftrace_<timestamp>/  # DeepSeek测试日志
```

---

## 🚀 快速开始指南

### 1️⃣ 简单测试（Queue调试）

**用途**: 快速验证Queue使用，适合初步调试

```bash
# GEMM测试（3分钟）
./run_simple_tests.sh gemm

# PyTorch测试（3分钟）
./run_simple_tests.sh pytorch
```

**输出**: 容器日志，包含基本的GPU计算信息

---

### 2️⃣ GEMM + ftrace（详细trace）

**用途**: 理解ROCm runtime和KFD的交互关系

```bash
sudo ./run_gemm_with_ftrace.sh <container_name>
```

**输出**:
- `gemm_ftrace_<timestamp>/amd_log.txt` - AMD日志（Level 5）
- `gemm_ftrace_<timestamp>/ftrace.txt` - Kernel ftrace日志
- `gemm_ftrace_<timestamp>/analyze.sh` - 自动分析脚本

**推荐阅读**: [FTRACE_ANALYSIS_GUIDE.md](FTRACE_ANALYSIS_GUIDE.md)

---

### 3️⃣ Case-A vs Case-B 对比测试

**用途**: 对比Online-AI（CNN）和Offline-AI（Transformer）的Queue使用

```bash
./run_case_comparison.sh <container_name>
```

**输出**:
- `case_comparison_<timestamp>/case_a_cnn.log` - Case-A日志
- `case_comparison_<timestamp>/case_b_transformer.log` - Case-B日志
- `case_comparison_<timestamp>/pid_mapping.txt` - PID映射
- `case_comparison_<timestamp>/analyze_logs.sh` - 分析脚本

**已完成分析**: [case_comparison_20260205_155247](log/case_comparison_20260205_155247/ANALYSIS_SUMMARY.md)

**关键发现**:
- ✅ Case-A和Case-B都使用单Queue模型
- ✅ RPTR ≈ WPTR，无Queue积压
- ✅ POC设计适用

---

### 4️⃣ DeepSeek 3.2 测试（新增）⭐

**用途**: 验证POC设计在复杂AI模型（8 GPU）下的适用性

```bash
sudo ./run_deepseek_with_ftrace.sh <container_name> [test_duration]
```

**示例**:
```bash
# 120秒测试（默认）
sudo ./run_deepseek_with_ftrace.sh zhen_vllm_dsv3

# 300秒测试
sudo ./run_deepseek_with_ftrace.sh zhen_vllm_dsv3 300
```

**输出**:
- `deepseek_ftrace_<timestamp>/deepseek_amd_log.txt` - AMD日志（Level 3）
- `deepseek_ftrace_<timestamp>/ftrace.txt` - Kernel ftrace日志
- `deepseek_ftrace_<timestamp>/queue_info.txt` - Queue使用统计
- `deepseek_ftrace_<timestamp>/analyze_deepseek.sh` - 详细分析脚本

**推荐阅读**: [DEEPSEEK_TEST_GUIDE.md](DEEPSEEK_TEST_GUIDE.md)

**关键验证点**:
- DeepSeek使用几个Queue？
- 多GPU环境下的Queue分配策略？
- POC设计是否需要调整？

---

## 🔍 POC实现工具

### Queue Finder（已实现）✅

**用途**: 从AMD日志中提取Queue信息，生成Python配置文件

```bash
cd poc_implementation/
make
./queue_finder <target_pid> <amd_log_path>
```

**输出**: `queue_config_pid_<pid>.py`

**测试**:
```bash
./test_queue_finder.sh
```

**已测试**: Case-A（PID 158036）和Case-B（PID 158122）

---

## 📊 测试场景对比

| 测试工具                          | GPU数 | 时长   | AMD日志级别 | ftrace | 用途                         |
|-----------------------------------|-------|--------|-------------|--------|------------------------------|
| `run_simple_tests.sh`             | 1     | 3分钟  | 默认        | ❌     | 快速Queue调试                |
| `run_gemm_with_ftrace.sh`         | 1     | 10秒   | Level 5     | ✅     | 理解ROCm-KFD交互             |
| `run_case_comparison.sh`          | 1     | 2分钟  | Level 5     | ❌     | Online vs Offline对比        |
| `run_deepseek_with_ftrace.sh` ⭐  | 8     | 2-5分钟| Level 3     | ✅     | 复杂模型验证POC设计          |

---

## 🎯 测试路径建议

### 阶段1: 基础理解（已完成✅）
1. ✅ 运行简单GEMM测试 → 理解基本Queue使用
2. ✅ 运行GEMM + ftrace → 理解ROCm-KFD交互
3. ✅ 运行Case-A vs Case-B → 发现单Queue模型

### 阶段2: POC设计验证（当前）
4. 🔄 **运行DeepSeek测试 → 验证复杂模型下的设计适用性**

### 阶段3: POC实现（规划中）
5. ⏳ Queue识别自动化
6. ⏳ Queue suspend/resume实现
7. ⏳ 完整抢占流程测试

---

## 📝 AMD日志级别说明

| Level | 内容                          | 日志量 | 适用场景                  |
|-------|-------------------------------|--------|---------------------------|
| 0     | 无日志                        | 最小   | 生产环境                  |
| 1     | 错误信息                      | 很小   | 错误调试                  |
| 2     | 警告信息                      | 小     | 一般调试                  |
| **3** | **Queue、Kernel提交**         | **中** | **本次DeepSeek测试（推荐）**|
| 4     | 增加Memory操作                | 大     | 深度调试                  |
| 5     | 所有KFD交互                   | 很大   | 完整trace（短时测试）     |

**选择建议**:
- **快速测试**: Level 3（平衡信息量和日志大小）
- **详细分析**: Level 5（仅用于短时测试，如GEMM 10秒）
- **长时间运行**: Level 2-3（避免日志过大）

---

## 🛠️ 常用命令参考

### Docker相关
```bash
# 查看容器状态
docker ps

# 获取容器PID
docker inspect -f '{{.State.Pid}}' <container_name>

# 进入容器
docker exec -it <container_name> bash

# 查看GPU
docker exec <container_name> rocm-smi --showid
```

### ftrace相关
```bash
# 挂载debugfs
sudo mount -t debugfs none /sys/kernel/debug

# 查看当前tracer
cat /sys/kernel/debug/tracing/current_tracer

# 清空trace
sudo sh -c "echo > /sys/kernel/debug/tracing/trace"

# 查看buffer大小
cat /sys/kernel/debug/tracing/buffer_size_kb
```

### 日志分析
```bash
# 提取Queue地址
grep 'HWq=0x' amd_log.txt | grep -o 'HWq=0x[0-9a-f]*' | sort -u

# 统计Queue数量
grep 'HWq=0x' amd_log.txt | grep -o 'HWq=0x[0-9a-f]*' | sort -u | wc -l

# 统计Kernel提交
grep -c 'KernelExecution.*enqueued' amd_log.txt

# 查看Queue指针
grep 'rptr\|wptr' amd_log.txt | head -20
```

---

## 📚 相关文档索引

### 设计文档
- [POC Stage 1 实施方案](../ARCH_Design_01_POC_Stage1_实施方案.md)
- [创新方案：Map/Unmap抢占](../New_DESIGN_MAP_UNMAP_BASED_PREEMPTION.md)
- [测试场景定义](../test_scenaria.md)

### 分析报告
- [Case-A/Case-B分析](log/case_comparison_20260205_155247/ANALYSIS_SUMMARY.md)
- [GEMM + ftrace分析](log/gemm_ftrace_20260205_143555/ANALYSIS_REPORT.md)

### 操作指南
- [Docker PID映射方案](DOCKER_PID_SOLUTION.md)
- [ftrace分析指南](FTRACE_ANALYSIS_GUIDE.md)
- [DeepSeek测试指南](DEEPSEEK_TEST_GUIDE.md) ⭐新增
- [Queue Finder工具说明](poc_implementation/README.md)

### 进度跟踪
- [下一步计划](../NEXT_STEPS_PREEMPTION_POC.md)
- [进度更新 2026-02-05](../PROGRESS_UPDATE_20260205.md)
- [今日总结 2026-02-05](../TODAY_SUMMARY_20260205.md)

---

## 🆘 问题排查

### 常见问题1: sudo权限不足
```bash
❌ 无法打开 /sys/kernel/debug/tracing/...
```
**解决**: 使用sudo运行需要ftrace的脚本

### 常见问题2: 容器未运行
```bash
❌ 无法获取容器PID
```
**解决**: 
```bash
docker ps
docker start <container_name>
```

### 常见问题3: 日志过大
```bash
⚠️  AMD日志超过1GB
```
**解决**: 
- 降低AMD_LOG_LEVEL（3或4）
- 缩短测试时长
- 清理旧日志

### 常见问题4: ftrace事件未找到
```bash
ℹ️  未发现自定义KFD events
```
**说明**: 这是正常的，脚本会使用function tracer替代

---

## 🎓 学习路径

### 初学者
1. 阅读 [POC Stage 1 实施方案](../ARCH_Design_01_POC_Stage1_实施方案.md)
2. 运行 `run_simple_tests.sh`
3. 查看日志，理解Queue概念

### 进阶
1. 阅读 [ftrace分析指南](FTRACE_ANALYSIS_GUIDE.md)
2. 运行 `run_gemm_with_ftrace.sh`
3. 分析ROCm-KFD交互

### 专家
1. 阅读 [创新方案文档](../New_DESIGN_MAP_UNMAP_BASED_PREEMPTION.md)
2. 运行 `run_deepseek_with_ftrace.sh`
3. 评估Map/Unmap方案适用性

---

## 📊 测试里程碑

| 日期       | 测试              | 结果                          | 状态 |
|------------|-------------------|-------------------------------|------|
| 2026-02-05 | Case-A vs Case-B  | 单Queue模型，RPTR≈WPTR        | ✅   |
| 2026-02-05 | Queue Finder      | 成功提取Queue信息             | ✅   |
| 2026-02-05 | DeepSeek脚本创建  | 脚本ready，待测试             | 🔄   |
| TBD        | DeepSeek测试      | 验证多GPU场景                 | ⏳   |
| TBD        | Queue suspend实现 | 实现IOCTL调用                 | ⏳   |
| TBD        | 完整POC验证       | Online抢占Offline             | ⏳   |

---

## 🔗 快速链接

**立即开始DeepSeek测试:**
```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
sudo ./run_deepseek_with_ftrace.sh zhen_vllm_dsv3 120
```

**查看最新分析:**
```bash
./log/deepseek_ftrace_<latest>/analyze_deepseek.sh
```

**回顾Case-A/Case-B结果:**
```bash
cat ./log/case_comparison_20260205_155247/ANALYSIS_SUMMARY.md
```

---

**最后更新**: 2026-02-05  
**版本**: 1.0  
**维护者**: POC Stage 1 Team

