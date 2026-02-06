# POC Stage 1 - 代码目录

**路径**: `/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code`  
**最后更新**: 2026-02-05

---

## 📦 目录结构

```
code/
├── README.md                           # 本文档
│
├── C++ POC工具 (推荐使用)
│   ├── kfd_queue_monitor.hpp          # Queue监控器头文件
│   ├── kfd_queue_monitor.cpp          # Queue监控器实现
│   ├── queue_monitor_main.cpp         # Queue监控工具主程序
│   └── kfd_preemption_poc.cpp         # Queue抢占POC工具
│
├── C测试工具
│   ├── get_queue_info.c               # 快速Queue信息查询
│   ├── test_gpreempt_ioctl.c          # 测试自定义IOCTL
│   ├── preempt_queue_manual.c         # 手动抢占测试v1
│   └── preempt_queue_manual_v2.c      # 手动抢占测试v2
│
├── 简单测试脚本 (新增) ⭐⭐⭐⭐⭐
│   ├── test_simple_gemm_3min.py       # GEMM测试（3分钟）
│   ├── test_simple_pytorch_3min.py    # PyTorch测试（3分钟）
│   └── run_simple_tests.sh            # 测试启动脚本
│
├── Case对比测试 (新增) ⭐⭐⭐⭐
│   ├── case_a_cnn.py                  # Case-A: CNN测试
│   ├── case_b_transformer.py          # Case-B: Transformer测试
│   ├── run_case_comparison.sh         # 对比测试脚本
│   ├── test_preemption_simple.py      # 抢占测试
│   ├── analyze_queue_logs.py          # 日志分析工具
│   └── test_gemm_with_debug.sh        # 带AMD调试日志的测试
│
├── ftrace集成测试 (新增) ⭐⭐⭐⭐⭐
│   ├── test_gemm_mini.py              # Mini GEMM测试（100次迭代）
│   ├── run_gemm_with_ftrace.sh        # ftrace + AMD日志同步捕获
│   └── FTRACE_ANALYSIS_GUIDE.md       # ftrace分析详细指南
│
├── Shell测试脚本
│   ├── 智能监控 (新增) ⭐⭐⭐
│   │   ├── watch_new_gpu.sh           # 宿主机：等待新GPU进程
│   │   ├── watch_gpu_in_docker.sh     # 容器内：等待新GPU进程 ⭐
│   │   ├── auto_monitor.sh            # 宿主机：智能监控（多种模式）
│   │   ├── watch_docker_gpu.sh        # 宿主机：监控指定容器
│   │   ├── find_container_gpu_pids.sh # 宿主机：查找容器GPU进程
│   │   └── list_gpu_processes.sh      # 列出GPU进程（容器内/宿主机）
│   ├── sysfs监控脚本 (新增) ⭐⭐
│   │   ├── monitor_queue_sysfs.sh     # 从sysfs持续监控Queue状态
│   │   ├── compare_api_vs_sysfs.sh    # 对比API和sysfs两种方式
│   │   └── watch_queue_live.sh        # 实时监控（类似htop）
│   ├── 诊断工具 (新增) ⭐⭐
│   │   ├── diagnose_process.sh        # 诊断进程GPU使用情况
│   │   └── monitor_with_pid.sh        # 监控指定PID（带验证）
│   ├── 一键测试
│   │   └── test_userspace_poc.sh      # 完整POC测试
│   ├── 实验脚本
│   │   ├── exp01_with_api.sh          # 实验1（使用API）
│   │   ├── exp01_redesigned.sh        # 实验1（重新设计）
│   │   └── exp01_queue_monitor_fixed.sh # 实验1（修复版）
│   ├── Queue测试
│   │   ├── host_queue_test.sh         # 宿主机Queue测试
│   │   ├── host_queue_consistency_test.sh # 宿主机一致性测试
│   │   ├── quick_queue_test.sh        # 快速Queue测试
│   │   ├── quick_queue_test_hip.sh    # HIP快速测试
│   │   ├── reliable_queue_test.sh     # 可靠的Queue测试
│   │   └── test_queue_consistency.sh  # Queue一致性测试
│   └── 调试工具
│       └── fix_debugfs.sh             # Debugfs诊断修复
│
└── Makefile                           # 编译脚本
```

---

## 🚀 快速开始

### 0. 简单测试 + 监控 ⭐⭐⭐⭐⭐ (调试专用，最推荐)

**场景**: 不想用DSV3.2这种复杂程序，需要简单可控的测试

**为什么用这个**:
- ✅ 运行时长固定（3分钟），不会突然结束
- ✅ 无需加载大模型，启动快速
- ✅ Queue使用稳定，便于观察
- ✅ 相比DSV3.2更简单，更适合调试

**两个测试**:
1. **GEMM测试**: 纯矩阵乘法，单一操作
2. **PyTorch测试**: 神经网络推理，多种操作

#### 使用方法

```bash
# 终端1: 启动监控（会等待新的GPU进程）
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
sudo ./watch_new_gpu.sh

# 终端2: 运行简单测试
./run_simple_tests.sh gemm      # GEMM测试（3分钟）
./run_simple_tests.sh pytorch   # PyTorch测试（3分钟）
./run_simple_tests.sh both      # 两个都运行（6分钟）
```

**详细文档**: 查看 `SIMPLE_TESTS_GUIDE.md` ⭐⭐⭐⭐⭐

---

### 1. 无需PID的快速监控 ⭐⭐⭐ (最简单)

**不知道进程PID？使用自动监控！**

#### 在宿主机运行

```bash
# 终端1: 启动监控（等待新GPU进程）
./watch_new_gpu.sh

# 终端2: 启动测试程序
python3 your_model.py
```

#### 在Docker容器内运行 ⭐ (推荐用于vLLM)

```bash
# 终端1: 容器内启动监控
docker exec -it zhen_vllm_dsv3 bash
cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
./watch_gpu_in_docker.sh

# 终端2: 容器内启动vLLM
docker exec -it zhen_vllm_dsv3 bash
cd /data/code/rampup_doc/vLLM_test/scripts
./run_vLLM_v1_optimized.sh test
```

监控脚本会自动检测新的GPU进程并开始监控！

**详细说明**: 
- 宿主机: `MONITOR_WITHOUT_PID.md`
- Docker: `DOCKER_INSIDE_GUIDE.md` ⭐
- 快速参考: `QUICK_DOCKER_REFERENCE.md`

---

### 1. 编译所有工具

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 清理并编译
make clean
make all

# 验证编译结果
ls -lh queue_monitor kfd_preemption_poc get_queue_info
```

**编译产物**:
- `queue_monitor` - Queue监控工具（C++）
- `kfd_preemption_poc` - 抢占POC工具（C++）
- `get_queue_info` - Queue查询工具（C）

### 2. 一键测试

```bash
# 自动编译+测试所有工具
./test_userspace_poc.sh
```

### 3. 单独测试工具

```bash
# 获取目标进程PID
CONTAINER_PID=$(docker exec zhenaiter pgrep -f python3 | head -1)

# 监控Queue（30秒，每5秒采样）
sudo ./queue_monitor $CONTAINER_PID 30 5

# 测试抢占（10次迭代）
sudo ./kfd_preemption_poc $CONTAINER_PID 10

# 快速查询
sudo ./get_queue_info $CONTAINER_PID
```

---

## 📚 C++ POC工具（推荐）⭐⭐⭐

### queue_monitor - Queue监控工具

**源文件**:
- `kfd_queue_monitor.hpp` - 头文件
- `kfd_queue_monitor.cpp` - 实现
- `queue_monitor_main.cpp` - 主程序

**功能**:
- ✅ 持续监控Queue使用情况
- ✅ 统计分析（频率、稳定性、分布）
- ✅ 详细输出（Queue ID、GPU、Ring地址、CWSR地址）
- ✅ 自动生成POC代码片段

**用法**:
```bash
sudo ./queue_monitor <pid> [duration] [interval]

# 示例：监控60秒，每10秒采样
sudo ./queue_monitor 12345 60 10
```

**输出**:
- 实时采样结果
- 第一个快照的详细信息
- 统计分析和POC建议
- C++格式的queue_ids数组

---

### kfd_preemption_poc - 抢占POC工具

**源文件**:
- `kfd_preemption_poc.cpp` - 主程序
- `kfd_queue_monitor.cpp` - 依赖库

**功能**:
- ✅ 循环测试Queue的Suspend/Resume
- ✅ 性能测量（延迟、成功率）
- ✅ 模拟Online-AI推理
- ✅ 统计分析

**用法**:
```bash
sudo ./kfd_preemption_poc <offline_pid> [iterations]

# 示例：运行100次抢占测试
sudo ./kfd_preemption_poc 12345 100
```

**输出**:
- 每次迭代的Suspend/Resume延迟
- 总体统计（成功率、平均延迟、最小/最大延迟）
- POC性能指标

---

### get_queue_info - 快速查询工具

**源文件**:
- `get_queue_info.c` - C语言实现

**功能**:
- ✅ 快速查看Queue信息
- ✅ 单次查询
- ✅ 友好输出

**用法**:
```bash
sudo ./get_queue_info <pid>

# 示例
sudo ./get_queue_info 12345
```

**输出**:
- Queue详细信息
- 统计信息（类型分布、GPU分布）
- C格式的queue_ids数组

---

## 🧪 Shell测试脚本

### 智能监控脚本 (新增) ⭐⭐⭐

**不知道进程PID？使用智能监控！**

#### watch_new_gpu.sh - 等待新GPU进程（最简单）

**最适合**: 双终端协同工作

```bash
# 终端1: 启动监控
./watch_new_gpu.sh

# 终端2: 启动测试
python3 your_model.py
```

**功能**:
- ✅ 自动检测新的GPU进程
- ✅ 自动开始监控
- ✅ 无需任何参数
- ✅ 固定监控30秒，每5秒采样

**输出**:
```
╔════════════════════════════════════════════════════════╗
║  GPU进程监控 - 等待模式                                 ║
╚════════════════════════════════════════════════════════╝

⏳ 等待新的GPU进程启动...

💡 提示: 现在可以在另一个终端启动测试程序

✅ 检测到新的GPU进程!
  PID:    12345
  进程:   python3
```

---

#### auto_monitor.sh - 智能监控（多种模式）

**更灵活的监控方式**

```bash
# 模式1: 交互式选择
./auto_monitor.sh

# 模式2: 等待新进程
./auto_monitor.sh --wait

# 模式3: 按进程名匹配
./auto_monitor.sh --name python3

# 模式4: 监控容器内进程
./auto_monitor.sh --container zhenaiter

# 模式5: 查看所有GPU进程
./auto_monitor.sh --all

# 自定义监控参数
./auto_monitor.sh --wait --duration 60 --interval 10
```

**功能**:
- ✅ 多种自动检测模式
- ✅ 交互式进程选择
- ✅ Docker容器支持
- ✅ 可自定义监控参数
- ✅ 显示所有GPU进程信息

**完整说明**: 查看 `MONITOR_WITHOUT_PID.md`

---

### Docker容器内监控 ⭐ (vLLM用户专用)

**场景**: 在 `zhen_vllm_dsv3` 容器内运行监控工具

**路径映射**:
- 宿主机: `/mnt/md0/zhehan/code/coderampup/.../code`
- 容器内: `/data/code/coderampup/.../code`

#### watch_gpu_in_docker.sh - 容器内等待新进程

**最推荐的Docker内监控方式！**

```bash
# 终端1: 容器内启动监控
docker exec -it zhen_vllm_dsv3 bash
cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
./watch_gpu_in_docker.sh

# 终端2: 容器内启动vLLM
docker exec -it zhen_vllm_dsv3 bash
cd /data/code/rampup_doc/vLLM_test/scripts
./run_vLLM_v1_optimized.sh test
```

**功能**:
- ✅ 在容器内直接运行
- ✅ 自动检测容器内的新GPU进程
- ✅ 无需PID转换
- ✅ 监控60秒，每10秒采样

**详细说明**: 查看 `DOCKER_INSIDE_GUIDE.md` ⭐⭐⭐

#### list_gpu_processes.sh - 列出GPU进程

**快速查看当前GPU进程**

```bash
# 在容器内运行
./list_gpu_processes.sh
```

**输出**:
- 列出所有GPU进程（PID、进程名、命令）
- 显示如何监控这些进程

#### 快速参考

查看 `QUICK_DOCKER_REFERENCE.md` 获取快速命令参考。

---

### 5. sysfs/debugfs 监控 ⭐⭐ (新增)

**场景**: 直接从 `/sys/kernel/debug/kfd` 读取Queue状态

这些脚本使用 **sysfs/debugfs** 方式监控Queue，补充API方式的功能。

#### monitor_queue_sysfs.sh - 持续采样监控

**持续采样MQD和HQD状态，保存历史快照**

```bash
# 监控指定进程，60秒，每10秒采样
./monitor_queue_sysfs.sh 12345 60 10

# 监控所有进程，30秒，每5秒采样（默认）
./monitor_queue_sysfs.sh
```

**功能**:
- ✅ 从 `/sys/kernel/debug/kfd/mqds` 读取MQD（软件队列）
- ✅ 从 `/sys/kernel/debug/kfd/hqds` 读取HQD（硬件队列）
- ✅ 保存所有采样快照到文件
- ✅ 分析队列数量的稳定性
- ✅ 统计队列分布情况

**输出**:
```
━━━ 采样 1 (15:30:45) ━━━
  MQD: PID 12345 有 24 个队列
       - Compute queue on device f7bc
       - Compute queue on device f7bc
       ...
  HQD: 96 个活跃硬件队列
       - Inst 0, CP Pipe 1, Queue 2
       ...

监控结果分析:
  队列数量完全稳定
  所有采样都是 24 个队列
```

#### compare_api_vs_sysfs.sh - 对比验证

**对比API和sysfs两种方式，验证数据一致性**

```bash
./compare_api_vs_sysfs.sh 12345
```

**功能**:
- ✅ 方法1: GET_QUEUE_SNAPSHOT API (IOCTL)
- ✅ 方法2: /sys/kernel/debug/kfd/mqds (sysfs)
- ✅ 方法3: /sys/kernel/debug/kfd/hqds (sysfs)
- ✅ 对比三种方式的队列数量
- ✅ 验证数据一致性

**输出**:
```
| 方法 | 队列数量 | 状态 |
|------|---------|------|
| API (GET_QUEUE_SNAPSHOT) | 24 | ✅ |
| sysfs (MQD) | 24 | ✅ |
| sysfs (HQD总计) | 96 | ✅ |

✅ API和sysfs数据一致！
   队列数量: 24

MQD到HQD映射比例: 1:4.0
ℹ️  MI308X典型映射: 1个MQD → 4个HQD (每个XCC一个)
```

#### watch_queue_live.sh - 实时监控

**类似htop的实时监控显示**

```bash
# 监控指定进程，每2秒刷新
./watch_queue_live.sh 12345 2

# 监控所有进程，每3秒刷新（默认）
./watch_queue_live.sh
```

**功能**:
- ✅ 实时刷新显示（类似htop）
- ✅ 自动检测进程状态
- ✅ 显示所有GPU进程概览
- ✅ 显示HQD使用情况
- ✅ 彩色输出，易于阅读
- ✅ 按Ctrl+C优雅退出

**输出**:
```
╔════════════════════════════════════════════════════════╗
║  KFD Queue 实时监控 - sysfs方式                        ║
╚════════════════════════════════════════════════════════╝

时间: 2026-02-05 15:30:45   刷新间隔: 3秒   (按Ctrl+C退出)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
目标进程: PID 12345
进程名:   python3
状态:     运行中
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Queue统计:
  总队列数: 24

队列详情:
No.  Type            Device    
───────────────────────────────
1    Compute         f7bc      
2    Compute         f7bc      
...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
硬件队列 (HQD):
  总活跃HQD: 96
  映射比例:  1:4.0 (MQD:HQD)
  ℹ️  MI308X: 每个MQD对应4个HQD (4个XCC)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**适用场景**:
- 实时观察Queue数量变化
- 监控模型初始化过程
- 调试Queue创建/销毁问题

#### sysfs监控的优势

| 对比项 | API方式 | sysfs方式 |
|--------|---------|-----------|
| 访问方式 | IOCTL | 读取文件 |
| 权限要求 | sudo | sudo |
| 数据内容 | Queue ID + 地址 | 完整MQD/HQD内容 |
| 适用场景 | 抢占控制 | 诊断分析 |
| 历史记录 | ❌ | ✅ (可保存快照) |
| 实时性 | ✅ 高 | ✅ 高 |

**何时使用**:
- ✅ 需要查看MQD/HQD内部结构时
- ✅ 需要保存历史快照进行分析时
- ✅ 验证API数据一致性时
- ✅ 调试队列问题时
- ✅ 不需要抢占控制，只需监控时

**兼容性**:
- 这些脚本可以在宿主机或Docker容器内运行
- 需要确保 `/sys/kernel/debug/kfd` 可访问
- 可与API方式的工具（如 `queue_monitor`）配合使用

---

### 一键测试脚本

#### test_userspace_poc.sh ⭐⭐⭐

**最推荐的测试方式**

```bash
./test_userspace_poc.sh
```

**功能**:
1. 自动编译所有工具
2. 在Docker内启动测试模型
3. 测试queue_monitor（采样4次）
4. 测试kfd_preemption_poc（10次迭代）
5. 自动清理

**预计时间**: ~2分钟

---

### 实验脚本（Experiment 1）

#### exp01_with_api.sh ⭐

**使用GET_QUEUE_SNAPSHOT API的实验1**

```bash
./exp01_with_api.sh
```

**功能**:
- 启动测试模型
- 使用API持续采样Queue信息（10次）
- 分析队列稳定性
- 生成POC代码片段

**优点**: 不需要解析debugfs hex dump

---

#### exp01_redesigned.sh

**重新设计的实验1（解析debugfs方式）**

```bash
./exp01_redesigned.sh
```

**功能**:
- 解析`/sys/kernel/debug/kfd/mqds`和`hqds`
- 提取"queue on device"和HQD headers
- 适用于没有API访问权限的情况

---

#### exp01_queue_monitor_fixed.sh

**修复版实验1**

```bash
./exp01_queue_monitor_fixed.sh
```

**功能**: 早期实验脚本，修复了f-string语法错误

---

### Queue测试脚本

#### host_queue_test.sh

```bash
./host_queue_test.sh
```

**功能**: 从宿主机测试Docker容器内模型的Queue

---

#### host_queue_consistency_test.sh

```bash
./host_queue_consistency_test.sh
```

**功能**: 测试Queue ID的一致性（多次采样）

---

#### quick_queue_test.sh / quick_queue_test_hip.sh

```bash
./quick_queue_test.sh        # PyTorch版本
./quick_queue_test_hip.sh    # HIP版本
```

**功能**: 快速测试Queue创建和检测

---

#### reliable_queue_test.sh

```bash
./reliable_queue_test.sh
```

**功能**: 可靠的Queue测试（包含环境检查）

---

#### test_queue_consistency.sh

```bash
./test_queue_consistency.sh
```

**功能**: Queue一致性测试

---

### 调试脚本

#### fix_debugfs.sh

```bash
./fix_debugfs.sh
```

**功能**: 
- 检查debugfs是否可用
- 检查KFD驱动状态
- 诊断常见问题
- 提供修复建议

---

## 🔨 Makefile

### 编译目标

```bash
make all          # 编译所有工具
make clean        # 清理编译产物
make install      # 安装到系统（可选）
make test         # 测试编译
```

### 编译配置

- **编译器**: g++ (C++17) / gcc
- **头文件路径**: `/usr/src/amdgpu-6.12.12-2194681.el8_preempt/include/uapi`
- **优化级别**: -O2
- **依赖**: pthread

### 生成的可执行文件

- `queue_monitor` - C++工具
- `kfd_preemption_poc` - C++工具
- `get_queue_info` - C工具

---

## 🎯 推荐使用流程

### 第一次使用

1. **编译工具**:
   ```bash
   make clean && make all
   ```

2. **一键测试**:
   ```bash
   ./test_userspace_poc.sh
   ```

3. **查看文档**:
   - 查看 `../README_USERSPACE_POC.md` 了解详细用法
   - 查看 `../QUICKSTART_CPP_POC.md` 快速开始

### 日常开发

1. **启动测试模型** (Docker内):
   ```bash
   docker exec -it zhenaiter bash
   python3 your_model.py
   ```

2. **监控Queue** (宿主机):
   ```bash
   CONTAINER_PID=$(docker exec zhenaiter pgrep -f your_model.py)
   sudo ./queue_monitor $CONTAINER_PID 30 5
   ```

3. **测试抢占** (宿主机):
   ```bash
   sudo ./kfd_preemption_poc $CONTAINER_PID 100
   ```

### 实验场景

1. **验证单模型Queue稳定性**:
   ```bash
   ./exp01_with_api.sh
   ```

2. **对比两个模型**:
   ```bash
   # 分别测试模型A和模型B
   sudo ./queue_monitor $PID_A 20 5 > model_a.txt
   sudo ./queue_monitor $PID_B 20 5 > model_b.txt
   diff model_a.txt model_b.txt
   ```

3. **压力测试抢占**:
   ```bash
   # 1000次迭代
   sudo ./kfd_preemption_poc $PID 1000
   ```

---

## ⚠️ 注意事项

### 权限要求

所有工具都需要以下权限之一：
- **Root权限**: `sudo ./queue_monitor ...`
- **CAP_SYS_PTRACE**: `sudo setcap cap_sys_ptrace=eip ./queue_monitor`

### 常见问题

1. **"Failed to open /dev/kfd"**
   - 检查: `ls -l /dev/kfd`
   - 解决: `sudo modprobe amdgpu`

2. **"Failed to enable debug trap"**
   - 确保目标进程是GPU进程
   - 确保没有其他调试器附加
   - 使用sudo运行

3. **"No queues found"**
   - 等待模型初始化完成（10-20秒）
   - 确保模型真正使用GPU（`device='cuda'`）

4. **编译错误 "linux/kfd_ioctl.h: No such file"**
   - 修改Makefile中的INCLUDES路径
   - 查找头文件: `find /usr/src -name "kfd_ioctl.h"`

---

## 📊 性能指标（MI308X）

| 操作 | 典型延迟 |
|------|---------|
| `open_kfd()` | ~1 ms |
| `enable_debug_trap()` | ~1-5 ms |
| `get_queue_snapshot()` | ~100-200 μs |
| `suspend_queues()` | ~400-500 μs |
| `resume_queues()` | ~300-400 μs |

**单次完整抢占周期**: ~1-2 ms

---

## 📝 代码统计

```
C++ 文件:
  - 头文件: 1 个 (kfd_queue_monitor.hpp)
  - 源文件: 3 个 (*.cpp)
  - 总行数: ~900 行

C 文件:
  - 源文件: 4 个 (*.c)
  - 总行数: ~400 行

Python测试脚本 (新增):
  - 简单测试: 2 个 (test_simple_*.py)
  - 总行数: ~550 行

Shell 脚本:
  - 监控脚本: 14 个 (*.sh)
  - 总行数: ~1400 行

总计: ~3250 行代码
```

---

## 🔗 相关文档

### 本目录文档 (code/)

#### 测试和对比
- **QUICKSTART_FTRACE.md** ⭐⭐⭐⭐⭐ - ftrace快速开始（新）
- **FTRACE_ANALYSIS_GUIDE.md** ⭐⭐⭐⭐⭐ - ftrace详细分析指南（新）
- **CASE_COMPARISON_GUIDE.md** ⭐⭐⭐⭐⭐ - Case对比测试完整指南
- **PREEMPTION_DESIGN.md** ⭐⭐⭐⭐ - GPU抢占机制设计文档
- **HQD_INSPECTION_GUIDE.md** ⭐⭐⭐⭐ - HQD信息查看指南（新）
- **SIMPLE_TESTS_GUIDE.md** ⭐⭐⭐⭐⭐ - 简单测试使用指南
- **AMD_DEBUG_GUIDE.md** ⭐⭐⭐⭐ - AMD调试日志指南
- **QUICK_TEST_GUIDE.md** - 快速测试指南
- **QUICKSTART_CASE_COMPARISON.md** - Case对比快速开始

#### 监控相关
- **MONITORING_GUIDE_DSV3.md** - DeepSeek V3监控完整指南
- **DOCKER_INSIDE_GUIDE.md** ⭐⭐⭐ - Docker容器内监控指南
- **DOCKER_MONITORING_GUIDE.md** - Docker监控详细说明
- **QUICK_DOCKER_REFERENCE.md** - Docker快速参考
- **MONITOR_WITHOUT_PID.md** - 无需PID的监控方法

#### 其他
- **COMPILE_FIX.md** - 编译问题修复记录

### 上级目录文档 (DOC_POC_stage1/)

- **../README_USERSPACE_POC.md** - 完整使用指南（618行）
- **../QUICKSTART_CPP_POC.md** - 5分钟快速开始
- **../CPP_POC_FILES_SUMMARY.md** - 工具集总结
- **../GET_QUEUE_SNAPSHOT_API_GUIDE.md** - API详细说明
- **../00_INDEX_POC_TOOLS_AND_DOCS.md** - 完整索引

---

## 💡 下一步

### 新手推荐流程

1. **先用简单测试验证监控工具** ⭐⭐⭐⭐⭐
   ```bash
   # 终端1: 监控
   sudo ./watch_new_gpu.sh
   
   # 终端2: 测试
   ./run_simple_tests.sh gemm
   ```
   → 查看 `SIMPLE_TESTS_GUIDE.md`

2. **成功后，尝试DSV3或其他模型**
   ```bash
   # 参考 MONITORING_GUIDE_DSV3.md
   sudo ./watch_docker_gpu.sh zhen_vllm_dsv3 300 5
   ```

3. **深入测试**
   - 对比API和sysfs监控方式
   - 测试抢占功能
   - 分析性能数据

### 高级用户

1. **测试工具**: 运行 `./test_userspace_poc.sh`
2. **实际应用**: 用真实AI模型测试
3. **性能优化**: 分析Suspend/Resume延迟
4. **集成**: 将API集成到实际的GPU调度系统

---

**最后更新**: 2026-02-05  
**维护者**: AI Assistant  
**测试平台**: MI308X + ROCm 6.x
