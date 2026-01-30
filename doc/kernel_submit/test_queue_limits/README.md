# Stream Queue Limits 测试程序

**目的**: 验证软件队列（AQL Queue）和硬件队列（HQD）的数量限制和行为

**对应文档**: `SOFTWARE_HARDWARE_QUEUE_LIMITS.md`

---

## 📋 测试内容

### 测试场景

1. **16个Streams** - 硬件资源充足（50%利用率）
2. **32个Streams** - 硬件资源刚好够用（100%利用率）
3. **64个Streams** - 硬件资源不足，需要HQD复用

### 验证目标

✅ 每个Stream创建独立的AQL Queue  
✅ 每个AQL Queue有独立的ring buffer和doorbell  
✅ 硬件HQD的实际使用情况  
✅ CPSCH vs NOCPSCH模式的行为差异  
✅ HQD复用时的性能影响

---

## 🚀 快速开始

### 方式1: 使用脚本（推荐）

```bash
# 进入测试目录
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/test_queue_limits

# 给脚本执行权限
chmod +x run_tests.sh

# 运行所有对比测试（16, 32, 64 streams）
./run_tests.sh

# 或者测试特定数量的streams
./run_tests.sh 16   # 测试16个streams
./run_tests.sh 32   # 测试32个streams
./run_tests.sh 64   # 测试64个streams

# 启用KFD debug日志后测试
./run_tests.sh -d 32
```

### 方式2: 手动编译和运行

```bash
# 编译
hipcc -o test_multiple_streams test_multiple_streams.cpp

# 清空dmesg（可选）
sudo dmesg -C

# 运行测试
./test_multiple_streams 16        # 测试16个streams
./test_multiple_streams 32 -w     # 测试32个streams，创建后等待5秒
./test_multiple_streams 64 -t 10  # 测试64个streams，创建后等待10秒

# 查看结果
sudo dmesg | grep -E 'CREATE_QUEUE|hqd slot' | tail -50
```

---

## 📊 预期结果

### 16个Streams

```
软件层:
  ✓ 创建16个独立的AQL Queue
  ✓ 每个有独立的ring buffer和doorbell
  
硬件层:
  ✓ 使用16个HQD（NOCPSCH模式）
  ✓ 或者显示pipe=0, queue=0（CPSCH模式）
  ✓ HQD资源充足（使用率50%）
  
性能:
  ✓ 最优（无HQD复用，无Context Switch开销）
```

**dmesg输出示例**:

```bash
$ sudo dmesg | grep CREATE_QUEUE | wc -l
16

$ sudo dmesg | grep "hqd slot" | tail -5
[timestamp] kfd: hqd slot - pipe 0, queue 0
[timestamp] kfd: hqd slot - pipe 1, queue 0
[timestamp] kfd: hqd slot - pipe 2, queue 0
[timestamp] kfd: hqd slot - pipe 3, queue 0
[timestamp] kfd: hqd slot - pipe 0, queue 1
```

### 32个Streams

```
软件层:
  ✓ 创建32个独立的AQL Queue
  
硬件层:
  ✓ 使用全部32个HQD（100%利用率）
  ✓ 刚好用完所有硬件资源
  
性能:
  ✓ 良好（每个Queue仍有独占HQD）
```

**dmesg输出**:

```bash
$ sudo dmesg | grep CREATE_QUEUE | wc -l
32

$ sudo dmesg | grep "hqd slot" | wc -l
32  # NOCPSCH模式
0   # CPSCH模式（不使用固定HQD）
```

### 64个Streams

```
软件层:
  ✓ 创建64个独立的AQL Queue
  
硬件层:
  ⚠ 只有32个HQD可用
  ⚠ 需要HQD复用（2:1的共享比例）
  
性能:
  ⚠ 下降（Context Switch开销）
  ⚠ 预计性能下降20-40%
```

**dmesg输出**:

```bash
$ sudo dmesg | grep CREATE_QUEUE | wc -l
64

$ sudo dmesg | grep "hqd slot" | wc -l
32  # NOCPSCH: 只能分配32个HQD
0   # CPSCH: 不显示固定HQD分配

# 观察: 64个软件Queue，但只有32个HQD可用
```

---

## 🔍 结果分析

### 检查软件队列创建

```bash
# 查看所有CREATE_QUEUE事件
sudo dmesg | grep CREATE_QUEUE | tail -50

# 统计Queue数量
sudo dmesg | grep -c CREATE_QUEUE

# 查看Queue ID和doorbell地址
sudo dmesg | grep CREATE_QUEUE | awk '{print $NF}'
```

### 检查硬件HQD分配（NOCPSCH模式）

```bash
# 查看HQD分配
sudo dmesg | grep "hqd slot" | tail -50

# 统计HQD使用情况
sudo dmesg | grep "hqd slot" | awk '{print "Pipe "$5", Queue "$7}' | sort | uniq -c

# 示例输出:
#   2 Pipe 0, Queue 0
#   2 Pipe 0, Queue 1
#   2 Pipe 1, Queue 0
#   2 Pipe 1, Queue 1
#   ... (平均分布)
```

### 检查CPSCH模式行为

```bash
# 查看map_queues_cpsch调用
sudo dmesg | grep "map_queues_cpsch"

# 查看runlist操作
sudo dmesg | grep "runlist"

# 注意: CPSCH模式下，所有队列可能显示pipe=0, queue=0
# 这是正常的，实际HQD由MEC Firmware动态分配
```

---

## 📝 程序参数说明

### test_multiple_streams

```
用法: ./test_multiple_streams <num_streams> [options]

参数:
  <num_streams>    要创建的stream数量（1-1024）
  
选项:
  -w, --wait       创建后等待检查（默认5秒）
  -t, --time <sec> 指定等待时间（秒）

示例:
  ./test_multiple_streams 16        # 测试16个streams
  ./test_multiple_streams 32 -w     # 测试32个streams并等待
  ./test_multiple_streams 64 -t 10  # 测试64个streams并等待10秒
```

### run_tests.sh

```
用法: ./run_tests.sh [options] [num_streams]

选项:
  -h, --help       显示帮助信息
  -c, --compile    仅编译测试程序
  -d, --debug      启用KFD debug日志
  -a, --all        运行所有对比测试（16, 32, 64）
  
参数:
  num_streams      要测试的stream数量（默认运行全部）

示例:
  ./run_tests.sh               # 运行全部对比测试
  ./run_tests.sh 16            # 仅测试16个streams
  ./run_tests.sh -d 32         # 启用debug并测试32个streams
  ./run_tests.sh --compile     # 仅编译
```

---

## 📁 文件结构

```
test_queue_limits/
├── README.md                      # 本文档
├── test_multiple_streams.cpp      # 测试程序源码
├── run_tests.sh                   # 自动化测试脚本
└── logs/                          # 测试日志目录（自动创建）
    ├── test_16_streams_*.log      # 16 streams测试日志
    ├── test_32_streams_*.log      # 32 streams测试日志
    ├── test_64_streams_*.log      # 64 streams测试日志
    ├── *_dmesg_*.log              # dmesg日志
    └── comparison_report_*.txt    # 对比报告
```

---

## 🔧 故障排除

### 问题1: 编译失败

```bash
# 检查HIP环境
which hipcc
hipcc --version

# 检查GPU设备
rocm-smi

# 手动编译查看详细错误
hipcc -v -o test_multiple_streams test_multiple_streams.cpp
```

### 问题2: 看不到dmesg日志

```bash
# 检查权限
sudo dmesg | tail

# 启用KFD debug日志
sudo bash ../scripts/enable_kfd_debug.sh

# 检查模块是否加载
lsmod | grep amdgpu
lsmod | grep amdkfd

# 检查日志级别
cat /proc/sys/kernel/printk
```

### 问题3: CPSCH vs NOCPSCH模式

```bash
# 查看调度模式
sudo dmesg | grep -i "scheduling policy"

# 查看是否使用MES
sudo dmesg | grep -i "enable_mes"

# MI308X通常使用CPSCH模式
# 在CPSCH模式下，不会看到"hqd slot"日志
# 而是看到"map_queues_cpsch"日志
```

---

## 📚 相关文档

- `../SOFTWARE_HARDWARE_QUEUE_LIMITS.md` - 详细的队列限制文档
- `../multiple_doorbellQueue/SOFTWARE_VS_HARDWARE_QUEUES.md` - 软件vs硬件队列概念
- `../multiple_doorbellQueue/DIRECTION1_ANALYSIS.md` - CPSCH模式验证报告
- `../STREAM_PRIORITY_AND_QUEUE_MAPPING.md` - Stream到Queue的映射关系

---

## ✅ 验收标准

测试通过标准：

1. **编译成功** ✓
   - 程序正常编译，无错误

2. **Stream创建成功** ✓
   - 能够创建指定数量的streams
   - 每个stream创建独立的AQL Queue

3. **日志输出正确** ✓
   - dmesg中能看到CREATE_QUEUE事件
   - Queue数量与创建的stream数量一致

4. **HQD行为符合预期** ✓
   - ≤32 streams: 硬件资源充足
   - >32 streams: 需要HQD复用

5. **Kernel执行正常** ✓
   - 所有streams上的kernel都能正常执行
   - 无错误或崩溃

---

**创建时间**: 2026-01-30  
**测试环境**: MI308X GPU, CPSCH调度模式  
**预计测试时间**: 5-10分钟（全部测试）  
**前置要求**: ROCm环境, sudo权限（可选，用于查看dmesg）
