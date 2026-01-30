# 快速开始 - Stream Queue Limits 测试

**5分钟快速验证软件队列和硬件队列行为**

---

## 🚀 最简单的方式（推荐）

```bash
# 1. 进入测试目录
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/test_queue_limits

# 2. 运行所有测试（自动编译）
./run_tests.sh

# 完成！查看日志目录中的结果
ls -l logs/
```

---

## 📋 三种使用方式

### 方式1: 使用自动化脚本（最简单）

```bash
# 运行所有对比测试（16, 32, 64 streams）
./run_tests.sh

# 测试特定数量
./run_tests.sh 16   # 16个streams
./run_tests.sh 32   # 32个streams
./run_tests.sh 64   # 64个streams
```

### 方式2: 使用Makefile（方便）

```bash
# 编译
make

# 运行所有测试
make test

# 运行特定测试
make test-16    # 16个streams
make test-32    # 32个streams
make test-64    # 64个streams
```

### 方式3: 手动运行（最灵活）

```bash
# 编译
hipcc -o test_multiple_streams test_multiple_streams.cpp

# 运行
./test_multiple_streams 16        # 16 streams
./test_multiple_streams 32 -w     # 32 streams, 创建后等待5秒
./test_multiple_streams 64 -t 10  # 64 streams, 等待10秒
```

---

## 🔍 查看结果

### 实时查看（推荐）

在运行测试的同时，打开另一个终端：

```bash
# 监控dmesg
watch -n 1 'sudo dmesg | grep -E "CREATE_QUEUE|hqd slot" | tail -20'

# 或者实时跟踪
sudo dmesg -w | grep --line-buffered -E "CREATE_QUEUE|hqd slot"
```

### 测试完成后查看

```bash
# 查看最近的Queue创建事件
sudo dmesg | grep CREATE_QUEUE | tail -50

# 统计Queue数量
sudo dmesg | grep -c CREATE_QUEUE

# 查看HQD分配（NOCPSCH模式）
sudo dmesg | grep "hqd slot" | tail -50

# 查看CPSCH调度
sudo dmesg | grep "map_queues_cpsch" | tail -50
```

---

## 📊 预期输出示例

### 程序输出

```
========================================
Stream Tester Initialization
========================================
Number of Streams: 16
Process ID: 12345
========================================

[CREATE] Creating 16 streams...
[CREATE] Created 8/16 streams
[CREATE] Created 16/16 streams
[CREATE] ✓ Successfully created 16 streams in 45 ms

[LAUNCH] Launching kernels on all streams...
[LAUNCH] ✓ All kernels completed in 23 ms

[CONCURRENT] Testing concurrent kernel submission...
[CONCURRENT] ✓ Submitted 80 kernels (16 streams × 5 rounds) in 67 ms

========================================
Test Summary
========================================
Process ID:        12345
Streams Created:   16
Expected AQL Queues: 16 (1 per stream)
Expected HQD Usage:  16/32
HQD Status:        ✓ Sufficient (each queue gets dedicated HQD)
========================================
```

### dmesg输出（关键部分）

```bash
$ sudo dmesg | grep CREATE_QUEUE | tail -5
[12345.678] kfd: CREATE_QUEUE: pid=12345 queue_id=100 doorbell=0x1000
[12345.679] kfd: CREATE_QUEUE: pid=12345 queue_id=101 doorbell=0x1008
[12345.680] kfd: CREATE_QUEUE: pid=12345 queue_id=102 doorbell=0x1010
[12345.681] kfd: CREATE_QUEUE: pid=12345 queue_id=103 doorbell=0x1018
[12345.682] kfd: CREATE_QUEUE: pid=12345 queue_id=104 doorbell=0x1020

$ sudo dmesg | grep -c CREATE_QUEUE
16

# NOCPSCH模式会看到:
$ sudo dmesg | grep "hqd slot" | tail -5
[12345.678] kfd: hqd slot - pipe 0, queue 0
[12345.679] kfd: hqd slot - pipe 1, queue 0
[12345.680] kfd: hqd slot - pipe 2, queue 0
[12345.681] kfd: hqd slot - pipe 3, queue 0
[12345.682] kfd: hqd slot - pipe 0, queue 1

# CPSCH模式会看到:
$ sudo dmesg | grep "map_queues_cpsch"
# 多个map_queues_cpsch调用
```

---

## ✅ 验证要点

**16个Streams测试**:
- ✅ 看到16个CREATE_QUEUE事件
- ✅ 每个Queue有不同的doorbell地址（+8递增）
- ✅ HQD使用率: 16/32 = 50%
- ✅ 硬件资源充足

**32个Streams测试**:
- ✅ 看到32个CREATE_QUEUE事件
- ✅ HQD使用率: 32/32 = 100%
- ✅ 刚好用完所有硬件资源

**64个Streams测试**:
- ✅ 看到64个CREATE_QUEUE事件
- ⚠️ HQD需要复用（64 queues > 32 HQDs）
- ⚠️ 可能观察到性能下降

---

## 🔧 常见问题

**Q: 看不到dmesg输出？**
```bash
# 检查权限
sudo dmesg | tail

# 启用KFD debug
sudo bash ../scripts/enable_kfd_debug.sh

# 重新运行测试
./run_tests.sh -d 16
```

**Q: 编译失败？**
```bash
# 检查HIP环境
which hipcc
hipcc --version

# 检查GPU
rocm-smi

# 手动编译查看详细错误
hipcc -v -o test_multiple_streams test_multiple_streams.cpp
```

**Q: 如何区分CPSCH和NOCPSCH？**
```bash
# 查看调度模式
sudo dmesg | grep -i "scheduling policy"

# NOCPSCH: 会看到"hqd slot"日志
# CPSCH: 会看到"map_queues_cpsch"日志，所有队列pipe=0, queue=0
```

---

## 📁 生成的文件

```
test_queue_limits/
├── test_multiple_streams          # 编译后的可执行文件
└── logs/                          # 测试日志（自动创建）
    ├── test_16_streams_*.log      # 测试输出
    ├── test_16_streams_dmesg_*.log # dmesg日志
    ├── test_32_streams_*.log
    ├── test_64_streams_*.log
    └── comparison_report_*.txt    # 对比报告
```

---

## 🎯 一键测试命令

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/test_queue_limits && \
./run_tests.sh && \
echo "✓ 测试完成！查看日志:" && \
ls -lh logs/*.txt | tail -1
```

---

## 📚 更多信息

- 详细说明: 查看 `README.md`
- 理论基础: 查看 `../SOFTWARE_HARDWARE_QUEUE_LIMITS.md`
- 故障排除: 查看 `README.md` 中的"故障排除"部分

---

**祝测试顺利！如有问题，请查看README.md获取详细帮助。**
