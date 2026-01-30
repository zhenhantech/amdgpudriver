# 日志启用指南 - HIP vs KFD

**关键结论**: 两者都**不需要重新编译**！都可以通过简单配置启用。

---

## 🎯 快速对比

| 日志类型 | 需要重新编译? | 启用方式 | 难度 | 可见内容 |
|---------|------------|---------|------|---------|
| **HIP日志** | ❌ 不需要 | 环境变量 | ⭐ 简单 | HIP API调用、Queue创建 |
| **KFD日志** | ❌ 不需要 | Dynamic Debug | ⭐⭐ 中等 | 内核层Queue分配、HQD分配 |

---

## 🚀 方式1: HIP日志（最简单）

### 启用方法（无需重新编译）

```bash
# 设置环境变量即可
export AMD_LOG_LEVEL=3        # 0=Error, 1=Warning, 2=Info, 3=Verbose, 4=Debug
export HIP_VISIBLE_DEVICES=0  # 指定使用的GPU
export HIP_DB=0x1              # 启用HIP调试输出

# 运行测试
./test_multiple_streams 16
```

### 一键测试（推荐）

```bash
# 使用我们的脚本
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/test_queue_limits

# 测试16个streams，启用HIP日志
./test_with_logs.sh 16

# 测试32个streams
./test_with_logs.sh 32
```

### 可用的环境变量

```bash
# HIP Runtime日志
export AMD_LOG_LEVEL=4        # 最详细的日志
export HSA_ENABLE_DEBUG=1     # HSA层调试

# 选择性日志
export AMD_LOG_LEVEL=3        # Verbose（推荐）
export HIP_TRACE_API=1        # 跟踪API调用
export HIP_PRINT_ENV=1        # 打印环境配置

# 性能分析
export AMD_SERIALIZE_KERNEL=3 # 串行化kernel（用于调试）
```

### 预期输出示例

```
:3:hip_stream.cpp:299: hipStreamCreateWithPriority ( 0x7ffd12345678, 0, 0 )
:3:hip_stream.cpp:188: Creating new hip::Stream
:3:hip_queue.cpp:123: Creating HSA Queue
:3:hip_queue.cpp:156: Queue created successfully
```

---

## 🔧 方式2: KFD内核日志（中等难度）

### ✅ 不需要重新编译！

**关键**: KFD驱动支持 `dynamic_debug`，可以在**运行时**启用日志。

### 前置条件检查

```bash
# 检查内核是否支持 dynamic_debug
ls /sys/kernel/debug/dynamic_debug/control

# 如果存在这个文件 → ✅ 可以启用KFD日志
# 如果不存在 → ❌ 内核未启用 CONFIG_DYNAMIC_DEBUG（需要重新编译内核）
```

### 启用方法（无需重新编译模块）

```bash
# 方式1: 使用我们的脚本（推荐）
sudo bash /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/scripts/enable_kfd_debug.sh

# 方式2: 手动启用
sudo bash -c "echo 'file kfd_device_queue_manager.c +p' > /sys/kernel/debug/dynamic_debug/control"
sudo bash -c "echo 'file kfd_chardev.c +p' > /sys/kernel/debug/dynamic_debug/control"
sudo bash -c "echo 'file kfd_process_queue_manager.c +p' > /sys/kernel/debug/dynamic_debug/control"
```

### 工作原理

```
Dynamic Debug 机制:
  ✓ 内核编译时包含 pr_debug() 调用
  ✓ 默认情况下这些调用是禁用的（零性能开销）
  ✓ 通过 /sys/kernel/debug/dynamic_debug/control 动态启用
  ✓ 无需重新编译内核或模块！
```

### 查看日志

```bash
# 实时查看
sudo dmesg -w | grep kfd

# 查看所有KFD日志
sudo dmesg | grep kfd

# 查看特定类型
sudo dmesg | grep "CREATE_QUEUE"    # Queue创建
sudo dmesg | grep "hqd slot"        # HQD分配（NOCPSCH模式）
sudo dmesg | grep "map_queues"      # CPSCH调度
```

### 预期输出示例

```bash
[12345.678] kfd kfd: ioctl_create_queue: pid=12345 queue_id=100
[12345.679] kfd kfd: allocate_hqd: hqd slot - pipe 0, queue 0
[12345.680] kfd kfd: init_mqd: configuring MQD for queue 100
```

---

## 📊 完整测试流程

### 测试脚本（已创建）

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/test_queue_limits

# 运行带完整日志的测试
./test_with_logs.sh 16    # 16 streams
./test_with_logs.sh 32    # 32 streams
./test_with_logs.sh 64    # 64 streams
```

### 脚本会自动完成

1. ✅ 启用HIP日志（环境变量）
2. ✅ 尝试启用KFD日志（dynamic_debug）
3. ✅ 运行测试程序
4. ✅ 收集所有日志
5. ✅ 生成分析报告

### 生成的日志文件

```
logs_with_debug/
├── test_16_streams_20260130_*.log      # 测试输出（含HIP日志）
├── dmesg_16_streams_20260130_*.log     # 内核日志（含KFD日志）
└── report_16_streams_20260130_*.txt    # 分析报告
```

---

## 🔍 故障排除

### 问题1: 看不到HIP日志

**解决方案**:
```bash
# 确认环境变量设置
echo $AMD_LOG_LEVEL    # 应该是 3 或 4

# 提高日志级别
export AMD_LOG_LEVEL=4

# 重新运行测试
./test_with_logs.sh 16
```

### 问题2: 看不到KFD日志

**原因A: Dynamic Debug不可用**
```bash
# 检查
ls /sys/kernel/debug/dynamic_debug/control

# 如果不存在 → 内核未启用 CONFIG_DYNAMIC_DEBUG
# 这种情况下确实需要重新编译内核（但很罕见，大多数发行版都启用了）
```

**原因B: 权限问题**
```bash
# 需要root权限
sudo bash enable_kfd_debug.sh
```

**原因C: 日志级别太低**
```bash
# 提高内核日志级别
sudo dmesg -n 8    # 8 = debug level
```

### 问题3: 大量无关日志

**解决方案**: 只启用特定文件
```bash
# 只启用queue管理相关的日志
sudo bash -c "echo 'file kfd_device_queue_manager.c func allocate_hqd +p' > /sys/kernel/debug/dynamic_debug/control"
```

---

## 📝 日志级别对照表

### HIP日志级别

| AMD_LOG_LEVEL | 级别 | 输出内容 |
|--------------|------|---------|
| 0 | Error | 仅错误 |
| 1 | Warning | 错误 + 警告 |
| 2 | Info | 错误 + 警告 + 信息 |
| 3 | Verbose | 详细信息 ⭐ 推荐 |
| 4 | Debug | 所有调试信息 |

### KFD日志控制

| 模式 | 命令 | 效果 |
|-----|------|-----|
| 启用所有 | `file kfd_*.c +p` | 所有KFD文件 |
| 启用单文件 | `file kfd_device_queue_manager.c +p` | 单个文件 |
| 启用单函数 | `func allocate_hqd +p` | 单个函数 |
| 启用单行 | `line 992 +p` | 单行代码 |
| 禁用 | `-p` 代替 `+p` | 禁用日志 |

---

## 🎓 总结

### ✅ 好消息

1. **HIP日志**: 零配置，设置环境变量即可 ⭐⭐⭐⭐⭐
2. **KFD日志**: 不需要重新编译，动态启用即可 ⭐⭐⭐⭐

### ⚠️ 注意事项

1. KFD日志需要内核启用 `CONFIG_DYNAMIC_DEBUG`（大多数发行版默认启用）
2. 日志会影响性能，测试完记得禁用
3. 大量日志可能淹没 dmesg，建议只启用需要的部分

### 🚀 推荐流程

```bash
# 1. 先用HIP日志测试（最简单）
export AMD_LOG_LEVEL=3
./test_multiple_streams 16

# 2. 如果需要更底层信息，启用KFD日志
sudo bash enable_kfd_debug.sh
./test_with_logs.sh 16

# 3. 查看日志
sudo dmesg | grep -E "kfd|CREATE_QUEUE|hqd slot"

# 4. 测试完成，清理
sudo bash disable_kfd_debug.sh
unset AMD_LOG_LEVEL
```

---

**创建时间**: 2026-01-30  
**关键优势**: 两种日志都不需要重新编译！
