# Kernel 提交流程验证指南

**目的**: 通过实际测试验证文档描述的 Kernel 提交流程的正确性  
**测试程序**: `test_kernel_trace.cpp`  
**验证脚本**: `verify_kernel_flow.sh`

---

## 📋 快速开始

### 1. 编译测试程序

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/

# 编译测试程序
hipcc -o test_kernel_trace test_kernel_trace.cpp
```

### 2. 基础运行

```bash
# 直接运行测试程序
./test_kernel_trace

# 预期输出：
# - GPU 信息
# - MES/CPSCH 模式
# - Kernel 启动配置
# - 执行时间
# - 验证结果 ✅
```

### 3. 完整验证

```bash
# 运行完整验证脚本（部分功能需要 root）
sudo ./verify_kernel_flow.sh

# 或者非 root 运行（跳过 ftrace）
./verify_kernel_flow.sh
```

---

## 🔍 验证方法详解

### 方法 1: 基础程序输出验证

**验证点**：
- [x] GPU 设备信息
- [x] MES/CPSCH 调度器模式
- [x] Kernel 正确执行
- [x] 结果计算正确

**运行**：
```bash
./test_kernel_trace
```

**预期输出**：
```
=== Kernel Submission Flow Test ===

[1] GPU Information:
  - Device Name: AMD Radeon Graphics
  - PCI Bus ID: 1
  - PCI Device ID: 29859
  - Compute Units: 60

[2] Scheduler Mode:
  - MES enabled: 0    # 0=CPSCH, 1=MES

[6] Launching Kernel:
  - Kernel: vectorAdd
  - Flow: hipLaunchKernel -> HIP Runtime -> HSA Runtime -> KFD -> MES/CPSCH
  - Kernel execution time: 245 us

[8] Verification:
  - ✅ All results correct!
```

---

### 方法 2: ROCm Profiler 追踪 (推荐)

**验证文档章节**：
- ✅ KERNEL_TRACE_01 - HIP API 调用
- ✅ KERNEL_TRACE_02 - HSA API 调用
- ✅ Kernel 执行时间

**使用 rocprofv3** (推荐):
```bash
# 完整追踪
rocprofv3 \
    --hip-api \
    --hsa-api \
    --kernel-trace \
    --output-file trace_output/rocprof.csv \
    ./test_kernel_trace

# 分析结果
cat trace_output/rocprof.csv | grep -i "hipLaunchKernel\|hsa_queue\|hsa_signal"
```

**使用 rocprof** (旧版本):
```bash
rocprof \
    --hip-trace \
    --hsa-trace \
    --timestamp on \
    -o trace_output/rocprof.csv \
    ./test_kernel_trace

# 分析结果
cat trace_output/rocprof.csv
```

**验证点**：
1. **HIP API 层**：
   - `hipLaunchKernel` 被调用
   - `hipMemcpy` 用于数据传输
   - `hipDeviceSynchronize` 用于同步

2. **HSA API 层**：
   - `hsa_queue_create` - 创建 AQL Queue
   - `hsa_signal_create` - 创建同步信号
   - `hsa_queue_store_write_index_relaxed` - 更新写指针

---

### 方法 3: ftrace 内核函数追踪 (需要 root)

**验证文档章节**：
- ✅ KERNEL_TRACE_03 - KFD ioctl 调用
- ✅ Queue 创建流程

**设置 ftrace**：
```bash
# 需要 root 权限
sudo su

# 清理之前的追踪
echo 0 > /sys/kernel/debug/tracing/tracing_on
echo > /sys/kernel/debug/tracing/trace

# 设置追踪的函数
echo 'kfd_ioctl' > /sys/kernel/debug/tracing/set_ftrace_filter
echo 'pqm_create_queue' >> /sys/kernel/debug/tracing/set_ftrace_filter
echo 'create_queue' >> /sys/kernel/debug/tracing/set_ftrace_filter
echo 'create_queue_cpsch' >> /sys/kernel/debug/tracing/set_ftrace_filter
echo 'create_queue_mes' >> /sys/kernel/debug/tracing/set_ftrace_filter

# 启用追踪
echo function > /sys/kernel/debug/tracing/current_tracer
echo 1 > /sys/kernel/debug/tracing/tracing_on

# 运行测试程序
./test_kernel_trace

# 停止追踪
echo 0 > /sys/kernel/debug/tracing/tracing_on

# 查看结果
cat /sys/kernel/debug/tracing/trace

# 清理
echo > /sys/kernel/debug/tracing/trace
echo nop > /sys/kernel/debug/tracing/current_tracer
```

**验证点**：
1. **kfd_ioctl** - 应该看到多次调用（CREATE_QUEUE, MAP_MEMORY 等）
2. **pqm_create_queue** - Process Queue Manager 创建队列
3. **create_queue_cpsch** 或 **create_queue_mes** - 根据调度器模式

**预期输出示例**：
```
# tracer: function
#
     test_kernel_tr-12345 [001] .... 123.456789: kfd_ioctl <-do_vfs_ioctl
     test_kernel_tr-12345 [001] .... 123.456790: pqm_create_queue <-kfd_ioctl_create_queue
     test_kernel_tr-12345 [001] .... 123.456791: create_queue_cpsch <-pqm_create_queue
```

---

### 方法 4: strace 系统调用追踪

**验证文档章节**：
- ✅ `/dev/kfd` 打开
- ✅ `ioctl` 调用
- ✅ `mmap` doorbell 映射

**运行**：
```bash
strace -e trace=open,openat,ioctl,mmap,munmap -o trace_output/strace.log ./test_kernel_trace
```

**分析结果**：
```bash
# 查看 /dev/kfd 打开
grep "/dev/kfd" trace_output/strace.log

# 查看 ioctl 调用
grep "ioctl" trace_output/strace.log | grep -v "TCGETS"

# 查看 mmap (doorbell 映射)
grep "mmap" trace_output/strace.log
```

**验证点**：
1. **打开 /dev/kfd**：
   ```
   openat(AT_FDCWD, "/dev/kfd", O_RDWR|O_CLOEXEC) = 3
   ```

2. **CREATE_QUEUE ioctl**：
   ```
   ioctl(3, AMDKFD_IOC_CREATE_QUEUE, ...) = 0
   ```

3. **Doorbell mmap**：
   ```
   mmap(NULL, 8192, PROT_READ|PROT_WRITE, MAP_SHARED, 3, 0x...) = 0x7f...
   ```

---

### 方法 5: dmesg 内核日志

**验证点**：
- ✅ GPU 初始化
- ✅ MES/CPSCH 模式
- ✅ Queue 创建

**运行**：
```bash
# 清理旧日志
sudo dmesg -c > /dev/null

# 运行测试
./test_kernel_trace

# 查看新日志
dmesg | grep -i "amdgpu\|kfd\|mes"
```

**查找关键信息**：
```bash
# 1. GPU IP 版本
dmesg | grep -i "ip.*version"

# 2. MES 状态
dmesg | grep -i "mes"

# 3. Queue 创建
dmesg | grep -i "queue"
```

---

### 方法 6: /proc 文件系统检查

**验证 Doorbell 映射**：

```bash
# 找到测试程序的 PID
PID=$(pgrep -f test_kernel_trace)

# 查看内存映射（需要程序运行中）
cat /proc/$PID/maps | grep -E "kfd|doorbell"

# 查看打开的文件
lsof -p $PID | grep kfd
```

**验证 /dev/kfd 使用**：
```bash
# 查看所有使用 /dev/kfd 的进程
lsof /dev/kfd

# 查看 /dev/kfd 设备信息
ls -l /dev/kfd
```

---

## 📊 完整验证流程

### 步骤 1: 确认系统配置

```bash
# 1. 检查 GPU
rocminfo | grep "Name:"

# 2. 检查调度器模式
cat /sys/module/amdgpu/parameters/mes
# 输出: 0 = CPSCH, 1 = MES

# 3. 检查 ROCm 版本
hipcc --version
```

### 步骤 2: 编译和基础运行

```bash
# 编译
hipcc -o test_kernel_trace test_kernel_trace.cpp

# 运行
./test_kernel_trace
```

**验证**: 程序应该输出 ✅ All results correct!

### 步骤 3: ROCprofiler 追踪

```bash
# 使用 rocprofv3（推荐）
rocprofv3 --hip-api --hsa-api --kernel-trace \
    --output-file trace_rocprof.csv \
    ./test_kernel_trace

# 或使用 rocprof
rocprof --hip-trace --hsa-trace \
    -o trace_rocprof.csv \
    ./test_kernel_trace

# 分析追踪
cat trace_rocprof.csv | grep "hipLaunchKernel"
cat trace_rocprof.csv | grep "hsa_queue"
```

**验证文档对应关系**：
- `hipLaunchKernel` → **KERNEL_TRACE_01** 第 3 节
- `hsa_queue_create` → **KERNEL_TRACE_02** 第 2 节
- `hsa_signal_store_relaxed` → **KERNEL_TRACE_02** 第 4 节

### 步骤 4: ftrace 追踪 (可选，需要 root)

```bash
sudo ./verify_kernel_flow.sh
```

或手动执行：
```bash
# 参考"方法 3: ftrace 内核函数追踪"部分
```

### 步骤 5: 对比文档验证

| 文档章节 | 验证方法 | 关键观察点 |
|---------|---------|-----------|
| **KERNEL_TRACE_01** | rocprof | `hipLaunchKernel` 调用 |
| **KERNEL_TRACE_02** | rocprof | `hsa_queue_create`, doorbell 写入 |
| **KERNEL_TRACE_03** | ftrace | `kfd_ioctl`, `create_queue` |
| **KERNEL_TRACE_04** | dmesg | MES 启用状态, queue 注册 |

---

## 🎯 关键验证点清单

### ✅ Application Layer (KERNEL_TRACE_01)

- [ ] `hipLaunchKernel` 被调用
- [ ] HIP Runtime 处理启动请求
- [ ] Stream 管理正常

**验证命令**：
```bash
rocprofv3 --hip-api ./test_kernel_trace | grep hipLaunch
```

### ✅ HSA Runtime Layer (KERNEL_TRACE_02)

- [ ] `/dev/kfd` 被打开
- [ ] AQL Queue 创建成功
- [ ] Doorbell 映射到用户空间
- [ ] AQL Packet 写入 Queue
- [ ] Doorbell 更新触发硬件

**验证命令**：
```bash
# 检查 /dev/kfd
strace -e openat ./test_kernel_trace 2>&1 | grep kfd

# 检查 doorbell mmap
strace -e mmap ./test_kernel_trace 2>&1 | grep -A2 kfd
```

### ✅ KFD Driver Layer (KERNEL_TRACE_03)

- [ ] `kfd_ioctl` 处理 CREATE_QUEUE
- [ ] Queue properties 设置
- [ ] Device Queue Manager 工作
- [ ] 选择正确的调度器（MES 或 CPSCH）

**验证命令**：
```bash
# ftrace (需要 root)
sudo ./verify_kernel_flow.sh
```

### ✅ MES/Hardware Layer (KERNEL_TRACE_04)

- [ ] MES 模式：`add_hw_queue` 调用
- [ ] CPSCH 模式：软件队列管理
- [ ] Doorbell 传递到硬件

**验证命令**：
```bash
# 检查模式
cat /sys/module/amdgpu/parameters/mes

# 检查 dmesg
dmesg | grep -i "mes\|cpsch"
```

---

## 🔧 故障排查

### 问题 1: 程序编译失败

**错误**：`hipcc: command not found`

**解决**：
```bash
# 检查 ROCm 安装
which hipcc

# 如果未安装，设置环境变量
export PATH=/opt/rocm/bin:$PATH
```

### 问题 2: rocprofv3 不可用

**解决**：
```bash
# 使用旧版 rocprof
rocprof --hip-trace --hsa-trace ./test_kernel_trace

# 或检查 ROCm 版本
rocminfo | grep "Runtime Version"
```

### 问题 3: ftrace 权限不足

**错误**：`Permission denied`

**解决**：
```bash
# 使用 root 权限
sudo ./verify_kernel_flow.sh

# 或手动切换到 root
sudo su
```

### 问题 4: /dev/kfd 不存在

**错误**：`/dev/kfd: No such file or directory`

**解决**：
```bash
# 检查 amdgpu 驱动加载
lsmod | grep amdgpu

# 重新加载驱动
sudo modprobe amdgpu

# 检查设备文件
ls -l /dev/kfd
```

---

## 📈 预期结果示例

### MES 模式（MI300A/X, MI250X）

```
[2] Scheduler Mode:
  - MES enabled: 1

dmesg 输出:
[  123.456] [drm] MES enabled
[  123.457] amdgpu: MES scheduler registered

ftrace 输出:
create_queue_mes <-pqm_create_queue
```

### CPSCH 模式（MI308X, MI100）

```
[2] Scheduler Mode:
  - MES enabled: 0

dmesg 输出:
[  123.456] amdgpu: CPSCH mode enabled
[  123.457] kfd: Using CPSCH scheduler

ftrace 输出:
create_queue_cpsch <-pqm_create_queue
```

---

## 📚 参考文档对应关系

| 测试观察 | 对应文档 | 章节 |
|---------|---------|------|
| `hipLaunchKernel` 调用 | KERNEL_TRACE_01 | 3.1 |
| `hsa_queue_create` | KERNEL_TRACE_02 | 2.1 |
| Doorbell 写入 | KERNEL_TRACE_02 | 4.2 |
| `kfd_ioctl` | KERNEL_TRACE_03 | 2.1 |
| `create_queue` | KERNEL_TRACE_03 | 5.1 |
| MES vs CPSCH | KERNEL_TRACE_03 | 8.2 |
| MES `add_hw_queue` | KERNEL_TRACE_04 | 2.2 |

---

## 🎉 验证完成标准

当您完成验证后，应该能够确认：

1. ✅ **程序正常运行**：测试程序输出正确结果
2. ✅ **调度器模式确认**：明确系统使用 MES 或 CPSCH
3. ✅ **API 调用链**：通过 rocprof 观察到 HIP → HSA 调用
4. ✅ **驱动交互**：通过 strace/ftrace 观察到 KFD 交互
5. ✅ **文档一致性**：观察到的流程与文档描述一致

**恭喜！您已经成功验证了 Kernel 提交流程文档的正确性！** 🎊

