# Stream Priority 测试套件

验证 AMD GPU 上每个 Stream 都有独立的 Queue (ring-buffer) 的实验程序。

## 📁 文件说明

| 文件 | 说明 |
|-----|------|
| `test_app_A.cpp` | 应用程序 A，创建 2 个 Stream (HIGH, LOW) |
| `test_app_B.cpp` | 应用程序 B，创建 2 个 Stream (HIGH, NORMAL) |
| `test_concurrent.cpp` | 单进程测试，创建 4 个 Stream，便于追踪 |
| `Makefile` | 编译脚本 |
| `run_test.sh` | 自动化测试脚本 |
| `run_test_with_log.sh` | **新增**: 启用详细 HIP/HSA 日志的测试脚本 ⭐ |
| `view_source_code.sh` | **新增**: 查看文档引用的原始代码 |
| `CODE_LOCATIONS.md` | **新增**: 代码原始文件位置参考 |
| `QUICKSTART.md` | 快速开始指南 |
| `README.md` | 本文件 |

---

## 🚀 快速开始

### 方法 1: 自动化测试（推荐）

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/test_stream_priority

# 基本测试
./run_test.sh

# 完整测试（包含 dmesg 监控，需要 root）
sudo ./run_test.sh
```

### 方法 2: 启用详细日志运行（新增！推荐用于调试）⭐

```bash
# 自动启用 HIP/HSA 详细日志并运行测试
./run_test_with_log.sh

# 这会：
# 1. 编译所有程序
# 2. 设置 AMD_LOG_LEVEL=5（最详细日志）
# 3. 运行测试并收集日志
# 4. 自动分析和分类日志
# 5. 生成测试报告

# 日志保存在 logs_YYYYMMDD_HHMMSS/ 目录
```

**日志包含**:
- `test_concurrent.log` - 完整输出
- `stream_create.txt` - Stream 创建记录
- `queue_create.txt` - Queue 创建记录
- `doorbell.txt` - Doorbell 信息
- `priority.txt` - 优先级设置
- `warnings.txt` - 所有警告和错误
- `TEST_REPORT.md` - 测试总结报告

### 方法 3: 手动编译和运行

```bash
# 编译
make all

# 运行单进程测试（4 个 Stream）
./test_concurrent

# 运行双进程测试（需要两个终端）
# 终端 1:
./test_app_A

# 终端 2:
./test_app_B
```

---

## 📖 查看源代码

想看文档中引用的原始代码？

```bash
# 查看所有关键代码
./view_source_code.sh

# 或查看代码位置文档
cat CODE_LOCATIONS.md
```

**显示内容**:
1. HIP Stream 创建 (`hip_stream.cpp`)
2. AQL Queue 构造 (`amd_aql_queue.cpp`)
3. GPU Agent Queue 创建 (`amd_gpu_agent.cpp`)
4. KFD MQD 优先级设置 (`kfd_mqd_manager_v11.c`)
5. 优先级映射表 (`kfd_mqd_manager.c`)

---

## 🔬 测试内容

### 测试 1: 单进程 4 个 Stream

**程序**: `test_concurrent`

**验证内容**:
- ✅ 4 个 Stream 有不同的地址
- ✅ 每个 Stream 有独立的优先级
- ✅ 所有 Stream 可以并发提交 kernel

**运行**:
```bash
./test_concurrent
```

**预期输出**:
```
═══════════════════════════════════════════════════════════
并发测试 - 4 个 Stream 的独立性验证
═══════════════════════════════════════════════════════════
PID: 12345

GPU Device: AMD Instinct MI300X

═══════════════════════════════════════════════════════════
阶段 1: 创建 Stream（模拟应用 A）
═══════════════════════════════════════════════════════════
✅ [应用 A] Stream-1 (HIGH):   0x7f1234567890
✅ [应用 A] Stream-2 (LOW):    0x7f1234567a00

═══════════════════════════════════════════════════════════
阶段 2: 创建 Stream（模拟应用 B）
═══════════════════════════════════════════════════════════
✅ [应用 B] Stream-3 (HIGH):   0x7f1234567b10
✅ [应用 B] Stream-4 (NORMAL): 0x7f1234567c20

═══════════════════════════════════════════════════════════
验证: 所有 Stream 地址唯一性
═══════════════════════════════════════════════════════════
✅ 所有 4 个 Stream 地址唯一 → 4 个独立的 Stream 对象

...
```

### 测试 2: 双进程独立运行

**程序**: `test_app_A` + `test_app_B`

**验证内容**:
- ✅ 不同进程的 Stream 完全独立
- ✅ 不同进程的 Queue ID 不同
- ✅ 不同进程的 doorbell 地址不同

**运行**:
```bash
# 终端 1
./test_app_A

# 终端 2（在 test_app_A 运行期间启动）
./test_app_B
```

---

## 📊 使用 rocprofv3 追踪

### 追踪 Queue 信息

```bash
rocprofv3 --hip-trace ./test_concurrent
```

**生成的文件**:
- `hip_api_trace.csv` - HIP API 调用记录
- `hip_activity_trace.csv` - GPU 活动记录

**查看 Queue 信息**:
```bash
# 查看 Stream 创建
grep -i "hipStreamCreate" hip_api_trace.csv

# 查看 Queue 信息
grep -i queue hip_activity_trace.csv | head -20
```

**预期结果**:
- 看到 4 个不同的 `hipStreamCreateWithPriority` 调用
- 看到 4 个不同的 Queue ID
- 看到每个 Stream 的 kernel 提交记录

### 使用 Perfetto 可视化

```bash
# 生成 Perfetto 格式
rocprofv3 --hip-trace --output-format perfetto ./test_concurrent

# 在浏览器中打开 https://ui.perfetto.dev/
# 加载生成的 .pftrace 文件
```

**预期观察**:
- 4 条独立的 Stream 时间线
- 每个 Stream 的 kernel 执行时间
- 高优先级 Stream 的调度优先级

---

## 🔍 使用 dmesg 监控内核消息

### 启用 KFD Debug（可选）

```bash
# 启用详细日志
sudo su
echo 0xff > /sys/module/amdkfd/parameters/debug_evictions
exit
```

### 监控 Queue 创建

```bash
# 清空 dmesg
sudo dmesg -C

# 在另一个终端启动监控
sudo dmesg -w | grep -E "create queue|doorbell|priority"

# 在原终端运行测试
./test_concurrent
```

**预期输出**:
```
[12345.678] amdkfd: create queue id=1001, priority=11, doorbell_off=0x1000
[12345.679] amdkfd: create queue id=1002, priority=1, doorbell_off=0x1008
[12345.680] amdkfd: create queue id=1003, priority=11, doorbell_off=0x1010
[12345.681] amdkfd: create queue id=1004, priority=7, doorbell_off=0x1018
```

**关键观察**:
- ✅ 4 个不同的 `queue_id`
- ✅ 4 个不同的 `doorbell_off` (doorbell 偏移)
- ✅ 每个 Queue 有自己的 `priority`

---

## 🛠️ 高级验证

### 1. 检查进程打开的文件

```bash
# 在测试程序运行期间（保持运行 10 秒）
PID=$(pgrep test_concurrent)
lsof -p $PID | grep kfd
```

**预期输出**:
```
test_conc 12345 user    3u   CHR  234,0      0t0  12345 /dev/kfd
```

### 2. 检查内存映射

```bash
# 查看 doorbell 映射
cat /proc/$PID/maps | grep doorbell
```

**预期输出**:
```
7f1234000000-7f1234001000 rw-s 00001000 00:11 12345 /dev/kfd (doorbell)
```

### 3. 检查 KFD Queue 信息（需要 debugfs）

```bash
# 挂载 debugfs（如果未挂载）
sudo mount -t debugfs none /sys/kernel/debug

# 查看所有 Queue
sudo cat /sys/kernel/debug/kfd/queues
```

**预期输出**:
```
PID 12345:
  Queue 1001: type=COMPUTE, priority=11, doorbell=0x1000
  Queue 1002: type=COMPUTE, priority=1,  doorbell=0x1008
  Queue 1003: type=COMPUTE, priority=11, doorbell=0x1010
  Queue 1004: type=COMPUTE, priority=7,  doorbell=0x1018
```

---

## 📈 性能测试（可选）

### 测试优先级调度

修改 `test_concurrent.cpp`，在并发阶段提交更多 kernel：

```cpp
// 高优先级 Stream 提交大量 kernel
for (int i = 0; i < 100; i++) {
    hipLaunchKernelGGL(dummy_kernel, dim3(256), dim3(256), 0, stream_A1, d_data[0], i);
}

// 低优先级 Stream 提交 kernel
for (int i = 0; i < 10; i++) {
    hipLaunchKernelGGL(dummy_kernel, dim3(256), dim3(256), 0, stream_A2, d_data[1], i);
}
```

**预期行为**:
- 高优先级 Stream 的 kernel 优先被调度
- 低优先级 Stream 的 kernel 可能需要等待

---

## 🔬 代码分析

### Stream 创建流程

```
hipStreamCreateWithPriority()
  ↓
ihipStreamCreate(..., priority)
  ↓
new hip::Stream(device, priority, ...)
  ↓
hip::Stream::Create()
  ↓
hsa_queue_create(...)
  ↓
GpuAgent::QueueCreate(...)
  ↓
new AqlQueue(...)              // ⭐ 分配独立的 ring buffer
  ↓
AllocRegisteredRingBuffer()    // ⭐ 独立的 ring buffer
  ↓
driver().CreateQueue(...)      // ⭐ 调用 KFD
  ↓
ioctl(AMDKFD_IOC_CREATE_QUEUE) // ⭐ 创建内核层 Queue
  ↓
kfd_ioctl_create_queue()
  ↓
pqm_create_queue()
  ↓
queue_id = new_id              // ⭐ 分配独立的 Queue ID
doorbell_off = allocate()      // ⭐ 分配独立的 doorbell
```

### 关键数据结构

```cpp
// 每个 Stream
struct hip::Stream {
    hsa_queue_t* hsa_queue_;     // 指向独立的 HSA Queue
    Priority priority_;          // 优先级
    Device* device_;
};

// 每个 HSA Queue
struct AqlQueue {
    void* ring_buf_;             // ⭐ 独立的 ring buffer
    uint64_t queue_id_;          // ⭐ 独立的 Queue ID
    void* doorbell_ptr_;         // ⭐ 独立的 doorbell MMIO 地址
    HSA_QUEUE_PRIORITY priority_; // 优先级
};

// 每个 KFD Queue
struct queue {
    unsigned int queue_id;       // ⭐ 内核层 Queue ID
    struct queue_properties {
        uint32_t priority;       // ⭐ 0-15 的优先级值
        uint64_t queue_address;  // ⭐ ring buffer 物理地址
        uint32_t doorbell_off;   // ⭐ doorbell 偏移
    };
    struct mqd {
        uint32_t cp_hqd_pipe_priority;   // ⭐ 硬件 pipe 优先级
        uint32_t cp_hqd_queue_priority;  // ⭐ Queue 优先级
    };
};
```

---

## 📊 测试结果总结

### 验证的关键点

| 验证项 | 方法 | 预期结果 |
|-------|------|---------|
| **Stream 地址唯一** | `printf("%p", stream)` | ✅ 4 个不同地址 |
| **Queue ID 唯一** | `dmesg` 或 `rocprof` | ✅ 4 个不同 ID |
| **doorbell 地址唯一** | `/proc/PID/maps` | ✅ 4 个不同偏移 |
| **优先级独立** | `hipStreamGetPriority()` | ✅ 每个 Stream 有自己的优先级 |
| **并发提交** | 测试程序运行 | ✅ 所有 Stream 可以并发提交 |

### 核心结论

```
✅ 每个 Stream 都有独立的 HSA Queue
✅ 每个 Queue 都有独立的 ring-buffer
✅ 每个 Queue 都有独立的 Queue ID
✅ 每个 Queue 都有独立的 doorbell 地址
✅ 优先级不影响 Queue 的独立性
✅ 不同进程的 Queue 完全隔离
```

---

## 🐛 故障排查

### 问题 1: 编译失败

**症状**: `hipcc: command not found`

**解决**:
```bash
# 检查 ROCm 安装
ls /opt/rocm/

# 添加到 PATH
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

### 问题 2: 运行时错误

**症状**: `hipErrorInvalidDevice`

**解决**:
```bash
# 检查 GPU 是否可见
rocminfo | grep "Name:"

# 检查权限
ls -l /dev/kfd
sudo usermod -aG render $USER
sudo usermod -aG video $USER

# 重新登录
```

### 问题 3: dmesg 无输出

**症状**: 运行 `dmesg` 没有看到 Queue 创建消息

**解决**:
```bash
# 启用 KFD debug
sudo su
echo 0xff > /sys/module/amdkfd/parameters/debug_evictions
exit

# 或者重新加载 amdgpu 模块
sudo modprobe -r amdgpu
sudo modprobe amdgpu
```

---

## 📚 相关文档

- [STREAM_PRIORITY_AND_QUEUE_MAPPING.md](../STREAM_PRIORITY_AND_QUEUE_MAPPING.md) - 理论分析
- [KERNEL_TRACE_STREAM_MANAGEMENT.md](../KERNEL_TRACE_STREAM_MANAGEMENT.md) - Stream 管理
- [KERNEL_TRACE_02_HSA_RUNTIME.md](../KERNEL_TRACE_02_HSA_RUNTIME.md) - HSA Runtime Queue 创建
- [KERNEL_TRACE_03_KFD_QUEUE.md](../KERNEL_TRACE_03_KFD_QUEUE.md) - KFD Queue 管理

---

## ⚠️ 重要提醒

**当前状态**: HSA Runtime 中优先级被写死为 NORMAL！

**影响**: 
- 测试程序可以运行，但所有 Queue 都是相同优先级
- 无法真正测试优先级调度效果
- MQD 中的 `cp_hqd_pipe_priority` 都是相同的值 (1=NORMAL)

**修复方案**: 见 [../PRIORITY_CODE_FIX_TODO.md](../PRIORITY_CODE_FIX_TODO.md)

**下一步**: 
1. 修改 `amd_aql_queue.cpp` Line 100
2. 重新编译 HSA Runtime
3. 再次运行测试验证

---

## 🎯 下一步

**在修复代码之前**: 测试可以验证 Stream 和 Queue 的独立性 ✅

**修复代码之后**: 可以进行：

1. **修改优先级**: 改变 Stream 的优先级，观察调度行为 ⭐
2. **增加 Stream 数量**: 创建更多 Stream，观察 Queue 数量
3. **性能测试**: 提交大量 kernel，测试优先级调度效果 ⭐
4. **跨进程测试**: 运行多个应用，观察 Queue 隔离

---

**创建时间**: 2026-01-28  
**更新时间**: 2026-01-29  
**目的**: 验证每个 Stream 都有独立的 Queue (ring-buffer)  
**结论**: ✅ 已验证 Stream 和 Queue 的 1:1 映射！  
**待完成**: ⚠️ 需要修复 HSA Runtime 才能真正测试优先级调度
