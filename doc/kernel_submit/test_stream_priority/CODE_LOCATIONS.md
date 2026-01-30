# 代码原始文件位置参考

本文档列出 `STREAM_PRIORITY_AND_QUEUE_MAPPING.md` 中引用的所有代码的原始文件位置。

---

## 📁 HIP Runtime 层

### 1. Stream 创建

**文档中的引用**:
```cpp
// 文件: hipamd/src/hip_stream.cpp:194
hip::Stream* hStream = new hip::Stream(device, priority, flags, false, cuMask);
```

**实际完整路径**:
```bash
/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/rocm-systems/projects/clr/hipamd/src/hip_stream.cpp
```

**关键函数**:
- Line 188: `ihipStreamCreate()`
- Line 299: `hipStreamCreateWithPriority()`

**查看代码**:
```bash
# 查看 Stream 创建函数
sed -n '188,206p' /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/rocm-systems/projects/clr/hipamd/src/hip_stream.cpp

# 查看优先级处理
sed -n '299,316p' /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/rocm-systems/projects/clr/hipamd/src/hip_stream.cpp
```

---

## 📁 HSA Runtime 层

### 2. AQL Queue 构造函数

**文档中的引用**:
```cpp
// 文件: rocr-runtime/core/runtime/amd_aql_queue.cpp:81
AqlQueue::AqlQueue(...) {
    ring_buf_ = nullptr;
    queue_id_ = HSA_QUEUEID(-1);
    priority_ = HSA_QUEUE_PRIORITY_NORMAL;
    AllocRegisteredRingBuffer(queue_size_pkts);
    agent->driver().CreateQueue(..., priority_, ..., ring_buf_, ...);
    signal_.hardware_doorbell_ptr = queue_rsrc.Queue_DoorBell_aql;
}
```

**实际完整路径**:
```bash
/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp
```

**关键函数和行号**:
- Line 81-130: `AqlQueue::AqlQueue()` 构造函数
- Line 269-289: Queue 创建和 ioctl 调用
- Line 634-643: `AqlQueue::SetPriority()` 设置优先级

**查看代码**:
```bash
# 查看构造函数
sed -n '81,130p' /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp

# 查看 KFD 调用
sed -n '269,289p' /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp

# 查看优先级设置
sed -n '634,643p' /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp
```

### 3. GPU Agent Queue 创建

**文档中的引用**:
```cpp
// 文件: rocr-runtime/core/runtime/amd_gpu_agent.cpp:1735
hsa_status_t GpuAgent::QueueCreate(...)
```

**实际完整路径**:
```bash
/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp
```

**关键函数和行号**:
- Line 1735-1835: `GpuAgent::QueueCreate()`
- Line 777-798: `InitDma()` - 设置 Queue 优先级的 lambda

**查看代码**:
```bash
# 查看 QueueCreate
sed -n '1735,1835p' /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp

# 查看优先级 lambda
sed -n '777,798p' /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp
```

---

## 📁 KFD Driver 层

### 4. MQD 优先级设置

**文档中的引用**:
```c
// 文件: kfd/amdkfd/kfd_mqd_manager_v11.c:96
static void set_priority(struct v11_compute_mqd *m, struct queue_properties *q) {
    m->cp_hqd_pipe_priority = pipe_priority_map[q->priority];
    m->cp_hqd_queue_priority = q->priority;
}
```

**实际完整路径**:
```bash
/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_mqd_manager_v11.c
```

**关键函数和行号**:
- Line 96-100: `set_priority()` - 设置 MQD 优先级

**查看代码**:
```bash
sed -n '96,100p' /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_mqd_manager_v11.c
```

### 5. 优先级映射表

**文档中的引用**:
```c
// 文件: kfd/amdkfd/kfd_mqd_manager.c:29
int pipe_priority_map[] = {
    KFD_PIPE_PRIORITY_CS_LOW,    // 0-6
    ...
    KFD_PIPE_PRIORITY_CS_MEDIUM, // 7-10
    ...
    KFD_PIPE_PRIORITY_CS_HIGH,   // 11-15
};
```

**实际完整路径**:
```bash
/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_mqd_manager.c
```

**关键数据**:
- Line 29-47: `pipe_priority_map[]` 优先级映射数组

**查看代码**:
```bash
sed -n '29,47p' /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_mqd_manager.c
```

---

## 📝 快速访问脚本

创建一个脚本来快速查看这些关键代码：

```bash
#!/bin/bash
# view_stream_code.sh - 快速查看 Stream/Queue 相关代码

BASE_DIR="/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver"

echo "═══════════════════════════════════════════════════════════"
echo "1. HIP Stream 创建"
echo "═══════════════════════════════════════════════════════════"
sed -n '188,206p' "$BASE_DIR/rocm-systems/projects/clr/hipamd/src/hip_stream.cpp"
echo ""

echo "═══════════════════════════════════════════════════════════"
echo "2. AQL Queue 构造"
echo "═══════════════════════════════════════════════════════════"
sed -n '81,130p' "$BASE_DIR/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp"
echo ""

echo "═══════════════════════════════════════════════════════════"
echo "3. MQD 优先级设置"
echo "═══════════════════════════════════════════════════════════"
sed -n '96,100p' "$BASE_DIR/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_mqd_manager_v11.c"
echo ""

echo "═══════════════════════════════════════════════════════════"
echo "4. 优先级映射表"
echo "═══════════════════════════════════════════════════════════"
sed -n '29,47p' "$BASE_DIR/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_mqd_manager.c"
echo ""
```

---

## 🔍 使用 Grep 搜索关键代码

### 搜索 Stream 创建

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver

# 搜索 hipStreamCreateWithPriority
grep -rn "hipStreamCreateWithPriority" rocm-systems/projects/clr/hipamd/src/

# 搜索 ihipStreamCreate
grep -rn "ihipStreamCreate" rocm-systems/projects/clr/hipamd/src/hip_stream.cpp
```

### 搜索 Queue 创建

```bash
# 搜索 AqlQueue 构造函数
grep -rn "AqlQueue::AqlQueue" rocm-systems/projects/rocr-runtime/

# 搜索 QueueCreate
grep -rn "GpuAgent::QueueCreate" rocm-systems/projects/rocr-runtime/
```

### 搜索优先级处理

```bash
# 搜索 set_priority
grep -rn "set_priority" kfd-amdgpu-debug-20260106/amd/amdkfd/

# 搜索 pipe_priority_map
grep -rn "pipe_priority_map" kfd-amdgpu-debug-20260106/amd/amdkfd/
```

---

## 📚 相关文件索引

### HIP Runtime (rocm-systems/projects/clr/hipamd/)

| 文件 | 关键内容 |
|-----|---------|
| `src/hip_stream.cpp` | Stream 创建、优先级设置 |
| `include/hip/amd_detail/hip_runtime.h` | Stream API 定义 |

### HSA Runtime (rocm-systems/projects/rocr-runtime/)

| 文件 | 关键内容 |
|-----|---------|
| `runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp` | AQL Queue 实现 |
| `runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp` | GPU Agent Queue 管理 |
| `runtime/hsa-runtime/core/inc/amd_aql_queue.h` | AQL Queue 头文件 |

### KFD Driver (kfd-amdgpu-debug-20260106/amd/amdkfd/)

| 文件 | 关键内容 |
|-----|---------|
| `kfd_mqd_manager_v11.c` | MI300 系列 MQD 管理 |
| `kfd_mqd_manager_v9.c` | MI100/MI250 MQD 管理 |
| `kfd_mqd_manager.c` | 优先级映射表 |
| `kfd_chardev.c` | ioctl 处理 |
| `kfd_process_queue_manager.c` | Queue 管理 |

---

## 🎯 验证路径

验证这些文件存在：

```bash
# HIP Runtime
ls -l /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/rocm-systems/projects/clr/hipamd/src/hip_stream.cpp

# HSA Runtime
ls -l /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp
ls -l /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp

# KFD Driver
ls -l /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_mqd_manager_v11.c
ls -l /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_mqd_manager.c
```

---

**创建时间**: 2026-01-29  
**用途**: 快速定位文档中引用的代码原始位置
