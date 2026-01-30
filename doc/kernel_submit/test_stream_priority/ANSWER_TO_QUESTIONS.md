# 问题解答

您的两个问题：
1. run_test.sh 有很多 warning，想打开 HIP log 测试
2. 代码块中的原始文件在哪里

---

## ✅ 问题 1: 启用 HIP 详细日志

### 快速运行（推荐）

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/test_stream_priority

# 使用新的日志测试脚本
./run_test_with_log.sh
```

### 这个脚本做了什么？

1. **设置详细日志级别**:
   ```bash
   AMD_LOG_LEVEL=5              # 最详细的 HIP 日志
   HIP_TRACE_API=1              # 追踪所有 HIP API 调用
   HIP_DB=0x1                   # 启用 debug 信息
   AMD_SERIALIZE_KERNEL=0       # 不串行化（看并发行为）
   GPU_MAX_HW_QUEUES=8          # 限制队列数（便于观察）
   ```

2. **自动收集和分类日志**:
   - `test_concurrent.log` - 完整输出
   - `stream_create.txt` - Stream 创建记录
   - `queue_create.txt` - Queue 创建记录
   - `doorbell.txt` - Doorbell 信息
   - `priority.txt` - 优先级记录
   - `warnings.txt` - ⭐ **所有警告汇总**

3. **生成分析报告**:
   - `TEST_REPORT.md` - 自动生成的测试报告

### 预期输出示例

```
═══════════════════════════════════════════════════════════
Stream Priority 测试 - 启用详细日志
═══════════════════════════════════════════════════════════
ℹ️  日志目录: logs_20260129_174500

═══════════════════════════════════════════════════════════
步骤 1: 编译测试程序
═══════════════════════════════════════════════════════════
✅ 编译成功

═══════════════════════════════════════════════════════════
步骤 2: 配置日志环境
═══════════════════════════════════════════════════════════
ℹ️  HIP/HSA 日志级别:
  AMD_LOG_LEVEL        = 5 (5=最详细)
  HIP_TRACE_API        = 1 (1=启用)
  HIP_DB               = 0x1 (0x1=debug)
  AMD_SERIALIZE_KERNEL = 0 (0=不串行化)
  GPU_MAX_HW_QUEUES    = 8

═══════════════════════════════════════════════════════════
步骤 3: 运行单进程测试 (test_concurrent)
═══════════════════════════════════════════════════════════
ℹ️  运行 ./test_concurrent
ℹ️  输出保存到: logs_20260129_174500/test_concurrent.log

✅ test_concurrent 运行成功
...

═══════════════════════════════════════════════════════════
步骤 4: 分析日志
═══════════════════════════════════════════════════════════
─── 搜索 Warning/Error ───
ℹ️  找到 15 条 Warning/Error
详细信息保存在: logs_20260129_174500/warnings.txt
前 20 条:
...（显示 warnings）...
```

### 查看生成的日志

```bash
# 进入日志目录
cd logs_20260129_174500/

# 查看所有警告
cat warnings.txt

# 查看完整日志
less test_concurrent.log

# 查看 Stream 创建
cat stream_create.txt

# 查看测试报告
cat TEST_REPORT.md
```

---

## ✅ 问题 2: 代码原始文件位置

### 文档中引用的代码块

您在 `STREAM_PRIORITY_AND_QUEUE_MAPPING.md` 中看到的代码：

```cpp
// 文件: hipamd/src/hip_stream.cpp:194
hip::Stream* hStream = new hip::Stream(device, priority, flags, false, cuMask);

// 文件: rocr-runtime/core/runtime/amd_aql_queue.cpp:81
AqlQueue::AqlQueue(...) {
    ring_buf_ = nullptr;
    ...
}
```

### 实际完整路径

| 文档引用 | 实际完整路径 |
|---------|------------|
| `hipamd/src/hip_stream.cpp` | `/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/rocm-systems/projects/clr/hipamd/src/hip_stream.cpp` |
| `rocr-runtime/core/runtime/amd_aql_queue.cpp` | `/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp` |
| `rocr-runtime/core/runtime/amd_gpu_agent.cpp` | `/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp` |
| `kfd/amdkfd/kfd_mqd_manager_v11.c` | `/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_mqd_manager_v11.c` |

### 快速查看原始代码

**方法 1: 使用提供的脚本**（最方便）

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/test_stream_priority

# 查看所有关键代码
./view_source_code.sh
```

**输出示例**:
```
═══════════════════════════════════════════════════════════
1. HIP Stream 创建 (hip_stream.cpp)
═══════════════════════════════════════════════════════════
文件: rocm-systems/projects/clr/hipamd/src/hip_stream.cpp

ihipStreamCreate() - Line 188-206:
static hipError_t ihipStreamCreate(hipStream_t* stream, unsigned int flags,
                                   hip::Stream::Priority priority,
                                   const std::vector<uint32_t>& cuMask = {}) {
  if (flags != hipStreamDefault && flags != hipStreamNonBlocking) {
    return hipErrorInvalidValue;
  }
  hip::Stream* hStream = new hip::Stream(hip::getCurrentDevice(), priority, flags, false, cuMask);
  ...
}

hipStreamCreateWithPriority() - Line 299-316:
hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority) {
  ...
}

═══════════════════════════════════════════════════════════
2. AQL Queue 构造函数 (amd_aql_queue.cpp)
═══════════════════════════════════════════════════════════
...
```

**方法 2: 查看详细文档**

```bash
cat CODE_LOCATIONS.md
```

这个文档包含：
- 所有代码的完整路径
- 关键函数的行号
- 快速访问命令
- grep 搜索示例

**方法 3: 直接用 vim 打开**

```bash
# 打开 HIP Stream 创建代码
vim /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/rocm-systems/projects/clr/hipamd/src/hip_stream.cpp +188

# 打开 AQL Queue 构造代码
vim /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp +81

# 打开 MQD 优先级设置代码
vim /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_mqd_manager_v11.c +96
```

**方法 4: 使用 grep 搜索**

```bash
BASE_DIR="/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver"

# 搜索 Stream 创建
grep -rn "hipStreamCreateWithPriority" $BASE_DIR/rocm-systems/projects/clr/hipamd/src/

# 搜索 Queue 创建
grep -rn "AqlQueue::AqlQueue" $BASE_DIR/rocm-systems/projects/rocr-runtime/

# 搜索优先级设置
grep -rn "set_priority" $BASE_DIR/kfd-amdgpu-debug-20260106/amd/amdkfd/
```

---

## 🎯 总结

### 解决 warnings 问题

1. **立即运行**: `./run_test_with_log.sh`
2. **查看警告**: `cat logs_*/warnings.txt`
3. **分析日志**: 检查 `logs_*/TEST_REPORT.md`

### 查看原始代码

1. **最快方式**: `./view_source_code.sh`
2. **查看文档**: `cat CODE_LOCATIONS.md`
3. **直接打开**: `vim 完整路径 +行号`

---

## 📚 相关文件

| 文件 | 用途 |
|-----|------|
| `run_test_with_log.sh` | ⭐ 启用详细日志运行测试 |
| `view_source_code.sh` | ⭐ 查看原始代码 |
| `CODE_LOCATIONS.md` | 代码位置参考 |
| `README.md` | 完整文档 |
| `QUICKSTART.md` | 快速开始 |

---

**创建时间**: 2026-01-29  
**用途**: 回答关于日志和代码位置的问题
