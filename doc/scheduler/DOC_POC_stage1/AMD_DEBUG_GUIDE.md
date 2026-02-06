# AMD/ROCm 调试日志指南

**更新**: 2026-02-05  
**用途**: 使用 `AMD_LOG_LEVEL` 查看详细的ROCm/KFD日志

---

## 🔍 问题：监控工具检测不到GPU使用

### 现象

```bash
sudo lsof /dev/kfd
# 没有任何输出
```

这说明**没有进程通过KFD使用GPU**。

### 解决方案：使用AMD调试日志

通过设置 `AMD_LOG_LEVEL=5`，可以看到详细的ROCm/HIP/KFD操作日志。

---

## 📊 AMD_LOG_LEVEL 级别

| 级别 | 说明 | 详细程度 |
|------|------|---------|
| `0` | 无日志 | - |
| `1` | 错误 | 最少 |
| `2` | 警告 | 少 |
| `3` | 信息 | 中 |
| `4` | 调试 | 多 |
| `5` | 跟踪 | **最详细** ⭐ |

**推荐使用级别5**来诊断KFD使用问题。

---

## 🚀 使用方法

### 方法1: 使用提供的脚本 ⭐⭐⭐⭐⭐（最简单）

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 运行带调试日志的GEMM测试
./test_gemm_with_debug.sh

# 或指定容器和时长
./test_gemm_with_debug.sh zhen_vllm_dsv3 60  # 60秒测试
```

**脚本会**:
- ✅ 自动设置 `AMD_LOG_LEVEL=5`
- ✅ 运行GEMM测试
- ✅ 显示所有AMD/ROCm日志
- ✅ 保存日志到文件
- ✅ 在初始化后等待10秒，让您检查 `lsof /dev/kfd`

---

### 方法2: 手动设置环境变量

#### 在Docker内

```bash
docker exec -it zhen_vllm_dsv3 bash

# 设置调试级别
export AMD_LOG_LEVEL=5

# 运行测试
cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
python3 test_simple_gemm_3min.py 2>&1 | tee /tmp/amd_debug.log
```

#### 在宿主机

```bash
# 设置环境变量
export AMD_LOG_LEVEL=5

# 运行PyTorch程序
python3 test_simple_gemm_3min.py 2>&1 | tee amd_debug.log
```

---

## 🔍 查看日志中的关键信息

### 1. KFD设备打开

**关键字**: `kfd`, `open`, `/dev/kfd`

```bash
grep -i 'kfd.*open\|open.*kfd' amd_debug.log
```

**预期输出**:
```
:3:hip_init.cpp     :361 : 14629 : hsa_init:Returned :0x0 : SUCCESS
:3:hip_device.cpp   :78  : 14629 : ihipDeviceOpen_: Opened /dev/kfd, fd=3
```

如果看到这个，说明**KFD已被打开**！

---

### 2. Queue创建

**关键字**: `queue`, `create`

```bash
grep -i 'queue.*creat\|creat.*queue' amd_debug.log
```

**预期输出**:
```
:3:hip_stream.cpp   :245 : 14629 : ihipStreamCreate: Created stream 0x7f3c00001000, priority=0
:4:kfd_queue.cpp    :123 : 14629 : CreateQueue: Queue created, id=1, type=COMPUTE
```

---

### 3. 内存分配

**关键字**: `malloc`, `alloc`, `memory`

```bash
grep -i 'malloc\|hipMalloc\|memory.*alloc' amd_debug.log
```

**预期输出**:
```
:4:hip_memory.cpp   :456 : 14629 : ihipMalloc: Allocated 33554432 bytes at 0x7f3c40000000
```

---

### 4. Kernel提交

**关键字**: `launch`, `kernel`, `dispatch`

```bash
grep -i 'launch\|kernel.*launch\|dispatch' amd_debug.log
```

**预期输出**:
```
:4:hip_module.cpp   :789 : 14629 : hipLaunchKernel: Launching kernel gemm_kernel
:4:kfd_dispatch.cpp :234 : 14629 : DispatchKernel: Submitted to queue 1
```

---

## 📋 完整示例

### 步骤1: 启动监控

**终端1** - 实时监控 `lsof`:
```bash
watch -n 1 'sudo lsof /dev/kfd 2>/dev/null || echo "无GPU进程"'
```

### 步骤2: 运行带调试的测试

**终端2** - 运行测试:
```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
./test_gemm_with_debug.sh
```

**输出示例**:
```
╔════════════════════════════════════════════════════════════════════╗
║  GEMM测试 - 带AMD/ROCm详细日志                                     ║
╚════════════════════════════════════════════════════════════════════╝

容器: zhen_vllm_dsv3
时长: 180秒

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 使用 AMD_LOG_LEVEL=5 查看详细的ROCm/KFD日志
   可以看到：
   - KFD设备打开
   - Queue创建
   - 内存分配
   - Kernel提交
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

按Enter开始测试...

日志保存到: /tmp/gemm_test_with_amd_logs_20260205_133500.log

======================================================================
🔢 GEMM测试 - 带AMD调试日志
======================================================================

━━━ 环境变量 ━━━
AMD_LOG_LEVEL: 5
HIP_VISIBLE_DEVICES: 0,1,2,3,4,5,6,7

━━━ GPU信息 ━━━
PyTorch版本: 2.9.1+rocm7.2.0.git7e1940d4
CUDA可用: True
GPU数量: 8
GPU名称: AMD Instinct MI308X
当前进程PID: 12345

━━━ 初始化GPU ━━━
注意下面的AMD日志输出，应该能看到KFD相关信息...

:3:hip_init.cpp     :361 : 12345 : hsa_init:Returned :0x0 : SUCCESS
:3:hip_device.cpp   :78  : 12345 : ihipDeviceOpen_: Opened /dev/kfd, fd=3
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    看到这个说明KFD已打开！

━━━ 创建GPU矩阵 ━━━
:4:hip_memory.cpp   :456 : 12345 : ihipMalloc: Allocated 33554432 bytes
矩阵A: torch.Size([2048, 2048]), 在GPU: True, 设备: cuda:0
矩阵B: torch.Size([2048, 2048]), 在GPU: True, 设备: cuda:0

━━━ GPU预热 ━━━
:4:kfd_queue.cpp    :123 : 12345 : CreateQueue: Queue created, id=1
:4:hip_module.cpp   :789 : 12345 : hipLaunchKernel: Launching kernel
预热完成

⚠️  等待10秒，请在另一个终端检查:
   sudo lsof /dev/kfd
   应该能看到这个进程 (PID: 12345)

  倒计时 10秒...
...
```

**此时在终端1应该能看到**:
```
COMMAND     PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
python3   12345 root    3u   CHR  235,0      0t0 1234 /dev/kfd
          ^^^^^ 
          看到了！
```

---

## ✅ 成功标志

### 1. AMD日志中有KFD

```bash
grep -i kfd /tmp/gemm_test_with_amd_logs_*.log
```

**成功**:
```
:3:hip_device.cpp   :78  : 12345 : ihipDeviceOpen_: Opened /dev/kfd, fd=3
:4:kfd_queue.cpp    :123 : 12345 : CreateQueue: Queue created, id=1
```

### 2. lsof能看到进程

```bash
sudo lsof /dev/kfd
```

**成功**:
```
python3   12345 root    3u   CHR  235,0  /dev/kfd
```

### 3. Queue监控工具能检测到

```bash
sudo ./watch_new_gpu.sh
```

**成功**:
```
✅ 检测到新的GPU进程!
  PID:    12345
  进程:   python3
```

---

## ❌ 如果仍然看不到KFD

### 检查清单

#### 1. 容器是否有KFD访问权限？

```bash
docker exec zhen_vllm_dsv3 ls -la /dev/kfd
```

**应该看到**:
```
crw-rw-rw- 1 root render 235, 0 ... /dev/kfd
```

如果看不到，需要在启动容器时添加设备映射：
```bash
docker run --device=/dev/kfd --device=/dev/dri ...
```

#### 2. ROCm是否正确安装？

```bash
docker exec zhen_vllm_dsv3 rocminfo 2>&1 | head -20
```

**应该能看到GPU信息**。

#### 3. PyTorch是否为ROCm版本？

```bash
docker exec zhen_vllm_dsv3 python3 -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print('ROCm支持:', 'rocm' in torch.__version__.lower())
"
```

**应该显示**:
```
PyTorch版本: 2.9.1+rocm7.2.0.git7e1940d4
ROCm支持: True
```

#### 4. HSA环境是否正确？

```bash
docker exec zhen_vllm_dsv3 bash -c "
export AMD_LOG_LEVEL=5
python3 -c 'import torch; torch.cuda.init()' 2>&1 | grep -i 'hsa\|kfd' | head -10
"
```

**应该看到HSA和KFD初始化日志**。

---

## 📊 日志分析快速参考

```bash
LOG_FILE="/tmp/gemm_test_with_amd_logs_*.log"

# 查看KFD相关
echo "=== KFD 日志 ==="
grep -i kfd "$LOG_FILE" | head -20

# 查看Queue相关
echo "=== Queue 日志 ==="
grep -i queue "$LOG_FILE" | head -20

# 查看内存分配
echo "=== 内存分配 ==="
grep -i 'malloc\|memory.*alloc' "$LOG_FILE" | head -20

# 查看Kernel提交
echo "=== Kernel提交 ==="
grep -i 'launch\|dispatch' "$LOG_FILE" | head -20

# 查看所有错误
echo "=== 错误信息 ==="
grep -E ':1:|ERROR|FAIL' "$LOG_FILE"

# 统计日志行数
echo "=== 日志统计 ==="
echo "总行数: $(wc -l < "$LOG_FILE")"
echo "KFD相关: $(grep -i kfd "$LOG_FILE" | wc -l)"
echo "Queue相关: $(grep -i queue "$LOG_FILE" | wc -l)"
```

---

## 💡 常见AMD日志模式

### 正常的GPU初始化序列

```
:3:hip_init.cpp     :361 : PID : hsa_init:Returned :0x0 : SUCCESS
:3:hip_device.cpp   :78  : PID : ihipDeviceOpen_: Opened /dev/kfd, fd=3
:3:hip_device.cpp   :145 : PID : ihipDeviceGet: Found 8 GPU devices
:4:kfd_queue.cpp    :123 : PID : CreateQueue: Queue created, id=1, type=COMPUTE
:4:hip_memory.cpp   :456 : PID : ihipMalloc: Allocated memory on GPU
:4:hip_module.cpp   :789 : PID : hipLaunchKernel: Launching kernel
```

### 如果没有使用KFD

```
:3:hip_init.cpp     :361 : PID : hsa_init:Returned :0x0 : SUCCESS
(没有看到 ihipDeviceOpen_ 或 /dev/kfd)
(可能使用了CPU后端或其他实现)
```

---

## 🎯 推荐工作流

### 诊断GPU使用问题

1. **设置调试日志**:
   ```bash
   export AMD_LOG_LEVEL=5
   ```

2. **运行测试并保存日志**:
   ```bash
   ./test_gemm_with_debug.sh 2>&1 | tee test.log
   ```

3. **在测试运行时检查lsof**:
   ```bash
   # 另一个终端
   watch -n 1 'sudo lsof /dev/kfd'
   ```

4. **分析日志**:
   ```bash
   # 查找KFD打开
   grep -i 'open.*kfd\|kfd.*open' test.log
   
   # 如果找不到，可能是：
   # - PyTorch使用了CPU后端
   # - 容器没有KFD访问权限
   # - ROCm环境问题
   ```

---

## 📚 相关文档

- **test_gemm_with_debug.sh** - 带AMD日志的测试脚本
- **QUICK_TEST_GUIDE.md** - 快速测试指南
- **SIMPLE_TESTS_GUIDE.md** - 简单测试使用指南

---

**维护者**: AI Assistant  
**更新**: 2026-02-05  
**测试平台**: MI308X + ROCm 7.x
