# 快速测试指南 - 确保Queue监控成功

**更新**: 2026-02-05  
**问题**: 监控工具检测不到GPU进程

---

## 🔍 问题诊断

### 问题现象

```
# Docker内测试运行
✅ GEMM测试在运行

# 但宿主机监控检测不到
❌ (无GPU进程)
   等待中... (已等待 100 秒)
```

### 可能原因

1. ✅ **PyTorch没有真正使用GPU** - 最可能
   - `torch.cuda.is_available()` 返回 True，但tensor没有真正在GPU上
   - 或者使用了ROCm但没有通过KFD

2. ❌ GPU初始化延迟 - 不太可能（已经等了100秒）

3. ❌ 监控脚本bug - 不太可能（已验证逻辑）

---

## ✅ 解决方案

### 方案1: 使用修改后的测试脚本（已更新）⭐⭐⭐⭐⭐

我已经更新了测试脚本，添加了：
- ✅ 显示容器内PID
- ✅ 检查`/dev/kfd`是否存在
- ✅ 确认tensor确实在GPU上
- ✅ 在初始化后等待5秒，给监控工具时间检测

**重新运行测试**:
```bash
# 终端1: 宿主机
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
sudo ./debug_gpu_usage.sh zhen_vllm_dsv3

# 终端2: Docker内
docker exec -it zhen_vllm_dsv3 bash
cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
./run_simple_tests.sh gemm
```

**查看新的输出**:
```
━━━ GPU信息 ━━━
  PyTorch版本:    2.9.1+rocm7.2.0.git7e1940d4
  CUDA可用:       是
  GPU数量:        8
  GPU名称:        AMD Instinct MI308X
  CUDA版本:       None
  GPU总内存:      191.98 GB
  /dev/kfd:       存在              ← 检查这个
  当前进程PID:    12345             ← 记录这个PID

━━━ GPU预热 ━━━
  运行小规模GEMM预热...
  预热矩阵A在GPU: True              ← 确认在GPU上
  预热矩阵B在GPU: True              ← 确认在GPU上
  ✅ 预热完成 (torch.Size([1024, 1024]))

━━━ 开始GEMM测试 ━━━
  创建测试矩阵...
  矩阵A大小: (2048, 2048), 内存: 16.00 MB
  矩阵A在GPU: True, 设备: cuda:0   ← 确认在GPU上
  矩阵B大小: (2048, 2048), 内存: 16.00 MB
  矩阵B在GPU: True, 设备: cuda:0   ← 确认在GPU上

  ⚠️ 重要提示: 程序正在初始化GPU，Queue监控工具应该能检测到此进程
     容器内PID: 12345
     在宿主机检查: sudo lsof /dev/kfd
  等待5秒，确保Queue监控工具能检测到...    ← 等待期间检查
```

---

### 方案2: 手动验证GPU使用

在测试运行时，**在第三个终端**手动检查：

```bash
# 终端3: 宿主机
# 1. 检查是否有GPU进程
sudo lsof /dev/kfd

# 应该看到类似:
# python3   123456 root  mem  CHR  235,0  /dev/kfd
```

如果看到输出，说明GPU正在使用。记录宿主机PID (123456)。

然后手动启动监控：
```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
sudo ./queue_monitor 123456 180 10
```

---

### 方案3: 使用调试脚本实时监控

使用新创建的`debug_gpu_usage.sh`，每3秒检查一次：

```bash
# 终端1: 宿主机 - 实时诊断
sudo ./debug_gpu_usage.sh zhen_vllm_dsv3

# 终端2: Docker内 - 运行测试
docker exec -it zhen_vllm_dsv3 bash
cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
./run_simple_tests.sh gemm
```

**输出示例**:
```
━━━ 采样 1 10:23:45 ━━━
宿主机 /dev/kfd:
  ❌ 无GPU进程

容器内Python进程:
  ❌ 无Python进程

━━━ 采样 2 10:23:48 ━━━
宿主机 /dev/kfd:
  ✅ 1 个GPU进程           ← 出现了！
    PID 123456: python3

容器内Python进程:
  ✅ 1 个Python进程
    PID 67: python3 test_simple_gemm_3min.py
```

---

## 🐛 如果仍然检测不到

### 检查清单

1. **`/dev/kfd` 在容器内可访问吗？**
   ```bash
   docker exec zhen_vllm_dsv3 ls -la /dev/kfd
   ```
   应该看到: `crw-rw-rw- 1 root root 235, 0 ... /dev/kfd`

2. **PyTorch 真的在使用GPU吗？**
   ```bash
   docker exec zhen_vllm_dsv3 python3 -c "
   import torch
   print('CUDA available:', torch.cuda.is_available())
   if torch.cuda.is_available():
       x = torch.randn(100, 100, device='cuda')
       print('Tensor on GPU:', x.is_cuda)
       print('Device:', x.device)
   "
   ```

3. **容器有正确的设备映射吗？**
   ```bash
   docker inspect zhen_vllm_dsv3 | grep -A 10 Devices
   ```
   应该看到 `/dev/kfd` 和 `/dev/dri/*`

4. **ROCm 环境变量设置了吗？**
   ```bash
   docker exec zhen_vllm_dsv3 env | grep -E 'ROCM|HIP|HSA'
   ```

---

## 📊 预期的正确流程

### 时间线

```
T=0s    Docker内: 启动测试脚本
T=1s    Docker内: 显示GPU信息，PID=67
T=2s    Docker内: GPU预热（创建tensor）
T=3s    宿主机: lsof /dev/kfd 应该能看到 PID=123456
T=3s    Docker内: 等待5秒...
T=8s    宿主机: 监控工具应该已经检测到进程
T=8s    Docker内: 开始GEMM测试 (180秒)
...
T=188s  Docker内: 测试完成
T=188s  宿主机: 监控工具完成
```

### 成功的标志

**Docker内**:
```
━━━ GPU信息 ━━━
  /dev/kfd:       存在           ✅
  当前进程PID:    67

━━━ GPU预热 ━━━
  预热矩阵A在GPU: True            ✅
  预热矩阵B在GPU: True            ✅

━━━ 开始GEMM测试 ━━━
  矩阵A在GPU: True, 设备: cuda:0  ✅
  等待5秒，确保Queue监控工具能检测到...
```

**宿主机**:
```
sudo lsof /dev/kfd
COMMAND     PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
python3  123456 root  mem    CHR  235,0           /dev/kfd
                                                  ✅
```

---

## 🚀 一键测试脚本

创建一个完整的测试流程：

```bash
#!/bin/bash
# 完整测试流程

echo "=== 步骤1: 检查环境 ==="
docker exec zhen_vllm_dsv3 python3 -c "
import torch
assert torch.cuda.is_available()
print('✅ PyTorch + CUDA 可用')
"

echo ""
echo "=== 步骤2: 启动实时监控（后台） ==="
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
sudo ./debug_gpu_usage.sh zhen_vllm_dsv3 > debug.log 2>&1 &
MONITOR_PID=$!
echo "监控PID: $MONITOR_PID"

echo ""
echo "=== 步骤3: 运行测试 (30秒，而不是3分钟) ==="
docker exec zhen_vllm_dsv3 bash -c "
cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
python3 -c '
import torch
import time
print(\"测试开始...\")
A = torch.randn(2048, 2048, device=\"cuda:0\")
B = torch.randn(2048, 2048, device=\"cuda:0\")
print(f\"矩阵在GPU: {A.is_cuda}\")
for i in range(150):  # 30秒左右
    C = torch.matmul(A, B)
    torch.cuda.synchronize()
    if i % 10 == 0:
        print(f\"迭代 {i}...\")
print(\"测试完成\")
'
"

echo ""
echo "=== 步骤4: 停止监控并查看结果 ==="
kill $MONITOR_PID
cat debug.log

echo ""
echo "=== 步骤5: 检查是否检测到GPU ==="
if grep -q "个GPU进程" debug.log; then
    echo "✅ 成功: 监控工具检测到了GPU进程"
else
    echo "❌ 失败: 监控工具未检测到GPU进程"
fi
```

---

## 📝 总结

**问题**: `watch_docker_gpu.sh` 检测不到GPU进程

**解决**:
1. ✅ 使用更新后的测试脚本（已添加调试信息）
2. ✅ 使用 `debug_gpu_usage.sh` 实时监控
3. ✅ 手动检查 `sudo lsof /dev/kfd`
4. ✅ 确认 `/dev/kfd` 在容器内可访问
5. ✅ 确认tensor确实在GPU上

**下一步**: 重新运行测试，观察新的调试输出

---

**维护者**: AI Assistant  
**更新**: 2026-02-05
