# POC Stage 1 故障排除指南

**更新日期**: 2026-02-03  
**适用场景**: QUICKSTART 和完整实验中遇到的常见问题

---

## 🔧 问题 1: ROCm 库找不到

### 错误信息

```
./test_hip_preempt: error while loading shared libraries: librocprofiler-register.so.0: cannot open shared object file: No such file or directory
```

或类似的：
```
libamdhip64.so.6: cannot open shared object file
libhsa-runtime64.so.1: cannot open shared object file
```

---

### 🎯 解决方案 1: 设置 LD_LIBRARY_PATH (推荐)

```bash
# 在容器内执行
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH

# 验证库是否存在
ls -la /opt/rocm/lib/librocprofiler-register.so*
ls -la /opt/rocm/lib64/librocprofiler-register.so*

# 如果找到了，重新运行测试
HIP_DEVICE=0 ./test_hip_preempt 50000 10000 0
```

**永久设置**（在容器内）:
```bash
# 添加到 bashrc
echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

### 🎯 解决方案 2: 使用 PyTorch 测试 (最简单)

如果 HIP 测试程序有问题，改用 PyTorch：

**Step 1: 创建 PyTorch 测试脚本**

```bash
cat > /tmp/quick_queue_test.py << 'EOF'
#!/usr/bin/env python3
"""
Quick Queue ID Test - PyTorch Version
用于替代 test_hip_preempt，验证 Queue ID 可见性
"""

import torch
import time
import os
import sys

def main():
    print("="*60)
    print("Queue ID Quick Test (PyTorch Version)")
    print("="*60)
    print(f"PID: {os.getpid()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ GPU not available!")
        sys.exit(1)
    
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print("")
    
    # 创建数据
    print("Creating tensors on GPU...")
    x = torch.randn(2000, 2000, device='cuda')
    y = torch.randn(2000, 2000, device='cuda')
    
    print("Running computation for 30 seconds...")
    print("(You can check Queue ID now with: sudo cat /sys/kernel/debug/kfd/mqds)")
    print("")
    
    start_time = time.time()
    iteration = 0
    
    while time.time() - start_time < 30:
        # Matrix multiplication
        z = torch.mm(x, y)
        
        iteration += 1
        if iteration % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  Iteration {iteration}, Elapsed: {elapsed:.1f}s")
        
        time.sleep(0.01)  # 10ms between iterations
    
    print("")
    print(f"✅ Completed! Total iterations: {iteration}")
    print(f"Total time: {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    main()
EOF

chmod +x /tmp/quick_queue_test.py
```

**Step 2: 运行测试**

```bash
# 激活 PyTorch 环境（如果需要）
export MAMBA_EXE='/root/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/root/micromamba'
eval "$(/root/.local/bin/micromamba shell hook --shell=bash)"
micromamba activate flashinfer-rocm

# 运行测试（后台）
python3 /tmp/quick_queue_test.py &
PID=$!

echo "PID: $PID"
sleep 3

# 查看 Queue ID
echo ""
echo "=== Queue IDs for PID $PID ==="
sudo cat /sys/kernel/debug/kfd/mqds | grep -B 2 -A 5 "pid $PID"

echo ""
echo "=== Extracted Queue IDs ==="
sudo cat /sys/kernel/debug/kfd/mqds | grep -A 1 "pid $PID" | grep "Queue ID"
```

**预期输出**:
```
============================================================
Queue ID Quick Test (PyTorch Version)
============================================================
PID: 12345
PyTorch Version: 2.9.1+rocm6.4
CUDA Available: True
GPU Count: 8
GPU Name: AMD Instinct MI308X

Creating tensors on GPU...
Running computation for 30 seconds...
(You can check Queue ID now with: sudo cat /sys/kernel/debug/kfd/mqds)

  Iteration 50, Elapsed: 0.5s
  Iteration 100, Elapsed: 1.0s
  ...

✅ Completed! Total iterations: 3000
Total time: 30.0s
```

---

### 🎯 解决方案 3: 编译简化版 HIP 程序

如果需要纯 HIP 测试，可以编译一个不依赖额外库的版本：

```bash
cat > /tmp/minimal_hip_test.cpp << 'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <unistd.h>

__global__ void dummy_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 1.001f;
    }
}

int main() {
    printf("PID: %d\n", getpid());
    
    const int N = 10000000;
    float *d_data;
    
    hipMalloc(&d_data, N * sizeof(float));
    
    printf("Running kernel for 30 seconds...\n");
    printf("Check Queue ID with: sudo cat /sys/kernel/debug/kfd/mqds\n\n");
    
    for (int i = 0; i < 3000; i++) {
        dummy_kernel<<<1000, 256>>>(d_data, N);
        hipDeviceSynchronize();
        usleep(10000);  // 10ms
        
        if (i % 300 == 0) {
            printf("  Iteration %d\n", i);
        }
    }
    
    hipFree(d_data);
    printf("\nDone!\n");
    
    return 0;
}
EOF

# 编译
hipcc -o /tmp/minimal_hip_test /tmp/minimal_hip_test.cpp

# 运行
/tmp/minimal_hip_test &
PID=$!
echo "PID: $PID"
sleep 3
sudo cat /sys/kernel/debug/kfd/mqds | grep -B 2 -A 5 "pid $PID"
```

---

### 🎯 解决方案 4: 检查并修复 ROCm 安装

如果上述方案都失败，可能是 ROCm 安装有问题：

```bash
# 检查 ROCm 版本
/opt/rocm/bin/rocminfo | head -20

# 检查 ROCm 库
ls -la /opt/rocm/lib/ | grep -E "hip|hsa|profiler"

# 检查环境变量
echo $LD_LIBRARY_PATH
echo $PATH
echo $ROCM_PATH

# 重新设置完整的 ROCm 环境
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH

# 验证 HIP 可用
/opt/rocm/bin/hipcc --version
```

---

## 🔧 问题 2: 权限被拒绝

### 错误信息

```
cat: /sys/kernel/debug/kfd/mqds: Permission denied
```

---

### 🎯 解决方案

**方案 1: 使用 sudo**

```bash
sudo cat /sys/kernel/debug/kfd/mqds
```

**方案 2: 以 root 身份进入容器**

```bash
# 在宿主机执行
docker exec -u root -it zhenaiter /bin/bash

# 容器内不需要 sudo
cat /sys/kernel/debug/kfd/mqds
```

**方案 3: 检查 debugfs 挂载**

```bash
# 检查是否挂载
mount | grep debugfs

# 如果未挂载
sudo mount -t debugfs none /sys/kernel/debug

# 验证
ls -la /sys/kernel/debug/kfd/
```

---

## 🔧 问题 3: MQD/HQD 文件不存在

### 错误信息

```
cat: /sys/kernel/debug/kfd/mqds: No such file or directory
cat: /sys/kernel/debug/kfd/hqds: No such file or directory
```

---

### 🎯 诊断和解决

**Step 1: 检查 KFD 是否加载**

```bash
# 检查 KFD 模块
lsmod | grep amdkfd

# 检查 KFD 设备
ls -la /dev/kfd
```

**Step 2: 检查 debugfs**

```bash
# 检查 debugfs 目录
ls -la /sys/kernel/debug/

# 检查 KFD debugfs
ls -la /sys/kernel/debug/kfd/
```

如果 `/sys/kernel/debug/kfd/` 不存在：

```bash
# 可能的原因 1: debugfs 未挂载
sudo mount -t debugfs none /sys/kernel/debug

# 可能的原因 2: KFD debugfs 未启用
# 需要检查内核配置
```

**Step 3: 验证 KFD 功能**

```bash
# 检查 KFD 是否工作
python3 -c "import torch; print(torch.cuda.is_available())"

# 如果返回 True，说明 KFD 基本可用
# 但 debugfs 可能需要额外配置
```

---

## 🔧 问题 4: 找不到 Queue ID

### 症状

运行测试后，`grep "pid $PID"` 没有输出

---

### 🎯 解决方案

**原因 1: 程序运行太快**

```bash
# 增加运行时间
# PyTorch 版本
sed -i 's/while time.time() - start_time < 30:/while time.time() - start_time < 60:/' /tmp/quick_queue_test.py

# 或手动运行更久的任务
python3 -c "
import torch
import time
x = torch.randn(5000, 5000, device='cuda')
for i in range(10000):
    y = torch.mm(x, x)
    time.sleep(0.05)
"
```

**原因 2: PID 不正确**

```bash
# 确认进程还在运行
ps aux | grep python
ps aux | grep test_hip

# 手动找到正确的 PID
ps aux | grep "quick_queue_test"
# 例如看到: user  12345  ...  python3 /tmp/quick_queue_test.py

# 使用正确的 PID
sudo cat /sys/kernel/debug/kfd/mqds | grep -B 2 -A 5 "pid 12345"
```

**原因 3: 队列已经释放**

```bash
# 在程序运行的"中途"查看，不要等程序结束
python3 /tmp/quick_queue_test.py &
PID=$!
sleep 5  # 等待启动
sudo cat /sys/kernel/debug/kfd/mqds | grep -B 2 -A 5 "pid $PID"
# 不要 wait，让程序继续运行
```

---

## 🔧 问题 5: Docker 容器访问问题

### 错误信息

```
Cannot connect to the Docker daemon at unix:///var/run/docker.sock
Error response from daemon: No such container: zhenaiter
```

---

### 🎯 解决方案

**检查容器是否运行**

```bash
# 列出所有容器
docker ps -a | grep zhen

# 如果容器未运行，启动它
docker start zhenaiter

# 验证
docker ps | grep zhenaiter
```

**如果容器不存在**

```bash
# 查找类似的容器
docker ps -a

# 可能的替代容器名
docker exec -it <actual_container_name> /bin/bash
```

---

## 🔧 问题 6: 实验脚本创建失败

### 症状

使用 `cat > file.py << 'EOF'` 时出错

---

### 🎯 解决方案

**方案 1: 直接用编辑器创建**

```bash
# 使用 vim
vim /tmp/quick_queue_test.py
# 粘贴内容，保存

# 或使用 nano
nano /tmp/quick_queue_test.py
```

**方案 2: 从宿主机复制**

```bash
# 在宿主机创建文件
cat > /tmp/quick_queue_test.py << 'EOF'
# (内容)
EOF

# 复制到容器
docker cp /tmp/quick_queue_test.py zhenaiter:/tmp/
```

**方案 3: 下载预制脚本**

```bash
# 如果有 git 仓库
cd /data/dockercode
git pull  # 获取最新的测试脚本
```

---

## 🔧 问题 7: PyTorch 无法使用 GPU

### 错误信息

```python
torch.cuda.is_available() = False
```

或

```
RuntimeError: No HIP GPUs are available
```

---

### 🎯 解决方案

**Step 1: 检查基础环境**

```bash
# 检查 GPU 设备
ls -la /dev/kfd
ls -la /dev/dri/

# 检查 ROCm
rocm-smi

# 检查 HIP
hipconfig
```

**Step 2: 检查 PyTorch 安装**

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"ROCm version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
```

**Step 3: 重新激活环境**

```bash
# 确保在正确的 conda 环境
export MAMBA_EXE='/root/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/root/micromamba'
eval "$(/root/.local/bin/micromamba shell hook --shell=bash)"
micromamba activate flashinfer-rocm

# 验证环境
which python
python --version
```

---

## 📋 快速检查清单

在开始实验前，运行这个检查脚本：

```bash
#!/bin/bash
# pre_experiment_check.sh

echo "╔════════════════════════════════════════════════════════╗"
echo "║  POC Stage 1 环境检查                                   ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# 1. Docker 容器
echo "1. Docker 容器检查..."
docker ps | grep zhenaiter > /dev/null && echo "   ✅ zhenaiter 容器运行中" || echo "   ❌ zhenaiter 容器未运行"
echo ""

# 2. GPU 设备
echo "2. GPU 设备检查..."
docker exec zhenaiter ls -la /dev/kfd > /dev/null 2>&1 && echo "   ✅ /dev/kfd 存在" || echo "   ❌ /dev/kfd 不存在"
echo ""

# 3. debugfs
echo "3. debugfs 检查..."
docker exec zhenaiter sudo ls -la /sys/kernel/debug/kfd/mqds > /dev/null 2>&1 && echo "   ✅ mqds 可访问" || echo "   ❌ mqds 不可访问"
docker exec zhenaiter sudo ls -la /sys/kernel/debug/kfd/hqds > /dev/null 2>&1 && echo "   ✅ hqds 可访问" || echo "   ❌ hqds 不可访问"
echo ""

# 4. ROCm
echo "4. ROCm 检查..."
docker exec zhenaiter /opt/rocm/bin/rocminfo > /dev/null 2>&1 && echo "   ✅ rocminfo 可用" || echo "   ❌ rocminfo 不可用"
echo ""

# 5. PyTorch
echo "5. PyTorch 检查..."
docker exec zhenaiter bash -c "
export MAMBA_EXE='/root/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/root/micromamba'
eval \"\$(/root/.local/bin/micromamba shell hook --shell=bash)\"
micromamba activate flashinfer-rocm
python3 -c 'import torch; print(\"   ✅ PyTorch GPU:\" if torch.cuda.is_available() else \"   ❌ PyTorch GPU:\", torch.cuda.device_count())'
"

echo ""
echo "✅ 检查完成！"
echo ""
echo "如果有 ❌，请参考 TROUBLESHOOTING_常见问题解决.md"
```

保存为 `/tmp/pre_experiment_check.sh`，然后运行：

```bash
chmod +x /tmp/pre_experiment_check.sh
/tmp/pre_experiment_check.sh
```

---

## 🎯 推荐的故障排除流程

```
1. 运行预检查脚本
   └─> 发现问题

2. 根据问题查找对应章节
   └─> ROCm 库问题 → 解决方案 1
   └─> 权限问题 → 解决方案 2
   └─> debugfs 问题 → 解决方案 3
   └─> ...

3. 应用解决方案

4. 重新运行 QUICKSTART 测试

5. 如果成功 → 继续完整实验
   如果失败 → 使用 PyTorch 替代方案
```

---

## 💡 最佳实践

1. **优先使用 PyTorch 测试**
   - 更稳定
   - 依赖更少
   - 与实际 AI 模型更接近

2. **保存工作环境**
   ```bash
   # 记录成功的配置
   env > /tmp/working_env.txt
   ```

3. **使用脚本自动化**
   - 避免手动输入错误
   - 可重复执行

4. **及时记录问题**
   - 遇到新问题时记录在本文档
   - 方便后续排查

---

**最后更新**: 2026-02-03  
**维护者**: Zhehan

如有新问题，请添加到本文档！
