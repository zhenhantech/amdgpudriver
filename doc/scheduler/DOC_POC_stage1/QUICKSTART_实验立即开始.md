# 立即开始：Queue ID 实验 (5 分钟快速版)

**时间**: 5 分钟  
**目标**: 快速验证是否能看到模型的 Queue ID  
**环境**: zhenaiter Docker 容器

---

## 🚀 一行命令开始

```bash
docker exec -it zhenaiter bash -c "
export MAMBA_EXE='/root/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/root/micromamba'
eval \"\$(/root/.local/bin/micromamba shell hook --shell=bash)\"
micromamba activate flashinfer-rocm

# 设置 ROCm 库路径 (修复 librocprofiler-register.so.0 错误)
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:\$LD_LIBRARY_PATH

cd /data/dockercode/gpreempt_test

echo '=== 运行前的 MQD ==='
sudo cat /sys/kernel/debug/kfd/mqds | grep 'Queue ID' | wc -l

echo ''
echo '=== 启动测试 kernel (后台) ==='
HIP_DEVICE=0 ./test_hip_preempt 50000 10000 0 &
PID=\$!
echo \"PID: \$PID\"

sleep 3

echo ''
echo '=== 运行中的 MQD (查找 PID \$PID) ==='
sudo cat /sys/kernel/debug/kfd/mqds | grep -B 2 -A 5 \"pid \$PID\"

echo ''
echo '=== Queue ID 列表 ==='
sudo cat /sys/kernel/debug/kfd/mqds | grep -A 1 \"pid \$PID\" | grep 'Queue ID'

echo ''
echo '✅ 测试完成！'
echo ''
echo '如果看到了 Queue ID，说明方法可行！'
echo '下一步：运行完整实验 (EXP_Design_01)'
"
```

---

## 📋 预期输出

### 成功的情况

```
=== 运行前的 MQD ===
0

=== 启动测试 kernel (后台) ===
PID: 12345

=== 运行中的 MQD (查找 PID 12345) ===
Compute queue on device 0001:01:00.0
    Queue ID: 0 (0x0)
    Address: 0x7f8c00000000
    Process: pid 12345 pasid 0x8001
    is active: yes
    priority: 7
    queue count: 1

Compute queue on device 0001:01:00.0
    Queue ID: 1 (0x1)
    Address: 0x7f8c10000000
    Process: pid 12345 pasid 0x8001
    is active: yes
    priority: 7
    queue count: 2

=== Queue ID 列表 ===
    Queue ID: 0 (0x0)
    Queue ID: 1 (0x1)

✅ 测试完成！

如果看到了 Queue ID，说明方法可行！
下一步：运行完整实验 (EXP_Design_01)
```

**解读**:
- ✅ 这个程序使用了 Queue 0 和 Queue 1
- ✅ 下一步：多次运行，看是否一致

---

### 失败的情况

```
=== 运行前的 MQD ===
0

=== 启动测试 kernel (后台) ===
PID: 12345

=== 运行中的 MQD (查找 PID 12345) ===
(没有输出)

✅ 测试完成！
```

**可能原因**:
1. Kernel 运行太快，已经结束
   - 解决：增加迭代次数
2. 权限问题
   - 解决：使用 sudo
3. MQD debugfs 不存在
   - 解决：检查 `/sys/kernel/debug/kfd/` 目录

---

## 🛠️ 分步执行 (如果一行命令失败)

### Step 1: 进入容器

```bash
docker exec -it zhenaiter /bin/bash
```

### Step 2: 激活环境

```bash
export MAMBA_EXE='/root/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/root/micromamba'
eval "$(/root/.local/bin/micromamba shell hook --shell=bash)"
micromamba activate flashinfer-rocm

# 设置 ROCm 库路径
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH
```

### Step 3: 进入测试目录

```bash
cd /data/dockercode/gpreempt_test
```

### Step 4: 查看当前 MQD

```bash
sudo cat /sys/kernel/debug/kfd/mqds | grep "Queue ID"
```

应该看到 0 个或很少的队列（如果没有其他程序运行）

### Step 5: 启动测试程序

```bash
HIP_DEVICE=0 ./test_hip_preempt 50000 10000 0 &
PID=$!
echo "PID: $PID"
```

### Step 6: 等待程序启动

```bash
sleep 3
```

### Step 7: 查看该进程的 Queue

```bash
sudo cat /sys/kernel/debug/kfd/mqds | grep -B 2 -A 5 "pid $PID"
```

应该看到该 PID 对应的队列信息！

### Step 8: 提取 Queue ID

```bash
sudo cat /sys/kernel/debug/kfd/mqds | grep -A 1 "pid $PID" | grep "Queue ID"
```

输出类似：
```
    Queue ID: 0 (0x0)
    Queue ID: 1 (0x1)
```

---

## ✅ 成功标志

**看到了 Queue ID** ✅

- 说明 MQD debugfs 可用
- 说明可以追踪进程的队列
- **可以进行下一步：完整实验**

---

## ❌ 如果失败

### 问题 1: 权限被拒绝

```bash
cat: /sys/kernel/debug/kfd/mqds: Permission denied
```

**解决**:
```bash
# 确保使用 sudo
sudo cat /sys/kernel/debug/kfd/mqds

# 或在容器内切换到 root
docker exec -u root -it zhenaiter /bin/bash
```

---

### 问题 2: 文件不存在

```bash
cat: /sys/kernel/debug/kfd/mqds: No such file or directory
```

**解决**:
```bash
# 检查 debugfs 是否挂载
mount | grep debugfs

# 如果未挂载
sudo mount -t debugfs none /sys/kernel/debug

# 检查 KFD debugfs
ls -la /sys/kernel/debug/kfd/
```

---

### 问题 3: 找不到 PID

```bash
# grep "pid $PID" 没有输出
```

**原因**:
- Kernel 运行太快，已经结束

**解决**:
```bash
# 增加迭代次数，让程序运行更久
HIP_DEVICE=0 ./test_hip_preempt 500000 50000 0 &
```

---

### 问题 4: 缺少 ROCm 库

```bash
./test_hip_preempt: error while loading shared libraries: librocprofiler-register.so.0: cannot open shared object file: No such file or directory
```

**解决方案 1: 设置 LD_LIBRARY_PATH**

```bash
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH

# 验证库是否存在
ls -la /opt/rocm/lib*/librocprofiler-register.so*

# 重新运行测试
HIP_DEVICE=0 ./test_hip_preempt 50000 10000 0 &
```

**解决方案 2: 使用简单的 Python + PyTorch 测试**

如果 HIP 测试程序有问题，可以用 PyTorch 代替：

```bash
# 创建简单的 PyTorch 测试
cat > /tmp/quick_torch_test.py << 'EOF'
import torch
import time
import os

print(f"PID: {os.getpid()}")
print("Creating tensors on GPU...")

# 创建一些 GPU 操作
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()

print("Running computation for 30 seconds...")
start = time.time()
iteration = 0

while time.time() - start < 30:
    z = torch.mm(x, y)
    iteration += 1
    if iteration % 100 == 0:
        print(f"  Iteration {iteration}, elapsed: {time.time()-start:.1f}s")
    time.sleep(0.01)

print(f"Done! Total iterations: {iteration}")
EOF

# 运行 PyTorch 测试
python3 /tmp/quick_torch_test.py &
PID=$!
echo "PID: $PID"
sleep 3

# 查看该进程的 Queue
sudo cat /sys/kernel/debug/kfd/mqds | grep -B 2 -A 5 "pid $PID"
```

**解决方案 3: 使用 HIP 的简化版本**

```bash
# 创建一个不依赖 rocprofiler 的简单 HIP 程序
cat > /tmp/simple_hip_test.cpp << 'EOF'
#include <hip/hip_runtime.h>
#include <iostream>
#include <unistd.h>

__global__ void simple_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    std::cout << "PID: " << getpid() << std::endl;
    
    const int N = 1000000;
    float *d_data;
    
    hipMalloc(&d_data, N * sizeof(float));
    
    std::cout << "Running kernel for 30 seconds..." << std::endl;
    
    for (int i = 0; i < 3000; i++) {
        simple_kernel<<<100, 256>>>(d_data, N);
        hipDeviceSynchronize();
        usleep(10000);  // 10ms
    }
    
    hipFree(d_data);
    std::cout << "Done!" << std::endl;
    
    return 0;
}
EOF

# 编译（不链接 rocprofiler）
hipcc /tmp/simple_hip_test.cpp -o /tmp/simple_hip_test

# 运行
/tmp/simple_hip_test &
PID=$!
echo "PID: $PID"
sleep 3
sudo cat /sys/kernel/debug/kfd/mqds | grep -B 2 -A 5 "pid $PID"
```

---

## ➡️ 成功后的下一步

### 如果成功看到 Queue ID：

1. ✅ **多次运行测试**，看 Queue ID 是否一致
   ```bash
   # 运行 3 次
   for i in {1..3}; do
       echo "=== Run $i ==="
       HIP_DEVICE=0 ./test_hip_preempt 50000 10000 0 &
       PID=$!
       sleep 3
       sudo cat /sys/kernel/debug/kfd/mqds | grep -A 1 "pid $PID" | grep "Queue ID"
       wait $PID
       sleep 3
   done
   ```

2. ✅ 如果 Queue ID 一致 → **极好！**
   - 阅读 EXP_Design_01 的"场景 A"部分
   - 可以硬编码 Queue ID
   - POC Stage 1 只需 3-5 天

3. ⚠️ 如果 Queue ID 不一致 → **仍然可行**
   - 阅读 EXP_Design_01 的"场景 B"部分
   - 需要动态发现机制
   - POC Stage 1 需要 7-10 天

---

## 📊 记录你的结果

```bash
# 创建结果文件
cat > my_quick_test_result.txt << EOF
Quick Test Result
=================
Date: $(date)
Docker: zhenaiter

Test 1:
PID: ___
Queue IDs: ___

Test 2:
PID: ___
Queue IDs: ___

Test 3:
PID: ___
Queue IDs: ___

一致性: Yes / No

下一步策略: 
- [ ] 硬编码 (如果一致)
- [ ] 动态发现 (如果不一致)
EOF
```

---

## 🎯 这 5 分钟测试的价值

**投入**: 5 分钟  
**收获**: 
- ✅ 验证 MQD debugfs 可用
- ✅ 验证可以追踪进程队列
- ✅ 初步了解 Queue ID 模式
- ✅ 决定是否进行完整实验

**如果失败**: 
- 可以提前发现环境问题
- 避免浪费时间在完整实验上

---

**立即执行！** 🚀

复制上面的"一行命令"到终端即可开始！
