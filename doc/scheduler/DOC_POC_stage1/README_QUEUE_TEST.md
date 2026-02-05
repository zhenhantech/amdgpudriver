# Queue ID 测试脚本使用说明

**目录**: `/data/dockercode/gpreempt_test/` (Docker 内部)  
**对应宿主机**: `/mnt/md0/zhehan/code/flashinfer/dockercode/gpreempt_test/`  
**更新日期**: 2026-02-03

---

## 📋 脚本列表

### 🖥️ 宿主机脚本 (推荐！)

#### 1. host_queue_test.sh ⭐⭐⭐⭐⭐ (最推荐)

**功能**: 在宿主机运行，自动启动容器测试并查看 Queue ID  
**时间**: ~35 秒  
**稳定性**: ⭐⭐⭐⭐⭐ (最稳定，解决 debugfs 访问问题)

```bash
# 在宿主机运行（不在容器内！）
cd /mnt/md0/zhehan/code/flashinfer/dockercode/gpreempt_test
./host_queue_test.sh
```

**优点**:
- ✅ 解决了容器无法访问 debugfs 的问题
- ✅ 自动启动容器内测试
- ✅ 在宿主机查看 MQD
- ✅ 自动提取 Queue ID

---

#### 2. host_queue_consistency_test.sh ⭐⭐⭐⭐⭐ (核心实验)

**功能**: 在宿主机运行 5 次测试，验证 Queue ID 一致性  
**时间**: ~5 分钟  
**重要性**: ⭐⭐⭐⭐⭐ (决定 POC 实施策略)

```bash
# 在宿主机运行
cd /mnt/md0/zhehan/code/flashinfer/dockercode/gpreempt_test
./host_queue_consistency_test.sh
```

**输出**:
- 5 次运行的 Queue IDs
- 一致性分析
- POC Stage 1 实施建议

---

#### 3. fix_debugfs.sh

**功能**: 诊断 debugfs 访问问题  
**时间**: ~1 分钟  

```bash
# 可以在容器内或宿主机运行
./fix_debugfs.sh
```

---

### 🐳 容器内脚本 (如果 debugfs 可用)

#### 4. quick_queue_test.sh

**功能**: 使用 PyTorch 进行快速 Queue ID 测试  
**时间**: ~30 秒  
**稳定性**: ⭐⭐⭐⭐⭐ (最稳定)

```bash
# 在 Docker 内部运行
cd /data/dockercode/gpreempt_test
./quick_queue_test.sh
```

**注意**: 需要容器能访问 `/sys/kernel/debug/kfd/mqds`

---

#### 5. quick_queue_test_hip.sh

**功能**: 使用 HIP 测试程序（尝试修复库路径）  
**时间**: ~10 秒  
**稳定性**: ⭐⭐⭐☆☆ (可能有库依赖问题)

```bash
# 在 Docker 内部运行
cd /data/dockercode/gpreempt_test
./quick_queue_test_hip.sh
```

---

#### 6. test_queue_consistency.sh

**功能**: 在容器内运行 5 次测试  
**时间**: ~5 分钟  

```bash
# 在 Docker 内部运行
cd /data/dockercode/gpreempt_test
./test_queue_consistency.sh
```

**注意**: 需要容器能访问 debugfs

---

## 🚀 快速开始

### ⭐ 方案 A: 使用宿主机脚本（推荐！解决 debugfs 问题）

#### 步骤 1: 在宿主机运行快速测试

```bash
# 在宿主机执行（不在容器内！）
cd /mnt/md0/zhehan/code/flashinfer/dockercode/gpreempt_test
./host_queue_test.sh
```

这个脚本会：
1. 自动启动容器内的测试程序
2. 在宿主机上查看 MQD
3. 提取 Queue ID

#### 步骤 2: 验证一致性（如果步骤 1 成功）

```bash
# 在宿主机执行
./host_queue_consistency_test.sh
```

---

### 方案 B: 容器内运行（如果 debugfs 可访问）

#### 步骤 1: 进入 Docker 容器

```bash
# 在宿主机执行
docker exec -it zhenaiter /bin/bash
```

#### 步骤 2: 进入测试目录

```bash
# 在 Docker 内部执行
cd /data/dockercode/gpreempt_test
```

#### 步骤 3: 诊断 debugfs

```bash
# 检查 debugfs 是否可用
./fix_debugfs.sh
```

#### 步骤 4: 运行快速测试（如果 debugfs 可用）

```bash
# PyTorch 版本（推荐）
./quick_queue_test.sh
```

#### 步骤 5: 验证一致性

```bash
./test_queue_consistency.sh
```

---

### ⚠️ 如果容器内看不到 debugfs

**症状**: `/sys/kernel/debug/kfd/mqds: No such file or directory`

**解决方案**: 使用**方案 A**（宿主机脚本），它会：
1. 在宿主机访问 debugfs
2. 在容器内启动测试程序
3. 自动关联 PID 和 Queue ID

---

## 📊 预期结果

### 成功的 quick_queue_test.sh 输出

```
╔════════════════════════════════════════════════════════╗
║  Queue ID 快速测试 (PyTorch 版本)                       ║
╚════════════════════════════════════════════════════════╝

✅ 环境已激活

📝 测试脚本已创建: /tmp/queue_test_torch.py

🚀 启动 PyTorch 测试 (后台运行)...

✅ 测试进程已启动
   PID: 12345

⏳ 等待 5 秒让程序初始化...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 MQD 信息 (PID: 12345)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Compute queue on device 0001:01:00.0
    Queue ID: 0 (0x0)
    Address: 0x7f8c00000000
    Process: pid 12345 pasid 0x8001
    is active: yes
    priority: 7

Compute queue on device 0001:01:00.0
    Queue ID: 1 (0x1)
    Address: 0x7f8c10000000
    Process: pid 12345 pasid 0x8001
    is active: yes
    priority: 7

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 提取的 Queue IDs:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Queue ID: 0 (0x0)
    Queue ID: 1 (0x1)

✅ 成功找到 Queue 信息！
```

---

### 成功的 test_queue_consistency.sh 输出（一致性高）

```
╔════════════════════════════════════════════════════════╗
║  测试结果汇总                                           ║
╚════════════════════════════════════════════════════════╝

各次运行的 Queue IDs:
  Run 1: 0,1
  Run 2: 0,1
  Run 3: 0,1
  Run 4: 0,1
  Run 5: 0,1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 一致性分析:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 所有运行的 Queue IDs 完全一致！

   固定的 Queue IDs: 0,1

💡 结论: Queue ID 高度可预测
   → POC Stage 1 可以使用硬编码 Queue ID
   → 实施时间: 3-5 天
```

---

## 🔧 故障排除

### 问题 1: 权限被拒绝

```bash
# 如果看到 Permission denied
# 以 root 身份进入容器
docker exec -u root -it zhenaiter /bin/bash
cd /data/dockercode/gpreempt_test
./quick_queue_test.sh
```

---

### 问题 2: 脚本不可执行

```bash
# 添加执行权限
chmod +x *.sh
```

---

### 问题 3: 未找到 Queue ID

**可能原因**:
1. debugfs 不可用
2. 程序运行太快
3. PID 不正确

**解决方案**:

```bash
# 检查 debugfs
ls -la /sys/kernel/debug/kfd/

# 如果不存在，挂载 debugfs
sudo mount -t debugfs none /sys/kernel/debug

# 手动运行更长时间的测试
python3 << 'EOF'
import torch
import time
import os
print(f"PID: {os.getpid()}")
x = torch.randn(3000, 3000, device='cuda')
for i in range(10000):
    y = torch.mm(x, x)
    time.sleep(0.1)
EOF
```

然后在另一个终端查看：

```bash
# 获取 Python 进程的 PID
ps aux | grep python

# 查看该 PID 的 Queue
sudo cat /sys/kernel/debug/kfd/mqds | grep -B 2 -A 5 "pid <YOUR_PID>"
```

---

### 问题 4: PyTorch GPU 不可用

```bash
# 检查环境
python3 -c "import torch; print(torch.cuda.is_available())"

# 如果返回 False，检查 ROCm
rocm-smi

# 重新激活环境
export MAMBA_EXE='/root/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/root/micromamba'
eval "$(/root/.local/bin/micromamba shell hook --shell=bash)"
micromamba activate flashinfer-rocm
```

---

## 📖 相关文档

- **QUICKSTART_实验立即开始.md** - 快速开始指南
- **TROUBLESHOOTING_常见问题解决.md** - 详细故障排除
- **EXP_Design_01_MQD_HQD_模型关联性实验.md** - 完整实验设计
- **ARCH_Design_03_QueueID获取与环境配置.md** - Queue ID 获取方法

---

## 🎯 测试目标

### 主要目标

1. ✅ **验证 MQD debugfs 可用**
2. ✅ **验证可以追踪进程的 Queue ID**
3. ✅ **了解 Queue ID 的分配模式**

### 次要目标

1. **Queue ID 一致性**: 同一程序多次运行，Queue ID 是否相同？
2. **Queue ID 范围**: Queue ID 是小整数 (0, 1, 2) 还是随机？
3. **可预测性**: 能否预先知道程序会使用哪些 Queue？

---

## 💡 根据结果的下一步

### 如果 Queue ID 一致 ✅

→ 使用硬编码策略
→ 阅读 `ARCH_Design_01` 简化版
→ 3-5 天完成 POC Stage 1

### 如果 Queue ID 不一致 ⚠️

→ 使用动态发现策略
→ 阅读 `ARCH_Design_03` 动态发现部分
→ 7-10 天完成 POC Stage 1

---

## 🔍 手动验证（如果脚本失败）

```bash
# 1. 激活环境
export MAMBA_EXE='/root/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/root/micromamba'
eval "$(/root/.local/bin/micromamba shell hook --shell=bash)"
micromamba activate flashinfer-rocm

# 2. 启动测试（后台）
python3 -c "
import torch
import time
import os
print(f'PID: {os.getpid()}')
x = torch.randn(2000, 2000, device='cuda')
for i in range(3000):
    y = torch.mm(x, x)
    time.sleep(0.01)
" &

# 3. 记录 PID
PID=$!
echo "PID: $PID"

# 4. 等待启动
sleep 5

# 5. 查看 Queue ID
sudo cat /sys/kernel/debug/kfd/mqds | grep -B 2 -A 5 "pid $PID"
```

---

**最后更新**: 2026-02-03  
**维护者**: Zhehan

如有问题，参考 `TROUBLESHOOTING_常见问题解决.md`
