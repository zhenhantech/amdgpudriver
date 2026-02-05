# 用户空间POC - 5分钟快速开始

**日期**: 2026-02-05  
**目的**: 快速测试C++版本的Queue监控和抢占POC

---

## 🚀 一键测试

```bash
cd /mnt/md0/zhehan/code/flashinfer/dockercode/gpreempt_test

# 一键完成：编译 + 启动测试模型 + 运行POC
./test_userspace_poc.sh
```

**这个脚本会自动**：
1. ✅ 编译所有工具（queue_monitor, kfd_preemption_poc, get_queue_info）
2. ✅ 在Docker内启动测试模型
3. ✅ 测试queue_monitor（采样4次）
4. ✅ 测试kfd_preemption_poc（10次迭代）
5. ✅ 自动清理

**预计时间**: ~2分钟

---

## 📝 手动测试（逐步）

### 步骤1: 编译

```bash
cd /mnt/md0/zhehan/code/flashinfer/dockercode/gpreempt_test

make clean
make all

# 检查生成的文件
ls -lh queue_monitor kfd_preemption_poc get_queue_info
```

### 步骤2: 启动测试模型

**终端1 - Docker内**:
```bash
docker exec -it zhenaiter bash

# 激活环境
export MAMBA_EXE='/root/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/root/micromamba'
eval "$(/root/.local/bin/micromamba shell hook --shell=bash)"
micromamba activate flashinfer-rocm

# 运行测试模型
python3 << 'EOF'
import torch
import time
import os

print(f"PID: {os.getpid()}")  # 记下这个PID

x = torch.randn(3000, 3000, device='cuda')
y = torch.randn(3000, 3000, device='cuda')

print("开始GPU计算...")
while True:
    z = torch.mm(x, y)
    torch.cuda.synchronize()
    time.sleep(0.02)
EOF
```

### 步骤3: 监控Queue（宿主机）

**终端2 - 宿主机**:
```bash
# 获取容器内PID
CONTAINER_PID=$(docker exec zhenaiter pgrep -f python3 | head -1)
echo "Container PID: $CONTAINER_PID"

# 监控30秒，每5秒采样
sudo ./queue_monitor $CONTAINER_PID 30 5
```

**期望输出**:
```
[ 0s] 采样  1: 10 个队列 (IDs: 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
[ 5s] 采样  2: 10 个队列 (IDs: 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
...

统计分析:
  采样次数: 6
  平均队列数: 10.0
  稳定性: ✅ 稳定
```

### 步骤4: 测试抢占POC（宿主机）

**终端2 - 继续**:
```bash
# 运行10次抢占测试
sudo ./kfd_preemption_poc $CONTAINER_PID 10
```

**期望输出**:
```
[  1] Suspend:   450 μs | Online-AI:  100 ms | Resume:   380 μs
[  2] Suspend:   420 μs | Online-AI:  101 ms | Resume:   390 μs
...

POC 统计结果:
  总迭代次数: 10
  成功次数: 10
  成功率: 100.0%
  
  Suspend平均延迟: 425 μs
  Resume平均延迟:  387 μs
```

### 步骤5: 清理

**终端1 - Docker内**:
按 `Ctrl+C` 停止模型

---

## 🎯 典型使用场景

### 场景1: 快速查看Queue信息

```bash
CONTAINER_PID=$(docker exec zhenaiter pgrep -f your_model.py)
sudo ./get_queue_info $CONTAINER_PID
```

**用途**: 快速查看某个模型使用了哪些Queue ID

### 场景2: 持续监控

```bash
# 监控1小时，每10秒采样
sudo ./queue_monitor $CONTAINER_PID 3600 10 > monitor_log.txt
```

**用途**: 长时间观察Queue使用情况的稳定性

### 场景3: 压力测试抢占

```bash
# 运行500次抢占测试
sudo ./kfd_preemption_poc $CONTAINER_PID 500
```

**用途**: 测试抢占机制的稳定性和性能

### 场景4: 对比两个模型

```bash
# 启动模型A
docker exec zhenaiter python3 model_a.py &
PID_A=$(docker exec zhenaiter pgrep -f model_a.py)
sudo ./queue_monitor $PID_A 20 5 > model_a_queues.txt

# 启动模型B
docker exec zhenaiter python3 model_b.py &
PID_B=$(docker exec zhenaiter pgrep -f model_b.py)
sudo ./queue_monitor $PID_B 20 5 > model_b_queues.txt

# 对比
diff model_a_queues.txt model_b_queues.txt
```

**用途**: 了解不同模型的Queue使用差异

---

## 🔧 故障排除

### 问题1: 编译失败 "linux/kfd_ioctl.h: No such file"

**解决**:
```bash
# 查找头文件位置
find /usr/src -name "kfd_ioctl.h"

# 输出: /usr/src/amdgpu-x.x.x/include/uapi/linux/kfd_ioctl.h

# 修改Makefile中的INCLUDES路径
vim Makefile
# INCLUDES = -I/usr/src/amdgpu-x.x.x/include/uapi
```

### 问题2: "Failed to enable debug trap"

**可能原因**:
1. 进程不是GPU进程（未使用CUDA/ROCm）
2. 权限不足（需要sudo）
3. 进程已被其他调试器附加

**解决**:
```bash
# 1. 确认进程使用GPU
docker exec zhenaiter rocm-smi

# 2. 确保使用sudo
sudo ./queue_monitor $PID

# 3. 检查进程是否真正在使用KFD
sudo cat /sys/kernel/debug/kfd/mqds | grep "Process $PID"
```

### 问题3: "No queues found"

**可能原因**: 模型还未初始化完成

**解决**:
```bash
# 等待更长时间
sleep 10

# 或在模型代码中确保创建了GPU数据
# x = torch.randn(..., device='cuda')  # ✅ 正确
# x = torch.randn(...)                  # ❌ 错误（CPU）
```

### 问题4: Suspend/Resume失败

**可能原因**: 内核不支持或需要特定配置

**解决**:
```bash
# 检查KFD调试功能是否可用
sudo dmesg | grep kfd

# 检查是否有相关内核参数
cat /proc/cmdline | grep amdgpu

# 如果suspend_queues不工作，可能需要:
# 1. 更新驱动到最新版本
# 2. 启用KFD调试功能（内核编译选项）
# 3. 联系AMD支持
```

---

## 📊 理解输出

### queue_monitor输出解读

```
[  0s] 采样  1: 10 个队列 (IDs: 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
```

- `[0s]`: 从开始监控的时间
- `采样 1`: 第1次采样
- `10 个队列`: 该进程当前有10个活跃队列
- `IDs: ...`: 具体的Queue ID（用于suspend/resume）

### 统计分析解读

```
━━━ Queue ID 出现频率 ━━━
  Queue     5:   6/6 (100.0%)
```

- `Queue 5`: Queue ID为5的队列
- `6/6`: 在6次采样中出现了6次
- `100.0%`: 出现频率100%，说明这个Queue非常稳定

**如果频率<100%**: 说明该Queue是动态创建/销毁的

### kfd_preemption_poc输出解读

```
[  1] Suspend:   450 μs | Online-AI:  100 ms | Resume:   380 μs
```

- `Suspend: 450 μs`: 调用suspend_queues花费450微秒
- `Online-AI: 100 ms`: 模拟的Online-AI推理时间
- `Resume: 380 μs`: 调用resume_queues花费380微秒

**总抢占开销** = Suspend + Resume = ~830 μs

---

## 💡 下一步

完成基础测试后，您可以：

1. **实际模型测试**: 用真实的AI模型替换测试脚本
2. **性能优化**: 分析Suspend/Resume延迟瓶颈
3. **集成到调度器**: 将这些API集成到实际的GPU调度系统
4. **扩展功能**: 添加自定义的监控和分析逻辑

详细文档：
- `README_USERSPACE_POC.md` - 完整使用指南
- `GET_QUEUE_SNAPSHOT_API_GUIDE.md` - API详细说明

---

## 🎓 核心概念速查

| 概念 | 说明 |
|------|------|
| **Queue ID** | KFD内部的队列标识符，用于suspend/resume |
| **Debug Trap** | KFD的调试接口，提供Queue控制能力 |
| **Suspend** | 暂停队列执行（触发CWSR） |
| **Resume** | 恢复队列执行 |
| **CWSR** | 硬件级的Wave保存/恢复机制 |
| **MQD** | 软件层的队列描述符 |
| **HQD** | 硬件层的队列寄存器 |

---

**最后更新**: 2026-02-05  
**测试状态**: ✅ MI308X验证通过
