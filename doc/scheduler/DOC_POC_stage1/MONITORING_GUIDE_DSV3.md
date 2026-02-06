# DeepSeek V3 GPU队列监控完整指南

**路径**: `/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code`  
**更新**: 2026-02-05  
**容器**: zhen_vllm_dsv3

---

## 📋 目录

1. [监控方式选择](#监控方式选择)
2. [方式1: 自动监控（推荐）](#方式1-自动监控推荐)
3. [方式2: 手动监控](#方式2-手动监控)
4. [方式3: sysfs监控](#方式3-sysfs监控)
5. [常见问题](#常见问题)
6. [监控数据解读](#监控数据解读)

---

## 监控方式选择

| 方式 | 适用场景 | 难度 | 推荐度 |
|------|---------|------|--------|
| **自动监控** | 不知道PID，等待程序启动 | ⭐ 简单 | ⭐⭐⭐⭐⭐ |
| **手动监控** | 已知PID，精确控制 | ⭐⭐ 中等 | ⭐⭐⭐ |
| **sysfs监控** | 需要查看HQD详情 | ⭐⭐⭐ 较难 | ⭐⭐ |

---

## 方式1: 自动监控（推荐）

### 在宿主机监控Docker容器内的GPU进程

**最佳实践**: 两个终端同时操作

#### 终端1: 启动监控（等待模式）

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 监控指定容器，等待300秒，每5秒检查一次
sudo ./watch_docker_gpu.sh zhen_vllm_dsv3 300 5
```

#### 终端2: 启动DSV3

```bash
docker exec -it zhen_vllm_dsv3 bash
cd /data/code/rampup_doc/vLLM_test/scripts
./run_vLLM_v1_optimized.sh test
```

**监控脚本会自动**:
1. 检测到新的GPU进程
2. 获取宿主机PID
3. 启动queue_monitor监控
4. 持续记录队列信息

**输出示例**:
```
╔════════════════════════════════════════════════════════╗
║  Docker容器GPU进程监控                                  ║
╚════════════════════════════════════════════════════════╝

⏳ 等待容器内新的GPU进程启动...

✅ 检测到新的GPU进程!
  容器内PID:   151623
  宿主机PID:   2231234
  进程:        python3

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
开始监控队列 (PID: 2231234)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[  0s] 采样 1: 24 个队列
[  5s] 采样 2: 24 个队列
...
```

---

## 方式2: 手动监控

### 步骤1: 查找GPU进程

#### 在宿主机查找

```bash
sudo lsof /dev/kfd | grep python
```

**输出示例**:
```
python3   2231234 root  mem  CHR  235,0  /dev/kfd
```

→ 宿主机PID = **2231234**

#### 在容器内查找

```bash
docker exec zhen_vllm_dsv3 bash -c "ps aux | grep python | grep -v grep"
```

**输出示例**:
```
root  151623  25.5  0.0  ... python3 test_inference.py
```

→ 容器内PID = **151623**

### 步骤2: 诊断进程状态（可选）

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 诊断容器内进程
sudo ./diagnose_process.sh zhen_vllm_dsv3 151623
```

**输出会告诉您**:
- ✅ 进程是否运行中
- ✅ 是否使用GPU
- ✅ 容器PID到宿主机PID的映射
- ✅ 进程状态分析（Running/Sleeping等）

### 步骤3: 开始监控

#### 从宿主机监控（推荐）

使用宿主机PID:

```bash
# API方式 - 监控60秒，每10秒采样
sudo ./queue_monitor 2231234 60 10

# 或者使用监控助手（带验证）
sudo ./monitor_with_pid.sh 2231234 60 10
```

#### 从容器内监控（可能有权限问题）

```bash
docker exec zhen_vllm_dsv3 bash -c "cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code && sudo ./queue_monitor 151623 60 10"
```

⚠️ **注意**: 容器内监控可能遇到 "Operation not permitted" 错误，推荐从宿主机监控。

---

## 方式3: sysfs监控

### 特点

- ✅ 可以看到**完整的MQD和HQD信息**
- ✅ 可以看到**硬件队列细节**（Pipe、Queue、XCC分布）
- ✅ 可以保存历史快照
- ❌ 只能从宿主机运行（debugfs权限）

### 使用方法

#### 3.1 持续采样监控

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 使用宿主机PID
sudo ./monitor_queue_sysfs.sh 2231234 60 10
```

**输出**:
- MQD信息（软件队列）
- HQD信息（硬件队列）
- 队列数量统计
- 历史快照文件

#### 3.2 实时监控（htop风格）

```bash
# 实时刷新，类似htop
sudo ./watch_queue_live.sh 2231234 3
```

**输出示例**:
```
╔════════════════════════════════════════════════════════╗
║  KFD Queue 实时监控 - sysfs方式                        ║
╚════════════════════════════════════════════════════════╝

时间: 2026-02-05 15:30:45   刷新间隔: 3秒

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
目标进程: PID 2231234
进程名:   python3
状态:     运行中

Queue统计:
  总队列数: 24

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
硬件队列 (HQD):
  总活跃HQD: 96
  映射比例:  1:4.0 (MQD:HQD)
  ℹ️  MI308X: 每个MQD对应4个HQD (4个XCC)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### 3.3 对比API和sysfs数据

```bash
# 验证数据一致性
sudo ./compare_api_vs_sysfs.sh 2231234
```

**输出示例**:
```
| 方法 | 队列数量 | 状态 |
|------|---------|------|
| API (GET_QUEUE_SNAPSHOT) | 24 | ✅ |
| sysfs (MQD) | 24 | ✅ |
| sysfs (HQD总计) | 96 | ✅ |

✅ API和sysfs数据一致！

MQD到HQD映射比例: 1:4.0
ℹ️  MI308X典型映射: 1个MQD → 4个HQD (每个XCC一个)
```

---

## 常见问题

### Q1: "Operation not permitted" 错误

**原因**: KFD debug trap权限不足

**解决方案**:
1. ✅ 使用 `sudo` 运行
2. ✅ 从**宿主机**监控（而非容器内）
3. ✅ 使用宿主机PID（而非容器PID）

**示例**:
```bash
# ❌ 错误: 在容器内用容器PID
docker exec zhen_vllm_dsv3 bash -c "./queue_monitor 151623 ..."

# ✅ 正确: 在宿主机用宿主机PID
sudo ./queue_monitor 2231234 ...
```

### Q2: "进程不存在" 或 "未找到GPU进程"

**原因**: 
- 进程已结束
- 进程还未初始化GPU
- PID不正确

**解决方案**:
1. 检查进程是否运行:
   ```bash
   ps aux | grep python
   ```

2. 检查是否使用GPU:
   ```bash
   sudo lsof /dev/kfd | grep python
   ```

3. 使用自动监控等待:
   ```bash
   sudo ./watch_docker_gpu.sh zhen_vllm_dsv3 300 5
   ```

### Q3: DSV3程序运行很快就结束了

**原因**: 
- 测试程序可能只是推理几次就退出
- 可能有错误导致提前退出

**解决方案**:
1. 使用长时间运行的测试:
   ```bash
   # 修改test_inference.py，增加更多推理次数或循环
   ```

2. 在启动DSV3前就启动监控（等待模式）:
   ```bash
   # 终端1: 提前启动监控
   sudo ./watch_docker_gpu.sh zhen_vllm_dsv3 300 5
   
   # 终端2: 然后启动DSV3
   docker exec -it zhen_vllm_dsv3 bash
   ...
   ```

### Q4: 看不到HQD信息

**原因**: debugfs不可用或权限不足

**解决方案**:
1. 确保在宿主机运行:
   ```bash
   # 在宿主机（不是容器内）
   sudo cat /sys/kernel/debug/kfd/hqds
   ```

2. 检查debugfs挂载:
   ```bash
   mount | grep debugfs
   # 应该看到: debugfs on /sys/kernel/debug type debugfs
   ```

3. 使用root权限:
   ```bash
   sudo ./watch_queue_live.sh 2231234 3
   ```

### Q5: 如何同时监控多个模型？

**场景**: 同时运行模型A和模型B，对比它们的队列使用

**方案1: 两个终端分别监控**

```bash
# 终端1: 监控模型A
sudo ./queue_monitor PID_A 300 10 > model_a_queues.log

# 终端2: 监控模型B
sudo ./queue_monitor PID_B 300 10 > model_b_queues.log

# 终端3: 对比结果
diff model_a_queues.log model_b_queues.log
```

**方案2: 使用auto_monitor多进程模式**

```bash
# 显示所有GPU进程
sudo ./auto_monitor.sh --all

# 然后分别启动监控
```

---

## 监控数据解读

### Queue数量含义

#### MQD (Memory Queue Descriptor)

**典型值**:
- **24个队列** (常见于8 GPU setup)
  - 每个GPU: 3个队列
  - 8个GPU: 8 × 3 = 24个队列

**队列类型**:
- Compute queue (计算队列): 执行kernel
- SDMA queue (DMA队列): 数据传输
- 其他队列: 根据workload而定

#### HQD (Hardware Queue Descriptor)

**典型值**:
- **96个HQD** (MI308X with 8 GPUs)
  - 每个MQD对应4个HQD (4个XCC)
  - 24 MQD × 4 XCC = 96 HQD

**映射关系**:
```
MQD (软件)  →  HQD (硬件)
   1:1            对于单XCC GPU (MI250X)
   1:4            对于4-XCC GPU (MI308X)
```

### 队列状态变化

#### 稳定状态

```
[  0s] 采样 1: 24 个队列
[  5s] 采样 2: 24 个队列
[ 10s] 采样 3: 24 个队列
```

→ ✅ **队列数量稳定**，说明模型运行正常

#### 动态变化

```
[  0s] 采样 1: 0 个队列    ← 初始化中
[  5s] 采样 2: 8 个队列    ← 开始创建队列
[ 10s] 采样 3: 24 个队列   ← 所有GPU就绪
[ 15s] 采样 4: 24 个队列   ← 稳定运行
```

→ ✅ **正常启动过程**

#### 异常情况

```
[  0s] 采样 1: 24 个队列
[  5s] 采样 2: 12 个队列   ← ⚠️ 队列突然减少
[ 10s] 采样 3: 0 个队列    ← ❌ 队列全部消失
```

→ ❌ **可能的问题**:
- GPU崩溃
- 进程被终止
- 资源不足

### Queue ID的意义

**Queue ID** 是KFD分配的唯一标识符:

```
Queue ID: 12345
  ├─ GPU ID: 0 (哪个GPU)
  ├─ Queue Type: Compute (队列类型)
  ├─ Ring Buffer: 0x7f3c01a97c00 (命令缓冲区地址)
  └─ CWSR Area: 0xab780000b5c4e749 (上下文保存区地址)
```

**用途**:
1. **队列识别**: 确定是哪个队列
2. **抢占控制**: suspend/resume特定队列
3. **调试**: 追踪队列生命周期
4. **性能分析**: 关联队列与workload

---

## 完整工作流示例

### 场景: 对比两个模型的队列使用

#### 步骤1: 准备监控

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 确保工具已编译
make clean && make all
```

#### 步骤2: 测试模型A

```bash
# 终端1: 启动监控
sudo ./watch_docker_gpu.sh zhen_vllm_dsv3 300 5 > model_a.log 2>&1 &

# 终端2: 启动模型A
docker exec -it zhen_vllm_dsv3 bash
cd /data/code/rampup_doc/vLLM_test/scripts
./run_vLLM_v1_optimized.sh test_model_a

# 等待完成，查看日志
cat model_a.log
```

#### 步骤3: 测试模型B

```bash
# 重复步骤2，使用不同的测试脚本
sudo ./watch_docker_gpu.sh zhen_vllm_dsv3 300 5 > model_b.log 2>&1 &
docker exec -it zhen_vllm_dsv3 bash
./run_vLLM_v1_optimized.sh test_model_b

cat model_b.log
```

#### 步骤4: 对比结果

```bash
# 对比队列数量
echo "模型A队列数:" && grep "个队列" model_a.log | head -5
echo "模型B队列数:" && grep "个队列" model_b.log | head -5

# 详细对比
diff model_a.log model_b.log

# 提取Queue ID进行对比
grep "Queue ID" model_a.log | sort > queues_a.txt
grep "Queue ID" model_b.log | sort > queues_b.txt
comm -12 queues_a.txt queues_b.txt  # 共同的队列
comm -23 queues_a.txt queues_b.txt  # 只在A中的队列
comm -13 queues_a.txt queues_b.txt  # 只在B中的队列
```

#### 步骤5: 同时运行A+B

```bash
# 启动全局监控
sudo ./watch_queue_live.sh 3  # 监控所有进程

# 在另外的终端启动两个模型
# 观察队列是否重叠或独立
```

---

## 脚本速查表

| 脚本 | 用途 | 运行位置 | 需要PID | 难度 |
|------|------|---------|---------|------|
| `watch_docker_gpu.sh` | 等待Docker容器GPU进程 | 宿主机 | ❌ | ⭐ |
| `watch_gpu_in_docker.sh` | 等待GPU进程（容器内） | 容器内 | ❌ | ⭐ |
| `auto_monitor.sh` | 智能监控（多模式） | 宿主机 | ❌ | ⭐ |
| `diagnose_process.sh` | 诊断进程状态 | 宿主机 | ✅ | ⭐⭐ |
| `monitor_with_pid.sh` | 监控指定PID（带验证） | 宿主机 | ✅ | ⭐⭐ |
| `queue_monitor` | API方式监控Queue | 宿主机 | ✅ | ⭐⭐ |
| `monitor_queue_sysfs.sh` | sysfs持续监控 | 宿主机 | ✅ | ⭐⭐⭐ |
| `watch_queue_live.sh` | sysfs实时监控 | 宿主机 | ✅ | ⭐⭐⭐ |
| `compare_api_vs_sysfs.sh` | 对比API和sysfs | 宿主机 | ✅ | ⭐⭐⭐ |

---

## 推荐工作流

### 对于不熟悉PID的用户 ⭐⭐⭐⭐⭐

```bash
# 一个命令搞定！
sudo ./watch_docker_gpu.sh zhen_vllm_dsv3 300 5
```

然后在另一个终端启动模型即可。

### 对于需要详细调试的用户

```bash
# 1. 诊断当前状态
sudo ./diagnose_process.sh zhen_vllm_dsv3 <PID>

# 2. API监控
sudo ./queue_monitor <HOST_PID> 60 10

# 3. sysfs详细监控
sudo ./monitor_queue_sysfs.sh <HOST_PID> 60 10

# 4. 对比验证
sudo ./compare_api_vs_sysfs.sh <HOST_PID>
```

---

**最后更新**: 2026-02-05  
**维护者**: AI Assistant  
**测试平台**: MI308X + ROCm 7.x + Docker
