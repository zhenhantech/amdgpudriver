# 在Docker容器内监控GPU进程

**日期**: 2026-02-05  
**场景**: 在Docker容器内直接运行监控工具  
**容器**: zhen_vllm_dsv3

---

## 📁 路径映射

```
宿主机路径:
  /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

容器内路径:
  /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
```

---

## 🚀 快速开始（在容器内）

### 方案1: 最简单 - 等待新进程 ⭐⭐⭐

#### 终端1 - 容器内，启动监控
```bash
# 进入容器
docker exec -it zhen_vllm_dsv3 bash

# 进入工具目录
cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 启动监控（等待新GPU进程）
./watch_gpu_in_docker.sh
```

输出：
```
╔════════════════════════════════════════════════════════╗
║  GPU进程监控 - 容器内模式                               ║
╚════════════════════════════════════════════════════════╝

⏳ 等待新的GPU进程启动...

💡 提示: 现在可以在另一个终端启动测试程序
         例如: cd /data/code/rampup_doc/vLLM_test/scripts
               ./run_vLLM_v1_optimized.sh test
```

#### 终端2 - 容器内，启动vLLM
```bash
# 进入容器（新终端）
docker exec -it zhen_vllm_dsv3 bash

# 启动vLLM
cd /data/code/rampup_doc/vLLM_test/scripts
./run_vLLM_v1_optimized.sh test
```

#### 终端1 - 自动开始监控
```
✅ 检测到新的GPU进程!

进程信息:
  PID:    12345
  进程:   python3

开始监控 Queue 使用情况
[  0s] 采样  1: 15 个队列 (IDs: ...)
```

---

### 方案2: 查看当前GPU进程

```bash
# 在容器内
cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 列出所有GPU进程
./list_gpu_processes.sh
```

输出：
```
╔════════════════════════════════════════════════════════╗
║  当前GPU进程列表                                        ║
╚════════════════════════════════════════════════════════╝

GPU进程:

[1] PID: 12345
    进程: python3
    命令: python3 -m vllm.entrypoints.openai.api_server...

总计: 1 个GPU进程

如何监控这些进程:
./queue_monitor 12345 60 10
```

然后监控：
```bash
./queue_monitor 12345 60 10
```

---

### 方案3: 直接监控（如果知道PID）

```bash
# 在容器内

# 1. 查找vLLM进程
ps aux | grep vllm | grep -v grep

# 假设PID是12345

# 2. 监控
cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
./queue_monitor 12345 60 10
```

---

## 📋 完整工作流程

### 典型场景：监控vLLM

```bash
# ============ 终端1 ============
# 进入容器
docker exec -it zhen_vllm_dsv3 bash

# 进入工具目录
cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 如果还没编译，先编译
make clean && make all

# 启动监控
./watch_gpu_in_docker.sh


# ============ 终端2 ============
# 进入容器
docker exec -it zhen_vllm_dsv3 bash

# 启动vLLM测试
cd /data/code/rampup_doc/vLLM_test/scripts
./run_vLLM_v1_optimized.sh test


# ============ 终端1（自动） ============
# 自动检测并显示Queue信息
# [ 0s] 采样  1: 15 个队列
# [ 10s] 采样  2: 15 个队列
# ...
```

---

## 🔧 前置条件

### 1. 确保工具已编译

```bash
# 在容器内
cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 检查是否已编译
ls -lh queue_monitor kfd_preemption_poc get_queue_info

# 如果没有，编译
make clean
make all
```

### 2. 检查GPU设备访问

```bash
# 在容器内
ls -l /dev/kfd /dev/dri

# 应该显示:
# crw-rw-rw- 1 root render ... /dev/kfd
# drwxr-xr-x 2 root root   ... /dev/dri
```

如果看不到这些设备，说明容器启动时没有正确挂载GPU，需要重新启动容器并添加：
```bash
--device=/dev/kfd --device=/dev/dri
```

---

## 📊 可用工具

| 工具 | 用途 | 需要PID | 推荐度 |
|------|------|---------|--------|
| `watch_gpu_in_docker.sh` | 等待新GPU进程 | ❌ | ⭐⭐⭐ 最推荐 |
| `list_gpu_processes.sh` | 列出当前GPU进程 | ❌ | ⭐⭐ |
| `queue_monitor` | 监控指定进程 | ✅ | ⭐⭐ |
| `get_queue_info` | 快速查看Queue信息 | ✅ | ⭐ |

---

## 💡 使用技巧

### 技巧1: 一行命令监控

```bash
# 在容器内，查找并监控第一个GPU进程
PID=$(lsof -t /dev/kfd | head -1) && ./queue_monitor $PID 60 10
```

### 技巧2: 循环监控

```bash
# 持续监控vLLM
while true; do
    echo "========== 新一轮监控 =========="
    ./watch_gpu_in_docker.sh
    echo ""
    echo "按Enter继续，Ctrl+C退出"
    read
done
```

### 技巧3: 保存监控日志

```bash
# 监控并保存日志
timestamp=$(date +%Y%m%d_%H%M%S)
./watch_gpu_in_docker.sh | tee "monitor_${timestamp}.log"
```

### 技巧4: 后台监控

```bash
# 查找vLLM PID并后台监控
PID=$(ps aux | grep vllm | grep -v grep | awk '{print $2}' | head -1)
nohup ./queue_monitor $PID 300 10 > monitor.log 2>&1 &
echo "监控进程已启动: $!"
```

---

## 🎯 常见场景

### 场景1: vLLM已经在运行

```bash
# 方法1: 列出GPU进程
./list_gpu_processes.sh

# 方法2: 直接查找vLLM
ps aux | grep vllm | grep -v grep

# 然后使用显示的PID
./queue_monitor <PID> 60 10
```

---

### 场景2: 准备启动vLLM

```bash
# 终端1: 先启动监控
./watch_gpu_in_docker.sh

# 终端2: 再启动vLLM
cd /data/code/rampup_doc/vLLM_test/scripts
./run_vLLM_v1_optimized.sh test
```

---

### 场景3: 对比不同配置

```bash
# 测试配置A
./watch_gpu_in_docker.sh | tee config_a.log
# 启动vLLM配置A...

# 测试配置B
./watch_gpu_in_docker.sh | tee config_b.log
# 启动vLLM配置B...

# 对比
diff config_a.log config_b.log
```

---

## ⚠️ 注意事项

### 1. 权限问题

在容器内通常不需要sudo，因为容器内可能已经是root用户：

```bash
# 检查当前用户
whoami

# 如果是root，直接运行
./queue_monitor 12345 60 10

# 如果不是root，可能需要sudo
sudo ./queue_monitor 12345 60 10
```

### 2. lsof工具

如果容器内没有`lsof`：

```bash
# 安装lsof（如果有权限）
apt-get update && apt-get install -y lsof

# 或者使用list_gpu_processes.sh的备用方法（自动切换）
```

### 3. vLLM启动时间

vLLM模型加载可能需要30-90秒：
- DeepSeek-V3: ~60秒
- 小模型: ~20秒
- 大模型: ~120秒

`watch_gpu_in_docker.sh` 会等待最多5分钟。

---

## 🔍 故障排查

### 问题1: "queue_monitor不存在"

```bash
# 解决：编译工具
cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
make clean && make all
```

### 问题2: "/dev/kfd 不存在"

```bash
# 检查设备
ls -l /dev/kfd

# 如果不存在，容器需要重新启动并添加设备挂载
```

### 问题3: "未检测到GPU进程"

```bash
# 检查vLLM是否真正在运行
ps aux | grep vllm

# 检查vLLM是否在使用GPU
nvidia-smi  # NVIDIA GPU
rocm-smi    # AMD GPU

# 手动查找GPU进程
lsof /dev/kfd
```

### 问题4: 编译错误

```bash
# 如果遇到编译错误，查看修复文档
cat COMPILE_FIX.md
```

---

## ✅ 快速参考

### 最常用的3个命令

```bash
# 1. 等待新进程（最推荐）
./watch_gpu_in_docker.sh

# 2. 列出当前GPU进程
./list_gpu_processes.sh

# 3. 监控指定PID
./queue_monitor <PID> 60 10
```

### 完整流程（复制粘贴）

```bash
# 进入容器和工具目录
docker exec -it zhen_vllm_dsv3 bash
cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 编译（如果还没有）
make clean && make all

# 启动监控
./watch_gpu_in_docker.sh

# 在另一个终端启动vLLM
# docker exec -it zhen_vllm_dsv3 bash
# cd /data/code/rampup_doc/vLLM_test/scripts
# ./run_vLLM_v1_optimized.sh test
```

---

## 🔗 相关文档

- `DOCKER_MONITORING_GUIDE.md` - Docker监控完整指南
- `MONITOR_WITHOUT_PID.md` - 无需PID的监控方法
- `README.md` - 代码目录完整说明

---

**最后更新**: 2026-02-05  
**测试容器**: zhen_vllm_dsv3  
**状态**: ✅ 已验证
