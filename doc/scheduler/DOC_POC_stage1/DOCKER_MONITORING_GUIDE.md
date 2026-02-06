# Docker容器GPU进程监控指南

**日期**: 2026-02-05  
**场景**: 监控Docker容器内运行的GPU程序（如vLLM）

---

## 🎯 问题说明

当你的GPU程序运行在Docker容器内时：
- 容器内的PID与宿主机的PID不同
- `watch_new_gpu.sh` 无法直接检测容器内的进程
- 需要特殊方法来监控容器内的GPU进程

---

## ✅ 解决方案（3种方法）

### 方法1: 使用 `auto_monitor.sh --container` ⭐⭐⭐ (推荐)

**最简单、最可靠**

```bash
# 在宿主机运行
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

./auto_monitor.sh --container zhen_vllm_dsv3
```

**优点**:
- ✅ 自动找到容器内的GPU进程
- ✅ 自动转换为宿主机PID
- ✅ 如果有多个进程，会让你选择

**输出示例**:
```
容器 'zhen_vllm_dsv3' 内的GPU进程:

[1] 
PID: 12345  进程: python3
  命令: python3 -m vllm.entrypoints...

请选择 [1-1]: 1

开始监控...
```

---

### 方法2: 使用 `watch_docker_gpu.sh` ⭐⭐

**适合等待容器内新进程启动**

#### 终端1 - 宿主机运行监控
```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

./watch_docker_gpu.sh zhen_vllm_dsv3
```

输出：
```
╔════════════════════════════════════════════════════════╗
║  Docker容器GPU进程监控                                  ║
╚════════════════════════════════════════════════════════╝

目标容器: zhen_vllm_dsv3

⏳ 等待容器内新的GPU进程启动...

💡 提示: 现在可以在容器内启动测试程序
```

#### 终端2 - 容器内启动测试
```bash
# 如果已经在容器内
cd /data/code/rampup_doc/vLLM_test/scripts
./run_vLLM_v1_optimized.sh test

# 如果不在容器内
docker exec -it zhen_vllm_dsv3 bash
cd /data/code/rampup_doc/vLLM_test/scripts
./run_vLLM_v1_optimized.sh test
```

#### 终端1 - 自动检测并监控
```
✅ 检测到新的GPU进程!

进程信息:
  容器:      zhen_vllm_dsv3
  宿主机PID: 12345
  进程:      python3

开始监控 Queue 使用情况
[  0s] 采样  1: 15 个队列 (IDs: ...)
```

---

### 方法3: 手动查找PID ⭐

**适合调试和理解原理**

#### 步骤1: 查找容器内的GPU进程
```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

./find_container_gpu_pids.sh zhen_vllm_dsv3
```

输出：
```
╔════════════════════════════════════════════════════════╗
║  查找容器内的GPU进程                                    ║
╚════════════════════════════════════════════════════════╝

容器: zhen_vllm_dsv3

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
检测GPU进程（宿主机PID）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[进程 1]
  宿主机PID: 12345
  进程名:    python3
  完整命令:  python3 -m vllm.entrypoints.openai.api_server...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
如何监控这些进程:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 方法1: 使用queue_monitor
sudo ./queue_monitor 12345 60 10

# 方法2: 使用auto_monitor.sh
./auto_monitor.sh --container zhen_vllm_dsv3

# 方法3: 快速查看Queue信息
sudo ./get_queue_info 12345
```

#### 步骤2: 使用获取的PID监控
```bash
# 使用上面显示的PID
sudo ./queue_monitor 12345 60 10
```

---

## 🎯 你的具体场景

### 场景说明
- **容器名**: `zhen_vllm_dsv3`
- **工作目录**: `/data/code/rampup_doc/vLLM_test/scripts`
- **启动命令**: `./run_vLLM_v1_optimized.sh test`

### 推荐操作步骤

#### 方案A: 最简单（推荐）⭐⭐⭐

```bash
# 终端1 - 宿主机
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
./auto_monitor.sh --container zhen_vllm_dsv3 --duration 120 --interval 10

# 终端2 - 容器内（如果还未启动）
docker exec -it zhen_vllm_dsv3 bash
cd /data/code/rampup_doc/vLLM_test/scripts
./run_vLLM_v1_optimized.sh test
```

#### 方案B: 等待模式

```bash
# 终端1 - 宿主机
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
./watch_docker_gpu.sh zhen_vllm_dsv3

# 终端2 - 容器内
docker exec -it zhen_vllm_dsv3 bash
cd /data/code/rampup_doc/vLLM_test/scripts
./run_vLLM_v1_optimized.sh test
```

---

## 🔍 常见问题

### Q1: 为什么 `watch_new_gpu.sh` 检测不到？

**A**: `watch_new_gpu.sh` 设计用于检测**宿主机上**的新GPU进程。Docker容器内的进程虽然在宿主机有PID，但：
1. 检测逻辑没有考虑容器映射
2. 需要通过容器ID来过滤进程
3. vLLM启动较慢，可能超过默认检测时间

**解决**: 使用专门的 `watch_docker_gpu.sh` 或 `auto_monitor.sh --container`

---

### Q2: 容器内的进程已经运行，怎么监控？

**A**: 使用查找脚本：

```bash
# 1. 查找GPU进程
./find_container_gpu_pids.sh zhen_vllm_dsv3

# 2. 复制显示的PID（假设是12345）
sudo ./queue_monitor 12345 60 10

# 或者直接用
./auto_monitor.sh --container zhen_vllm_dsv3
```

---

### Q3: vLLM启动很慢，检测需要多久？

**A**: vLLM通常需要：
- **模型加载**: 30-60秒
- **GPU初始化**: 10-20秒
- **队列创建**: 5-10秒

`watch_docker_gpu.sh` 会等待最多5分钟，并且会在检测到进程后等待5秒初始化。

---

### Q4: 多个vLLM进程怎么办？

**A**: 使用交互模式：

```bash
./auto_monitor.sh --container zhen_vllm_dsv3
```

会列出所有GPU进程让你选择。

---

### Q5: 如何监控更长时间？

**A**: 自定义参数：

```bash
# 监控300秒（5分钟），每10秒采样
./auto_monitor.sh --container zhen_vllm_dsv3 --duration 300 --interval 10
```

---

## 📊 工具对比

| 工具 | 适用场景 | 是否需要PID | 容器支持 | 难度 |
|------|---------|------------|---------|------|
| `watch_new_gpu.sh` | 宿主机新进程 | ❌ | ❌ 不支持 | ⭐ |
| `watch_docker_gpu.sh` | 容器新进程 | ❌ | ✅ 专门支持 | ⭐ |
| `auto_monitor.sh --container` | 容器已运行进程 | ❌ | ✅ 专门支持 | ⭐ 推荐 |
| `find_container_gpu_pids.sh` | 查找容器进程 | 输出PID | ✅ | ⭐ |
| `queue_monitor <pid>` | 已知PID | ✅ 需要 | ⚠️ 手动 | ⭐⭐ |

---

## 💡 最佳实践

### 日常开发（vLLM测试）

```bash
# 一行命令搞定
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
./auto_monitor.sh --container zhen_vllm_dsv3 --duration 120 --interval 10
```

然后在容器内启动测试：
```bash
docker exec -it zhen_vllm_dsv3 bash
cd /data/code/rampup_doc/vLLM_test/scripts
./run_vLLM_v1_optimized.sh test
```

---

### 持续监控（循环）

```bash
while true; do
    echo "========== 新一轮监控 =========="
    ./auto_monitor.sh --container zhen_vllm_dsv3 --duration 60 --interval 5
    echo ""
    echo "按Enter继续下一轮，Ctrl+C退出"
    read
done
```

---

### 监控并保存日志

```bash
timestamp=$(date +%Y%m%d_%H%M%S)
./auto_monitor.sh --container zhen_vllm_dsv3 --duration 120 --interval 10 \
    | tee "vllm_monitor_${timestamp}.log"
```

---

## 🎓 理解原理

### Docker容器内进程的PID映射

```
容器内视角:
  PID: 123 (python3进程)

宿主机视角:
  PID: 12345 (同一个进程)
```

**KFD监控需要使用宿主机PID！**

### 如何找到映射关系

```bash
# 方法1: 通过lsof
sudo lsof -t /dev/kfd  # 列出所有使用KFD的进程（宿主机PID）

# 方法2: 通过docker ps --filter
docker ps --filter "pid=12345" --format "{{.Names}}"  # 查看PID属于哪个容器

# 方法3: 我们的脚本自动化了这个过程
./find_container_gpu_pids.sh zhen_vllm_dsv3
```

---

## ✅ 快速参考

### 最常用的3个命令

```bash
# 1. 已运行的vLLM - 交互选择
./auto_monitor.sh --container zhen_vllm_dsv3

# 2. 等待新启动的vLLM
./watch_docker_gpu.sh zhen_vllm_dsv3

# 3. 查找当前GPU进程
./find_container_gpu_pids.sh zhen_vllm_dsv3
```

---

## 🔗 相关文档

- `MONITOR_WITHOUT_PID.md` - 无需PID的监控方法
- `README.md` - 完整代码目录说明
- `../QUICKSTART_CPP_POC.md` - 快速开始指南

---

**最后更新**: 2026-02-05  
**测试容器**: zhen_vllm_dsv3  
**状态**: ✅ 已验证
