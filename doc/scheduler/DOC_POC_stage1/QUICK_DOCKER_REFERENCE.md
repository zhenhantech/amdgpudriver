# Docker容器内监控 - 快速参考卡

**容器**: zhen_vllm_dsv3  
**工具目录**: `/data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code`

---

## 🚀 最简单的方法（推荐）

### 终端1 - 启动监控
```bash
docker exec -it zhen_vllm_dsv3 bash
cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
./watch_gpu_in_docker.sh
```

### 终端2 - 启动vLLM
```bash
docker exec -it zhen_vllm_dsv3 bash
cd /data/code/rampup_doc/vLLM_test/scripts
./run_vLLM_v1_optimized.sh test
```

**终端1会自动检测并开始监控！**

---

## 📋 常用命令

```bash
# 进入容器和工具目录
docker exec -it zhen_vllm_dsv3 bash
cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 列出当前GPU进程
./list_gpu_processes.sh

# 监控指定PID（假设PID是12345）
./queue_monitor 12345 60 10

# 快速查看Queue信息
./get_queue_info 12345

# 一行命令：查找并监控第一个GPU进程
PID=$(lsof -t /dev/kfd | head -1) && ./queue_monitor $PID 60 10
```

---

## 🔧 首次使用

```bash
# 1. 进入容器
docker exec -it zhen_vllm_dsv3 bash

# 2. 进入工具目录
cd /data/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 3. 编译工具（只需一次）
make clean && make all

# 4. 启动监控
./watch_gpu_in_docker.sh
```

---

## 📚 完整文档

- **DOCKER_INSIDE_GUIDE.md** - 容器内监控完整指南
- **DOCKER_MONITORING_GUIDE.md** - 宿主机监控Docker指南
- **README.md** - 所有工具说明

---

## ⚡ 故障排查

```bash
# 编译工具
make clean && make all

# 检查GPU设备
ls -l /dev/kfd /dev/dri

# 查找GPU进程
lsof /dev/kfd
ps aux | grep vllm

# 检查是否在容器内
ls /.dockerenv
```

---

**更新**: 2026-02-05
