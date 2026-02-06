# Docker PID映射完全指南

**日期**: 2026-02-05  
**适用场景**: 测试Model-A和Model-B时的PID映射

---

## 🎯 问题背景

### Docker PID Namespace

Docker容器有独立的PID空间，导致：
- **AMD日志**: 显示容器内PID
- **ftrace**: 显示主机PID  
- **需要映射**: 才能关联两个日志

---

## ✅ 推荐方案：docker inspect

### 核心命令

```bash
# 获取容器主进程的主机PID
docker inspect -f '{{.State.Pid}}' <container_name>
```

### 示例

```bash
$ docker inspect -f '{{.State.Pid}}' zhen_vllm_dsv3
7064
```

---

## 🛠️ 使用工具

### 工具1: get_docker_pid_mapping.sh ⭐⭐⭐⭐⭐（推荐）

**功能**: 一键获取完整的PID映射信息

**使用**:
```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 基本用法
./get_docker_pid_mapping.sh zhen_vllm_dsv3

# 指定进程名
./get_docker_pid_mapping.sh zhen_vllm_dsv3 python3

# 保存映射到文件
./get_docker_pid_mapping.sh zhen_vllm_dsv3 python3 --save
```

**输出示例**:
```
╔════════════════════════════════════════════════════════════════════╗
║  Docker PID映射查询                                                 ║
╚════════════════════════════════════════════════════════════════════╝

容器: zhen_vllm_dsv3
进程: python3

━━━ 1. 容器主进程 ━━━
主机PID: 7064

━━━ 2. 查找 python3 进程 ━━━
主机PID  | 命令
─────────┼─────────────────────────────────────────────────
3934101  | python3 test_gemm_mini.py

━━━ 3. 容器内PID（对比用）━━━
容器内PID | 命令
──────────┼─────────────────────────────────────────────────
157868    | /usr/bin/python3
```

---

## 📊 自动化集成

### 已更新的脚本

所有测试脚本已自动集成PID映射：

#### 1. run_gemm_with_ftrace.sh

**自动记录**:
- 容器主进程PID
- 测试进程的容器内PID (从AMD日志)
- 测试进程的主机PID (从ftrace)
- 生成 `pid_mapping.txt`

**使用**:
```bash
sudo ./run_gemm_with_ftrace.sh zhen_vllm_dsv3

# PID映射自动保存在:
cat log/gemm_ftrace_*/pid_mapping.txt
```

#### 2. run_case_comparison.sh

**自动记录**:
- 容器信息
- 每个Case的PID映射
- 保存到 `log/case_comparison_*/pid_mapping.txt`

**使用**:
```bash
./run_case_comparison.sh zhen_vllm_dsv3 60

# 查看映射
cat log/case_comparison_*/pid_mapping.txt
```

---

## 🔍 手动查找PID映射

### 方法1: docker inspect + docker top

```bash
# 1. 容器主进程
MAIN_PID=$(docker inspect -f '{{.State.Pid}}' zhen_vllm_dsv3)
echo "容器主进程(主机): $MAIN_PID"

# 2. 查找特定进程
docker top zhen_vllm_dsv3 | grep python3
```

### 方法2: 对比ps输出

```bash
# 在容器内
docker exec zhen_vllm_dsv3 ps aux | grep python3

# 在主机
ps aux | grep "python3.*zhen_vllm_dsv3"
```

### 方法3: 通过命令行特征匹配

```bash
# 容器内运行测试并记录PID
docker exec zhen_vllm_dsv3 bash -c 'python3 test.py & echo $!'

# 在主机查找对应进程
docker top zhen_vllm_dsv3 | grep test.py
```

---

## 📝 测试Model-A和Model-B的完整流程

### 步骤1: 运行测试（自动记录PID）

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 方式A: 使用ftrace
sudo ./run_gemm_with_ftrace.sh zhen_vllm_dsv3

# 方式B: Case对比
./run_case_comparison.sh zhen_vllm_dsv3 60
```

### 步骤2: 查看PID映射

```bash
# ftrace测试
LOG_DIR=$(ls -dt log/gemm_ftrace_* | head -1)
cat $LOG_DIR/pid_mapping.txt

# Case对比测试
LOG_DIR=$(ls -dt log/case_comparison_* | head -1)
cat $LOG_DIR/pid_mapping.txt
```

### 步骤3: 关联日志

```bash
cd $LOG_DIR

# 提取关键信息
echo "=== PID映射 ==="
cat pid_mapping.txt

echo ""
echo "=== AMD日志示例 ==="
grep "当前PID:" gemm_amd_log.txt || grep "当前PID:" case_*.log

echo ""
echo "=== ftrace示例 ==="
grep "python3-" ftrace.txt | head -3
```

---

## 💡 实际案例

### 案例：GEMM测试的PID映射

**测试运行**:
```bash
sudo ./run_gemm_with_ftrace.sh zhen_vllm_dsv3
```

**生成的pid_mapping.txt**:
```
# PID映射信息
# 生成时间: Wed Feb 5 14:35:55 CST 2026

容器名称: zhen_vllm_dsv3
容器主进程(主机PID): 7064

# 从AMD日志提取的容器内PID
容器内PID (AMD日志): 157868

# 从ftrace提取的主机PID
主机PID (ftrace): 3934101

# PID映射关系
容器内 157868 → 主机 3934101
```

**验证**:
```bash
# AMD日志
grep "当前PID: 157868" gemm_amd_log.txt

# ftrace
grep "python3-3934101" ftrace.txt | head -3
```

---

## 🎯 关键命令速查

### 基本查询

```bash
# 容器主进程PID
docker inspect -f '{{.State.Pid}}' zhen_vllm_dsv3

# 查找python进程
docker top zhen_vllm_dsv3 | grep python3

# 容器内PID
docker exec zhen_vllm_dsv3 ps aux | grep python3
```

### 实时监控

```bash
# 持续监控进程
watch -n 1 'docker top zhen_vllm_dsv3 | grep python3'

# ftrace实时过滤（需要主机PID）
tail -f /sys/kernel/debug/tracing/trace | grep "python3-<主机PID>"
```

### 日志分析

```bash
# 从AMD日志提取容器内PID
grep "当前PID:" gemm_amd_log.txt | awk '{print $NF}'

# 从ftrace提取主机PID
grep "python3-" ftrace.txt | head -1 | grep -o "python3-[0-9]*" | cut -d'-' -f2
```

---

## 🔧 故障排查

### 问题1: docker inspect返回0

**原因**: 容器未运行

**解决**:
```bash
# 检查容器状态
docker ps -a | grep zhen_vllm_dsv3

# 启动容器
docker start zhen_vllm_dsv3
```

### 问题2: docker top没有进程

**原因**: 测试还未开始

**解决**:
```bash
# 在测试运行时查询
# 或使用 --save 参数在测试后保存
```

### 问题3: PID不匹配

**原因**: 这是正常的！Docker namespace隔离

**解决**: 使用时间戳关联日志，不要依赖PID匹配

---

## 📚 相关文档

| 文档 | 说明 |
|------|------|
| `PID_MAPPING_GUIDE.md` | PID映射基础指南 |
| `get_docker_pid_mapping.sh` | PID映射查询工具 |
| `run_gemm_with_ftrace.sh` | ftrace测试（自动记录PID） |
| `run_case_comparison.sh` | Case对比（自动记录PID） |
| `FTRACE_ANALYSIS_GUIDE.md` | ftrace分析指南 |

---

## ✅ 最佳实践

### 1. 使用自动化工具

**推荐**: 让脚本自动记录PID映射
```bash
# 不要手动查询PID
# 使用自动化脚本
sudo ./run_gemm_with_ftrace.sh zhen_vllm_dsv3
```

### 2. 测试前验证容器

```bash
# 确保容器运行
docker ps | grep zhen_vllm_dsv3

# 获取容器PID
docker inspect -f '{{.State.Pid}}' zhen_vllm_dsv3
```

### 3. 保存PID映射

```bash
# 使用工具保存
./get_docker_pid_mapping.sh zhen_vllm_dsv3 python3 --save

# 或在测试时自动保存（已集成）
```

### 4. 通过时间戳关联

**不要依赖PID匹配，使用时间戳！**

```python
# AMD日志: 177770.497秒
# ftrace:   177770.631秒
# → 是同一个操作（133ms延迟）
```

---

## 🎉 总结

| 方面 | 方案 |
|------|------|
| **推荐方法** | `docker inspect -f '{{.State.Pid}}'` |
| **自动化工具** | `get_docker_pid_mapping.sh` |
| **集成支持** | 所有测试脚本已自动记录 |
| **日志关联** | 使用时间戳，不依赖PID |
| **验证方法** | 检查 `pid_mapping.txt` |

**核心要点**: 
- ✅ 使用 `docker inspect` 获取PID映射
- ✅ 测试脚本自动记录映射关系
- ✅ 通过时间戳关联AMD日志和ftrace
- ✅ `pid_mapping.txt` 包含所有映射信息

---

**更新**: 2026-02-05  
**维护**: AI Assistant  
**状态**: ✅ 已集成到所有测试脚本
