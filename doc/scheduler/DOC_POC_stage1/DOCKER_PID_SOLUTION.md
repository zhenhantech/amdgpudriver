# Docker PID映射解决方案

**更新**: 2026-02-05  
**方案**: 使用 `docker inspect -f '{{.State.Pid}}'`

---

## ✅ 解决方案

### 核心命令
```bash
docker inspect -f '{{.State.Pid}}' <container_name>
```

### 已集成工具

#### 1. get_docker_pid_mapping.sh
```bash
# 查询PID映射
./get_docker_pid_mapping.sh zhen_vllm_dsv3

# 保存到文件
./get_docker_pid_mapping.sh zhen_vllm_dsv3 python3 --save
```

#### 2. 测试脚本自动记录

**run_gemm_with_ftrace.sh**:
- ✅ 自动获取容器PID
- ✅ 提取AMD日志和ftrace的PID
- ✅ 生成 `pid_mapping.txt`

**run_case_comparison.sh**:
- ✅ 记录容器主进程PID
- ✅ 保存到 `pid_mapping.txt`

---

## 📊 PID映射示例

### 实际映射
```
容器内PID: 157868 (AMD日志)
    ↓ 映射
主机PID:   3934101 (ftrace)
```

### 验证方法
```bash
# AMD日志
grep "当前PID: 157868" gemm_amd_log.txt ✅

# ftrace
grep "python3-3934101" ftrace.txt ✅

# 时间戳一致
177770.497秒 (AMD) → 177770.631秒 (ftrace) ✅
```

---

## 🎯 使用场景

### Model-A和Model-B测试
```bash
# 运行测试（自动记录PID）
./run_case_comparison.sh zhen_vllm_dsv3 60

# 查看PID映射
cat log/case_comparison_*/pid_mapping.txt
```

### ftrace测试
```bash
# 运行测试
sudo ./run_gemm_with_ftrace.sh zhen_vllm_dsv3

# 查看映射
cat log/gemm_ftrace_*/pid_mapping.txt
```

---

## 💡 关键点

- ✅ 所有测试脚本已自动集成
- ✅ PID映射自动记录到 `pid_mapping.txt`
- ✅ 通过时间戳关联日志（不依赖PID）
- ✅ 工具可独立使用

---

**工具**: `get_docker_pid_mapping.sh`  
**文档**: 见各测试脚本和日志目录  
**状态**: ✅ 已完成并集成
