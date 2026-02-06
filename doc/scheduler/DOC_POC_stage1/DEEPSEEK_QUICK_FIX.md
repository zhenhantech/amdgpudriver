# DeepSeek测试快速修复指南

## 🔧 问题: vLLM启动失败

### 错误信息
```
pydantic_core._pydantic_core.ValidationError: 1 validation error for AttentionConfig
  Value error, Invalid value 'ROCM_FLASH' for VLLM_ATTENTION_BACKEND.
```

### 原因分析
容器中设置了环境变量 `VLLM_ATTENTION_BACKEND=ROCM_FLASH`，但当前vLLM版本不支持这个值。

**有效的backend选项包括**:
- `ROCM_ATTN` ✅
- `FLASH_ATTN` ✅
- `FLASHINFER` ✅
- `ROCM_AITER_FA` ✅
- 等等...

但**不包括** `ROCM_FLASH` ❌

---

## ✅ 解决方案

### 方案1: 使用简化PyTorch测试（推荐）⭐

**优点**:
- 最可靠，不依赖vLLM配置
- 直接使用PyTorch多GPU计算
- 同样能验证Queue使用模式

**步骤**:

1. **将测试脚本复制到容器**:
```bash
docker cp test_deepseek_simple.py zhen_vllm_dsv3:/workspace/
```

2. **运行测试**:
```bash
sudo ./run_deepseek_with_ftrace.sh zhen_vllm_dsv3 120
```

脚本会自动检测到 `test_deepseek_simple.py` 并使用它。

---

### 方案2: 修复vLLM环境变量

如果你确实需要使用vLLM：

**步骤1**: 进入容器检查环境变量
```bash
docker exec -it zhen_vllm_dsv3 bash
env | grep VLLM
```

**步骤2**: 修改或删除冲突的环境变量
```bash
# 临时修复（当前会话）
unset VLLM_ATTENTION_BACKEND

# 或者设置为有效值
export VLLM_ATTENTION_BACKEND=ROCM_ATTN
```

**步骤3**: 永久修复（修改容器配置）
```bash
# 找到容器的启动脚本或环境配置文件
# 通常在 ~/.bashrc, /etc/environment, 或Docker启动参数中
```

**步骤4**: 重启容器
```bash
docker restart zhen_vllm_dsv3
```

---

### 方案3: 手动运行简化测试（调试用）

如果自动脚本有问题，可以手动运行：

**步骤1**: 启动ftrace（host上，需要sudo）
```bash
sudo su
cd /sys/kernel/debug/tracing
echo 0 > tracing_on
echo > trace
echo 20480 > buffer_size_kb
echo function > current_tracer
echo :mod:amdgpu > set_ftrace_filter
echo 1 > tracing_on
```

**步骤2**: 在容器内运行测试
```bash
docker exec zhen_vllm_dsv3 bash -c "
    cd /workspace
    export AMD_LOG_LEVEL=3
    export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    python3 test_deepseek_simple.py --duration 120 --gpus 8
" 2>&1 | tee deepseek_log.txt
```

**步骤3**: 保存ftrace（host上）
```bash
cat /sys/kernel/debug/tracing/trace > ftrace_log.txt
echo 0 > /sys/kernel/debug/tracing/tracing_on
```

---

## 📊 测试脚本说明

### `test_deepseek_simple.py` 参数

```bash
python3 test_deepseek_simple.py --help
```

**参数**:
- `--duration SECONDS`: 测试时长（默认120秒）
- `--gpus N`: 使用的GPU数量（默认8）
- `--single-process`: 单进程模式（更简单，调试友好）

**示例**:
```bash
# 120秒，8 GPU，多进程模式（推荐）
python3 test_deepseek_simple.py --duration 120 --gpus 8

# 60秒，4 GPU，单进程模式（调试）
python3 test_deepseek_simple.py --duration 60 --gpus 4 --single-process
```

### 两种运行模式对比

| 模式          | 命令参数           | 特点                          | 适用场景          |
|---------------|--------------------|------------------------------ |-------------------|
| 多进程并行    | （默认）           | 每个GPU一个进程，更真实       | 模拟真实DeepSeek  |
| 单进程串行    | `--single-process` | 一个进程轮询所有GPU，更简单   | 快速调试          |

---

## 🔍 验证测试是否成功

### 1. 检查AMD日志
```bash
# 应该看到多个GPU的代码加载
grep "Using native code object for device" deepseek_amd_log.txt | wc -l
# 预期: 应该是8的倍数（8个GPU）

# 应该看到Queue使用
grep "HWq=0x" deepseek_amd_log.txt | wc -l
# 预期: >0
```

### 2. 检查ftrace日志
```bash
# 应该看到KFD函数调用
grep -c "kfd" ftrace.txt
# 预期: >100

# 应该看到queue相关操作
grep -c "queue" ftrace.txt
# 预期: >10
```

### 3. 检查Queue数量
```bash
# 查看唯一Queue地址
grep 'HWq=0x' deepseek_amd_log.txt | \
    grep -o 'HWq=0x[0-9a-f]*' | \
    sort -u

# 统计数量
grep 'HWq=0x' deepseek_amd_log.txt | \
    grep -o 'HWq=0x[0-9a-f]*' | \
    sort -u | \
    wc -l
```

**预期结果**:
- **1个Queue**: ✅ 完美！与Case-A/Case-B一致，POC设计完全适用
- **8个Queue**: ⚠️  每个GPU一个Queue，需要批量操作
- **其他数量**: ❓ 需要进一步分析

---

## 🚀 完整测试流程（推荐）

### Step 1: 准备
```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 复制测试脚本到容器
docker cp test_deepseek_simple.py zhen_vllm_dsv3:/workspace/
```

### Step 2: 运行测试
```bash
# 使用sudo运行（ftrace需要）
sudo ./run_deepseek_with_ftrace.sh zhen_vllm_dsv3 120
```

### Step 3: 查看结果
```bash
# 测试会自动运行分析脚本
# 如需重新分析：
./log/deepseek_ftrace_<timestamp>/analyze_deepseek.sh
```

### Step 4: 对比Case-A/Case-B
```bash
# 查看之前的结果
cat ./log/case_comparison_20260205_155247/ANALYSIS_SUMMARY.md

# 对比Queue数量
echo "Case-A Queue数量: 1"
echo "Case-B Queue数量: 1"
echo "DeepSeek Queue数量: ?"
```

---

## 🎯 测试目标回顾

**核心问题**: DeepSeek 3.2（8 GPU）使用几个Hardware Queue？

**可能结果**:

1. **单Queue模型** ✅
   - 即使8个GPU，也只用1个Queue
   - POC设计完全适用，不需要修改

2. **多Queue模型（≤8）** ⚠️
   - 可能每个GPU一个Queue
   - 需要批量suspend/resume
   - 建议使用创新方案（batch_unmap）

3. **大量Queue（>8）** ❌
   - 需要重新评估POC设计
   - 可能需要选择性抢占策略

---

## 📚 相关文档

- [DeepSeek测试指南](DEEPSEEK_TEST_GUIDE.md) - 完整测试文档
- [测试工具总览](TEST_TOOLS_OVERVIEW.md) - 所有测试工具说明
- [Case-A/Case-B分析](log/case_comparison_20260205_155247/ANALYSIS_SUMMARY.md) - 之前的发现

---

## 🤝 故障排查

### Q1: 脚本找不到 test_deepseek_simple.py
**A**: 需要先复制到容器内
```bash
docker cp test_deepseek_simple.py zhen_vllm_dsv3:/workspace/
```

### Q2: AMD日志很少或为空
**A**: 检查AMD_LOG_LEVEL是否生效
```bash
docker exec zhen_vllm_dsv3 bash -c "
    export AMD_LOG_LEVEL=3
    python3 -c 'import torch; print(torch.cuda.is_available())'
"
```

### Q3: GPU数量不对
**A**: 检查HIP_VISIBLE_DEVICES
```bash
docker exec zhen_vllm_dsv3 bash -c "
    export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    python3 -c 'import torch; print(torch.cuda.device_count())'
"
```

### Q4: 权限错误
**A**: 确保使用sudo运行需要ftrace的脚本
```bash
sudo ./run_deepseek_with_ftrace.sh zhen_vllm_dsv3 120
```

---

**创建时间**: 2026-02-05  
**更新时间**: 2026-02-05  
**状态**: ✅ 已修复，ready to test

