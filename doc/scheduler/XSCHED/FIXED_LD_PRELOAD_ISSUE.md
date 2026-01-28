# LD_PRELOAD 问题修复说明

**日期**: 2026-01-28  
**问题**: `export LD_PRELOAD` 导致所有命令尝试加载 `libshimhip.so`  
**状态**: ✅ 已修复

---

## 🐛 问题描述

### 原始错误

```bash
bash: symbol lookup error: /data/dockercode/xsched-build/output/lib/libshimhip.so: 
  undefined symbol: _ZTIN6xsched3hip10HipCommandE

grep: symbol lookup error: /data/dockercode/xsched-build/output/lib/libshimhip.so: 
  undefined symbol: _ZTIN6xsched3hip10HipCommandE
```

---

## 🔍 根本原因

### 错误的做法

```bash
#!/bin/bash
# ❌ 错误：export LD_PRELOAD 会影响所有命令
export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so

# 这会导致所有后续命令都尝试加载 libshimhip.so
ldd ...       # ❌ ldd 也尝试加载，导致错误
grep ...      # ❌ grep 也尝试加载，导致错误
bash ...      # ❌ bash 也尝试加载，导致错误
python3 ...   # ✅ 只有 python3 需要加载
```

---

### 正确的做法

```bash
#!/bin/bash
# ✅ 正确：只在需要的命令前设置 LD_PRELOAD
XSCHED_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so

# 普通命令不受影响
ldd ...       # ✅ 正常执行
grep ...      # ✅ 正常执行
bash ...      # ✅ 正常执行

# 只在运行 Python 时设置
LD_PRELOAD=$XSCHED_PRELOAD python3 ...  # ✅ Python 加载 XSched
```

---

## ✅ 修复方案

### 修复后的脚本

已创建 `run_intensive_xsched_only_fixed.sh`：

```bash
#!/bin/bash
set -e
cd /data/dockercode

# LD_LIBRARY_PATH 可以 export（标准库路径）
export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:$LD_LIBRARY_PATH

# LD_PRELOAD 不能 export，只保存到变量
XSCHED_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so

# 验证库（不使用 LD_PRELOAD）
ldd /data/dockercode/xsched-build/output/lib/libshimhip.so | grep -E "libpreempt|libhalhip"

# 测试 PyTorch（使用 LD_PRELOAD）
LD_PRELOAD=$XSCHED_PRELOAD python3 -c 'import torch; print(torch.cuda.is_available())'

# 运行测试（使用 LD_PRELOAD）
LD_PRELOAD=$XSCHED_PRELOAD python3 test_phase4_dual_model_intensive.py \
  --duration 180 \
  --output /data/dockercode/test_results_phase4/xsched_intensive_result.json
```

---

## 🚀 使用方法

### 在 Docker 内执行（推荐）

```bash
# 先清除任何已设置的 LD_PRELOAD
unset LD_PRELOAD

# 运行修复后的脚本
bash /data/dockercode/run_intensive_xsched_only_fixed.sh 2>&1 | tee testlog/xsched_intensive_fixed.log
```

---

## 📊 预期输出

```
========================================================================
XSched 高负载测试（Docker 内部执行）
========================================================================

环境变量:
  LD_LIBRARY_PATH: /data/dockercode/xsched-build/output/lib:...
  XSCHED_PRELOAD: /data/dockercode/xsched-build/output/lib/libshimhip.so (只在 Python 中使用)

验证库依赖...
	libpreempt.so => /data/dockercode/xsched-build/output/lib/libpreempt.so
	libhalhip.so => /data/dockercode/xsched-build/output/lib/libhalhip.so
  ✅ 库依赖正常

测试基本 PyTorch 功能（带 XSched）...
  PyTorch: 2.6.0+rocm6.4.0
  CUDA: True
  ✅ PyTorch + XSched 正常

========================================================================
开始高负载测试 (20 req/s, batch=1024, 180s)
========================================================================

[测试运行中，约 3 分钟...]
```

---

## 💡 关键学习点

### 1. LD_PRELOAD 的作用域

```
export LD_PRELOAD=xxx
  → 所有子进程都会继承
  → 导致不需要的命令也尝试加载库
  → 容易出错

LD_PRELOAD=xxx command
  → 只对该命令有效
  → 其他命令不受影响
  → 更安全、更精确
```

---

### 2. LD_LIBRARY_PATH vs LD_PRELOAD

```
LD_LIBRARY_PATH:
  ✅ 可以 export
  ✅ 用于查找动态库
  ✅ 影响所有命令通常是安全的

LD_PRELOAD:
  ⚠️  不应该 export
  ⚠️  强制加载特定库
  ⚠️  只应用于特定命令
```

---

### 3. 调试方法

```bash
# 检查哪个命令出错
bash -x script.sh 2>&1 | grep "symbol lookup error" -B 5

# 查看当前环境变量
echo $LD_PRELOAD

# 清除环境变量
unset LD_PRELOAD

# 测试单个命令
LD_PRELOAD=/path/to/lib.so command args
```

---

## 🎯 立即执行

### 在 Docker 内运行

```bash
# 在 Docker 容器内执行以下命令：
cd /data/dockercode/xsched

# 清除环境变量（如果之前设置过）
unset LD_PRELOAD

# 运行修复后的脚本
bash /data/dockercode/run_intensive_xsched_only_fixed.sh 2>&1 | tee testlog/xsched_intensive_fixed.log
```

**预计时间**: 3-4 分钟  
**预期结果**: XSched P99 大幅降低（期待 <50ms，改善 >90%）🚀

---

## 📝 相关文件

- **修复后的脚本**: `/data/dockercode/run_intensive_xsched_only_fixed.sh`
- **原始脚本**: `/data/dockercode/run_intensive_xsched_only.sh` (有问题)
- **测试脚本**: `/data/dockercode/test_phase4_dual_model_intensive.py`
- **结果文件**: `/data/dockercode/test_results_phase4/xsched_intensive_result.json`

---

## 🔧 如果还有问题

### 检查库依赖

```bash
ldd /data/dockercode/xsched-build/output/lib/libshimhip.so
```

应该看到：
```
libpreempt.so => /data/dockercode/xsched-build/output/lib/libpreempt.so
libhalhip.so => /data/dockercode/xsched-build/output/lib/libhalhip.so
```

如果看到 `not found`，需要设置 `LD_LIBRARY_PATH`。

---

### 测试最小配置

```bash
# 测试 Python + XSched
cd /data/dockercode
export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:$LD_LIBRARY_PATH
LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so \
  python3 -c 'import torch; print(torch.cuda.is_available())'
```

如果成功输出 `True`，说明环境正常。

---

## ✅ 状态

- [x] 问题诊断完成
- [x] 修复脚本已创建
- [x] 脚本已复制到容器
- [ ] 等待用户运行测试
- [ ] 等待分析结果

**下一步**: 在 Docker 内运行修复后的脚本 🚀
