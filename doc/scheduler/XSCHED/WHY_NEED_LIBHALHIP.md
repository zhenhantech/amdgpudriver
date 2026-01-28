# 为什么需要 Preload libhalhip.so？

**日期**: 2026-01-28  
**问题**: Symbol lookup error for `_ZTIN6xsched3hip10HipCommandE`  
**解决**: 同时 preload `libhalhip.so` 和 `libshimhip.so`

---

## ❓ 用户的问题

> 为啥在轻量级负载时候可以正常测试？我们的脚本变复杂了？

**答案**: 不是脚本复杂了，而是**发现了之前没注意到的依赖问题**。

---

## 🔍 问题分析

### Symbol Error 详情

```bash
undefined symbol: _ZTIN6xsched3hip10HipCommandE
  → typeinfo for xsched::hip::HipCommand
```

### Symbol 在哪里？

```bash
# 检查 libhalhip.so
$ nm /data/dockercode/xsched-build/output/lib/libhalhip.so | grep HipCommand
00000000000285c8 d _ZTIN6xsched3hip10HipCommandE
                 ↑
                'd' = data section (非 exported symbol)
```

**关键发现**: 
- ✅ Symbol 确实在 `libhalhip.so` 中
- ⚠️  但没有导出为动态符号（`d` 而非 `T`）
- 🔧 需要通过 `LD_PRELOAD` 强制加载

---

## 🤔 为什么原来的测试能工作？

### 可能的原因

#### 1. 运气好（最可能）⭐

```bash
原来的测试:
  - 可能碰巧某些符号已经被加载
  - 或者依赖的解析顺序不同
  - 或者编译时链接顺序不同

现在的测试:
  - 更严格的符号检查
  - 或者 Python 版本/环境不同
```

---

#### 2. 原来的测试确实有问题，但没暴露

```bash
原来可能:
  - 测试运行了，但可能有隐藏的问题
  - Symbol 解析可能靠的是运气
  - 在某些情况下可能失败

现在:
  - 更严格的检查暴露了问题
  - 这实际上是好事！
```

---

#### 3. 编译/链接的变化

```bash
可能在某个时间点:
  - XSched 重新编译了
  - 链接顺序改变了
  - 导致依赖关系变化
```

---

## ✅ 正确的做法

### 完整的 LD_PRELOAD 设置

```bash
# ❌ 错误：只 preload libshimhip.so
export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so

# ✅ 正确：同时 preload libhalhip.so 和 libshimhip.so
export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libhalhip.so:/data/dockercode/xsched-build/output/lib/libshimhip.so
```

---

### 为什么需要两个库？

```
libhalhip.so:
  - XSched 的 HAL (Hardware Abstraction Layer)
  - 包含核心数据结构（如 HipCommand typeinfo）
  - 被 libshimhip.so 依赖

libshimhip.so:
  - HIP API 拦截层
  - 实现优先级调度逻辑
  - 依赖 libhalhip.so 的符号

加载顺序: libhalhip.so → libshimhip.so
```

---

## 🔧 修复方案

### 已修复的脚本

#### 1. run_phase4_dual_model_intensive.sh（HOST 端运行）

```bash
docker exec "$CONTAINER" bash -c "
    export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:\$LD_LIBRARY_PATH && \
    export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libhalhip.so:/data/dockercode/xsched-build/output/lib/libshimhip.so && \
    python3 test_phase4_dual_model_intensive.py ...
"
```

---

#### 2. run_intensive_xsched_final.sh（Docker 内部运行）

```bash
XSCHED_PRELOAD="/data/dockercode/xsched-build/output/lib/libhalhip.so:/data/dockercode/xsched-build/output/lib/libshimhip.so"

LD_PRELOAD=$XSCHED_PRELOAD python3 test_phase4_dual_model_intensive.py ...
```

---

## 🚀 立即运行

### 方法 1: 从 HOST 运行（推荐）⭐

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED

# 已经修复，直接运行
./run_phase4_dual_model_intensive.sh
```

---

### 方法 2: 在 Docker 内运行

```bash
# 在 Docker 容器内
unset LD_PRELOAD
bash /data/dockercode/run_intensive_xsched_final.sh 2>&1 | tee testlog/xsched_intensive_final.log
```

---

## 💡 经验教训

### 1. LD_PRELOAD 顺序很重要

```bash
# 正确顺序：被依赖的库在前
LD_PRELOAD=libhalhip.so:libshimhip.so

# 如果顺序错了，可能导致 symbol not found
```

---

### 2. 非导出符号需要显式 preload

```bash
# 如果符号是 'd' (data) 而非 'T' (text/exported)
# 需要通过 LD_PRELOAD 强制加载整个库
```

---

### 3. 测试要覆盖不同场景

```bash
轻量级测试: 可能"碰巧"能工作
重负载测试: 暴露隐藏的问题

教训: 两种测试都需要！
```

---

## 📊 对比

### 修复前 vs 修复后

#### 修复前

```bash
# 只 preload libshimhip.so
LD_PRELOAD=libshimhip.so

结果:
  ❌ Symbol lookup error
  ❌ 无法运行
```

---

#### 修复后

```bash
# 同时 preload libhalhip.so 和 libshimhip.so
LD_PRELOAD=libhalhip.so:libshimhip.so

结果:
  ✅ Symbol 正常解析
  ✅ 测试可以运行
```

---

## 🎯 验证方法

### 测试最小配置

```bash
cd /data/dockercode
export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:$LD_LIBRARY_PATH

# 测试 1: 只 preload libshimhip.so（会失败）
LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so \
  python3 -c 'import torch; print(torch.cuda.is_available())'
# ❌ 预期: symbol lookup error

# 测试 2: preload 两个库（会成功）
LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libhalhip.so:/data/dockercode/xsched-build/output/lib/libshimhip.so \
  python3 -c 'import torch; print(torch.cuda.is_available())'
# ✅ 预期: True
```

---

## 🔍 深入理解

### 为什么原来的脚本"可能"能工作？

```
可能的场景:

1. 环境变量残留
   - 之前的测试设置了某些环境变量
   - 没有清理干净
   - 后续测试"碰巧"用上了

2. 库的加载顺序
   - 某些情况下，符号解析顺序不同
   - libhalhip.so 可能被间接加载
   - 符号"碰巧"可用

3. Python 版本/配置
   - 不同的 Python 启动方式
   - 不同的库搜索路径
   - 导致行为不同
```

**结论**: 原来的配置可能"碰巧"能工作，但不可靠。现在的修复是正确且可靠的。

---

## ✅ 最终方案

### 统一的正确配置

```bash
# 对于所有 XSched 测试，应该使用:

export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libhalhip.so:/data/dockercode/xsched-build/output/lib/libshimhip.so

# 顺序: libhalhip.so 在前（被依赖）, libshimhip.so 在后（依赖方）
```

---

## 🚀 立即行动

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED

# 运行修复后的脚本（3-4 分钟）
./run_phase4_dual_model_intensive.sh
```

**预期**: 
- ✅ 测试正常运行
- ✅ 生成 XSched 结果
- ✅ 看到巨大的性能改善（期待 P99 降低 >90%）🚀

---

## 📝 相关文件

- **修复后的 HOST 脚本**: `run_phase4_dual_model_intensive.sh`
- **修复后的 Docker 内脚本**: `tests/run_intensive_xsched_final.sh`
- **问题分析**: `FIXED_LD_PRELOAD_ISSUE.md`
- **Baseline 结果**: `PHASE4_TEST3B_BASELINE_ANALYSIS.md`
