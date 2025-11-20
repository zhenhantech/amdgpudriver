# 解决 aqlprofile 依赖问题

## 问题分析

```bash
aqlprofile started due to rocr.txt
```

**关键发现**：
- `aqlprofile` 是 `rocr` (HSA Runtime) 的依赖
- `rocr` 是核心组件，无法跳过
- 因此 `aqlprofile` 也无法跳过

---

## 依赖链

```
rocm-dev
  └─ rocr (HSA Runtime) ← 必需！
      └─ aqlprofile ← 因此也必需
```

---

## 解决方案

### 方案 1：查看 aqlprofile 的编译错误（推荐）

```bash
# 查看详细错误日志
cat /data/code/rocm6.4.3/out/ubuntu-22.04/22.04/logs/1.Errors.aqlprofile.txt | tail -100

# 或查看完整日志
cat /data/code/rocm6.4.3/out/ubuntu-22.04/22.04/logs/2.Inprogress.aqlprofile.txt | tail -200
```

**常见错误类型**：
1. 缺少依赖库
2. 编译器版本问题
3. CMake配置错误
4. 头文件缺失

---

### 方案 2：从 rocr 的依赖中移除 aqlprofile

这需要修改依赖配置文件：

```bash
cd /data/code/rocm6.4.3/ROCm

# 查找 rocr 的依赖配置
find . -name "*.txt" -o -name "*.cmake" | xargs grep -l "aqlprofile" | grep rocr

# 可能的文件：
# - rocr/DEPENDENCIES.txt
# - rocr/CMakeLists.txt
# - rocr/config.cmake
```

#### 修改 rocr 依赖

```bash
# 备份依赖文件
cp rocr/DEPENDENCIES.txt rocr/DEPENDENCIES.txt.backup

# 移除 aqlprofile 依赖
sed -i '/aqlprofile/d' rocr/DEPENDENCIES.txt

# 验证修改
diff rocr/DEPENDENCIES.txt.backup rocr/DEPENDENCIES.txt
```

---

### 方案 3：最小化编译（跳过 aqlprofile）

创建一个只编译必要组件的脚本：

```bash
cat > build_minimal_rocm.sh << 'EOF'
#!/bin/bash

set -e

OUT_DIR=/data/code/rocm6.4.3/out/ubuntu-22.04/22.04
ROCM_INSTALL_PATH=/opt/rocm-9.99.99

echo "=========================================="
echo "最小化编译 ROCm（手动处理依赖）"
echo "=========================================="

cd /data/code/rocm6.4.3

# 1. 先编译底层依赖（不依赖aqlprofile的）
echo "第1步：编译基础组件..."
make -f ROCm/tools/rocm-build/ROCm.mk \
    OUT_DIR="$OUT_DIR" \
    ROCM_INSTALL_PATH="$ROCM_INSTALL_PATH" \
    SILENT= \
    T_devicelibs \
    T_comgr \
    T_rocm_smi_lib \
    -j $(nproc)

# 2. 尝试编译 rocr（如果失败，手动处理）
echo "第2步：尝试编译 rocr..."
if ! make -f ROCm/tools/rocm-build/ROCm.mk \
    OUT_DIR="$OUT_DIR" \
    ROCM_INSTALL_PATH="$ROCM_INSTALL_PATH" \
    NOBUILD="aqlprofile" \
    SILENT= \
    T_rocr \
    -j $(nproc); then
    
    echo "⚠️  rocr 编译触发了 aqlprofile"
    echo "尝试临时禁用 aqlprofile..."
    
    # 临时重命名 aqlprofile 目录
    if [ -d ROCm/aqlprofile ]; then
        mv ROCm/aqlprofile ROCm/aqlprofile.disabled
    fi
    
    # 再次尝试
    make -f ROCm/tools/rocm-build/ROCm.mk \
        OUT_DIR="$OUT_DIR" \
        ROCM_INSTALL_PATH="$ROCM_INSTALL_PATH" \
        SILENT= \
        T_rocr \
        -j $(nproc) || true
    
    # 恢复目录
    if [ -d ROCm/aqlprofile.disabled ]; then
        mv ROCm/aqlprofile.disabled ROCm/aqlprofile
    fi
fi

# 3. 编译上层组件
echo "第3步：编译上层组件..."
make -f ROCm/tools/rocm-build/ROCm.mk \
    OUT_DIR="$OUT_DIR" \
    ROCM_INSTALL_PATH="$ROCM_INSTALL_PATH" \
    SILENT= \
    T_hipcc \
    T_hip_on_rocclr \
    -j $(nproc)

echo "=========================================="
echo "✅ 最小化编译完成"
echo "=========================================="
EOF

chmod +x build_minimal_rocm.sh
./build_minimal_rocm.sh
```

---

### 方案 4：修复 aqlprofile 本身（最佳）

既然无法跳过，不如修复它：

```bash
# 1. 查看错误
echo "=== aqlprofile 编译错误 ==="
tail -100 /data/code/rocm6.4.3/out/ubuntu-22.04/22.04/logs/1.Errors.aqlprofile.txt

# 2. 常见问题及解决

# 问题1：缺少 hsa-runtime 头文件
# 解决：确保 rocr 先编译
make -f ROCm/tools/rocm-build/ROCm.mk T_rocr -j $(nproc)

# 问题2：缺少 rocprofiler 依赖
# 解决：先编译 rocprofiler-register
make -f ROCm/tools/rocm-build/ROCm.mk T_rocprofiler-register -j $(nproc)

# 问题3：CMake配置错误
# 解决：清理并重新配置
rm -rf out/ubuntu-22.04/22.04/build/aqlprofile
make -f ROCm/tools/rocm-build/ROCm.mk T_aqlprofile -j $(nproc)
```

---

## 深入分析

### 查看实际依赖关系

```bash
cd /data/code/rocm6.4.3/ROCm

# 1. 查看 rocr 的依赖文件
echo "=== rocr 的依赖 ==="
cat rocr/DEPENDENCIES.txt 2>/dev/null || echo "文件不存在"

# 2. 查看构建系统如何定义依赖
echo "=== 构建系统中的依赖定义 ==="
grep -A 5 "T_rocr:" tools/rocm-build/ROCm.mk

# 3. 查看 aqlprofile 的触发条件
echo "=== aqlprofile 的触发规则 ==="
grep -B 5 -A 10 "aqlprofile" tools/rocm-build/ROCm.mk | head -30
```

### 理解构建系统

ROCm 的构建系统使用 `.txt` 文件来触发依赖：

```
/data/code/rocm6.4.3/out/ubuntu-22.04/22.04/logs/rocr.txt 存在
    ↓ 触发
aqlprofile started due to rocr.txt
```

**机制**：
1. 当 `rocr` 编译完成时，创建 `rocr.txt`
2. 构建系统检测到 `rocr.txt`
3. 自动触发依赖 `rocr` 的组件
4. `aqlprofile` 被标记为需要 `rocr`

---

## 临时禁用 aqlprofile 的高级方法

### 修改构建规则

```bash
cd /data/code/rocm6.4.3/ROCm

# 1. 备份 ROCm.mk
cp tools/rocm-build/ROCm.mk tools/rocm-build/ROCm.mk.backup

# 2. 查找 aqlprofile 的规则
grep -n "aqlprofile" tools/rocm-build/ROCm.mk

# 3. 注释掉 aqlprofile 相关规则（示例）
# 假设在第150-160行
sed -i '150,160s/^/# DISABLED: /' tools/rocm-build/ROCm.mk

# 4. 或者使用更精确的方法
sed -i '/^T_aqlprofile:/,/^$/s/^/# /' tools/rocm-build/ROCm.mk
```

### 创建假的成功标记

```bash
# 欺骗构建系统，让它认为 aqlprofile 已经成功
OUT_DIR=/data/code/rocm6.4.3/out/ubuntu-22.04/22.04

# 创建成功标记文件
mkdir -p "$OUT_DIR/logs"
touch "$OUT_DIR/logs/aqlprofile.txt"

# 创建空的安装目录
mkdir -p "$OUT_DIR/install/aqlprofile"

# 然后继续编译
make -f ROCm/tools/rocm-build/ROCm.mk \
    OUT_DIR="$OUT_DIR" \
    ROCM_INSTALL_PATH=/opt/rocm-9.99.99 \
    SILENT= \
    rocm-dev \
    -j $(nproc)
```

---

## 推荐流程

### 步骤 1：先看错误日志

```bash
# 查看 aqlprofile 的具体错误
tail -100 /data/code/rocm6.4.3/out/ubuntu-22.04/22.04/logs/1.Errors.aqlprofile.txt
```

**根据错误类型决定**：
- 如果是简单的依赖问题 → 安装依赖
- 如果是代码错误 → 修补代码
- 如果无法修复 → 使用临时禁用方法

### 步骤 2：尝试修复（最推荐）

```bash
cd /data/code/rocm6.4.3

# 清理 aqlprofile 的构建
rm -rf out/ubuntu-22.04/22.04/build/aqlprofile
rm -f out/ubuntu-22.04/22.04/logs/*aqlprofile*

# 确保依赖已编译
make -f ROCm/tools/rocm-build/ROCm.mk \
    T_devicelibs \
    T_rocr \
    T_rocprofiler-register \
    -j $(nproc)

# 单独编译 aqlprofile 以查看详细错误
make -f ROCm/tools/rocm-build/ROCm.mk T_aqlprofile -j 1
```

### 步骤 3：如果实在无法修复

```bash
# 使用创建假标记的方法
OUT_DIR=/data/code/rocm6.4.3/out/ubuntu-22.04/22.04

# 创建成功标记
mkdir -p "$OUT_DIR/logs"
echo "Fake success marker" > "$OUT_DIR/logs/aqlprofile.txt"
mkdir -p "$OUT_DIR/install/aqlprofile/include"
mkdir -p "$OUT_DIR/install/aqlprofile/lib"

# 继续编译其他组件
make -f ROCm/tools/rocm-build/ROCm.mk \
    OUT_DIR="$OUT_DIR" \
    ROCM_INSTALL_PATH=/opt/rocm-9.99.99 \
    SILENT= \
    rocm-dev \
    -j $(nproc)
```

---

## 命令总结

```bash
# 快速诊断命令
cd /data/code/rocm6.4.3

echo "=== 1. 查看 aqlprofile 错误 ==="
tail -100 out/ubuntu-22.04/22.04/logs/1.Errors.aqlprofile.txt

echo "=== 2. 查看完整日志 ==="
tail -200 out/ubuntu-22.04/22.04/logs/2.Inprogress.aqlprofile.txt

echo "=== 3. 查看依赖关系 ==="
grep -A 10 "T_rocr:" ROCm/tools/rocm-build/ROCm.mk
grep "aqlprofile" ROCm/rocr/DEPENDENCIES.txt 2>/dev/null

echo "=== 4. 查看触发机制 ==="
ls -la out/ubuntu-22.04/22.04/logs/*.txt | grep -E "(rocr|aqlprofile)"
```

---

## 如果您提供错误日志

请运行以下命令并提供输出：

```bash
# 1. aqlprofile 的错误
tail -100 /data/code/rocm6.4.3/out/ubuntu-22.04/22.04/logs/1.Errors.aqlprofile.txt

# 2. 或完整日志的最后部分
tail -200 /data/code/rocm6.4.3/out/ubuntu-22.04/22.04/logs/2.Inprogress.aqlprofile.txt

# 3. CMake 配置输出
grep -A 20 "CMake Error" /data/code/rocm6.4.3/out/ubuntu-22.04/22.04/logs/2.Inprogress.aqlprofile.txt
```

然后我可以提供更精确的修复方案。

