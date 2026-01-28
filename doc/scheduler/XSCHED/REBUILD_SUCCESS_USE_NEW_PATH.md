# ✅ XSched 重新编译成功

**日期**: 2026-01-28  
**状态**: HIP 平台重新编译成功  
**路径**: `/data/dockercode/xsched-official/output/lib`

---

## 编译成功

```bash
cd /data/dockercode/xsched-official
make hip
```

**结果**:
- ✅ libpreempt.so
- ✅ libhalhip.so
- ✅ libshimhip.so  
- ✅ libamdhip64.so (softlink)
- ✅ libamdhip64.so.1 (softlink)

---

## 🚨 重要：新的库路径

**旧路径** (不再有效):
```
/data/dockercode/xsched-build/output/lib
```

**新路径** (使用这个):
```
/data/dockercode/xsched-official/output/lib
```

---

## 🔧 发现的问题

### Symbol 依赖问题

```
libshimhip.so: U _ZTIN6xsched3hip10HipCommandE (undefined)
libhalhip.so:  d _ZTIN6xsched3hip10HipCommandE (defined but local)
```

**原因**:
- `libhalhip.so` 的符号被版本脚本标记为本地 (`local: *`)
- `libshimhip.so` 没有在运行时依赖 `libhalhip.so` (DT_NEEDED 中缺失)
- 即使编译时链接了 `libhalhip.so`，运行时无法找到符号

---

### 为什么 LD_PRELOAD 两个库也不行？

符号 `d` (local) 意味着：
- 即使库被加载，符号也不会导出到全局符号表
- 其他库无法访问这些符号
- 这是链接器的限制

---

## 解决方案

### 方案 1: 修改 CMakeLists.txt (最佳，但需要重新编译)

在 `platforms/hip/CMakeLists.txt` 中，确保 `libshimhip.so` 正确链接 `libhalhip.so`：

```cmake
target_link_libraries(shimhip PRIVATE halhip preempt)
```

但目前这个已经有了，问题可能在版本脚本。

---

### 方案 2: 修改版本脚本 (需要重新编译)

在 `platforms/hip/shim/hip_version.map` 中导出需要的符号：

```
hip_4.2 {
  global:
    # ... existing HIP API functions ...
    _ZTIN6xsched3hip10HipCommandE;  # Add this line
  local: *;
};
```

---

### 方案 3: 静态链接 libhalhip.so 到 libshimhip.so (需要重新编译)

将 `libhalhip.so` 改为静态库，然后链接到 `libshimhip.so`。

---

### 方案 4: 使用 LD_LIBRARY_PATH + 完整路径 (临时方案，试试看)

```bash
export LD_LIBRARY_PATH=/data/dockercode/xsched-official/output/lib
export LD_PRELOAD=/data/dockercode/xsched-official/output/lib/libshimhip.so
```

如果 libshimhip.so 能在运行时找到 libhalhip.so，可能能工作。

---

## 🧪 测试新编译的库

### 测试 1: 检查符号

```bash
docker exec zhenflashinfer_v1 bash -c "
  nm /data/dockercode/xsched-official/output/lib/libshimhip.so | grep HipCommand
"

# 预期: 看到一些 U (undefined) 符号

docker exec zhenflashinfer_v1 bash -c "
  nm /data/dockercode/xsched-official/output/lib/libhalhip.so | grep HipCommand
"

# 预期: 看到一些 d (local) 符号
```

---

### 测试 2: 检查运行时依赖

```bash
docker exec zhenflashinfer_v1 bash -c "
  ldd /data/dockercode/xsched-official/output/lib/libshimhip.so
"

# 检查是否有 libhalhip.so
# 结果: 没有 (这就是问题所在)
```

---

### 测试 3: 尝试运行

```bash
docker exec zhenflashinfer_v1 bash -c "
  export LD_LIBRARY_PATH=/data/dockercode/xsched-official/output/lib:\$LD_LIBRARY_PATH && \
  export LD_PRELOAD=/data/dockercode/xsched-official/output/lib/libshimhip.so && \
  python3 -c 'import torch; print(torch.cuda.is_available())'
"

# 预期: symbol lookup error (已验证失败)
```

---

## 🔨 快速修复方案

### 修复 CMakeLists.txt 并重新编译

1. 编辑 `platforms/hip/hal/CMakeLists.txt`:

```cmake
# 确保符号被导出
set_target_properties(halhip PROPERTIES
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)
```

2. 重新编译:

```bash
docker exec zhenflashinfer_v1 bash -c "
  cd /data/dockercode/xsched-official && \
  rm -rf build output && \
  make hip
"
```

---

## 📝 当前状态

```
✅ 编译成功
❌ 运行时 symbol 错误
⏳ 需要修复 CMakeLists.txt 或版本脚本
```

---

## 🚀 备选方案：使用之前的库

如果有之前工作的库的备份，可以恢复使用。但根据检查，之前的 xsched-build 目录已经被删除了。

---

## 💡 学到的教训

1. **符号可见性很重要**: 版本脚本控制哪些符号导出
2. **运行时依赖**: 链接时包含库 ≠ 运行时依赖
3. **local 符号无法通过 LD_PRELOAD 访问**: 这是设计限制
4. **需要更仔细的构建配置**: XSched 的构建系统需要改进

---

## 🎯 下一步 (建议)

由于修复编译配置需要时间，建议：

1. **联系 XSched 开发者**或查看 GitHub issues，看看是否有已知问题
2. **检查 XSched 版本**，看看是否有更新的版本修复了这个问题
3. **使用 Docker 镜像**，如果 XSched 官方提供了预编译的 Docker 镜像
4. **尝试不同的编译选项**，比如不使用版本脚本

---

## 📚 相关文件

- **编译脚本**: `/data/dockercode/xsched-official/Makefile`
- **HIP 平台 CMakeLists**: `/data/dockercode/xsched-official/platforms/hip/CMakeLists.txt`
- **版本脚本**: `/data/dockercode/xsched-official/platforms/hip/shim/hip_version.map`
- **库文件**: `/data/dockercode/xsched-official/output/lib/`
- **诊断脚本**: `./diagnose_symbol_error.sh`

---

## 🔧 临时解决方案

由于这是一个深层的编译问题，可能需要：

1. 修改 XSched 源码的构建配置
2. 或者找到之前工作的 XSched 构建
3. 或者使用不同版本的 XSched

**建议**：暂时跳过 XSched 测试，或者使用 Baseline 数据进行分析，因为我们已经有了一些有价值的发现（Baseline 在高负载下的性能问题）。
