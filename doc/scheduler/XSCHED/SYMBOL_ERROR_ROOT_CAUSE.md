# Symbol Error 根本原因分析

**日期**: 2026-01-28  
**问题**: `undefined symbol: _ZTIN6xsched3hip10HipCommandE`  
**状态**: 🔧 需要重新编译 XSched

---

## 🔍 根本原因

### 诊断结果

```bash
# libshimhip.so 需要这个符号
$ nm libshimhip.so | grep HipCommandE
                 U _ZTIN6xsched3hip10HipCommandE
                 ↑
                'U' = Undefined (未定义，需要从其他库获取)

# libhalhip.so 定义了这个符号
$ nm libhalhip.so | grep HipCommandE
00000000000285c8 d _ZTIN6xsched3hip10HipCommandE
                 ↑
                'd' = data section, LOCAL (本地符号，未导出)
```

### 问题所在

```
libshimhip.so:  需要符号 (U)
libhalhip.so:   有符号但未导出 (d, 不是 D)
                ↓
           无法在运行时解析符号
                ↓
          symbol lookup error
```

---

## ❓ 为什么之前能工作？

### 时间线

```
09:36 - XSched 编译
17:33 - Phase 4 Test 3a 成功 ✅
18:51 - Phase 4 Test 3b 失败 ❌ (现在)
```

### 可能的原因

#### 1. 环境变化（最可能）

```
可能发生了:
  - Python 环境重新加载
  - 动态链接器缓存清除
  - 某些环境变量改变
  - Docker 容器重启

导致:
  - 之前"碰巧"能工作的配置失效
  - 符号解析更严格
```

#### 2. 之前的测试可能有隐藏问题

```
可能情况:
  - 之前测试"碰巧"工作但不可靠
  - 符号解析依赖运气
  - 现在问题暴露出来了
```

---

## 🔧 解决方案

### 方案 1: 重新编译 XSched（推荐）⭐

```bash
./rebuild_xsched.sh
```

**目标**: 
- 正确导出符号
- 使 libhalhip.so 的符号从 'd' 变为 'D' (导出)

**预计时间**: 2-3 分钟

---

### 方案 2: 手动重新编译

```bash
docker exec zhenflashinfer_v1 bash -c "
    cd /data/dockercode/xsched-build
    make clean
    make -j\$(nproc)
    make install
"
```

---

### 方案 3: 从源码重新构建（如果方案 1/2 失败）

```bash
docker exec zhenflashinfer_v1 bash -c "
    cd /data/dockercode
    rm -rf xsched-build
    mkdir xsched-build
    cd xsched-build
    
    cmake ../xsched-official \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/data/dockercode/xsched-build/output
    
    make -j\$(nproc)
    make install
"
```

---

## 🎯 验证方法

### 重新编译后，检查符号是否正确导出

```bash
docker exec zhenflashinfer_v1 bash -c "
    nm /data/dockercode/xsched-build/output/lib/libhalhip.so | grep HipCommandE
"
```

**期望看到**:
```
00000000000285c8 D _ZTIN6xsched3hip10HipCommandE
                 ↑
                'D' = 导出的数据符号 (不是 'd')
```

**如果看到 'd'**: 需要修改编译配置

---

### 测试 PyTorch + XSched

```bash
docker exec zhenflashinfer_v1 bash -c "
    export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:\$LD_LIBRARY_PATH
    export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so
    python3 -c 'import torch; print(torch.cuda.is_available())'
"
```

**期望**: `True`

---

## 💡 为什么需要导出符号？

### 符号类型说明

```
'U' = Undefined    - 符号未定义，需要从其他库获取
'T' = Text         - 导出的代码符号
'D' = Data         - 导出的数据符号
'd' = data         - 本地数据符号（未导出）
't' = text         - 本地代码符号（未导出）
```

---

### 动态链接的要求

```
libshimhip.so (U) ──需要──> _ZTIN6xsched3hip10HipCommandE
                                              ↑
libhalhip.so (d) ──有但未导出──┘

问题: 'd' (本地) 不能被其他库引用
解决: 改为 'D' (导出)
```

---

## 🔍 深入分析：为什么是 'd' 而不是 'D'？

### 可能的原因

#### 1. 编译器优化

```cmake
# CMakeLists.txt 中可能有:
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

→ 默认隐藏所有符号
→ 只导出明确标记的符号
```

#### 2. 链接器设置

```cmake
# 链接时可能使用了:
target_link_options(libhalhip PRIVATE "-Wl,--exclude-libs,ALL")

→ 排除静态库的符号
→ 导致某些符号变为本地
```

#### 3. 符号定义方式

```cpp
// 如果定义时没有 export 标记:
struct HipCommand { ... };  // 默认可能是本地符号

// 正确的方式:
__attribute__((visibility("default")))
struct HipCommand { ... };  // 明确导出
```

---

## 📝 修复 CMakeLists.txt (如果需要)

### 检查 XSched 源码

```bash
docker exec zhenflashinfer_v1 cat /data/dockercode/xsched-official/platforms/hip/CMakeLists.txt | grep -i visibility
```

### 可能需要的修改

```cmake
# 在 platforms/hip/CMakeLists.txt 中添加:

# 导出所有符号
set_target_properties(libhalhip PROPERTIES
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)

# 或者使用链接器选项
target_link_options(libhalhip PRIVATE 
    "-Wl,--export-dynamic"
)
```

---

## 🚀 立即行动

### Step 1: 尝试重新编译

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED

./rebuild_xsched.sh
```

**预计**: 2-3 分钟

---

### Step 2: 如果重新编译成功

```bash
# 快速测试 (10 秒)
./test_xsched_quick.sh

# 如果成功，运行完整测试 (180 秒)
./test_xsched_only.sh
```

---

### Step 3: 如果重新编译后还是 'd'

需要修改 CMakeLists.txt:

```bash
docker exec zhenflashinfer_v1 bash -c "
    cd /data/dockercode/xsched-official/platforms/hip
    
    # 备份
    cp CMakeLists.txt CMakeLists.txt.backup
    
    # 添加导出选项
    echo 'set_target_properties(halhip PROPERTIES CXX_VISIBILITY_PRESET default)' >> CMakeLists.txt
    
    # 重新编译
    cd /data/dockercode/xsched-build
    make clean
    make -j\$(nproc)
    make install
"
```

---

## 📊 当前状态总结

```
问题: Symbol lookup error
根因: libhalhip.so 符号未导出 (d 而非 D)
解决: 重新编译 XSched
优先级: 🔴 高（阻塞测试）
```

---

## 🎯 预期结果

```
重新编译后:
  ✅ 符号正确导出 (D)
  ✅ PyTorch + XSched 正常
  ✅ 可以运行测试
  ✅ 获得 XSched 性能数据
```

---

## 💡 经验教训

### 1. 符号可见性很重要

```
动态链接库需要:
  - 明确导出需要的符号
  - 使用正确的编译/链接选项
  - 验证符号类型
```

### 2. 测试要覆盖不同场景

```
之前测试可能"碰巧"工作
真正的压力测试暴露了问题
这实际上是好事！
```

### 3. 诊断工具很有价值

```
nm 命令可以:
  - 查看符号定义
  - 检查符号类型
  - 诊断链接问题
```

---

**立即执行**: `./rebuild_xsched.sh` 🚀
