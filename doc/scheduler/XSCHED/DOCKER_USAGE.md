# XSched 测试 - Docker 使用指南

**更新日期**: 2026-01-28  
**Docker 容器**: zhenflashinfer_v1

---

## 🐳 Docker 环境说明

### 容器信息
```bash
容器名称: zhenflashinfer_v1
基础镜像: PyTorch ROCm
ROCm 版本: 6.4.0
GPU: AMD MI308X
```

### 目录映射
```
宿主机 → Docker 容器
/mnt/md0/zhehan/code/flashinfer/dockercode → /data/dockercode
```

---

## 🚀 运行测试的三种方式

### 方式 1: 使用宿主机包装脚本（推荐）⭐

**最简单！一键运行**

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED

# 运行测试
./run_test_1_1.sh
```

**优点**:
- ✅ 自动检查容器状态
- ✅ 自动复制脚本到容器
- ✅ 显示清晰的输出
- ✅ 自动报告成功/失败

**输出示例**:
```
================================================
Running Test 1.1 in Docker Container
================================================

Container: zhenflashinfer_v1
Script:    /data/dockercode/test_1_1_compilation_docker.sh

[1/2] Copying test script to container...
  ✅ Script copied

[2/2] Executing test in container...

================================================
Test 1.1: XSched Compilation & Installation
================================================
...
✅ Test 1.1 PASSED
```

---

### 方式 2: 直接在 Docker 内运行

**适合调试和交互式操作**

```bash
# 1. 进入容器
docker exec -it zhenflashinfer_v1 bash

# 2. 在容器内运行
cd /data/dockercode
bash test_1_1_compilation_docker.sh

# 3. 查看结果
cat test_results/test_1_1_report.json
```

**优点**:
- ✅ 可以交互式调试
- ✅ 可以查看详细日志
- ✅ 可以手动检查中间结果

---

### 方式 3: 一行命令执行

**快速验证**

```bash
docker exec -it zhenflashinfer_v1 bash /data/dockercode/test_1_1_compilation_docker.sh
```

**优点**:
- ✅ 最简洁
- ✅ 适合脚本化

---

## 📂 文件说明

### 宿主机文件

```
/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED/
├── run_test_1_1.sh                          ← 宿主机包装脚本（推荐使用）
├── tests/
│   ├── test_1_1_compilation.sh              ← 原版（已更新，支持 Docker）
│   └── test_1_1_compilation_docker.sh       ← Docker 版（纯净）
└── README.md
```

### Docker 容器内文件（运行后）

```
/data/dockercode/
├── test_1_1_compilation_docker.sh           ← 测试脚本（从宿主机复制）
├── xsched-test/                             ← XSched 源码
├── xsched-test-build/                       ← 编译目录
│   ├── cmake_output.log
│   ├── build_output.log
│   └── install_output.log
├── xsched-test-install/                     ← 安装目录
│   └── lib/
│       ├── libhalhip.so
│       └── libshimhip.so
└── test_results/                            ← 测试结果
    └── test_1_1_report.json
```

---

## 🔍 查看结果

### 查看测试报告

```bash
# 在宿主机
docker exec zhenflashinfer_v1 cat /data/dockercode/test_results/test_1_1_report.json

# 或通过映射的目录（如果有映射）
cat /mnt/md0/zhehan/code/flashinfer/dockercode/test_results/test_1_1_report.json
```

### 查看编译日志

```bash
# CMake 日志
docker exec zhenflashinfer_v1 cat /data/dockercode/xsched-test-build/cmake_output.log

# 编译日志
docker exec zhenflashinfer_v1 cat /data/dockercode/xsched-test-build/build_output.log

# 安装日志
docker exec zhenflashinfer_v1 cat /data/dockercode/xsched-test-build/install_output.log
```

### 查看安装的库

```bash
docker exec zhenflashinfer_v1 ls -lh /data/dockercode/xsched-test-install/lib/
```

---

## ⚙️ 环境设置

### 在 Docker 内使用 XSched

测试完成后，在 Docker 容器内设置环境变量：

```bash
# 进入容器
docker exec -it zhenflashinfer_v1 bash

# 设置环境变量
export LD_LIBRARY_PATH=/data/dockercode/xsched-test-install/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/data/dockercode/xsched-test-install/lib/libshimhip.so

# 验证
python -c "import torch; print('PyTorch with XSched:', torch.__version__)"
```

### 创建环境设置脚本

```bash
# 在容器内创建
cat > /data/dockercode/setup_xsched.sh << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH=/data/dockercode/xsched-test-install/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/data/dockercode/xsched-test-install/lib/libshimhip.so
echo "✅ XSched environment configured"
EOF

chmod +x /data/dockercode/setup_xsched.sh

# 使用
source /data/dockercode/setup_xsched.sh
```

---

## 🐛 问题排查

### 问题 1: 容器未运行

**错误**:
```
❌ Error: Docker container 'zhenflashinfer_v1' is not running!
```

**解决**:
```bash
# 启动容器
docker start zhenflashinfer_v1

# 确认运行
docker ps | grep zhenflashinfer_v1
```

---

### 问题 2: hipcc 未找到

**错误**:
```
❌ hipcc not found!
```

**解决**:
```bash
# 在容器内
export PATH=/opt/rocm/bin:$PATH

# 或检查 ROCm 安装
ls -la /opt/rocm*/bin/hipcc
```

---

### 问题 3: 编译失败

**查看详细日志**:
```bash
docker exec zhenflashinfer_v1 tail -100 /data/dockercode/xsched-test-build/build_output.log
```

**常见问题**:
1. CMake 版本太旧
2. ROCm 版本不兼容
3. 编译器标志问题

---

### 问题 4: 权限问题

**错误**:
```
Permission denied
```

**解决**:
```bash
# 在宿主机
docker exec -u root zhenflashinfer_v1 chmod +x /data/dockercode/test_1_1_compilation_docker.sh

# 或进入容器修改
docker exec -it zhenflashinfer_v1 bash
chmod +x /data/dockercode/test_1_1_compilation_docker.sh
```

---

## 📊 预期结果

### 成功输出

```
================================================
✅ Test 1.1 PASSED
================================================

Installation summary:
  Source:  /data/dockercode/xsched-test
  Build:   /data/dockercode/xsched-test-build
  Install: /data/dockercode/xsched-test-install

Libraries installed:
-rwxr-xr-x 1 root root 2.3M Jan 28 10:30 libhalhip.so
-rwxr-xr-x 1 root root 856K Jan 28 10:30 libshimhip.so

Environment setup:
  export LD_LIBRARY_PATH=/data/dockercode/xsched-test-install/lib:$LD_LIBRARY_PATH
  export LD_PRELOAD=/data/dockercode/xsched-test-install/lib/libshimhip.so
```

### 测试报告示例

```json
{
  "test_id": "1.1",
  "test_name": "Compilation & Installation",
  "date": "2026-01-28T02:30:45Z",
  "container": "hjbog-srdc-26",
  "hardware": "AMD MI308X",
  "rocm_version": "HIP version: 6.4.0",
  "status": "PASS",
  "metrics": {
    "compilation_time_sec": 180,
    "install_path": "/data/dockercode/xsched-test-install",
    "libhalhip_size": "2.3M",
    "libshimhip_size": "856K"
  },
  "code_size": {
    "shim_loc": "316 total",
    "lv1_loc": "841 total"
  }
}
```

---

## 🎯 快速命令参考

```bash
# 运行测试（推荐）
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED
./run_test_1_1.sh

# 查看结果
docker exec zhenflashinfer_v1 cat /data/dockercode/test_results/test_1_1_report.json

# 进入容器调试
docker exec -it zhenflashinfer_v1 bash

# 清理重新测试
docker exec zhenflashinfer_v1 rm -rf /data/dockercode/xsched-test*
./run_test_1_1.sh
```

---

## 📝 下一步

测试成功后，继续下一个测试：

```bash
# Stage 1.2: 运行官方示例
./run_test_1_2.sh  # (待创建)

# Stage 1.3: API 验证
./run_test_1_3.sh  # (待创建)
```

---

**准备好了吗？立即开始！**

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED
./run_test_1_1.sh
```

预期时间：10-15 分钟 ⏱️  
预期结果：✅ XSched successfully compiled and installed
