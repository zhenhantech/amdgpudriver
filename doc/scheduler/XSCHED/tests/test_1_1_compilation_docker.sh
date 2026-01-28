#!/bin/bash
#####################################################################
# Test 1.1: XSched Compilation & Installation (Docker Version)
# 在 Docker 容器内验证 XSched 的编译和安装
# 
# 用法：
#   docker exec zhenflashinfer_v1 bash /data/dockercode/test_1_1_compilation_docker.sh
#####################################################################

set -e

# Docker 容器内的路径
XSCHED_SOURCE="/data/dockercode/xsched-test"
XSCHED_BUILD="/data/dockercode/xsched-test-build"
XSCHED_INSTALL="/data/dockercode/xsched-test-install"
RESULTS_DIR="/data/dockercode/test_results"

echo "================================================"
echo "Test 1.1: XSched Compilation & Installation"
echo "================================================"
echo ""
echo "Running inside Docker container: $(hostname)"
echo ""

# 1. 克隆或更新源码
echo "[Step 1/6] Checking XSched source..."
if [ ! -d "$XSCHED_SOURCE" ]; then
    echo "  Cloning XSched..."
    cd /data/dockercode
    git clone https://github.com/XpuOS/xsched.git xsched-test
    echo "  ✅ Cloned"
else
    echo "  ✅ Source exists: $XSCHED_SOURCE"
fi

# 2. 检查依赖
echo ""
echo "[Step 2/6] Checking dependencies..."
which hipcc >/dev/null 2>&1 || { echo "❌ hipcc not found!"; exit 1; }
which cmake >/dev/null 2>&1 || { echo "❌ cmake not found!"; exit 1; }
echo "  ✅ hipcc: $(which hipcc)"
echo "  ✅ cmake: $(which cmake)"
echo "  ✅ ROCm: $(hipcc --version | head -1)"

# 3. 配置 CMake
echo ""
echo "[Step 3/6] Configuring CMake..."
mkdir -p "$XSCHED_BUILD"
cd "$XSCHED_BUILD"

cmake "$XSCHED_SOURCE" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DXSCHED_PLATFORM=hip \
    -DCMAKE_INSTALL_PREFIX="$XSCHED_INSTALL" \
    -DCMAKE_CXX_FLAGS="-Wno-error=maybe-uninitialized" \
    2>&1 | tee cmake_output.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "  ✅ CMake configured"
else
    echo "  ❌ CMake failed!"
    echo ""
    echo "Last 20 lines of cmake_output.log:"
    tail -20 cmake_output.log
    exit 1
fi

# 4. 编译
echo ""
echo "[Step 4/6] Building XSched..."
start_time=$(date +%s)

make -j$(nproc) 2>&1 | tee build_output.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    end_time=$(date +%s)
    build_time=$((end_time - start_time))
    echo "  ✅ Build completed in ${build_time}s"
else
    echo "  ❌ Build failed!"
    echo ""
    echo "Last 50 lines of build_output.log:"
    tail -50 build_output.log
    exit 1
fi

# 5. 安装
echo ""
echo "[Step 5/6] Installing XSched..."
make install 2>&1 | tee install_output.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "  ✅ Installed to $XSCHED_INSTALL"
else
    echo "  ❌ Install failed!"
    tail -20 install_output.log
    exit 1
fi

# 6. 验证安装
echo ""
echo "[Step 6/6] Verifying installation..."

check_file() {
    if [ -f "$1" ]; then
        echo "  ✅ $1 ($(du -h "$1" | cut -f1))"
        return 0
    else
        echo "  ❌ Missing: $1"
        return 1
    fi
}

check_file "$XSCHED_INSTALL/lib/libhalhip.so" || exit 1
check_file "$XSCHED_INSTALL/lib/libshimhip.so" || exit 1

# 7. 代码量统计
echo ""
echo "[Bonus] Code Size Statistics:"
echo "  Shim LoC:"
shim_loc=$(find "$XSCHED_SOURCE/platforms/hip/shim" -name "*.cpp" -o -name "*.h" 2>/dev/null | xargs wc -l 2>/dev/null | tail -1)
echo "    $shim_loc"

echo "  Lv1 LoC:"
lv1_loc=$(find "$XSCHED_SOURCE/platforms/hip/hal" -name "*.cpp" -o -name "*.h" 2>/dev/null | xargs wc -l 2>/dev/null | tail -1)
echo "    $lv1_loc"

# 8. 生成测试报告
echo ""
echo "Generating test report..."
mkdir -p "$RESULTS_DIR"

cat > "$RESULTS_DIR/test_1_1_report.json" << EOF
{
  "test_id": "1.1",
  "test_name": "Compilation & Installation",
  "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "container": "$(hostname)",
  "hardware": "AMD MI308X",
  "rocm_version": "$(hipcc --version | head -1 | tr -d '\n')",
  "status": "PASS",
  "metrics": {
    "compilation_time_sec": ${build_time:-0},
    "install_path": "$XSCHED_INSTALL",
    "libhalhip_size": "$(du -h "$XSCHED_INSTALL/lib/libhalhip.so" | cut -f1)",
    "libshimhip_size": "$(du -h "$XSCHED_INSTALL/lib/libshimhip.so" | cut -f1)"
  },
  "code_size": {
    "shim_loc": "$shim_loc",
    "lv1_loc": "$lv1_loc"
  }
}
EOF

echo "  ✅ Report saved to $RESULTS_DIR/test_1_1_report.json"

echo ""
echo "================================================"
echo "✅ Test 1.1 PASSED"
echo "================================================"
echo ""
echo "Installation summary:"
echo "  Source:  $XSCHED_SOURCE"
echo "  Build:   $XSCHED_BUILD"
echo "  Install: $XSCHED_INSTALL"
echo ""
echo "Libraries installed:"
ls -lh "$XSCHED_INSTALL/lib/"libhalhip.so "$XSCHED_INSTALL/lib/libshimhip.so"
echo ""
echo "Environment setup:"
echo "  export LD_LIBRARY_PATH=$XSCHED_INSTALL/lib:\$LD_LIBRARY_PATH"
echo "  export LD_PRELOAD=$XSCHED_INSTALL/lib/libshimhip.so"
echo ""
echo "Next step: Run test_1_2_native_examples.sh"
