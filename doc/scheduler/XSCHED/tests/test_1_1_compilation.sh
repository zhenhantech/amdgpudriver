#!/bin/bash
#####################################################################
# Test 1.1: XSched Compilation & Installation
# 验证 XSched 在 MI308X 上的编译和安装
# 
# 用法：
#   在宿主机运行（通过 docker exec）:
#     ./test_1_1_compilation.sh
#   
#   在 Docker 内运行:
#     docker exec zhenflashinfer_v1 bash -c "cd /data/dockercode && ./test_1_1_compilation.sh"
#####################################################################

set -e

# 检测是否在 Docker 内
if [ -f /.dockerenv ]; then
    echo "✅ Running inside Docker container"
    SCRIPT_DIR="/data/dockercode"
else
    echo "ℹ️  Running on host, will execute in Docker container"
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# Docker 容器内的路径
XSCHED_SOURCE="/data/dockercode/xsched-test"
XSCHED_BUILD="/data/dockercode/xsched-test-build"
XSCHED_INSTALL="/data/dockercode/xsched-test-install"

# 如果在宿主机，通过 docker exec 执行
if [ ! -f /.dockerenv ]; then
    echo "Executing inside Docker container: zhenflashinfer_v1"
    docker exec -i zhenflashinfer_v1 bash << 'DOCKER_EOF'
        cd /data/dockercode
        
        # 重新定义路径（在容器内）
        XSCHED_SOURCE="/data/dockercode/xsched-test"
        XSCHED_BUILD="/data/dockercode/xsched-test-build"
        XSCHED_INSTALL="/data/dockercode/xsched-test-install"
        
        # 以下是实际的测试脚本
        set -e
        
        echo "================================================"
        echo "Test 1.1: XSched Compilation & Installation"
        echo "================================================"
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
            exit 1
        fi
        
        # 6. 验证安装
        echo ""
        echo "[Step 6/6] Verifying installation..."
        
        if [ -f "$XSCHED_INSTALL/lib/libhalhip.so" ]; then
            echo "  ✅ $XSCHED_INSTALL/lib/libhalhip.so ($(du -h "$XSCHED_INSTALL/lib/libhalhip.so" | cut -f1))"
        else
            echo "  ❌ Missing: libhalhip.so"
            exit 1
        fi
        
        if [ -f "$XSCHED_INSTALL/lib/libshimhip.so" ]; then
            echo "  ✅ $XSCHED_INSTALL/lib/libshimhip.so ($(du -h "$XSCHED_INSTALL/lib/libshimhip.so" | cut -f1))"
        else
            echo "  ❌ Missing: libshimhip.so"
            exit 1
        fi
        
        # 7. 代码量统计
        echo ""
        echo "[Bonus] Code Size Statistics:"
        echo "  Shim LoC:"
        find "$XSCHED_SOURCE/platforms/hip/shim" -name "*.cpp" -o -name "*.h" 2>/dev/null | xargs wc -l 2>/dev/null | tail -1 || echo "    (不可用)"
        echo "  Lv1 LoC:"
        find "$XSCHED_SOURCE/platforms/hip/hal" -name "*.cpp" -o -name "*.h" 2>/dev/null | xargs wc -l 2>/dev/null | tail -1 || echo "    (不可用)"
        
        # 8. 生成测试报告
        echo ""
        echo "Generating test report..."
        mkdir -p /data/dockercode/test_results
        cat > /data/dockercode/test_results/test_1_1_report.json << EOF
{
  "test_id": "1.1",
  "test_name": "Compilation & Installation",
  "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hardware": "AMD MI308X",
  "rocm_version": "$(hipcc --version | head -1)",
  "status": "PASS",
  "metrics": {
    "compilation_time_sec": ${build_time:-0},
    "install_path": "$XSCHED_INSTALL"
  }
}
EOF
        
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
        echo "Next step: Run test_1_2_native_examples.sh"
DOCKER_EOF
    
    exit $?
fi

echo "================================================"
echo "Test 1.1: XSched Compilation & Installation"
echo "================================================"
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
    exit 1
fi

# 6. 验证安装
echo ""
echo "[Step 6/6] Verifying installation..."

check_file() {
    if [ -f "$1" ]; then
        echo "  ✅ $1 ($(du -h "$1" | cut -f1))"
    else
        echo "  ❌ Missing: $1"
        return 1
    fi
}

check_file "$XSCHED_INSTALL/lib/libhalhip.so"
check_file "$XSCHED_INSTALL/lib/libshimhip.so"

# 7. 代码量统计
echo ""
echo "[Bonus] Code Size Statistics:"
echo "  Shim LoC:"
find "$XSCHED_SOURCE/platforms/hip/shim" -name "*.cpp" -o -name "*.h" | xargs wc -l 2>/dev/null | tail -1 || echo "    (不可用)"
echo "  Lv1 LoC:"
find "$XSCHED_SOURCE/platforms/hip/hal" -name "*.cpp" -o -name "*.h" | xargs wc -l 2>/dev/null | tail -1 || echo "    (不可用)"

# 8. 生成测试报告
echo ""
echo "Generating test report..."
cat > "$SCRIPT_DIR/../test_results/test_1_1_report.json" << EOF
{
  "test_id": "1.1",
  "test_name": "Compilation & Installation",
  "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hardware": "AMD MI308X",
  "rocm_version": "$(hipcc --version | head -1)",
  "status": "PASS",
  "metrics": {
    "compilation_time_sec": ${build_time},
    "install_path": "$XSCHED_INSTALL"
  }
}
EOF

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
echo "Next step: Run test_1_2_native_examples.sh"
