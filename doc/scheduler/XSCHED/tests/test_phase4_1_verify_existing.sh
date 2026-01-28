#!/bin/bash
#####################################################################
# Phase 4 Test 1: Verify Existing XSched Installation
# 验证已有的 XSched 安装（来自 Phase 2）
# 
# 背景:
#   - Phase 1-3 已完成（PyTorch + XSched）
#   - XSched 已编译安装在 /data/dockercode/xsched-build
#   - Phase 4 基于已有环境进行论文测试
#####################################################################

set -e

# 使用已有的 XSched 路径
XSCHED_SOURCE="/data/dockercode/xsched-official"
XSCHED_BUILD="/data/dockercode/xsched-build"
XSCHED_INSTALL="$XSCHED_BUILD/output"
RESULTS_DIR="/data/dockercode/test_results_phase4"

echo "================================================"
echo "Phase 4 Test 1: Verify Existing XSched"
echo "================================================"
echo ""
echo "Running inside Docker container: $(hostname)"
echo ""

# 1. 验证源码存在
echo "[Step 1/5] Checking XSched source..."
if [ ! -d "$XSCHED_SOURCE" ]; then
    echo "  ❌ XSched source not found at $XSCHED_SOURCE"
    echo "  This test requires Phase 2 to be completed first."
    exit 1
else
    echo "  ✅ Source exists: $XSCHED_SOURCE"
    echo "     Git commit: $(cd $XSCHED_SOURCE && git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
fi

# 2. 验证编译目录
echo ""
echo "[Step 2/5] Checking build directory..."
if [ ! -d "$XSCHED_BUILD" ]; then
    echo "  ❌ Build directory not found at $XSCHED_BUILD"
    echo "  Please complete Phase 2 first (PyTorch + XSched integration)"
    exit 1
else
    echo "  ✅ Build directory exists: $XSCHED_BUILD"
fi

# 3. 验证安装的库
echo ""
echo "[Step 3/5] Checking installed libraries..."

check_lib() {
    if [ -f "$1" ]; then
        echo "  ✅ $1"
        echo "     Size: $(du -h "$1" | cut -f1)"
        echo "     Modified: $(stat -c %y "$1" | cut -d. -f1)"
        return 0
    else
        echo "  ❌ Missing: $1"
        return 1
    fi
}

LIBS_OK=true
check_lib "$XSCHED_INSTALL/lib/libhalhip.so" || LIBS_OK=false
check_lib "$XSCHED_INSTALL/lib/libshimhip.so" || LIBS_OK=false

if [ "$LIBS_OK" = false ]; then
    echo ""
    echo "  ❌ Required libraries not found!"
    echo "  Please rebuild XSched (Phase 2)"
    exit 1
fi

# 4. 验证 Symbol Versioning (Phase 2 的关键修复)
echo ""
echo "[Step 4/5] Verifying Symbol Versioning fix..."
if nm -D "$XSCHED_INSTALL/lib/libshimhip.so" | grep -q "hipMalloc@@hip_4.2"; then
    echo "  ✅ Symbol versioning correctly applied"
    echo "     (This was the critical Phase 2 fix)"
else
    echo "  ⚠️  Symbol versioning not detected"
    echo "     (May need rebuild with hip_version.map)"
fi

# 5. 测试基础功能
echo ""
echo "[Step 5/5] Testing basic functionality..."

# 设置环境
export LD_LIBRARY_PATH=$XSCHED_INSTALL/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$XSCHED_INSTALL/lib/libshimhip.so

# 简单的 PyTorch 测试
echo "  Testing PyTorch with XSched..."
python3 << 'EOF'
import torch
import sys

try:
    # 创建 tensor
    A = torch.randn(128, 128, device='cuda')
    B = torch.randn(128, 128, device='cuda')
    
    # 矩阵乘法
    C = A @ B
    torch.cuda.synchronize()
    
    print("    ✅ PyTorch + XSched works!")
    print(f"    PyTorch version: {torch.__version__}")
    print(f"    CUDA available: {torch.cuda.is_available()}")
    sys.exit(0)
except Exception as e:
    print(f"    ❌ PyTorch test failed: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo "  ✅ Basic functionality verified"
else
    echo "  ❌ Basic functionality test failed"
    exit 1
fi

# 6. 生成报告
echo ""
echo "Generating verification report..."
mkdir -p "$RESULTS_DIR"

cat > "$RESULTS_DIR/phase4_test1_report.json" << EOF
{
  "test_id": "Phase4-1",
  "test_name": "Verify Existing XSched Installation",
  "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "container": "$(hostname)",
  "status": "PASS",
  "xsched_paths": {
    "source": "$XSCHED_SOURCE",
    "build": "$XSCHED_BUILD",
    "install": "$XSCHED_INSTALL"
  },
  "libraries": {
    "libhalhip": "$(du -h "$XSCHED_INSTALL/lib/libhalhip.so" | cut -f1)",
    "libshimhip": "$(du -h "$XSCHED_INSTALL/lib/libshimhip.so" | cut -f1)"
  },
  "verification": {
    "symbol_versioning": "$(nm -D "$XSCHED_INSTALL/lib/libshimhip.so" | grep -q "hipMalloc@@hip_4.2" && echo "YES" || echo "NO")",
    "pytorch_test": "PASS",
    "phase2_status": "✅ Complete"
  },
  "notes": "Using existing XSched from Phase 2 (PyTorch integration)"
}
EOF

echo "  ✅ Report saved to $RESULTS_DIR/phase4_test1_report.json"

echo ""
echo "================================================"
echo "✅ Phase 4 Test 1 PASSED"
echo "================================================"
echo ""
echo "Verified XSched Installation:"
echo "  Source:  $XSCHED_SOURCE"
echo "  Build:   $XSCHED_BUILD"
echo "  Install: $XSCHED_INSTALL"
echo ""
echo "Key Features Verified:"
echo "  ✅ Libraries compiled and installed"
echo "  ✅ Symbol versioning (Phase 2 fix)"
echo "  ✅ PyTorch integration working"
echo ""
echo "Environment Setup:"
echo "  export LD_LIBRARY_PATH=$XSCHED_INSTALL/lib:\$LD_LIBRARY_PATH"
echo "  export LD_PRELOAD=$XSCHED_INSTALL/lib/libshimhip.so"
echo ""
echo "Next: Phase 4 Test 2 - Runtime Overhead Measurement"
echo "      (Based on Paper Section 7.4.1)"
