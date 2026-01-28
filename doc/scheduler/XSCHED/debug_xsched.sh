#!/bin/bash
# XSched Debug Script - 渐进式测试
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="zhenflashinfer_v1"
DOCKER_WORKDIR="/data/dockercode"
TEST_SCRIPT="debug_xsched_step_by_step.py"
LOG_DIR="$SCRIPT_DIR/phase4_log"

mkdir -p "$LOG_DIR"

echo "========================================================================"
echo "XSched Debug - Progressive Testing"
echo "========================================================================"
echo ""
echo "This will run tests in two modes:"
echo "  1. Baseline (no XSched)"
echo "  2. XSched enabled"
echo ""

# 复制测试脚本
echo "[1/4] Copying test script..."
docker cp "$SCRIPT_DIR/tests/$TEST_SCRIPT" "$CONTAINER:$DOCKER_WORKDIR/"
echo "  ✅ Script copied"
echo ""

# Test 1: Baseline
echo "========================================================================"
echo "[2/4] Running Baseline Tests (no XSched)"
echo "========================================================================"
echo ""

docker exec "$CONTAINER" bash -c "
    cd $DOCKER_WORKDIR && \
    python3 $TEST_SCRIPT
" 2>&1 | tee "$LOG_DIR/debug_baseline_$(date +%Y%m%d_%H%M%S).log"

BASELINE_EXIT=$?
echo ""
if [ $BASELINE_EXIT -eq 0 ]; then
    echo "✅ Baseline tests PASSED"
else
    echo "❌ Baseline tests FAILED (exit $BASELINE_EXIT)"
    echo "   Stopping here - baseline should work"
    exit 1
fi

echo ""
echo "========================================================================"
echo "[3/4] Running XSched Tests (with XSched)"
echo "========================================================================"
echo ""

docker exec "$CONTAINER" bash -c "
    cd $DOCKER_WORKDIR && \
    export LD_LIBRARY_PATH=$DOCKER_WORKDIR/xsched-build/output/lib:\$LD_LIBRARY_PATH && \
    export LD_PRELOAD=$DOCKER_WORKDIR/xsched-build/output/lib/libshimhip.so && \
    python3 $TEST_SCRIPT
" 2>&1 | tee "$LOG_DIR/debug_xsched_$(date +%Y%m%d_%H%M%S).log"

XSCHED_EXIT=$?
echo ""

# 分析结果
echo "========================================================================"
echo "[4/4] Analysis"
echo "========================================================================"
echo ""

if [ $XSCHED_EXIT -eq 0 ]; then
    echo "🎉 XSched tests PASSED!"
    echo ""
    echo "Great! XSched is working. The previous test failures might be due to:"
    echo "  - Specific model/batch size combinations"
    echo "  - Multi-threading issues"
    echo "  - Load conditions"
elif [ $XSCHED_EXIT -eq 139 ]; then
    echo "⚠️  XSched tests exited with 139 (segfault during cleanup)"
    echo ""
    echo "This is likely the known cleanup issue. Check if tests actually passed"
    echo "before the segfault."
else
    echo "❌ XSched tests FAILED (exit $XSCHED_EXIT)"
    echo ""
    echo "Check the logs to see which step failed:"
    echo "  - Step 1 (basic tensor): Basic GPU operations"
    echo "  - Step 2 (matmul): BLAS operations"
    echo "  - Step 3 (conv2d): MIOpen kernels ← likely failure point"
    echo "  - Step 4 (simple model): Small network"
    echo "  - Step 5 (ResNet): Full model"
fi

echo ""
echo "Logs saved in:"
echo "  Baseline: $LOG_DIR/debug_baseline_*.log"
echo "  XSched:   $LOG_DIR/debug_xsched_*.log"
echo ""

exit 0
