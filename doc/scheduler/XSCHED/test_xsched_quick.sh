#!/bin/bash
# 快速测试 XSched (10 秒)，用于调试环境
# 用法: ./test_xsched_quick.sh

set -e

CONTAINER="zhenflashinfer_v1"
TEST_SCRIPT="test_phase4_dual_model_intensive.py"
DOCKER_WORKDIR="/data/dockercode"
RESULTS_DIR="/data/dockercode/test_results_phase4"

echo "========================================================================"
echo "XSched 快速测试 (10 秒，调试用)"
echo "========================================================================"
echo

# 检查容器
if ! docker ps | grep -q "$CONTAINER"; then
    echo "❌ Docker 容器未运行"
    exit 1
fi

# 复制测试脚本
echo "[1/3] 复制测试脚本..."
docker cp tests/test_phase4_dual_model_intensive.py "$CONTAINER:$DOCKER_WORKDIR/$TEST_SCRIPT" 2>/dev/null || true
echo "  ✅ 脚本已复制"
echo

# 验证库文件
echo "[2/3] 验证 XSched 库..."
docker exec "$CONTAINER" bash -c "
    export LD_LIBRARY_PATH=$DOCKER_WORKDIR/xsched-build/output/lib:\$LD_LIBRARY_PATH
    echo '  检查库文件:'
    ls -lh $DOCKER_WORKDIR/xsched-build/output/lib/libhalhip.so | awk '{print \"    libhalhip.so:\", \$5}'
    ls -lh $DOCKER_WORKDIR/xsched-build/output/lib/libshimhip.so | awk '{print \"    libshimhip.so:\", \$5}'
    echo
    echo '  检查库依赖:'
    ldd $DOCKER_WORKDIR/xsched-build/output/lib/libshimhip.so | grep -E 'libpreempt|libhalhip|not found'
"
echo

# 测试 PyTorch + XSched
echo "[3/3] 测试 PyTorch + XSched..."
docker exec "$CONTAINER" bash -c "
    export LD_LIBRARY_PATH=$DOCKER_WORKDIR/xsched-build/output/lib:\$LD_LIBRARY_PATH
    export LD_PRELOAD=$DOCKER_WORKDIR/xsched-build/output/lib/libhalhip.so:$DOCKER_WORKDIR/xsched-build/output/lib/libshimhip.so
    python3 -c 'import torch; print(\"  PyTorch:\", torch.__version__); print(\"  CUDA:\", torch.cuda.is_available())'
"

if [ $? -ne 0 ]; then
    echo
    echo "❌ 环境测试失败"
    echo
    echo "尝试诊断..."
    echo
    echo "方法 1: 检查是否需要 libhalhip.so"
    docker exec "$CONTAINER" bash -c "
        export LD_LIBRARY_PATH=$DOCKER_WORKDIR/xsched-build/output/lib:\$LD_LIBRARY_PATH
        nm $DOCKER_WORKDIR/xsched-build/output/lib/libhalhip.so | grep HipCommand | head -5
    "
    echo
    echo "方法 2: 检查 symbol 是否在其他库中"
    docker exec "$CONTAINER" bash -c "
        export LD_LIBRARY_PATH=$DOCKER_WORKDIR/xsched-build/output/lib:\$LD_LIBRARY_PATH
        nm $DOCKER_WORKDIR/xsched-build/output/lib/libpreempt.so | grep HipCommand | head -5
    "
    exit 1
fi

echo "  ✅ 环境正常"
echo

# 运行 10 秒快速测试
echo "========================================================================"
echo "运行 10 秒快速测试 (20 req/s, batch=1024)"
echo "========================================================================"
echo

docker exec "$CONTAINER" bash -c "
    cd $DOCKER_WORKDIR && \
    export LD_LIBRARY_PATH=$DOCKER_WORKDIR/xsched-build/output/lib:\$LD_LIBRARY_PATH && \
    export LD_PRELOAD=$DOCKER_WORKDIR/xsched-build/output/lib/libhalhip.so:$DOCKER_WORKDIR/xsched-build/output/lib/libshimhip.so && \
    python3 $TEST_SCRIPT --duration 10 --output $RESULTS_DIR/xsched_quick_test.json
"

if [ $? -eq 0 ]; then
    echo
    echo "========================================================================"
    echo "✅ 快速测试成功"
    echo "========================================================================"
    echo
    echo "环境验证通过，可以运行完整测试:"
    echo "  ./test_xsched_only.sh"
    echo
else
    echo
    echo "❌ 快速测试失败"
    exit 1
fi
