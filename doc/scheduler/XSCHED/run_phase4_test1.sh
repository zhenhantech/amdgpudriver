#!/bin/bash
#####################################################################
# Phase 4: XSched Paper Tests - Test 1
# 验证已有的 XSched 安装（无需重新编译）
#####################################################################

set -e

DOCKER_CONTAINER="zhenflashinfer_v1"
SCRIPT_PATH="/data/dockercode/test_phase4_1_verify_existing.sh"

echo "================================================"
echo "Phase 4 Test 1: Verify Existing XSched"
echo "================================================"
echo ""
echo "Background:"
echo "  Phase 1-2: PyTorch + XSched integration ✅"
echo "  Phase 3:   Real models testing         ✅"
echo "  Phase 4:   XSched paper tests          ← Current"
echo ""
echo "Container: $DOCKER_CONTAINER"
echo ""

# 检查容器是否运行
if ! docker ps | grep -q "$DOCKER_CONTAINER"; then
    echo "❌ Error: Docker container '$DOCKER_CONTAINER' is not running!"
    exit 1
fi

# 将脚本复制到容器
echo "[1/2] Copying test script to container..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker cp "$SCRIPT_DIR/tests/test_phase4_1_verify_existing.sh" \
    "$DOCKER_CONTAINER:/data/dockercode/test_phase4_1_verify_existing.sh"
echo "  ✅ Script copied"

# 运行测试
echo ""
echo "[2/2] Executing test in container..."
echo ""

docker exec -it "$DOCKER_CONTAINER" bash -c "
    chmod +x /data/dockercode/test_phase4_1_verify_existing.sh
    /data/dockercode/test_phase4_1_verify_existing.sh
"

EXIT_CODE=$?

echo ""
echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Test completed successfully"
    echo ""
    echo "View results:"
    echo "  docker exec $DOCKER_CONTAINER cat /data/dockercode/test_results_phase4/phase4_test1_report.json"
    echo ""
    echo "Next step:"
    echo "  ./run_phase4_test2.sh  # Runtime overhead measurement"
else
    echo "❌ Test failed with exit code $EXIT_CODE"
    echo ""
    echo "Possible reasons:"
    echo "  - Phase 2 (PyTorch + XSched) not completed"
    echo "  - XSched libraries not built"
    echo "  - Missing /data/dockercode/xsched-official or xsched-build"
fi
echo "================================================"

exit $EXIT_CODE
