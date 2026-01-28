#!/bin/bash
#####################################################################
# 宿主机脚本：在 Docker 容器内运行 Test 1.1
#####################################################################

set -e

DOCKER_CONTAINER="zhenflashinfer_v1"
SCRIPT_PATH="/data/dockercode/test_1_1_compilation_docker.sh"

echo "================================================"
echo "Running Test 1.1 in Docker Container"
echo "================================================"
echo ""
echo "Container: $DOCKER_CONTAINER"
echo "Script:    $SCRIPT_PATH"
echo ""

# 检查容器是否运行
if ! docker ps | grep -q "$DOCKER_CONTAINER"; then
    echo "❌ Error: Docker container '$DOCKER_CONTAINER' is not running!"
    echo ""
    echo "Please start the container first:"
    echo "  docker start $DOCKER_CONTAINER"
    exit 1
fi

# 将脚本复制到容器
echo "[1/2] Copying test script to container..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker cp "$SCRIPT_DIR/tests/test_1_1_compilation_docker.sh" \
    "$DOCKER_CONTAINER:/data/dockercode/test_1_1_compilation_docker.sh"
echo "  ✅ Script copied"

# 设置执行权限并运行
echo ""
echo "[2/2] Executing test in container..."
echo ""

docker exec -it "$DOCKER_CONTAINER" bash -c "
    chmod +x /data/dockercode/test_1_1_compilation_docker.sh
    /data/dockercode/test_1_1_compilation_docker.sh
"

EXIT_CODE=$?

echo ""
echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Test completed successfully"
    echo ""
    echo "View results:"
    echo "  docker exec $DOCKER_CONTAINER cat /data/dockercode/test_results/test_1_1_report.json"
else
    echo "❌ Test failed with exit code $EXIT_CODE"
    echo ""
    echo "Check logs in container:"
    echo "  docker exec $DOCKER_CONTAINER ls -lh /data/dockercode/xsched-test-build/*.log"
fi
echo "================================================"

exit $EXIT_CODE
