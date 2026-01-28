#!/bin/bash
# 运行 Test 3，启用 XSched 的 DEBUG 日志级别

set -e

CONTAINER="zhenflashinfer_v1"
TEST_SCRIPT="test_phase4_dual_model.py"
DOCKER_WORKDIR="/data/dockercode"

echo "========================================================================"
echo "Phase 4 Test 3: 启用 DEBUG 日志运行"
echo "========================================================================"
echo
echo "目的: 查看 XSched 的详细调度日志"
echo "  - 是否有优先级设置？"
echo "  - 高优先级任务如何影响低优先级？"
echo "  - 任务提交和等待的日志"
echo
echo "日志级别: DEBG (最详细)"
echo "环境变量: XLOG_LEVEL=DEBG"
echo

# 检查 Docker 容器
if ! docker ps | grep -q "$CONTAINER"; then
    echo "❌ Error: Docker container '$CONTAINER' is not running"
    exit 1
fi

echo "========================================================================"
echo "运行 XSched 测试 (启用 DEBUG 日志)"
echo "========================================================================"
echo

# 运行测试，设置 XLOG_LEVEL=DEBG
docker exec "$CONTAINER" bash -c "
    cd $DOCKER_WORKDIR && \
    export LD_LIBRARY_PATH=$DOCKER_WORKDIR/xsched-build/output/lib:\$LD_LIBRARY_PATH && \
    export LD_PRELOAD=$DOCKER_WORKDIR/xsched-build/output/lib/libshimhip.so && \
    export XLOG_LEVEL=DEBG && \
    python3 $TEST_SCRIPT --duration 30 --output /tmp/xsched_debug.json 2>&1 | tee /tmp/xsched_debug_log.txt
"

echo
echo "========================================================================"
echo "测试完成"
echo "========================================================================"
echo
echo "查看完整日志:"
echo "  docker exec $CONTAINER cat /tmp/xsched_debug_log.txt"
echo
echo "查看调度相关的日志:"
echo "  docker exec $CONTAINER grep -i 'priority\\|enqueue\\|wait\\|schedule' /tmp/xsched_debug_log.txt"
echo
echo "查看流创建日志:"
echo "  docker exec $CONTAINER grep -i 'stream' /tmp/xsched_debug_log.txt | head -20"
echo
echo "查看 kernel 启动日志:"
echo "  docker exec $CONTAINER grep -i 'launch\\|kernel' /tmp/xsched_debug_log.txt | head -20"
echo
echo "========================================================================"
