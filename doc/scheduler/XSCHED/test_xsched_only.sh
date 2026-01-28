#!/bin/bash
# 只测试 XSched（跳过 Baseline），方便调试
# 用法: ./test_xsched_only.sh

set -e

CONTAINER="zhenflashinfer_v1"
TEST_SCRIPT="test_phase4_dual_model_intensive.py"
DOCKER_WORKDIR="/data/dockercode"
RESULTS_DIR="/data/dockercode/test_results_phase4"

echo "========================================================================"
echo "XSched 测试 Only (调试模式)"
echo "========================================================================"
echo
echo "配置:"
echo "  Duration: 180s (3 minutes)"
echo "  High Priority: ResNet-18 (20 reqs/sec, 50ms interval)"
echo "  Low Priority:  ResNet-50 (batch=1024, continuous)"
echo

# 检查容器
if ! docker ps | grep -q "$CONTAINER"; then
    echo "❌ Docker 容器未运行"
    exit 1
fi

echo "✅ Docker 容器正在运行"
echo

# 复制测试脚本
echo "[1/3] 复制测试脚本..."
docker cp tests/test_phase4_dual_model_intensive.py "$CONTAINER:$DOCKER_WORKDIR/$TEST_SCRIPT"
echo "  ✅ 脚本已复制"
echo

# 验证 XSched 库文件
echo "[2/3] 验证 XSched 库文件..."
docker exec "$CONTAINER" bash -c "
    ls -lh $DOCKER_WORKDIR/xsched-build/output/lib/libhalhip.so
    ls -lh $DOCKER_WORKDIR/xsched-build/output/lib/libshimhip.so
    ls -lh $DOCKER_WORKDIR/xsched-build/output/lib/libpreempt.so
"
echo "  ✅ 库文件存在"
echo

# 测试基本 PyTorch + XSched
echo "测试基本 PyTorch + XSched..."
docker exec "$CONTAINER" bash -c "
    export LD_LIBRARY_PATH=$DOCKER_WORKDIR/xsched-build/output/lib:\$LD_LIBRARY_PATH
    export LD_PRELOAD=$DOCKER_WORKDIR/xsched-build/output/lib/libhalhip.so:$DOCKER_WORKDIR/xsched-build/output/lib/libshimhip.so
    python3 -c 'import torch; print(\"  PyTorch:\", torch.__version__); print(\"  CUDA:\", torch.cuda.is_available())'
"

if [ $? -ne 0 ]; then
    echo
    echo "❌ PyTorch + XSched 测试失败"
    echo
    echo "调试信息:"
    echo "检查库依赖..."
    docker exec "$CONTAINER" bash -c "
        export LD_LIBRARY_PATH=$DOCKER_WORKDIR/xsched-build/output/lib:\$LD_LIBRARY_PATH
        ldd $DOCKER_WORKDIR/xsched-build/output/lib/libshimhip.so
    "
    exit 1
fi

echo "  ✅ PyTorch + XSched 正常"
echo

# 运行 XSched 测试
echo "========================================================================"
echo "[3/3] 运行 XSched 测试 (20 req/s, batch=1024, 180s)"
echo "========================================================================"
echo "预计时间: 3 分钟"
echo

docker exec "$CONTAINER" bash -c "
    cd $DOCKER_WORKDIR && \
    export LD_LIBRARY_PATH=$DOCKER_WORKDIR/xsched-build/output/lib:\$LD_LIBRARY_PATH && \
    export LD_PRELOAD=$DOCKER_WORKDIR/xsched-build/output/lib/libhalhip.so:$DOCKER_WORKDIR/xsched-build/output/lib/libshimhip.so && \
    python3 $TEST_SCRIPT --duration 180 --output $RESULTS_DIR/xsched_intensive_result.json
"

XSCHED_EXIT=$?
echo

if [ $XSCHED_EXIT -ne 0 ]; then
    echo "❌ XSched 测试失败 (exit code: $XSCHED_EXIT)"
    exit 1
fi

echo "========================================================================"
echo "✅ XSched 测试完成"
echo "========================================================================"
echo

# 显示结果
echo "XSched 结果:"
docker exec "$CONTAINER" python3 << 'PYEOF'
import json
try:
    with open('/data/dockercode/test_results_phase4/xsched_intensive_result.json') as f:
        result = json.load(f)
    
    high = result['high_priority']
    low = result['low_priority']
    
    print(f"  High Priority:")
    print(f"    Requests:    {high['requests']}")
    print(f"    P99 Latency: {high['latency_p99_ms']:.2f} ms")
    print(f"    Avg Latency: {high['latency_avg_ms']:.2f} ms")
    print(f"    Throughput:  {high['throughput_rps']:.2f} req/s")
    print()
    print(f"  Low Priority:")
    print(f"    Iterations:  {low['iterations']}")
    print(f"    Throughput:  {low['throughput_ips']:.2f} iter/s")
    print(f"    Images/sec:  {low['images_per_sec']:.1f}")
except Exception as e:
    print(f"  无法读取结果: {e}")
PYEOF

echo
echo "结果文件: $RESULTS_DIR/xsched_intensive_result.json"
echo

# 如果有 baseline 结果，做对比
if docker exec "$CONTAINER" test -f "$RESULTS_DIR/baseline_intensive_result.json"; then
    echo "========================================================================"
    echo "对比 Baseline 结果"
    echo "========================================================================"
    echo
    
    docker exec "$CONTAINER" python3 << 'PYEOF'
import json
try:
    with open('/data/dockercode/test_results_phase4/baseline_intensive_result.json') as f:
        baseline = json.load(f)
    with open('/data/dockercode/test_results_phase4/xsched_intensive_result.json') as f:
        xsched = json.load(f)
    
    b_p99 = baseline['high_priority']['latency_p99_ms']
    x_p99 = xsched['high_priority']['latency_p99_ms']
    change = ((x_p99 - b_p99) / b_p99) * 100
    
    print(f"High Priority P99 Latency:")
    print(f"  Baseline: {b_p99:.2f} ms")
    print(f"  XSched:   {x_p99:.2f} ms")
    print(f"  Change:   {change:+.1f}%")
    
    if change < -10:
        print(f"\n🎉 XSched 改善了 {abs(change):.1f}%！")
    elif change < 0:
        print(f"\n✅ XSched 略有改善 ({abs(change):.1f}%)")
    else:
        print(f"\n⚠️  XSched P99 增加了 {change:.1f}%")
except Exception as e:
    print(f"无法对比: {e}")
PYEOF
    echo
fi

echo "========================================================================"
echo "✅ 完成"
echo "========================================================================"
