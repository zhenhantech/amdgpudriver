#!/bin/bash
# 修复 symbol error 并重新运行高负载测试

set -e

CONTAINER="zhenflashinfer_v1"
DOCKER_WORKDIR="/data/dockercode"

echo "========================================================================"
echo "修复 Symbol Error 并重新运行 XSched 高负载测试"
echo "========================================================================"
echo

# Step 1: 验证库文件
echo "[1/4] 验证 XSched 库文件..."
docker exec "$CONTAINER" bash -c "
    ls -lh /data/dockercode/xsched-build/output/lib/libshimhip.so
    ls -lh /data/dockercode/xsched-build/output/lib/libhalhip.so
    ls -lh /data/dockercode/xsched-build/output/lib/libpreempt.so
"

echo
echo "  ✅ 库文件存在"
echo

# Step 2: 测试简单的 PyTorch 命令
echo "[2/4] 测试 XSched 环境..."
docker exec "$CONTAINER" bash -c "
    export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:\$LD_LIBRARY_PATH && \
    export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so && \
    python3 -c 'import torch; print(\"PyTorch:\", torch.__version__); print(\"CUDA:\", torch.cuda.is_available())'
" 2>&1 | head -20

if [ $? -eq 0 ]; then
    echo "  ✅ XSched 环境正常"
else
    echo "  ❌ XSched 环境有问题"
    echo
    echo "  尝试修复: 检查 ldd"
    docker exec "$CONTAINER" bash -c "
        export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:\$LD_LIBRARY_PATH && \
        ldd /data/dockercode/xsched-build/output/lib/libshimhip.so | grep 'not found'
    "
    exit 1
fi

echo

# Step 3: 运行 XSched 高负载测试
echo "[3/4] 运行 XSched 高负载测试..."
echo "  配置: 20 req/s, batch=1024, 180s"
echo "  预计时间: 3 分钟"
echo

docker exec "$CONTAINER" bash -c '
    cd /data/dockercode && \
    export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:$LD_LIBRARY_PATH && \
    export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so && \
    python3 test_phase4_dual_model_intensive.py \
      --duration 180 \
      --output /data/dockercode/test_results_phase4/xsched_intensive_result.json
'

XSCHED_EXIT=$?

if [ $XSCHED_EXIT -ne 0 ]; then
    echo
    echo "❌ XSched 测试失败"
    echo
    echo "调试信息:"
    docker exec "$CONTAINER" bash -c "
        export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:\$LD_LIBRARY_PATH && \
        ldd /data/dockercode/xsched-build/output/lib/libshimhip.so
    "
    exit 1
else
    echo
    echo "✅ XSched 测试完成"
fi

echo

# Step 4: 对比结果
echo "[4/4] 对比结果..."
echo

docker exec "$CONTAINER" python3 << 'PYEOF'
import json

try:
    with open('/data/dockercode/test_results_phase4/baseline_intensive_result.json') as f:
        baseline = json.load(f)
    with open('/data/dockercode/test_results_phase4/xsched_intensive_result.json') as f:
        xsched = json.load(f)
    
    print("=" * 70)
    print("COMPARISON: XSched vs Baseline (Intensive)")
    print("=" * 70)
    print()
    
    # 高优先级
    print("High Priority (ResNet-18, 20 req/s):")
    print("-" * 70)
    
    b_h = baseline['high_priority']
    x_h = xsched['high_priority']
    
    print(f"  P99 Latency:")
    print(f"    Baseline: {b_h['latency_p99_ms']:.2f} ms")
    print(f"    XSched:   {x_h['latency_p99_ms']:.2f} ms")
    change = ((x_h['latency_p99_ms'] - b_h['latency_p99_ms']) / b_h['latency_p99_ms']) * 100
    print(f"    Change:   {change:+.1f}%")
    
    print(f"\n  Max Latency:")
    print(f"    Baseline: {b_h['latency_max_ms']:.2f} ms")
    print(f"    XSched:   {x_h['latency_max_ms']:.2f} ms")
    
    print(f"\n  Throughput:")
    print(f"    Baseline: {b_h['throughput_rps']:.2f} req/s")
    print(f"    XSched:   {x_h['throughput_rps']:.2f} req/s")
    
    print()
    print()
    
    # 低优先级
    print("Low Priority (ResNet-50, batch=1024):")
    print("-" * 70)
    
    b_l = baseline['low_priority']
    x_l = xsched['low_priority']
    
    print(f"  Throughput:")
    print(f"    Baseline: {b_l['throughput_ips']:.2f} iter/s")
    print(f"    XSched:   {x_l['throughput_ips']:.2f} iter/s")
    change_low = ((x_l['throughput_ips'] - b_l['throughput_ips']) / b_l['throughput_ips']) * 100
    print(f"    Change:   {change_low:+.1f}%")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if change < -50:
        print(f"🎉 XSched 改善 {abs(change):.1f}% - 巨大提升！")
    elif change < -20:
        print(f"✅ XSched 改善 {abs(change):.1f}% - 显著提升！")
    elif change < 0:
        print(f"✅ XSched 改善 {abs(change):.1f}%")
    else:
        print(f"⚠️  XSched P99 增加 {change:.1f}%")
    
    print("=" * 70)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

PYEOF

echo
echo "========================================================================"
echo "✅ 测试完成"
echo "========================================================================"
echo
echo "查看详细结果:"
echo "  docker exec $CONTAINER cat /data/dockercode/test_results_phase4/baseline_intensive_result.json"
echo "  docker exec $CONTAINER cat /data/dockercode/test_results_phase4/xsched_intensive_result.json"
echo
