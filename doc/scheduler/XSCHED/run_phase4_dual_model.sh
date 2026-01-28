#!/bin/bash
#####################################################################
# Phase 4: 运行双模型优先级测试
# 
# 测试场景:
#   - 高优先级: ResNet-18 (10 reqs/sec)
#   - 低优先级: ResNet-50 (连续运行)
# 
# 运行模式:
#   1. Baseline (无 XSched)
#   2. XSched (有优先级调度)
#   3. 对比分析
#####################################################################

set -e

DOCKER_CONTAINER="zhenflashinfer_v1"
TEST_DURATION=60  # 秒
RESULTS_DIR="/data/dockercode/test_results_phase4"

echo "========================================================================"
echo "Phase 4: Dual Model Priority Scheduling Test"
echo "========================================================================"
echo ""
echo "Test Configuration:"
echo "  Duration: ${TEST_DURATION}s"
echo "  High Priority: ResNet-18 (10 reqs/sec)"
echo "  Low Priority:  ResNet-50 (continuous)"
echo ""

# 检查容器
if ! docker ps | grep -q "$DOCKER_CONTAINER"; then
    echo "❌ Error: Docker container '$DOCKER_CONTAINER' is not running!"
    exit 1
fi

# 复制测试脚本
echo "[1/5] Copying test script to container..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker cp "$SCRIPT_DIR/tests/test_phase4_dual_model.py" \
    "$DOCKER_CONTAINER:/data/dockercode/test_phase4_dual_model.py"
echo "  ✅ Script copied"

# 创建结果目录
docker exec "$DOCKER_CONTAINER" mkdir -p "$RESULTS_DIR"

echo ""
echo "========================================================================"
echo "[2/5] Running BASELINE test (Native scheduler, no XSched)"
echo "========================================================================"
echo ""

docker exec "$DOCKER_CONTAINER" bash -c "
    # 取消 LD_PRELOAD
    unset LD_PRELOAD
    
    # 运行测试
    cd /data/dockercode
    python3 test_phase4_dual_model.py \
        --duration $TEST_DURATION \
        --output $RESULTS_DIR/baseline_result.json
"

BASELINE_EXIT=$?
if [ $BASELINE_EXIT -ne 0 ]; then
    echo "❌ Baseline test failed!"
    exit 1
fi

echo ""
echo "========================================================================"
echo "[3/5] Running XSCHED test (With priority scheduling)"
echo "========================================================================"
echo ""

docker exec "$DOCKER_CONTAINER" bash -c "
    # 设置 XSched 环境
    export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:\$LD_LIBRARY_PATH
    export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so
    
    # 运行测试
    cd /data/dockercode
    python3 test_phase4_dual_model.py \
        --duration $TEST_DURATION \
        --output $RESULTS_DIR/xsched_result.json
"

XSCHED_EXIT=$?
if [ $XSCHED_EXIT -ne 0 ]; then
    echo "❌ XSched test failed!"
    exit 1
fi

echo ""
echo "========================================================================"
echo "[4/5] Comparing results..."
echo "========================================================================"
echo ""

# 创建对比脚本
docker exec "$DOCKER_CONTAINER" bash -c "cat > /data/dockercode/compare_results.py << 'EOF'
import json
import sys

def load_result(filename):
    with open(filename, 'r') as f:
        return json.load(f)

baseline = load_result('$RESULTS_DIR/baseline_result.json')
xsched = load_result('$RESULTS_DIR/xsched_result.json')

print('=' * 70)
print('COMPARISON: XSched vs Baseline')
print('=' * 70)

# 高优先级对比
print('\nHigh Priority Task (ResNet-18):')
print('-' * 70)

if 'error' not in baseline['high_priority'] and 'error' not in xsched['high_priority']:
    b_high = baseline['high_priority']
    x_high = xsched['high_priority']
    
    print(f'  Metric             Baseline      XSched        Change')
    print(f'  ' + '-' * 60)
    
    # P99 延迟
    b_p99 = b_high['latency_p99_ms']
    x_p99 = x_high['latency_p99_ms']
    change_p99 = ((x_p99 - b_p99) / b_p99) * 100
    status_p99 = '✅' if change_p99 < 10 else '⚠️ '
    print(f'  P99 Latency (ms)   {b_p99:>8.2f}      {x_p99:>8.2f}      {change_p99:>+6.1f}% {status_p99}')
    
    # 平均延迟
    b_avg = b_high['latency_avg_ms']
    x_avg = x_high['latency_avg_ms']
    change_avg = ((x_avg - b_avg) / b_avg) * 100
    print(f'  Avg Latency (ms)   {b_avg:>8.2f}      {x_avg:>8.2f}      {change_avg:>+6.1f}%')
    
    # 吞吐量
    b_thr = b_high['throughput_rps']
    x_thr = x_high['throughput_rps']
    change_thr = ((x_thr - b_thr) / b_thr) * 100
    print(f'  Throughput (rps)   {b_thr:>8.2f}      {x_thr:>8.2f}      {change_thr:>+6.1f}%')

# 低优先级对比
print('\n\nLow Priority Task (ResNet-50):')
print('-' * 70)

if 'error' not in baseline['low_priority'] and 'error' not in xsched['low_priority']:
    b_low = baseline['low_priority']
    x_low = xsched['low_priority']
    
    print(f'  Metric             Baseline      XSched        Change')
    print(f'  ' + '-' * 60)
    
    # 吞吐量
    b_thr = b_low['throughput_ips']
    x_thr = x_low['throughput_ips']
    change_thr = ((x_thr - b_thr) / b_thr) * 100
    status_thr = '✅' if x_thr > b_thr * 0.3 else '⚠️ '  # 至少保留 30%
    print(f'  Throughput (ips)   {b_thr:>8.2f}      {x_thr:>8.2f}      {change_thr:>+6.1f}% {status_thr}')
    
    # 图像/秒
    b_img = b_low['images_per_sec']
    x_img = x_low['images_per_sec']
    change_img = ((x_img - b_img) / b_img) * 100
    print(f'  Images/sec         {b_img:>8.1f}      {x_img:>8.1f}      {change_img:>+6.1f}%')

print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)

# 判断标准
success = True

if 'error' not in baseline['high_priority'] and 'error' not in xsched['high_priority']:
    x_p99 = xsched['high_priority']['latency_p99_ms']
    b_p99 = baseline['high_priority']['latency_p99_ms']
    
    if x_p99 < b_p99 * 1.1:  # XSched P99 不超过 baseline 的 110%
        print('✅ High priority latency: GOOD (XSched P99 < 110% baseline)')
    else:
        print(f'⚠️  High priority latency: Could be better (XSched P99 = {(x_p99/b_p99)*100:.1f}% of baseline)')
        success = False

if 'error' not in baseline['low_priority'] and 'error' not in xsched['low_priority']:
    x_thr = xsched['low_priority']['throughput_ips']
    b_thr = baseline['low_priority']['throughput_ips']
    
    if x_thr > b_thr * 0.3:  # 低优先级至少保留 30% 吞吐量
        print(f'✅ Low priority throughput: GOOD (XSched = {(x_thr/b_thr)*100:.1f}% of baseline, > 30%)')
    else:
        print(f'⚠️  Low priority throughput: Low (XSched = {(x_thr/b_thr)*100:.1f}% of baseline, < 30%)')
        success = False

print('')
if success:
    print('🎉 Overall: PASS')
    print('')
    print('Key findings:')
    print('  - High priority task maintains good latency')
    print('  - Low priority task is not starved')
    print('  - XSched priority scheduling is working')
else:
    print('⚠️  Overall: Needs Investigation')
    print('')
    print('Possible issues:')
    print('  - XSched overhead too high')
    print('  - Priority scheduling not effective')
    print('  - Need to adjust configuration')

print('=' * 70)

sys.exit(0 if success else 1)
EOF

python3 /data/dockercode/compare_results.py
"

COMPARE_EXIT=$?

echo ""
echo "========================================================================"
echo "[5/5] Generating final report..."
echo "========================================================================"
echo ""

docker exec "$DOCKER_CONTAINER" bash -c "
cat > $RESULTS_DIR/phase4_dual_model_report.md << 'EOF'
# Phase 4 Dual Model Test Report

**Date**: $(date)
**Duration**: ${TEST_DURATION}s
**Container**: $(hostname)

## Test Configuration

- **High Priority Task**: ResNet-18, 10 req/s
- **Low Priority Task**: ResNet-50, continuous

## Results

### Baseline (Native Scheduler)

\`\`\`json
$(cat $RESULTS_DIR/baseline_result.json | python3 -m json.tool)
\`\`\`

### XSched (Priority Scheduler)

\`\`\`json
$(cat $RESULTS_DIR/xsched_result.json | python3 -m json.tool)
\`\`\`

## Comparison

Run \`python3 compare_results.py\` to see detailed comparison.

EOF

echo '✅ Report generated: $RESULTS_DIR/phase4_dual_model_report.md'
"

echo ""
echo "========================================================================"
if [ $COMPARE_EXIT -eq 0 ]; then
    echo "✅ Phase 4 Dual Model Test: PASSED"
else
    echo "⚠️  Phase 4 Dual Model Test: Completed (needs review)"
fi
echo "========================================================================"
echo ""
echo "Results saved in:"
echo "  - Baseline:  $DOCKER_CONTAINER:$RESULTS_DIR/baseline_result.json"
echo "  - XSched:    $DOCKER_CONTAINER:$RESULTS_DIR/xsched_result.json"
echo "  - Report:    $DOCKER_CONTAINER:$RESULTS_DIR/phase4_dual_model_report.md"
echo ""
echo "View results:"
echo "  docker exec $DOCKER_CONTAINER cat $RESULTS_DIR/baseline_result.json"
echo "  docker exec $DOCKER_CONTAINER cat $RESULTS_DIR/xsched_result.json"
echo ""
echo "Next step: Analyze results and proceed to Test 4.2 (Multi-tenant)"

exit 0
