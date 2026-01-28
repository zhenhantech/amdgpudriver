#!/bin/bash
# Phase 4 Test: Dual Model Priority Scheduling (Intensive Configuration)
# 高负载测试：ResNet-18 (20 req/s) + ResNet-50 (batch=1024), 3 分钟

set -e

CONTAINER="zhenflashinfer_v1"
TEST_SCRIPT="test_phase4_dual_model_intensive.py"
DOCKER_WORKDIR="/data/dockercode"
RESULTS_DIR="/data/dockercode/test_results_phase4"

echo "========================================================================"
echo "Phase 4: Dual Model Priority Scheduling Test (Intensive)"
echo "========================================================================"
echo
echo "Test Configuration:"
echo "  Duration: 180s (3 minutes)"
echo "  High Priority: ResNet-18 (20 reqs/sec, 50ms interval)"
echo "  Low Priority:  ResNet-50 (batch=1024, continuous)"
echo

# 检查 Docker 容器
if ! docker ps | grep -q "$CONTAINER"; then
    echo "❌ Error: Docker container '$CONTAINER' is not running"
    exit 1
fi

# 复制测试脚本
echo "[1/5] Copying test script to container..."
docker cp "tests/$TEST_SCRIPT" "$CONTAINER:$DOCKER_WORKDIR/$TEST_SCRIPT"
if [ $? -eq 0 ]; then
    echo "  ✅ Script copied"
else
    echo "  ❌ Failed to copy script"
    exit 1
fi

echo

# 运行 Baseline 测试
echo "========================================================================"
echo "[2/5] Running BASELINE test (Native scheduler, no XSched)"
echo "========================================================================"
echo

docker exec "$CONTAINER" bash -c "
    cd $DOCKER_WORKDIR && \
    unset LD_PRELOAD && \
    python3 $TEST_SCRIPT --duration 180 --output $RESULTS_DIR/baseline_intensive_result.json
"

BASELINE_EXIT=$?
echo

if [ $BASELINE_EXIT -ne 0 ]; then
    echo "⚠️  Baseline test failed, but continuing..."
else
    echo "✅ Baseline test completed"
fi

echo

# 运行 XSched 测试
echo "========================================================================"
echo "[3/5] Running XSCHED test (With priority scheduling)"
echo "========================================================================"
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
    echo "❌ XSched test failed"
    exit 1
else
    echo "✅ XSched test completed"
fi

echo

# 对比结果
echo "========================================================================"
echo "[4/5] Comparing results..."
echo "========================================================================"
echo

# 创建对比脚本
cat > /tmp/compare_intensive.py << 'EOF'
import json
import sys

def load_result(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

baseline = load_result('/data/dockercode/test_results_phase4/baseline_intensive_result.json')
xsched = load_result('/data/dockercode/test_results_phase4/xsched_intensive_result.json')

if not baseline or not xsched:
    print("❌ Failed to load results")
    sys.exit(1)

print("=" * 70)
print("COMPARISON: XSched vs Baseline (Intensive Configuration)")
print("=" * 70)
print()

# 高优先级对比
print("High Priority Task (ResNet-18, 20 req/s):")
print("-" * 70)
print(f"  {'Metric':<25} {'Baseline':>12} {'XSched':>12} {'Change':>12}")
print("  " + "-" * 58)

b_high = baseline['high_priority']
x_high = xsched['high_priority']

if 'error' not in b_high and 'error' not in x_high:
    metrics = [
        ('Requests', 'requests', ''),
        ('P99 Latency (ms)', 'latency_p99_ms', 'ms'),
        ('Avg Latency (ms)', 'latency_avg_ms', 'ms'),
        ('Throughput (rps)', 'throughput_rps', 'rps'),
    ]
    
    for label, key, unit in metrics:
        b_val = b_high[key]
        x_val = x_high[key]
        
        if isinstance(b_val, (int, float)) and isinstance(x_val, (int, float)):
            if key == 'requests':
                change = ""
                status = ""
            else:
                change_pct = ((x_val - b_val) / b_val) * 100
                if abs(change_pct) < 0.1:
                    change = f"{change_pct:+.1f}%"
                else:
                    change = f"{change_pct:+.1f}%"
                
                if 'latency' in key.lower():
                    status = " ✅" if change_pct < 10 else " ⚠️"
                else:
                    status = ""
            
            print(f"  {label:<25} {b_val:>12.2f} {x_val:>12.2f} {change:>11}{status}")
        else:
            print(f"  {label:<25} {str(b_val):>12} {str(x_val):>12}")

print()
print()

# 低优先级对比
print("Low Priority Task (ResNet-50, batch=1024):")
print("-" * 70)
print(f"  {'Metric':<25} {'Baseline':>12} {'XSched':>12} {'Change':>12}")
print("  " + "-" * 58)

b_low = baseline['low_priority']
x_low = xsched['low_priority']

if 'error' not in b_low and 'error' not in x_low:
    metrics = [
        ('Iterations', 'iterations', ''),
        ('Throughput (ips)', 'throughput_ips', 'ips'),
        ('Images/sec', 'images_per_sec', ''),
    ]
    
    for label, key, unit in metrics:
        b_val = b_low[key]
        x_val = x_low[key]
        
        if isinstance(b_val, (int, float)) and isinstance(x_val, (int, float)):
            if key == 'iterations':
                change = ""
                status = ""
            else:
                change_pct = ((x_val - b_val) / b_val) * 100
                change = f"{change_pct:+.1f}%"
                status = " ✅" if change_pct > -10 else " ⚠️"
            
            print(f"  {label:<25} {b_val:>12.1f} {x_val:>12.1f} {change:>11}{status}")
        else:
            print(f"  {label:<25} {str(b_val):>12} {str(x_val):>12}")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

# 判断
if 'error' not in b_high and 'error' not in x_high:
    b_p99 = b_high['latency_p99_ms']
    x_p99 = x_high['latency_p99_ms']
    
    if x_p99 <= b_p99 * 1.1:
        print("✅ High priority latency: GOOD (XSched P99 <= 110% baseline)")
    else:
        print("⚠️  High priority latency: WARNING (XSched P99 > 110% baseline)")

if 'error' not in b_low and 'error' not in x_low:
    b_tput = b_low['throughput_ips']
    x_tput = x_low['throughput_ips']
    retention = (x_tput / b_tput) * 100
    
    if retention >= 30:
        print(f"✅ Low priority throughput: GOOD (XSched = {retention:.1f}% of baseline, >= 30%)")
    else:
        print(f"⚠️  Low priority throughput: WARNING (XSched = {retention:.1f}% of baseline, < 30%)")

print()

if x_p99 <= b_p99 * 1.1 and retention >= 30:
    print("🎉 Overall: PASS")
else:
    print("⚠️  Overall: NEEDS REVIEW")

print()
print("Key findings:")
print("  - Test duration: 3 minutes (180s)")
print("  - High priority: 20 req/s (2x original load)")
print("  - Low priority: batch=1024 (128x original load)")
print("=" * 70)
EOF

# 复制并运行对比脚本
docker cp /tmp/compare_intensive.py "$CONTAINER:/tmp/compare_intensive.py"
docker exec "$CONTAINER" python3 /tmp/compare_intensive.py

echo

# 生成最终报告
echo "========================================================================"
echo "[5/5] Generating final report..."
echo "========================================================================"
echo

docker exec "$CONTAINER" bash -c "
python3 << 'PYEOF'
import json
import sys

try:
    with open('/data/dockercode/test_results_phase4/baseline_intensive_result.json') as f:
        baseline = json.load(f)
    with open('/data/dockercode/test_results_phase4/xsched_intensive_result.json') as f:
        xsched = json.load(f)
    
    report = f'''# Phase 4 Dual Model Test Report (Intensive Configuration)

**Date**: {baseline['timestamp']}
**Duration**: 180s (3 minutes)

## Configuration

- **High Priority**: ResNet-18, 20 req/s (50ms interval)
- **Low Priority**: ResNet-50, batch=1024, continuous

## Baseline Results (Native Scheduler)

### High Priority
- Requests: {baseline['high_priority']['requests']}
- P99 Latency: {baseline['high_priority']['latency_p99_ms']:.2f} ms
- Throughput: {baseline['high_priority']['throughput_rps']:.2f} req/s

### Low Priority
- Iterations: {baseline['low_priority']['iterations']}
- Throughput: {baseline['low_priority']['throughput_ips']:.2f} iter/s

## XSched Results

### High Priority
- Requests: {xsched['high_priority']['requests']}
- P99 Latency: {xsched['high_priority']['latency_p99_ms']:.2f} ms
- Throughput: {xsched['high_priority']['throughput_rps']:.2f} req/s

### Low Priority
- Iterations: {xsched['low_priority']['iterations']}
- Throughput: {xsched['low_priority']['throughput_ips']:.2f} iter/s

## Comparison

- High Priority P99 Change: {((xsched['high_priority']['latency_p99_ms'] - baseline['high_priority']['latency_p99_ms']) / baseline['high_priority']['latency_p99_ms'] * 100):+.1f}%
- Low Priority Throughput Change: {((xsched['low_priority']['throughput_ips'] - baseline['low_priority']['throughput_ips']) / baseline['low_priority']['throughput_ips'] * 100):+.1f}%
'''
    
    with open('/data/dockercode/test_results_phase4/phase4_dual_model_intensive_report.md', 'w') as f:
        f.write(report)
    
    print('✅ Report generated: /data/dockercode/test_results_phase4/phase4_dual_model_intensive_report.md')
    
except Exception as e:
    print(f'Error generating report: {e}')
    sys.exit(1)
PYEOF
"

echo

echo "========================================================================"
echo "✅ Phase 4 Dual Model Test (Intensive): COMPLETED"
echo "========================================================================"
echo
echo "Results saved in:"
echo "  - Baseline:  $CONTAINER:$RESULTS_DIR/baseline_intensive_result.json"
echo "  - XSched:    $CONTAINER:$RESULTS_DIR/xsched_intensive_result.json"
echo "  - Report:    $CONTAINER:$RESULTS_DIR/phase4_dual_model_intensive_report.md"
echo
echo "View results:"
echo "  docker exec $CONTAINER cat $RESULTS_DIR/baseline_intensive_result.json"
echo "  docker exec $CONTAINER cat $RESULTS_DIR/xsched_intensive_result.json"
echo "  docker exec $CONTAINER cat $RESULTS_DIR/phase4_dual_model_intensive_report.md"
echo
echo "Next step: Analyze results and compare with original test (10 req/s, batch=8)"
echo "========================================================================"
