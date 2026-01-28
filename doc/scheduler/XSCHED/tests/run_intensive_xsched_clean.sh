#!/bin/bash
# 在 Docker 内部运行 XSched 高负载测试（清洁版本）
# 用法: bash /data/dockercode/run_intensive_xsched_clean.sh

set -e

cd /data/dockercode

echo "========================================================================"
echo "XSched 高负载测试（Docker 内部执行）"
echo "========================================================================"
echo

# 先清除可能存在的环境变量
unset LD_PRELOAD

# 设置库路径（但不设置 LD_PRELOAD，让 Python 脚本启动时才设置）
export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:$LD_LIBRARY_PATH

echo "环境变量:"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo

# 验证库文件存在
echo "验证库文件..."
if [ ! -f /data/dockercode/xsched-build/output/lib/libshimhip.so ]; then
    echo "❌ libshimhip.so 不存在"
    exit 1
fi

if [ ! -f /data/dockercode/xsched-build/output/lib/libpreempt.so ]; then
    echo "❌ libpreempt.so 不存在"
    exit 1
fi

if [ ! -f /data/dockercode/xsched-build/output/lib/libhalhip.so ]; then
    echo "❌ libhalhip.so 不存在"
    exit 1
fi

echo "  ✅ 库文件存在"
echo

# 验证库依赖（不使用 LD_PRELOAD）
echo "验证库依赖..."
LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib ldd /data/dockercode/xsched-build/output/lib/libshimhip.so | grep -E "libpreempt|libhalhip"

if LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib ldd /data/dockercode/xsched-build/output/lib/libshimhip.so | grep -q "not found"; then
    echo "❌ 库依赖有问题"
    LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib ldd /data/dockercode/xsched-build/output/lib/libshimhip.so
    exit 1
fi

echo "  ✅ 库依赖正常"
echo

# 测试基本功能（不使用 XSched）
echo "测试基本 PyTorch 功能（无 XSched）..."
python3 -c 'import torch; print("  PyTorch:", torch.__version__); print("  CUDA:", torch.cuda.is_available())'

if [ $? -ne 0 ]; then
    echo "❌ PyTorch 测试失败"
    exit 1
fi

echo "  ✅ PyTorch 正常"
echo

# 测试 XSched 环境
echo "测试 XSched 环境..."
LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so \
  python3 -c 'import torch; print("  PyTorch with XSched:", torch.__version__); print("  CUDA:", torch.cuda.is_available())' 2>&1 | head -5

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ XSched 环境测试失败"
    exit 1
fi

echo "  ✅ XSched 环境正常"
echo

# 运行高负载测试（只在 Python 进程中设置 LD_PRELOAD）
echo "========================================================================"
echo "开始高负载测试 (20 req/s, batch=1024, 180s)"
echo "========================================================================"
echo
echo "预计时间: 3 分钟"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo

LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so \
  python3 test_phase4_dual_model_intensive.py \
    --duration 180 \
    --output /data/dockercode/test_results_phase4/xsched_intensive_result.json

TEST_EXIT=$?

echo
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo

if [ $TEST_EXIT -ne 0 ]; then
    echo "❌ 测试失败 (exit code: $TEST_EXIT)"
    exit 1
fi

echo "========================================================================"
echo "✅ 测试完成"
echo "========================================================================"
echo
echo "结果文件: /data/dockercode/test_results_phase4/xsched_intensive_result.json"
echo

# 显示结果摘要
if [ -f /data/dockercode/test_results_phase4/xsched_intensive_result.json ]; then
    echo "结果摘要:"
    python3 << 'PYEOF'
import json
try:
    with open('/data/dockercode/test_results_phase4/xsched_intensive_result.json') as f:
        result = json.load(f)
    
    if 'high_priority' in result:
        h = result['high_priority']
        print(f"  High Priority (ResNet-18):")
        print(f"    Requests:   {h.get('requests', 'N/A')}")
        print(f"    P99:        {h.get('latency_p99_ms', 'N/A'):.2f} ms")
        print(f"    Throughput: {h.get('throughput_rps', 'N/A'):.2f} req/s")
    
    if 'low_priority' in result:
        l = result['low_priority']
        print(f"  Low Priority (ResNet-50):")
        print(f"    Iterations: {l.get('iterations', 'N/A')}")
        print(f"    Throughput: {l.get('throughput_ips', 'N/A'):.2f} iter/s")

except Exception as e:
    print(f"  无法解析结果: {e}")
PYEOF
fi

echo
