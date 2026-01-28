#!/bin/bash
# 在 Docker 内部运行 XSched 高负载测试（最终修复版本）
# 用法: bash /data/dockercode/run_intensive_xsched_final.sh

set -e

cd /data/dockercode

echo "========================================================================"
echo "XSched 高负载测试（Docker 内部执行 - 最终版）"
echo "========================================================================"
echo

# LD_LIBRARY_PATH 可以 export（因为是标准库路径）
export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:$LD_LIBRARY_PATH

# 需要 preload 两个库：libhalhip.so 和 libshimhip.so
# libhalhip.so 包含必需的符号（HipCommand typeinfo）
XSCHED_PRELOAD="/data/dockercode/xsched-build/output/lib/libhalhip.so:/data/dockercode/xsched-build/output/lib/libshimhip.so"

echo "环境变量:"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "  XSCHED_PRELOAD:"
echo "    - libhalhip.so (提供 HipCommand 符号)"
echo "    - libshimhip.so (HIP API 拦截)"
echo

# 验证库加载（不使用 LD_PRELOAD）
echo "验证库依赖..."
echo "  检查 libshimhip.so:"
ldd /data/dockercode/xsched-build/output/lib/libshimhip.so | grep -E "libpreempt" | head -1

echo "  检查 libhalhip.so:"
ls -lh /data/dockercode/xsched-build/output/lib/libhalhip.so | awk '{print "    Size:", $5}'

echo "  ✅ 库文件正常"
echo

# 测试基本功能（使用 LD_PRELOAD）
echo "测试基本 PyTorch 功能（带 XSched）..."
LD_PRELOAD=$XSCHED_PRELOAD python3 -c 'import torch; print("  PyTorch:", torch.__version__); print("  CUDA:", torch.cuda.is_available())'

if [ $? -ne 0 ]; then
    echo "❌ PyTorch 测试失败"
    exit 1
fi

echo "  ✅ PyTorch + XSched 正常"
echo

# 运行高负载测试（使用 LD_PRELOAD）
echo "========================================================================"
echo "开始高负载测试 (20 req/s, batch=1024, 180s)"
echo "========================================================================"
echo "预计时间: 3 分钟"
echo

LD_PRELOAD=$XSCHED_PRELOAD python3 test_phase4_dual_model_intensive.py \
  --duration 180 \
  --output /data/dockercode/test_results_phase4/xsched_intensive_result.json

echo
echo "========================================================================"
echo "✅ 测试完成"
echo "========================================================================"
echo
echo "结果文件: /data/dockercode/test_results_phase4/xsched_intensive_result.json"
echo

# 显示关键指标
echo "关键指标:"
python3 << 'PYEOF'
import json
try:
    with open('/data/dockercode/test_results_phase4/xsched_intensive_result.json') as f:
        result = json.load(f)
    
    high = result['high_priority']
    low = result['low_priority']
    
    print(f"  High Priority P99: {high['latency_p99_ms']:.2f} ms")
    print(f"  High Priority Throughput: {high['throughput_rps']:.2f} req/s")
    print(f"  Low Priority Throughput: {low['throughput_ips']:.2f} iter/s")
except Exception as e:
    print(f"  (无法读取结果: {e})")
PYEOF

echo
