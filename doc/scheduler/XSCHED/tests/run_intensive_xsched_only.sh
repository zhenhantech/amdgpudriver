#!/bin/bash
# 在 Docker 内部运行 XSched 高负载测试
# 用法: docker exec zhenflashinfer_v1 bash /data/dockercode/run_intensive_xsched_only.sh

set -e

cd /data/dockercode

echo "========================================================================"
echo "XSched 高负载测试（Docker 内部执行）"
echo "========================================================================"
echo

# 设置环境变量
export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so

echo "环境变量:"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "  LD_PRELOAD: $LD_PRELOAD"
echo

# 验证库加载
echo "验证库依赖..."
ldd /data/dockercode/xsched-build/output/lib/libshimhip.so | grep -E "libpreempt|libhalhip|not found"

if ldd /data/dockercode/xsched-build/output/lib/libshimhip.so | grep -q "not found"; then
    echo
    echo "❌ 库依赖有问题"
    exit 1
fi

echo "  ✅ 库依赖正常"
echo

# 测试基本功能
echo "测试基本 PyTorch 功能..."
python3 -c 'import torch; print("  PyTorch:", torch.__version__); print("  CUDA:", torch.cuda.is_available())'

if [ $? -ne 0 ]; then
    echo "❌ PyTorch 测试失败"
    exit 1
fi

echo "  ✅ PyTorch 正常"
echo

# 运行高负载测试
echo "========================================================================"
echo "开始高负载测试 (20 req/s, batch=1024, 180s)"
echo "========================================================================"
echo

python3 test_phase4_dual_model_intensive.py \
  --duration 180 \
  --output /data/dockercode/test_results_phase4/xsched_intensive_result.json

echo
echo "========================================================================"
echo "✅ 测试完成"
echo "========================================================================"
echo
echo "结果文件: /data/dockercode/test_results_phase4/xsched_intensive_result.json"
echo
