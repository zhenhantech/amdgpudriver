#!/bin/bash
# 检查日志配置并重新运行 Test 3

set -e

echo "========================================================================"
echo "检查 XSched 日志配置并重新运行 Test 3"
echo "========================================================================"
echo

# 1. 检查是否有 Debug 日志
echo "[1/5] 检查 shim.cpp 是否有 fprintf Debug 日志..."
DEBUG_COUNT=$(docker exec zhenflashinfer_v1 bash -c "grep -c 'fprintf.*TRACE' /data/dockercode/xsched-official/platforms/hip/shim/src/shim.cpp || true")

if [ "$DEBUG_COUNT" -gt 0 ]; then
    echo "  ⚠️  发现 $DEBUG_COUNT 行 Debug 日志"
    echo "  建议移除以获得准确性能数据"
    echo
    echo "  移除方法:"
    echo "    docker exec -it zhenflashinfer_v1 bash"
    echo "    cd /data/dockercode/xsched-official/platforms/hip/shim/src"
    echo "    # 编辑 shim.cpp，注释掉 fprintf 行"
    echo "    # 然后重新编译"
    echo
else
    echo "  ✅ 没有发现 Debug 日志"
fi

echo

# 2. 检查 XSched 日志级别环境变量
echo "[2/5] 检查 XSched 日志级别环境变量..."
docker exec zhenflashinfer_v1 bash -c 'env | grep -i "XSCHED\|LOG_LEVEL" || echo "  ℹ️  未设置日志级别环境变量"'

echo

# 3. 查找可能的日志控制方式
echo "[3/5] 查找 XSched 日志控制方式..."
echo "  查找 LOG_LEVEL 相关代码..."
LOG_FILES=$(docker exec zhenflashinfer_v1 bash -c 'find /data/dockercode/xsched-official/platforms/hip -name "*.cpp" -o -name "*.h" | xargs grep -l "LOG_LEVEL\|log_level" | head -5' || echo "未找到")
if [ -n "$LOG_FILES" ]; then
    echo "$LOG_FILES"
else
    echo "  ℹ️  未找到明显的日志级别控制"
fi

echo

# 4. 提供重新运行选项
echo "[4/5] 重新运行 Test 3 的选项..."
echo
echo "选项 A: 直接重新运行（使用现有配置）"
echo "  cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED"
echo "  ./run_phase4_dual_model.sh"
echo
echo "选项 B: 在 Docker 内直接运行（baseline）"
echo "  docker exec zhenflashinfer_v1 bash -c '"
echo "    cd /data/dockercode && \\"
echo "    unset LD_PRELOAD && \\"
echo "    python3 test_phase4_dual_model.py --duration 60 --output /tmp/baseline_v2.json"
echo "  '"
echo
echo "选项 C: 在 Docker 内直接运行（XSched）"
echo "  docker exec zhenflashinfer_v1 bash -c '"
echo "    cd /data/dockercode && \\"
echo "    export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:\$LD_LIBRARY_PATH && \\"
echo "    export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so && \\"
echo "    python3 test_phase4_dual_model.py --duration 60 --output /tmp/xsched_v2.json"
echo "  '"
echo
echo "选项 D: 减少测试时间（快速验证，30 秒）"
echo "  docker exec zhenflashinfer_v1 bash -c '"
echo "    cd /data/dockercode && \\"
echo "    export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:\$LD_LIBRARY_PATH && \\"
echo "    export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so && \\"
echo "    python3 test_phase4_dual_model.py --duration 30 --output /tmp/xsched_quick.json"
echo "  '"
echo

# 5. 交互式选择
echo "[5/5] 是否立即重新运行 Test 3？"
echo
read -p "请选择 (A/B/C/D/N): " choice

case "$choice" in
    A|a)
        echo
        echo "执行选项 A: 完整重新运行..."
        cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED
        ./run_phase4_dual_model.sh
        ;;
    B|b)
        echo
        echo "执行选项 B: Baseline 测试..."
        docker exec zhenflashinfer_v1 bash -c "
            cd /data/dockercode && \
            unset LD_PRELOAD && \
            python3 test_phase4_dual_model.py --duration 60 --output /tmp/baseline_v2.json
        "
        echo
        echo "✅ 结果保存在: zhenflashinfer_v1:/tmp/baseline_v2.json"
        ;;
    C|c)
        echo
        echo "执行选项 C: XSched 测试..."
        docker exec zhenflashinfer_v1 bash -c "
            cd /data/dockercode && \
            export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:\$LD_LIBRARY_PATH && \
            export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so && \
            python3 test_phase4_dual_model.py --duration 60 --output /tmp/xsched_v2.json
        "
        echo
        echo "✅ 结果保存在: zhenflashinfer_v1:/tmp/xsched_v2.json"
        ;;
    D|d)
        echo
        echo "执行选项 D: 快速验证（30秒）..."
        docker exec zhenflashinfer_v1 bash -c "
            cd /data/dockercode && \
            export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:\$LD_LIBRARY_PATH && \
            export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so && \
            python3 test_phase4_dual_model.py --duration 30 --output /tmp/xsched_quick.json
        "
        echo
        echo "✅ 结果保存在: zhenflashinfer_v1:/tmp/xsched_quick.json"
        ;;
    N|n|*)
        echo
        echo "取消运行。"
        echo "你可以稍后手动执行上述命令。"
        ;;
esac

echo
echo "========================================================================"
echo "检查完成"
echo "========================================================================"
