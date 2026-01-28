#!/bin/bash
# 重新编译 XSched 以修复 symbol error
# 用法: ./rebuild_xsched.sh

set -e

CONTAINER="zhenflashinfer_v1"

echo "========================================================================"
echo "重新编译 XSched"
echo "========================================================================"
echo

# 检查容器
if ! docker ps | grep -q "$CONTAINER"; then
    echo "❌ Docker 容器未运行"
    exit 1
fi

echo "✅ Docker 容器正在运行"
echo

# 备份当前的库文件
echo "[1/5] 备份当前库文件..."
docker exec "$CONTAINER" bash -c "
    cd /data/dockercode/xsched-build/output/lib
    mkdir -p backup_$(date +%Y%m%d_%H%M%S)
    cp *.so backup_$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
"
echo "  ✅ 备份完成"
echo

# 清理旧的编译文件
echo "[2/5] 清理旧的编译文件..."
docker exec "$CONTAINER" bash -c "
    cd /data/dockercode/xsched-build
    make clean || true
"
echo "  ✅ 清理完成"
echo

# 重新编译
echo "[3/5] 重新编译 XSched (预计 2-3 分钟)..."
docker exec "$CONTAINER" bash -c "
    cd /data/dockercode/xsched-build
    make -j\$(nproc)
"

if [ $? -ne 0 ]; then
    echo "❌ 编译失败"
    exit 1
fi

echo "  ✅ 编译完成"
echo

# 安装
echo "[4/5] 安装库文件..."
docker exec "$CONTAINER" bash -c "
    cd /data/dockercode/xsched-build
    make install
"
echo "  ✅ 安装完成"
echo

# 验证
echo "[5/5] 验证库文件..."
docker exec "$CONTAINER" bash -c "
    export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:\$LD_LIBRARY_PATH
    
    echo '  检查库文件:'
    ls -lh /data/dockercode/xsched-build/output/lib/lib*.so | awk '{print \"    \" \$9, \$5}'
    
    echo
    echo '  检查依赖:'
    ldd /data/dockercode/xsched-build/output/lib/libshimhip.so | grep -E 'libpreempt|not found'
    
    echo
    echo '  检查符号:'
    nm /data/dockercode/xsched-build/output/lib/libshimhip.so | grep -i hipcommand | wc -l | xargs echo \"    HipCommand symbols:\"
"
echo

# 测试
echo "========================================================================"
echo "测试 PyTorch + XSched"
echo "========================================================================"
echo

docker exec "$CONTAINER" bash -c "
    export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:\$LD_LIBRARY_PATH
    export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so
    python3 -c 'import torch; print(\"  PyTorch:\", torch.__version__); print(\"  CUDA:\", torch.cuda.is_available())'
"

if [ $? -eq 0 ]; then
    echo
    echo "========================================================================"
    echo "✅ 重新编译成功！"
    echo "========================================================================"
    echo
    echo "现在可以运行测试:"
    echo "  ./test_xsched_quick.sh    # 10 秒快速测试"
    echo "  ./test_xsched_only.sh     # 完整测试"
    echo
else
    echo
    echo "❌ 测试失败，可能还有其他问题"
    echo
    echo "尝试添加 libhalhip.so:"
    docker exec "$CONTAINER" bash -c "
        export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:\$LD_LIBRARY_PATH
        export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libhalhip.so:/data/dockercode/xsched-build/output/lib/libshimhip.so
        python3 -c 'import torch; print(\"  PyTorch:\", torch.__version__); print(\"  CUDA:\", torch.cuda.is_available())'
    "
    
    if [ $? -eq 0 ]; then
        echo
        echo "✅ 使用 libhalhip.so + libshimhip.so 成功"
        echo
        echo "请更新测试脚本使用两个库:"
        echo "  LD_PRELOAD=libhalhip.so:libshimhip.so"
    else
        echo
        echo "❌ 仍然失败，需要进一步诊断"
    fi
fi
