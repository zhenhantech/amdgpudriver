#!/bin/bash
# 诊断 XSched symbol error
# 用法: ./diagnose_symbol_error.sh

CONTAINER="zhenflashinfer_v1"

echo "========================================================================"
echo "XSched Symbol Error 诊断"
echo "========================================================================"
echo

echo "[1] 检查缺失的符号..."
SYMBOL="_ZTIN6xsched3hip10HipCommandE"
echo "  Symbol: $SYMBOL"
echo "  解码: typeinfo for xsched::hip::HipCommand"
echo

echo "[2] 在 libshimhip.so 中查找..."
docker exec "$CONTAINER" bash -c "
    nm /data/dockercode/xsched-build/output/lib/libshimhip.so 2>/dev/null | grep -q '$SYMBOL' && echo '  ✅ 找到' || echo '  ❌ 未找到'
"

echo "[3] 在 libhalhip.so 中查找..."
docker exec "$CONTAINER" bash -c "
    nm /data/dockercode/xsched-build/output/lib/libhalhip.so 2>/dev/null | grep -q '$SYMBOL' && echo '  ✅ 找到' || echo '  ❌ 未找到'
"

echo "[4] 在 libpreempt.so 中查找..."
docker exec "$CONTAINER" bash -c "
    nm /data/dockercode/xsched-build/output/lib/libpreempt.so 2>/dev/null | grep -q '$SYMBOL' && echo '  ✅ 找到' || echo '  ❌ 未找到'
"

echo
echo "[5] 检查所有库中的 HipCommand 相关符号..."
for lib in libshimhip.so libhalhip.so libpreempt.so; do
    echo "  $lib:"
    docker exec "$CONTAINER" bash -c "
        nm /data/dockercode/xsched-build/output/lib/$lib 2>/dev/null | grep -i hipcommand | wc -l
    " | xargs echo "    HipCommand symbols:"
done

echo
echo "========================================================================"
echo "诊断结果"
echo "========================================================================"
echo

# 检查是否在任何库中找到了符号
FOUND=0
for lib in libshimhip.so libhalhip.so libpreempt.so; do
    if docker exec "$CONTAINER" bash -c "nm /data/dockercode/xsched-build/output/lib/$lib 2>/dev/null | grep -q '$SYMBOL'"; then
        FOUND=1
        echo "✅ 符号在 $lib 中找到"
        
        # 检查是否是导出符号
        TYPE=$(docker exec "$CONTAINER" bash -c "nm /data/dockercode/xsched-build/output/lib/$lib 2>/dev/null | grep '$SYMBOL' | awk '{print \$2}'")
        echo "   符号类型: $TYPE"
        
        if [ "$TYPE" = "T" ] || [ "$TYPE" = "D" ] || [ "$TYPE" = "B" ]; then
            echo "   ✅ 是导出符号"
        else
            echo "   ⚠️  不是导出符号 (type: $TYPE)"
            echo "   可能需要通过 LD_PRELOAD 显式加载"
        fi
        
        echo
        echo "解决方案:"
        if [ "$lib" = "libshimhip.so" ]; then
            echo "  export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so"
        elif [ "$lib" = "libhalhip.so" ]; then
            echo "  export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libhalhip.so:/data/dockercode/xsched-build/output/lib/libshimhip.so"
        else
            echo "  export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/$lib:/data/dockercode/xsched-build/output/lib/libshimhip.so"
        fi
        break
    fi
done

if [ $FOUND -eq 0 ]; then
    echo "❌ 符号在任何库中都未找到"
    echo
    echo "可能的原因:"
    echo "  1. XSched 编译不完整"
    echo "  2. 库文件损坏"
    echo "  3. 版本不匹配"
    echo
    echo "建议操作:"
    echo "  ./rebuild_xsched.sh    # 重新编译 XSched"
fi

echo
echo "========================================================================"
echo "库文件信息"
echo "========================================================================"
docker exec "$CONTAINER" bash -c "
    ls -lh /data/dockercode/xsched-build/output/lib/*.so
"

echo
echo "========================================================================"
echo "编译时间"
echo "========================================================================"
docker exec "$CONTAINER" bash -c "
    stat /data/dockercode/xsched-build/output/lib/libshimhip.so | grep 'Modify'
"
