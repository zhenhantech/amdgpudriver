#!/bin/bash

# 对比不同优化级别的效果

KERNEL=${1:-test_promote_alloca.hip}
ARCH=${2:-gfx90a}
SCRIPT_DIR=$(dirname $(readlink -f $0))

if [ ! -f "$SCRIPT_DIR/$KERNEL" ]; then
    echo "错误: 找不到 $KERNEL"
    echo "用法: $0 [kernel.hip] [arch]"
    exit 1
fi

echo "======================================"
echo "对比优化级别"
echo "Kernel: $KERNEL"
echo "架构: $ARCH"
echo "======================================"

cd $SCRIPT_DIR

for OPT in O0 O1 O2 O3; do
    echo -e "\n=== -$OPT ==="
    
    # 编译
    hipcc -$OPT --offload-arch=$ARCH \
          -save-temps \
          -mllvm -stats \
          $KERNEL -o test_$OPT 2>&1 | \
          grep -c "promoted\|optimized\|combined" | \
          xargs -I {} echo "优化数量: {}"
    
    # 可执行文件大小
    ls -lh test_$OPT | awk '{print "可执行文件: " $5}'
    
    # VGPR使用（从汇编中提取）
    ASM_FILE=$(ls ${KERNEL%.hip}-hip-amdgcn-*.s 2>/dev/null | head -1)
    if [ -f "$ASM_FILE" ]; then
        VGPR=$(grep -m1 "NumVgprs" "$ASM_FILE" | awk '{print $2}')
        echo "VGPR使用: ${VGPR:-N/A}"
        SGPR=$(grep -m1 "NumSgprs" "$ASM_FILE" | awk '{print $2}')
        echo "SGPR使用: ${SGPR:-N/A}"
    fi
    
    # 清理中间文件
    rm -f ${KERNEL%.hip}-hip-amdgcn-*
done

echo -e "\n======================================"
echo "测试完成"
echo "======================================"

