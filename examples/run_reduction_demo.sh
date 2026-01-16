#!/bin/bash

# HIPIFY 归约优化示例 - 运行脚本
# 用法: ./run_reduction_demo.sh [数组大小]

set -e  # 遇到错误立即退出

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 HIPIFY 归约优化示例 - 编译和运行"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 检查HIP环境
if ! command -v hipcc &> /dev/null; then
    echo "❌ 错误: hipcc 未找到"
    echo "   请确保已安装ROCm并设置环境变量"
    echo "   安装指南: https://rocm.docs.amd.com/"
    exit 1
fi

echo "✅ HIP环境检查通过"
hipcc --version | head -n 1
echo ""

# 获取数组大小参数
ARRAY_SIZE=${1:-52428800}  # 默认50M元素

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 编译选项"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 方法1: 基础编译（快速测试）
echo "🔹 方法1: 基础编译"
hipcc -O3 hipify_reduction_demo.hip -o reduction_demo_basic
echo "   ✓ 生成: reduction_demo_basic"
echo ""

# 方法2: 显示资源使用（学习寄存器优化）
echo "🔹 方法2: 显示资源使用（推荐学习）"
hipcc -O3 --resource-usage hipify_reduction_demo.hip -o reduction_demo_verbose 2>&1 | tee compile_info.txt
echo "   ✓ 生成: reduction_demo_verbose"
echo "   ✓ 编译信息保存到: compile_info.txt"
echo ""

# 方法3: 调试版本
echo "🔹 方法3: 调试版本"
hipcc -g -O0 hipify_reduction_demo.hip -o reduction_demo_debug
echo "   ✓ 生成: reduction_demo_debug"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏃 运行测试"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 基础运行
echo "▶️  基础运行 (数组大小: $ARRAY_SIZE)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./reduction_demo_basic $ARRAY_SIZE
echo ""

# 使用rocprof性能分析（如果可用）
if command -v rocprof &> /dev/null; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 rocprof 性能分析"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    rocprof --stats ./reduction_demo_basic $ARRAY_SIZE 2>&1 | tee rocprof_stats.txt
    echo ""
    echo "✓ 性能分析结果保存到: rocprof_stats.txt"
    echo "✓ 详细报告保存到: results.csv, results.json"
else
    echo "⚠️  rocprof 未找到，跳过性能分析"
    echo "   安装: sudo apt install rocprofiler-dev"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📚 学习建议:"
echo "   1. 查看 compile_info.txt 了解寄存器使用情况"
echo "   2. 对比三个版本的性能数据"
echo "   3. 修改代码尝试不同的展开倍数 (2x, 8x, 32x)"
echo "   4. 使用 rocprof 分析内存带宽利用率"
echo ""
echo "🔧 进阶实验:"
echo "   - 修改 blockSize (128, 256, 512)"
echo "   - 修改 gridSize (256, 512, 1024)"
echo "   - 测试不同数组大小的性能特征"
echo ""

