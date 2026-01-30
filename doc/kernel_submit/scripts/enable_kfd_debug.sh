#!/bin/bash
# enable_kfd_debug.sh - 启用 KFD 驱动的调试日志
# 使用方法: sudo bash enable_kfd_debug.sh

if [ "$EUID" -ne 0 ]; then 
    echo "❌ 错误: 此脚本需要 root 权限"
    echo "请使用: sudo bash $0"
    exit 1
fi

echo "============================================"
echo "启用 KFD (Kernel Fusion Driver) 调试日志"
echo "============================================"
echo

# 检查 dynamic debug 是否可用
if [ ! -f /sys/kernel/debug/dynamic_debug/control ]; then
    echo "❌ 错误: Dynamic Debug 功能不可用"
    echo "请确保内核编译时启用了 CONFIG_DYNAMIC_DEBUG"
    exit 1
fi

echo "✅ Dynamic Debug 功能可用"
echo

# 启用 KFD 相关的调试日志
echo "启用以下调试日志:"
echo

# 1. HQD 分配日志 (hqd slot)
echo "1️⃣ 启用 HQD 分配日志..."
echo "file kfd_device_queue_manager.c line 992 +p" > /sys/kernel/debug/dynamic_debug/control
echo "   ✓ hqd slot - pipe X, queue Y"

# 2. Queue 创建日志
echo "2️⃣ 启用 Queue 创建日志..."
echo "file kfd_device_queue_manager.c func create_queue +p" > /sys/kernel/debug/dynamic_debug/control
echo "   ✓ create_queue()"

# 3. Queue 销毁日志
echo "3️⃣ 启用 Queue 销毁日志..."
echo "file kfd_device_queue_manager.c func destroy_queue +p" > /sys/kernel/debug/dynamic_debug/control
echo "   ✓ destroy_queue()"

# 4. 可选: 启用所有 kfd_device_queue_manager.c 的调试日志
# 注意: 这会产生大量日志
# echo "file kfd_device_queue_manager.c +p" > /sys/kernel/debug/dynamic_debug/control

echo
echo "============================================"
echo "✅ KFD 调试日志已启用"
echo "============================================"
echo
echo "📝 查看日志方法:"
echo "   sudo dmesg -w                    # 实时查看内核日志"
echo "   sudo dmesg | grep 'hqd slot'     # 查看 HQD 分配日志"
echo "   sudo dmesg | grep 'kfd'          # 查看所有 KFD 日志"
echo
echo "🔧 运行测试程序:"
echo "   cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/tests"
echo "   ./test_kernel_trace"
echo
echo "⚠️  注意: 调试日志会影响性能，测试完成后建议禁用"
echo "   sudo bash disable_kfd_debug.sh"
echo

