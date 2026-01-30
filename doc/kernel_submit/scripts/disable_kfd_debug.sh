#!/bin/bash
# disable_kfd_debug.sh - 禁用 KFD 驱动的调试日志
# 使用方法: sudo bash disable_kfd_debug.sh

if [ "$EUID" -ne 0 ]; then 
    echo "❌ 错误: 此脚本需要 root 权限"
    echo "请使用: sudo bash $0"
    exit 1
fi

echo "============================================"
echo "禁用 KFD (Kernel Fusion Driver) 调试日志"
echo "============================================"
echo

# 禁用 KFD 相关的调试日志
echo "禁用 kfd_device_queue_manager.c 的调试日志..."
echo "file kfd_device_queue_manager.c -p" > /sys/kernel/debug/dynamic_debug/control

echo
echo "✅ KFD 调试日志已禁用"
echo

