#!/bin/bash

# 读取MQD和HQD信息来确定实际硬件队列使用情况
# 这是查看真实硬件状态的正确方法！

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "=========================================="
echo "MQD/HQD Hardware Queue Inspector"
echo "=========================================="
echo ""

# 检查是否有root权限
if ! sudo -n true 2>/dev/null; then
    print_error "This script requires sudo access"
    echo "Please run with sudo or configure passwordless sudo"
    exit 1
fi

# 检查debugfs是否挂载
if [ ! -d /sys/kernel/debug/kfd ]; then
    print_error "/sys/kernel/debug/kfd not found"
    echo ""
    echo "Possible reasons:"
    echo "  1. debugfs not mounted"
    echo "  2. KFD module not loaded"
    echo "  3. Kernel CONFIG_DEBUG_FS not enabled"
    echo ""
    echo "Try:"
    echo "  sudo mount -t debugfs none /sys/kernel/debug"
    exit 1
fi

print_success "KFD debugfs is available"
echo ""

# ============================================
# Part 1: 读取 HQDs (Hardware Queue Descriptors)
# ============================================
print_info "Reading Hardware Queue Descriptors (HQDs)..."
echo ""

if [ -r /sys/kernel/debug/kfd/hqds ]; then
    HQD_FILE="/tmp/kfd_hqds_$(date +%Y%m%d_%H%M%S).txt"
    sudo cat /sys/kernel/debug/kfd/hqds > "${HQD_FILE}"
    
    print_success "HQDs data saved to: ${HQD_FILE}"
    echo ""
    
    # 分析HQD内容
    echo "HQD Analysis:"
    echo "----------------------------------------"
    
    # 统计活跃的HQD数量
    active_hqds=$(grep -c "Active:" "${HQD_FILE}" 2>/dev/null || echo "0")
    echo "  Total HQD entries: ${active_hqds}"
    
    # 显示前几个HQD
    if [ ${active_hqds} -gt 0 ]; then
        echo ""
        echo "First 5 HQDs (preview):"
        head -100 "${HQD_FILE}"
        
        if [ $(wc -l < "${HQD_FILE}") -gt 100 ]; then
            echo "... (see full content in ${HQD_FILE})"
        fi
    else
        print_warning "No active HQDs found"
    fi
    echo ""
else
    print_warning "Cannot read /sys/kernel/debug/kfd/hqds"
fi

# ============================================
# Part 2: 读取 MQDs (Memory Queue Descriptors)
# ============================================
print_info "Reading Memory Queue Descriptors (MQDs)..."
echo ""

if [ -r /sys/kernel/debug/kfd/mqds ]; then
    MQD_FILE="/tmp/kfd_mqds_$(date +%Y%m%d_%H%M%S).txt"
    sudo cat /sys/kernel/debug/kfd/mqds > "${MQD_FILE}"
    
    print_success "MQDs data saved to: ${MQD_FILE}"
    echo ""
    
    # 分析MQD内容
    echo "MQD Analysis:"
    echo "----------------------------------------"
    
    # 查找Queue ID信息
    queue_count=$(grep -c "Queue" "${MQD_FILE}" 2>/dev/null || echo "0")
    echo "  Total Queue entries: ${queue_count}"
    
    # 查找pipe和queue字段
    pipe_info=$(grep -i "pipe" "${MQD_FILE}" 2>/dev/null | head -10 || echo "")
    if [ -n "${pipe_info}" ]; then
        echo ""
        echo "Pipe/Queue information found:"
        echo "${pipe_info}"
    fi
    
    # 查找优先级信息
    priority_info=$(grep -i "priority" "${MQD_FILE}" 2>/dev/null | head -10 || echo "")
    if [ -n "${priority_info}" ]; then
        echo ""
        echo "Priority information found:"
        echo "${priority_info}"
    fi
    
    # 查找ring buffer地址
    ringbuf_count=$(grep -c "cp_hqd_pq_base" "${MQD_FILE}" 2>/dev/null || echo "0")
    if [ ${ringbuf_count} -gt 0 ]; then
        echo ""
        echo "Ring buffer configurations: ${ringbuf_count}"
    fi
    
    echo ""
    echo "Preview (first 200 lines):"
    head -200 "${MQD_FILE}"
    
    if [ $(wc -l < "${MQD_FILE}") -gt 200 ]; then
        echo "... (see full content in ${MQD_FILE})"
    fi
    echo ""
else
    print_warning "Cannot read /sys/kernel/debug/kfd/mqds"
fi

# ============================================
# Part 3: 读取 RLS (RunList Status)
# ============================================
print_info "Reading RunList Status (RLS)..."
echo ""

if [ -r /sys/kernel/debug/kfd/rls ]; then
    RLS_FILE="/tmp/kfd_rls_$(date +%Y%m%d_%H%M%S).txt"
    sudo cat /sys/kernel/debug/kfd/rls > "${RLS_FILE}"
    
    print_success "RLS data saved to: ${RLS_FILE}"
    echo ""
    cat "${RLS_FILE}"
    echo ""
else
    print_warning "Cannot read /sys/kernel/debug/kfd/rls"
fi

# ============================================
# Part 4: 总结
# ============================================
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""
echo "Key Findings:"
echo "  ✓ 通过 /sys/kernel/debug/kfd/mqds 可以看到所有软件队列的MQD"
echo "  ✓ 通过 /sys/kernel/debug/kfd/hqds 可以看到实际硬件队列使用情况"
echo "  ✓ 这是查看真实硬件状态的正确方法（不是理论计算）"
echo ""
echo "Saved Files:"
if [ -f "${HQD_FILE}" ]; then
    echo "  - HQDs: ${HQD_FILE}"
fi
if [ -f "${MQD_FILE}" ]; then
    echo "  - MQDs: ${MQD_FILE}"
fi
if [ -f "${RLS_FILE}" ]; then
    echo "  - RLS:  ${RLS_FILE}"
fi
echo ""

echo "Understanding the output:"
echo "  - MQDs: 显示每个软件队列的配置（包括ring buffer地址、优先级等）"
echo "  - HQDs: 显示实际加载到GPU硬件的队列（这是真实的硬件状态！）"
echo "  - RLS:  显示CPSCH模式下的RunList状态"
echo ""

echo "Key MQD fields to check:"
echo "  - cp_hqd_pq_base: Ring buffer地址（每个队列不同）"
echo "  - cp_hqd_pipe_priority: 硬件优先级（0=LOW, 1=MEDIUM, 2=HIGH）"
echo "  - cp_hqd_pq_doorbell_control: Doorbell配置"
echo ""

echo "=========================================="
print_success "Inspection Complete"
echo "=========================================="
