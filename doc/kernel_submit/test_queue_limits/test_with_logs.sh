#!/bin/bash

# 带详细日志的Queue测试脚本
# 启用HIP log和KFD debug log来观察queue创建过程

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_PROG="${SCRIPT_DIR}/test_multiple_streams"
LOG_DIR="${SCRIPT_DIR}/logs_with_debug"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

mkdir -p "${LOG_DIR}"

echo "======================================================"
echo "Stream Queue Test with Full Logging"
echo "======================================================"
echo ""

# ============================================
# Part 1: 检查并编译测试程序
# ============================================
if [ ! -f "${TEST_PROG}" ]; then
    print_info "Compiling test program..."
    cd "${SCRIPT_DIR}"
    if hipcc -o test_multiple_streams test_multiple_streams.cpp; then
        print_success "Compilation successful"
    else
        print_error "Compilation failed"
        exit 1
    fi
fi

# ============================================
# Part 2: 启用 HIP Logging (不需要重新编译!)
# ============================================
print_info "Enabling HIP logging..."
echo ""
echo "HIP环境变量 (无需重新编译，直接生效):"
echo "  - AMD_LOG_LEVEL=3             # 详细日志"
echo "  - HIP_VISIBLE_DEVICES=0       # 使用GPU 0"
echo "  - HIP_DB=0x1                  # 启用调试输出"
echo ""

export AMD_LOG_LEVEL=3        # 0=Error, 1=Warning, 2=Info, 3=Verbose, 4=Debug
export HIP_VISIBLE_DEVICES=0
export HIP_DB=0x1              # Enable HIP debugging
# export HSA_ENABLE_DEBUG=1    # 可选: HSA调试

print_success "HIP logging enabled"
echo ""

# ============================================
# Part 3: 启用 KFD Debug Logging (不需要重新编译!)
# ============================================
print_info "Checking KFD debug capability..."

# 检查是否有sudo权限
if sudo -n true 2>/dev/null; then
    print_info "Enabling KFD debug logging..."
    
    # 检查 dynamic_debug 是否可用
    if [ -f /sys/kernel/debug/dynamic_debug/control ]; then
        print_success "Dynamic Debug is available"
        
        # 启用KFD调试日志（无需重新编译内核模块！）
        print_info "Enabling KFD debug messages via dynamic_debug..."
        
        # 启用关键的调试点
        sudo bash -c "echo 'file kfd_device_queue_manager.c +p' > /sys/kernel/debug/dynamic_debug/control" 2>/dev/null || true
        sudo bash -c "echo 'file kfd_chardev.c +p' > /sys/kernel/debug/dynamic_debug/control" 2>/dev/null || true
        sudo bash -c "echo 'file kfd_process_queue_manager.c +p' > /sys/kernel/debug/dynamic_debug/control" 2>/dev/null || true
        
        print_success "KFD debug logging enabled (via dynamic_debug)"
        echo "  ✓ 无需重新编译内核模块"
        echo "  ✓ 通过 /sys/kernel/debug/dynamic_debug/control 动态启用"
        echo ""
    else
        print_warning "Dynamic Debug not available (need CONFIG_DYNAMIC_DEBUG in kernel)"
        print_info "继续测试，但可能看不到KFD内核日志"
        echo ""
    fi
else
    print_warning "No sudo access - cannot enable KFD debug logging"
    print_info "仍然可以看到HIP层的日志"
    echo ""
fi

# ============================================
# Part 4: 清空 dmesg
# ============================================
print_info "Clearing dmesg buffer..."
sudo dmesg -C 2>/dev/null || print_warning "Cannot clear dmesg (need sudo)"
echo ""

# ============================================
# Part 5: 运行测试
# ============================================
NUM_STREAMS=${1:-16}

echo "======================================================"
echo "Running Test with ${NUM_STREAMS} Streams"
echo "======================================================"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_LOG="${LOG_DIR}/test_${NUM_STREAMS}_streams_${TIMESTAMP}.log"
DMESG_LOG="${LOG_DIR}/dmesg_${NUM_STREAMS}_streams_${TIMESTAMP}.log"
HIP_LOG="${LOG_DIR}/hip_${NUM_STREAMS}_streams_${TIMESTAMP}.log"

print_info "Test configuration:"
echo "  Streams: ${NUM_STREAMS}"
echo "  HIP Log Level: ${AMD_LOG_LEVEL}"
echo "  Test Log: ${TEST_LOG}"
echo "  Dmesg Log: ${DMESG_LOG}"
echo ""

# 运行测试并捕获输出
print_info "Executing test program..."
if "${TEST_PROG}" "${NUM_STREAMS}" 2>&1 | tee "${TEST_LOG}"; then
    print_success "Test completed"
else
    print_error "Test failed"
    exit 1
fi

echo ""

# 等待一下让日志写入
sleep 2

# ============================================
# Part 6: 收集日志
# ============================================
print_info "Collecting logs..."
echo ""

# 收集 dmesg
if sudo dmesg > "${DMESG_LOG}" 2>/dev/null; then
    print_success "Saved dmesg to ${DMESG_LOG}"
    
    # 分析 dmesg 内容
    echo ""
    echo "Dmesg Analysis:"
    
    queue_count=$(grep -c "CREATE_QUEUE" "${DMESG_LOG}" 2>/dev/null || echo "0")
    hqd_count=$(grep -c "hqd slot" "${DMESG_LOG}" 2>/dev/null || echo "0")
    cpsch_count=$(grep -c "map_queues_cpsch" "${DMESG_LOG}" 2>/dev/null || echo "0")
    kfd_count=$(grep -c "kfd" "${DMESG_LOG}" 2>/dev/null || echo "0")
    
    echo "  - Total KFD messages: ${kfd_count}"
    echo "  - CREATE_QUEUE events: ${queue_count}"
    echo "  - HQD allocations: ${hqd_count} (NOCPSCH mode)"
    echo "  - map_queues_cpsch calls: ${cpsch_count} (CPSCH mode)"
    echo ""
    
    if [ "${kfd_count}" -gt 0 ]; then
        print_success "KFD debug logs captured!"
        echo ""
        echo "Recent KFD logs (last 20):"
        grep "kfd" "${DMESG_LOG}" | tail -20
    else
        print_warning "No KFD debug logs found"
        echo "  可能原因:"
        echo "  1. Dynamic debug 未正确启用"
        echo "  2. 需要更高的内核日志级别"
        echo "  3. 内核未编译 CONFIG_DYNAMIC_DEBUG"
    fi
else
    print_warning "Cannot save dmesg (need sudo)"
fi

echo ""

# 分析 HIP 日志
print_info "Analyzing HIP logs..."
if grep -i "hipStreamCreate" "${TEST_LOG}" > /dev/null 2>&1; then
    print_success "HIP function calls logged"
else
    print_info "HIP详细日志可能需要更高的AMD_LOG_LEVEL"
fi

echo ""

# ============================================
# Part 7: 生成报告
# ============================================
REPORT="${LOG_DIR}/report_${NUM_STREAMS}_streams_${TIMESTAMP}.txt"

{
    echo "======================================================"
    echo "Stream Queue Test Report with Full Logging"
    echo "======================================================"
    echo "Generated: $(date)"
    echo ""
    echo "Test Configuration:"
    echo "  - Streams: ${NUM_STREAMS}"
    echo "  - AMD_LOG_LEVEL: ${AMD_LOG_LEVEL}"
    echo "  - HIP_DB: ${HIP_DB}"
    echo ""
    echo "Logging Status:"
    echo "  - HIP Logging: ✓ Enabled (via environment variables)"
    echo "  - KFD Logging: $([ ${kfd_count} -gt 0 ] && echo '✓ Enabled' || echo '⚠ Not captured')"
    echo ""
    echo "Results:"
    echo "  - Test Status: $([ -f "${TEST_LOG}" ] && echo 'SUCCESS' || echo 'FAILED')"
    echo "  - KFD Messages: ${kfd_count}"
    echo "  - Queue Creations: ${queue_count}"
    echo "  - HQD Allocations: ${hqd_count}"
    echo ""
    echo "Log Files:"
    echo "  - Test Output: ${TEST_LOG}"
    echo "  - Kernel Dmesg: ${DMESG_LOG}"
    echo "  - This Report: ${REPORT}"
    echo ""
} > "${REPORT}"

cat "${REPORT}"

print_success "Report saved to: ${REPORT}"

echo ""
echo "======================================================"
echo "✅ Test with Logging Complete"
echo "======================================================"
echo ""
echo "查看日志:"
echo "  cat ${TEST_LOG}     # 测试输出"
echo "  cat ${DMESG_LOG}    # 内核日志"
echo "  cat ${REPORT}       # 摘要报告"
echo ""
echo "关键发现:"
echo "  1. ✓ HIP日志 - 通过环境变量启用，无需重新编译"
echo "  2. $([ ${kfd_count} -gt 0 ] && echo '✓' || echo '⚠') KFD日志 - 通过dynamic_debug启用，无需重新编译"
echo ""

# 可选：禁用 debug （清理）
if [ "${2}" = "--cleanup" ]; then
    print_info "Disabling debug logs..."
    sudo bash -c "echo 'file kfd_device_queue_manager.c -p' > /sys/kernel/debug/dynamic_debug/control" 2>/dev/null || true
    print_success "Debug logs disabled"
fi
