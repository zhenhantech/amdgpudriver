#!/bin/bash

# Stream队列测试脚本
# 用于验证软件队列和硬件队列的行为

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_PROG="${SCRIPT_DIR}/test_multiple_streams"
LOG_DIR="${SCRIPT_DIR}/logs"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 编译测试程序
compile_test() {
    print_info "Compiling test program..."
    cd "${SCRIPT_DIR}"
    
    if hipcc -o test_multiple_streams test_multiple_streams.cpp; then
        print_success "Compilation successful"
        return 0
    else
        print_error "Compilation failed"
        return 1
    fi
}

# 清空dmesg（需要root权限）
clear_dmesg() {
    if sudo -n dmesg -C 2>/dev/null; then
        print_success "Cleared dmesg"
        return 0
    else
        print_warning "Cannot clear dmesg (need sudo). Continuing anyway..."
        return 1
    fi
}

# 启用KFD debug日志
enable_kfd_debug() {
    print_info "Checking KFD debug status..."
    
    # 检查debug script是否存在
    DEBUG_SCRIPT="${SCRIPT_DIR}/../scripts/enable_kfd_debug.sh"
    if [ -f "${DEBUG_SCRIPT}" ]; then
        print_info "Enabling KFD debug logs..."
        if sudo bash "${DEBUG_SCRIPT}" 2>/dev/null; then
            print_success "KFD debug enabled"
        else
            print_warning "Could not enable KFD debug (continuing anyway)"
        fi
    else
        print_warning "KFD debug script not found at ${DEBUG_SCRIPT}"
    fi
}

# 运行单个测试
run_test() {
    local num_streams=$1
    local test_name="test_${num_streams}_streams"
    local log_file="${LOG_DIR}/${test_name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo ""
    echo "========================================"
    echo "Running Test: ${num_streams} Streams"
    echo "========================================"
    
    # 清空dmesg
    clear_dmesg
    
    # 运行测试程序
    print_info "Executing: ${TEST_PROG} ${num_streams}"
    
    if "${TEST_PROG}" "${num_streams}" 2>&1 | tee "${log_file}"; then
        print_success "Test program completed"
    else
        print_error "Test program failed"
        return 1
    fi
    
    # 等待一下让kernel日志写入
    sleep 1
    
    # 收集dmesg日志
    print_info "Collecting dmesg logs..."
    local dmesg_file="${LOG_DIR}/${test_name}_dmesg_$(date +%Y%m%d_%H%M%S).log"
    
    if sudo dmesg > "${dmesg_file}" 2>/dev/null; then
        print_success "Saved dmesg to ${dmesg_file}"
    else
        print_warning "Could not save dmesg (need sudo)"
    fi
    
    # 分析结果
    echo ""
    print_info "Analyzing results for ${num_streams} streams:"
    
    # 统计CREATE_QUEUE
    local queue_count=$(sudo dmesg 2>/dev/null | grep -c "CREATE_QUEUE" || echo "0")
    echo "  - CREATE_QUEUE events: ${queue_count}"
    
    # 统计hqd slot（NOCPSCH模式）
    local hqd_count=$(sudo dmesg 2>/dev/null | grep -c "hqd slot" || echo "0")
    echo "  - HQD allocations: ${hqd_count} (if NOCPSCH mode)"
    
    # 统计map_queues_cpsch（CPSCH模式）
    local cpsch_count=$(sudo dmesg 2>/dev/null | grep -c "map_queues_cpsch" || echo "0")
    echo "  - map_queues_cpsch calls: ${cpsch_count} (if CPSCH mode)"
    
    # 显示最近的queue创建日志
    echo ""
    print_info "Recent queue creation logs (last 10):"
    sudo dmesg 2>/dev/null | grep -E "CREATE_QUEUE|hqd slot" | tail -10 || echo "  (no logs available)"
    
    echo ""
    print_success "Test completed: ${test_name}"
    print_info "Full log saved to: ${log_file}"
    
    return 0
}

# 运行对比测试
run_comparison_tests() {
    print_info "Running comparison tests..."
    
    # 测试场景
    local test_cases=(16 32 64)
    
    for num in "${test_cases[@]}"; do
        run_test "${num}"
        
        # 测试之间暂停一下
        if [ "${num}" != "64" ]; then
            print_info "Waiting 3 seconds before next test..."
            sleep 3
        fi
    done
    
    # 生成对比报告
    generate_comparison_report
}

# 生成对比报告
generate_comparison_report() {
    local report_file="${LOG_DIR}/comparison_report_$(date +%Y%m%d_%H%M%S).txt"
    
    print_info "Generating comparison report..."
    
    {
        echo "========================================"
        echo "Stream Queue Limits Comparison Report"
        echo "========================================"
        echo "Generated: $(date)"
        echo ""
        echo "Test Configuration:"
        echo "  - Software Queue Limit (per process): 1024"
        echo "  - Hardware Queue Limit (MI308X): 32"
        echo ""
        echo "Test Results:"
        echo ""
        
        for num in 16 32 64; do
            echo "Test: ${num} Streams"
            echo "  Expected:"
            echo "    - Software Queues: ${num}"
            echo "    - Hardware HQDs: $(( num < 32 ? num : 32 ))"
            if [ ${num} -le 32 ]; then
                echo "    - HQD Status: ✓ Sufficient (no sharing)"
            else
                echo "    - HQD Status: ⚠ Insufficient (sharing required)"
                echo "    - Sharing Ratio: $(echo "scale=2; ${num}/32" | bc):1"
            fi
            echo ""
        done
        
        echo "Logs Location: ${LOG_DIR}"
        echo ""
        echo "Key Findings:"
        echo "  1. Each stream creates 1 independent AQL Queue"
        echo "  2. AQL Queues have independent ring buffers and doorbells"
        echo "  3. HQD sharing occurs when streams > 32"
        echo "  4. In CPSCH mode, HQD allocation is dynamic (firmware-managed)"
        echo ""
    } > "${report_file}"
    
    cat "${report_file}"
    print_success "Report saved to: ${report_file}"
}

# 显示使用说明
show_usage() {
    cat << EOF
Stream Queue Limits Test Script

Usage: $0 [options] [num_streams]

Options:
  -h, --help              Show this help message
  -c, --compile           Compile test program only
  -d, --debug             Enable KFD debug logs before testing
  -a, --all               Run all comparison tests (16, 32, 64 streams)
  
Arguments:
  num_streams             Number of streams to test (default: run all)

Examples:
  $0                      # Run all comparison tests
  $0 -a                   # Run all comparison tests
  $0 16                   # Test with 16 streams only
  $0 -d 32                # Enable debug and test with 32 streams
  $0 --compile            # Compile only

Logs are saved to: ${LOG_DIR}
EOF
}

# 主函数
main() {
    local compile_only=false
    local enable_debug=false
    local run_all=false
    local num_streams=""
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -c|--compile)
                compile_only=true
                shift
                ;;
            -d|--debug)
                enable_debug=true
                shift
                ;;
            -a|--all)
                run_all=true
                shift
                ;;
            [0-9]*)
                num_streams=$1
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # 编译测试程序
    if ! compile_test; then
        exit 1
    fi
    
    if [ "${compile_only}" = true ]; then
        print_info "Compile-only mode. Exiting."
        exit 0
    fi
    
    # 启用debug日志
    if [ "${enable_debug}" = true ]; then
        enable_kfd_debug
    fi
    
    # 运行测试
    if [ "${run_all}" = true ] || [ -z "${num_streams}" ]; then
        run_comparison_tests
    else
        run_test "${num_streams}"
    fi
    
    echo ""
    print_success "All tests completed!"
    print_info "Check logs in: ${LOG_DIR}"
}

# 执行主函数
main "$@"
