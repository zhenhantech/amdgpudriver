#!/bin/bash

# 完整测试流程：创建streams → 检查MQD/HQD → 获取真实硬件状态
# 这个脚本回答："真正有多少个硬件Queue？"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_PROG="${SCRIPT_DIR}/test_multiple_streams"
LOG_DIR="${SCRIPT_DIR}/logs_mqd_inspection"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

NUM_STREAMS=${1:-16}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "Stream Queue Test with MQD/HQD Real-Time Inspection"
echo "============================================================"
echo ""
echo "Test: ${NUM_STREAMS} Streams"
echo "Goal: 查看真实的硬件Queue使用情况（不是理论计算）"
echo ""

# 检查编译
if [ ! -f "${TEST_PROG}" ]; then
    print_info "Compiling test program..."
    cd "${SCRIPT_DIR}"
    hipcc -o test_multiple_streams test_multiple_streams.cpp || exit 1
fi

# 检查debugfs
if [ ! -d /sys/kernel/debug/kfd ]; then
    print_error "/sys/kernel/debug/kfd not available"
    exit 1
fi

# ============================================
# Step 1: Baseline - 读取测试前的状态
# ============================================
print_info "Step 1: Reading baseline state (before creating streams)..."
echo ""

BASELINE_MQD="${LOG_DIR}/baseline_mqds_${TIMESTAMP}.txt"
BASELINE_HQD="${LOG_DIR}/baseline_hqds_${TIMESTAMP}.txt"

sudo cat /sys/kernel/debug/kfd/mqds > "${BASELINE_MQD}" 2>/dev/null || print_warning "Cannot read mqds"
sudo cat /sys/kernel/debug/kfd/hqds > "${BASELINE_HQD}" 2>/dev/null || print_warning "Cannot read hqds"

baseline_queues=$(grep -c "Queue" "${BASELINE_MQD}" 2>/dev/null || echo "0")
print_info "Baseline: ${baseline_queues} existing queues"
echo ""

# ============================================
# Step 2: 启动测试程序（后台运行，保持streams存活）
# ============================================
print_info "Step 2: Creating ${NUM_STREAMS} streams..."
echo ""

TEST_LOG="${LOG_DIR}/test_${NUM_STREAMS}_streams_${TIMESTAMP}.log"

# 启动测试程序，带等待参数
"${TEST_PROG}" "${NUM_STREAMS}" -t 15 > "${TEST_LOG}" 2>&1 &
TEST_PID=$!

print_success "Test program started (PID: ${TEST_PID})"
print_info "Waiting 3 seconds for streams to be created..."
sleep 3

# ============================================
# Step 3: 读取测试中的MQD/HQD状态（关键！）
# ============================================
print_info "Step 3: Reading MQD/HQD state (while streams are active)..."
echo ""

ACTIVE_MQD="${LOG_DIR}/active_mqds_${NUM_STREAMS}_${TIMESTAMP}.txt"
ACTIVE_HQD="${LOG_DIR}/active_hqds_${NUM_STREAMS}_${TIMESTAMP}.txt"

sudo cat /sys/kernel/debug/kfd/mqds > "${ACTIVE_MQD}"
sudo cat /sys/kernel/debug/kfd/hqds > "${ACTIVE_HQD}"

print_success "Captured active state"
echo ""

# ============================================
# Step 4: 分析MQD/HQD内容
# ============================================
print_info "Step 4: Analyzing captured data..."
echo ""

echo "=========================================="
echo "Analysis Results"
echo "=========================================="
echo ""

# 分析MQD
print_info "MQD Analysis (Memory Queue Descriptors):"
active_queues=$(grep -c "Queue" "${ACTIVE_MQD}" 2>/dev/null || echo "0")
new_queues=$((active_queues - baseline_queues))

echo "  - Total queues: ${active_queues}"
echo "  - Baseline queues: ${baseline_queues}"
echo "  - New queues: ${new_queues}"
echo "  - Expected new queues: ${NUM_STREAMS}"

if [ ${new_queues} -eq ${NUM_STREAMS} ]; then
    print_success "✓ Queue count matches! (${new_queues} = ${NUM_STREAMS})"
else
    print_warning "⚠ Queue count mismatch (${new_queues} != ${NUM_STREAMS})"
fi
echo ""

# 查找pipe/queue信息
print_info "Searching for Pipe/Queue assignments in MQD..."
if grep -i "pipe\|queue" "${ACTIVE_MQD}" | head -20; then
    echo ""
else
    print_info "No explicit pipe/queue info in MQD (normal for CPSCH mode)"
    echo ""
fi

# 统计不同的ring buffer地址（每个队列应该有不同的地址）
print_info "Checking Ring Buffer addresses (cp_hqd_pq_base)..."
ringbuf_count=$(grep -c "cp_hqd_pq_base" "${ACTIVE_MQD}" 2>/dev/null || echo "0")
echo "  - Ring buffer entries: ${ringbuf_count}"

# 统计不同的doorbell配置
doorbell_count=$(grep -c "doorbell" "${ACTIVE_MQD}" 2>/dev/null || echo "0")
echo "  - Doorbell entries: ${doorbell_count}"

# 检查优先级配置
priority_count=$(grep -c "priority" "${ACTIVE_MQD}" 2>/dev/null || echo "0")
echo "  - Priority entries: ${priority_count}"
echo ""

# 分析HQD
print_info "HQD Analysis (Hardware Queue Descriptors):"
if [ -s "${ACTIVE_HQD}" ]; then
    hqd_entries=$(grep -c "pipe\|queue\|HQD" "${ACTIVE_HQD}" 2>/dev/null || echo "0")
    echo "  - HQD entries: ${hqd_entries}"
    
    echo ""
    echo "HQD Content (first 50 lines):"
    head -50 "${ACTIVE_HQD}"
    
    if [ $(wc -l < "${ACTIVE_HQD}") -gt 50 ]; then
        echo "... (see full content in ${ACTIVE_HQD})"
    fi
else
    print_warning "HQD file is empty or not readable"
fi
echo ""

# ============================================
# Step 5: 生成详细报告
# ============================================
REPORT="${LOG_DIR}/mqd_hqd_report_${NUM_STREAMS}_${TIMESTAMP}.txt"

{
    echo "=========================================="
    echo "MQD/HQD Inspection Report"
    echo "=========================================="
    echo "Generated: $(date)"
    echo "Test Process PID: ${TEST_PID}"
    echo ""
    echo "Test Configuration:"
    echo "  - Streams Created: ${NUM_STREAMS}"
    echo ""
    echo "Results:"
    echo "  - Baseline Queues: ${baseline_queues}"
    echo "  - Active Queues: ${active_queues}"
    echo "  - New Queues: ${new_queues}"
    echo "  - Match: $([ ${new_queues} -eq ${NUM_STREAMS} ] && echo 'YES' || echo 'NO')"
    echo ""
    echo "Key Findings:"
    echo "  1. MQDs显示所有软件队列的配置"
    echo "  2. 每个MQD包含独立的ring buffer地址和doorbell配置"
    echo "  3. HQDs显示实际硬件队列的使用情况"
    echo "  4. 在CPSCH模式下，HQD分配是动态的"
    echo ""
    echo "Data Files:"
    echo "  - Baseline MQDs: ${BASELINE_MQD}"
    echo "  - Active MQDs: ${ACTIVE_MQD}"
    echo "  - Baseline HQDs: ${BASELINE_HQD}"
    echo "  - Active HQDs: ${ACTIVE_HQD}"
    echo "  - Test Log: ${TEST_LOG}"
    echo ""
} > "${REPORT}"

cat "${REPORT}"
print_success "Report saved to: ${REPORT}"

# ============================================
# Step 6: 等待测试程序完成
# ============================================
echo ""
print_info "Waiting for test program to complete..."
wait ${TEST_PID}

print_success "Test program completed"
echo ""

# ============================================
# Step 7: 读取测试后的状态
# ============================================
print_info "Step 7: Reading post-test state..."
echo ""

AFTER_MQD="${LOG_DIR}/after_mqds_${TIMESTAMP}.txt"
AFTER_HQD="${LOG_DIR}/after_hqds_${TIMESTAMP}.txt"

sudo cat /sys/kernel/debug/kfd/mqds > "${AFTER_MQD}" 2>/dev/null || print_warning "Cannot read mqds"
sudo cat /sys/kernel/debug/kfd/hqds > "${AFTER_HQD}" 2>/dev/null || print_warning "Cannot read hqds"

after_queues=$(grep -c "Queue" "${AFTER_MQD}" 2>/dev/null || echo "0")
print_info "After test: ${after_queues} queues (should return to baseline)"
echo ""

# ============================================
# Final Summary
# ============================================
echo "=========================================="
echo "✅ MQD/HQD Inspection Complete"
echo "=========================================="
echo ""
echo "关键发现："
echo ""
echo "1. ✓ MQD（Memory Queue Descriptor）"
echo "   - 存储在内存中，每个软件队列一个MQD"
echo "   - 包含完整的硬件配置（ring buffer、doorbell、优先级等）"
echo "   - 可以通过 /sys/kernel/debug/kfd/mqds 查看"
echo "   - 数量 = 软件队列数量"
echo ""
echo "2. ✓ HQD（Hardware Queue Descriptor）"
echo "   - GPU硬件中的队列槽位"
echo "   - 数量固定 = 32个（4 Pipes × 8 Queues）"
echo "   - 可以通过 /sys/kernel/debug/kfd/hqds 查看"
echo "   - 显示实际加载到硬件的队列"
echo ""
echo "3. ⭐ 真实硬件状态"
echo "   - 创建 ${NUM_STREAMS} 个streams"
echo "   - 软件队列: ${new_queues} 个（从MQD读取）"
echo "   - 硬件队列: 需要查看HQD文件确认实际使用数量"
echo ""
echo "查看详细数据："
echo "  cat ${ACTIVE_MQD}    # 所有队列的MQD配置"
echo "  cat ${ACTIVE_HQD}    # 实际硬件队列状态"
echo "  cat ${REPORT}        # 分析报告"
echo ""
