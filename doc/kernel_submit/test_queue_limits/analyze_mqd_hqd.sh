#!/bin/bash

# 分析MQD和HQD测试结果
# 提取真实的软件队列和硬件队列数量

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

LOG_DIR="${1:-logs_mqd_inspection}"

if [ ! -d "${LOG_DIR}" ]; then
    echo "Usage: $0 [log_directory]"
    echo "Default: logs_mqd_inspection"
    exit 1
fi

echo "============================================================"
echo "MQD/HQD Analysis Report"
echo "============================================================"
echo ""

# 分析每个测试
for STREAMS in 16 32 64; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_info "Test: ${STREAMS} Streams"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # 找到对应的文件
    MQD_FILE=$(ls -1 "${LOG_DIR}"/active_mqds_${STREAMS}_*.txt 2>/dev/null | head -1)
    HQD_FILE=$(ls -1 "${LOG_DIR}"/active_hqds_${STREAMS}_*.txt 2>/dev/null | head -1)
    
    if [ -z "${MQD_FILE}" ] || [ -z "${HQD_FILE}" ]; then
        print_warning "Files not found for ${STREAMS} streams test"
        echo ""
        continue
    fi
    
    echo "Files:"
    echo "  MQD: ${MQD_FILE}"
    echo "  HQD: ${HQD_FILE}"
    echo ""
    
    # ========================================
    # Part 1: 分析MQD（软件队列）
    # ========================================
    print_info "Software Queue Analysis (MQD):"
    echo ""
    
    # 统计"Compute queue on device"
    mqd_compute_queues=$(grep -c "Compute queue on device" "${MQD_FILE}" 2>/dev/null || echo "0")
    echo "  - Compute queue entries: ${mqd_compute_queues}"
    
    # 统计唯一的ring buffer地址（关键指标！）
    # 地址在偏移0x200的位置（MQD中的cp_hqd_pq_base_lo）
    unique_ringbufs=$(grep "00000200:" "${MQD_FILE}" | awk '{print $2}' | grep -v "^00000000$" | sort | uniq | wc -l)
    echo "  - Unique ring buffer addresses: ${unique_ringbufs}"
    
    # 显示前5个ring buffer地址
    echo ""
    echo "  First 5 Ring Buffer Addresses:"
    grep "00000200:" "${MQD_FILE}" | awk '{print "    " $2}' | grep -v "^    00000000$" | head -5
    
    echo ""
    print_success "✓ Software Queues (MQD): ${unique_ringbufs}"
    echo ""
    
    # ========================================
    # Part 2: 分析HQD（硬件队列）
    # ========================================
    print_info "Hardware Queue Analysis (HQD):"
    echo ""
    
    # 统计活跃的HQD（ring buffer地址非0）
    active_hqds=$(awk '/CP Pipe/ {pipe=$4; queue=$6; getline; if ($2 != "00000000" || $3 != "00000000") {count++}} END {print count}' "${HQD_FILE}")
    
    if [ -z "${active_hqds}" ]; then
        active_hqds=0
    fi
    
    echo "  - Active HQDs (non-zero address): ${active_hqds}"
    
    # 统计使用的Pipe
    pipes_used=$(awk '/CP Pipe/ {if ($4 != "") pipes[$4]++} END {for (p in pipes) count++; print count}' "${HQD_FILE}")
    echo "  - Pipes used: ${pipes_used:-0}"
    
    # 统计每个Pipe的使用情况
    echo ""
    echo "  HQDs per Pipe:"
    awk '/CP Pipe/ {pipe=$4; queue=$6; getline; if ($2 != "00000000" || $3 != "00000000") {pipes[pipe]++}} END {for (p in pipes) printf "    Pipe %s: %d queues\n", p, pipes[p]}' "${HQD_FILE}" | sort
    
    # 显示前5个活跃的HQD
    echo ""
    echo "  First 5 Active HQDs:"
    awk '/CP Pipe/ {pipe=$4; queue=$6; getline; addr=$2; if (addr != "00000000") {print "    Pipe " pipe ", Queue " queue ": " addr; count++; if (count >= 5) exit}}' "${HQD_FILE}"
    
    echo ""
    print_success "✓ Hardware Queues (HQD): ${active_hqds}"
    echo ""
    
    # ========================================
    # Part 3: 对比和验证
    # ========================================
    print_info "Verification:"
    echo ""
    
    echo "  Expected (from test):"
    echo "    - Software Queues: ${STREAMS}"
    echo "    - Hardware Queues: ≤32 (hardware limit: 4 Pipes × 8 Queues)"
    echo ""
    
    echo "  Actual (from debugfs):"
    echo "    - Software Queues (MQD): ${unique_ringbufs}"
    echo "    - Hardware Queues (HQD): ${active_hqds}"
    echo ""
    
    # 验证结果
    if [ ${unique_ringbufs} -eq ${STREAMS} ]; then
        print_success "✓ Software queue count MATCHES expected (${unique_ringbufs} = ${STREAMS})"
    else
        print_warning "⚠ Software queue count MISMATCH (${unique_ringbufs} != ${STREAMS})"
    fi
    
    if [ ${active_hqds} -le 32 ]; then
        print_success "✓ Hardware queue count within limit (${active_hqds} ≤ 32)"
    else
        print_warning "⚠ Hardware queue count EXCEEDS limit (${active_hqds} > 32)"
    fi
    
    echo ""
    
    # 分析HQD复用情况
    if [ ${unique_ringbufs} -gt ${active_hqds} ]; then
        ratio=$(echo "scale=2; ${unique_ringbufs} / ${active_hqds}" | bc)
        print_info "HQD Sharing: ${unique_ringbufs} software queues → ${active_hqds} hardware queues (ratio: ${ratio}:1)"
    elif [ ${unique_ringbufs} -eq ${active_hqds} ]; then
        print_info "HQD Mapping: 1:1 (each software queue has dedicated hardware queue)"
    else
        print_warning "Unexpected: More HQDs than software queues?"
    fi
    
    echo ""
    echo ""
done

# ========================================
# 总结
# ========================================
echo "============================================================"
echo "Key Findings"
echo "============================================================"
echo ""

print_success "✅ 通过读取 /sys/kernel/debug/kfd/mqds 和 hqds 可以看到真实的队列状态"
echo ""

echo "关键发现："
echo ""
echo "1. 软件队列（MQD）："
echo "   - 每个HIP stream创建一个软件队列（AQL Queue）"
echo "   - 每个队列有独立的ring buffer地址"
echo "   - 可以通过MQD中的cp_hqd_pq_base地址来统计"
echo ""

echo "2. 硬件队列（HQD）："
echo "   - 实际加载到GPU硬件的队列数量"
echo "   - 受硬件限制：32个（4 Pipes × 8 Queues）"
echo "   - 当软件队列 > 32时，会发生HQD复用"
echo ""

echo "3. CPSCH模式："
echo "   - HQD由MEC Firmware动态分配"
echo "   - 软件队列可能时分复用硬件队列"
echo "   - HQD allocation对驱动层透明"
echo ""

echo "4. 验证方法："
echo "   - MQD: 统计唯一的ring buffer地址（cp_hqd_pq_base）"
echo "   - HQD: 统计非零ring buffer地址的硬件队列"
echo "   - 对比：软件队列数 vs 硬件队列数"
echo ""

echo "============================================================"
print_success "Analysis Complete"
echo "============================================================"
