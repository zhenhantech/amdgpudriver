#!/bin/bash
#
# Stream Priority 测试 - 启用详细日志
#
# 用途: 运行测试并收集详细的 HIP/HSA 日志
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# 创建日志目录
LOG_DIR="logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

print_header "Stream Priority 测试 - 启用详细日志"
print_info "日志目录: $LOG_DIR"
echo ""

# 编译程序
print_header "步骤 1: 编译测试程序"
if make all > "$LOG_DIR/compile.log" 2>&1; then
    print_success "编译成功"
else
    print_error "编译失败，查看日志: $LOG_DIR/compile.log"
    exit 1
fi
echo ""

# 设置 HIP/HSA 日志环境变量
print_header "步骤 2: 配置日志环境"

# HIP Runtime 日志
export AMD_LOG_LEVEL=5              # 最详细的日志级别
export HIP_TRACE_API=1              # 追踪 HIP API 调用
export HIP_DB=0x1                   # 启用 debug 信息
export AMD_SERIALIZE_KERNEL=0       # 不串行化 kernel（看并发行为）
export AMD_SERIALIZE_COPY=0         # 不串行化 copy

# HSA Runtime 日志
export HSA_TOOLS_LIB=libhsa-runtime64.so.1
export HSA_ENABLE_INTERRUPT=0       # 禁用中断模式（看更多日志）
export HSA_ENABLE_SDMA=1            # 启用 SDMA

# ROCm 调试
export ROCR_VISIBLE_DEVICES=0
export GPU_MAX_HW_QUEUES=8          # 限制硬件队列数量（便于观察）

# 显示配置
print_info "HIP/HSA 日志级别:"
echo "  AMD_LOG_LEVEL        = $AMD_LOG_LEVEL (5=最详细)"
echo "  HIP_TRACE_API        = $HIP_TRACE_API (1=启用)"
echo "  HIP_DB               = $HIP_DB (0x1=debug)"
echo "  AMD_SERIALIZE_KERNEL = $AMD_SERIALIZE_KERNEL (0=不串行化)"
echo "  GPU_MAX_HW_QUEUES    = $GPU_MAX_HW_QUEUES"
echo ""

# 运行测试 - test_concurrent
print_header "步骤 3: 运行单进程测试 (test_concurrent)"
print_info "运行 ./test_concurrent"
print_info "输出保存到: $LOG_DIR/test_concurrent.log"
echo ""

if ./test_concurrent > "$LOG_DIR/test_concurrent.log" 2>&1; then
    print_success "test_concurrent 运行成功"
else
    print_error "test_concurrent 运行失败"
fi

# 显示部分日志
print_info "最后 30 行输出:"
tail -30 "$LOG_DIR/test_concurrent.log"
echo ""

# 分析日志
print_header "步骤 4: 分析日志"

echo "─── 搜索 Stream 创建 ───"
if grep -i "stream.*create\|hsa_queue_create" "$LOG_DIR/test_concurrent.log" > "$LOG_DIR/stream_create.txt" 2>/dev/null; then
    print_info "找到 $(wc -l < "$LOG_DIR/stream_create.txt") 条 Stream 创建记录"
    echo "详细信息保存在: $LOG_DIR/stream_create.txt"
    echo "前 10 条:"
    head -10 "$LOG_DIR/stream_create.txt"
else
    print_info "未找到 Stream 创建日志"
fi
echo ""

echo "─── 搜索 Queue 创建 ───"
if grep -i "queue.*create\|create.*queue\|aql.*queue" "$LOG_DIR/test_concurrent.log" > "$LOG_DIR/queue_create.txt" 2>/dev/null; then
    print_info "找到 $(wc -l < "$LOG_DIR/queue_create.txt") 条 Queue 创建记录"
    echo "详细信息保存在: $LOG_DIR/queue_create.txt"
    echo "前 10 条:"
    head -10 "$LOG_DIR/queue_create.txt"
else
    print_info "未找到 Queue 创建日志"
fi
echo ""

echo "─── 搜索 Doorbell 信息 ───"
if grep -i "doorbell\|door_bell" "$LOG_DIR/test_concurrent.log" > "$LOG_DIR/doorbell.txt" 2>/dev/null; then
    print_info "找到 $(wc -l < "$LOG_DIR/doorbell.txt") 条 Doorbell 记录"
    echo "详细信息保存在: $LOG_DIR/doorbell.txt"
    echo "前 10 条:"
    head -10 "$LOG_DIR/doorbell.txt"
else
    print_info "未找到 Doorbell 日志"
fi
echo ""

echo "─── 搜索 Priority 信息 ───"
if grep -i "priority" "$LOG_DIR/test_concurrent.log" > "$LOG_DIR/priority.txt" 2>/dev/null; then
    print_info "找到 $(wc -l < "$LOG_DIR/priority.txt") 条 Priority 记录"
    echo "详细信息保存在: $LOG_DIR/priority.txt"
    echo "前 10 条:"
    head -10 "$LOG_DIR/priority.txt"
else
    print_info "未找到 Priority 日志"
fi
echo ""

echo "─── 搜索 Warning/Error ───"
if grep -iE "warning|error|fail" "$LOG_DIR/test_concurrent.log" > "$LOG_DIR/warnings.txt" 2>/dev/null; then
    WARNING_COUNT=$(wc -l < "$LOG_DIR/warnings.txt")
    if [ "$WARNING_COUNT" -gt 0 ]; then
        print_info "找到 $WARNING_COUNT 条 Warning/Error"
        echo "详细信息保存在: $LOG_DIR/warnings.txt"
        echo "前 20 条:"
        head -20 "$LOG_DIR/warnings.txt"
    else
        print_success "没有 Warning/Error"
    fi
else
    print_success "没有 Warning/Error"
fi
echo ""

# 运行 test_app_A（可选）
print_header "步骤 5: 运行应用 A (可选)"
read -p "是否运行 test_app_A？(y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "运行 ./test_app_A"
    ./test_app_A > "$LOG_DIR/test_app_A.log" 2>&1 &
    APP_A_PID=$!
    print_info "test_app_A 已在后台运行 (PID: $APP_A_PID)"
    
    sleep 2
    
    print_info "是否运行 test_app_B？(y/N)"
    read -p "" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "运行 ./test_app_B"
        ./test_app_B > "$LOG_DIR/test_app_B.log" 2>&1
        
        wait $APP_A_PID 2>/dev/null || true
        
        print_success "test_app_A 和 test_app_B 运行完成"
        print_info "日志: $LOG_DIR/test_app_A.log, $LOG_DIR/test_app_B.log"
    fi
fi
echo ""

# 生成总结报告
print_header "步骤 6: 生成测试报告"

REPORT="$LOG_DIR/TEST_REPORT.md"

cat > "$REPORT" << EOF
# Stream Priority 测试报告（详细日志）

**生成时间**: $(date)  
**日志目录**: $LOG_DIR

---

## 测试环境

\`\`\`
操作系统: $(uname -a)
ROCm 版本: $(hipcc --version 2>/dev/null | head -1 || echo 'N/A')
GPU 设备: $(rocminfo 2>/dev/null | grep "Name:" | head -1 || echo 'N/A')
\`\`\`

---

## 日志配置

\`\`\`bash
AMD_LOG_LEVEL        = 5 (最详细)
HIP_TRACE_API        = 1 (启用)
HIP_DB               = 0x1 (debug)
AMD_SERIALIZE_KERNEL = 0 (不串行化)
GPU_MAX_HW_QUEUES    = 8
\`\`\`

---

## 测试结果

### test_concurrent
- 日志文件: \`test_concurrent.log\`
- Stream 创建: \`stream_create.txt\` ($(wc -l < "$LOG_DIR/stream_create.txt" 2>/dev/null || echo 0) 条)
- Queue 创建: \`queue_create.txt\` ($(wc -l < "$LOG_DIR/queue_create.txt" 2>/dev/null || echo 0) 条)
- Doorbell 信息: \`doorbell.txt\` ($(wc -l < "$LOG_DIR/doorbell.txt" 2>/dev/null || echo 0) 条)
- Priority 信息: \`priority.txt\` ($(wc -l < "$LOG_DIR/priority.txt" 2>/dev/null || echo 0) 条)
- Warnings: \`warnings.txt\` ($(wc -l < "$LOG_DIR/warnings.txt" 2>/dev/null || echo 0) 条)

---

## 关键发现

### Stream 创建记录

\`\`\`
$(head -20 "$LOG_DIR/stream_create.txt" 2>/dev/null || echo '未找到')
\`\`\`

### Queue 创建记录

\`\`\`
$(head -20 "$LOG_DIR/queue_create.txt" 2>/dev/null || echo '未找到')
\`\`\`

### Doorbell 信息

\`\`\`
$(head -20 "$LOG_DIR/doorbell.txt" 2>/dev/null || echo '未找到')
\`\`\`

### Warnings (前 20 条)

\`\`\`
$(head -20 "$LOG_DIR/warnings.txt" 2>/dev/null || echo '无警告')
\`\`\`

---

## 分析建议

1. 检查 \`stream_create.txt\` 确认 4 个 Stream 创建
2. 检查 \`queue_create.txt\` 确认 4 个 Queue 创建
3. 检查 \`doorbell.txt\` 确认 4 个不同的 doorbell 地址
4. 检查 \`warnings.txt\` 分析是否有实质性问题

---

## 文件列表

\`\`\`bash
ls -lh $LOG_DIR/
\`\`\`

\`\`\`
$(ls -lh "$LOG_DIR/" 2>/dev/null)
\`\`\`

EOF

cat "$REPORT"

print_success "测试报告已生成: $REPORT"
echo ""

# 提供后续分析建议
print_header "后续分析"

echo "1️⃣ 查看完整日志:"
echo "   cat $LOG_DIR/test_concurrent.log"
echo ""

echo "2️⃣ 搜索特定关键词:"
echo "   grep -i 'queue' $LOG_DIR/test_concurrent.log"
echo "   grep -i 'stream' $LOG_DIR/test_concurrent.log"
echo "   grep -i 'doorbell' $LOG_DIR/test_concurrent.log"
echo ""

echo "3️⃣ 查看 Warning 详情:"
echo "   cat $LOG_DIR/warnings.txt"
echo ""

echo "4️⃣ 查看测试报告:"
echo "   cat $LOG_DIR/TEST_REPORT.md"
echo ""

echo "5️⃣ 使用 less 分页查看:"
echo "   less $LOG_DIR/test_concurrent.log"
echo ""

print_header "测试完成"
print_success "所有日志已保存到: $LOG_DIR/"

echo ""
echo "如果发现 Warnings，请检查:"
echo "  • $LOG_DIR/warnings.txt - 所有警告信息"
echo "  • $LOG_DIR/test_concurrent.log - 完整日志"
echo ""
