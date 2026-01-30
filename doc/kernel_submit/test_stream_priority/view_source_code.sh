#!/bin/bash
#
# 查看 Stream/Queue 相关源代码
#
# 用途: 快速查看文档中引用的原始代码

BASE_DIR="/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/ROCm_keyDriver"

# 颜色定义
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'

print_section() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
}

# 1. HIP Stream 创建
print_section "1. HIP Stream 创建 (hip_stream.cpp)"
echo "文件: rocm-systems/projects/clr/hipamd/src/hip_stream.cpp"
echo ""
echo -e "${GREEN}ihipStreamCreate() - Line 188-206:${NC}"
sed -n '188,206p' "$BASE_DIR/rocm-systems/projects/clr/hipamd/src/hip_stream.cpp"
echo ""
echo -e "${GREEN}hipStreamCreateWithPriority() - Line 299-316:${NC}"
sed -n '299,316p' "$BASE_DIR/rocm-systems/projects/clr/hipamd/src/hip_stream.cpp"
echo ""
echo ""

# 2. AQL Queue 构造
print_section "2. AQL Queue 构造函数 (amd_aql_queue.cpp)"
echo "文件: rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp"
echo ""
echo -e "${GREEN}AqlQueue::AqlQueue() - Line 81-130:${NC}"
sed -n '81,130p' "$BASE_DIR/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp"
echo ""
echo -e "${GREEN}KFD CreateQueue 调用 - Line 269-289:${NC}"
sed -n '269,289p' "$BASE_DIR/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp"
echo ""
echo -e "${GREEN}SetPriority() - Line 634-643:${NC}"
sed -n '634,643p' "$BASE_DIR/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp"
echo ""
echo ""

# 3. GPU Agent QueueCreate
print_section "3. GPU Agent QueueCreate (amd_gpu_agent.cpp)"
echo "文件: rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp"
echo ""
echo -e "${GREEN}InitDma() Priority Lambda - Line 777-798:${NC}"
sed -n '777,798p' "$BASE_DIR/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp"
echo ""
echo -e "${GREEN}GpuAgent::QueueCreate() - Line 1735-1760:${NC}"
sed -n '1735,1760p' "$BASE_DIR/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp"
echo ""
echo ""

# 4. KFD MQD 优先级
print_section "4. KFD MQD 优先级设置 (kfd_mqd_manager_v11.c)"
echo "文件: kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_mqd_manager_v11.c"
echo ""
echo -e "${GREEN}set_priority() - Line 96-100:${NC}"
sed -n '96,100p' "$BASE_DIR/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_mqd_manager_v11.c"
echo ""
echo ""

# 5. 优先级映射表
print_section "5. 优先级映射表 (kfd_mqd_manager.c)"
echo "文件: kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_mqd_manager.c"
echo ""
echo -e "${GREEN}pipe_priority_map[] - Line 29-47:${NC}"
sed -n '29,47p' "$BASE_DIR/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_mqd_manager.c"
echo ""
echo ""

# 总结
print_section "完成"
echo "所有关键代码已显示"
echo ""
echo "查看完整文件:"
echo "  vim $BASE_DIR/rocm-systems/projects/clr/hipamd/src/hip_stream.cpp +188"
echo "  vim $BASE_DIR/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp +81"
echo "  vim $BASE_DIR/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp +1735"
echo "  vim $BASE_DIR/kfd-amdgpu-debug-20260106/amd/amdkfd/kfd_mqd_manager_v11.c +96"
echo ""
