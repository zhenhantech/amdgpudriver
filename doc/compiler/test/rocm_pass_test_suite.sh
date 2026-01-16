#!/bin/bash

set -e

ARCH=${1:-gfx90a}
OUTDIR=/tmp/rocm_pass_tests
mkdir -p $OUTDIR

echo "======================================"
echo "ROCm LLVM Pass测试套件"
echo "GPU架构: $ARCH"
echo "输出目录: $OUTDIR"
echo "======================================"

# 测试1: PromoteAlloca
echo -e "\n[测试1] PromoteAlloca优化..."
cat > $OUTDIR/test_alloca.hip << 'HIPEOF'
#include <hip/hip_runtime.h>
__global__ void kernel(int* out) {
    int arr[8];
    for (int i = 0; i < 8; i++) arr[i] = i;
    int sum = 0;
    for (int i = 0; i < 8; i++) sum += arr[i];
    out[threadIdx.x] = sum;
}
int main() { return 0; }
HIPEOF

hipcc -O3 --offload-arch=$ARCH \
      -mllvm -stats \
      $OUTDIR/test_alloca.hip -o $OUTDIR/test_alloca 2>&1 | \
      grep "promote-alloca" | tee $OUTDIR/result_alloca.txt

# 测试2: AtomicOptimizer
echo -e "\n[测试2] AtomicOptimizer..."
cat > $OUTDIR/test_atomic.hip << 'HIPEOF'
#include <hip/hip_runtime.h>
__global__ void kernel(int* counter) {
    atomicAdd(counter, 1);
}
int main() { return 0; }
HIPEOF

hipcc -O3 --offload-arch=$ARCH \
      -mllvm -amdgpu-atomic-optimizer-strategy=DPP \
      -mllvm -stats \
      $OUTDIR/test_atomic.hip -o $OUTDIR/test_atomic 2>&1 | \
      grep "atomic-optimizer" | tee $OUTDIR/result_atomic.txt

# 测试3: LoadStoreOptimizer
echo -e "\n[测试3] LoadStoreOptimizer..."
cat > $OUTDIR/test_loadstore.hip << 'HIPEOF'
#include <hip/hip_runtime.h>
__global__ void kernel(float* in, float* out) {
    int i = threadIdx.x * 4;
    out[threadIdx.x] = in[i] + in[i+1] + in[i+2] + in[i+3];
}
int main() { return 0; }
HIPEOF

hipcc -O3 --offload-arch=$ARCH \
      -mllvm -stats \
      $OUTDIR/test_loadstore.hip -o $OUTDIR/test_loadstore 2>&1 | \
      grep "load-store" | tee $OUTDIR/result_loadstore.txt

# 测试4: Pass执行时间
echo -e "\n[测试4] Pass执行时间统计..."
hipcc -O3 --offload-arch=$ARCH \
      -mllvm -time-passes \
      $OUTDIR/test_alloca.hip -o $OUTDIR/test_timing 2>&1 | \
      grep -E "AMDGPU|SI|GCN" | head -20 | tee $OUTDIR/result_timing.txt

echo -e "\n======================================"
echo "测试完成！结果保存在: $OUTDIR/"
echo "======================================"

ls -lh $OUTDIR/result_*.txt

