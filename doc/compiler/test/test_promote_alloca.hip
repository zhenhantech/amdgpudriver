#include <hip/hip_runtime.h>
#include <stdio.h>

// 测试：局部数组应该被提升到寄存器或LDS
__global__ void test_promote_alloca(int* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 小数组 - 应该提升到寄存器
    int local_array[4];
    local_array[0] = tid;
    local_array[1] = tid + 1;
    local_array[2] = tid + 2;
    local_array[3] = tid + 3;
    
    // 计算
    int sum = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        sum += local_array[i];
    }
    
    output[tid] = sum;
}

int main() {
    int *d_output;
    int h_output[256];
    
    hipMalloc(&d_output, 256 * sizeof(int));
    
    hipLaunchKernelGGL(test_promote_alloca, 
                       dim3(1), dim3(256), 0, 0, 
                       d_output);
    
    hipMemcpy(h_output, d_output, 256 * sizeof(int), hipMemcpyDeviceToHost);
    
    printf("First result: %d (expected: %d)\n", h_output[0], 0+1+2+3);
    printf("Test %s\n", (h_output[0] == 6) ? "PASSED" : "FAILED");
    
    hipFree(d_output);
    return 0;
}

