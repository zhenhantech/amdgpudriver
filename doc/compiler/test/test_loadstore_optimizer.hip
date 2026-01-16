#include <hip/hip_runtime.h>
#include <stdio.h>

// 测试：连续内存访问优化
__global__ void test_load_merge(float* input, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 4;
    
    // 连续的4个load - 应该被合并为向量load
    float a = input[idx + 0];
    float b = input[idx + 1];
    float c = input[idx + 2];
    float d = input[idx + 3];
    
    // 计算
    float sum = a + b + c + d;
    
    // 写回
    output[tid] = sum;
}

int main() {
    const int N = 256;
    float *d_input, *d_output;
    float h_input[N * 4], h_output[N];
    
    // 初始化
    for (int i = 0; i < N * 4; i++) h_input[i] = i;
    
    hipMalloc(&d_input, N * 4 * sizeof(float));
    hipMalloc(&d_output, N * sizeof(float));
    
    hipMemcpy(d_input, h_input, N * 4 * sizeof(float), hipMemcpyHostToDevice);
    
    hipLaunchKernelGGL(test_load_merge, 
                       dim3(1), dim3(N), 0, 0, 
                       d_input, d_output);
    
    hipMemcpy(h_output, d_output, N * sizeof(float), hipMemcpyDeviceToHost);
    
    float expected = 0.0f + 1.0f + 2.0f + 3.0f;
    printf("First result: %.0f (expected: %.0f)\n", h_output[0], expected);
    printf("Test %s\n", (h_output[0] == expected) ? "PASSED" : "FAILED");
    
    hipFree(d_input);
    hipFree(d_output);
    return 0;
}

