#include <hip/hip_runtime.h>
#include <stdio.h>

// 测试：原子操作优化
__global__ void test_atomic_add(int* counter) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程原子加1
    // AtomicOptimizer应该将wavefront内的64个操作合并为1个
    atomicAdd(counter, 1);
}

int main() {
    int *d_counter;
    int h_counter = 0;
    
    hipMalloc(&d_counter, sizeof(int));
    hipMemcpy(d_counter, &h_counter, sizeof(int), hipMemcpyHostToDevice);
    
    // 启动64个线程（1个wavefront）
    hipLaunchKernelGGL(test_atomic_add, 
                       dim3(1), dim3(64), 0, 0, 
                       d_counter);
    
    hipMemcpy(&h_counter, d_counter, sizeof(int), hipMemcpyDeviceToHost);
    
    printf("Counter: %d (expected: 64)\n", h_counter);
    printf("Test %s\n", (h_counter == 64) ? "PASSED" : "FAILED");
    
    hipFree(d_counter);
    return 0;
}

