#include <hip/hip_runtime.h>
#include <stdio.h>
#include <unistd.h>

__global__ void simpleKernel(int *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = idx * 2;
}

int main() {
    printf("=== Test Queue Creation for Debug Logging ===\n");
    printf("This program creates new HIP streams to trigger HQD allocation\n\n");
    
    // 初始化 HIP
    hipSetDevice(0);
    
    // 创建多个 Stream，每个都会分配一个新的 Queue
    const int NUM_STREAMS = 4;
    hipStream_t streams[NUM_STREAMS];
    
    printf("Creating %d HIP streams...\n", NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        hipStreamCreate(&streams[i]);
        printf("  Stream %d created\n", i);
        usleep(100000);  // 延迟 100ms，方便观察日志
    }
    
    printf("\nLaunching kernels on each stream...\n");
    
    // 在每个 Stream 上启动一个 kernel
    int *d_data;
    hipMalloc(&d_data, 1024 * sizeof(int));
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        hipLaunchKernelGGL(simpleKernel, dim3(32), dim3(32), 0, streams[i], d_data);
        printf("  Kernel launched on Stream %d\n", i);
        usleep(100000);  // 延迟 100ms
    }
    
    printf("\nSynchronizing streams...\n");
    for (int i = 0; i < NUM_STREAMS; i++) {
        hipStreamSynchronize(streams[i]);
        printf("  Stream %d synchronized\n", i);
    }
    
    printf("\nDestroying streams...\n");
    for (int i = 0; i < NUM_STREAMS; i++) {
        hipStreamDestroy(streams[i]);
        printf("  Stream %d destroyed\n", i);
        usleep(100000);  // 延迟 100ms
    }
    
    hipFree(d_data);
    
    printf("\n✅ Test completed!\n");
    printf("\nCheck dmesg for 'hqd slot' messages:\n");
    printf("  sudo dmesg | grep 'hqd slot'\n\n");
    
    return 0;
}

