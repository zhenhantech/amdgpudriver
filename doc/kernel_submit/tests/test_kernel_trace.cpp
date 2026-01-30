// 测试程序：验证 Kernel 提交流程
// 编译: hipcc -o test_kernel_trace test_kernel_trace.cpp
// 运行: ./test_kernel_trace

#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>

#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = (cmd); \
        if (error != hipSuccess) { \
            std::cerr << "Error: '" << hipGetErrorString(error) << "' (" << error << ") at " \
                      << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 简单的向量加法 kernel
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char** argv) {
    std::cout << "=== Kernel Submission Flow Test ===" << std::endl;
    
    // 1. 获取 GPU 信息
    int deviceId = 0;
    HIP_CHECK(hipSetDevice(deviceId));
    
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, deviceId));
    
    std::cout << "\n[1] GPU Information:" << std::endl;
    std::cout << "  - Device Name: " << prop.name << std::endl;
    std::cout << "  - PCI Bus ID: " << prop.pciBusID << std::endl;
    std::cout << "  - PCI Device ID: " << prop.pciDeviceID << std::endl;
    std::cout << "  - Compute Units: " << prop.multiProcessorCount << std::endl;
    std::cout << "  - Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
    
    // 检查 MES 支持 (通过读取系统参数)
    std::cout << "\n[2] Scheduler Mode:" << std::endl;
    system("cat /sys/module/amdgpu/parameters/mes 2>/dev/null | xargs -I {} echo '  - MES enabled: {}'");
    
    // 2. 准备数据
    const int N = 1024 * 1024; // 1M elements
    const size_t bytes = N * sizeof(float);
    
    std::cout << "\n[3] Memory Allocation:" << std::endl;
    std::cout << "  - Array size: " << N << " elements" << std::endl;
    std::cout << "  - Memory size: " << bytes / (1024 * 1024) << " MB per array" << std::endl;
    
    // Host 内存
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];
    
    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }
    
    // Device 内存
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, bytes));
    HIP_CHECK(hipMalloc(&d_B, bytes));
    HIP_CHECK(hipMalloc(&d_C, bytes));
    
    std::cout << "  - Device memory allocated" << std::endl;
    
    // 3. 拷贝数据到 GPU
    std::cout << "\n[4] Memory Transfer (Host -> Device):" << std::endl;
    auto start_transfer = std::chrono::high_resolution_clock::now();
    
    HIP_CHECK(hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice));
    
    auto end_transfer = std::chrono::high_resolution_clock::now();
    auto transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(end_transfer - start_transfer).count();
    std::cout << "  - Transfer time: " << transfer_time << " us" << std::endl;
    
    // 4. 配置 kernel 启动参数
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    std::cout << "\n[5] Kernel Launch Configuration:" << std::endl;
    std::cout << "  - Block size: " << blockSize << " threads" << std::endl;
    std::cout << "  - Grid size: " << numBlocks << " blocks" << std::endl;
    std::cout << "  - Total threads: " << numBlocks * blockSize << std::endl;
    
    // 5. 启动 kernel（这里会触发整个提交流程）
    std::cout << "\n[6] Launching Kernel:" << std::endl;
    std::cout << "  - Kernel: vectorAdd" << std::endl;
    std::cout << "  - Flow: hipLaunchKernel -> HIP Runtime -> HSA Runtime -> KFD -> MES/CPSCH" << std::endl;
    
    auto start_kernel = std::chrono::high_resolution_clock::now();
    
    // 这个调用会经过我们文档描述的完整流程
    hipLaunchKernelGGL(vectorAdd, 
                       dim3(numBlocks), 
                       dim3(blockSize), 
                       0,                  // shared memory
                       0,                  // stream (default stream)
                       d_A, d_B, d_C, N);
    
    // 等待 kernel 完成
    HIP_CHECK(hipDeviceSynchronize());
    
    auto end_kernel = std::chrono::high_resolution_clock::now();
    auto kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(end_kernel - start_kernel).count();
    std::cout << "  - Kernel execution time: " << kernel_time << " us" << std::endl;
    
    // 6. 拷贝结果回 host
    std::cout << "\n[7] Memory Transfer (Device -> Host):" << std::endl;
    HIP_CHECK(hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost));
    
    // 7. 验证结果
    std::cout << "\n[8] Verification:" << std::endl;
    bool success = true;
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            std::cerr << "  - Error at index " << i << ": expected " << expected 
                      << ", got " << h_C[i] << std::endl;
            success = false;
            break;
        }
    }
    
    if (success) {
        std::cout << "  - ✅ All results correct!" << std::endl;
    } else {
        std::cout << "  - ❌ Verification failed!" << std::endl;
    }
    
    // 清理
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    
    return success ? 0 : 1;
}

