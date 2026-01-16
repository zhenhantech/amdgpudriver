/**
 * HIPIFY 真实案例：归约操作的寄存器优化
 * 来源: HIPIFY官方测试套件 tests/unit_tests/samples/reduction.cu
 * 
 * 本示例展示3种不同优化级别的归约操作：
 * 1. 未优化版本 (基础循环)
 * 2. 4倍循环展开优化
 * 3. 16倍循环展开优化
 * 
 * 学习目标: 理解寄存器优化如何影响GPU性能
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>

// ❌ 版本1: 未优化 - 简单循环，寄存器使用效率低
__global__ void atomic_reduction_kernel(int *in, int *out, int ARRAYSIZE) {
    int sum = int(0);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每次迭代只处理一个元素
    for (int i = idx; i < ARRAYSIZE; i += blockDim.x * gridDim.x) {
        sum += in[i];
    }
    
    atomicAdd(out, sum);
}

// ✅ 版本2: 优化 - 4倍循环展开
__global__ void atomic_reduction_kernel3(int *in, int *out, int ARRAYSIZE) {
    int sum = int(0);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 策略：循环展开4倍，减少循环开销和寄存器溢出
    for(int i = idx * 4; i < ARRAYSIZE - 3; i += blockDim.x * gridDim.x * 4) {
        sum += in[i] + in[i+1] + in[i+2] + in[i+3];
    }
    
    // 处理剩余元素
    for(int i = idx * 4 + ((ARRAYSIZE / 4) * 4); i < ARRAYSIZE; i += blockDim.x * gridDim.x) {
        if(i < ARRAYSIZE) sum += in[i];
    }
    
    atomicAdd(out, sum);
}

// ✅✅ 版本3: 激进优化 - 16倍循环展开
__global__ void atomic_reduction_kernel2(int *in, int *out, int ARRAYSIZE) {
    int sum = int(0);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 展开16倍
    for(int i = idx * 16; i < ARRAYSIZE - 15; i += blockDim.x * gridDim.x * 16) {
        sum += in[i] + in[i+1] + in[i+2] + in[i+3] + in[i+4] + in[i+5] 
             + in[i+6] + in[i+7] + in[i+8] + in[i+9] + in[i+10]
             + in[i+11] + in[i+12] + in[i+13] + in[i+14] + in[i+15];
    }
    
    // 处理剩余元素
    for(int i = idx * 16 + ((ARRAYSIZE / 16) * 16); i < ARRAYSIZE; i += blockDim.x * gridDim.x) {
        if(i < ARRAYSIZE) sum += in[i];
    }
    
    atomicAdd(out, sum);
}

// 错误检查宏
#define HIP_CHECK(call) \
{ \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) \
                  << " (line " << __LINE__ << ")" << std::endl; \
        exit(1); \
    } \
}

// 性能测试函数
template<typename KernelFunc>
double benchmark_kernel(KernelFunc kernel, int *d_in, int *d_out, int arraySize, 
                       int blockSize, int gridSize, const char* name) {
    // 重置输出
    int zero = 0;
    HIP_CHECK(hipMemcpy(d_out, &zero, sizeof(int), hipMemcpyHostToDevice));
    
    // 预热
    hipLaunchKernelGGL(kernel, dim3(gridSize), dim3(blockSize), 0, 0, d_in, d_out, arraySize);
    HIP_CHECK(hipDeviceSynchronize());
    
    // 计时
    auto start = std::chrono::high_resolution_clock::now();
    
    for(int i = 0; i < 10; i++) {
        HIP_CHECK(hipMemcpy(d_out, &zero, sizeof(int), hipMemcpyHostToDevice));
        hipLaunchKernelGGL(kernel, dim3(gridSize), dim3(blockSize), 0, 0, d_in, d_out, arraySize);
    }
    
    HIP_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count() / 10.0;
    
    // 验证结果
    int result;
    HIP_CHECK(hipMemcpy(&result, d_out, sizeof(int), hipMemcpyDeviceToHost));
    
    std::cout << "🔹 " << name << std::endl;
    std::cout << "   执行时间: " << elapsed << " ms" << std::endl;
    std::cout << "   结果: " << result << " (期望: " << arraySize << ")" << std::endl;
    std::cout << "   带宽: " << (arraySize * sizeof(int) / elapsed / 1e6) << " GB/s" << std::endl;
    
    return elapsed;
}

int main(int argc, char** argv) {
    // 默认数组大小
    int arraySize = 52428800;  // ~50M elements
    if(argc > 1) {
        arraySize = atoi(argv[1]);
    }
    
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "📊 HIPIFY 归约优化示例" << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "数组大小: " << arraySize << " elements" << std::endl;
    std::cout << "内存占用: " << (arraySize * sizeof(int) / 1024.0 / 1024.0) << " MB" << std::endl;
    
    // 分配内存
    int *h_in = new int[arraySize];
    for(int i = 0; i < arraySize; i++) {
        h_in[i] = 1;  // 每个元素为1，总和应等于arraySize
    }
    
    int *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, arraySize * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(int)));
    HIP_CHECK(hipMemcpy(d_in, h_in, arraySize * sizeof(int), hipMemcpyHostToDevice));
    
    // 配置
    int blockSize = 256;
    int gridSize = 512;
    
    std::cout << "\n⚙️  配置:" << std::endl;
    std::cout << "   Block Size: " << blockSize << std::endl;
    std::cout << "   Grid Size: " << gridSize << std::endl;
    std::cout << "   总线程数: " << (blockSize * gridSize) << std::endl;
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" << std::endl;
    
    // 测试三个版本
    double time1 = benchmark_kernel(atomic_reduction_kernel, d_in, d_out, arraySize, 
                                    blockSize, gridSize, "版本1: 未优化 (基础循环)");
    std::cout << std::endl;
    
    double time2 = benchmark_kernel(atomic_reduction_kernel3, d_in, d_out, arraySize, 
                                    blockSize, gridSize, "版本2: 4倍循环展开");
    std::cout << std::endl;
    
    double time3 = benchmark_kernel(atomic_reduction_kernel2, d_in, d_out, arraySize, 
                                    blockSize, gridSize, "版本3: 16倍循环展开");
    
    // 性能对比
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "📈 性能对比:" << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "版本2 vs 版本1: " << (time1 / time2) << "x 加速 ("
              << ((time1 - time2) / time1 * 100) << "% 提升)" << std::endl;
    std::cout << "版本3 vs 版本1: " << (time1 / time3) << "x 加速 ("
              << ((time1 - time3) / time1 * 100) << "% 提升)" << std::endl;
    std::cout << "版本3 vs 版本2: " << (time2 / time3) << "x 加速 ("
              << ((time2 - time3) / time2 * 100) << "% 提升)" << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" << std::endl;
    
    // 学习要点
    std::cout << "💡 学习要点:" << std::endl;
    std::cout << "   1. 循环展开减少循环开销和分支判断" << std::endl;
    std::cout << "   2. 更好的指令级并行（ILP）" << std::endl;
    std::cout << "   3. 但过度展开可能导致寄存器溢出" << std::endl;
    std::cout << "   4. 需要在寄存器压力和ILP之间权衡" << std::endl;
    std::cout << "\n🔧 优化建议:" << std::endl;
    std::cout << "   - 使用 'hipcc --resource-usage' 查看寄存器使用" << std::endl;
    std::cout << "   - 使用 'rocprof --stats' 进行详细性能分析" << std::endl;
    std::cout << "   - 根据具体GPU架构调整展开倍数" << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" << std::endl;
    
    // 清理
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    delete[] h_in;
    
    return 0;
}

