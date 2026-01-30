/**
 * 测试程序 A - 验证 Stream 和 Queue 的独立性
 * 
 * 目的: 
 *   1. 创建 2 个不同优先级的 Stream
 *   2. 验证每个 Stream 有独立的 Queue
 *   3. 验证每个 Queue 有独立的 doorbell
 * 
 * 编译: hipcc test_app_A.cpp -o test_app_A
 * 运行: ./test_app_A
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>

// 简单的 kernel，用于触发 Queue 活动
__global__ void dummy_kernel(int* data, int val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = val + idx;
}

int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("应用 A - Stream 和 Queue 独立性测试\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("PID: %d\n", getpid());
    printf("\n");
    
    // 获取设备信息
    int device;
    hipGetDevice(&device);
    
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device);
    printf("GPU Device: %s\n", prop.name);
    printf("\n");
    
    // 创建 2 个不同优先级的 Stream
    hipStream_t stream_high, stream_low;
    
    printf("─── 创建 Stream ───\n");
    
    // Stream 1: High Priority
    hipError_t err1 = hipStreamCreateWithPriority(&stream_high, hipStreamDefault, -1);
    if (err1 != hipSuccess) {
        printf("❌ 创建 stream_high 失败: %s\n", hipGetErrorString(err1));
        return 1;
    }
    printf("✅ Stream-1 (HIGH):   创建成功\n");
    printf("   地址: %p\n", stream_high);
    
    // Stream 2: Low Priority
    hipError_t err2 = hipStreamCreateWithPriority(&stream_low, hipStreamDefault, 1);
    if (err2 != hipSuccess) {
        printf("❌ 创建 stream_low 失败: %s\n", hipGetErrorString(err2));
        return 1;
    }
    printf("✅ Stream-2 (LOW):    创建成功\n");
    printf("   地址: %p\n", stream_low);
    printf("\n");
    
    // 验证 Stream 地址不同
    printf("─── 验证 Stream 独立性 ───\n");
    if (stream_high != stream_low) {
        printf("✅ Stream 地址不同 → 独立的 Stream 对象\n");
    } else {
        printf("❌ Stream 地址相同 → 异常！\n");
    }
    printf("\n");
    
    // 分配设备内存
    int* d_data1;
    int* d_data2;
    size_t size = 1024 * sizeof(int);
    
    hipMalloc(&d_data1, size);
    hipMalloc(&d_data2, size);
    
    printf("─── 提交 Kernel 到不同 Stream ───\n");
    printf("提示: 使用 'sudo dmesg -w | grep -E \"create queue|doorbell\"' 监控 Queue 创建\n");
    printf("提示: 使用 'rocprofv3 --hip-trace ./test_app_A' 查看 Queue 详情\n");
    printf("\n");
    
    // 让用户有时间启动监控
    printf("等待 3 秒，请启动监控工具...\n");
    sleep(3);
    
    printf("开始提交 Kernel...\n\n");
    
    // 提交 kernel 到 stream_high
    printf("[Stream-1 HIGH] 提交 kernel...\n");
    hipLaunchKernelGGL(dummy_kernel, dim3(32), dim3(32), 0, stream_high, d_data1, 100);
    hipStreamSynchronize(stream_high);
    printf("[Stream-1 HIGH] kernel 完成\n\n");
    
    sleep(1);
    
    // 提交 kernel 到 stream_low
    printf("[Stream-2 LOW]  提交 kernel...\n");
    hipLaunchKernelGGL(dummy_kernel, dim3(32), dim3(32), 0, stream_low, d_data2, 200);
    hipStreamSynchronize(stream_low);
    printf("[Stream-2 LOW]  kernel 完成\n\n");
    
    // 获取优先级
    int priority_high, priority_low;
    hipStreamGetPriority(stream_high, &priority_high);
    hipStreamGetPriority(stream_low, &priority_low);
    
    printf("─── Stream 优先级 ───\n");
    printf("Stream-1: priority = %d (HIGH)\n", priority_high);
    printf("Stream-2: priority = %d (LOW)\n", priority_low);
    printf("\n");
    
    // 保持运行，让用户检查
    printf("─── 保持运行 10 秒 ───\n");
    printf("提示: 此时可以检查 /proc/%d/fd/ 查看 /dev/kfd\n", getpid());
    printf("提示: 可以使用 'lsof -p %d' 查看打开的文件\n", getpid());
    printf("\n");
    
    for (int i = 10; i > 0; i--) {
        printf("\r剩余 %2d 秒...", i);
        fflush(stdout);
        sleep(1);
    }
    printf("\n\n");
    
    // 清理
    printf("─── 清理资源 ───\n");
    hipFree(d_data1);
    hipFree(d_data2);
    hipStreamDestroy(stream_high);
    hipStreamDestroy(stream_low);
    printf("✅ 清理完成\n\n");
    
    printf("═══════════════════════════════════════════════════════════\n");
    printf("应用 A 测试完成\n");
    printf("═══════════════════════════════════════════════════════════\n");
    
    return 0;
}
