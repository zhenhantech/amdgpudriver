/**
 * 测试程序 B - 验证跨进程的 Stream 和 Queue 独立性
 * 
 * 目的: 
 *   1. 创建 2 个不同优先级的 Stream
 *   2. 与应用 A 一起运行，验证跨进程的 Queue 独立性
 *   3. 验证不同进程的 doorbell 地址不同
 * 
 * 编译: hipcc test_app_B.cpp -o test_app_B
 * 运行: ./test_app_B
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
    printf("应用 B - Stream 和 Queue 独立性测试\n");
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
    hipStream_t stream_high, stream_normal;
    
    printf("─── 创建 Stream ───\n");
    
    // Stream 3: High Priority
    hipError_t err1 = hipStreamCreateWithPriority(&stream_high, hipStreamDefault, -1);
    if (err1 != hipSuccess) {
        printf("❌ 创建 stream_high 失败: %s\n", hipGetErrorString(err1));
        return 1;
    }
    printf("✅ Stream-3 (HIGH):   创建成功\n");
    printf("   地址: %p\n", stream_high);
    
    // Stream 4: Normal Priority
    hipError_t err2 = hipStreamCreateWithPriority(&stream_normal, hipStreamDefault, 0);
    if (err2 != hipSuccess) {
        printf("❌ 创建 stream_normal 失败: %s\n", hipGetErrorString(err2));
        return 1;
    }
    printf("✅ Stream-4 (NORMAL): 创建成功\n");
    printf("   地址: %p\n", stream_normal);
    printf("\n");
    
    // 验证 Stream 地址不同
    printf("─── 验证 Stream 独立性 ───\n");
    if (stream_high != stream_normal) {
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
    printf("提示: 观察 dmesg 输出，应该看到不同的 Queue ID\n");
    printf("\n");
    
    // 让用户有时间启动监控
    printf("等待 3 秒，请启动监控工具...\n");
    sleep(3);
    
    printf("开始提交 Kernel...\n\n");
    
    // 提交 kernel 到 stream_high
    printf("[Stream-3 HIGH]   提交 kernel...\n");
    hipLaunchKernelGGL(dummy_kernel, dim3(32), dim3(32), 0, stream_high, d_data1, 300);
    hipStreamSynchronize(stream_high);
    printf("[Stream-3 HIGH]   kernel 完成\n\n");
    
    sleep(1);
    
    // 提交 kernel 到 stream_normal
    printf("[Stream-4 NORMAL] 提交 kernel...\n");
    hipLaunchKernelGGL(dummy_kernel, dim3(32), dim3(32), 0, stream_normal, d_data2, 400);
    hipStreamSynchronize(stream_normal);
    printf("[Stream-4 NORMAL] kernel 完成\n\n");
    
    // 获取优先级
    int priority_high, priority_normal;
    hipStreamGetPriority(stream_high, &priority_high);
    hipStreamGetPriority(stream_normal, &priority_normal);
    
    printf("─── Stream 优先级 ───\n");
    printf("Stream-3: priority = %d (HIGH)\n", priority_high);
    printf("Stream-4: priority = %d (NORMAL)\n", priority_normal);
    printf("\n");
    
    // 保持运行，让用户检查
    printf("─── 保持运行 10 秒 ───\n");
    printf("提示: 此时可以对比应用 A 和应用 B 的 Queue ID\n");
    printf("提示: 使用 'cat /proc/%d/fd/* | grep kfd' 查看 KFD 连接\n", getpid());
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
    hipStreamDestroy(stream_normal);
    printf("✅ 清理完成\n\n");
    
    printf("═══════════════════════════════════════════════════════════\n");
    printf("应用 B 测试完成\n");
    printf("═══════════════════════════════════════════════════════════\n");
    
    return 0;
}
