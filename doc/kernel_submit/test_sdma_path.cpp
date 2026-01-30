#include <hip/hip_runtime.h>
#include <stdio.h>
#include <unistd.h>

int main() {
    printf("=== SDMA Path Verification Test ===\n");
    printf("PID: %d\n", getpid());
    printf("\n请在另一个终端运行：\n");
    printf("sudo sh -c 'echo 1 > /sys/kernel/debug/tracing/events/drm/drm_run_job/enable'\n");
    printf("sudo sh -c 'echo > /sys/kernel/debug/tracing/trace'\n");
    printf("\n按回车开始测试...\n");
    getchar();
    
    float *h_data, *d_data;
    int size = 1024 * 1024;
    
    // 分配内存
    h_data = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        h_data[i] = i;
    }
    
    hipMalloc(&d_data, size * sizeof(float));
    
    printf("\n开始测试 H2D 拷贝（纯 SDMA 操作，无 compute kernel）\n");
    printf("如果 SDMA 走 KFD Ring，应该能在 ftrace 看到 sdma 事件\n");
    printf("如果 SDMA 走 Doorbell，ftrace 不会有任何事件\n");
    
    // 连续做 10 次 H2D 拷贝
    for (int i = 0; i < 10; i++) {
        hipMemcpy(d_data, h_data, size * sizeof(float), hipMemcpyHostToDevice);
        printf("  H2D copy %d/10\n", i+1);
    }
    
    hipDeviceSynchronize();
    
    printf("\n测试完成！\n");
    printf("现在查看 ftrace:\n");
    printf("sudo cat /sys/kernel/debug/tracing/trace | grep drm_run_job\n");
    printf("\n如果看到 sdma 事件：SDMA 走 KFD Ring ✓\n");
    printf("如果没有任何事件：SDMA 走 Doorbell ✓\n");
    
    hipFree(d_data);
    free(h_data);
    
    return 0;
}
