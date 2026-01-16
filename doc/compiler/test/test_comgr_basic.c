#include <amd_comgr.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 辅助函数：打印 comgr 状态
const char* status_string(amd_comgr_status_t status) {
    switch (status) {
        case AMD_COMGR_STATUS_SUCCESS: return "SUCCESS";
        case AMD_COMGR_STATUS_ERROR: return "ERROR";
        case AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES: return "OUT_OF_RESOURCES";
        default: return "UNKNOWN";
    }
}

#define CHECK(call) do { \
    amd_comgr_status_t status = call; \
    if (status != AMD_COMGR_STATUS_SUCCESS) { \
        fprintf(stderr, "Error at line %d: %s (%s)\n", \
                __LINE__, #call, status_string(status)); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("========================================\n");
    printf("comgr 基础功能测试\n");
    printf("========================================\n\n");
    
    // 1. 获取 comgr 版本
    size_t major, minor;
    CHECK(amd_comgr_get_version(&major, &minor));
    printf("1. comgr 版本: %zu.%zu\n\n", major, minor);
    
    // 2. 查询支持的 ISA
    size_t isa_count;
    CHECK(amd_comgr_get_isa_count(&isa_count));
    printf("2. 支持的 ISA 数量: %zu\n", isa_count);
    printf("   支持的架构列表:\n");
    for (size_t i = 0; i < isa_count && i < 10; i++) {  // 只显示前10个
        const char* isa_name;
        CHECK(amd_comgr_get_isa_name(i, &isa_name));
        printf("   [%zu] %s\n", i, isa_name);
    }
    printf("\n");
    
    // 3. 简单的编译测试：HIP 内核源码 → Bitcode
    const char* hip_source = 
        "extern \"C\" __global__ void simple_add(int* a, int* b, int* c, int n) {\n"
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (i < n) {\n"
        "        c[i] = a[i] + b[i];\n"
        "    }\n"
        "}\n";
    
    printf("3. 编译测试内核:\n");
    printf("   源码长度: %zu 字节\n", strlen(hip_source));
    
    // 创建输入数据
    amd_comgr_data_t input_data;
    CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &input_data));
    CHECK(amd_comgr_set_data(input_data, strlen(hip_source), hip_source));
    CHECK(amd_comgr_set_data_name(input_data, "simple_add.hip"));
    
    // 创建输入数据集
    amd_comgr_data_set_t input_set;
    CHECK(amd_comgr_create_data_set(&input_set));
    CHECK(amd_comgr_data_set_add(input_set, input_data));
    
    // 创建编译动作配置
    amd_comgr_action_info_t action;
    CHECK(amd_comgr_create_action_info(&action));
    CHECK(amd_comgr_action_info_set_language(action, AMD_COMGR_LANGUAGE_HIP));
    
    // 设置目标架构（使用 gfx900 作为通用架构）
    const char* target_isa = "amdgcn-amd-amdhsa--gfx900";
    CHECK(amd_comgr_action_info_set_isa_name(action, target_isa));
    printf("   目标架构: %s\n", target_isa);
    
    // 设置编译选项
    const char* options[] = {"-O2"};
    CHECK(amd_comgr_action_info_set_option_list(action, options, 1));
    
    // 创建输出数据集
    amd_comgr_data_set_t output_set;
    CHECK(amd_comgr_create_data_set(&output_set));
    
    // 编译：源码 → LLVM Bitcode
    printf("   编译中...\n");
    amd_comgr_status_t compile_status = amd_comgr_do_action(
        AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
        action,
        input_set,
        output_set
    );
    
    if (compile_status == AMD_COMGR_STATUS_SUCCESS) {
        printf("   ✅ 编译成功！\n");
        
        // 获取生成的 Bitcode 数量
        size_t bc_count;
        CHECK(amd_comgr_action_data_count(output_set, AMD_COMGR_DATA_KIND_BC, &bc_count));
        printf("   生成的 Bitcode 对象数量: %zu\n", bc_count);
        
        if (bc_count > 0) {
            // 获取 Bitcode 大小
            amd_comgr_data_t bc_data;
            CHECK(amd_comgr_action_data_get_data(output_set, AMD_COMGR_DATA_KIND_BC, 0, &bc_data));
            
            size_t bc_size;
            CHECK(amd_comgr_get_data(bc_data, &bc_size, NULL));
            printf("   Bitcode 大小: %zu 字节\n", bc_size);
            
            amd_comgr_release_data(bc_data);
        }
    } else {
        printf("   ❌ 编译失败: %s\n", status_string(compile_status));
        
        // 尝试获取编译日志
        size_t log_count;
        amd_comgr_action_data_count(output_set, AMD_COMGR_DATA_KIND_LOG, &log_count);
        
        if (log_count > 0) {
            amd_comgr_data_t log_data;
            amd_comgr_action_data_get_data(output_set, AMD_COMGR_DATA_KIND_LOG, 0, &log_data);
            
            size_t log_size;
            amd_comgr_get_data(log_data, &log_size, NULL);
            
            if (log_size > 0) {
                char* log = malloc(log_size + 1);
                amd_comgr_get_data(log_data, &log_size, log);
                log[log_size] = '\0';
                
                printf("\n   编译日志:\n%s\n", log);
                free(log);
            }
            
            amd_comgr_release_data(log_data);
        }
    }
    
    // 清理
    amd_comgr_destroy_data_set(input_set);
    amd_comgr_destroy_data_set(output_set);
    amd_comgr_destroy_data(input_data);
    amd_comgr_destroy_action_info(action);
    
    printf("\n========================================\n");
    printf("测试完成！\n");
    printf("========================================\n");
    
    return 0;
}

