# NVIDIA GPreempt 双路径分析：任务提交 vs 抢占控制

**日期**: 2026-01-28  
**核心问题**: Pushbuffer 提交和 ioctl 抢占是两条完全不同的路径！  
**关键修正**: 必须区分"应用提交"和"调度器控制"

---

## 🎯 核心混淆点

```
我之前的错误理解:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

错误: 认为 GPreempt 的任务提交需要 ioctl
      Pushbuffer 和 ioctl 在同一个流程中

❌ 这是完全错误的！

正确理解:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPreempt 有**两条完全独立的路径**:

路径 1: 应用任务提交（数据平面）
  • cuLaunchKernel() → Pushbuffer + MMIO
  • 用户态，快速（~100ns）
  • 不经过 ioctl ✅
  • 任何 CUDA 应用都是这样

路径 2: 调度器抢占控制（控制平面）⭐⭐⭐
  • GPreempt 用户态调度器 → ioctl → 驱动 → GPU
  • 用户态（但是独立的 GPreempt 进程）
  • 慢（~1-10μs）
  • 专门用于抢占控制

这是两个完全不同的角色和流程！
```

---

## 📊 NVIDIA GPreempt 完整架构

### 架构全景图

```
NVIDIA GPreempt 系统架构:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌──────────────────────────────────────────────────────────────┐
│ 用户态 - 应用进程（训练/推理）                                │
│                                                               │
│ Application 1 (低优先级训练):                                 │
│   void train() {                                             │
│     cuLaunchKernel(train_kernel, ...);  ← 路径 1 ⭐          │
│       ↓                                                       │
│     libcuda.so:                                              │
│       • 写 Pushbuffer (Ring Buffer)                          │
│       • *pushbuf_put = new_value  (MMIO write ~100ns)       │
│       • 不经过内核！✅                                        │
│   }                                                           │
│                                                               │
│ Application 2 (高优先级推理):                                 │
│   void inference() {                                         │
│     cuLaunchKernel(infer_kernel, ...);  ← 路径 1 ⭐          │
│       ↓                                                       │
│     libcuda.so:                                              │
│       • 写 Pushbuffer                                        │
│       • MMIO write (~100ns)                                  │
│       • 不经过内核！✅                                        │
│   }                                                           │
│                                                               │
└───────────────────────────────┬──────────────────────────────┘
                                │ Pushbuffer + MMIO (~100ns)
                                ↓
┌──────────────────────────────────────────────────────────────┐
│ GPU Hardware - 任务接收                                       │
│   • PFIFO Engine 监控 Pushbuffer                             │
│   • 检测 PUT pointer 更新                                     │
│   • DMA 读取命令                                              │
│   • 但按照默认调度策略执行（可能不是最优）❌                  │
└──────────────────────────────────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    这是第一条路径（数据平面）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━



┌──────────────────────────────────────────────────────────────┐
│ 用户态 - GPreempt 调度器进程（独立进程！）⭐⭐⭐              │
│                                                               │
│ GPreempt Scheduler (用户态守护进程):                         │
│   int main() {                                               │
│     while (1) {                                              │
│       // 1. 监控所有应用的 Context                           │
│       for (ctx in all_contexts) {                            │
│         query_context_status(ctx);  ← 路径 2.1               │
│       }                                                       │
│                                                               │
│       // 2. 检测优先级倒置                                   │
│       if (detect_inversion(&high_ctx, &low_ctx)) {          │
│         // 3. 触发抢占 ⭐⭐⭐                                 │
│         preempt_context(low_ctx);  ← 路径 2.2 ⭐             │
│       }                                                       │
│                                                               │
│       usleep(1000);  // 1ms 轮询                             │
│     }                                                         │
│   }                                                           │
│                                                               │
│   // ⭐ 路径 2.1: 查询 Context 状态                          │
│   void query_context_status(NvContext ctx) {                │
│     int fd = open("/dev/nvidiactl", O_RDWR);                │
│     NvRmQuery(&ctx);  // ioctl (1-10μs)                     │
│     // 返回: channels, active_waves, pending_work           │
│   }                                                           │
│                                                               │
│   // ⭐⭐⭐ 路径 2.2: 触发抢占                                │
│   void preempt_context(NvContext ctx) {                     │
│     int fd = open("/dev/nvidiactl", O_RDWR);                │
│     NvRmPreempt(ctx);  // ioctl (1-10μs) ⭐⭐⭐              │
│     // 这会触发驱动向 GPU 发送抢占命令                       │
│   }                                                           │
│                                                               │
└───────────────────────────────┬──────────────────────────────┘
                                │ ioctl (系统调用)
                                ↓
┌──────────────────────────────────────────────────────────────┐
│ NVIDIA Kernel Driver (nvidia.ko) - 抢占控制                  │
│                                                               │
│ kfd_ioctl_handler() {                                        │
│   switch (cmd) {                                             │
│     case NV_QUERY_CONTEXT:                                   │
│       return query_context_info(...);                        │
│                                                               │
│     case NV_PREEMPT_CONTEXT: ⭐⭐⭐                           │
│       // 1. 查找目标 Context                                 │
│       ctx = find_context(hClient, hObject);                  │
│                                                               │
│       // 2. 标记为待抢占                                     │
│       ctx->preemption_pending = true;                        │
│                                                               │
│       // 3. ⭐ 向 GPU 发送抢占命令                           │
│       //    (写特殊的控制寄存器)                             │
│       writel(ctx->tsg_id, GPU_TSG_PREEMPT_TRIGGER_REG);     │
│                                                               │
│       // 4. 异步返回（bWait=FALSE）                         │
│       return 0;  // 不等待 GPU 完成                         │
│   }                                                           │
│ }                                                             │
│                                                               │
└───────────────────────────────┬──────────────────────────────┘
                                │ MMIO write (控制寄存器)
                                ↓
┌──────────────────────────────────────────────────────────────┐
│ GPU Hardware - 抢占执行                                       │
│   • 检测 PREEMPT_TRIGGER 寄存器                               │
│   • 执行 Thread Block Preemption                             │
│   • 保存状态 (10-100μs)                                      │
│   • Context 切换                                              │
└──────────────────────────────────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    这是第二条路径（控制平面）⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🔍 两条路径的详细对比

### 路径 1: 应用任务提交（数据平面）

```
角色: 任何 CUDA 应用（训练、推理、计算）
目的: 向 GPU 提交计算任务
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// 应用代码
__global__ void my_kernel(float *data) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    data[tid] = data[tid] * 2.0f;
}

int main() {
    // 1. 准备数据
    float *d_data;
    cudaMalloc(&d_data, size);
    
    // 2. ⭐ 提交 kernel
    my_kernel<<<grid, block>>>(d_data);
    //   ↓
    // cuLaunchKernel() 展开后:
}

cuLaunchKernel() 内部实现（libcuda.so）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CUresult cuLaunchKernel(
    CUfunction f,
    unsigned int gridDimX, gridDimY, gridDimZ,
    unsigned int blockDimX, blockDimY, blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void **kernelParams,
    void **extra
) {
    // 1. 构造 GPU 命令 packet
    struct gpu_command_packet pkt;
    pkt.type = GPU_CMD_LAUNCH_KERNEL;
    pkt.kernel_addr = get_kernel_gpu_addr(f);
    pkt.grid = {gridDimX, gridDimY, gridDimZ};
    pkt.block = {blockDimX, blockDimY, blockDimZ};
    pkt.params_addr = upload_params(kernelParams);
    pkt.shared_mem = sharedMemBytes;
    
    // 2. 获取当前 Context 的 Pushbuffer
    CUcontext ctx = get_current_context();
    struct pushbuffer *pb = ctx->pushbuffer;
    
    // 3. ⭐⭐⭐ 写入 Pushbuffer（用户态内存）
    uint32_t put = pb->put_ptr;  // 当前写位置
    pb->buffer[put] = pkt;       // 写入命令
    put = (put + 1) % pb->size;  // 更新位置
    
    // 4. ⭐⭐⭐ 更新 PUT pointer（MMIO write）
    //    这是关键的"doorbell"操作！
    *pb->put_mmio = put;  // 写到 GPU 的 MMIO 地址
    //  ~~~~~~~~~~~~~~~~
    //  这是一个 MMIO 写操作
    //  地址: 0xF000_0000 + ctx_id * 0x1000
    //  延迟: ~100ns
    //  完全在用户态！不经过内核！✅
    
    pb->put_ptr = put;
    
    // 5. 立即返回
    return CUDA_SUCCESS;  // ~200ns 总延迟
}

GPU 接收（PFIFO Engine）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// 硬件状态机（伪代码）
while (1) {
    // 监控所有 Context 的 PUT pointer（MMIO 地址）
    for (ctx in all_contexts) {
        uint32_t put_hw = read_mmio(ctx->put_mmio_addr);
        uint32_t get_hw = ctx->get_ptr;  // 当前读位置
        
        if (put_hw != get_hw) {
            // 有新命令！
            struct gpu_command_packet pkt = dma_read(
                ctx->pushbuffer_gpu_addr + get_hw * sizeof(pkt)
            );
            
            // 处理命令
            if (pkt.type == GPU_CMD_LAUNCH_KERNEL) {
                schedule_kernel(pkt);  // 调度 kernel
            }
            
            // 更新 GET pointer
            ctx->get_ptr = (get_hw + 1) % ctx->pb_size;
        }
    }
}

关键点:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 完全在用户态（libcuda.so）
✅ Pushbuffer 是用户态内存（可直接访问）
✅ PUT pointer 是 MMIO（类似 AMD doorbell）
✅ 延迟 ~100-200ns
✅ 不需要 ioctl！
✅ 不需要系统调用！
✅ 不经过内核驱动！
```

### 路径 2: GPreempt 抢占控制（控制平面）⭐⭐⭐

```
角色: GPreempt 用户态调度器（独立进程）
目的: 监控优先级，触发抢占
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// GPreempt 调度器进程
// 这是一个独立的守护进程，不是应用！

int main() {
    int fd = open("/dev/nvidiactl", O_RDWR);
    if (fd < 0) {
        perror("Cannot open nvidiactl");
        return 1;
    }
    
    // 主循环：监控和抢占
    while (1) {
        // ⭐ 步骤 1: 查询所有 Context 的状态
        std::vector<NvContext> contexts = query_all_contexts(fd);
        
        // ⭐ 步骤 2: 检测优先级倒置
        NvContext high_ctx, low_ctx;
        if (detect_priority_inversion(contexts, &high_ctx, &low_ctx)) {
            printf("Priority inversion detected!\n");
            printf("  High priority context %p is waiting\n", high_ctx.hObject);
            printf("  Low priority context %p is running\n", low_ctx.hObject);
            
            // ⭐⭐⭐ 步骤 3: 触发抢占（关键！）
            int r = trigger_preemption(fd, low_ctx);
            if (r < 0) {
                printf("Preemption failed: %d\n", r);
            } else {
                printf("Preemption triggered successfully\n");
            }
        }
        
        // 休眠 1ms（监控周期）
        usleep(1000);
    }
    
    close(fd);
    return 0;
}

// ⭐ 查询所有 Context（需要 ioctl）
std::vector<NvContext> query_all_contexts(int fd) {
    std::vector<NvContext> contexts;
    
    // 遍历所有进程的 CUDA Context
    // 这需要特殊的权限和驱动支持
    for (int pid : get_cuda_processes()) {
        NvContext ctx;
        ctx.hClient = get_client_handle(pid);
        
        // ⭐ ioctl: 查询 Context 信息
        int r = ioctl(fd, NV_ESC_RM_QUERY_GROUP, &ctx);
        if (r == 0) {
            // 成功查询到 Context 信息
            // ctx.hObject, ctx.channels 等都已填充
            contexts.push_back(ctx);
        }
    }
    
    return contexts;
}

// ⭐⭐⭐ 触发抢占（关键函数！）
int trigger_preemption(int fd, NvContext low_ctx) {
    // 构造 ioctl 参数
    NVOS54_PARAMETERS controlArgs;
    controlArgs.hClient = low_ctx.hClient;
    controlArgs.hObject = low_ctx.hObject;
    controlArgs.cmd = NVA06C_CTRL_CMD_PREEMPT;  // 0xa06c0105
    
    // 抢占参数
    NVA06C_CTRL_PREEMPT_PARAMS preemptParams;
    preemptParams.bWait = NV_FALSE;          // 异步
    preemptParams.bManualTimeout = NV_FALSE;
    preemptParams.timeoutUs = 0;
    
    controlArgs.params = (NvP64)&preemptParams;
    controlArgs.paramsSize = sizeof(preemptParams);
    controlArgs.flags = 0;
    controlArgs.status = 0;
    
    // ⭐⭐⭐ ioctl: 触发抢占
    int r = ioctl(fd, OP_CONTROL, &controlArgs);
    //        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //        这是系统调用！
    //        进入内核态
    //        延迟: 1-10μs
    
    if (r < 0) {
        return -errno;
    }
    
    // 检查驱动返回的状态
    if (controlArgs.status != NV_OK) {
        return -EIO;
    }
    
    return 0;
}

内核驱动处理（nvidia.ko）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// drivers/gpu/nvidia/nvidia-drm/nvidia-drm-ioctl.c

long nvidia_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
    switch (cmd) {
        case OP_CONTROL: {
            NVOS54_PARAMETERS args;
            copy_from_user(&args, (void *)arg, sizeof(args));
            
            switch (args.cmd) {
                case NVA06C_CTRL_CMD_PREEMPT: {
                    // ⭐ 处理抢占命令
                    return handle_preempt_command(&args);
                }
                
                case NV_ESC_RM_QUERY_GROUP: {
                    // 处理查询命令
                    return handle_query_command(&args);
                }
            }
            break;
        }
    }
    
    return -EINVAL;
}

int handle_preempt_command(NVOS54_PARAMETERS *args) {
    // 1. 查找目标 Context
    struct nv_context *ctx = find_context(
        args->hClient,
        args->hObject
    );
    
    if (!ctx) {
        return -ENOENT;
    }
    
    // 2. 解析抢占参数
    NVA06C_CTRL_PREEMPT_PARAMS params;
    copy_from_user(&params, args->params, args->paramsSize);
    
    // 3. 标记为待抢占
    ctx->preemption_pending = true;
    
    // 4. ⭐⭐⭐ 向 GPU 发送抢占命令
    //    写入特殊的控制寄存器
    writel(ctx->tsg_id, dev->mmio_base + TSG_PREEMPT_TRIGGER_REG);
    //  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //  这是驱动写 GPU 的控制寄存器
    //  不是应用的 Pushbuffer！
    //  这是一个特殊的硬件接口
    
    // 5. 异步返回（bWait=FALSE）
    if (!params.bWait) {
        return 0;  // 立即返回，不等待 GPU 完成
    }
    
    // 如果 bWait=TRUE，轮询等待
    int timeout = params.timeoutUs;
    while (timeout > 0) {
        if (readl(dev->mmio_base + ctx->status_reg) & STATUS_PREEMPTED) {
            break;  // 抢占完成
        }
        udelay(10);
        timeout -= 10;
    }
    
    return 0;
}

GPU 接收抢占命令:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// GPU Host Interface (硬件状态机)
while (1) {
    // 监控控制寄存器（不是 Pushbuffer！）
    uint32_t preempt_trigger = read_reg(TSG_PREEMPT_TRIGGER_REG);
    
    if (preempt_trigger != 0) {
        uint32_t tsg_id = preempt_trigger;
        
        // 找到对应的 TSG (Thread State Group)
        struct tsg *target = find_tsg(tsg_id);
        
        // ⭐ 执行抢占
        execute_thread_block_preemption(target);
        //   • 等待 Thread Block 边界
        //   • 保存状态到 GPU 内存
        //   • 释放 SM 资源
        //   • 延迟: 10-100μs
        
        // 清除触发寄存器
        write_reg(TSG_PREEMPT_TRIGGER_REG, 0);
        
        // 更新状态
        write_reg(target->status_reg, STATUS_PREEMPTED);
        
        // 可能触发中断通知驱动
        if (target->interrupt_enabled) {
            trigger_interrupt(IRQ_PREEMPT_COMPLETE);
        }
    }
}

关键点:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 这是独立的 GPreempt 调度器进程（不是应用）
✅ 需要 ioctl 系统调用（进入内核态）
✅ 延迟 ~1-10μs（比 Pushbuffer 慢 10-100 倍）
⚠️ 但这不在应用的关键路径上！
✅ 应用的任务提交仍然很快（Pushbuffer）
✅ 只是抢占控制需要 ioctl
```

---

## 🎯 关键对比总结

### 为什么 GPreempt 需要 ioctl？

```
正确理解:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

问题: 应用使用 Pushbuffer 提交（~100ns，无 ioctl），
      为什么 GPreempt 还需要 ioctl？

答案: 因为它们是两个不同的角色和路径！
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

角色 1: 应用进程
  • 目的: 提交计算任务
  • 方法: cuLaunchKernel() → Pushbuffer + MMIO
  • 延迟: ~100ns
  • 路径: 用户态 → GPU（直接）
  • ioctl: ❌ 不需要

角色 2: GPreempt 调度器进程（独立！）
  • 目的: 监控优先级，触发抢占
  • 方法: ioctl → 驱动 → GPU 控制寄存器
  • 延迟: ~1-10μs
  • 路径: 用户态 → 内核态 → GPU
  • ioctl: ✅ 必须（需要特权操作）

为什么 GPreempt 不能用 Pushbuffer？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 权限问题:
   • Pushbuffer 是 per-context 的
   • GPreempt 需要访问其他进程的 Context
   • 需要特权操作（root/CAP_SYS_ADMIN）

2. 不同的硬件接口:
   • Pushbuffer: 提交任务（数据命令）
   • 控制寄存器: 抢占控制（控制命令）
   • 这是两个不同的硬件接口！

3. 需要驱动协助:
   • 查找 Context（需要驱动维护的数据结构）
   • 验证权限
   • 管理状态
   • 只有驱动能做这些

结论:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Pushbuffer 用于快速的"数据平面"（应用提交任务）✅
ioctl 用于可靠的"控制平面"（调度器管理）✅

这是经典的"数据/控制分离"设计！
```

### 完整对比表

| 维度 | 路径 1: 应用任务提交 | 路径 2: GPreempt 抢占控制 |
|------|---------------------|--------------------------|
| **角色** | 任何 CUDA 应用 | GPreempt 调度器（独立进程）|
| **目的** | 提交计算任务 | 监控优先级、触发抢占 |
| **API** | cuLaunchKernel() | NvRmPreempt() |
| **底层** | libcuda.so | ioctl() 系统调用 |
| **路径** | Pushbuffer + MMIO | ioctl → 驱动 → 控制寄存器 |
| **延迟** | ~100ns ✅ | ~1-10μs ⚠️ |
| **系统调用** | ❌ 无 | ✅ 有 |
| **进入内核** | ❌ 否 | ✅ 是 |
| **频率** | 高（每个 kernel）| 低（检测到倒置时）|
| **关键路径** | ✅ 是（性能关键）| ❌ 否（控制操作）|
| **权限** | 普通用户 | 需要特权 |
| **硬件接口** | Pushbuffer MMIO | 控制寄存器 MMIO |

---

## 📝 AMD 的对应架构

### AMD GPREEMPT 的优势

```
AMD vs NVIDIA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NVIDIA GPreempt (用户态调度器):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

路径 1 (应用):
  Application → cuLaunchKernel() → Pushbuffer + MMIO
  延迟: ~100ns ✅

路径 2 (调度器): ⚠️
  GPreempt Scheduler → ioctl → nvidia.ko → GPU
  延迟: ~1-10μs（系统调用开销）

为什么用户态调度器？
  ❌ 驱动闭源，无法在内核态实现
  ❌ 必须通过 ioctl 触发抢占


AMD GPREEMPT (内核态调度器): ✅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

路径 1 (应用):
  Application → hipLaunchKernel() → Doorbell + MMIO
  延迟: ~100ns ✅ (与 NVIDIA 相同)

路径 2 (调度器): ✅✅✅
  KFD Monitor Thread → 直接调用 kfd_queue_preempt() → GPU
  延迟: <1μs（无系统调用开销！）

为什么更快？
  ✅ 驱动开源，可以在内核态实现
  ✅ 内核线程直接调用函数，无 ioctl
  ✅ 省去 1-10μs 的系统调用开销


详细对比:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                        NVIDIA              AMD
应用提交（路径1）:      ~100ns              ~100ns         ✅ 相同
调度器监控位置:         用户态              内核态         ✅ AMD 优
调度器触发抢占:         ioctl (1-10μs)     直接调用 (<1μs) ✅ AMD 快
硬件抢占:               10-100μs            1-10μs         ✅ AMD 快10倍
驱动开源:               ❌ 闭源            ✅ 开源         ✅ AMD 优
部署便利:               ⚠️ 需补丁          ✅ DKMS        ✅ AMD 优

结论:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• 应用提交性能相同（都是 ~100ns）✅
• AMD 的调度器控制更快（内核态，无 ioctl）✅
• AMD 的硬件抢占更快（CWSR 1-10μs）✅
• AMD 开源驱动更易部署 ✅

AMD GPREEMPT 在所有维度上都优于 NVIDIA GPreempt！
```

---

## 🎯 核心总结

### 关键洞察

1. **双路径设计**：
   - 数据平面（应用提交）：Pushbuffer/Doorbell，~100ns
   - 控制平面（调度器）：ioctl（NVIDIA）或直接调用（AMD）

2. **为什么 NVIDIA 需要 ioctl**：
   - ioctl 是给 **GPreempt 调度器进程** 用的
   - 不是给**应用进程**用的
   - 应用仍然用 Pushbuffer（快速）

3. **AMD 的优势**：
   - 应用提交同样快（Doorbell）
   - 但调度器在内核态，无 ioctl 开销
   - 开源驱动，易修改和部署

4. **数据/控制分离**：
   - 这是经典的网络架构设计
   - 数据平面要快（Pushbuffer/Doorbell）
   - 控制平面要可靠（ioctl/内核函数）

---

**文档版本**: v1.0  
**创建日期**: 2026-01-28  
**核心修正**: 明确区分应用提交（Pushbuffer）和调度器控制（ioctl）

**关键结论**:
- Pushbuffer 和 ioctl 是两条不同的路径
- 应用用 Pushbuffer（快速）
- GPreempt 调度器用 ioctl（控制）
- AMD 在内核态实现调度器，更优


