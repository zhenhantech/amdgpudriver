# suspend_queues() API 使用指南（POC专用）

**日期**: 2026-02-04  
**目的**: 说明POC中如何使用suspend_queues/resume_queues实现抢占

---

## 📌 核心答案

### Q1: POC是不是只调用suspend_queues就可以了？

**A: 不完全是，需要4个步骤** ⭐⭐⭐:

```
1. 启用debug trap (一次性)
2. suspend_queues() - 暂停Offline队列 ← ⭐ 这是核心
3. [Online-AI执行]
4. resume_queues() - 恢复Offline队列
```

### Q2: 传入参数是什么？需要指定哪个queue吗？

**A: 需要传入queue_id数组** ⭐⭐⭐:

```c
// 内核函数签名
int suspend_queues(
    struct kfd_process *p,           // 目标进程
    uint32_t num_queues,             // 要暂停的队列数量
    uint32_t grace_period,           // 宽限期(GPU clock cycles)
    uint64_t exception_clear_mask,   // 异常清除mask
    uint32_t *usr_queue_id_array     // 队列ID数组 ⭐
);

// 用户态ioctl参数
struct kfd_ioctl_dbg_trap_suspend_queues_args {
    __u64 exception_mask;      // 异常清除mask
    __u64 queue_array_ptr;     // 指向queue_id数组的指针 ⭐
    __u32 num_queues;          // 数组中的队列数量
    __u32 grace_period;        // 宽限期
};
```

**关键**: 必须指定具体的queue_id！

---

## 🔍 API详细说明

### 1. 内核函数签名

**定义位置**: `/usr/src/amdgpu-6.12.12-2194681.el8_preempt/amd/amdkfd/kfd_device_queue_manager.h:316`

```316:323:usr/src/amdgpu-6.12.12-2194681.el8_preempt/amd/amdkfd/kfd_device_queue_manager.h
int suspend_queues(struct kfd_process *p,
			uint32_t num_queues,
			uint32_t grace_period,
			uint64_t exception_clear_mask,
			uint32_t *usr_queue_id_array);
int resume_queues(struct kfd_process *p,
		uint32_t num_queues,
		uint32_t *usr_queue_id_array);
```

### 2. 用户态ioctl参数

**定义位置**: `/usr/src/amdgpu-6.12.12-2194681.el8_preempt/include/uapi/linux/kfd_ioctl.h:1421`

```1421:1426:usr/src/amdgpu-6.12.12-2194681.el8_preempt/include/uapi/linux/kfd_ioctl.h
struct kfd_ioctl_dbg_trap_suspend_queues_args {
	__u64 exception_mask;
	__u64 queue_array_ptr;
	__u32 num_queues;
	__u32 grace_period;
};
```

### 3. 参数说明

| 参数 | 类型 | 说明 | POC建议值 |
|------|------|------|-----------|
| **queue_array_ptr** | uint64_t | 指向`uint32_t`数组的指针，包含要暂停的queue_id | 从Offline进程获取 ⭐⭐⭐ |
| **num_queues** | uint32_t | 数组中的队列数量 | Offline进程的队列数（如10个） |
| **grace_period** | uint32_t | 宽限期（单位：1K GPU clock cycles）| `0` (立即抢占) 或 `100` (允许100K cycles) |
| **exception_mask** | uint64_t | 异常清除mask | `0` (POC不需要清除异常) |

---

## 📋 完整使用流程

### Step 1: 启用Debug Trap（一次性操作）⭐⭐⭐

**重要**: `suspend_queues`是调试API，必须先启用debug trap！

```c
// 打开KFD设备
int kfd_fd = open("/dev/kfd", O_RDWR);

// 准备debug trap参数
struct kfd_ioctl_dbg_trap_args trap_args = {
    .op = KFD_IOC_DBG_TRAP_ENABLE,  // 启用debug trap
    .pid = target_pid,               // 要调试的进程PID（Offline-AI进程）
    .enable = {
        .dbg_fd = kfd_fd,           // 调试器的fd
        .rinfo_ptr = 0,             // runtime info指针（可选）
        .rinfo_size = 0,            // runtime info大小
        .exception_mask = 0         // 异常mask（POC不需要）
    }
};

// 调用ioctl启用
int ret = ioctl(kfd_fd, AMDKFD_IOC_DBG_TRAP, &trap_args);
if (ret != 0) {
    perror("Failed to enable debug trap");
    exit(1);
}
```

**注意**: 
- 这个操作在POC开始时做一次即可
- 需要**root权限**或**ptrace权限**
- 目标进程必须是你自己启动的，或者通过`ptrace`附加

---

### Step 2: 获取Offline队列的Queue ID ⭐⭐⭐

**方法1: 通过`/sys/kernel/debug/kfd/process`**（推荐）

```bash
# 查找目标进程的队列
sudo cat /sys/kernel/debug/kfd/process | grep -A 20 "PID $OFFLINE_PID"

# 输出示例:
# Process 12345:
#   Queue 0 (active):
#     queue id: 123
#   Queue 1 (active):
#     queue id: 124
#   ...
```

**方法2: 从程序内部获取**（如果修改Offline-AI代码）

```cpp
// 在Offline-AI程序中记录queue_id
// 创建队列时，ioctl返回queue_id
struct kfd_ioctl_create_queue_args args = {...};
ioctl(kfd_fd, AMDKFD_IOC_CREATE_QUEUE, &args);
uint32_t queue_id = args.queue_id;  // ← 记录这个ID

// 写入文件供POC读取
FILE* fp = fopen("/tmp/offline_queue_ids.txt", "w");
fprintf(fp, "%u\n", queue_id);
fclose(fp);
```

**方法3: 解析HIP Runtime的Queue对象**（复杂，不推荐）

```cpp
// 需要hack HIP Runtime内部结构，不稳定
```

---

### Step 3: 调用suspend_queues ⭐⭐⭐⭐⭐

```c
// 假设我们获取到了Offline进程的10个queue_id
uint32_t offline_queue_ids[] = {123, 124, 125, 126, 127, 
                                 128, 129, 130, 131, 132};
uint32_t num_queues = 10;

// 准备参数
struct kfd_ioctl_dbg_trap_args trap_args = {
    .op = KFD_IOC_DBG_TRAP_SUSPEND_QUEUES,  // ← suspend操作
    .pid = offline_pid,                      // Offline进程的PID
    .suspend_queues = {
        .exception_mask = 0,                 // POC不需要
        .queue_array_ptr = (uint64_t)offline_queue_ids,  // ⭐ 队列ID数组
        .num_queues = num_queues,            // ⭐ 队列数量
        .grace_period = 0                    // ⭐ 立即抢占
    }
};

// 调用ioctl暂停队列
int ret = ioctl(kfd_fd, AMDKFD_IOC_DBG_TRAP, &trap_args);
if (ret < 0) {
    perror("Failed to suspend queues");
    exit(1);
}

printf("Successfully suspended %d queues\n", ret);
// ret返回实际暂停的队列数量
```

**关键参数**:
- `queue_array_ptr`: 指向包含queue_id的数组 ⭐⭐⭐
- `num_queues`: 数组大小
- `grace_period = 0`: 立即抢占（POC推荐）

---

### Step 4: Online-AI执行

```cpp
// 此时Offline队列已被暂停（unmap）
// Online-AI可以使用GPU资源

// 启动Online-AI任务
hipLaunchKernel<<<...>>>(online_kernel, ...);
hipStreamSynchronize(online_stream);

// 测量延迟
auto start = std::chrono::high_resolution_clock::now();
// ... Online-AI执行 ...
auto end = std::chrono::high_resolution_clock::now();
auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

printf("Online-AI latency: %ld ms\n", latency.count());
```

---

### Step 5: 恢复Offline队列

```c
// 准备resume参数（不需要grace_period和exception_mask）
struct kfd_ioctl_dbg_trap_args trap_args = {
    .op = KFD_IOC_DBG_TRAP_RESUME_QUEUES,   // ← resume操作
    .pid = offline_pid,
    .resume_queues = {
        .queue_array_ptr = (uint64_t)offline_queue_ids,  // ⭐ 同一批队列
        .num_queues = num_queues                         // ⭐ 同样数量
    }
};

// 调用ioctl恢复队列
int ret = ioctl(kfd_fd, AMDKFD_IOC_DBG_TRAP, &trap_args);
if (ret < 0) {
    perror("Failed to resume queues");
    exit(1);
}

printf("Successfully resumed %d queues\n", ret);
```

**注意**: resume时必须传入**相同的queue_id数组**！

---

## 🔑 POC关键问题解答

### Q1: 为什么不能suspend所有队列？

**A: 可以，但需要知道所有queue_id**

```c
// 如果想suspend进程的所有队列
// 需要遍历/sys/kernel/debug/kfd/process获取所有queue_id

// 示例：suspend进程的所有10个队列
uint32_t all_queue_ids[10];
// ... 从debugfs获取 ...
suspend_queues(p, 10, 0, 0, all_queue_ids);
```

### Q2: grace_period该设置多少？

**POC建议**: `0` (立即抢占)

```c
.grace_period = 0  // 立即触发CWSR保存，最快抢占
```

**如果需要"温和"抢占**:
```c
.grace_period = 100  // 允许100K GPU cycles完成当前Wave
                     // 约= 100,000 / 3,800,000,000 = 0.026ms
```

**计算公式**:
```
实际宽限时间(ms) = grace_period * 1000 / GPU频率(MHz)
                 = grace_period * 1000 / 3800 (MI308X)
```

### Q3: 如何知道suspend成功？

**方法1: 检查返回值**
```c
int num_suspended = ioctl(kfd_fd, AMDKFD_IOC_DBG_TRAP, &trap_args);
if (num_suspended == num_queues) {
    printf("All %d queues suspended successfully\n", num_queues);
} else {
    printf("Warning: only %d/%d queues suspended\n", num_suspended, num_queues);
}
```

**方法2: 检查queue_array中的错误标志**
```c
// ioctl会修改queue_array，标记失败的队列
for (int i = 0; i < num_queues; i++) {
    if (offline_queue_ids[i] & KFD_DBG_QUEUE_ERROR_MASK) {
        printf("Queue %d: Hardware error\n", offline_queue_ids[i] & ~KFD_DBG_QUEUE_ERROR_MASK);
    }
    if (offline_queue_ids[i] & KFD_DBG_QUEUE_INVALID_MASK) {
        printf("Queue %d: Invalid (destroyed or new)\n", offline_queue_ids[i] & ~KFD_DBG_QUEUE_INVALID_MASK);
    }
}
```

**方法3: 查看MQD状态**
```bash
# suspend后，MQD的is_active标志应该变为false
sudo cat /sys/kernel/debug/kfd/mqds | grep "queue_id: 123" -A 5
# 输出应该显示 "active: 0"
```

### Q4: 需要root权限吗？

**A: 是的** ⭐⭐⭐

```bash
# 方法1: 以root运行POC程序
sudo ./poc_preemption

# 方法2: 添加CAP_SYS_PTRACE权限
sudo setcap cap_sys_ptrace=eip ./poc_preemption
./poc_preemption

# 方法3: 使用ptrace附加（需要同用户或root）
# 在POC程序中：
ptrace(PTRACE_ATTACH, offline_pid, NULL, NULL);
// ... 然后可以使用debug trap API ...
```

### Q5: suspend_queues会不会丢失数据？

**A: 不会！CWSR机制保证数据安全** ⭐⭐⭐

```
1. suspend_queues → unmap队列
2. HWS触发CWSR保存 ⭐⭐⭐
   - 所有Wave状态保存到MQD的ctx_save_restore_area
   - Ring-buffer中未处理的命令保持不变
   - Read/Write指针保存在MQD
3. resume_queues → map队列
4. HWS触发CWSR恢复 ⭐⭐⭐
   - 从ctx_save_restore_area恢复Wave状态
   - 从Read指针位置继续读取命令
5. 继续执行，就像没有中断过一样
```

**验证方法**:
```bash
# 检查CWSR区域是否有数据
sudo cat /sys/kernel/debug/kfd/mqds | grep ctx_save -A 2
# 输出应该显示非零的ctx_save_base_addr
```

---

## 📊 POC完整示例代码

```c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/kfd_ioctl.h>  // 需要包含KFD ioctl头文件

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <offline_pid>\n", argv[0]);
        return 1;
    }
    
    pid_t offline_pid = atoi(argv[1]);
    
    // ===== Step 1: 打开KFD设备 =====
    int kfd_fd = open("/dev/kfd", O_RDWR);
    if (kfd_fd < 0) {
        perror("Failed to open /dev/kfd");
        return 1;
    }
    
    // ===== Step 2: 启用Debug Trap =====
    printf("Enabling debug trap for PID %d...\n", offline_pid);
    struct kfd_ioctl_dbg_trap_args enable_trap = {
        .op = KFD_IOC_DBG_TRAP_ENABLE,
        .pid = offline_pid,
        .enable = {
            .dbg_fd = kfd_fd,
            .rinfo_ptr = 0,
            .rinfo_size = 0,
            .exception_mask = 0
        }
    };
    
    if (ioctl(kfd_fd, AMDKFD_IOC_DBG_TRAP, &enable_trap) != 0) {
        perror("Failed to enable debug trap");
        close(kfd_fd);
        return 1;
    }
    printf("Debug trap enabled successfully\n");
    
    // ===== Step 3: 获取Offline队列ID =====
    // TODO: 从debugfs或程序协商方式获取queue_id
    // 这里假设我们已经知道了queue_id
    uint32_t offline_queue_ids[] = {10, 11, 12, 13, 14, 
                                     15, 16, 17, 18, 19};  // 示例ID
    uint32_t num_queues = 10;
    
    printf("Target queues: ");
    for (int i = 0; i < num_queues; i++) {
        printf("%u ", offline_queue_ids[i]);
    }
    printf("\n");
    
    // ===== Step 4: Suspend Offline队列 ⭐⭐⭐ =====
    printf("Suspending %d queues...\n", num_queues);
    struct kfd_ioctl_dbg_trap_args suspend_trap = {
        .op = KFD_IOC_DBG_TRAP_SUSPEND_QUEUES,
        .pid = offline_pid,
        .suspend_queues = {
            .exception_mask = 0,
            .queue_array_ptr = (uint64_t)offline_queue_ids,
            .num_queues = num_queues,
            .grace_period = 0  // 立即抢占
        }
    };
    
    int num_suspended = ioctl(kfd_fd, AMDKFD_IOC_DBG_TRAP, &suspend_trap);
    if (num_suspended < 0) {
        perror("Failed to suspend queues");
        close(kfd_fd);
        return 1;
    }
    printf("Successfully suspended %d/%d queues\n", num_suspended, num_queues);
    
    // ===== Step 5: Online-AI执行（这里用sleep模拟）=====
    printf("Online-AI executing...\n");
    sleep(2);  // 实际POC中这里是hipLaunchKernel等
    printf("Online-AI completed\n");
    
    // ===== Step 6: Resume Offline队列 =====
    printf("Resuming %d queues...\n", num_queues);
    struct kfd_ioctl_dbg_trap_args resume_trap = {
        .op = KFD_IOC_DBG_TRAP_RESUME_QUEUES,
        .pid = offline_pid,
        .resume_queues = {
            .queue_array_ptr = (uint64_t)offline_queue_ids,
            .num_queues = num_queues
        }
    };
    
    int num_resumed = ioctl(kfd_fd, AMDKFD_IOC_DBG_TRAP, &resume_trap);
    if (num_resumed < 0) {
        perror("Failed to resume queues");
        close(kfd_fd);
        return 1;
    }
    printf("Successfully resumed %d/%d queues\n", num_resumed, num_queues);
    
    // ===== Step 7: 清理 =====
    close(kfd_fd);
    printf("POC completed successfully\n");
    
    return 0;
}
```

**编译**:
```bash
gcc -o poc_preemption poc_preemption.c -I/usr/src/amdgpu-6.12.12-2194681.el8_preempt/include/uapi
```

**运行**:
```bash
# 启动Offline-AI程序（记录PID）
python offline_training.py &
OFFLINE_PID=$!

# 运行POC（需要root）
sudo ./poc_preemption $OFFLINE_PID
```

---

## ⚠️ 重要限制和注意事项

### 1. 必须启用Debug Trap ⭐⭐⭐

```c
// suspend_queues是调试API，必须先enable debug trap
if (args->op != KFD_IOC_DBG_TRAP_ENABLE && !target->debug_trap_enabled) {
    pr_err("PID %i not debug enabled for op %i\n", args->pid, args->op);
    return -EINVAL;  // ← 如果没有enable，会返回错误
}
```

### 2. 需要权限

- **Root权限** 或
- **CAP_SYS_PTRACE capability** 或
- **ptrace附加**到目标进程

### 3. 队列销毁被阻塞

```
suspend_queues会阻塞队列销毁，直到resume
→ Offline进程无法正常退出
→ POC必须确保resume被调用
```

**解决方案**: 使用信号处理器
```c
void cleanup_handler(int signum) {
    printf("Caught signal %d, resuming queues...\n", signum);
    // resume queues
    exit(0);
}

signal(SIGINT, cleanup_handler);
signal(SIGTERM, cleanup_handler);
```

### 4. Runtime状态要求

```c
// 目标进程必须在DEBUG_RUNTIME_STATE_ENABLED状态
if (target->runtime_info.runtime_state != DEBUG_RUNTIME_STATE_ENABLED) {
    return -EPERM;  // ← 会返回权限错误
}
```

这个状态在`KFD_IOC_DBG_TRAP_ENABLE`时自动设置。

---

## 📚 相关文档

- `AQL_QUEUE_VS_MQD_RELATIONSHIP.md` - MQD与抢占的关系
- `QUEUE_CREATION_TIMELINE.md` - 队列创建时CWSR区域的分配
- `New_DEEP_DIVE_MI308X_QUEUE_MECHANISMS.md` - CWSR保存/恢复机制
- `ARCH_Design_01_POC_Stage1_实施方案.md` - 完整POC方案

---

## 🎯 总结：POC只需要做什么？

### 最小化POC流程 ⭐⭐⭐⭐⭐

```
1. 获取Offline进程的queue_id（从debugfs或协商）
2. 启用debug trap（一次）
3. 循环:
   a. suspend_queues(offline_queue_ids)
   b. [Online-AI执行]
   c. resume_queues(offline_queue_ids)
4. 测量Online-AI的延迟

就这么简单！ ✨
```

### 不需要做什么 ❌

```
❌ 不需要手动unmap/map队列（suspend/resume自动做）
❌ 不需要管理CWSR区域（内核自动保存/恢复）
❌ 不需要修改MQD（内核自动更新）
❌ 不需要发送runlist（内核自动重建）
❌ 不需要重新创建队列（suspend/resume保持队列存活）
```

---

**最后更新**: 2026-02-04  
**验证状态**: ✅ 基于内核代码和ioctl定义  
**适用平台**: MI308X (CPSCH模式)
