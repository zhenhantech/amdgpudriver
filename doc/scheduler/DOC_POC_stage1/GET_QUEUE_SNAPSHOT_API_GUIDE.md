# GET_QUEUE_SNAPSHOT API使用指南

**日期**: 2026-02-04  
**目的**: 使用ioctl直接获取Queue ID和MQD信息，避免cat sysfs的问题

---

## 📌 核心答案

### Q: 有从代码中直接获取MQD对应的queue ID信息的ioctl吗？

**A: 有！`KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT`** ⭐⭐⭐⭐⭐

这个API可以直接从内核获取所有queue的详细信息，包括：
- queue_id
- gpu_id  
- ring_base_address（ring-buffer地址）
- write/read_pointer_address
- ctx_save_restore_address（CWSR区域地址）
- queue_type
- ring_size
- exception_status

**比cat sysfs更好**:
- ✅ 编程友好：直接C结构体
- ✅ 原子操作：一次调用获取所有信息
- ✅ 无需root：只需debug trap权限
- ✅ 稳定：不依赖debugfs格式
- ✅ 完整：包含所有MQD关键字段

---

## 🔍 API定义

### 结构体定义

**位置**: `/usr/src/amdgpu-6.12.12-2194681.el8_preempt/include/uapi/linux/kfd_ioctl.h:1198`

```c
/* Queue information */
struct kfd_queue_snapshot_entry {
    __u64 exception_status;           // 异常状态
    __u64 ring_base_address;          // Ring-buffer GPU地址 ⭐
    __u64 write_pointer_address;      // Write指针地址 ⭐
    __u64 read_pointer_address;       // Read指针地址 ⭐
    __u64 ctx_save_restore_address;   // CWSR保存区地址 ⭐⭐⭐
    __u32 queue_id;                   // 队列ID ⭐⭐⭐
    __u32 gpu_id;                     // GPU ID
    __u32 ring_size;                  // Ring大小
    __u32 queue_type;                 // 队列类型（AQL/PM4/SDMA）
    __u32 ctx_save_restore_area_size; // CWSR区域大小
    __u32 reserved;
};
```

### ioctl参数定义

**位置**: `/usr/src/amdgpu-6.12.12-2194681.el8_preempt/include/uapi/linux/kfd_ioctl.h:1601`

```c
struct kfd_ioctl_dbg_trap_queue_snapshot_args {
    __u64 exception_mask;        // (IN) 异常mask（通常为0）
    __u64 snapshot_buf_ptr;      // (IN) 指向snapshot数组的指针 ⭐
    __u32 num_queues;            // (IN/OUT) 输入=缓冲区大小，输出=实际队列数 ⭐
    __u32 entry_size;            // (IN/OUT) 每个entry的字节大小
};
```

**关键**:
- `num_queues` 是 **IN/OUT** 参数：
  - **IN**: 你分配的缓冲区能装多少个queue
  - **OUT**: 内核告诉你实际有多少个queue
  - 如果实际 > 缓冲区，不会溢出，但你需要再次调用用更大的缓冲区

---

## 📋 完整使用步骤

### Step 1: 启用Debug Trap（前提条件）

```c
int kfd_fd = open("/dev/kfd", O_RDWR);

// 必须先启用debug trap
struct kfd_ioctl_dbg_trap_args enable_args = {
    .op = KFD_IOC_DBG_TRAP_ENABLE,
    .pid = target_pid,  // 目标进程PID
    .enable = {
        .dbg_fd = kfd_fd,
        .rinfo_ptr = 0,
        .rinfo_size = 0,
        .exception_mask = 0
    }
};

ioctl(kfd_fd, AMDKFD_IOC_DBG_TRAP, &enable_args);
```

### Step 2: 分配Snapshot缓冲区

```c
// 假设最多有100个队列
#define MAX_QUEUES 100
struct kfd_queue_snapshot_entry snapshots[MAX_QUEUES];
```

### Step 3: 调用GET_QUEUE_SNAPSHOT ⭐⭐⭐

```c
struct kfd_ioctl_dbg_trap_args snapshot_args = {
    .op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT,
    .pid = target_pid,  // 目标进程PID
    .get_queue_snapshot = {
        .exception_mask = 0,
        .snapshot_buf_ptr = (uint64_t)snapshots,  // ⭐ 指向缓冲区
        .num_queues = MAX_QUEUES,                  // ⭐ 缓冲区大小
        .entry_size = sizeof(struct kfd_queue_snapshot_entry)
    }
};

int ret = ioctl(kfd_fd, AMDKFD_IOC_DBG_TRAP, &snapshot_args);
if (ret < 0) {
    perror("Failed to get queue snapshot");
    exit(1);
}

// 返回值是实际的队列数量
int num_queues = snapshot_args.get_queue_snapshot.num_queues;  // ⭐
printf("Found %d queues\n", num_queues);
```

### Step 4: 解析Snapshot数据

```c
printf("\nQueue Information:\n");
printf("%-8s %-6s %-18s %-10s %-18s\n", 
       "Queue ID", "GPU ID", "Ring Address", "Ring Size", "CWSR Address");
printf("--------------------------------------------------------------------\n");

for (int i = 0; i < num_queues; i++) {
    struct kfd_queue_snapshot_entry *entry = &snapshots[i];
    
    printf("%-8u %-6u 0x%016llx %-10u 0x%016llx\n",
           entry->queue_id,              // ⭐ Queue ID
           entry->gpu_id,
           entry->ring_base_address,     // Ring-buffer地址
           entry->ring_size,
           entry->ctx_save_restore_address);  // ⭐ CWSR区域
    
    // 队列类型
    const char* type_str;
    switch (entry->queue_type) {
        case 0: type_str = "COMPUTE"; break;
        case 1: type_str = "SDMA"; break;
        case 2: type_str = "AQL"; break;
        default: type_str = "UNKNOWN"; break;
    }
    printf("  Type: %s\n", type_str);
}
```

---

## 💡 完整示例程序

```c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/kfd_ioctl.h>

#define MAX_QUEUES 100

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <pid>\n", argv[0]);
        return 1;
    }
    
    pid_t target_pid = atoi(argv[1]);
    
    // ===== Step 1: 打开KFD设备 =====
    int kfd_fd = open("/dev/kfd", O_RDWR);
    if (kfd_fd < 0) {
        perror("Failed to open /dev/kfd");
        return 1;
    }
    
    // ===== Step 2: 启用Debug Trap =====
    printf("Enabling debug trap for PID %d...\n", target_pid);
    struct kfd_ioctl_dbg_trap_args enable_trap = {
        .op = KFD_IOC_DBG_TRAP_ENABLE,
        .pid = target_pid,
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
    printf("Debug trap enabled\n");
    
    // ===== Step 3: 分配snapshot缓冲区 =====
    struct kfd_queue_snapshot_entry *snapshots = 
        malloc(MAX_QUEUES * sizeof(struct kfd_queue_snapshot_entry));
    if (!snapshots) {
        perror("Failed to allocate snapshot buffer");
        close(kfd_fd);
        return 1;
    }
    
    // ===== Step 4: 获取Queue Snapshot ⭐⭐⭐ =====
    printf("\nGetting queue snapshot...\n");
    struct kfd_ioctl_dbg_trap_args snapshot_trap = {
        .op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT,
        .pid = target_pid,
        .queue_snapshot = {  // ⭐ 注意字段名
            .exception_mask = 0,
            .snapshot_buf_ptr = (uint64_t)snapshots,
            .num_queues = MAX_QUEUES,
            .entry_size = sizeof(struct kfd_queue_snapshot_entry)
        }
    };
    
    if (ioctl(kfd_fd, AMDKFD_IOC_DBG_TRAP, &snapshot_trap) != 0) {
        perror("Failed to get queue snapshot");
        free(snapshots);
        close(kfd_fd);
        return 1;
    }
    
    int num_queues = snapshot_trap.get_queue_snapshot.num_queues;
    printf("Found %d queues\n\n", num_queues);
    
    // ===== Step 5: 打印Queue信息 =====
    printf("Queue Snapshot:\n");
    printf("================================================================================\n");
    printf("%-8s %-6s %-18s %-10s %-10s %-18s\n",
           "QueueID", "GPU", "RingAddress", "RingSize", "Type", "CWSR Address");
    printf("================================================================================\n");
    
    for (int i = 0; i < num_queues; i++) {
        struct kfd_queue_snapshot_entry *entry = &snapshots[i];
        
        const char* type_str;
        switch (entry->queue_type) {
            case 0: type_str = "COMPUTE"; break;
            case 1: type_str = "SDMA"; break;
            case 2: type_str = "AQL"; break;
            case 3: type_str = "SDMA_XGMI"; break;
            default: type_str = "UNKNOWN"; break;
        }
        
        printf("%-8u %-6u 0x%016llx %-10u %-10s 0x%016llx\n",
               entry->queue_id,
               entry->gpu_id,
               entry->ring_base_address,
               entry->ring_size,
               type_str,
               entry->ctx_save_restore_address);
        
        // 详细信息（可选）
        if (entry->exception_status != 0) {
            printf("    Exception Status: 0x%llx\n", entry->exception_status);
        }
        printf("    Write Ptr: 0x%llx, Read Ptr: 0x%llx\n",
               entry->write_pointer_address,
               entry->read_pointer_address);
        printf("    CWSR Size: %u bytes\n", entry->ctx_save_restore_area_size);
        printf("\n");
    }
    
    // ===== Step 6: 提取Queue ID数组（用于suspend）=====
    printf("Queue IDs for suspend operation:\n");
    printf("uint32_t queue_ids[] = {");
    for (int i = 0; i < num_queues; i++) {
        printf("%u", snapshots[i].queue_id);
        if (i < num_queues - 1) printf(", ");
    }
    printf("};\n");
    printf("uint32_t num_queues = %d;\n", num_queues);
    
    // ===== Step 7: 清理 =====
    free(snapshots);
    close(kfd_fd);
    
    return 0;
}
```

### 编译和运行

```bash
# 编译
gcc -o get_queue_info get_queue_info.c \
    -I/usr/src/amdgpu-6.12.12-2194681.el8_preempt/include/uapi

# 运行（需要root或ptrace权限）
sudo ./get_queue_info 12345  # 12345是目标进程PID

# 输出示例:
# Found 10 queues
#
# Queue Snapshot:
# ================================================================================
# QueueID GPU    RingAddress        RingSize   Type       CWSR Address      
# ================================================================================
# 10      0      0x00007f1234000000 65536      AQL        0x00007f5678000000
#     Write Ptr: 0x00007f1234010000, Read Ptr: 0x00007f1234010008
#     CWSR Size: 2097152 bytes
#
# 11      0      0x00007f1234020000 65536      AQL        0x00007f5678200000
# ...
#
# Queue IDs for suspend operation:
# uint32_t queue_ids[] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
# uint32_t num_queues = 10;
```

---

## 🎯 POC集成：完整工作流

### 完整POC流程（使用GET_QUEUE_SNAPSHOT）

```c
int poc_preemption(pid_t offline_pid) {
    int kfd_fd = open("/dev/kfd", O_RDWR);
    
    // 1. 启用debug trap
    enable_debug_trap(kfd_fd, offline_pid);
    
    // 2. 获取Offline进程的所有queue_id ⭐⭐⭐
    struct kfd_queue_snapshot_entry snapshots[MAX_QUEUES];
    int num_queues = get_queue_snapshot(kfd_fd, offline_pid, snapshots, MAX_QUEUES);
    
    if (num_queues < 0) {
        fprintf(stderr, "Failed to get queue snapshot\n");
        return -1;
    }
    
    printf("Offline process has %d queues\n", num_queues);
    
    // 3. 提取queue_id数组
    uint32_t *queue_ids = malloc(num_queues * sizeof(uint32_t));
    for (int i = 0; i < num_queues; i++) {
        queue_ids[i] = snapshots[i].queue_id;
        printf("  Queue %d: ID=%u, GPU=%u, Type=%u\n",
               i, snapshots[i].queue_id, snapshots[i].gpu_id, snapshots[i].queue_type);
    }
    
    // 4. POC测试循环
    for (int iter = 0; iter < 100; iter++) {
        // a. Suspend offline queues
        suspend_queues(kfd_fd, offline_pid, queue_ids, num_queues);
        
        // b. Online-AI执行并测量延迟
        auto start = now();
        run_online_ai();
        auto latency = now() - start;
        printf("Iteration %d: Online-AI latency = %ld ms\n", iter, latency);
        
        // c. Resume offline queues
        resume_queues(kfd_fd, offline_pid, queue_ids, num_queues);
        
        sleep(1);  // 等待下一次迭代
    }
    
    free(queue_ids);
    close(kfd_fd);
    return 0;
}

// 辅助函数：获取queue snapshot
int get_queue_snapshot(int kfd_fd, pid_t pid,
                       struct kfd_queue_snapshot_entry *snapshots,
                       int max_queues) {
    struct kfd_ioctl_dbg_trap_args args = {
        .op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT,
        .pid = pid,
        .queue_snapshot = {  // ⭐ 字段名
            .exception_mask = 0,
            .snapshot_buf_ptr = (uint64_t)snapshots,
            .num_queues = max_queues,
            .entry_size = sizeof(struct kfd_queue_snapshot_entry)
        }
    };
    
    if (ioctl(kfd_fd, AMDKFD_IOC_DBG_TRAP, &args) != 0) {
        perror("get_queue_snapshot failed");
        return -1;
    }
    
    return args.queue_snapshot.num_queues;  // ⭐ 返回实际队列数
}
```

---

## 🔑 关键优势

### vs cat sysfs/debugfs

| 特性 | GET_QUEUE_SNAPSHOT | cat sysfs/debugfs |
|------|-------------------|-------------------|
| **编程友好** | ✅ C结构体，类型安全 | ❌ 文本解析，易出错 |
| **原子性** | ✅ 一次调用获取所有 | ❌ 多次读取，可能不一致 |
| **权限** | ✅ debug trap权限 | ❌ 需要root |
| **性能** | ✅ 直接内核调用 | ❌ 文件I/O开销 |
| **稳定性** | ✅ UAPI保证 | ❌ debugfs格式可能变化 |
| **完整性** | ✅ 所有MQD关键字段 | ⚠️ 部分信息 |
| **错误处理** | ✅ 明确的返回码 | ❌ 解析错误难发现 |

### 包含的关键信息

```c
struct kfd_queue_snapshot_entry {
    // POC最需要的信息 ⭐⭐⭐
    __u32 queue_id;                   // 用于suspend/resume
    __u32 gpu_id;                     // 区分不同GPU
    __u32 queue_type;                 // 区分COMPUTE/SDMA
    
    // MQD关键信息 ⭐⭐
    __u64 ring_base_address;          // Ring-buffer地址
    __u64 write_pointer_address;      // Write指针
    __u64 read_pointer_address;       // Read指针
    __u32 ring_size;                  // Ring大小
    
    // CWSR信息（抢占相关）⭐⭐⭐
    __u64 ctx_save_restore_address;   // Wave状态保存区
    __u32 ctx_save_restore_area_size; // 保存区大小
    
    // 异常状态
    __u64 exception_status;           // 用于调试
};
```

---

## ⚠️ 注意事项

### 1. 必须启用Debug Trap

```c
// GET_QUEUE_SNAPSHOT需要先enable debug trap
if (!debug_trap_enabled) {
    return -EINVAL;  // 会返回错误
}
```

### 2. 缓冲区大小

```c
// 如果队列数量超过缓冲区大小
struct kfd_ioctl_dbg_trap_get_queue_snapshot_args args = {
    .num_queues = 10  // 缓冲区只有10个slot
};
ioctl(...);  // 假设实际有15个队列

// 返回后：
args.num_queues = 15  // ⚠️ 告诉你实际有15个队列
// 但只会填充前10个到缓冲区

// 解决方案：第二次调用，增大缓冲区
```

### 3. 队列动态变化

```c
// 队列可能在两次调用之间被创建/销毁
int num1 = get_queue_snapshot(...);  // 返回10个队列
// ... Offline进程创建了新队列 ...
int num2 = get_queue_snapshot(...);  // 返回12个队列

// 解决方案：POC测试时，让Offline进程先稳定运行，再获取snapshot
```

### 4. 需要权限

```bash
# 需要以下之一：
# 1. Root权限
sudo ./get_queue_info 12345

# 2. CAP_SYS_PTRACE capability
sudo setcap cap_sys_ptrace=eip ./get_queue_info
./get_queue_info 12345

# 3. ptrace附加（同用户）
# 在代码中：
ptrace(PTRACE_ATTACH, target_pid, NULL, NULL);
```

---

## 📚 相关API对比

| API | 用途 | 输入 | 输出 | 需要root |
|-----|------|------|------|----------|
| **GET_QUEUE_SNAPSHOT** | 获取所有queue信息 | PID | Queue数组 | 否（需debug trap） |
| cat /sys/kernel/debug/kfd/process | 查看进程队列 | - | 文本 | 是 |
| cat /sys/kernel/debug/kfd/mqds | 查看MQD状态 | - | 文本 | 是 |
| SUSPEND_QUEUES | 暂停队列 | PID + queue_ids | 成功数量 | 否（需debug trap） |
| RESUME_QUEUES | 恢复队列 | PID + queue_ids | 成功数量 | 否（需debug trap） |

---

## 🎯 推荐用法

**POC最佳实践** ⭐⭐⭐:

```c
// 1. 启动Offline-AI进程
pid_t offline_pid = fork_offline_ai();

// 2. 等待Offline进程稳定（创建完所有队列）
sleep(5);

// 3. 获取所有queue_id
int num_queues = get_queue_snapshot(kfd_fd, offline_pid, snapshots, MAX_QUEUES);
uint32_t *queue_ids = extract_queue_ids(snapshots, num_queues);

// 4. POC测试循环
for (int i = 0; i < 100; i++) {
    suspend_queues(kfd_fd, offline_pid, queue_ids, num_queues);
    run_online_ai_and_measure_latency();
    resume_queues(kfd_fd, offline_pid, queue_ids, num_queues);
}
```

**优点**:
- ✅ 无需root权限（只需debug trap）
- ✅ 编程友好（C结构体）
- ✅ 稳定可靠（UAPI接口）
- ✅ 完整信息（包含所有MQD关键字段）

---

## 📝 总结

### 核心答案回顾

**Q: 有从代码中直接获取MQD对应的queue ID信息的ioctl吗？**

**A: `KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT` ⭐⭐⭐⭐⭐**

```c
// 一行获取所有信息
int num = get_queue_snapshot(kfd_fd, pid, snapshots, MAX_QUEUES);

// 每个snapshot包含：
// - queue_id (用于suspend)
// - gpu_id
// - ring_base_address
// - cwsr_address (抢占用)
// - queue_type
// - 等等...
```

**不需要cat sysfs！** ✨

---

## 📚 相关文档

- `POC_SUSPEND_QUEUES_API_GUIDE.md` - suspend/resume API使用
- `AQL_QUEUE_VS_MQD_RELATIONSHIP.md` - MQD结构详解
- `QUEUE_CREATION_TIMELINE.md` - Queue创建流程

---

**最后更新**: 2026-02-04  
**验证状态**: ✅ 基于内核UAPI定义  
**适用平台**: MI308X + 所有支持debug trap的AMD GPU
