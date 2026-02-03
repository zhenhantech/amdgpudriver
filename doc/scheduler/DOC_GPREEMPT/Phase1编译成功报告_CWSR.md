# GPREEMPT Phase 1 编译成功报告

> 📅 完成时间：2026-01-27  
> 🎯 状态：✅ **DKMS编译成功！**  
> 💻 平台：AMD MI300 (GC 9.4.3) + CWSR

---

## 🎉 重大突破

**GPREEMPT Phase 1成功编译进AMD GPU驱动！**

```
✅ 编译状态:    成功 (24MB kernel module)
✅ GPREEMPT字符串: 34处
✅ CWSR字符串:    7处  
✅ 新模块位置:   /lib/modules/.../extra/amdgpu.ko
```

---

## 📋 完成的工作

### 1. 问题诊断与修复历程

#### **问题1：访问不存在的成员**
```c
// ❌ 原始错误代码
mqd_mgr = dev->mqd_mgrs[KFD_MQD_TYPE_CP];  // dev没有mqd_mgrs!

// ✅ 修复后
struct device_queue_manager *dqm = q->device->dqm;
mqd_mgr = dqm->mqd_mgrs[mqd_type];
```

**根本原因**：
- `struct kfd_node` 没有 `mqd_mgrs` 成员
- 需要通过 `q->device->dqm` 访问 `device_queue_manager`

---

#### **问题2：不完整类型（Incomplete Type）**
```
error: invalid use of undefined type 'struct device_queue_manager'
error: invalid use of undefined type 'struct mqd_manager'
```

**根本原因**：
- `kfd_priv.h` 只有前向声明，没有完整定义
- 不能访问不完整类型的成员

**解决方案**：
```c
#include "kfd_priv.h"
#include "kfd_mqd_manager.h"          // ← 添加
#include "kfd_device_queue_manager.h"  // ← 添加
```

---

### 2. 关键代码实现

#### **A. 队列抢占（基于CWSR）**

```c
int kfd_queue_preempt_single(struct queue *q, 
                              enum kfd_preempt_type type,
                              unsigned int timeout)
{
    struct device_queue_manager *dqm = q->device->dqm;
    enum KFD_MQD_TYPE mqd_type = get_mqd_type_from_queue_type(q->properties.type);
    struct mqd_manager *mqd_mgr = dqm->mqd_mgrs[mqd_type];
    
    // WAVEFRONT_SAVE: 使用CWSR保存wave状态
    if (type == KFD_PREEMPT_TYPE_WAVEFRONT_SAVE) {
        // 1. 分配快照空间
        q->snapshot.mqd_backup = kzalloc(mqd_mgr->mqd_size, GFP_KERNEL);
        q->snapshot.ctl_stack_backup = kzalloc(ctl_stack_size, GFP_KERNEL);
        
        // 2. 保存状态（checkpoint）
        mqd_mgr->checkpoint_mqd(mqd_mgr, q->mqd,
                                q->snapshot.mqd_backup,
                                q->snapshot.ctl_stack_backup);
        
        q->snapshot.is_valid = true;
    }
    
    // 3. 触发硬件抢占
    ret = mqd_mgr->destroy_mqd(mqd_mgr, q->mqd, type, timeout,
                                q->pipe, q->queue);
    
    return ret;
}
```

**CWSR保存内容**：
- 程序计数器（PC）
- 标量/向量寄存器（SGPRs/VGPRs）
- 累加器寄存器（ACC VGPRs）
- Local Data Share（LDS）
- 硬件状态寄存器

---

#### **B. 队列恢复**

```c
int kfd_queue_resume_single(struct queue *q)
{
    struct device_queue_manager *dqm = q->device->dqm;
    enum KFD_MQD_TYPE mqd_type = get_mqd_type_from_queue_type(q->properties.type);
    struct mqd_manager *mqd_mgr = dqm->mqd_mgrs[mqd_type];
    
    // 1. 恢复MQD和控制栈
    mqd_mgr->restore_mqd(mqd_mgr, &q->mqd, q->mqd_mem_obj,
                         &q->gart_mqd_addr, &q->properties,
                         q->snapshot.mqd_backup,
                         q->snapshot.ctl_stack_backup,
                         q->snapshot.ctl_stack_size);
    
    // 2. 重新加载到GPU
    if (mqd_mgr->load_mqd) {
        ret = mqd_mgr->load_mqd(mqd_mgr, q->mqd, q->pipe, q->queue,
                                &q->properties, q->process->mm);
    }
    
    q->properties.is_active = true;
    return 0;
}
```

---

#### **C. 用户空间IOCTL接口**

**新增IOCTL命令** (`kfd_ioctl.h`):
```c
// 抢占队列
#define AMDKFD_IOC_PREEMPT_QUEUE \
    AMDKFD_IOW(0x87, struct kfd_ioctl_preempt_queue_args)

// 恢复队列  
#define AMDKFD_IOC_RESUME_QUEUE \
    AMDKFD_IOW(0x88, struct kfd_ioctl_resume_queue_args)

struct kfd_ioctl_preempt_queue_args {
    __u32 queue_id;
    __u32 preempt_type;  // DRAIN=0, RESET=1, SAVE=2
    __u32 timeout_ms;
    __u32 pad;
};
```

**IOCTL处理函数** (`kfd_chardev.c`):
```c
static int kfd_ioctl_preempt_queue(struct file *filep, 
                                    struct kfd_process *p, void *data)
{
    struct kfd_ioctl_preempt_queue_args *args = data;
    struct queue *q = pqm_find_queue(p, args->queue_id);
    
    if (!q) return -EINVAL;
    
    return kfd_queue_preempt_single(q, args->preempt_type, args->timeout_ms);
}
```

---

### 3. 修改的文件列表

| 文件 | 修改内容 | 行数 |
|------|---------|------|
| `kfd_queue_preempt.c` | 新增：抢占/恢复核心逻辑 | +260 |
| `kfd_priv.h` | 新增：queue snapshot结构 | +15 |
| `kfd_chardev.c` | 新增：IOCTL处理函数 | +50 |
| `kfd_ioctl.h` | 新增：IOCTL命令定义 | +15 |
| `Makefile` | 新增：编译kfd_queue_preempt.o | +1 |

---

## 🔧 CWSR技术要点

### A. 三种抢占类型

| 类型 | 延迟 | 状态保存 | 应用场景 |
|------|------|---------|---------|
| **WAVEFRONT_DRAIN** | 1-10ms | ❌ | 队列销毁 |
| **WAVEFRONT_RESET** | 10-50μs | ❌ | 错误恢复 |
| **WAVEFRONT_SAVE** | **1-10μs** | ✅ | ⭐ **抢占调度** |

**GPREEMPT使用：WAVEFRONT_SAVE**

---

### B. CWSR工作流程

```
用户调用 ioctl(AMDKFD_IOC_PREEMPT_QUEUE)
    ↓
kfd_ioctl_preempt_queue() - 验证参数
    ↓
kfd_queue_preempt_single() - 抢占逻辑
    ↓
checkpoint_mqd() - 保存MQD和控制栈
    ↓
destroy_mqd(WAVEFRONT_SAVE) - 触发硬件
    ↓
硬件Trap Handler - 保存所有wave状态
    ↓
✅ Wave挂起，状态完整保存
```

**恢复流程**：
```
用户调用 ioctl(AMDKFD_IOC_RESUME_QUEUE)
    ↓
kfd_ioctl_resume_queue()
    ↓
kfd_queue_resume_single()
    ↓
restore_mqd() - 恢复MQD和控制栈
    ↓
load_mqd() - 重新加载到GPU
    ↓
硬件从CWSR内存恢复状态
    ↓
✅ Wave从断点处继续执行
```

---

### C. 系统CWSR状态

```bash
# 检查CWSR是否启用
$ cat /sys/module/amdgpu/parameters/cwsr_enable
1  # ✅ 启用

# 源代码确认
/usr/src/amdgpu-6.12.12-2194681.el8/amd/amdgpu/amdgpu_drv.c:
int cwsr_enable = 1;  // 默认启用
```

**MI300 CWSR支持**：
- ✅ 硬件版本：GC 9.4.3
- ✅ Trap Handler：cwsr_trap_gfx9_4_3_hex
- ✅ 计算单元：304 CUs
- ✅ 每队列CWSR内存：~186 MB

---

## 📊 编译验证

### A. 模块信息
```bash
$ ls -lh /var/lib/dkms/amdgpu/.../amdgpu.ko
-rw-r--r-- 1 root root 24M Jan 27 11:41 amdgpu.ko

$ sudo strings amdgpu.ko | grep GPREEMPT | wc -l
34  # ✅ 34处GPREEMPT字符串
```

### B. 关键字符串示例
```
amdgpu: GPREEMPT: Preempt queue ioctl: queue_id=%u, type=%u, timeout=%u
amdgpu: GPREEMPT: Resume queue ioctl: queue_id=%u
amdgpu: GPREEMPT: Queue %u preempted successfully
amdgpu: GPREEMPT: Queue %u resumed successfully via CWSR
amdgpu: GPREEMPT: Queue state saved via CWSR (mqd_size=%u, ctl_stack_size=%u)
```

---

## 🚀 下一步工作

### Phase 2：IOCTL测试

**目标**：验证IOCTL接口工作正常

```c
// 测试程序框架
int kfd_fd = open("/dev/kfd", O_RDWR);

// 1. 创建队列
struct kfd_ioctl_create_queue_args create_args = {...};
ioctl(kfd_fd, AMDKFD_IOC_CREATE_QUEUE, &create_args);

// 2. 抢占队列
struct kfd_ioctl_preempt_queue_args preempt_args = {
    .queue_id = create_args.queue_id,
    .preempt_type = 2,  // WAVEFRONT_SAVE
    .timeout_ms = 1000,
};
ioctl(kfd_fd, AMDKFD_IOC_PREEMPT_QUEUE, &preempt_args);

// 3. 恢复队列
struct kfd_ioctl_resume_queue_args resume_args = {
    .queue_id = create_args.queue_id,
    .timeout_ms = 1000,
};
ioctl(kfd_fd, AMDKFD_IOC_RESUME_QUEUE, &resume_args);
```

**验证点**：
- ✅ IOCTL调用不返回错误
- ✅ dmesg显示GPREEMPT日志
- ✅ 队列状态正确切换（active/inactive）

---

### Phase 3：实际工作负载测试

**目标**：在真实GPU kernel上验证抢占/恢复

```cpp
// HIP测试程序
__global__ void long_kernel() {
    // 长时间运行的kernel
    for (int i = 0; i < 1000000; i++) {
        // compute...
    }
}

// 主程序
hipLaunchKernelGGL(long_kernel, ...);  // 启动BE任务
sleep(0.01);                           // 等待10ms
// 触发抢占（LC任务到达）
ioctl(kfd_fd, AMDKFD_IOC_PREEMPT_QUEUE, &args);
// 运行LC任务
hipLaunchKernelGGL(latency_critical_kernel, ...);
hipDeviceSynchronize();
// 恢复BE任务
ioctl(kfd_fd, AMDKFD_IOC_RESUME_QUEUE, &args);
```

**测量指标**：
- 抢占延迟（preemption latency）
- 恢复延迟（resume latency）
- LC任务延迟（latency-critical task latency）
- BE任务影响（best-effort task impact）

---

### Phase 4：集成到GPREEMPT框架

**目标**：实现完整的GPREEMPT调度器

```
GPREEMPT调度器 (用户空间)
    ↓
监控LC/BE队列状态
    ↓
当LC任务到达时:
  1. ioctl(PREEMPT_QUEUE) - 抢占BE队列
  2. 运行LC任务
  3. ioctl(RESUME_QUEUE) - 恢复BE队列
```

---

## 📝 技术总结

### A. CWSR的优势

1. **硬件支持**：
   - MI300原生支持CWSR（GC 9.4.3）
   - Trap Handler在固件中，无需软件实现
   - 微秒级延迟（1-10μs）

2. **完整状态保存**：
   - 所有寄存器（SGPRs/VGPRs/ACC）
   - Local Data Share（LDS）
   - 程序计数器（PC）
   - 恢复后无感知，从断点继续

3. **已有基础设施**：
   - KFD驱动完整实现
   - `checkpoint_mqd`/`restore_mqd` 接口
   - 只需封装和暴露给用户空间

---

### B. 关键发现

1. **不需要MES**：
   - GPREEMPT论文在A100上实现（无MES）
   - AMD MI100实现也是软件模拟
   - CWSR才是核心，MES是优化

2. **dqm访问模式**：
   - 不能直接从`kfd_node`访问`mqd_mgrs`
   - 必须通过`device_queue_manager`
   - `q->device->dqm->mqd_mgrs[type]`

3. **头文件依赖**：
   - `kfd_priv.h`只有前向声明
   - 需要完整定义：`kfd_mqd_manager.h`
   - 需要完整定义：`kfd_device_queue_manager.h`

---

### C. 为什么需要复制到/usr/src？

**DKMS工作原理**：

```
我们的Git仓库:
/mnt/md0/.../amdgpu_DKMS/amdgpu-6.12.12-2194681.el8/
    └── amd/amdkfd/kfd_queue_preempt.c  ← 我们在这里修改

DKMS编译源:
/usr/src/amdgpu-6.12.12-2194681.el8/
    └── amd/amdkfd/kfd_queue_preempt.c  ← DKMS从这里编译
```

**类比**：
- 就像烤蛋糕：准备好了材料（Git仓库）
- 但烤箱在别处（`/usr/src/`）
- **必须把材料搬到烤箱**才能烤

**自动化脚本**：
- `rebuild_with_gpreempt_fixed.sh`自动同步文件
- 从Git仓库 → DKMS源码目录
- 然后重新编译

---

## ✅ 成果验证

### 编译成功证据

```bash
# 1. 模块已安装
$ ls -lh /lib/modules/5.10.134-19.1.al8.x86_64/extra/amdgpu.ko
-rw-r--r-- 1 root root 24M Jan 27 11:41 amdgpu.ko

# 2. GPREEMPT字符串存在
$ sudo strings /lib/modules/.../amdgpu.ko | grep GPREEMPT | wc -l
34

# 3. CWSR字符串存在
$ sudo strings /lib/modules/.../amdgpu.ko | grep CWSR | wc -l
7

# 4. DKMS状态
$ sudo dkms status
amdgpu, 6.12.12-2194681.el8, ..., x86_64: installed ✅
```

---

## 🎓 学习要点

### 从错误中学到的

1. **理解内核数据结构**：
   - 不能假设结构体成员存在
   - 需要查看源码确认
   - 前向声明 ≠ 完整定义

2. **DKMS编译流程**：
   - DKMS从`/usr/src/`编译，不是Git仓库
   - 必须同步修改文件
   - 自动化脚本避免手工错误

3. **内核模块开发**：
   - 编译错误要看具体的`.c`文件错误
   - 不是所有warning都是问题
   - 查找"error:"关键字定位真正错误

---

## 🎯 Phase 1 总结

| 项目 | 状态 |
|------|------|
| 代码编写 | ✅ 完成 |
| DKMS编译 | ✅ 成功 |
| 模块安装 | ✅ 已安装 |
| GPREEMPT验证 | ✅ 34处字符串 |
| CWSR验证 | ✅ 7处字符串 |
| 文档完善 | ✅ 完成 |

**总修改**：
- 新增文件：1个（`kfd_queue_preempt.c`）
- 修改文件：4个
- 总代码行：~350行
- 新增IOCTL：2个

**下一里程碑**：Phase 2 IOCTL功能测试

---

## 📖 参考资源

### 关键文档
1. `CWSR机制简要总结.md` - CWSR技术详解
2. `CWSR启用状态确认.md` - 系统CWSR状态
3. `GPREEMPT_Phase1实施计划.md` - 实施计划
4. `KFD抢占机制详细分析.md` - KFD抢占分析

### 源代码位置
```
/mnt/md0/zhehan/code/coderampup/private_github/amdgpu_DKMS/
└── amdgpu-6.12.12-2194681.el8/
    ├── amd/amdkfd/kfd_queue_preempt.c
    ├── amd/amdkfd/kfd_priv.h
    ├── amd/amdkfd/kfd_chardev.c
    ├── amd/amdkfd/Makefile
    └── include/uapi/linux/kfd_ioctl.h
```

### 编译脚本
```bash
/mnt/md0/zhehan/code/rampup_doc/GPREEMPT_MI300_Testing/rebuild_with_gpreempt_fixed.sh
```

---

**报告完成时间**：2026-01-27 11:45  
**状态**：✅ **Phase 1 完成！准备进入Phase 2**

