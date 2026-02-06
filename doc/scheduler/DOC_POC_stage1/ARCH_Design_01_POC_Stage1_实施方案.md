# POC Stage 1: Online/Offline AI 模型优先级调度实施方案

**日期**: 2026-02-03  
**目标**: 验证 AMD GPU 上的 Queue-level 优先级抢占可行性  
**场景**: Online-AI 推理（高优先级）抢占 Offline-AI 训练（低优先级）

---

## 📋 测试场景定义

### 角色定义

| 角色 | 优先级 | 实时性要求 | 典型负载 | 队列类型 |
|------|-------|-----------|---------|---------|
| **Online-AI-model** | 高 (15) | ✅ 强（< 50ms） | 推理 (小 kernel) | 1-2 个队列 |
| **Offline-AI-model** | 低 (2) | ❌ 无 | 训练 (长 kernel) | 多个队列 |

### 测试目标

✅ **核心目标**: 当 Online-AI 任务到达时，能够快速暂停 Offline-AI，让 Online-AI 优先执行

**成功标准**:
1. Online-AI 端到端延迟 < 50ms
2. Offline-AI 正确恢复执行（无数据丢失）
3. 吞吐量损失 < 5%
4. 系统稳定（无崩溃）

---

## 🔍 三种可用 API 分析

### API 1: CWSR (Compute Wave Save/Restore) - ⭐⭐⭐⭐⭐

**优势**:
- ✅ Wave-level 细粒度抢占（最精确）
- ✅ 硬件自动保存/恢复状态
- ✅ 不丢失 kernel 执行进度
- ✅ 已在 CRIU 中验证可用

**API 接口**:
```c
// 位置: kfd_process_queue_manager.c:809-829
int pqm_checkpoint_mqd(struct process_queue_manager *pqm,
                       unsigned int qid,
                       void *mqd,
                       void *ctl_stack);

// MQD Manager 层
mqd_mgr->checkpoint_mqd(...);  // 保存 MQD + control stack
mqd_mgr->destroy_mqd(...);     // 触发 CWSR，停止队列
mqd_mgr->restore_mqd(...);     // 恢复状态
mqd_mgr->load_mqd(...);        // 重新激活队列
```

**复杂度**: 中等（需要内存管理）

---

### API 2: KFD_IOC_DBG_TRAP_SUSPEND_QUEUES - ⭐⭐⭐⭐

**优势**:
- ✅ 专门的队列暂停接口
- ✅ 支持批量暂停多个队列
- ✅ grace_period 支持（优雅停止）
- ✅ 已有完整实现

**API 接口**:
```c
// 位置: kfd_chardev.c:3310-3316
int suspend_queues(struct kfd_process *target,
                   uint32_t num_queues,
                   uint32_t grace_period,
                   uint64_t exception_mask,
                   uint32_t *queue_array_ptr);

int resume_queues(struct kfd_process *target,
                  uint32_t num_queues,
                  uint32_t *queue_array_ptr);
```

**用途**: 原本用于调试（GDB 调试 GPU 程序）

**复杂度**: 低（直接调用即可）

**限制**:
- ⚠️ 需要知道队列 ID
- ⚠️ 可能包含额外的调试逻辑

---

### API 3: amdgpu_amdkfd_stop_sched - ⭐⭐

**优势**:
- ✅ 最简单的接口
- ✅ 停止整个 KFD 调度器

**API 接口**:
```c
// 位置: amdgpu_amdkfd.c:898-903
int amdgpu_amdkfd_stop_sched(struct amdgpu_device *adev, uint32_t node_id);
int amdgpu_amdkfd_start_sched(struct amdgpu_device *adev, uint32_t node_id);
```

**复杂度**: 极低（一行调用）

**限制**:
- ❌ 粒度太粗（停止整个 GPU 的调度）
- ❌ 影响所有队列（包括高优先级）
- ❌ 不适合细粒度控制

---

## 🎯 POC Stage 1 推荐方案

### 方案选择: KFD_IOC_DBG_TRAP_SUSPEND_QUEUES ⭐⭐⭐⭐⭐

**为什么选择这个**:
1. ✅ **复杂度最低** - 已有完整实现，直接调用
2. ✅ **粒度合适** - 可以精确暂停指定队列
3. ✅ **快速验证** - 能快速证明概念可行性
4. ✅ **可扩展** - 后续可以升级到 CWSR

**实施路径**:
```
POC Stage 1 (suspend_queues)  →  POC Stage 2 (CWSR)  →  Production
     ↓                              ↓                      ↓
  快速验证概念              优化性能和细粒度            完整调度器
```

---

## 📐 POC Stage 1 架构设计

### 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│ Test Framework (Python/C++)                                   │
│                                                                │
│  ┌─────────────────┐        ┌─────────────────┐             │
│  │ Online-AI Model │        │ Offline-AI Model│             │
│  │  (推理，高优先级)│        │  (训练，低优先级)│             │
│  │  Priority = 15  │        │  Priority = 2   │             │
│  └────────┬────────┘        └────────┬────────┘             │
│           │ hipLaunchKernel          │ hipLaunchKernel      │
│           │ (Doorbell)               │ (Doorbell)           │
└───────────┼──────────────────────────┼──────────────────────┘
            │                          │
            ↓                          ↓
┌──────────────────────────────────────────────────────────────┐
│ Test Framework - Scheduler Thread (User space)                │
│                                                                │
│  1. 监控任务提交                                               │
│     • Online-AI 任务到达 → 设置 online_task_pending = true   │
│                                                                │
│  2. 查询队列信息                                               │
│     • 通过 /sys/kernel/debug/kfd/mqds 读取队列 ID             │
│     • 识别 Online/Offline 队列                                │
│                                                                │
│  3. 触发抢占                                                   │
│     • ioctl(KFD_IOC_DBG_TRAP_SUSPEND_QUEUES) 暂停 Offline    │
│                                                                │
│  4. 等待 Online-AI 完成                                        │
│     • 轮询或回调通知                                           │
│                                                                │
│  5. 恢复 Offline-AI                                            │
│     • ioctl(KFD_IOC_DBG_TRAP_RESUME_QUEUES) 恢复 Offline     │
└──────────────────────────────────────────────────────────────┘
            │                          │
            ↓ ioctl                    ↓
┌──────────────────────────────────────────────────────────────┐
│ KFD (内核驱动)                                                 │
│                                                                │
│  suspend_queues(target, num_queues, grace_period,             │
│                exception_mask, queue_array_ptr);              │
│  ↓                                                             │
│  • 遍历队列数组                                                │
│  • 调用 DQM 的 evict_process_queues_cpsch()                   │
│  • 触发 CWSR 保存状态                                          │
│  • 队列从 Runlist 中移除                                       │
│                                                                │
│  resume_queues(target, num_queues, queue_array_ptr);          │
│  ↓                                                             │
│  • 调用 DQM 的 restore_process_queues_cpsch()                 │
│  • 恢复队列状态                                                │
│  • 队列重新加入 Runlist                                        │
└──────────────────────────────────────────────────────────────┘
            │
            ↓ PM4 Commands
┌──────────────────────────────────────────────────────────────┐
│ GPU Hardware (CPSCH 模式)                                      │
│                                                                │
│  • CP Scheduler 处理 UNMAP_QUEUES / MAP_QUEUES               │
│  • 触发 CWSR 机制                                              │
│  • Wave-level 状态保存/恢复                                    │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔎 关键内核调用路径（文件 + 行号）

**ioctl 入口 → suspend/resume**
```
3310:3321:/usr/src/amdgpu-6.12.12-2194681.el8_preempt/amd/amdkfd/kfd_chardev.c
case KFD_IOC_DBG_TRAP_SUSPEND_QUEUES:
    r = suspend_queues(target, ...);
    break;
case KFD_IOC_DBG_TRAP_RESUME_QUEUES:
    r = resume_queues(target, ...);
    break;
```

**CPSCH 路径：evict/restore → execute_queues**
```
1253:1305:/usr/src/amdgpu-6.12.12-2194681.el8_preempt/amd/amdkfd/kfd_device_queue_manager.c
static int evict_process_queues_cpsch(...) { ... execute_queues_cpsch(...); }

1393:1447:/usr/src/amdgpu-6.12.12-2194681.el8_preempt/amd/amdkfd/kfd_device_queue_manager.c
static int restore_process_queues_cpsch(...) { ... execute_queues_cpsch(...); }
```

**execute_queues_cpsch = unmap + map**
```
2442:2455:/usr/src/amdgpu-6.12.12-2194681.el8_preempt/amd/amdkfd/kfd_device_queue_manager.c
static int execute_queues_cpsch(...)
{
  retval = unmap_queues_cpsch(...);
  if (!retval)
      retval = map_queues_cpsch(...);
}
```

---

## 🧭 MES 路径 vs CPSCH 路径（分支图）

```
SUSPEND_QUEUES / RESUME_QUEUES
            │
            ▼
     suspend_queues() / resume_queues()
            │
            ├── if (enable_mes = true)
            │       │
            │       ├─ suspend: remove_queue_mes()
            │       └─ resume : add_queue_mes()
            │
            └── if (enable_mes = false)  ← CPSCH
                    │
                    ├─ evict/restore_process_queues_cpsch()
                    │     └─ execute_queues_cpsch()
                    │           ├─ unmap_queues_cpsch()
                    │           │     └─ pm_send_unmap_queue()
                    │           └─ map_queues_cpsch()
                    │                 └─ pm_send_runlist()
```

## 📝 详细实施步骤

### Step 1: 队列识别机制 (1-2天)

**目标**: Test Framework 能识别 Online/Offline 队列

**实现方法**:

**方法 A: 通过环境变量标记** (推荐)
```python
# Online-AI 模型
import os
os.environ['AMD_QUEUE_PRIORITY'] = '15'
os.environ['AMD_QUEUE_TAG'] = 'ONLINE_AI'

# Offline-AI 模型
os.environ['AMD_QUEUE_PRIORITY'] = '2'
os.environ['AMD_QUEUE_TAG'] = 'OFFLINE_AI'

# Test Framework 读取
online_queues = find_queues_by_tag('ONLINE_AI')
offline_queues = find_queues_by_tag('OFFLINE_AI')
```

**方法 B: 通过 /proc 解析** (备选)
```python
def get_process_queues(pid):
    # 解析 /sys/kernel/debug/kfd/mqds
    # 找到属于该进程的队列
    queues = []
    with open('/sys/kernel/debug/kfd/mqds', 'r') as f:
        content = f.read()
        # 解析队列 ID, 优先级等信息
    return queues
```

**方法 C: 通过优先级过滤** (最简单)
```python
def classify_queues():
    queues = parse_mqd_debugfs()
    online_queues = [q for q in queues if q['priority'] >= 10]
    offline_queues = [q for q in queues if q['priority'] < 10]
    return online_queues, offline_queues
```

---

### Step 2: suspend_queues API 封装 (1天)

**C 库封装**: `libgpreempt_poc.so`

```c
// gpreempt_poc.h

#include <stdint.h>
#include <stdbool.h>

// 初始化
int gpreempt_poc_init(void);
void gpreempt_poc_cleanup(void);

// 队列操作
int gpreempt_suspend_queues(uint32_t *queue_ids, 
                           uint32_t num_queues,
                           uint32_t grace_period_us);

int gpreempt_resume_queues(uint32_t *queue_ids,
                          uint32_t num_queues);

// 队列查询
typedef struct {
    uint32_t queue_id;
    uint32_t priority;
    uint32_t gpu_id;
    char tag[64];
    bool is_active;
} queue_info_t;

int gpreempt_get_all_queues(queue_info_t **queues, uint32_t *num_queues);
int gpreempt_find_queues_by_priority(uint32_t min_priority, 
                                    uint32_t max_priority,
                                    queue_info_t **queues,
                                    uint32_t *num_queues);
```

**实现**: `gpreempt_poc.c`

```c
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/kfd_ioctl.h>
#include "gpreempt_poc.h"

static int kfd_fd = -1;

int gpreempt_poc_init(void) {
    kfd_fd = open("/dev/kfd", O_RDWR);
    if (kfd_fd < 0) {
        perror("Failed to open /dev/kfd");
        return -1;
    }
    return 0;
}

void gpreempt_poc_cleanup(void) {
    if (kfd_fd >= 0) {
        close(kfd_fd);
        kfd_fd = -1;
    }
}

int gpreempt_suspend_queues(uint32_t *queue_ids, 
                           uint32_t num_queues,
                           uint32_t grace_period_us) {
    struct kfd_ioctl_dbg_trap_args args = {0};
    
    args.op = KFD_IOC_DBG_TRAP_SUSPEND_QUEUES;
    args.suspend_queues.num_queues = num_queues;
    args.suspend_queues.grace_period = grace_period_us;
    args.suspend_queues.exception_mask = 0;
    args.suspend_queues.queue_array_ptr = (uint64_t)queue_ids;
    
    int ret = ioctl(kfd_fd, AMDKFD_IOC_DBG_TRAP, &args);
    if (ret < 0) {
        perror("suspend_queues ioctl failed");
        return -1;
    }
    
    return 0;
}

int gpreempt_resume_queues(uint32_t *queue_ids, uint32_t num_queues) {
    struct kfd_ioctl_dbg_trap_args args = {0};
    
    args.op = KFD_IOC_DBG_TRAP_RESUME_QUEUES;
    args.resume_queues.num_queues = num_queues;
    args.resume_queues.queue_array_ptr = (uint64_t)queue_ids;
    
    int ret = ioctl(kfd_fd, AMDKFD_IOC_DBG_TRAP, &args);
    if (ret < 0) {
        perror("resume_queues ioctl failed");
        return -1;
    }
    
    return 0;
}

int gpreempt_get_all_queues(queue_info_t **queues, uint32_t *num_queues) {
    // 解析 /sys/kernel/debug/kfd/mqds
    FILE *fp = fopen("/sys/kernel/debug/kfd/mqds", "r");
    if (!fp) {
        perror("Failed to open mqds");
        return -1;
    }
    
    // 解析逻辑...
    // TODO: 实现 MQD debugfs 解析
    
    fclose(fp);
    return 0;
}
```

---

### Step 3: Test Framework 主程序 (2天)

**Python 实现**: `test_priority_scheduling.py`

```python
#!/usr/bin/env python3
"""
POC Stage 1: Online/Offline AI 优先级调度测试框架
"""

import ctypes
import time
import threading
from dataclasses import dataclass
from typing import List

# 加载 C 库
lib = ctypes.CDLL('./libgpreempt_poc.so')

@dataclass
class QueueInfo:
    queue_id: int
    priority: int
    gpu_id: int
    tag: str
    is_active: bool

class GPreemptScheduler:
    def __init__(self):
        self.lib = lib
        self.lib.gpreempt_poc_init()
        
        self.online_queues = []
        self.offline_queues = []
        self.online_task_pending = False
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _monitor_loop(self):
        """监控线程：检测 Online 任务并触发抢占"""
        while True:
            time.sleep(0.001)  # 1ms 检测间隔
            
            if self.online_task_pending:
                print("[SCHED] 检测到 Online 任务，触发抢占...")
                self._handle_online_task()
    
    def _handle_online_task(self):
        """处理 Online 任务到达"""
        
        # 1. 暂停所有 Offline 队列
        offline_ids = [q.queue_id for q in self.offline_queues]
        if offline_ids:
            print(f"[SCHED] 暂停 {len(offline_ids)} 个 Offline 队列")
            ret = self.lib.gpreempt_suspend_queues(
                (ctypes.c_uint32 * len(offline_ids))(*offline_ids),
                len(offline_ids),
                1000  # 1ms grace period
            )
            if ret == 0:
                print(f"[SCHED] ✅ Offline 队列已暂停")
            else:
                print(f"[SCHED] ❌ 暂停失败")
        
        # 2. 等待 Online 任务完成
        # (通过某种机制检测，例如回调、轮询 rptr/wptr 等)
        self._wait_for_online_completion()
        
        # 3. 恢复 Offline 队列
        if offline_ids:
            print(f"[SCHED] 恢复 {len(offline_ids)} 个 Offline 队列")
            ret = self.lib.gpreempt_resume_queues(
                (ctypes.c_uint32 * len(offline_ids))(*offline_ids),
                len(offline_ids)
            )
            if ret == 0:
                print(f"[SCHED] ✅ Offline 队列已恢复")
            else:
                print(f"[SCHED] ❌ 恢复失败")
        
        self.online_task_pending = False
    
    def _wait_for_online_completion(self):
        """等待 Online 任务完成"""
        # TODO: 实现完成检测
        # 方法1: 固定时间片 (简单)
        time.sleep(0.010)  # 10ms
        
        # 方法2: 轮询队列状态 (精确)
        # while not online_queue_idle():
        #     time.sleep(0.001)
    
    def register_online_queue(self, queue_id, priority=15):
        """注册 Online 队列"""
        q = QueueInfo(queue_id, priority, 0, "ONLINE", True)
        self.online_queues.append(q)
        print(f"[SCHED] 注册 Online 队列: {queue_id}, priority={priority}")
    
    def register_offline_queue(self, queue_id, priority=2):
        """注册 Offline 队列"""
        q = QueueInfo(queue_id, priority, 0, "OFFLINE", True)
        self.offline_queues.append(q)
        print(f"[SCHED] 注册 Offline 队列: {queue_id}, priority={priority}")
    
    def notify_online_task(self):
        """通知有 Online 任务到达"""
        self.online_task_pending = True
    
    def cleanup(self):
        self.lib.gpreempt_poc_cleanup()


# ═══════════════════════════════════════════════════════════════
#  测试场景
# ═══════════════════════════════════════════════════════════════

def test_online_offline_scheduling():
    """测试 Online/Offline 调度"""
    
    sched = GPreemptScheduler()
    
    # 1. 启动 Offline-AI 模型（训练）
    print("\n[TEST] 启动 Offline-AI 模型（训练）...")
    offline_process = launch_offline_ai()
    
    # 等待队列创建
    time.sleep(2)
    
    # 2. 扫描并注册队列
    print("\n[TEST] 扫描队列...")
    offline_queues = find_queues_by_priority(min_prio=0, max_prio=5)
    for q in offline_queues:
        sched.register_offline_queue(q.queue_id, q.priority)
    
    # 3. 启动 Online-AI 模型（推理）
    print("\n[TEST] 启动 Online-AI 模型（推理）...")
    online_process = launch_online_ai()
    
    time.sleep(1)
    
    online_queues = find_queues_by_priority(min_prio=10, max_prio=15)
    for q in online_queues:
        sched.register_online_queue(q.queue_id, q.priority)
    
    # 4. 模拟 Online 任务到达
    print("\n[TEST] 模拟 Online 任务到达...")
    for i in range(10):
        print(f"\n[TEST] === Online 任务 #{i+1} ===")
        sched.notify_online_task()
        time.sleep(0.5)  # 每 500ms 一个推理请求
    
    # 5. 清理
    sched.cleanup()
    offline_process.terminate()
    online_process.terminate()


if __name__ == '__main__':
    test_online_offline_scheduling()
```

---

## 🔬 测试用例设计

### Test Case 1: 基本抢占测试

**场景**:
1. Offline-AI 持续运行（长 kernel）
2. Online-AI 间歇提交（短 kernel）
3. 验证 Online 任务能快速执行

**验证点**:
- [ ] Offline 队列被正确暂停
- [ ] Online 任务延迟 < 50ms
- [ ] Offline 队列正确恢复
- [ ] 无 kernel 执行错误

---

### Test Case 2: 频繁抢占测试

**场景**:
1. Online-AI 高频提交（每 100ms）
2. Offline-AI 持续运行

**验证点**:
- [ ] 频繁 suspend/resume 不导致错误
- [ ] Offline 吞吐量下降 < 10%
- [ ] 系统稳定运行 10 分钟+

---

### Test Case 3: 边界条件测试

**场景**:
- Offline 队列为空时的处理
- Online 和 Offline 同时提交
- 多个 Online 任务并发

**验证点**:
- [ ] 边界条件不崩溃
- [ ] 错误处理正确

---

## 📊 性能指标

### 关键延迟指标

| 指标 | 目标 | 测量方法 |
|------|------|---------|
| **Online 任务端到端延迟** | < 50ms | timestamp 对比 |
| **Suspend 操作延迟** | < 5ms | ioctl 返回时间 |
| **Resume 操作延迟** | < 5ms | ioctl 返回时间 |
| **Offline 吞吐量损失** | < 10% | 对比baseline |

### 测量代码

```python
import time

def measure_online_latency():
    start = time.time()
    
    # 触发抢占
    sched.notify_online_task()
    
    # 等待 Online 任务完成
    # （通过某种方式检测）
    
    end = time.time()
    latency_ms = (end - start) * 1000
    
    print(f"Online 任务延迟: {latency_ms:.2f} ms")
    return latency_ms
```

---

## 🚧 已知限制和风险

### 限制

1. **需要 root 权限**
   - suspend_queues 需要访问调试接口
   - 可能需要 `sudo` 运行

2. **调试接口的副作用**
   - `KFD_IOC_DBG_TRAP_SUSPEND_QUEUES` 原本用于调试
   - 可能包含额外的检查和日志

3. **队列识别挑战**
   - 需要可靠的方法识别 Online/Offline 队列
   - MQD debugfs 格式可能变化

### 风险

⚠️ **系统稳定性**
- 频繁 suspend/resume 可能导致驱动不稳定
- 需要充分测试

⚠️ **性能开销**
- ioctl 系统调用开销（~1-10μs）
- 可能不满足极低延迟要求（< 1ms）

---

## 🛠️ 实施计划

### Week 1: 基础框架

- [x] Day 1-2: 队列识别机制实现
- [x] Day 3: C 库封装 (libgpreempt_poc)
- [ ] Day 4: Python Test Framework 主程序
- [ ] Day 5: 基本测试用例

### Week 2: 测试和优化

- [ ] Day 6-7: 功能测试
- [ ] Day 8-9: 性能测试和调优
- [ ] Day 10: 文档和报告

---

## 📚 参考资料

### KFD API 参考

1. **suspend_queues 实现**
   - 位置: `kfd_chardev.c:3310-3316`
   - 调用路径: `ioctl → suspend_queues → evict_process_queues_cpsch`

2. **resume_queues 实现**
   - 位置: `kfd_chardev.c:3318-3321`
   - 调用路径: `ioctl → resume_queues → restore_process_queues_cpsch`

### 相关文档

- `TODOLIST.md` - 完整实施计划
- `CWSR_API_USAGE_REFERENCE.md` - CWSR API 参考
- `CRIU_CODE_REUSE_ANALYSIS.md` - CRIU 代码复用分析

---

## ➡️ 下一步: POC Stage 2

如果 POC Stage 1 成功验证概念，Stage 2 将升级到更优的方案：

**POC Stage 2: 直接使用 CWSR API**
- 绕过 debugfs trap 接口
- 直接调用 `pqm_checkpoint_mqd` 等 CWSR API
- 更低的延迟和开销
- 更精确的控制

**POC Stage 3: 内核态调度器**
- 实现完整的 GPREEMPT 调度器（TODOLIST.md 中的 Phase 1-5）
- 无 ioctl 开销
- 完全自动化的优先级调度

---

**最后更新**: 2026-02-03  
**维护者**: Zhehan

**结论**: POC Stage 1 使用 `KFD_IOC_DBG_TRAP_SUSPEND_QUEUES` API 进行快速概念验证，实施复杂度低，能快速证明队列级抢占的可行性。✅
