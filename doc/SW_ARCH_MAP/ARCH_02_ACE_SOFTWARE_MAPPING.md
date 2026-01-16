# ACE在软件驱动中的表现形式分析

**文档类型**: 硬件架构与软件映射分析  
**创建时间**: 2025-12-29  
**关联文档**: `ARCH_01_MI300_HARDWARE_QUEUE_ANALYSIS.md`

## ACE在软件驱动中的表现形式

### ✅ 核心发现：ACE = Hardware Queue (硬件队列)

根据AMD官方文档和社区资料，**ACE在软件驱动中表现为硬件队列（Hardware Queue）**，也称为**Compute Ring**或**Compute Queue**。

### 1. OpenCL中的映射关系

根据[AMD OpenCL文档](https://community.amd.com/t5/opencl/how-to-use-opencl-multiple-command-queues/td-p/599543)：

- **每个OpenCL命令队列**在创建时会被分配到一个**硬件队列**
- **分配方式**: 如果硬件支持K个并发硬件队列，第N个OpenCL队列会被分配到第 `(N mod K)` 个硬件队列
- **环境变量控制**: ⚠️ **注意**: 文档提到可以通过环境变量限制可用的计算队列数量，但`GPU_NUM_COMPUTE_RINGS`这个具体变量名是**基于文档描述的推断**，需要验证是否真实存在

**映射关系**：
```
OpenCL Command Queue → Hardware Queue (ACE)
分配算法: Queue_ID = N mod K
其中:
  N = OpenCL队列在上下文中的创建顺序
  K = 硬件支持的并发硬件队列数量（ACE数量）
```

### 2. ROCm/HIP中的映射关系

在ROCm/HIP中，ACE表现为：

1. **HIP Stream → Hardware Queue**
   - 每个HIP Stream对应一个硬件队列
   - 多个Stream可以映射到不同的ACE
   - Stream的创建顺序决定映射到哪个ACE

2. **Compute Ring**
   - ACE在驱动层表现为Compute Ring
   - Ring是硬件队列的软件抽象
   - 多个Ring可以并行执行

3. **User Mode Queue (UMQ)**
   - 现代AMD GPU支持多个用户模式队列
   - 调度固件动态地将用户队列映射到硬件队列槽位
   - 当用户队列数量超过硬件队列槽位时，使用优先级和时间片调度

### 3. 驱动层的实现

根据[Linux内核文档](https://docs.kernel.org/gpu/amdgpu/userq.html)：

- **硬件队列槽位**: 每个ACE对应一个硬件队列槽位
- **动态映射**: 调度固件动态地将用户队列映射到可用的硬件队列槽位
- **超量订阅**: 当用户队列数量超过硬件队列槽位时，使用调度策略管理

**关键概念**：
```
User Mode Queue (UMQ) → Hardware Queue Slot (ACE)
调度固件负责动态映射和调度
```

## MI300的ACE到Queue映射

### 硬件配置

根据`ARCH_01_MI300_HARDWARE_QUEUE_ANALYSIS.md`：

- **MI300X**: 8 XCDs × 4 ACEs = **32个硬件队列**
- **MI300A**: 6 XCDs × 4 ACEs = **24个硬件队列**

### 软件映射

1. **HIP Stream映射**
   ```
   Stream 0 → ACE 0 (XCD 0, ACE 0)
   Stream 1 → ACE 1 (XCD 0, ACE 1)
   Stream 2 → ACE 2 (XCD 0, ACE 2)
   Stream 3 → ACE 3 (XCD 0, ACE 3)
   Stream 4 → ACE 4 (XCD 1, ACE 0)
   ...
   Stream 31 → ACE 31 (XCD 7, ACE 3)
   ```

2. **环境变量控制** ⚠️ **OpenCL特有，HIP中不存在**
   ```bash
   # ⚠️ 重要：GPU_NUM_COMPUTE_RINGS是OpenCL API栈中的环境变量
   # ❌ 在ROCm/HIP API栈中不存在（已验证：FWK_15_HIP_CODE_SEARCH_RESULTS.md）
   # 
   # 如果使用OpenCL，可以这样设置：
   export GPU_NUM_COMPUTE_RINGS=32  # 使用所有32个ACE
   export GPU_NUM_COMPUTE_RINGS=16  # 只使用16个ACE
   export GPU_NUM_COMPUTE_RINGS=4   # 只使用4个ACE（单XCD）
   #
   # 对于HIP应用，需要寻找其他机制：
   # - HIP Stream优先级设置
   # - 驱动层参数
   # - 应用层限制Stream数量
   ```

3. **进程到ACE映射**
   - 默认情况下，进程创建的Stream会按顺序映射到ACE
   - 多个进程可能竞争同一个ACE
   - **没有进程到ACE的显式绑定机制**

## 🔍 关键发现：Runlist过度订阅

**从系统dmesg日志中发现**：
```
amdgpu: Runlist is getting oversubscribed due to too many queues. 
Expect reduced ROCm performance.
```

**这是多进程性能问题的根本原因！**

### Runlist过度订阅的含义

1. **Runlist是什么**
   - Runlist是GPU调度器维护的运行列表
   - 包含所有待执行的队列（Queue）
   - 当队列数量超过硬件能力时，会发生过度订阅

2. **过度订阅的影响**
   - GPU调度器无法及时处理所有队列
   - 队列在Runlist中排队等待
   - 导致性能下降和延迟增加

3. **与我们的问题关联**
   - 8个进程创建了太多队列
   - Runlist被过度订阅
   - 导致job提交效率只有66%
   - 最大job count达到287，说明严重排队

## 与多进程性能问题的关联

### 问题分析

基于`DRIVER_21_1PROC_VS_2PROC_COMPARISON.md`的分析和dmesg日志：

1. **8进程时效率只有66%**
   - 硬件有32个ACE，理论上可以支持32个并行队列
   - 但8个进程时效率只有66%，说明没有充分利用硬件能力
   - **dmesg显示Runlist过度订阅，这是直接原因**

2. **可能的原因**：

   a. **Runlist过度订阅** ⚠️ **根本原因**
      - 8个进程创建了太多队列
      - Runlist无法及时处理所有队列
      - 导致job排队和性能下降
      - **dmesg明确显示此问题**

   b. **默认映射策略问题**
      - 多个进程的Stream可能映射到相同的ACE
      - 没有进程到ACE的智能分配
      - 导致某些ACE空闲，某些ACE过载

   c. **队列数量限制** ⚠️ **HIP中没有环境变量控制**
      - ⚠️ `GPU_NUM_COMPUTE_RINGS`是OpenCL特有的，HIP中不存在
      - HIP中可能通过其他机制限制队列数量
      - 需要查找HIP中的替代方案（驱动层参数、Stream优先级等）

   d. **驱动层调度瓶颈**
      - 虽然硬件有32个ACE，但驱动层调度器可能成为瓶颈
      - Runlist调度器无法高效处理大量队列

### 验证方法

1. **检查环境变量** ⚠️ **仅适用于OpenCL**
   ```bash
   # ⚠️ 注意：此环境变量仅适用于OpenCL，HIP中不存在
   echo $GPU_NUM_COMPUTE_RINGS  # 如果使用OpenCL
   
   # 对于HIP应用，需要查找其他机制
   ```

2. **检查驱动层队列配置**
   ```bash
   # 查看GPU信息
   rocm-smi --showid
   
   # 查看队列信息（如果可用）
   cat /sys/kernel/debug/dri/*/amdgpu_ring_info
   ```

3. **监控ACE利用率**
   - 使用`rocprof`或性能计数器
   - 监控各ACE的负载分布
   - 确认是否存在ACE空闲或过载

## 优化建议

### 1. 队列数量控制 ⚠️ **HIP中需要替代方案**

```bash
# ⚠️ GPU_NUM_COMPUTE_RINGS是OpenCL特有的，HIP中不存在
# 
# 对于HIP应用，可能的替代方案：
# 1. 应用层限制Stream数量
# 2. 使用HIP Stream优先级
# 3. 查找驱动层参数（需要进一步研究）
#
# 如果使用OpenCL，可以设置：
# export GPU_NUM_COMPUTE_RINGS=32  # 使用所有32个ACE
```

### 2. 进程到ACE绑定

- **研究HIP Stream到ACE的绑定API**
- **实现进程到ACE的显式映射**
- **避免多个进程竞争同一个ACE**

### 3. 驱动层优化

- **确认驱动层是否实现多队列调度**
- **优化调度器，充分利用32个ACE**
- **减少调度器锁竞争**

### 4. CUDA Graph优化

- **确保Graph replay使用多个ACE**
- **为每个进程创建独立的Graph，绑定到不同ACE**
- **避免Graph replay时的ACE竞争**

## 关键术语对照表

| 硬件层 | 软件驱动层 | 应用层 |
|--------|-----------|--------|
| ACE (Asynchronous Compute Engine) | Hardware Queue / Compute Ring | HIP Stream / OpenCL Queue |
| XCD (Accelerator Complex Die) | GPU Device | GPU Device |
| HWS (Hardware Scheduler) | Kernel Scheduler / Runlist | - |
| Ring (SDMA) | DMA Queue | - |
| Runlist | Queue调度列表 | - |

**环境变量控制**：
- ⚠️ `GPU_NUM_COMPUTE_RINGS`：**仅适用于OpenCL**，HIP中不存在（已验证：`FWK_15_HIP_CODE_SEARCH_RESULTS.md`）
- HIP中需要寻找其他机制（Stream优先级、驱动层参数等）

## 🔍 关键发现总结

### ACE在软件驱动中的表现形式

1. **ACE = Hardware Queue (硬件队列)**
   - 在OpenCL中：每个命令队列映射到一个硬件队列
   - 在HIP中：每个Stream映射到一个硬件队列
   - 映射方式：`Queue_ID = N mod K`（N是队列序号，K是ACE数量）

2. **环境变量控制**
   - `GPU_NUM_COMPUTE_RINGS`：限制可用的计算队列数量
   - 默认值可能小于32，限制了硬件能力的使用

3. **Runlist过度订阅问题** ⚠️ **关键发现**
   - dmesg显示："Runlist is getting oversubscribed due to too many queues"
   - 这是多进程性能问题的直接原因
   - 8个进程创建了太多队列，导致Runlist无法及时处理

### 解决方案方向

1. **减少队列数量**
   - 限制每个进程创建的Stream数量
   - 使用`GPU_NUM_COMPUTE_RINGS`控制总队列数
   - 避免Runlist过度订阅

2. **优化队列映射**
   - 确保队列均匀分布到不同ACE
   - 避免多个进程竞争同一个ACE
   - 实现进程到ACE的智能映射

3. **驱动层优化**
   - 优化Runlist调度算法
   - 提高调度器处理大量队列的能力
   - 减少队列排队时间

## 参考资料

1. [AMD OpenCL Multiple Command Queues](https://community.amd.com/t5/opencl/how-to-use-opencl-multiple-command-queues/td-p/599543)
2. [Linux Kernel AMDGPU User Mode Queues](https://docs.kernel.org/gpu/amdgpu/userq.html)
3. `ARCH_01_MI300_HARDWARE_QUEUE_ANALYSIS.md` - MI300硬件架构分析
4. `DRIVER_21_1PROC_VS_2PROC_COMPARISON.md` - 多进程性能对比分析

## 下一步行动

1. **查找HIP中的队列数量控制机制** ⚠️ **GPU_NUM_COMPUTE_RINGS在HIP中不存在**
2. **验证HIP Stream到ACE的映射关系**
3. **监控ACE利用率，确认负载分布**
4. **研究进程到ACE的绑定机制**
5. **研究HIP Stream优先级对队列分配的影响**

