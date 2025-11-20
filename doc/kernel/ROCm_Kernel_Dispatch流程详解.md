# ROCm Kernel Dispatch 流程详解

## 概述

本文档以**提交一个compute kernel**为例，详细剖析从应用层到硬件层的完整调用流程，涵盖：
- **HIP API**：用户接口
- **CLR/ROCclr**：设备抽象层
- **HSA Runtime**：AQL packet管理
- **ROCt**：Doorbell映射
- **KFD**：队列和硬件管理

---

## 核心概念

### AQL (Architected Queuing Language)

AQL是HSA标准定义的packet格式，用于控制GPU执行：
- **Packet结构**：64字节固定大小
- **类型**：Kernel Dispatch、Barrier、Agent Dispatch
- **队列**：环形缓冲区（Ring Buffer）
- **提交机制**：Doorbell寄存器

### Doorbell机制

**Doorbell**是一种低延迟的GPU通知机制：
- **用户态直接写入**：应用可直接写MMIO寄存器
- **无需系统调用**：避免内核上下文切换
- **高性能**：典型延迟 < 1μs
- **原理**：写doorbell = 告诉GPU "队列中有新工作"

---

## 五层架构总览

```
┌─────────────────────────────────────────────────────────┐
│  应用代码                                                │
│  myKernel<<<grid, block>>>(args);                       │
└──────────────────────┬──────────────────────────────────┘
                       │ 编译器展开为 hipLaunchKernel()
┌──────────────────────▼──────────────────────────────────┐
│  Layer 1: HIP API                                       │
│  clr/hipamd/src/hip_module.cpp                          │
│  • hipLaunchKernel()                                    │
│  • 参数验证                                              │
│  • Grid/Block尺寸检查                                    │
│  职责: 提供CUDA兼容的启动接口                            │
└──────────────────────┬──────────────────────────────────┘
                       │ 调用 ROCclr
┌──────────────────────▼──────────────────────────────────┐
│  Layer 2: CLR/ROCclr                                    │
│  clr/rocclr/device/rocm/rocvirtual.cpp                  │
│  • submitKernel()                                       │
│  • submitKernelInternal()                               │
│  • 构建 AQL Packet                                       │
│  • 写入队列环形缓冲区                                     │
│  职责: 管理命令队列，准备AQL packet                       │
└──────────────────────┬──────────────────────────────────┘
                       │ 写AQL + Ring Doorbell
┌──────────────────────▼──────────────────────────────────┐
│  Layer 3: HSA Runtime                                   │
│  rocr-runtime/runtime/hsa-runtime                       │
│  • hsa_queue_add_write_index()                          │
│  • 原子更新write_index                                   │
│  • 写入AQL packet到队列                                  │
│  • hsa_signal_store() -> Ring Doorbell                  │
│  职责: 实现HSA标准队列操作                                │
└──────────────────────┬──────────────────────────────────┘
                       │ 直接MMIO写入（无ioctl！）
┌──────────────────────▼──────────────────────────────────┐
│  Layer 4: Doorbell MMIO                                 │
│  • CPU直接写doorbell寄存器                               │
│  • *doorbell_ptr = write_index                          │
│  • PCIe写事务                                            │
│  职责: 通知GPU有新任务                                    │
└──────────────────────┬──────────────────────────────────┘
                       │ 硬件中断
┌──────────────────────▼──────────────────────────────────┐
│  Layer 5: KFD + GPU Hardware                            │
│  kfd/amdkfd/                                            │
│  • GPU读取队列                                           │
│  • 解析AQL packet                                        │
│  • CP (Command Processor) 调度                          │
│  • 启动compute unit执行                                  │
│  职责: 硬件调度和执行                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 详细流程分析

### Layer 1: HIP API

#### 入口点：`hipLaunchKernel()`

**文件**：`clr/hipamd/src/hip_module.cpp`

```cpp
// 应用代码
__global__ void myKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] *= 2.0f;
}

// 调用方式1：CUDA风格语法（编译器展开）
myKernel<<<gridDim, blockDim>>>(d_data, size);

// 调用方式2：显式API
hipLaunchKernel((void*)myKernel, gridDim, blockDim, 
                args, 0, stream);
```

#### HIP层实现

```cpp
// clr/hipamd/src/hip_platform.cpp
hipError_t ihipLaunchKernel(
    const void* hostFunction,  // 内核函数指针
    dim3 gridDim,              // Grid尺寸
    dim3 blockDim,             // Block尺寸
    void** args,               // 参数数组
    size_t sharedMemBytes,     // 共享内存大小
    hipStream_t stream,        // 执行流
    hipEvent_t startEvent,     // 开始事件
    hipEvent_t stopEvent,      // 结束事件
    int flags)                 // 标志位
{
  // 1. 验证stream和函数
  if (!hip::isValid(stream)) {
    return hipErrorInvalidValue;
  }
  if (hostFunction == nullptr) {
    return hipErrorInvalidDeviceFunction;
  }
  
  // 2. 获取设备ID
  int deviceId = hip::Stream::DeviceId(stream);
  
  // 3. 查找函数对象
  hipFunction_t func = nullptr;
  hipError_t hip_error = 
      PlatformState::instance().getStatFunc(&func, hostFunction, deviceId);
  
  // 4. 验证Grid和Block尺寸
  amd::HIPLaunchParams launch_params(
      gridDim.x, gridDim.y, gridDim.z, 
      blockDim.x, blockDim.y, blockDim.z, 
      sharedMemBytes);
  
  if (!launch_params.IsValidConfig()) {
    return hipErrorInvalidConfiguration;
  }
  
  // 5. 调用模块启动
  return ihipModuleLaunchKernel(
      func, launch_params, stream, args, 
      nullptr, startEvent, stopEvent, flags);
}
```

#### 关键验证

```cpp
// clr/hipamd/src/hip_module.cpp
hipError_t ihipLaunchKernel_validate(
    hipFunction_t f,
    amd::LaunchParams& launch_params,
    void** kernelParams,
    void** extra,
    int deviceId,
    uint32_t params)
{
  // 验证kernel对象
  hip::DeviceFunc* function = hip::DeviceFunc::asFunction(f);
  amd::Kernel* kernel = function->kernel();
  
  // 验证设备
  auto device = g_devices[deviceId]->devices()[0];
  
  // 验证workgroup大小
  if (launch_params.local_.product() > 
      device->info().maxWorkGroupSize_) {
    return hipErrorInvalidConfiguration;
  }
  
  // 验证共享内存大小
  if (launch_params.sharedMemBytes_ > 
      device->info().localMemSizePerCU_) {
    return hipErrorInvalidValue;
  }
  
  return hipSuccess;
}
```

#### HIP层职责总结

| 功能 | 说明 |
|-----|------|
| **接口转换** | CUDA风格 → ROCclr调用 |
| **参数验证** | Grid/Block尺寸、共享内存 |
| **错误检查** | Stream、函数、设备有效性 |
| **性能分析** | HIP_TRACE_API、时间戳记录 |
| **流管理** | 获取stream对应的命令队列 |

---

### Layer 2: CLR/ROCclr

#### 命令队列提交

**文件**：`clr/rocclr/device/rocm/rocvirtual.cpp`

```cpp
// ================================================================================================
void VirtualGPU::submitKernel(amd::NDRangeKernelCommand& vcmd) {
  // 处理cooperative groups（特殊情况）
  if (vcmd.cooperativeGroups()) {
    // 使用设备队列进行独占访问
    VirtualGPU* queue = dev().xferQueue();
    // ... 特殊处理
  } else {
    // 常规路径
    
    // 1. 获取排他锁
    amd::ScopedLock lock(execution());
    
    // 2. 开始性能分析
    profilingBegin(vcmd);
    
    // 3. 提交kernel到硬件
    if (!submitKernelInternal(
            vcmd.sizes(),           // Grid/Block尺寸
            vcmd.kernel(),          // Kernel对象
            vcmd.parameters(),      // 参数
            vcmd.event(),           // 事件
            vcmd.sharedMemBytes(),  // 共享内存
            &vcmd))                 // 命令对象
    {
      vcmd.setStatus(CL_INVALID_OPERATION);
      return;
    }
    
    // 4. 结束性能分析
    profilingEnd();
  }
}
```

#### 构建AQL Packet

```cpp
// ================================================================================================
bool VirtualGPU::submitKernelInternal(
    const amd::NDRangeContainer& sizes,
    const amd::Kernel& kernel,
    const_address parameters,
    void* event_handle,
    uint32_t sharedMemBytes,
    amd::NDRangeKernelCommand* vcmd)
{
  // 1. 获取设备kernel对象
  const device::Kernel& devKernel = 
      *(kernel.getDeviceKernel(dev()));
  
  // 2. 处理printf缓冲区
  bool printfEnabled = (devKernel.printfInfo().size() > 0);
  if (printfEnabled && !printfDbgHSA().init(*this, printfEnabled)) {
    LogError("Printf debug buffer initialization failed!");
    return false;
  }
  
  // 3. 处理内存依赖
  size_t ldsSize;
  if (!processMemObjectsHSA(kernel, parameters, nativeMem, ldsSize, 
                            imageBufferWrtBack, wrtBackImageBuffer)) {
    LogError("Wrong memory objects!");
    return false;
  }
  
  // 4. 分配AQL packet
  uint64_t index = acquireWriteIndex();
  hsa_kernel_dispatch_packet_t* aql_packet = 
      obtainPacketSlot(index);
  
  // 5. 填充AQL packet
  fillAqlPacket(aql_packet, devKernel, sizes, parameters, 
                sharedMemBytes, ldsSize);
  
  // 6. 设置completion signal
  aql_packet->completion_signal = 
      Barriers().ActiveSignal(kInitSignalValueOne, timestamp_, true);
  
  // 7. 使packet生效（原子写header）
  atomic::Store(&aql_packet->header, 
                createPacketHeader(),
                std::memory_order_release);
  
  // 8. Ring doorbell
  Hsa::hsa_signal_store_screlease(
      gpu_queue_->doorbell_signal, 
      index);
  
  return true;
}
```

#### AQL Packet结构填充

```cpp
void VirtualGPU::fillAqlPacket(
    hsa_kernel_dispatch_packet_t* aql,
    const device::Kernel& devKernel,
    const amd::NDRangeContainer& sizes,
    const_address parameters,
    uint32_t sharedMemBytes,
    size_t ldsSize)
{
  // 设置workgroup尺寸
  aql->workgroup_size_x = sizes.local()[0];
  aql->workgroup_size_y = sizes.local()[1];
  aql->workgroup_size_z = sizes.local()[2];
  
  // 设置grid尺寸
  aql->grid_size_x = sizes.global()[0];
  aql->grid_size_y = sizes.global()[1];
  aql->grid_size_z = sizes.global()[2];
  
  // 设置内存段大小
  aql->private_segment_size = devKernel.workGroupInfo()->privateMemSize_;
  aql->group_segment_size = ldsSize + sharedMemBytes;
  
  // 设置kernel对象（GPU代码地址）
  aql->kernel_object = devKernel.gpuAqlCode();
  
  // 设置kernel参数地址
  aql->kernarg_address = const_cast<void*>(
      reinterpret_cast<const void*>(parameters));
  
  // 初始化header为INVALID（稍后原子更新）
  aql->header = HSA_PACKET_TYPE_INVALID;
  
  // 设置保留字段
  aql->reserved0 = 0;
  aql->reserved1 = 0;
  aql->reserved2 = 0;
}
```

#### 写队列并Ring Doorbell

```cpp
void VirtualGPU::dispatchPacket(
    hsa_kernel_dispatch_packet_t* packet,
    uint64_t index)
{
  // 1. 确保队列槽可用（等待GPU消费）
  while ((index - Hsa::queue_load_read_index_scacquire(gpu_queue_)) 
         >= sw_queue_size) {
    amd::Os::yield();  // 让出CPU，等待GPU处理
  }
  
  // 2. 计算packet在环形缓冲区中的位置
  uint32_t mask = gpu_queue_->size - 1;
  uint32_t slot = index & mask;
  
  hsa_kernel_dispatch_packet_t* queue_slot = 
      reinterpret_cast<hsa_kernel_dispatch_packet_t*>(
          gpu_queue_->base_address) + slot;
  
  // 3. 先写packet body（header仍为INVALID）
  queue_slot->setup = packet->setup;
  queue_slot->workgroup_size_x = packet->workgroup_size_x;
  queue_slot->workgroup_size_y = packet->workgroup_size_y;
  queue_slot->workgroup_size_z = packet->workgroup_size_z;
  queue_slot->grid_size_x = packet->grid_size_x;
  queue_slot->grid_size_y = packet->grid_size_y;
  queue_slot->grid_size_z = packet->grid_size_z;
  queue_slot->private_segment_size = packet->private_segment_size;
  queue_slot->group_segment_size = packet->group_segment_size;
  queue_slot->kernel_object = packet->kernel_object;
  queue_slot->kernarg_address = packet->kernarg_address;
  queue_slot->completion_signal = packet->completion_signal;
  
  // 4. 内存屏障（确保body写入完成）
  std::atomic_thread_fence(std::memory_order_release);
  
  // 5. 原子写header（使packet生效）
  atomic::Store(&queue_slot->header, 
                createPacketHeader(),
                std::memory_order_release);
  
  // 6. Ring doorbell（通知GPU）
  Hsa::hsa_signal_store_screlease(
      gpu_queue_->doorbell_signal, 
      index);
}
```

#### 创建Packet Header

```cpp
uint16_t VirtualGPU::createPacketHeader() {
  uint16_t header = 0;
  
  // Packet type
  header |= (HSA_PACKET_TYPE_KERNEL_DISPATCH 
             << HSA_PACKET_HEADER_TYPE);
  
  // Barrier bit（如果需要等待前面的操作）
  header |= (0 << HSA_PACKET_HEADER_BARRIER);
  
  // Acquire fence scope
  header |= (HSA_FENCE_SCOPE_AGENT 
             << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE);
  
  // Release fence scope
  header |= (HSA_FENCE_SCOPE_AGENT 
             << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  
  return header;
}
```

#### ROCclr层职责总结

| 功能 | 说明 |
|-----|------|
| **队列管理** | 维护环形缓冲区、write/read index |
| **AQL构建** | 填充64字节dispatch packet |
| **内存依赖** | 处理buffer、image依赖关系 |
| **信号管理** | Completion signal、barrier signal |
| **资源跟踪** | Kernel代码、参数、内存对象 |

---

### Layer 3: HSA Runtime

#### 队列索引管理

**文件**：`rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp`

```cpp
// ================================================================================================
uint64_t AqlQueue::AddWriteIndexAcqRel(uint64_t value) {
  // 原子递增write_index，返回旧值
  return atomic::Add(&amd_queue_.write_dispatch_id, 
                     value,
                     std::memory_order_acq_rel);
}

// ================================================================================================
uint64_t AqlQueue::LoadReadIndexRelaxed() {
  // 读取GPU当前处理到的位置
  return atomic::Load(&amd_queue_.read_dispatch_id, 
                      std::memory_order_relaxed);
}
```

#### Doorbell写入

```cpp
// ================================================================================================
void AqlQueue::StoreRelaxed(hsa_signal_value_t value) {
  if (core::Runtime::runtime_singleton_->thunkLoader()->IsDXG()) {
    // Windows DXG模式：需要ioctl
    HSAKMT_CALL(hsaKmtQueueRingDoorbell(queue_id_));
  } else {
    // Linux模式：直接MMIO写入
    
    // 1. 内存屏障（确保packet已写入）
    _mm_sfence();
    
    // 2. 写doorbell寄存器
    *(signal_.hardware_doorbell_ptr) = uint64_t(value);
    
    // signal_分配在uncached内存，无需回读
  }
}

// ================================================================================================
void AqlQueue::StoreRelease(hsa_signal_value_t value) {
  // Release语义：确保之前的写操作可见
  std::atomic_thread_fence(std::memory_order_release);
  StoreRelaxed(value);
}
```

#### HSA Queue结构

```cpp
// HSA标准队列结构
typedef struct hsa_queue_s {
  hsa_queue_type32_t type;           // 队列类型
  uint32_t features;                 // 特性标志
  
  void* base_address;                // 环形缓冲区基地址
  hsa_signal_t doorbell_signal;      // Doorbell信号
  
  uint32_t size;                     // 队列大小（2的幂）
  uint32_t reserved1;
  
  uint64_t id;                       // 队列ID
} hsa_queue_t;

// AMD扩展队列结构
typedef struct amd_queue_s {
  hsa_queue_t hsa_queue;             // HSA标准部分
  
  uint64_t write_dispatch_id;        // 写索引
  uint64_t read_dispatch_id;         // 读索引
  
  uint64_t max_cu_id;                // 最大CU ID
  uint64_t max_wave_id;              // 最大Wave ID
  
  void* queue_properties;            // 队列属性
} amd_queue_t;
```

#### HSA Runtime层职责总结

| 功能 | 说明 |
|-----|------|
| **标准实现** | 实现HSA Runtime标准API |
| **原子操作** | write_index的原子递增 |
| **Doorbell** | 直接MMIO写入（Linux）或ioctl（Windows） |
| **队列管理** | 环形缓冲区地址计算 |
| **信号操作** | Signal create/wait/store |

---

### Layer 4: ROCt (Doorbell映射)

#### Doorbell内存映射

**文件**：`rocr-runtime/libhsakmt/src/queues.c`

```c
// ================================================================================================
static HSAKMT_STATUS map_doorbell(
    HSAuint32 NodeId,
    HSAuint32 gpu_id,
    HSAuint64 doorbell_mmap_offset)
{
  HSAKMT_STATUS status = HSAKMT_STATUS_SUCCESS;
  
  pthread_mutex_lock(&doorbells[NodeId].mutex);
  
  // 如果已经映射，直接返回
  if (doorbells[NodeId].size) {
    pthread_mutex_unlock(&doorbells[NodeId].mutex);
    return HSAKMT_STATUS_SUCCESS;
  }
  
  // 获取doorbell映射信息
  get_doorbell_map_info(NodeId, &doorbells[NodeId]);
  
  if (doorbells[NodeId].use_gpuvm) {
    // dGPU：在GPU虚拟地址空间中分配
    status = map_doorbell_dgpu(NodeId, gpu_id, doorbell_mmap_offset);
  } else {
    // APU：直接mmap
    status = map_doorbell_apu(NodeId, gpu_id, doorbell_mmap_offset);
  }
  
  pthread_mutex_unlock(&doorbells[NodeId].mutex);
  return status;
}

// ================================================================================================
static HSAKMT_STATUS map_doorbell_apu(
    HSAuint32 NodeId,
    HSAuint32 gpu_id,
    HSAuint64 doorbell_mmap_offset)
{
  // APU模式：直接mmap KFD提供的doorbell页
  void *ptr = mmap(
      0,                           // 地址由内核选择
      doorbells[NodeId].size,      // 大小
      PROT_READ|PROT_WRITE,        // 读写权限
      MAP_SHARED,                  // 共享映射
      hsakmt_kfd_fd,               // KFD文件描述符
      doorbell_mmap_offset);       // 偏移（由KFD提供）
  
  if (ptr == MAP_FAILED)
    return HSAKMT_STATUS_ERROR;
  
  doorbells[NodeId].mapping = ptr;
  return HSAKMT_STATUS_SUCCESS;
}

// ================================================================================================
static HSAKMT_STATUS map_doorbell_dgpu(
    HSAuint32 NodeId,
    HSAuint32 gpu_id,
    HSAuint64 doorbell_mmap_offset)
{
  // dGPU模式：在GPU地址空间分配doorbell
  void *ptr = hsakmt_fmm_allocate_doorbell(
      gpu_id,
      doorbells[NodeId].size,
      doorbell_mmap_offset);
  
  if (!ptr)
    return HSAKMT_STATUS_ERROR;
  
  // 映射到GPU（使GPU可以看到）
  if (hsakmt_fmm_map_to_gpu(ptr, doorbells[NodeId].size, NULL)) {
    hsakmt_fmm_release(ptr);
    return HSAKMT_STATUS_ERROR;
  }
  
  doorbells[NodeId].mapping = ptr;
  return HSAKMT_STATUS_SUCCESS;
}
```

#### Doorbell分配（创建队列时）

```c
// rocr-runtime/libhsakmt/src/queues.c
HSAKMT_STATUS hsaKmtCreateQueue(
    HSAuint32 NodeId,
    HSA_QUEUE_TYPE Type,
    HSAuint32 QueuePercentage,
    HSA_QUEUE_PRIORITY Priority,
    void* QueueAddress,
    HSAuint64 QueueSizeInBytes,
    HsaEvent* Event,
    HsaQueueResource* QueueResource)
{
  // 1. 准备ioctl参数
  struct kfd_ioctl_create_queue_args args = {0};
  args.gpu_id = get_gpu_id(NodeId);
  
  // 2. 队列类型
  if (Type == HSA_QUEUE_COMPUTE) {
    args.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE;
  } else if (Type == HSA_QUEUE_SDMA) {
    args.queue_type = KFD_IOC_QUEUE_TYPE_SDMA;
  }
  
  // 3. 队列缓冲区地址和大小
  args.ring_base_address = (uint64_t)QueueAddress;
  args.ring_size = QueueSizeInBytes;
  
  // 4. 写指针地址（GPU从这里读取当前写位置）
  args.write_pointer_address = (uint64_t)&write_ptr;
  args.read_pointer_address = (uint64_t)&read_ptr;
  
  // 5. 调用KFD创建队列
  int ret = ioctl(hsakmt_kfd_fd, 
                  AMDKFD_IOC_CREATE_QUEUE, 
                  &args);
  
  if (ret != 0)
    return HSAKMT_STATUS_ERROR;
  
  // 6. 返回doorbell信息
  QueueResource->QueueId = args.queue_id;
  QueueResource->Queue_DoorBell = 
      (uint64_t*)((char*)doorbells[NodeId].mapping + 
                  args.doorbell_offset);
  
  // 7. 映射doorbell（如果尚未映射）
  if (doorbells[NodeId].mapping == NULL) {
    map_doorbell(NodeId, args.gpu_id, args.doorbell_offset);
  }
  
  return HSAKMT_STATUS_SUCCESS;
}
```

#### ROCt层职责总结

| 功能 | 说明 |
|-----|------|
| **队列创建** | ioctl(AMDKFD_IOC_CREATE_QUEUE) |
| **Doorbell映射** | mmap doorbell页到用户空间 |
| **地址管理** | 管理doorbell、queue buffer地址 |
| **GPU/CPU映射** | 确保doorbell CPU/GPU都可访问 |

---

### Layer 5: KFD (Kernel Driver)

#### 队列创建（ioctl处理）

**文件**：`kfd/amdkfd/kfd_chardev.c`

```c
// ================================================================================================
static int kfd_ioctl_create_queue(
    struct file *filep,
    struct kfd_process *p,
    void *data)
{
  struct kfd_ioctl_create_queue_args *args = data;
  struct kfd_node *dev;
  struct queue_properties q_properties;
  struct kfd_process_device *pdd;
  unsigned int queue_id;
  uint32_t doorbell_offset_in_process = 0;
  
  // 1. 从用户参数设置队列属性
  set_queue_properties_from_user(&q_properties, args);
  
  // 2. 查找GPU设备
  dev = kfd_device_by_id(args->gpu_id);
  if (!dev)
    return -EINVAL;
  
  // 3. 获取进程设备数据
  pdd = kfd_process_device_data_by_id(p, args->gpu_id);
  if (!pdd)
    return -EINVAL;
  
  // 4. 绑定进程到设备
  pdd = kfd_bind_process_to_device(dev, p);
  if (IS_ERR(pdd))
    return -ESRCH;
  
  // 5. 分配doorbell（如果尚未分配）
  if (!pdd->qpd.proc_doorbells) {
    err = kfd_alloc_process_doorbells(dev->kfd, pdd);
    if (err)
      return err;
  }
  
  // 6. 获取队列缓冲区
  err = kfd_queue_acquire_buffers(pdd, &q_properties);
  if (err)
    return err;
  
  // 7. 创建队列
  err = pqm_create_queue(
      &p->pqm,                      // 进程队列管理器
      dev,                          // 设备
      &q_properties,                // 队列属性
      &queue_id,                    // 返回队列ID
      NULL, NULL, NULL,
      &doorbell_offset_in_process); // 返回doorbell偏移
  
  if (err != 0)
    return err;
  
  // 8. 返回结果给用户空间
  args->queue_id = queue_id;
  args->doorbell_offset = doorbell_offset_in_process;
  
  return 0;
}
```

#### 队列管理器创建队列

**文件**：`kfd/amdkfd/kfd_process_queue_manager.c`

```c
// ================================================================================================
int pqm_create_queue(
    struct process_queue_manager *pqm,
    struct kfd_node *dev,
    struct queue_properties *properties,
    unsigned int *qid,
    const struct kfd_criu_queue_priv_data *q_data,
    const void *restore_mqd,
    const void *restore_ctl_stack,
    uint32_t *p_doorbell_offset_in_process)
{
  struct kfd_process_device *pdd;
  struct queue *q;
  struct process_queue_node *pqn;
  enum kfd_queue_type type = properties->type;
  
  // 1. 获取进程设备数据
  pdd = kfd_get_process_device_data(dev, pqm->process);
  if (!pdd)
    return -1;
  
  // 2. 检查队列数量限制
  if (pdd->qpd.queue_count >= max_queues)
    return -ENOSPC;
  
  // 3. 分配队列slot
  retval = find_available_queue_slot(pqm, qid);
  if (retval != 0)
    return retval;
  
  // 4. 分配队列节点
  pqn = kzalloc(sizeof(*pqn), GFP_KERNEL);
  if (!pqn)
    return -ENOMEM;
  
  // 5. 根据类型创建队列
  switch (type) {
  case KFD_QUEUE_TYPE_COMPUTE:
    // 初始化用户队列
    retval = init_user_queue(pqm, dev, &q, properties, *qid);
    if (retval != 0)
      goto err_create_queue;
    
    pqn->q = q;
    pqn->kq = NULL;
    
    // 调用设备队列管理器创建队列
    retval = dev->dqm->ops.create_queue(
        dev->dqm,          // 设备队列管理器
        q,                 // 队列对象
        &pdd->qpd,         // 进程队列描述符
        q_data,
        restore_mqd,
        restore_ctl_stack);
    
    if (retval != 0)
      goto err_create_queue;
    
    break;
  
  case KFD_QUEUE_TYPE_SDMA:
    // SDMA队列...
    break;
  
  default:
    return -EINVAL;
  }
  
  // 6. 添加到进程队列列表
  pqn->q->properties.queue_id = *qid;
  list_add(&pqn->q_list, &pdd->qpd.queues_list);
  
  // 7. 返回doorbell偏移
  if (p_doorbell_offset_in_process)
    *p_doorbell_offset_in_process = q->properties.doorbell_off;
  
  return 0;
}
```

#### 设备队列管理器（DQM）

**文件**：`kfd/amdkfd/kfd_device_queue_manager.c`

```c
// 创建compute队列
static int create_compute_queue(
    struct device_queue_manager *dqm,
    struct queue *q,
    struct qcm_process_device *qpd)
{
  // 1. 分配MQD（Memory Queue Descriptor）
  retval = mqd_mgr->allocate_mqd(
      mqd_mgr,
      &q->mqd,
      &q->mqd_mem_obj);
  
  if (retval)
    return retval;
  
  // 2. 初始化MQD
  mqd_mgr->init_mqd(
      mqd_mgr,
      &q->mqd,
      &q->mqd_mem_obj,
      &q->gart_mqd_addr,
      &q->properties);
  
  // 3. 分配doorbell
  retval = allocate_doorbell(qpd, q);
  if (retval)
    goto out_deallocate_mqd;
  
  // 4. 设置GPU页表
  retval = amdgpu_amdkfd_gpuvm_map_memory_to_gpu(
      dqm->dev->adev,
      q->mqd_mem_obj->gtt_mem,
      qpd->vm);
  
  if (retval)
    goto out_deallocate_doorbell;
  
  // 5. 激活队列（写入HW寄存器）
  retval = dqm->ops.update_queue(dqm, q);
  if (retval)
    goto out_unmap;
  
  return 0;
}
```

#### MQD (Memory Queue Descriptor)

MQD是AMD GPU的队列描述符，包含队列的所有硬件配置：

```c
struct mqd {
  uint32_t queue_base_addr_lo;     // 队列基地址（低32位）
  uint32_t queue_base_addr_hi;     // 队列基地址（高32位）
  uint32_t queue_size;              // 队列大小
  uint32_t doorbell_ptr;            // Doorbell指针
  uint32_t read_ptr;                // 读指针
  uint32_t write_ptr;               // 写指针
  uint32_t queue_state;             // 队列状态
  
  // Compute shader相关
  uint32_t compute_static_thread_mgmt_se0;
  uint32_t compute_static_thread_mgmt_se1;
  uint32_t compute_tmpring_size;
  
  // 其他硬件配置...
};
```

#### GPU Command Processor (CP)

GPU的CP（Command Processor）负责：

1. **监控Doorbell**：检测write_index变化
2. **读取AQL Packet**：从队列环形缓冲区读取
3. **解析Packet**：提取grid尺寸、kernel地址、参数
4. **分配资源**：VGPR、SGPR、LDS
5. **调度Workgroup**：分配到CU（Compute Unit）
6. **执行**：启动shader执行
7. **更新read_index**：告知CPU已处理

#### KFD层职责总结

| 功能 | 说明 |
|-----|------|
| **队列创建** | 分配MQD、doorbell、queue buffer |
| **Doorbell分配** | 分配GPU MMIO区域 |
| **MQD管理** | 初始化硬件队列描述符 |
| **页表设置** | 映射队列内存到GPU地址空间 |
| **硬件编程** | 写入CP寄存器激活队列 |
| **进程隔离** | 每个进程独立的队列和doorbell |

---

## AQL Packet详解

### Packet结构（64字节）

```cpp
typedef struct hsa_kernel_dispatch_packet_s {
  // [0-1] Packet header（2字节）
  uint16_t header;
  // - [0-7]  : Type (KERNEL_DISPATCH = 2)
  // - [8]    : Barrier (0/1)
  // - [9-10] : Acquire fence scope
  // - [11-12]: Release fence scope
  // - [13-15]: Reserved
  
  // [2-3] Setup（2字节）
  uint16_t setup;
  // - [0-1]: Dimensions (1/2/3)
  // - [2-31]: Reserved
  
  // [4-5] Workgroup size X（2字节）
  uint16_t workgroup_size_x;
  
  // [6-7] Workgroup size Y（2字节）
  uint16_t workgroup_size_y;
  
  // [8-9] Workgroup size Z（2字节）
  uint16_t workgroup_size_z;
  
  // [10-11] Reserved（2字节）
  uint16_t reserved0;
  
  // [12-15] Grid size X（4字节）
  uint32_t grid_size_x;
  
  // [16-19] Grid size Y（4字节）
  uint32_t grid_size_y;
  
  // [20-23] Grid size Z（4字节）
  uint32_t grid_size_z;
  
  // [24-27] Private segment size（4字节）
  uint32_t private_segment_size;
  
  // [28-31] Group segment size（4字节）
  uint32_t group_segment_size;
  
  // [32-39] Kernel object（8字节）
  uint64_t kernel_object;         // GPU代码地址
  
  // [40-47] Kernarg address（8字节）
  void* kernarg_address;          // 参数缓冲区地址
  
  // [48-55] Reserved（8字节）
  uint64_t reserved2;
  
  // [56-63] Completion signal（8字节）
  hsa_signal_t completion_signal; // 完成信号
  
} hsa_kernel_dispatch_packet_t;
```

### Packet Header位域

```
15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
├──┴──┴──┼──┴──┼──┴──┼──┼──┴──┴──┴──┴──┴──┴──┴──┤
│  RSV  │ RLS │ ACQ │ B │       TYPE           │
└───────┴─────┴─────┴───┴──────────────────────┘

TYPE [0-7]:   Packet类型
  - 0: INVALID
  - 1: VENDOR_SPECIFIC
  - 2: KERNEL_DISPATCH  ← compute kernel
  - 3: BARRIER_AND
  - 4: AGENT_DISPATCH
  - 5: BARRIER_OR

B [8]:        Barrier位
  - 0: 不等待前面的packet
  - 1: 等待前面所有packet完成

ACQ [9-10]:   Acquire fence scope
  - 0: None
  - 1: Agent (GPU)
  - 2: System (CPU+GPU)

RLS [11-12]:  Release fence scope
  - 0: None
  - 1: Agent (GPU)
  - 2: System (CPU+GPU)

RSV [13-15]:  Reserved
```

---

## 完整时序图

```
应用层    HIP       ROCclr     HSA-RT    Doorbell   GPU-CP     CU
  │        │          │          │          │         │         │
  │ kernel<<<>>>()    │          │          │         │         │
  ├────────>│         │          │          │         │         │
  │         │ submitKernel()     │          │         │         │
  │         ├─────────>│         │          │         │         │
  │         │          │ 获取write_index     │         │         │
  │         │          ├─────────>│         │         │         │
  │         │          │<─────────┤         │         │         │
  │         │          │ index=N  │         │         │         │
  │         │          │          │         │         │         │
  │         │          │ 填充AQL packet      │         │         │
  │         │          │ (ring buffer[N])    │         │         │
  │         │          ├──────────┐          │         │         │
  │         │          │          │          │         │         │
  │         │          │<─────────┘          │         │         │
  │         │          │          │          │         │         │
  │         │          │ 写header（原子）     │         │         │
  │         │          ├──────────┐          │         │         │
  │         │          │<─────────┘          │         │         │
  │         │          │          │          │         │         │
  │         │          │ Ring doorbell       │         │         │
  │         │          ├──────────>│         │         │         │
  │         │          │           │ *db=N   │         │         │
  │         │          │           ├─────────>│        │         │
  │         │          │           │         │ 检测doorbell     │
  │         │          │           │         ├────────>│         │
  │         │          │           │         │ read=0, write=N  │
  │         │          │           │         │         │         │
  │         │          │           │         │ 读packet[0..N-1] │
  │         │          │           │         ├────────>│         │
  │         │          │           │         │<────────┤         │
  │         │          │           │         │         │         │
  │         │          │           │         │ 解析packet[0]    │
  │         │          │           │         ├────────>│         │
  │         │          │           │         │         │         │
  │         │          │           │         │ 分配资源         │
  │         │          │           │         │ (VGPR/LDS)       │
  │         │          │           │         ├────────>│         │
  │         │          │           │         │         │         │
  │         │          │           │         │ 调度workgroup    │
  │         │          │           │         ├─────────────────>│
  │         │          │           │         │                  │ 执行
  │         │          │           │         │                  ├───┐
  │         │          │           │         │                  │   │
  │         │          │           │         │                  │<──┘
  │         │          │           │         │<─────────────────┤
  │         │          │           │         │ 完成             │
  │         │          │           │         │                  │
  │         │          │           │         │ 写completion_signal
  │         │          │           │         ├────────>│         │
  │         │          │           │         │         │         │
  │         │          │           │         │ 更新read_index=N │
  │         │          │           │         ├────────>│         │
  │         │          │           │         │         │         │
  │         │ 检查signal │           │         │         │         │
  │         ├─────────>│           │         │         │         │
  │         │          │ hsa_signal_wait()   │         │         │
  │         │          ├───────────>│        │         │         │
  │         │          │            │ 检查signal.value │         │
  │         │          │<───────────┤ = 0 (完成)       │         │
  │<────────┤          │            │        │         │         │
  │ 返回     │          │            │        │         │         │
```

---

## 关键技术点

### 1. Doorbell的优势

**传统方式（ioctl）**：
```
用户态 → 系统调用 → 内核态 → 硬件
延迟：~5-10 μs
```

**Doorbell方式**：
```
用户态 → MMIO写 → 硬件
延迟：~0.5-1 μs
```

**关键差异**：
- ✅ 无需上下文切换
- ✅ 无需内核参与
- ✅ CPU直接写GPU寄存器
- ✅ 适合高频率小任务

### 2. 无锁环形缓冲区

```
队列结构：

 +---+---+---+---+---+---+---+---+
 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |  (size = 8)
 +---+---+---+---+---+---+---+---+
   ^               ^
   │               │
  read=2        write=5

可用空间 = size - (write - read) = 8 - (5-2) = 5
已用空间 = write - read = 3

环形索引计算：
  slot = write_index & (size - 1)
       = 5 & 7 = 5
```

**并发控制**：
- CPU写入：原子递增 write_index
- GPU读取：原子递增 read_index
- 无锁设计：避免锁开销

### 3. Packet生效的两步操作

```cpp
// 步骤1：写packet body（header=INVALID）
queue_slot->grid_size_x = 1024;
queue_slot->kernel_object = 0x7f...;
// ... 其他字段

// 步骤2：内存屏障
std::atomic_thread_fence(std::memory_order_release);

// 步骤3：原子写header（使packet生效）
atomic::Store(&queue_slot->header, 
              KERNEL_DISPATCH,
              std::memory_order_release);
```

**为什么这样做？**
- ✅ 避免GPU读到部分写入的packet
- ✅ Header=INVALID → GPU跳过该slot
- ✅ Header更新为DISPATCH → GPU开始处理
- ✅ 原子操作保证可见性

### 4. Signal机制

**作用**：
- 同步CPU和GPU
- 跟踪kernel完成状态
- 支持event依赖

**操作**：
```cpp
// 创建signal（初始值=1）
hsa_signal_create(1, 0, NULL, &signal);

// GPU完成时递减（写入completion_signal）
// 硬件自动：signal.value--

// CPU等待
hsa_signal_wait_scacquire(signal, 
                          HSA_SIGNAL_CONDITION_EQ, 
                          0,    // 等待值变为0
                          timeout, 
                          HSA_WAIT_STATE_BLOCKED);
```

### 5. 队列大小考虑

**队列过小**：
- ❌ CPU经常等待slot可用
- ❌ 吞吐量下降

**队列过大**：
- ❌ 内存占用高
- ❌ 延迟增加

**典型配置**：
- Compute queue: 128-1024 packets
- SDMA queue: 64-256 packets
- Size必须是2的幂（方便位运算）

---

## 性能分析

### 各层开销

| 层次 | 操作 | 典型延迟 |
|-----|-----|---------|
| HIP | 函数调用 | ~20 ns |
| ROCclr | AQL构建 | ~100 ns |
| HSA Runtime | 原子操作 | ~50 ns |
| Doorbell | MMIO写 | ~100 ns |
| **总CPU开销** | | **~300 ns** |
| PCIe传输 | 写传播 | ~500 ns |
| GPU CP | 唤醒+读packet | ~500 ns |
| **总延迟** | | **~1 μs** |

### 优化技巧

#### 1. 批量提交

```cpp
// 差：每次单独提交
for (int i = 0; i < 1000; i++) {
  kernel<<<1, 256>>>(data);  // 1000次doorbell写
}

// 好：使用stream并行
for (int i = 0; i < 1000; i++) {
  kernel<<<1, 256, 0, stream>>>(data);
}
hipStreamSynchronize(stream);  // 只等一次
```

#### 2. 减少同步

```cpp
// 差：频繁同步
kernel1<<<...>>>();
hipDeviceSynchronize();  // 阻塞等待
kernel2<<<...>>>();
hipDeviceSynchronize();

// 好：推迟同步
kernel1<<<...>>>();
kernel2<<<...>>>();
kernel3<<<...>>>();
hipDeviceSynchronize();  // 一次同步
```

#### 3. 使用异步API

```cpp
// 同步：CPU等待
hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);

// 异步：CPU继续执行
hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToHost, stream);
// ... 做其他工作
hipStreamSynchronize(stream);
```

---

## 调试方法

### 1. 启用API跟踪

```bash
export HIP_TRACE_API=1
./myapp

# 输出：
hipLaunchKernel(0x7f..., 256, 256, ...) = hipSuccess
  Grid: (256, 1, 1), Block: (256, 1, 1)
  Shared mem: 0 bytes
  Stream: 0x7f...
```

### 2. 查看队列状态

```bash
# ROCm工具
rocm-smi --showqueues

# 输出：
GPU[0]: gfx908
  Queue 0: Compute, Active, Depth: 5/128
  Queue 1: SDMA, Idle
```

### 3. GDB断点

```bash
gdb ./myapp

# 各层断点
(gdb) break hipLaunchKernel
(gdb) break VirtualGPU::submitKernel
(gdb) break AqlQueue::StoreRelease

(gdb) run

# 查看AQL packet内容
(gdb) p *(hsa_kernel_dispatch_packet_t*)queue_slot
$1 = {
  header = 8194,  # KERNEL_DISPATCH
  workgroup_size_x = 256,
  workgroup_size_y = 1,
  workgroup_size_z = 1,
  grid_size_x = 65536,
  grid_size_y = 1,
  grid_size_z = 1,
  kernel_object = 0x7f1234567890,
  kernarg_address = 0x7f9876543210,
  ...
}
```

### 4. ROCProfiler

```bash
rocprof --hip-trace ./myapp

# 生成results.csv，包含：
# - Kernel launch时间
# - 队列等待时间
# - Kernel执行时间
```

### 5. ROCgdb（GPU调试）

```bash
rocgdb ./myapp

(rocgdb) set scheduler-locking on
(rocgdb) break myKernel
(rocgdb) run

# 停在GPU kernel内部
(rocgdb) info threads   # 显示所有wavefront
(rocgdb) thread 1.1.1   # 切换到wavefront (1,1,1)
(rocgdb) print data[0]  # 查看GPU内存
```

---

## 总结

### 五层职责总结

```
┌───────────┬─────────────┬──────────────┬──────────────┐
│    层次    │    职责      │    接口       │    特点      │
├───────────┼─────────────┼──────────────┼──────────────┤
│ HIP       │ 用户接口    │ hipLaunchKernel │ CUDA兼容    │
│ ROCclr    │ AQL构建     │ submitKernel  │ 队列管理     │
│ HSA-RT    │ 标准实现    │ hsa_signal_*  │ 原子操作     │
│ Doorbell  │ 硬件通知    │ MMIO写        │ 无系统调用   │
│ KFD+GPU   │ 硬件执行    │ ioctl（初始化）│ CP调度      │
└───────────┴─────────────┴──────────────┴──────────────┘
```

### 关键流程

1. **应用** → `kernel<<<>>>()` → 编译器展开
2. **HIP** → 参数验证 → 调用ROCclr
3. **ROCclr** → 构建AQL packet → 写队列
4. **HSA Runtime** → 原子更新write_index → Ring doorbell
5. **Doorbell** → CPU直接MMIO写 → 通知GPU
6. **GPU CP** → 读取AQL → 调度执行

### 设计亮点

- ✅ **Doorbell**：用户态直接通知GPU，延迟 < 1μs
- ✅ **AQL标准**：64字节packet，硬件直接理解
- ✅ **无锁队列**：环形缓冲区 + 原子操作
- ✅ **分层清晰**：每层职责明确，易于维护
- ✅ **零拷贝**：Packet直接在共享内存，GPU读取

### 与CUDA对比

| 特性 | ROCm | CUDA |
|-----|------|------|
| **队列标准** | HSA AQL | NVIDIA专有 |
| **Doorbell** | 用户态MMIO | 用户态MMIO |
| **开源** | ✅ 完全开源 | ❌ 闭源 |
| **跨平台** | AMD GPU | NVIDIA GPU |
| **API兼容** | HIP ≈ CUDA | CUDA原生 |

---

## 参考资料

- **HSA标准**：[HSA Foundation](http://www.hsafoundation.com/)
- **AQL规范**：HSA Programmer's Reference Manual
- **ROCm文档**：[ROCm Documentation](https://rocm.docs.amd.com/)
- **源码**：本地 `amdgpudriver/` 目录

---

**文档版本**：1.0  
**创建日期**：2024年11月  
**目标读者**：ROCm驱动开发者、GPU系统工程师

