# AMD ROCm 详细调用流程与接口分析

## 完整调用链示例

### 示例1: hipLaunchKernel完整调用链

```c++
// ========== 1. 应用层 ==========
// app.cpp
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    hipLaunchKernelGGL(vectorAdd, dim3(blocks), dim3(threads), 0, stream, 
                       d_a, d_b, d_c, n);
}
```

#### 第1层: HIP API (clr/hipamd/src/hip_module.cpp)

```c++
// clr/hipamd/src/hip_module.cpp
hipError_t hipLaunchKernel(const void* hostFunction, 
                           dim3 gridDim, dim3 blockDim,
                           void** args, size_t sharedMem, 
                           hipStream_t stream) {
  HIP_INIT_API(hipLaunchKernel, ...);
  
  // 1. 获取设备函数对象
  hip::DeviceFunc* function = hip::DeviceFunc::asFunction(hostFunction);
  
  // 2. 获取amd::Kernel对象
  amd::Kernel* kernel = function->kernel();
  
  // 3. 获取或创建stream对应的amd::HostQueue
  hip::Stream* hip_stream = hip::getStream(stream);
  amd::HostQueue* queue = hip_stream->asHostQueue();
  
  // 4. 创建并提交NDRangeKernel命令
  amd::NDRangeContainer ndrange(3);
  ndrange[0] = amd::NDRange(gridDim.x * blockDim.x, blockDim.x);
  ndrange[1] = amd::NDRange(gridDim.y * blockDim.y, blockDim.y);
  ndrange[2] = amd::NDRange(gridDim.z * blockDim.z, blockDim.z);
  
  amd::Command* command = new amd::NDRangeKernelCommand(
      *queue, kernel, ndrange, args);
  
  command->enqueue();  // 提交到队列
  
  HIP_RETURN(hipSuccess);
}
```

#### 第2层: ROCclr Device Layer (clr/rocclr/device/rocm/)

```c++
// clr/rocclr/device/rocm/rocvirtual.cpp
bool VirtualGPU::submitKernel(
    const amd::NDRangeKernelCommand& vcmd) {
  
  // 1. 获取kernel对象
  const amd::Kernel& kernel = vcmd.kernel();
  roc::Kernel* rocKernel = static_cast<roc::Kernel*>(kernel.getDeviceKernel());
  
  // 2. 设置kernel参数
  void* kernarg_buffer = allocKernarg(kernel.parameters().size());
  memcpy(kernarg_buffer, vcmd.parameters(), kernel.parameters().size());
  
  // 3. 构建AQL Dispatch Packet
  hsa_kernel_dispatch_packet_t aql;
  memset(&aql, 0, sizeof(aql));
  
  // 设置工作组和网格尺寸
  aql.workgroup_size_x = vcmd.lws()[0];
  aql.workgroup_size_y = vcmd.lws()[1];
  aql.workgroup_size_z = vcmd.lws()[2];
  aql.grid_size_x = vcmd.gws()[0];
  aql.grid_size_y = vcmd.gws()[1];
  aql.grid_size_z = vcmd.gws()[2];
  
  // 设置kernel对象和参数地址
  aql.kernel_object = rocKernel->getKernelCodeHandle();
  aql.kernarg_address = reinterpret_cast<uint64_t>(kernarg_buffer);
  
  // 设置内存段大小
  aql.private_segment_size = rocKernel->workitemPrivateSegmentSize();
  aql.group_segment_size = rocKernel->workgroupGroupSegmentSize() + 
                           vcmd.sharedMemBytes();
  
  // 4. 创建完成信号
  hsa_signal_t completion_signal;
  Hsa::signal_create(1, 0, nullptr, &completion_signal);
  aql.completion_signal = completion_signal;
  
  // 5. 设置AQL包头（最后设置以原子激活包）
  aql.header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
               (1 << HSA_PACKET_HEADER_BARRIER) |
               (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
               (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
  aql.setup = 3;  // 3维
  
  // 6. 提交到HSA队列
  dispatchAqlPacket(&aql, gpu_queue_);
  
  // 7. 等待完成（异步则不等待）
  if (!async_mode) {
    Hsa::signal_wait_acquire(completion_signal, ...);
  }
  
  return true;
}

void VirtualGPU::dispatchAqlPacket(
    hsa_kernel_dispatch_packet_t* packet,
    hsa_queue_t* queue) {
  
  // 1. 获取写入索引
  uint64_t write_index = Hsa::queue_add_write_index_screlease(queue, 1);
  
  // 2. 计算队列槽位
  uint32_t slot = write_index % queue->size;
  hsa_kernel_dispatch_packet_t* queue_packet = 
      &((hsa_kernel_dispatch_packet_t*)queue->base_address)[slot];
  
  // 3. 先写入除header外的所有字段
  memcpy(((uint8_t*)queue_packet) + sizeof(uint16_t),
         ((uint8_t*)packet) + sizeof(uint16_t),
         sizeof(hsa_kernel_dispatch_packet_t) - sizeof(uint16_t));
  
  // 4. 原子写入header激活包
  __atomic_store_n((uint16_t*)queue_packet, packet->header, __ATOMIC_RELEASE);
  
  // 5. 写入doorbell通知GPU
  Hsa::signal_store_relaxed(queue->doorbell_signal, write_index);
}
```

#### 第3层: HSA Runtime (rocr-runtime/runtime/hsa-runtime/core/)

```c++
// rocr-runtime/runtime/hsa-runtime/core/runtime/queue.cpp
hsa_status_t Queue::Create(
    hsa_region_t region,
    uint32_t ring_size,
    hsa_queue_type32_t type,
    void (*callback)(hsa_status_t, hsa_queue_t*, void*),
    void* data,
    uint32_t private_segment_size,
    uint32_t group_segment_size,
    core::Queue** queue) {
  
  // 1. 分配队列环形缓冲区
  void* ring_buffer;
  hsa_memory_allocate(region, 
                      ring_size * sizeof(hsa_kernel_dispatch_packet_t),
                      &ring_buffer);
  
  // 2. 调用libhsakmt创建队列
  HSAuint32 queue_id;
  HsaQueueResource queue_resource;
  
  HSAKMT_STATUS status = hsaKmtCreateQueue(
      node_id_,
      HSA_QUEUE_COMPUTE_AQL,
      100,  // percentage
      HSA_QUEUE_PRIORITY_NORMAL,
      ring_buffer,
      ring_size,
      nullptr,  // event
      &queue_resource);
  
  if (status != HSAKMT_STATUS_SUCCESS) {
    return HSA_STATUS_ERROR;
  }
  
  // 3. 映射doorbell页
  void* doorbell_ptr = mmap(
      nullptr,
      queue_resource.QueueDoorBell_Size,
      PROT_READ | PROT_WRITE,
      MAP_SHARED,
      queue_resource.QueueDoorBell_FD,
      0);
  
  // 4. 创建doorbell信号
  hsa_signal_t doorbell_signal;
  doorbell_signal.handle = reinterpret_cast<uint64_t>(doorbell_ptr);
  
  // 5. 构建HSA队列对象
  *queue = new core::Queue();
  (*queue)->base_address = ring_buffer;
  (*queue)->size = ring_size;
  (*queue)->doorbell_signal = doorbell_signal;
  (*queue)->id = queue_id;
  
  return HSA_STATUS_SUCCESS;
}

// 队列操作
uint64_t hsa_queue_add_write_index_screlease(
    const hsa_queue_t* queue, uint64_t value) {
  // 原子递增write_index
  return __atomic_fetch_add(
      (uint64_t*)&queue->write_index, value, __ATOMIC_RELEASE);
}

void hsa_signal_store_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
  // 写入doorbell寄存器
  volatile uint32_t* doorbell_addr = 
      reinterpret_cast<volatile uint32_t*>(signal.handle);
  *doorbell_addr = (uint32_t)value;
}
```

#### 第4层: libhsakmt (rocr-runtime/libhsakmt/src/)

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
    HsaQueueResource* QueueResource) {
  
  // 1. 准备ioctl参数
  struct kfd_ioctl_create_queue_args args = {0};
  args.gpu_id = get_gpu_id(NodeId);
  args.queue_type = convert_queue_type(Type);
  args.queue_percentage = QueuePercentage;
  args.queue_priority = Priority;
  args.ring_base_address = (uint64_t)QueueAddress;
  args.ring_size = QueueSizeInBytes;
  args.write_pointer_address = (uint64_t)&write_ptr;
  args.read_pointer_address = (uint64_t)&read_ptr;
  
  if (Event) {
    args.eop_buffer_address = Event->EventData.HWData2;
    args.eop_buffer_size = Event->EventData.EventDataSizeInBytes;
  }
  
  // 2. 调用KFD ioctl
  int ret = ioctl(kfd_fd, AMDKFD_IOC_CREATE_QUEUE, &args);
  if (ret != 0) {
    return HSAKMT_STATUS_ERROR;
  }
  
  // 3. 返回队列资源
  QueueResource->QueueId = args.queue_id;
  QueueResource->QueueDoorBell_FD = kfd_fd;
  QueueResource->QueueDoorBell_Offset = args.doorbell_offset;
  QueueResource->QueueDoorBell_Size = PAGE_SIZE;
  
  return HSAKMT_STATUS_SUCCESS;
}
```

#### 第5层: KFD驱动 (kfd/amdkfd/)

```c
// kfd/amdkfd/kfd_chardev.c
static int kfd_ioctl_create_queue(
    struct file *filep,
    struct kfd_process *p,
    void *data) {
  
  struct kfd_ioctl_create_queue_args *args = data;
  struct kfd_node *dev;
  struct kfd_process_device *pdd;
  struct queue *q;
  int err;
  
  // 1. 查找设备
  dev = kfd_device_by_id(args->gpu_id);
  if (!dev)
    return -EINVAL;
  
  // 2. 获取进程设备上下文
  pdd = kfd_bind_process_to_device(dev, p);
  if (IS_ERR(pdd))
    return PTR_ERR(pdd);
  
  // 3. 创建队列对象
  err = kfd_create_queue(p, dev, pdd, args, &q);
  if (err)
    return err;
  
  // 4. 返回队列ID和doorbell偏移
  args->queue_id = q->properties.queue_id;
  args->doorbell_offset = q->properties.doorbell_off;
  
  return 0;
}

// kfd/amdkfd/kfd_queue.c
int kfd_create_queue(
    struct kfd_process *p,
    struct kfd_node *dev,
    struct kfd_process_device *pdd,
    struct kfd_ioctl_create_queue_args *args,
    struct queue **q_out) {
  
  struct queue_properties q_properties;
  struct queue *q;
  int err;
  
  // 1. 初始化队列属性
  memset(&q_properties, 0, sizeof(q_properties));
  q_properties.type = args->queue_type;
  q_properties.queue_size = args->ring_size;
  q_properties.priority = args->queue_priority;
  q_properties.queue_address = args->ring_base_address;
  q_properties.read_ptr = (uint32_t *)args->read_pointer_address;
  q_properties.write_ptr = (uint32_t *)args->write_pointer_address;
  
  // 2. 分配doorbell
  err = kfd_doorbell_init(pdd, &q_properties);
  if (err)
    return err;
  
  // 3. 创建队列（通过设备队列管理器）
  err = pqm_create_queue(p, dev, &pdd->qpd, &q_properties, &q);
  if (err)
    goto err_doorbell;
  
  // 4. 分配MQD (Memory Queue Descriptor)
  err = dev->dqm->ops.create_queue_mqd(dev->dqm, q, &q_properties);
  if (err)
    goto err_queue;
  
  // 5. 映射队列到硬件
  err = dev->dqm->ops.map_queues_cpsch(dev->dqm);
  if (err)
    goto err_mqd;
  
  *q_out = q;
  return 0;
  
err_mqd:
  dev->dqm->ops.destroy_queue_mqd(dev->dqm, q);
err_queue:
  pqm_destroy_queue(p, &pdd->qpd, q->properties.queue_id);
err_doorbell:
  kfd_doorbell_fini(pdd, &q_properties);
  return err;
}

// kfd/amdkfd/kfd_device_queue_manager.c
static int create_queue_mqd_v9(
    struct device_queue_manager *dqm,
    struct queue *q,
    struct queue_properties *qprop) {
  
  struct mqd_manager *mqd_mgr = dqm->mqd_mgrs[q->properties.type];
  struct amdgpu_device *adev = dev->adev;
  void *mqd;
  
  // 1. 分配MQD内存
  mqd = kzalloc(mqd_mgr->mqd_size, GFP_KERNEL);
  if (!mqd)
    return -ENOMEM;
  
  // 2. 初始化MQD
  mqd_mgr->init_mqd(mqd_mgr, &q->mqd, mqd, 
                    qprop->queue_address,
                    qprop);
  
  // 3. 配置硬件寄存器
  // MQD包含队列的所有硬件配置：
  // - 队列基地址
  // - 队列大小
  // - read/write指针地址
  // - doorbell地址
  // - 优先级
  // - 内存策略等
  
  q->mqd_mem_obj = mqd;
  
  return 0;
}
```

### 示例2: hipMemcpy完整调用链

```c++
// ========== 应用层 ==========
hipMemcpy(d_dst, d_src, size, hipMemcpyDeviceToDevice);
```

#### 第1层: HIP API

```c++
// clr/hipamd/src/hip_memory.cpp
hipError_t hipMemcpy(void* dst, const void* src, 
                     size_t sizeBytes, hipMemcpyKind kind) {
  HIP_INIT_API(hipMemcpy, dst, src, sizeBytes, kind);
  
  // 1. 确定拷贝类型
  hip::Stream* stream = hip::getStream(nullptr);  // 默认流
  
  if (kind == hipMemcpyDeviceToDevice) {
    // GPU到GPU拷贝
    return ihipMemcpy_validate(dst, src, sizeBytes, kind, *stream);
  }
  // ... 其他类型
}

static hipError_t ihipMemcpy_validate(
    void* dst, const void* src, size_t sizeBytes,
    hipMemcpyKind kind, hip::Stream& stream) {
  
  // 2. 获取内存对象
  size_t src_offset = 0, dst_offset = 0;
  amd::Memory* srcMemory = getMemoryObject(src, src_offset);
  amd::Memory* dstMemory = getMemoryObject(dst, dst_offset);
  
  // 3. 创建拷贝命令
  amd::Command* command = nullptr;
  
  if (kind == hipMemcpyDeviceToDevice) {
    command = new amd::CopyMemoryCommand(
        stream.asHostQueue(),
        CL_COMMAND_COPY_BUFFER,
        *srcMemory, *dstMemory,
        src_offset, dst_offset,
        sizeBytes);
  }
  
  // 4. 提交并等待完成
  command->enqueue();
  command->awaitCompletion();
  
  HIP_RETURN(hipSuccess);
}
```

#### 第2层: ROCclr

```c++
// clr/rocclr/device/rocm/rocblit.cpp
bool DmaBlitManager::copyBuffer(
    device::Memory& srcMemory,
    device::Memory& dstMemory,
    const amd::Coord3D& srcOrigin,
    const amd::Coord3D& dstOrigin,
    const amd::Coord3D& size) {
  
  roc::Memory& srcRocMem = static_cast<roc::Memory&>(srcMemory);
  roc::Memory& dstRocMem = static_cast<roc::Memory&>(dstMemory);
  
  // 1. 获取GPU虚拟地址
  void* src_addr = srcRocMem.getDeviceMemory() + srcOrigin[0];
  void* dst_addr = dstRocMem.getDeviceMemory() + dstOrigin[0];
  
  // 2. 选择拷贝方法
  if (prefer_sdma_) {
    // 使用SDMA引擎（异步DMA）
    return copyBufferViaSdma(src_addr, dst_addr, size[0]);
  } else {
    // 使用HSA内存拷贝
    hsa_status_t status = Hsa::memory_copy(dst_addr, src_addr, size[0]);
    return (status == HSA_STATUS_SUCCESS);
  }
}

bool DmaBlitManager::copyBufferViaSdma(
    void* src, void* dst, size_t size) {
  
  // 1. 获取SDMA队列
  hsa_queue_t* sdma_queue = getSdmaQueue();
  
  // 2. 构建SDMA命令包
  hsa_agent_dispatch_packet_t sdma_packet;
  memset(&sdma_packet, 0, sizeof(sdma_packet));
  
  // 设置SDMA拷贝命令
  sdma_packet.type = HSA_PACKET_TYPE_AGENT_DISPATCH;
  sdma_packet.arg[0] = reinterpret_cast<uint64_t>(src);
  sdma_packet.arg[1] = reinterpret_cast<uint64_t>(dst);
  sdma_packet.arg[2] = size;
  
  // 3. 提交到SDMA队列
  dispatchSdmaPacket(&sdma_packet, sdma_queue);
  
  return true;
}
```

#### 第3层: HSA Runtime

```c++
// rocr-runtime/runtime/hsa-runtime/core/runtime/runtime.cpp
hsa_status_t hsa_memory_copy(void* dst, const void* src, size_t size) {
  
  // 1. 查询内存属性
  hsa_amd_pointer_info_t src_info, dst_info;
  hsa_amd_pointer_info(src, &src_info, ...);
  hsa_amd_pointer_info(dst, &dst_info, ...);
  
  // 2. 根据内存类型选择拷贝路径
  if (src_info.type == HSA_EXT_POINTER_TYPE_HSA &&
      dst_info.type == HSA_EXT_POINTER_TYPE_HSA) {
    // GPU到GPU
    if (src_info.agentOwner.handle == dst_info.agentOwner.handle) {
      // 同一GPU，直接拷贝
      return copyOnSameGpu(dst, src, size);
    } else {
      // 跨GPU，使用P2P或通过系统内存
      return copyAcrossGpus(dst, src, size, 
                           src_info.agentOwner, 
                           dst_info.agentOwner);
    }
  }
  // ... 其他情况
}

static hsa_status_t copyOnSameGpu(void* dst, const void* src, size_t size) {
  
  // 1. 使用Blit内核或SDMA
  // 通常GPU内部拷贝很快，可以使用简单的内核
  
  // 2. 或者直接使用HSA信号同步的memcpy
  hsa_signal_t signal;
  hsa_signal_create(1, 0, nullptr, &signal);
  
  // 启动异步拷贝
  hsa_amd_memory_async_copy(dst, dst_agent, src, src_agent, size,
                            0, nullptr, signal);
  
  // 等待完成
  hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_LT, 1, ...);
  hsa_signal_destroy(signal);
  
  return HSA_STATUS_SUCCESS;
}
```

#### 第4层: libhsakmt - 不涉及

对于GPU内部的内存拷贝，通常不需要调用libhsakmt，因为：
- 队列已经创建
- 内存已经映射
- 直接通过队列提交AQL包即可

但如果需要新的内存操作（如pin内存），则会调用：

```c
// rocr-runtime/libhsakmt/src/memory.c
HSAKMT_STATUS hsaKmtMapMemoryToGPU(
    void* MemoryAddress,
    HSAuint64 MemorySizeInBytes,
    HSAuint64* AlternateVAGPU) {
  
  struct kfd_ioctl_map_memory_to_gpu_args args = {0};
  args.handle = get_mem_handle(MemoryAddress);
  
  int ret = ioctl(kfd_fd, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &args);
  
  if (AlternateVAGPU)
    *AlternateVAGPU = args.device_va_addr;
  
  return (ret == 0) ? HSAKMT_STATUS_SUCCESS : HSAKMT_STATUS_ERROR;
}
```

---

## 关键数据结构

### HIP层数据结构

```c++
// clr/hipamd/include/hip/hip_runtime_api.h

// 设备属性
struct hipDeviceProp_t {
  char name[256];
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int clockRate;
  int memoryClockRate;
  int memoryBusWidth;
  // ... 更多
};

// 流
typedef struct ihipStream_t* hipStream_t;

// 事件
typedef struct ihipEvent_t* hipEvent_t;

// 内存拷贝类型
typedef enum hipMemcpyKind {
  hipMemcpyHostToHost,
  hipMemcpyHostToDevice,
  hipMemcpyDeviceToHost,
  hipMemcpyDeviceToDevice,
  hipMemcpyDefault
} hipMemcpyKind;
```

### ROCclr层数据结构

```c++
// clr/rocclr/platform/command.hpp

namespace amd {

// 命令基类
class Command {
  enum Type {
    CL_COMMAND_NDRANGE_KERNEL,
    CL_COMMAND_COPY_BUFFER,
    CL_COMMAND_READ_BUFFER,
    CL_COMMAND_WRITE_BUFFER,
    // ... 更多
  };
  
  Type type_;
  HostQueue& queue_;
  Event* event_;
  // ...
  
  virtual void submit(device::VirtualDevice& device) = 0;
};

// 内核命令
class NDRangeKernelCommand : public Command {
  Kernel& kernel_;
  NDRangeContainer sizes_;  // 工作组和网格大小
  void* parameters_;        // 内核参数
  size_t sharedMemBytes_;   // 共享内存大小
  // ...
};

// 内存对象
class Memory {
  enum Type {
    Buffer,
    Image1D,
    Image2D,
    Image3D,
    Pipe
  };
  
  void* hostMemory_;        // 主机内存指针
  device::Memory* deviceMemory_;  // 设备内存对象
  size_t size_;
  // ...
};

} // namespace amd
```

### HSA Runtime层数据结构

```c
// rocr-runtime/runtime/hsa-runtime/inc/hsa.h

// HSA代理（设备）
typedef struct hsa_agent_s {
  uint64_t handle;
} hsa_agent_t;

// HSA队列
typedef struct hsa_queue_s {
  hsa_queue_type32_t type;        // 队列类型
  uint32_t features;               // 特性标志
  void* base_address;              // 环形缓冲区基地址
  hsa_signal_t doorbell_signal;    // Doorbell信号
  uint32_t size;                   // 队列大小
  uint32_t reserved1;
  uint64_t id;                     // 队列ID
} hsa_queue_t;

// HSA信号
typedef struct hsa_signal_s {
  uint64_t handle;
} hsa_signal_t;

// HSA内存区域
typedef struct hsa_region_s {
  uint64_t handle;
} hsa_region_t;

// AQL内核调度包
typedef struct hsa_kernel_dispatch_packet_s {
  uint16_t header;                    // 包头
  uint16_t setup;                     // 设置（维度等）
  uint16_t workgroup_size_x;          // 工作组X维度
  uint16_t workgroup_size_y;          // 工作组Y维度
  uint16_t workgroup_size_z;          // 工作组Z维度
  uint16_t reserved0;
  uint32_t grid_size_x;               // 网格X维度
  uint32_t grid_size_y;               // 网格Y维度
  uint32_t grid_size_z;               // 网格Z维度
  uint32_t private_segment_size;      // 私有内存大小
  uint32_t group_segment_size;        // 组内存大小
  uint64_t kernel_object;             // 内核对象地址
  uint64_t kernarg_address;           // 内核参数地址
  uint64_t reserved2;
  hsa_signal_t completion_signal;     // 完成信号
} hsa_kernel_dispatch_packet_t;
```

### libhsakmt层数据结构

```c
// rocr-runtime/libhsakmt/include/hsakmt/hsakmttypes.h

// 节点属性
typedef struct _HsaNodeProperties {
  HSAuint32 NumCPUCores;
  HSAuint32 NumFComputeCores;
  HSAuint32 NumMemoryBanks;
  HSAuint32 NumCaches;
  HSAuint32 NumIOLinks;
  HSAuint32 CComputeIdLo;
  HSAuint32 CComputeIdHi;
  HSAuint32 FComputeIdLo;
  HSAuint32 FComputeIdHi;
  // ... 更多
} HsaNodeProperties;

// 内存标志
typedef union {
  struct {
    unsigned int NonPaged : 1;       // 不可换页
    unsigned int CachePolicy : 2;    // 缓存策略
    unsigned int ReadOnly : 1;       // 只读
    unsigned int PageSize : 2;       // 页面大小
    unsigned int HostAccess : 1;     // 主机可访问
    unsigned int NoSubstitute : 1;   // 不替换
    unsigned int GDSMemory : 1;      // GDS内存
    unsigned int Scratch : 1;        // Scratch内存
    // ... 更多
  } ui32;
  HSAuint32 Value;
} HsaMemFlags;

// 队列资源
typedef struct _HsaQueueResource {
  HSAuint32 QueueId;                  // 队列ID
  HSAuint32 QueueDoorBell_FD;         // Doorbell文件描述符
  HSAuint64 QueueDoorBell_Offset;     // Doorbell偏移
  HSAuint64 QueueDoorBell_Size;       // Doorbell大小
  void* QueueRptrValue;               // 读指针地址
  void* QueueWptrValue;               // 写指针地址
} HsaQueueResource;
```

### KFD层数据结构

```c
// kfd/amdkfd/kfd_priv.h

// KFD进程
struct kfd_process {
  struct hlist_node kfd_processes;        // 进程链表
  struct mm_struct *mm;                   // 内存描述符
  struct mutex mutex;                     // 互斥锁
  struct kfd_process_device *pdds;        // 设备数组
  uint32_t n_pdds;                        // 设备数量
  struct process_queue_manager pqm;       // 队列管理器
  // ... 更多
};

// KFD进程设备
struct kfd_process_device {
  struct kfd_process *process;            // 所属进程
  struct kfd_node *dev;                   // 设备
  struct kfd_dev_apertures apertures;     // 地址空间
  struct kfd_vm *vm;                      // 虚拟内存
  struct qcm_process_device qpd;          // 队列上下文
  // ... 更多
};

// KFD队列
struct queue {
  struct list_head list;                  // 队列链表
  struct kfd_process *process;            // 所属进程
  struct kfd_node *device;                // 所属设备
  void *mqd;                              // MQD内存
  void *mqd_mem_obj;                      // MQD对象
  uint32_t queue_id;                      // 队列ID
  struct queue_properties properties;     // 队列属性
  // ... 更多
};

// 队列属性
struct queue_properties {
  enum kfd_queue_type type;               // 队列类型
  enum kfd_queue_format format;           // 队列格式
  unsigned int queue_size;                // 队列大小
  uint64_t queue_address;                 // 队列地址
  uint64_t doorbell_off;                  // Doorbell偏移
  uint32_t *read_ptr;                     // 读指针
  uint32_t *write_ptr;                    // 写指针
  // ... 更多
};

// 设备队列管理器
struct device_queue_manager {
  struct kfd_node *dev;                   // 设备
  struct list_head queues;                // 队列列表
  struct process_queue_manager *active_pqm; // 当前活跃进程队列管理器
  struct packet_manager packets;          // 包管理器
  struct device_queue_manager_ops ops;    // 操作函数表
  // ... 更多
};
```

---

## 同步机制

### HSA信号 (Signals)

```c
// 创建信号
hsa_signal_t signal;
hsa_signal_create(1,  // 初始值
                 0,   // consumer数量（0表示任意）
                 NULL, // consumer agents
                 &signal);

// 等待信号
hsa_signal_value_t value = hsa_signal_wait_acquire(
    signal,
    HSA_SIGNAL_CONDITION_LT,  // 条件：小于
    1,                        // 比较值
    UINT64_MAX,               // 超时（无限）
    HSA_WAIT_STATE_BLOCKED);  // 等待状态

// 发信号
hsa_signal_store_release(signal, 0);

// 销毁信号
hsa_signal_destroy(signal);
```

### HIP事件 (Events)

```c++
// 创建事件
hipEvent_t event;
hipEventCreate(&event);

// 记录事件
hipEventRecord(event, stream);

// 等待事件
hipEventSynchronize(event);

// 查询事件状态
hipError_t status = hipEventQuery(event);
if (status == hipSuccess) {
  // 事件已完成
}

// 销毁事件
hipEventDestroy(event);
```

### HIP流同步

```c++
// 同步流
hipStreamSynchronize(stream);

// 等待流中事件
hipStreamWaitEvent(stream, event, 0);

// 设置流回调
hipStreamAddCallback(stream, callback, userData, 0);
```

### KFD事件

```c
// kfd/amdkfd/kfd_events.c

// 创建事件
struct kfd_event *kfd_event_create(
    struct kfd_process *p,
    uint32_t event_type,
    bool auto_reset,
    uint32_t node_id);

// 设置事件
int kfd_event_set(struct kfd_event *ev);

// 等待事件
int kfd_wait_on_events(
    struct kfd_process *p,
    uint32_t num_events,
    void __user *data,
    bool all,
    uint32_t timeout);
```

---

## 性能优化技术

### 1. Doorbell机制

**传统方式**：
```
用户态 → syscall → 内核 → 通知GPU
延迟：~1-2μs
```

**Doorbell方式**：
```
用户态 → 直接写MMIO寄存器 → GPU立即感知
延迟：~100ns
```

**实现**：
```c++
// 写入doorbell
volatile uint32_t* doorbell = 
    (uint32_t*)queue->doorbell_signal.handle;
*doorbell = write_index;  // 一次写操作，GPU立即看到
```

### 2. User-mode Queues

- 应用直接写入队列环形缓冲区
- 无需内核介入
- GPU Command Processor直接读取

### 3. 异步执行

```c++
// HIP异步操作
hipMemcpyAsync(dst, src, size, kind, stream);
hipLaunchKernel(kernel, ..., stream);
// 不等待，立即返回

// 后续同步
hipStreamSynchronize(stream);
```

### 4. 多流并发

```c++
// 创建多个流
hipStream_t stream1, stream2;
hipStreamCreate(&stream1);
hipStreamCreate(&stream2);

// 在不同流上提交任务
hipMemcpyAsync(d_a, h_a, size, ..., stream1);
hipMemcpyAsync(d_b, h_b, size, ..., stream2);
hipLaunchKernel(kernel1, ..., stream1);
hipLaunchKernel(kernel2, ..., stream2);

// 并发执行
```

### 5. Unified Memory (SVM)

```c++
// 分配统一内存
void* ptr;
hipMallocManaged(&ptr, size);

// CPU和GPU都可以直接访问
ptr[0] = 42;  // CPU写入
kernel<<<...>>>(ptr);  // GPU读取

// 自动迁移
- 页错误触发迁移
- KFD驱动处理迁移
- 对应用透明
```

---

## 调试工具和接口

### ROCm调试工具

```bash
# rocm-smi: GPU监控
rocm-smi

# rocprof: 性能分析
rocprof --stats ./my_app

# rocgdb: GPU调试器
rocgdb ./my_app
```

### KFD Debugfs接口

```bash
# 查看设备信息
cat /sys/kernel/debug/kfd/topology/nodes/*/properties

# 查看进程信息
cat /sys/kernel/debug/kfd/process/*/pasid

# 查看队列信息
cat /sys/kernel/debug/kfd/process/*/queues
```

### HSA工具接口

```c
// HSA工具API（用于profiler）
#include "hsa_api_trace.h"

// 注册API回调
CoreApiTable.hsa_queue_create_fn = my_hsa_queue_create;
```

---

## 错误处理

### HIP错误

```c++
hipError_t err = hipMalloc(&ptr, size);
if (err != hipSuccess) {
  printf("Error: %s\n", hipGetErrorString(err));
  return -1;
}

// 获取最后一个错误
hipError_t last_err = hipGetLastError();
```

### HSA错误

```c
hsa_status_t status = hsa_init();
if (status != HSA_STATUS_SUCCESS) {
  const char* error_string;
  hsa_status_string(status, &error_string);
  fprintf(stderr, "HSA Error: %s\n", error_string);
}
```

### KFD错误

```c
// ioctl返回值
int ret = ioctl(kfd_fd, AMDKFD_IOC_CREATE_QUEUE, &args);
if (ret != 0) {
  perror("KFD ioctl failed");
  // errno包含错误码
}
```

---

## 总结

AMD ROCm软件栈通过精心设计的分层架构，实现了：

1. **高性能**：
   - Doorbell机制实现低延迟提交
   - 用户态队列避免系统调用开销
   - AQL标准化包格式
   - 异步执行和多流并发

2. **可移植性**：
   - HIP提供类CUDA接口
   - HSA标准化底层接口
   - ROCclr提供统一设备抽象

3. **完整功能**：
   - 内核执行
   - 内存管理（VRAM/GTT/System/SVM）
   - 同步机制（信号/事件）
   - 调试和profiling支持

4. **清晰分层**：
   - 每层职责明确
   - 接口定义清晰
   - 便于维护和扩展

这个架构使得AMD GPU能够高效支持各种GPU计算工作负载。

