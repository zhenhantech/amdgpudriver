# GEMM + ftrace 分析报告

**日期**: 2026-02-05  
**测试**: Mini GEMM (100次迭代)

---

## 📊 关键发现总结

### 1. 进程信息
- **容器内PID**: 157868
- **主机PID**: 3934101  
- **GPU**: 0xf7bc
- **Kernel提交**: 105次
- **Queue创建**: 2个 (qid=0, qid=1)

### 2. MQD生命周期 ✅

**创建** (177770.879738s):
```
KFD_MQD_CREATE: pid=3934101 gpu=0xf7bc qid=0 type=0 active=1 | Total_MQDs=1
```

**销毁** (177777.438854s & 177777.445126s):
```
KFD_MQD_DESTROY: qid=1 | Remaining_MQDs=1
KFD_MQD_DESTROY: qid=0 | Remaining_MQDs=0
```

### 3. Doorbell机制 ✅

- ✅ Doorbell已分配: `kfd_alloc_process_doorbells`
- ✅ Doorbell写入: `amdgpu_mm_wdoorbell64`
- ⚠️ 用户空间Doorbell提交不可见（直接MMIO）

### 4. KCQ使用 ❌

- **没有KCQ操作** - GEMM测试完全使用用户Queue

### 5. 完整调用链

```
ROCr: acquireQueue (177770.497126s)
  ↓ (133ms)
KFD: kfd_open (177770.631067s)
  ↓ (248ms)
KFD: kfd_ioctl_create_queue (177770.879589s)
  ↓ (20μs)
KFD: allocate_mqd (177770.879609s)
  ↓ (101μs)
KFD: init_mqd_v9_4_3 (177770.879710s)
  ↓ (28μs)
KFD: KFD_MQD_CREATE完成 ✅
```

**MQD操作延迟**: ~150μs（非常快！）

### 6. 核心洞察

1. **MQD是软硬件桥梁**: KFD分配MQD → 写入GPU HQD寄存器
2. **Doorbell高性能**: 用户空间直接MMIO，绕过内核
3. **Queue创建一次性**: 之后全部通过Doorbell提交
4. **2个Queue**: qid=0和qid=1（原因待分析）
5. **init_mqd调用4次**: 可能对应MI300X的4个XCC

---

详细分析见完整报告。
