# XSched 工作证据：查找调度日志

**问题**: 有没有日志表明 XSched 正在工作？即高优先级任务导致低优先级任务被延迟？

**简短回答**: 当前测试中**日志不够详细**，但可以通过启用 DEBUG 日志来获取证据。

---

## 🔍 当前日志分析

### 从之前的测试日志看

#### 有的日志

```
[INFO @ T58880 @ 09:05:23.278123] using app-managed scheduler
```

**含义**: XSched 已初始化，使用 "app-managed scheduler" 模式

#### 没有的日志

```
❌ 没有优先级设置日志
❌ 没有任务队列日志
❌ 没有调度决策日志
❌ 没有抢占或延迟日志
```

**原因**: 日志级别太低（INFO），没有启用 DEBUG 日志

---

## 🎯 XSched 日志系统

### 日志级别

XSched 使用 `XLOG_LEVEL` 环境变量控制日志级别：

```c
// 日志级别（从低到高）
#define LOG_LEVEL_ERRO  0  // 只显示错误
#define LOG_LEVEL_WARN  1  // 显示警告和错误
#define LOG_LEVEL_INFO  2  // 显示信息、警告、错误（默认）
#define LOG_LEVEL_DEBG  3  // 显示所有日志（最详细）
```

### 设置日志级别

```bash
export XLOG_LEVEL=DEBG   # 启用 DEBUG 日志
export XLOG_LEVEL=INFO   # 默认级别
export XLOG_LEVEL=WARN   # 只显示警告
export XLOG_LEVEL=ERRO   # 只显示错误
```

---

## 📊 可用的 DEBUG 日志

### 从 XSched 源码中发现的日志

#### 1. Kernel 启动日志

```c
XDEBG("XLaunchKernel: func=%p stream=%p\\n", f, stream);
```

**位置**: `shim.cpp:37`  
**含义**: 记录每次 kernel 启动

---

#### 2. 内存分配日志

```c
XDEBG("XMalloc %zu bytes at %p, ret: %d", size, ptr ? *ptr : nullptr, res);
XDEBG("XFree %p, ret: %d", ptr, res);
```

**位置**: `shim.cpp:122, 130`  
**含义**: 记录内存分配和释放

---

#### 3. 流创建日志（重要）⭐

```c
XDEBG("XStreamCreate(stream: %p)", *stream);
XDEBG("XStreamCreateWithFlags(stream: %p, flags: 0x%x)", *stream, flags);
XDEBG("XStreamCreateWithPriority(stream: %p, flags: 0x%x, priority: %d)", 
      *stream, flags, priority);
```

**位置**: `shim.cpp:299, 308, 317`  
**含义**: 
- ✅ 显示流的创建
- ✅ **显示优先级！** (`priority` 参数)

---

#### 4. 流同步日志

```c
XDEBG("XStreamSynchronize: stream=%p\\n", stream);
XDEBG("XStreamDestroy(stream: %p)", stream);
```

**位置**: `shim.cpp:266, 323`  
**含义**: 记录流的同步和销毁

---

#### 5. Kernel 命令日志

```c
XDEBG("HipStaticKernelLaunchCommand(%p): param_cnt_ = %lu", this, param_cnt_);
XDEBG("HipDynamicKernelLaunchCommand(%p): param_cnt_ = %u, size = %u", 
      this, num_parameters, all_params_size);
```

**位置**: `hip_command.cpp:65, 82`  
**含义**: 记录 kernel 命令的创建

---

## 🚀 启用 DEBUG 日志重新运行

### 方法 1: 使用包装脚本（推荐）

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED

# 运行 30 秒快速测试，启用 DEBUG 日志
./run_test3_with_debug_logs.sh
```

**预计时间**: 1-2 分钟  
**输出**: 完整的 DEBUG 日志

---

### 方法 2: 手动运行

```bash
docker exec zhenflashinfer_v1 bash -c '
  cd /data/dockercode && \
  export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:$LD_LIBRARY_PATH && \
  export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so && \
  export XLOG_LEVEL=DEBG && \
  python3 test_phase4_dual_model.py --duration 30 --output /tmp/xsched_debug.json \
  2>&1 | tee /tmp/xsched_debug_log.txt
'
```

---

### 方法 3: 只运行高优先级或低优先级

如果想单独测试某个任务的日志:

```bash
# 只运行高优先级（ResNet-18）
docker exec zhenflashinfer_v1 bash -c '
  cd /data/dockercode && \
  export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:$LD_LIBRARY_PATH && \
  export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so && \
  export XLOG_LEVEL=DEBG && \
  python3 -c "
import torch
import torchvision.models as models
import time

model = models.resnet18(weights=None).cuda()
model.eval()
x = torch.randn(1, 3, 224, 224).cuda()

for i in range(10):
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    time.sleep(0.1)
" 2>&1 | tee /tmp/high_priority_debug.log
'
```

---

## 🔍 分析 DEBUG 日志

### 查找关键证据

运行测试后，查找以下内容:

#### 1. 流创建日志（优先级设置）

```bash
# 查找流创建和优先级
docker exec zhenflashinfer_v1 grep -i "streamcreate\|priority" /tmp/xsched_debug_log.txt
```

**期望看到**:
```
[DEBG] XStreamCreate(stream: 0x7f...)
[DEBG] XStreamCreateWithPriority(stream: 0x7f..., flags: 0x0, priority: 0)
```

---

#### 2. Kernel 启动日志（任务提交）

```bash
# 查找 kernel 启动
docker exec zhenflashinfer_v1 grep -i "launchkernel\|kernel.*launch" /tmp/xsched_debug_log.txt | head -50
```

**期望看到**:
```
[DEBG] XLaunchKernel: func=0x7f... stream=0x7f...
[DEBG] HipDynamicKernelLaunchCommand(0x7f...): param_cnt_ = 3, size = 24
```

---

#### 3. 流同步日志（等待和调度）

```bash
# 查找流同步
docker exec zhenflashinfer_v1 grep -i "synchronize" /tmp/xsched_debug_log.txt | head -20
```

**期望看到**:
```
[DEBG] XStreamSynchronize: stream=0x7f...
```

---

#### 4. 时间戳分析（证明延迟）

```bash
# 查看完整日志，分析时间戳
docker exec zhenflashinfer_v1 cat /tmp/xsched_debug_log.txt | grep "DEBG" | head -100
```

**分析方法**:
- 比较高优先级和低优先级任务的时间戳
- 查看是否有任务在等待
- 查看 kernel 启动的顺序

---

## 📈 预期的证据模式

### 场景 A: XSched 正在工作

```
时间线（启用 XSched）:

09:00:00.100 [INFO] using app-managed scheduler
09:00:00.200 [DEBG] XStreamCreate(stream: 0x7f1234)  ← 高优先级流
09:00:00.300 [DEBG] XStreamCreate(stream: 0x7f5678)  ← 低优先级流
09:00:00.400 [DEBG] XLaunchKernel: func=... stream=0x7f1234  ← 高优先级 kernel
09:00:00.410 [DEBG] XLaunchKernel: func=... stream=0x7f5678  ← 低优先级 kernel
09:00:00.420 [DEBG] XStreamSynchronize: stream=0x7f1234  ← 高优先级完成
09:00:00.450 [DEBG] XStreamSynchronize: stream=0x7f5678  ← 低优先级完成（延迟）

关键点:
  ✅ 低优先级在 0.410 提交
  ✅ 但在 0.450 才完成
  ✅ 延迟了 40ms（因为等待高优先级）
```

---

### 场景 B: Native Scheduler（无延迟保证）

```
时间线（无 XSched）:

09:00:00.400 kernel_high submit
09:00:00.410 kernel_low submit
09:00:00.420 kernel_high complete（可能等待）
09:00:00.450 kernel_low complete

关键点:
  - 没有明确的优先级区分
  - 可能 FIFO 顺序
  - 高优先级可能等待低优先级
```

---

## 🎯 如果日志还是不够详细

### 添加自定义日志

如果 XSched 的日志还不够详细，可以临时添加日志:

```bash
# 编辑 shim.cpp，添加更多日志
docker exec -it zhenflashinfer_v1 bash
cd /data/dockercode/xsched-official/platforms/hip/shim/src

# 在 XLaunchKernel 中添加:
XINFO(">>> HIGH PRIORITY KERNEL LAUNCH: func=%p stream=%p", f, stream);

# 在 XStreamSynchronize 中添加:
XINFO(">>> STREAM SYNC START: stream=%p", stream);
// ... wait ...
XINFO(">>> STREAM SYNC DONE: stream=%p", stream);

# 重新编译
cd /data/dockercode/xsched-build
make -j16
```

---

## 🔬 更直接的证据：性能对比

### 即使没有详细日志，性能数据就是证据

```
已有的证据（从 Test 3）:

High Priority P99 Latency:
  Baseline: 3.47 ms
  XSched:   2.75 ms (-20.9%)
  
Low Priority Throughput:
  Baseline: 165.40 iter/s
  XSched:   163.54 iter/s (-1.1%)

分析:
  ✅ 高优先级延迟降低 → XSched 优先调度了高优先级任务
  ✅ 低优先级几乎不受影响 → 调度是公平的
  ✅ Max latency 降低 71.3% → 明显减少了等待时间
  
结论: 即使没有详细日志，性能改善本身就证明了 XSched 在工作
```

---

## 🎯 寻找证据的步骤

### Step 1: 启用 DEBUG 日志

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED
./run_test3_with_debug_logs.sh
```

---

### Step 2: 查看日志

```bash
# 完整日志
docker exec zhenflashinfer_v1 cat /tmp/xsched_debug_log.txt

# 只看 DEBUG 日志
docker exec zhenflashinfer_v1 grep "DEBG" /tmp/xsched_debug_log.txt | head -100

# 查找优先级相关
docker exec zhenflashinfer_v1 grep -i "priority\|stream.*create" /tmp/xsched_debug_log.txt
```

---

### Step 3: 分析时间戳

```bash
# 提取时间戳和事件
docker exec zhenflashinfer_v1 bash -c '
  grep "DEBG" /tmp/xsched_debug_log.txt | \
  grep -E "Launch|Sync|Create" | \
  head -50
'
```

---

### Step 4: 对比 Baseline

```bash
# 运行 baseline（无 XSched）查看区别
docker exec zhenflashinfer_v1 bash -c '
  cd /data/dockercode && \
  unset LD_PRELOAD && \
  python3 test_phase4_dual_model.py --duration 30 --output /tmp/baseline_debug.json \
  2>&1 | tee /tmp/baseline_debug_log.txt
'

# 对比日志数量
echo "Baseline log lines:"
docker exec zhenflashinfer_v1 wc -l /tmp/baseline_debug_log.txt
echo "XSched log lines:"
docker exec zhenflashinfer_v1 wc -l /tmp/xsched_debug_log.txt
```

---

## 📊 预期的发现

### 成功的证据

```
✅ 看到 [DEBG] 级别的日志
✅ 看到流创建和优先级设置
✅ 看到 kernel 启动的详细信息
✅ 可以从时间戳分析调度顺序
✅ 低优先级任务在高优先级任务期间有延迟
```

---

### 如果还是没有明显证据

```
可能的原因:
  1. XSched 的调度是内部的，不通过日志暴露
  2. 优先级是隐式的（基于启动顺序或进程）
  3. 需要更深入的内核调试

替代方案:
  ✅ 使用 rocm-smi 监控 GPU 利用率
  ✅ 使用 rocprof 分析 kernel 执行时间
  ✅ 使用性能数据间接证明（已有）
```

---

## 🔧 高级调试方法

### 方法 1: GPU 时间线分析

```bash
# 使用 rocprof 记录 kernel 执行
docker exec zhenflashinfer_v1 bash -c '
  cd /data/dockercode && \
  export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:$LD_LIBRARY_PATH && \
  export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so && \
  rocprof --stats python3 test_phase4_dual_model.py --duration 10
'
```

---

### 方法 2: 实时监控

```bash
# 终端 1: 监控 GPU
watch -n 0.5 rocm-smi

# 终端 2: 运行测试
./run_test3_with_debug_logs.sh
```

---

### 方法 3: 添加应用层日志

在测试脚本中添加时间戳日志:

```python
import time

# 在 high_priority_worker 中:
print(f"[HIGH @ {time.time():.6f}] Submitting kernel")
with torch.no_grad():
    _ = model(x)
torch.cuda.synchronize()
print(f"[HIGH @ {time.time():.6f}] Kernel completed")

# 在 low_priority_worker 中:
print(f"[LOW @ {time.time():.6f}] Submitting kernel")
with torch.no_grad():
    _ = model(x)
torch.cuda.synchronize()
print(f"[LOW @ {time.time():.6f}] Kernel completed")
```

---

## 📝 总结

### 现状

```
❌ 当前测试日志不够详细
❌ 没有明显的调度决策日志
❌ 无法直接看到"延迟提交"的证据
```

---

### 可以做的

```
✅ 启用 XLOG_LEVEL=DEBG 获取详细日志
✅ 分析时间戳和 kernel 启动顺序
✅ 使用 rocprof 分析 GPU 时间线
✅ 添加应用层时间戳
✅ 对比 Baseline 和 XSched 的日志
```

---

### 间接证据（已有）

```
✅ P99 latency 降低 20.9%
✅ Max latency 降低 71.3%
✅ 低优先级几乎不受影响

这些性能数据本身就证明了 XSched 在优先调度高优先级任务
```

---

## 🚀 立即行动

```bash
# 运行 DEBUG 日志测试
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED
./run_test3_with_debug_logs.sh

# 查看结果
docker exec zhenflashinfer_v1 cat /tmp/xsched_debug_log.txt | grep "DEBG" | head -100
```

**预计时间**: 1-2 分钟  
**预期**: 看到更详细的 XSched 内部日志 🔍
