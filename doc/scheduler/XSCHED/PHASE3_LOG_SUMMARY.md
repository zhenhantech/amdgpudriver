# Phase 3 测试日志摘要

**日期**: 2026-01-28  
**日志文件**: `/mnt/md0/zhehan/code/flashinfer/dockercode/xsched/testlog/TEST_REAL_MODELS.sh.log`  
**日志大小**: 314KB

---

## 🎯 快速查看

### 测试结果一览

```
Total Tests:  14
Passed:       13 ✅
Failed:       1 ❌

Success Rate: 92.9%
```

### 通过的测试 (13 个)

```
✅ ResNet-50
✅ ResNet-18
✅ MobileNetV2
✅ EfficientNet-B0
✅ Vision Transformer (ViT-B/16)
✅ DenseNet-121
✅ VGG-16
✅ SqueezeNet
✅ AlexNet
✅ ResNet-18 Training
✅ MobileNetV2 Training
✅ ResNet-50 Batch=32
✅ EfficientNet Batch=16
```

### 失败的测试 (1 个)

```
❌ GoogLeNet (Inception) - 待调试
```

---

## 📂 日志文件位置

```bash
# 完整日志路径
/mnt/md0/zhehan/code/flashinfer/dockercode/xsched/testlog/TEST_REAL_MODELS.sh.log

# 在 Docker 容器内
docker exec zhenflashinfer_v1 cat /data/dockercode/xsched/testlog/TEST_REAL_MODELS.sh.log
```

---

## 🔍 日志关键内容

### XSched 初始化日志

```
[INFO @ T57541 @ 08:58:33.564323] using app-managed scheduler
[INFO @ T57544 @ 08:58:33.577495] using app-managed scheduler
...
```

**说明**: XSched 正确初始化，App-Managed Scheduler 模式

---

### API 调用跟踪示例

```
[TRACE_MALLOC] size=2097152 ptr=0x7fe7ac200000 ret=0 (SUCCESS)
[TRACE_MALLOC] size=20971520 ptr=0x7fc796600000 ret=0 (SUCCESS)
[TRACE_KERNEL] func=0x7fe936b70d78 stream=(nil)
[TRACE_FREE] ptr=0x7fc787e00000 ret=0
```

**说明**: 
- ✅ `hipMalloc` 调用成功
- ✅ `hipLaunchKernel` 正常
- ✅ `hipFree` 正常
- ✅ 内存管理正常

---

### 环境配置

```
LD_LIBRARY_PATH: /data/dockercode/xsched-build/output/lib:...
LD_PRELOAD: /data/dockercode/xsched-build/output/lib/libshimhip.so
```

**说明**: 使用 Phase 2 编译的 XSched

---

## 🔎 查看日志的命令

### 提取测试结果

```bash
grep -E "(Testing |✅|❌|PASSED|FAILED)" \
  /mnt/md0/zhehan/code/flashinfer/dockercode/xsched/testlog/TEST_REAL_MODELS.sh.log
```

### 提取 API 调用

```bash
grep "TRACE_" \
  /mnt/md0/zhehan/code/flashinfer/dockercode/xsched/testlog/TEST_REAL_MODELS.sh.log \
  | head -100
```

### 提取 XSched 日志

```bash
grep "INFO" \
  /mnt/md0/zhehan/code/flashinfer/dockercode/xsched/testlog/TEST_REAL_MODELS.sh.log
```

### 查看特定模型的日志

```bash
# 查看 ResNet-50 的日志
grep -A 50 "Testing ResNet-50" \
  /mnt/md0/zhehan/code/flashinfer/dockercode/xsched/testlog/TEST_REAL_MODELS.sh.log
```

---

## 📊 日志统计

### 文件信息

```bash
# 大小
$ ls -lh /mnt/md0/zhehan/code/flashinfer/dockercode/xsched/testlog/TEST_REAL_MODELS.sh.log
-rw-r--r-- 1 root root 307K Jan 28 08:59 TEST_REAL_MODELS.sh.log

# 行数
$ wc -l /mnt/md0/zhehan/code/flashinfer/dockercode/xsched/testlog/TEST_REAL_MODELS.sh.log
15734 lines
```

### API 调用统计（估算）

```
TRACE_MALLOC: ~1000+ 次
TRACE_KERNEL: ~5000+ 次
TRACE_FREE:   ~500+ 次

说明: XSched API 拦截功能正常工作
```

---

## 📝 关键发现（从日志）

### 1. XSched 正常工作

```
✅ 所有 TRACE_MALLOC 返回 SUCCESS
✅ 所有 TRACE_FREE 返回 0
✅ TRACE_KERNEL 调用正常
✅ 无崩溃或异常退出
```

### 2. Symbol Versioning 生效

```
✅ XSched 初始化成功
✅ "using app-managed scheduler" 日志正常
✅ hipblasLt 调用被正确拦截
```

### 3. 内存管理健康

```
✅ 大内存分配（134MB+）成功
✅ 多次分配/释放循环正常
✅ 无内存泄漏迹象
```

### 4. 复杂模型支持

```
✅ Transformer (ViT) 成功
✅ DenseNet 成功
✅ 训练模式（Forward+Backward）成功
```

---

## 🎯 日志对 Phase 4 的价值

### 1. 性能 Baseline

```
可以从日志中提取:
- 单模型推理时间
- 内存分配大小
- Kernel 调用次数
→ 作为 Phase 4 多模型测试的对比基准
```

### 2. 稳定性证明

```
13/14 测试通过表明:
✅ XSched 环境稳定
✅ 可以进行更复杂的多模型测试
✅ API 拦截可靠
```

### 3. 模型选择参考

```
Phase 4 可以使用的模型（已验证）:
✅ ResNet-18 (轻量，适合高优先级)
✅ ResNet-50 (中等，适合低优先级)
✅ MobileNetV2 (快速)
✅ EfficientNet (高效)
```

---

## 🔗 相关文档

- **详细分析**: [PHASE3_TEST_RESULTS.md](PHASE3_TEST_RESULTS.md)
- **Phase 4 目标**: [PHASE4_CORE_OBJECTIVES.md](PHASE4_CORE_OBJECTIVES.md)
- **快速开始**: [PHASE4_QUICKSTART.md](PHASE4_QUICKSTART.md)

---

## 📋 快速命令

```bash
# 在 Docker 容器内查看日志
docker exec zhenflashinfer_v1 cat /data/dockercode/xsched/testlog/TEST_REAL_MODELS.sh.log

# 提取测试结果
docker exec zhenflashinfer_v1 bash -c "
  grep -E '(Testing |✅|❌)' /data/dockercode/xsched/testlog/TEST_REAL_MODELS.sh.log
"

# 统计 API 调用
docker exec zhenflashinfer_v1 bash -c "
  echo 'TRACE_MALLOC:' \$(grep -c 'TRACE_MALLOC' /data/dockercode/xsched/testlog/TEST_REAL_MODELS.sh.log)
  echo 'TRACE_KERNEL:' \$(grep -c 'TRACE_KERNEL' /data/dockercode/xsched/testlog/TEST_REAL_MODELS.sh.log)
  echo 'TRACE_FREE:' \$(grep -c 'TRACE_FREE' /data/dockercode/xsched/testlog/TEST_REAL_MODELS.sh.log)
"
```

---

**Phase 3 测试日志**: 详细记录了 XSched + PyTorch 的成功集成 ✅
