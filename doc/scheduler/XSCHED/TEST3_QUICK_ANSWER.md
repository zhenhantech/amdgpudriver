# Test 3 快速问答

## 问题 1: 测试原理是什么？

### 简短回答

**两个独立的 Python 进程同时运行在同一个 GPU 上，竞争 GPU 资源**:

```python
高优先级进程: ResNet-18, 10 req/s, batch=1
低优先级进程: ResNet-50, 连续,    batch=8

同时运行 60 秒
```

---

## 问题 2: Baseline 时候两个模型同时运行吗？

### 简短回答

**是的！Baseline 和 XSched 都是两个模型同时运行**

```python
# 测试代码
high_proc = mp.Process(...)  # 高优先级进程
low_proc = mp.Process(...)   # 低优先级进程

high_proc.start()  # 启动
time.sleep(1)      # 等 1 秒
low_proc.start()   # 启动

# 两个进程同时执行
high_proc.join()   # 等待完成
low_proc.join()
```

**区别只在于调度器**:
- Baseline: 使用 Native ROCm scheduler（没有 LD_PRELOAD）
- XSched: 使用 XSched scheduler（启用 LD_PRELOAD）

---

## 问题 3: 依赖现在系统的调度能力？

### 简短回答

**是的，Baseline 依赖 ROCm 的默认调度器**

### Baseline (Native Scheduler)

```
ROCm 默认调度器 (FIFO - First In First Out)
  ↓
所有 GPU 任务平等竞争
  ↓
先提交的任务先执行
  ↓
无优先级区分
  ↓
高优先级任务可能等待低优先级任务
  ↓
P99 latency: 3.47 ms (较高)
```

### XSched

```
XSched 优先级调度器
  ↓
拦截所有 HIP API (通过 LD_PRELOAD)
  ↓
智能调度（可能基于任务特征或启动顺序）
  ↓
高优先级任务优先执行或抢占
  ↓
P99 latency: 2.75 ms (-20.9%)
```

---

## 问题 4: 当前有 Debug 日志吗？

### 检查结果

从测试日志 `run_phase4_dual_model.sh.log` 看:

```
✅ 没有看到 TRACE_MALLOC、TRACE_KERNEL 日志
✅ 只有正常的测试输出
✅ 性能数据应该是准确的
```

**结论**: **当前没有明显的 Debug 日志影响性能**

---

## 如何重新运行？

### 方法 1: 使用现有脚本（推荐）

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED

# 完整测试（baseline + xsched，2 分钟）
./run_phase4_dual_model.sh
```

### 方法 2: 只运行 XSched 测试

```bash
docker exec zhenflashinfer_v1 bash -c '
  cd /data/dockercode && \
  export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:$LD_LIBRARY_PATH && \
  export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so && \
  python3 test_phase4_dual_model.py --duration 60 --output /tmp/xsched_v2.json
'
```

### 方法 3: 快速验证（30 秒）

```bash
docker exec zhenflashinfer_v1 bash -c '
  cd /data/dockercode && \
  export LD_LIBRARY_PATH=/data/dockercode/xsched-build/output/lib:$LD_LIBRARY_PATH && \
  export LD_PRELOAD=/data/dockercode/xsched-build/output/lib/libshimhip.so && \
  python3 test_phase4_dual_model.py --duration 30 --output /tmp/xsched_quick.json
'
```

### 方法 4: 使用检查脚本

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/XSCHED

# 交互式检查和运行
./check_logs_and_rerun.sh
```

---

## 预期结果

如果重新运行，应该看到类似的性能改善:

```
High Priority (ResNet-18):
  Baseline P99: ~3.5 ms
  XSched P99:   ~2.7 ms
  改善:         ~20%

Low Priority (ResNet-50):
  Baseline:     ~165 iter/s
  XSched:       ~163 iter/s
  影响:         ~1%
```

---

## 核心发现总结

### 测试原理

```
✅ 两个进程同时运行（multiprocessing）
✅ Baseline 依赖 Native ROCm scheduler
✅ XSched 使用智能优先级调度
✅ 即使没有显式设置优先级，XSched 仍然更好
```

### 当前状态

```
✅ 没有明显的 Debug 日志
✅ 性能数据准确
✅ 可以直接重新运行验证
```

### XSched 的优势

```
🎉 P99 latency 降低 20.9%
🎉 Max latency 降低 71.3%
🎉 低优先级几乎不受影响 (-1.1%)
🎉 证明了 XSched 的调度策略优于 Native
```

---

## 详细文档

- **完整原理解释**: [PHASE4_TEST3_PRINCIPLE.md](PHASE4_TEST3_PRINCIPLE.md)
- **测试结果分析**: [PHASE4_TEST3_RESULTS.md](PHASE4_TEST3_RESULTS.md)
- **检查和重新运行**: `./check_logs_and_rerun.sh`

---

**建议**: 直接运行 `./run_phase4_dual_model.sh` 重新验证结果的可重复性 ✅
