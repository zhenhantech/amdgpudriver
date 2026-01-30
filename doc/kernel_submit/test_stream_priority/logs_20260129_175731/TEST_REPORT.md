# Stream Priority 测试报告（详细日志）

**生成时间**: Thu Jan 29 05:58:01 PM CST 2026  
**日志目录**: logs_20260129_175731

---

## 测试环境

```
操作系统: Linux hjbog-srdc-26.amd.com 5.10.134-19.1.al8.x86_64 #1 SMP Wed Jun 25 10:21:27 CST 2025 x86_64 x86_64 x86_64 GNU/Linux
ROCm 版本: HIP version: 7.0.51831-7c9236b16
GPU 设备:   Name:                    Intel(R) Xeon(R) Platinum 8480C    
```

---

## 日志配置

```bash
AMD_LOG_LEVEL        = 5 (最详细)
HIP_TRACE_API        = 1 (启用)
HIP_DB               = 0x1 (debug)
AMD_SERIALIZE_KERNEL = 0 (不串行化)
GPU_MAX_HW_QUEUES    = 8
```

---

## 测试结果

### test_concurrent
- 日志文件: `test_concurrent.log`
- Stream 创建: `stream_create.txt` (8 条)
- Queue 创建: `queue_create.txt` (0 条)
- Doorbell 信息: `doorbell.txt` (1 条)
- Priority 信息: `priority.txt` (28 条)
- Warnings: `warnings.txt` (0 条)

---

## 关键发现

### Stream 创建记录

```
:3:hip_stream.cpp           :293 : 20159561929 us: [pid:459769 tid: 0x7f8edc1c4400] [32m hipStreamCreateWithPriority ( 0x7fffea224e88, 0, -1 ) [0m
:3:hip_stream.cpp           :308 : 20159685100 us: [pid:459769 tid: 0x7f8edc1c4400] hipStreamCreateWithPriority: Returned hipSuccess : stream:0x1952020
:3:hip_stream.cpp           :293 : 20159685107 us: [pid:459769 tid: 0x7f8edc1c4400] [32m hipStreamCreateWithPriority ( 0x7fffea224e80, 0, 1 ) [0m
:3:hip_stream.cpp           :308 : 20159690220 us: [pid:459769 tid: 0x7f8edc1c4400] hipStreamCreateWithPriority: Returned hipSuccess : stream:0x20a2490
:3:hip_stream.cpp           :293 : 20159690227 us: [pid:459769 tid: 0x7f8edc1c4400] [32m hipStreamCreateWithPriority ( 0x7fffea224e78, 0, -1 ) [0m
:3:hip_stream.cpp           :308 : 20159695165 us: [pid:459769 tid: 0x7f8edc1c4400] hipStreamCreateWithPriority: Returned hipSuccess : stream:0x2099db0
:3:hip_stream.cpp           :293 : 20159695171 us: [pid:459769 tid: 0x7f8edc1c4400] [32m hipStreamCreateWithPriority ( 0x7fffea224e70, 0, 0 ) [0m
:3:hip_stream.cpp           :308 : 20159700277 us: [pid:459769 tid: 0x7f8edc1c4400] hipStreamCreateWithPriority: Returned hipSuccess : stream:0x208cd70
```

### Queue 创建记录

```

```

### Doorbell 信息

```
  2. cat /proc/459769/maps | grep doorbell
```

### Warnings (前 20 条)

```

```

---

## 分析建议

1. 检查 `stream_create.txt` 确认 4 个 Stream 创建
2. 检查 `queue_create.txt` 确认 4 个 Queue 创建
3. 检查 `doorbell.txt` 确认 4 个不同的 doorbell 地址
4. 检查 `warnings.txt` 分析是否有实质性问题

---

## 文件列表

```bash
ls -lh logs_20260129_175731/
```

```
total 128K
-rw-rw-r-- 1 zhehan zhehan   36 Jan 29 17:57 compile.log
-rw-rw-r-- 1 zhehan zhehan   43 Jan 29 17:57 doorbell.txt
-rw-rw-r-- 1 zhehan zhehan 4.1K Jan 29 17:57 priority.txt
-rw-rw-r-- 1 zhehan zhehan    0 Jan 29 17:57 queue_create.txt
-rw-rw-r-- 1 zhehan zhehan 1.2K Jan 29 17:57 stream_create.txt
-rw-rw-r-- 1 zhehan zhehan 107K Jan 29 17:57 test_concurrent.log
-rw-rw-r-- 1 zhehan zhehan    0 Jan 29 17:58 TEST_REPORT.md
-rw-rw-r-- 1 zhehan zhehan    0 Jan 29 17:57 warnings.txt
```

