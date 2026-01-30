# 快速开始 - Stream Priority 测试

5 分钟验证每个 Stream 都有独立的 Queue！

## 🚀 方法 1: 一键运行（最简单）

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/test_stream_priority

./run_test.sh
```

**预期输出**:
```
═══════════════════════════════════════════════════════════
检查依赖
═══════════════════════════════════════════════════════════
✅ hipcc 已安装
✅ rocprofv3 已安装

═══════════════════════════════════════════════════════════
编译测试程序
═══════════════════════════════════════════════════════════
✅ 所有程序编译成功

═══════════════════════════════════════════════════════════
测试 1: 单进程 4 个 Stream
═══════════════════════════════════════════════════════════
...
✅ [应用 A] Stream-1 (HIGH):   0x7f1234567890
✅ [应用 A] Stream-2 (LOW):    0x7f1234567a00
✅ [应用 B] Stream-3 (HIGH):   0x7f1234567b10
✅ [应用 B] Stream-4 (NORMAL): 0x7f1234567c20
...
✅ 所有 4 个 Stream 地址唯一 → 4 个独立的 Stream 对象
...
```

---

## 🚀 方法 2: 手动运行（更灵活）

### 步骤 1: 编译

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/test_stream_priority

make all
```

### 步骤 2: 运行测试

```bash
# 单进程测试（推荐）
./test_concurrent
```

**观察点**:
1. 看到 4 个不同的 Stream 地址 ✅
2. 看到 4 个不同的优先级 ✅
3. 看到所有 Stream 都能提交 kernel ✅

---

## 🔍 方法 3: 使用 rocprof 追踪（深入分析）

```bash
# 追踪 Queue 信息
rocprofv3 --hip-trace ./test_concurrent

# 查看生成的 CSV 文件
ls -lh results*/

# 查找 Queue 相关信息
grep -i "queue\|stream" results*/*.csv | head -20
```

**预期看到**:
- 4 个 `hipStreamCreateWithPriority` 调用
- 4 个不同的 Queue ID
- 4 个独立的时间线

---

## 🔍 方法 4: 监控内核消息（需要 root）

### 终端 1: 启动监控

```bash
# 启用 KFD debug（可选）
sudo su
echo 0xff > /sys/module/amdkfd/parameters/debug_evictions
exit

# 启动监控
sudo dmesg -w | grep -E "create queue|doorbell|priority"
```

### 终端 2: 运行测试

```bash
./test_concurrent
```

### 预期输出（终端 1）:

```
[12345.678] amdkfd: create queue id=1001, priority=11, doorbell_off=0x1000
[12345.679] amdkfd: create queue id=1002, priority=1,  doorbell_off=0x1008
[12345.680] amdkfd: create queue id=1003, priority=11, doorbell_off=0x1010
[12345.681] amdkfd: create queue id=1004, priority=7,  doorbell_off=0x1018
```

**关键观察**:
- ✅ 4 个不同的 `queue_id`
- ✅ 4 个不同的 `doorbell_off`
- ✅ 每个 Queue 有自己的 `priority`

---

## ✅ 验证总结

### 如果看到以下结果，说明验证成功：

| 验证项 | 看到的结果 | 结论 |
|-------|-----------|------|
| Stream 地址 | 4 个不同的指针值 | ✅ 独立的 Stream 对象 |
| Queue ID | 4 个不同的 ID（如 1001-1004） | ✅ 独立的 Queue |
| doorbell 偏移 | 4 个不同的偏移（如 0x1000, 0x1008...） | ✅ 独立的 doorbell |
| 优先级 | 每个 Stream 显示不同的 priority | ✅ 独立的优先级配置 |
| 并发执行 | 所有 Stream 都能提交和完成 | ✅ 可以并发工作 |

### 核心结论

```
✅ 每个 Stream = 1 个独立的 Queue
✅ 每个 Queue = 1 个独立的 ring-buffer
✅ 每个 Queue = 1 个独立的 doorbell 地址

不共享！不复用！完全独立！
```

---

## 🐛 常见问题

### Q1: 编译失败 "hipcc: command not found"

**A**: 添加 ROCm 到 PATH
```bash
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

### Q2: 运行失败 "hipErrorInvalidDevice"

**A**: 检查 GPU 权限
```bash
sudo usermod -aG render $USER
sudo usermod -aG video $USER
# 然后重新登录
```

### Q3: dmesg 没有输出

**A**: 启用 KFD debug
```bash
sudo bash -c 'echo 0xff > /sys/module/amdkfd/parameters/debug_evictions'
```

---

## 📚 下一步

测试完成后，可以：

1. **阅读详细文档**: [README.md](./README.md)
2. **理解理论**: [STREAM_PRIORITY_AND_QUEUE_MAPPING.md](../STREAM_PRIORITY_AND_QUEUE_MAPPING.md)
3. **修改测试**: 改变优先级、增加 Stream 数量等
4. **性能测试**: 提交大量 kernel，观察调度行为

---

**时间**: 5-10 分钟  
**难度**: ⭐⭐☆☆☆  
**收获**: 验证每个 Stream 都有独立的 Queue！
