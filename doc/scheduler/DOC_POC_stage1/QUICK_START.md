# 最快开始 - 一行命令

**如果你遇到了容器内无法访问 `/sys/kernel/debug/kfd/mqds` 的问题，这是最快的解决方案！**

---

## 🚀 立即测试 (在宿主机运行)

```bash
cd /mnt/md0/zhehan/code/flashinfer/dockercode/gpreempt_test && ./host_queue_test.sh
```

这会：
1. ✅ 启动容器内的 PyTorch 测试
2. ✅ 在宿主机查看 Queue ID
3. ✅ 显示完整的 MQD 信息

---

## 📊 预期输出

```
╔════════════════════════════════════════════════════════╗
║  Queue ID 测试 (宿主机版本)                             ║
║  在宿主机上运行，自动启动容器测试并查看 Queue            ║
╚════════════════════════════════════════════════════════╝

✅ 运行在宿主机上
✅ 容器 zhenaiter 正在运行
✅ /sys/kernel/debug/kfd/mqds 可访问
✅ 测试脚本已创建
🚀 测试程序已启动（后台运行）
✅ 找到测试进程
   容器内 PID: 12345

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 MQD 信息 (PID: 12345)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Compute queue on device 0001:01:00.0
    Queue ID: 0 (0x0)
    Process: pid 12345 pasid 0x8001
    is active: yes
    priority: 7

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 提取的 Queue IDs:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Queue IDs: 0 1

✅ 成功获取 Queue ID！

📝 下一步:
   1. 运行多次测试验证一致性
   2. 或运行自动化一致性测试
```

---

## ✅ 成功后的下一步

### 验证 Queue ID 一致性（5 次测试）

```bash
cd /mnt/md0/zhehan/code/flashinfer/dockercode/gpreempt_test && ./host_queue_consistency_test.sh
```

这会运行 5 次测试并分析 Queue ID 是否一致，直接告诉你：
- ✅ **如果一致** → 可以硬编码 (3-5 天完成 POC)
- ⚠️ **如果不一致** → 需要动态发现 (7-10 天完成 POC)

---

## 🔧 如果失败

### 错误 1: 容器未运行

```bash
# 启动容器
docker start zhenaiter

# 验证
docker ps | grep zhenaiter
```

---

### 错误 2: 无法访问 /sys/kernel/debug/kfd/mqds

```bash
# 在宿主机挂载 debugfs
sudo mount -t debugfs none /sys/kernel/debug

# 验证
ls -la /sys/kernel/debug/kfd/

# 检查 KFD 驱动
lsmod | grep amdkfd
```

---

### 错误 3: 权限被拒绝

```bash
# 使用 sudo 运行脚本
sudo ./host_queue_test.sh
```

---

## 📖 详细文档

- **README_QUEUE_TEST.md** - 完整使用指南
- **TROUBLESHOOTING_常见问题解决.md** - 故障排除
- **fix_debugfs.sh** - debugfs 诊断工具

---

## 💡 为什么使用宿主机脚本？

### 问题：容器内看不到 debugfs

```bash
# 在容器内
cat /sys/kernel/debug/kfd/mqds
# 输出: No such file or directory
```

### 原因

Docker 容器默认无法访问宿主机的 debugfs，除非：
1. 容器以特权模式运行
2. debugfs 被挂载到容器内

### 解决方案：宿主机脚本

```
宿主机脚本做的事情:
1. 在容器内启动测试 → 创建 GPU Queue
2. 在宿主机查看 MQD → 可以访问 debugfs
3. 自动关联进程和 Queue ID
```

这样既不需要修改容器配置，也不需要特权模式！

---

## 🎯 实验目标

这些测试的目的是验证：

1. ✅ **能否看到 Queue ID**
2. ✅ **Queue ID 是否一致**（多次运行）
3. ✅ **Queue ID 是否可预测**

根据结果决定 POC Stage 1 的实施策略：
- **一致** → 硬编码策略 (简单，快速)
- **不一致** → 动态发现策略 (复杂，灵活)

---

**现在就运行第一个命令！** 🚀

```bash
cd /mnt/md0/zhehan/code/flashinfer/dockercode/gpreempt_test && ./host_queue_test.sh
```
