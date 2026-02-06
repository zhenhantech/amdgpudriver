# 无需PID的GPU监控方法

**日期**: 2026-02-05  
**问题**: 如何在不知道PID的情况下监控GPU进程？

---

## 🎯 典型场景

你想这样工作：
- **终端1**: 启动监控，等待GPU进程
- **终端2**: 启动测试程序（你不知道它的PID）
- **终端1**: 自动检测并开始监控

---

## ✅ 解决方案

### 方案1: 使用 `watch_new_gpu.sh` ⭐⭐⭐ (最简单)

**最适合**: 两个终端协同工作

#### 终端1 - 启动监控
```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

./watch_new_gpu.sh
```

输出：
```
╔════════════════════════════════════════════════════════╗
║  GPU进程监控 - 等待模式                                 ║
╚════════════════════════════════════════════════════════╝

📋 当前GPU进程:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  (无GPU进程)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏳ 等待新的GPU进程启动...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 提示: 现在可以在另一个终端启动测试程序
         例如: python3 your_model.py

   按 Ctrl+C 可以取消

  等待中... (已等待 5 秒)
```

#### 终端2 - 启动测试
```bash
# 启动你的测试程序
python3 your_model.py

# 或者
docker exec -it zhenaiter bash
python3 /tmp/test_model.py
```

#### 终端1 - 自动开始监控
```
✅ 检测到新的GPU进程!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
进程信息:
  PID:    12345
  进程:   python3
  命令:   python3 your_model.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⏳ 等待进程初始化 (3秒)...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
开始监控 Queue 使用情况
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[  0s] 采样  1: 10 个队列 (IDs: 5, 6, 7, ...)
...
```

---

### 方案2: 使用 `auto_monitor.sh` (更灵活)

`auto_monitor.sh` 提供多种模式：

#### 模式1: 交互式选择

```bash
./auto_monitor.sh
```

会列出所有GPU进程让你选择：
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
检测到的GPU进程:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1] 
PID: 12345  进程: python3
  命令: python3 model_a.py

[2] 
PID: 12346  进程: python3
  容器: zhenaiter
  命令: python3 model_b.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
请选择要监控的进程 [1-2]: 
```

#### 模式2: 等待新进程

```bash
./auto_monitor.sh --wait
```

功能与 `watch_new_gpu.sh` 相同，但更灵活。

#### 模式3: 按进程名匹配

```bash
./auto_monitor.sh --name python3
```

自动监控第一个匹配 `python3` 的GPU进程。

#### 模式4: 监控容器内进程

```bash
./auto_monitor.sh --container zhenaiter
```

自动监控指定Docker容器内的GPU进程。

#### 模式5: 查看所有进程

```bash
./auto_monitor.sh --all
```

一次性显示所有GPU进程的Queue信息。

---

## 📚 完整用法

### watch_new_gpu.sh

```bash
# 简单等待新进程
./watch_new_gpu.sh
```

**特点**:
- ✅ 最简单，无需参数
- ✅ 自动监控30秒，每5秒采样
- ✅ 适合两个终端协同工作

---

### auto_monitor.sh

```bash
# 基础用法
./auto_monitor.sh [选项]

# 选项
--wait                  # 等待新进程
--name <进程名>         # 按名称匹配
--container <容器名>    # 监控容器内进程
--all                   # 显示所有GPU进程
--duration <秒>         # 监控时长（默认30）
--interval <秒>         # 采样间隔（默认5）
-h, --help             # 帮助信息
```

**示例**:

```bash
# 等待新进程，监控60秒，每10秒采样
./auto_monitor.sh --wait --duration 60 --interval 10

# 监控python3进程
./auto_monitor.sh --name python3

# 监控zhenaiter容器，采样更频繁
./auto_monitor.sh --container zhenaiter --interval 2

# 快速查看所有GPU进程
./auto_monitor.sh --all
```

---

## 🎯 推荐使用场景

### 场景1: 日常开发测试 ⭐⭐⭐

```bash
# 终端1
./watch_new_gpu.sh

# 终端2
python3 your_model.py
```

**优点**: 最简单，无需任何参数

---

### 场景2: 多个GPU进程

```bash
# 交互选择
./auto_monitor.sh

# 或者按名称
./auto_monitor.sh --name my_model
```

**优点**: 可以精确选择要监控的进程

---

### 场景3: Docker容器测试

```bash
# 监控容器内的GPU进程
./auto_monitor.sh --container zhenaiter
```

**优点**: 自动处理容器内的进程

---

### 场景4: 快速检查

```bash
# 查看所有GPU进程的Queue使用情况
./auto_monitor.sh --all
```

**优点**: 一次性看到所有信息

---

## 🔧 传统方法（仍然可用）

如果你已经知道PID，可以直接使用：

```bash
# 方法1: 直接指定PID
sudo ./queue_monitor 12345 30 5

# 方法2: 通过命令查找PID
PID=$(pgrep -f python3 | head -1)
sudo ./queue_monitor $PID 30 5

# 方法3: Docker容器内的进程
CONTAINER_PID=$(docker exec zhenaiter pgrep -f python3 | head -1)
sudo ./queue_monitor $CONTAINER_PID 30 5
```

---

## 💡 工作流对比

### 旧方法 (需要PID)

```bash
# 1. 启动测试
python3 model.py &

# 2. 查找PID
PID=$(pgrep -f model.py)

# 3. 监控
sudo ./queue_monitor $PID 30 5
```

**问题**: 需要手动查找PID，步骤繁琐

---

### 新方法 (自动检测) ⭐

```bash
# 终端1: 启动监控
./watch_new_gpu.sh

# 终端2: 启动测试
python3 model.py
```

**优点**: 
- ✅ 无需查找PID
- ✅ 自动检测新进程
- ✅ 自动开始监控
- ✅ 两步完成

---

## 🎓 高级技巧

### 1. 持续监控多次运行

```bash
while true; do
    echo "等待下一次测试..."
    ./watch_new_gpu.sh
    echo ""
    echo "按Enter继续，Ctrl+C退出"
    read
done
```

### 2. 监控并保存日志

```bash
./watch_new_gpu.sh | tee monitor_$(date +%Y%m%d_%H%M%S).log
```

### 3. 同时监控多个进程

```bash
# 终端1
./auto_monitor.sh --name model_a

# 终端2
./auto_monitor.sh --name model_b
```

### 4. 自定义监控参数

```bash
# 长时间监控，频繁采样
./auto_monitor.sh --wait --duration 300 --interval 2
```

---

## 📊 功能对比

| 功能 | queue_monitor | watch_new_gpu.sh | auto_monitor.sh |
|------|--------------|------------------|-----------------|
| **需要PID** | ✅ 必需 | ❌ 自动检测 | ❌ 自动检测 |
| **等待新进程** | ❌ | ✅ | ✅ |
| **交互选择** | ❌ | ❌ | ✅ |
| **按名称匹配** | ❌ | ❌ | ✅ |
| **容器支持** | 手动 | 自动 | ✅ 专门选项 |
| **查看所有进程** | ❌ | ❌ | ✅ |
| **自定义参数** | ✅ 完全 | ❌ 固定 | ✅ 部分 |

---

## ✅ 总结

### 快速选择指南

**我想...**

1. **最简单的监控** → 使用 `./watch_new_gpu.sh` ⭐⭐⭐

2. **选择特定进程** → 使用 `./auto_monitor.sh` (交互模式)

3. **监控容器进程** → 使用 `./auto_monitor.sh --container NAME`

4. **查看所有进程** → 使用 `./auto_monitor.sh --all`

5. **精确控制** → 使用 `./queue_monitor PID DURATION INTERVAL`

---

## 🔗 相关文档

- `README.md` - 代码目录完整说明
- `QUICKSTART_CPP_POC.md` - 快速开始指南
- `README_USERSPACE_POC.md` - 完整使用指南

---

**最后更新**: 2026-02-05  
**状态**: ✅ 已验证
