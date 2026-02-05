# 实验1脚本问题修复

**问题发现时间**: 2026-02-04  
**问题等级**: 🔴 阻塞（脚本无法运行）

---

## 🐛 问题分析

### 原始错误

```
SyntaxError: f-string: expecting '=', or '!', or ':', or '}'
```

**错误位置**:
```python
print(f"[{time.strftime(%H:%M:%S)}] 测试模型启动")
                           ^
```

### 根本原因

在bash heredoc中创建Python代码时，f-string中的引号处理有问题：

```bash
# 原始代码（错误）
docker exec $CONTAINER bash -c 'cat > /tmp/test_model.py << '\''PYEOF'\''
print(f"[{time.strftime('%H:%M:%S')}] 测试模型启动")
```

**问题**:
1. bash的heredoc处理了内部的单引号 `'%H:%M:%S'`
2. 导致Python代码中缺少引号
3. 最终Python看到的是 `time.strftime(%H:%M:%S)` 而不是 `time.strftime('%H:%M:%S')`

---

## ✅ 解决方案

### 修复1: 使用带引号的heredoc结束符

```bash
# 修复后（正确）
docker exec $CONTAINER bash -c 'cat > /tmp/test_model.py << "PYEOF"
print(f"[{time.strftime('%H:%M:%S')}] 测试模型启动")
```

**原理**: `"PYEOF"` 告诉bash不要展开heredoc内部的任何内容

### 修复2: 使用辅助函数避免复杂引号

```python
# 更安全的方式
def get_timestamp():
    return time.strftime("%H:%M:%S")

print(f"[{get_timestamp()}] 测试模型启动")
```

---

## 🚀 快速修复

### 方法1: 使用修复版脚本 ⭐推荐

```bash
cd /mnt/md0/zhehan/code/flashinfer/dockercode/gpreempt_test

# 使用修复版脚本
./exp01_queue_monitor_fixed.sh
```

### 方法2: 手动修复原脚本

```bash
# 1. 编辑原脚本
nano exp01_queue_monitor.sh

# 2. 找到第41行附近，将：
docker exec $CONTAINER bash -c 'cat > /tmp/test_model.py << '\''PYEOF'\''

# 改为：
docker exec $CONTAINER bash -c 'cat > /tmp/test_model.py << "PYEOF"

# 3. 保存并退出
```

---

## 📊 验证修复

运行修复后的脚本应该看到：

```
✅ 测试模型已启动（后台）
   宿主机进程PID: XXXXX

⏳ 等待模型初始化（20秒）...
....................

✅ 找到容器内进程
   容器内PID: XXXX
```

而不是：

```
⚠️ 未找到容器内的进程
   查看模型输出:
  File "/tmp/test_model.py", line 6
    print(f"[{time.strftime(%H:%M:%S)}] 测试模型启动")
                           ^
SyntaxError: f-string: expecting '=', or '!', or ':', or '}'
```

---

## 🎯 立即行动

```bash
cd /mnt/md0/zhehan/code/flashinfer/dockercode/gpreempt_test

# 运行修复版脚本
./exp01_queue_monitor_fixed.sh

# 等待完成后分析结果
python3 analyze_queue_usage.py ./exp01_results
```

---

## 📝 技术细节

### Bash Heredoc引号规则

| 写法 | 行为 |
|------|------|
| `<< EOF` | bash会展开变量和命令替换 |
| `<< 'EOF'` | bash不展开，但引号处理复杂 |
| `<< "EOF"` | **推荐**: bash不展开，引号清晰 |
| `<< \EOF` | bash不展开（转义方式） |

### Python f-string在heredoc中的陷阱

```bash
# ❌ 错误：引号被bash吃掉
cat << 'EOF'
print(f"{time.strftime('%H')}")
EOF
# 输出: print(f"{time.strftime(%H)}")  # 缺少引号！

# ✅ 正确：使用双引号heredoc
cat << "EOF"
print(f"{time.strftime('%H')}")
EOF
# 输出: print(f"{time.strftime('%H')}")  # 引号保留！
```

---

## 💡 最佳实践

### 1. heredoc创建Python代码时

```bash
# 推荐方式
docker exec container bash -c 'cat > /tmp/script.py << "PYEOF"
# Python代码
PYEOF'
```

### 2. 避免复杂的字符串嵌套

```python
# 不推荐：f-string + strftime + 格式字符串
print(f"[{time.strftime('%H:%M:%S')}] message")

# 推荐：使用辅助函数
def ts():
    return time.strftime("%H:%M:%S")
print(f"[{ts()}] message")
```

### 3. 测试heredoc内容

```bash
# 先测试heredoc生成的内容
cat << "EOF"
print(f"test")
EOF

# 确认正确后再写入文件
```

---

## 🔗 相关文档

- **实验设计**: `EXP_01_QUEUE_USAGE_ANALYSIS.md`
- **快速指南**: `EXP01_QUICK_START.md`
- **修复后脚本**: `exp01_queue_monitor_fixed.sh` ✅

---

**修复时间**: 2026-02-04  
**状态**: ✅ 已修复  
**验证**: 待运行

现在可以运行实验了！🚀
