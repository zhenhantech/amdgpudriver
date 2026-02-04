# KCQ 配置完整指南

**更新日期**: 2026-02-03  
**适用系统**: AMD MI300X / MI308X 系统  
**参数**: `num_kcq` (Kernel Compute Queue 数量)

---

## 📋 什么是 num_kcq？

### 定义

**num_kcq**: 每个 XCC (Execution Compute Core) 分配给内核使用的队列数量

### 影响

```
每个 XCC 有 32 个硬件队列：
  ├─ num_kcq 个 → KCQ (内核队列)
  └─ (32 - num_kcq) 个 → 用户队列

例如 num_kcq=2:
  - 2 个 KCQ
  - 30 个用户队列 ✅

例如 num_kcq=1:
  - 1 个 KCQ
  - 31 个用户队列 ✅ (+1 个可用队列)
```

### 对系统的影响 (MI308X: 4 XCC/GPU, 8 GPUs)

| num_kcq | 每 XCC KCQ | 每 XCC 用户队列 | 每 GPU 用户队列 | 全系统用户队列 |
|---------|-----------|----------------|----------------|---------------|
| **8** (默认旧版) | 8 | 24 | 96 | 768 |
| **2** (推荐) | 2 | 30 | 120 | 960 |
| **1** (优化) | 1 | 31 | 124 | 992 |

---

## 🔧 配置方法

### 方法 1: modprobe 配置文件（⭐推荐，永久生效）

#### Step 1: 创建/编辑配置文件

```bash
# 编辑配置文件
sudo nano /etc/modprobe.d/amdgpu.conf
```

#### Step 2: 添加配置

```bash
# 设置 num_kcq=2
options amdgpu num_kcq=2
```

或者一行命令：

```bash
echo 'options amdgpu num_kcq=2' | sudo tee /etc/modprobe.d/amdgpu.conf
```

#### Step 3: 重新生成 initramfs

```bash
# RHEL/CentOS
sudo dracut --force

# Ubuntu/Debian
sudo update-initramfs -u
```

#### Step 4: 重启系统

```bash
sudo reboot
```

#### Step 5: 验证配置

```bash
# 检查模块参数
cat /sys/module/amdgpu/parameters/num_kcq
# 应该输出: 2

# 检查启动参数
cat /proc/cmdline | grep num_kcq
```

---

### 方法 2: GRUB 内核启动参数（永久，优先级更高）

#### Step 1: 编辑 GRUB 配置

```bash
sudo nano /etc/default/grub
```

#### Step 2: 添加内核参数

找到 `GRUB_CMDLINE_LINUX` 行，添加 `amdgpu.num_kcq=2`:

```bash
GRUB_CMDLINE_LINUX="... amdgpu.num_kcq=2"
```

完整示例：

```bash
GRUB_CMDLINE_LINUX="crashkernel=auto rhgb quiet amdgpu.num_kcq=2"
```

#### Step 3: 更新 GRUB

```bash
# RHEL/CentOS 8
sudo grub2-mkconfig -o /boot/grub2/grub.cfg

# RHEL/CentOS 7
sudo grub2-mkconfig -o /boot/efi/EFI/centos/grub.cfg

# Ubuntu/Debian
sudo update-grub
```

#### Step 4: 重启

```bash
sudo reboot
```

#### Step 5: 验证

```bash
# 检查内核参数
cat /proc/cmdline | grep num_kcq
# 应该看到: amdgpu.num_kcq=2

# 检查实际值
cat /sys/module/amdgpu/parameters/num_kcq
# 应该输出: 2
```

---

### 方法 3: 运行时修改（❌ 不推荐，通常无效）

```bash
# 尝试运行时修改（通常不会生效）
echo 2 | sudo tee /sys/module/amdgpu/parameters/num_kcq
```

**为什么无效**:
- amdgpu 模块加载时已经初始化了队列
- 运行时无法重新分配硬件队列
- 必须重启才能生效

---

## 📊 当前配置检查

### 快速检查脚本

```bash
#!/bin/bash
# check_kcq_config.sh

echo "╔════════════════════════════════════════════════════════╗"
echo "║  KCQ 配置检查                                           ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

echo "1. 当前运行时配置:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "/sys/module/amdgpu/parameters/num_kcq" ]; then
    NUM_KCQ=$(cat /sys/module/amdgpu/parameters/num_kcq)
    echo "   num_kcq = $NUM_KCQ"
else
    echo "   ❌ 无法读取 num_kcq"
fi
echo ""

echo "2. 内核启动参数:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if cat /proc/cmdline | grep -q "num_kcq"; then
    echo "   ✅ 在启动参数中找到:"
    cat /proc/cmdline | grep -o "amdgpu.num_kcq=[0-9]*"
else
    echo "   ⚠️ 启动参数中未设置 num_kcq"
fi
echo ""

echo "3. modprobe 配置文件:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "/etc/modprobe.d/amdgpu.conf" ]; then
    echo "   ✅ 配置文件存在:"
    cat /etc/modprobe.d/amdgpu.conf | grep -i num_kcq || echo "   (未找到 num_kcq 配置)"
else
    echo "   ⚠️ /etc/modprobe.d/amdgpu.conf 不存在"
fi
echo ""

echo "4. sysfs 队列信息:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "/sys/class/kfd/kfd/topology/nodes/1/properties" ]; then
    NUM_CP_QUEUES=$(cat /sys/class/kfd/kfd/topology/nodes/1/properties | grep num_cp_queues | awk '{print $2}')
    echo "   num_cp_queues (每GPU) = $NUM_CP_QUEUES"
    
    if [ -n "$NUM_KCQ" ]; then
        echo ""
        echo "   计算验证:"
        echo "     期望: 32 - $NUM_KCQ = $((32 - NUM_KCQ)) 个用户队列/XCC"
        echo "     实际: $NUM_CP_QUEUES 个用户队列/GPU (应该是 $((32 - NUM_KCQ)) × 4)"
        
        EXPECTED=$((( 32 - NUM_KCQ ) * 4))
        if [ "$NUM_CP_QUEUES" -eq "$EXPECTED" ]; then
            echo "     ✅ 一致！"
        else
            echo "     ⚠️ 不一致 (期望 $EXPECTED, 实际 $NUM_CP_QUEUES)"
        fi
    fi
else
    echo "   ⚠️ 无法读取 sysfs"
fi
echo ""

echo "╔════════════════════════════════════════════════════════╗"
echo "║  检查完成                                               ║"
echo "╚════════════════════════════════════════════════════════╝"
```

---

## 🎯 推荐配置

### 对于 POC Stage 1 测试

**推荐**: `num_kcq=2`

**原因**:
1. ✅ 平衡：足够的 KCQ，也有足够的用户队列
2. ✅ 稳定：默认推荐配置
3. ✅ 充足：120 队列/GPU 对大多数场景足够

### 对于生产环境优化

**可考虑**: `num_kcq=1`

**收益**:
- +4 个队列/GPU
- +32 个队列/系统 (8 GPUs)

**风险**:
- 如果内核需要多个队列，可能不足
- 需要验证稳定性

---

## 🔍 配置生效验证

### 完整验证脚本

```bash
#!/bin/bash
# verify_kcq_config.sh

echo "╔════════════════════════════════════════════════════════╗"
echo "║  KCQ 配置验证                                           ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# 1. 检查当前值
CURRENT=$(cat /sys/module/amdgpu/parameters/num_kcq)
echo "✅ 当前 num_kcq = $CURRENT"
echo ""

# 2. 检查配置来源
echo "配置来源检查:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if cat /proc/cmdline | grep -q "amdgpu.num_kcq"; then
    echo "  ✅ 来自内核启动参数 (GRUB)"
    cat /proc/cmdline | grep -o "amdgpu.num_kcq=[0-9]*"
elif [ -f "/etc/modprobe.d/amdgpu.conf" ] && grep -q "num_kcq" /etc/modprobe.d/amdgpu.conf; then
    echo "  ✅ 来自 modprobe 配置"
    grep "num_kcq" /etc/modprobe.d/amdgpu.conf
else
    echo "  ℹ️ 使用默认值"
fi

echo ""

# 3. 计算实际队列数
echo "队列分配验证:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

NUM_CP_QUEUES=$(cat /sys/class/kfd/kfd/topology/nodes/1/properties | grep num_cp_queues | awk '{print $2}')
EXPECTED=$((( 32 - CURRENT ) * 4))

echo "  每 XCC:"
echo "    - 硬件队列: 32"
echo "    - KCQ: $CURRENT"
echo "    - 用户队列: $((32 - CURRENT))"
echo ""
echo "  每 GPU (4 XCC):"
echo "    - 期望用户队列: $EXPECTED"
echo "    - 实际用户队列: $NUM_CP_QUEUES"
echo ""

if [ "$NUM_CP_QUEUES" -eq "$EXPECTED" ]; then
    echo "  ✅ 配置正确生效！"
else
    echo "  ⚠️ 队列数不匹配"
    echo "     可能原因:"
    echo "     - 配置未生效（需要重启）"
    echo "     - XCC 数量不是 4"
    echo "     - 其他因素影响"
fi

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  验证完成                                               ║"
echo "╚════════════════════════════════════════════════════════╝"
```

---

## 📝 配置历史记录

### 你的系统（根据之前的日志）

**当前配置**: `num_kcq=2`

**验证日志**: 
```
/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/test_queue_limits/logs_KCQ_config/verify_kcq_config.sh_configure2.log
```

**结果**: ✅ 配置生效，num_kcq=2 正常工作

---

## 🛠️ 修改 num_kcq 的完整步骤

### 场景：将 num_kcq 从当前值改为 2

#### Step 1: 备份当前配置

```bash
# 记录当前状态
cat /sys/module/amdgpu/parameters/num_kcq > /tmp/num_kcq_backup.txt
cat /proc/cmdline > /tmp/cmdline_backup.txt

# 备份 modprobe 配置
if [ -f "/etc/modprobe.d/amdgpu.conf" ]; then
    sudo cp /etc/modprobe.d/amdgpu.conf /etc/modprobe.d/amdgpu.conf.bak
fi
```

#### Step 2: 设置新配置

**方案 A: 使用 modprobe 配置**

```bash
# 创建配置文件
echo 'options amdgpu num_kcq=2' | sudo tee /etc/modprobe.d/amdgpu.conf

# 查看配置
cat /etc/modprobe.d/amdgpu.conf
```

**方案 B: 使用 GRUB 参数（如果方案 A 不生效）**

```bash
# 编辑 GRUB
sudo nano /etc/default/grub

# 在 GRUB_CMDLINE_LINUX 中添加: amdgpu.num_kcq=2
# 例如:
# GRUB_CMDLINE_LINUX="crashkernel=auto rhgb quiet amdgpu.num_kcq=2"

# 更新 GRUB
sudo grub2-mkconfig -o /boot/grub2/grub.cfg
```

#### Step 3: 重新生成 initramfs (方案 A)

```bash
sudo dracut --force
```

#### Step 4: 重启

```bash
sudo reboot
```

#### Step 5: 验证

```bash
# 检查值
cat /sys/module/amdgpu/parameters/num_kcq

# 检查队列数
cat /sys/class/kfd/kfd/topology/nodes/1/properties | grep num_cp_queues

# 期望: 30 (如果 num_kcq=2 且 4 XCC)
```

---

## 🔍 故障排除

### 问题 1: 配置未生效

**症状**: 重启后 `num_kcq` 值没有改变

**诊断**:

```bash
# 检查配置文件
cat /etc/modprobe.d/amdgpu.conf

# 检查 initramfs 是否包含配置
lsinitrd | grep amdgpu.conf

# 检查内核启动参数
cat /proc/cmdline
```

**解决**:

1. 确认配置文件语法正确
   ```bash
   # 正确格式
   options amdgpu num_kcq=2
   
   # 错误格式（注意拼写和空格）
   option amdgpu num_kcq=2  # 错误：option → options
   options amdgpu num_kcq =2  # 错误：多余空格
   ```

2. 重新生成 initramfs
   ```bash
   sudo dracut --force --verbose
   ```

3. 使用 GRUB 方法（优先级更高）

---

### 问题 2: 不确定当前使用哪种配置方法

**诊断**:

```bash
# 1. 检查 GRUB 参数（最高优先级）
cat /proc/cmdline | grep num_kcq

# 2. 检查 modprobe 配置
grep num_kcq /etc/modprobe.d/*.conf 2>/dev/null

# 3. 检查 dmesg
dmesg | grep -i "num_kcq"
```

**优先级**:
```
GRUB 启动参数 > modprobe.d 配置 > 驱动默认值
```

---

### 问题 3: 修改后系统不稳定

**症状**: 设置 `num_kcq=1` 后系统出现问题

**解决**:

1. 恢复到 `num_kcq=2`
2. 检查 dmesg 错误日志
3. 确认内核队列是否足够

---

## 🧪 测试配置是否生效

### 测试脚本

```bash
#!/bin/bash
# test_kcq_effect.sh

NUM_KCQ=$(cat /sys/module/amdgpu/parameters/num_kcq)
echo "当前 num_kcq = $NUM_KCQ"

# 检查每个 GPU 的队列数
for node in /sys/class/kfd/kfd/topology/nodes/*/properties; do
    if grep -q "gpu_id" "$node"; then
        GPU_ID=$(grep gpu_id "$node" | awk '{print $2}')
        NUM_CP=$(grep num_cp_queues "$node" | awk '{print $2}')
        
        EXPECTED=$(( (32 - NUM_KCQ) * 4 ))
        
        echo ""
        echo "GPU $GPU_ID:"
        echo "  num_cp_queues = $NUM_CP"
        echo "  expected      = $EXPECTED (32-$NUM_KCQ)×4"
        
        if [ "$NUM_CP" -eq "$EXPECTED" ]; then
            echo "  ✅ 正确"
        else
            echo "  ⚠️ 不匹配"
        fi
    fi
done
```

---

## 📖 参考文档

### 官方文档

- [AMD GPU Driver Documentation](https://docs.kernel.org/gpu/amdgpu/driver-core.html)
- Kernel Module Parameters

### 项目文档

- `XCC_XCD_AND_QUEUE_COUNT_CLARIFICATION.md` - XCC 和队列数说明
- `CODE_ANALYSIS_30_QUEUES_SOURCE.md` - 30 队列的来源分析
- `logs_KCQ_config/verify_kcq_config.sh_configure2.log` - 你的历史验证日志

---

## 🎯 快速命令参考

### 查看当前配置

```bash
cat /sys/module/amdgpu/parameters/num_kcq
```

### 修改为 num_kcq=2

```bash
echo 'options amdgpu num_kcq=2' | sudo tee /etc/modprobe.d/amdgpu.conf
sudo dracut --force
sudo reboot
```

### 修改为 num_kcq=1（优化）

```bash
echo 'options amdgpu num_kcq=1' | sudo tee /etc/modprobe.d/amdgpu.conf
sudo dracut --force
sudo reboot
```

### 验证配置

```bash
cat /sys/module/amdgpu/parameters/num_kcq
cat /sys/class/kfd/kfd/topology/nodes/1/properties | grep num_cp_queues
```

---

**最后更新**: 2026-02-03  
**维护者**: Zhehan

**当前你的系统**: `num_kcq=2` ✅ (已验证生效)
