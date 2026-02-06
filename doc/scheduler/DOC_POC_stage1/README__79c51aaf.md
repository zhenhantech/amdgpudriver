# POC Stage 1 实施代码

**状态**: 🔄 开发中  
**日期**: 2026-02-05

---

## 📁 目录结构

```
poc_implementation/
├── README.md                    # 本文档
├── Makefile                     # 编译配置
├── queue_finder.c              # Queue查询工具
├── libgpreempt_poc.c           # C库实现（待开发）
├── libgpreempt_poc.h           # C库头文件（待开发）
├── test_preemption.py          # Python测试框架（待开发）
└── run_poc_test.sh             # 自动化测试脚本（待开发）
```

---

## 🚀 快速开始

### 步骤1: 编译Queue查询工具

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code/poc_implementation

# 编译
make

# 查看帮助
./queue_finder
```

### 步骤2: 使用Queue查询工具

基于我们之前的分析，测试Case-A和Case-B：

```bash
# Case-A (CNN)
./queue_finder 158036 ../log/case_comparison_20260205_155247/case_a_cnn.log

# Case-B (Transformer)
./queue_finder 158122 ../log/case_comparison_20260205_155247/case_b_transformer.log
```

**预期输出**：
```
╔════════════════════════════════════════════════════════════════╗
║  Queue信息汇总                                                  ║
╚════════════════════════════════════════════════════════════════╝

Queue #1:
  地址:     0x00007f9567e00000
  Queue ID: 1
  PID:      158036
  活跃:     是

总计: 1个Queue

✅ 已生成Python配置: queue_config_pid_158036.py
```

---

## 🔧 开发计划

### ✅ Phase 1: Queue识别（当前）

- [x] `queue_finder.c` - Queue查询工具
- [x] 支持从AMD日志提取Queue信息
- [x] 生成Python配置文件
- [ ] 支持从debugfs读取实时状态
- [ ] 支持Docker容器PID映射

### ⏳ Phase 2: 基础抢占API（下一步）

**文件**: `libgpreempt_poc.c` / `.h`

功能：
- [ ] `gpreempt_poc_init()` - 初始化KFD连接
- [ ] `gpreempt_suspend_queues()` - 暂停队列
- [ ] `gpreempt_resume_queues()` - 恢复队列
- [ ] `gpreempt_get_queue_status()` - 查询状态

基于API:
```c
// 使用 KFD_IOC_DBG_TRAP_SUSPEND_QUEUES
ioctl(kfd_fd, AMDKFD_IOC_DBG_TRAP, &args);
```

### ⏳ Phase 3: Python测试框架

**文件**: `test_preemption.py`

功能：
- [ ] 加载libgpreempt_poc.so
- [ ] 监控Online/Offline任务
- [ ] 自动触发抢占
- [ ] 性能统计和报告

### ⏳ Phase 4: 自动化测试

**文件**: `run_poc_test.sh`

功能：
- [ ] 启动Case-A和Case-B
- [ ] 运行抢占测试
- [ ] 收集日志和性能数据
- [ ] 生成测试报告

---

## 📋 使用说明

### Queue Finder工具

**功能**：从PID和AMD日志中提取Queue信息

**用法**：
```bash
./queue_finder <pid> [amd_log_file]

参数：
  pid            - 目标进程PID
  amd_log_file   - AMD日志文件路径（可选）

示例：
  # 使用已有日志
  ./queue_finder 158036 ../log/case_comparison_20260205_155247/case_a_cnn.log
  
  # 只查询PID（需要sudo读取debugfs）
  sudo ./queue_finder 158036
```

**输出**：
1. 终端显示Queue信息
2. 生成Python配置文件 `queue_config_pid_<pid>.py`

**配置文件格式**：
```python
# queue_config_pid_158036.py
queues = [
    {
        'addr': 0x00007f9567e00000,
        'queue_id': 1,
        'pid': 158036,
        'is_active': True,
    },
]
```

---

## 🧪 测试场景

### 场景1: 验证Queue识别

```bash
# 1. 使用之前的测试日志
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code/poc_implementation

# 2. 测试Case-A
./queue_finder 158036 ../log/case_comparison_20260205_155247/case_a_cnn.log

# 3. 测试Case-B
./queue_finder 158122 ../log/case_comparison_20260205_155247/case_b_transformer.log

# 4. 验证输出
ls queue_config_*.py
```

**验收标准**：
- ✅ 正确识别Queue地址
- ✅ 提取Queue ID
- ✅ 生成有效的Python配置

### 场景2: 实时Queue监控（需要sudo）

```bash
# 启动新的测试容器
docker exec -d zhen_vllm_dsv3 bash -c "
    export AMD_LOG_LEVEL=5
    cd /workspace/code
    python3 case_a_cnn.py > /tmp/case_a.log 2>&1
"

# 获取PID
CASE_A_PID=$(docker exec zhen_vllm_dsv3 pgrep -f case_a_cnn.py)

# 等待Queue创建
sleep 3

# 查询Queue
sudo ./queue_finder $CASE_A_PID

# 从debugfs验证
sudo cat /sys/kernel/debug/kfd/hqds | grep -A 20 "Queue"
```

---

## 📊 开发进度

| 阶段 | 任务 | 状态 | 完成时间 |
|------|------|------|----------|
| Phase 1 | Queue查询工具 | ✅ 完成 | 2026-02-05 |
| Phase 1 | 编译系统 | ✅ 完成 | 2026-02-05 |
| Phase 1 | 文档 | ✅ 完成 | 2026-02-05 |
| Phase 2 | C库封装 | ⏳ 待开始 | - |
| Phase 2 | suspend/resume API | ⏳ 待开始 | - |
| Phase 3 | Python框架 | ⏳ 待开始 | - |
| Phase 4 | 自动化测试 | ⏳ 待开始 | - |

---

## 🔍 技术细节

### Queue信息来源

1. **AMD日志**（推荐）:
   - 设置 `AMD_LOG_LEVEL=5`
   - 从日志中grep `HWq=0x` 提取地址
   - 可靠且详细

2. **debugfs** (需要sudo):
   - 路径: `/sys/kernel/debug/kfd/hqds`
   - 实时状态
   - 包含ACTIVE, RPTR, WPTR等

3. **procfs**:
   - 路径: `/proc/<pid>/maps`
   - 可以看到内存映射
   - 但不直接显示Queue信息

### PID映射（容器环境）

```bash
# 容器PID → 主机PID
docker inspect -f '{{.State.Pid}}' <container_name>

# 示例
docker inspect -f '{{.State.Pid}}' zhen_vllm_dsv3
# 输出: 7064 (主机PID)
```

---

## 🐛 常见问题

### Q1: 编译错误

```bash
# 确保有gcc
gcc --version

# 如果没有，安装
sudo yum install gcc
```

### Q2: 找不到Queue信息

**原因**: 没有运行AMD_LOG_LEVEL=5的测试

**解决**:
```bash
# 重新运行测试
docker exec zhen_vllm_dsv3 bash -c "
    export AMD_LOG_LEVEL=5
    cd /workspace/code
    python3 case_a_cnn.py > /tmp/case_a.log 2>&1
"

# 使用新日志
./queue_finder <pid> /tmp/case_a.log
```

### Q3: debugfs权限问题

```bash
# 需要sudo权限
sudo ./queue_finder <pid>

# 或者添加用户到相关组
sudo usermod -a -G video $USER
```

---

## 📚 参考文档

- **分析结果**: `../log/case_comparison_20260205_155247/ANALYSIS_SUMMARY.md`
- **实施计划**: `../NEXT_STEPS_PREEMPTION_POC.md`
- **POC设计**: `../ARCH_Design_01_POC_Stage1_实施方案.md`
- **创新方案**: `../New_DESIGN_MAP_UNMAP_BASED_PREEMPTION.md`

---

## 🚀 下一步

1. **测试Queue Finder**
   ```bash
   make
   ./queue_finder 158036 ../log/case_comparison_20260205_155247/case_a_cnn.log
   ```

2. **开发C库封装**
   - 创建 `libgpreempt_poc.c`
   - 实现suspend/resume API
   - 测试基本功能

3. **Python集成**
   - 加载C库
   - 创建调度器类
   - 实现自动抢占

---

**当前状态**: Phase 1 - Queue识别工具已完成 ✅  
**下一步**: Phase 2 - 开发C库封装（suspend/resume API）  
**预计完成**: 本周内

