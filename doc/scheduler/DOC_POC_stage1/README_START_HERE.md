# POC Stage 1 - 快速开始指南

**创建日期**: 2026-02-03  
**目标**: Online/Offline AI 模型优先级调度概念验证  
**预计时间**: 7-10 天

---

## 🎯 核心目标

验证在 AMD GPU (ROCm + HIP + KFD) 上，**使用 Queue-level 抢占实现 Online-AI 优先于 Offline-AI** 的可行性。

**测试场景**:
- **Online-AI**: 推理任务，高优先级 (15)，实时性要求 < 50ms
- **Offline-AI**: 训练任务，低优先级 (2)，无实时性要求

---

## 📚 文档导航

### 核心文档（必读）

1. **test_scenaria.md** - 测试场景定义 (5 行)
2. **EXP_Design_01_MQD_HQD_模型关联性实验.md** ⭐ - **前置实验**（必须先做！）
3. **ARCH_Design_01_POC_Stage1_实施方案.md** - 完整实施方案 (695 行)
4. **ARCH_Design_02_三种API技术对比.md** - 技术选型分析
5. **POC_Stage1_TODOLIST.md** - 详细任务清单

### 技术细节文档

6. **ARCH_Design_03_QueueID获取与环境配置.md** - Queue ID 获取方法
7. **ARCH_Design_04_HQD信息获取完整指南.md** - HQD 状态查看

---

## ⚡ 1 分钟快速理解

### 问题

如何让 **Online-AI 推理任务能快速抢占 Offline-AI 训练任务**？

### 方案

使用 KFD 的 **`KFD_IOC_DBG_TRAP_SUSPEND_QUEUES`** API：

```
1. Offline-AI 持续训练（低优先级）
2. Online-AI 任务到达
3. 调用 suspend_queues(offline_queue_ids) → 暂停 Offline
4. Online-AI 执行
5. 调用 resume_queues(offline_queue_ids) → 恢复 Offline
```

### 核心挑战

**如何获取 Queue ID**？

**解决方案**:
- **方法 1**: 暴力枚举 0-10（最快，2 分钟）
- **方法 2**: 解析 `/sys/kernel/debug/kfd/mqds`（精确，需要 root）
- **方法 3**: 修改 AI 模型打印 Queue ID（最可靠，需要代码修改）

---

## 🚀 立即开始的3种方式

### 方式 1: 读完整方案（推荐给架构师）

```bash
# 阅读顺序
1. ARCH_Design_01_POC_Stage1_实施方案.md       # 30分钟
2. ARCH_Design_02_三种API技术对比.md           # 20分钟
3. ARCH_Design_03_QueueID获取与环境配置.md     # 15分钟
                                               ────────
                                               总计: ~1小时

# 理解后开始实施
按照 POC_Stage1_TODOLIST.md 执行
```

### 方式 2: 立即动手实验（⭐⭐⭐⭐⭐ 强烈推荐！）

```bash
# 1. 阅读实验设计
cat EXP_Design_01_MQD_HQD_模型关联性实验.md  # 10 分钟

# 2. 进入 Docker 容器
docker exec -it zhenaiter /bin/bash

# 3. 激活环境
export MAMBA_EXE='/root/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/root/micromamba'
eval "$(/root/.local/bin/micromamba shell hook --shell=bash)"
micromamba activate flashinfer-rocm

# 4. 快速测试 Queue ID 可见性
cd /data/dockercode/gpreempt_test
sudo cat /sys/kernel/debug/kfd/mqds | head -20

# 5. 运行完整实验 (1.5 小时)
cd /data/dockercode
mkdir -p poc_stage1_experiment
cd poc_stage1_experiment
# 复制脚本并执行 ./run_experiment.sh

# 6. 根据实验结果决定实施策略
```

### 方式 3: 使用历史代码（推荐快速验证）

```bash
# 使用之前 CWSR 测试的代码
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_GPREEMPT/MI300_Testing

# 阅读快速开始指南
cat Docker容器内端到端测试_快速开始.md

# 运行已有的测试
docker exec -it zhenaiter /bin/bash -c "
cd /data/dockercode/gpreempt_test && \
HIP_DEVICE=0 ./test_hip_preempt 100000 20000 0
"
```

---

## 📐 技术架构速览

### 系统层次

```
┌─────────────────────────────────────┐
│ Online/Offline AI Models            │
│   hipLaunchKernel() → Doorbell      │
└────────────┬────────────────────────┘
             ↓ 用户空间提交（快，~100ns）
┌─────────────────────────────────────┐
│ Test Framework (Python)             │
│   • 监控 Online 任务到达            │
│   • ioctl suspend_queues → 暂停    │
│   • ioctl resume_queues → 恢复     │
└────────────┬────────────────────────┘
             ↓ ioctl (~1-10μs)
┌─────────────────────────────────────┐
│ KFD (内核驱动)                       │
│   suspend_queues()                  │
│    → evict_process_queues_cpsch()  │
│    → 触发 CWSR                      │
└────────────┬────────────────────────┘
             ↓ PM4 Commands
┌─────────────────────────────────────┐
│ GPU Hardware (CPSCH 模式)           │
│   • CP Scheduler 处理 UNMAP        │
│   • CWSR 保存 Wave 状态            │
│   • 队列从 Runlist 移除            │
└─────────────────────────────────────┘
```

---

## 🔑 关键 API 和数据结构

### 核心 API

```c
// KFD IOCTL
int suspend_queues(uint32_t *queue_ids, 
                  uint32_t num_queues,
                  uint32_t grace_period_us);

int resume_queues(uint32_t *queue_ids,
                 uint32_t num_queues);
```

### Queue 信息来源

**MQD (软件层)**:
```bash
/sys/kernel/debug/kfd/mqds
  → Queue ID (用于 suspend_queues)
  → Process PID
  → Priority
  → is active (软件状态)
```

**HQD (硬件层)**:
```bash
/sys/kernel/debug/kfd/hqds
  → CP_HQD_ACTIVE (0x1247) bit[0] (硬件状态)
  → CP_HQD_PQ_RPTR/WPTR (Ring Buffer 指针)
  → 56 个硬件寄存器
```

---

## 📊 实施阶段

### Phase 1: API 验证 (2 天)
- [ ] 编译 libgpreempt_poc.so
- [ ] 验证 suspend/resume API 可用
- [ ] 测试 Queue ID 获取

### Phase 2: 队列识别 (2 天)
- [ ] MQD debugfs 解析
- [ ] 按优先级分类
- [ ] 自动发现机制

### Phase 3: Test Framework (2 天)
- [ ] Python GPreemptScheduler
- [ ] AI 模型包装
- [ ] 监控线程

### Phase 4: 测试验证 (2-3 天)
- [ ] 功能测试
- [ ] 性能测试
- [ ] 稳定性测试

---

## 🧪 关键前置实验 ⭐⭐⭐⭐⭐

### 在开始实施前，必须先做这个实验！

**实验文档**: `EXP_Design_01_MQD_HQD_模型关联性实验.md`

**实验目的**: 验证 AI 模型使用的 Queue ID 是否可预测

**核心问题**:
1. 同一模型多次运行，Queue ID 是否一致？
2. 不同模型使用的 Queue 是否不同？
3. 多模型并发时，Queue 是否重合？

**为什么重要**:
- ✅ **如果 Queue ID 可预测** → POC Stage 1 只需 3-5 天（硬编码 Queue ID）
- ⚠️ **如果 Queue ID 不可预测** → POC Stage 1 需要 7-10 天（动态发现机制）

**实验时间**: 约 1.5 小时

**建议**: 在阅读完整方案后，**立即执行此实验**，结果将决定实施策略！

---

## 🎓 前置知识

### 需要理解的概念

**队列层次**:
- **MQD** (Memory Queue Descriptor) = 软件队列，在内存中
- **HQD** (Hardware Queue Descriptor) = 硬件队列，在 GPU 寄存器中
- **关系**: 1 个 MQD → 1 个 HQD (1:1 映射)

**Doorbell**:
- 用户空间 MMIO，直接提交 kernel
- 不通知 KFD，KFD 需要主动监控
- 优势：快（~100ns），缺点：KFD 不感知

**CWSR** (Compute Wave Save/Restore):
- AMD GPU 的 Wave-level 上下文切换
- 硬件自动保存 Wave 状态（PC, SGPR, VGPR, LDS）
- `suspend_queues` 内部自动触发

**CPSCH** (CP Scheduler):
- Command Processor Scheduler
- GPU firmware，管理队列调度
- 处理 MAP_QUEUES / UNMAP_QUEUES PM4 命令

---

## 🔧 环境准备

### Docker 容器

**推荐**: zhenaiter 容器

```bash
容器名:     zhenaiter
ROCm:      6.4
PyTorch:   2.9.1+rocm6.4
GPU:       8× AMD Instinct MI308X
Conda:     flashinfer-rocm (micromamba)
```

**进入方式**:
```bash
docker exec -it zhenaiter /bin/bash
export MAMBA_EXE='/root/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/root/micromamba'
eval "$(/root/.local/bin/micromamba shell hook --shell=bash)"
micromamba activate flashinfer-rocm
```

### 所需权限

```bash
# 需要 root 权限访问 debugfs
sudo cat /sys/kernel/debug/kfd/mqds
sudo cat /sys/kernel/debug/kfd/hqds

# 或添加到 sudo 组
sudo usermod -aG sudo $USER
```

---

## 📊 预期成果

### 成功标准

**功能**:
- [x] Online 任务能抢占 Offline 任务
- [x] Offline 任务能正确恢复
- [x] 系统稳定运行（无崩溃）

**性能**:
- [ ] Online 延迟 < 50ms (可接受)
- [ ] Online 延迟 < 10ms (理想)
- [ ] Offline 吞吐量损失 < 20%

### 交付物

**代码**:
- `libgpreempt_poc/` - C 库
  - gpreempt_poc.c/h
  - mqd_parser.c
  - hqd_monitor.c
  - Makefile

- `test_framework/` - Python 框架
  - gpreempt_scheduler.py
  - ai_model_wrapper.py
  - test_priority_scheduling.py

- `tests/` - 测试用例
  - test_basic_preemption.py
  - test_latency.py
  - test_throughput.py

**文档**:
- 测试报告
- 性能分析
- Stage 2 实施建议

---

## ❓ 常见问题

### Q1: 为什么选择 suspend_queues 而不是直接用 CWSR？

**A**: 
- suspend_queues 已有完整实现，无需修改内核
- 能快速验证概念（1-2 周）
- 如果性能满足要求，就不需要更复杂的方案
- 如果不满足，Stage 2 再升级到 CWSR 直接使用

### Q2: Queue ID 获取为什么这么复杂？

**A**: 
- KFD 没有提供标准的 Queue ID 查询接口
- `/sys/kernel/debug/kfd/queues` 在某些版本不存在
- 需要解析 `/sys/kernel/debug/kfd/mqds` 或暴力枚举

### Q3: MQD 和 HQD 有什么区别？

**A**: 
- **MQD**: 软件层的队列描述符（在内存中）
- **HQD**: 硬件层的队列寄存器（在 GPU 中）
- **关系**: 1:1 映射，但状态可能不同步
- **POC 使用**: 只需要 MQD Queue ID，HQD 用于验证

### Q4: 为什么不直接用 amdgpu_amdkfd_stop_sched？

**A**: 
- 粒度太粗（停止所有队列，包括 Online）
- 不能区分 Online/Offline
- 不适合本测试场景

### Q5: 需要修改 KFD 内核代码吗？

**A**: 
- **Stage 1**: ❌ 不需要（使用已有的 suspend_queues）
- **Stage 2**: ✅ 需要（新增 CWSR 直接调用接口）
- **Production**: ✅ 需要（完整的内核态调度器）

---

## 🎓 学习路径

### 新手入门 (2-3 小时)

```
1. 阅读 test_scenaria.md (5分钟)
   ↓ 了解测试场景

2. 阅读 ARCH_Design_01 的"系统架构"部分 (20分钟)
   ↓ 理解整体流程

3. 阅读 ARCH_Design_02 的"API 对比"部分 (15分钟)
   ↓ 理解为什么选 suspend_queues

4. 阅读 ARCH_Design_03 的"Queue ID 获取"部分 (20分钟)
   ↓ 理解如何找到队列

5. 动手测试 Queue ID 获取 (1-2小时)
   ↓ 在 Docker 中实际操作
```

### 进阶学习 (半天)

```
1. 详读 ARCH_Design_01 完整方案
2. 详读 ARCH_Design_04 HQD 技术细节
3. 研究 POC_Stage1_TODOLIST.md 任务分解
4. 开始实施 Phase 1
```

---

## 🔗 相关资源

### 历史测试经验

- `../DOC_GPREEMPT/MI300_Testing/` - CWSR 抢占测试
  - QUEUE_ID_SOLUTION.md - Queue ID 获取经验
  - Docker容器内端到端测试_快速开始.md - Docker 测试指南

- `../XSCHED/tests/` - XSched AI 模型测试
  - test_bert_with_xsched_api.py - BERT 测试脚本
  - RUN_IN_DOCKER.md - Docker 运行指南

### 技术背景

- `../DOC_GPREEMPT/TODOLIST.md` - 完整 GPREEMPT 实施计划
- `../DOC_GPREEMPT/CWSR_API_USAGE_REFERENCE.md` - CWSR API 参考
- `../DOC_GPREEMPT/CRIU_CODE_REUSE_ANALYSIS.md` - CRIU 代码复用

---

## 📞 需要帮助？

### 问题诊断

**找不到 Queue ID**:
- → 阅读 ARCH_Design_03 的"暴力枚举"部分
- → 使用 QUEUE_ID_SOLUTION.md 中的脚本

**不理解 HQD**:
- → 阅读 ARCH_Design_04
- → HQD 主要用于验证，POC 中可选

**不确定 Docker 环境**:
- → 阅读 ARCH_Design_03 的"Docker 环境配置"
- → 使用 zhenaiter 容器（已验证）

**性能不满足要求**:
- → 如果延迟 > 50ms，考虑升级到 Stage 2
- → 阅读 ARCH_Design_02 的"Stage 2 方案"

---

## ✅ 检查清单

**开始前**:
- [ ] 已读 test_scenaria.md
- [ ] 已读 ARCH_Design_01 或 ARCH_Design_02
- [ ] 理解 MQD vs HQD 的区别
- [ ] 知道如何获取 Queue ID

**实施中**:
- [ ] zhenaiter 容器可以访问
- [ ] 能读取 /sys/kernel/debug/kfd/mqds
- [ ] 能编译 C 库
- [ ] 能运行测试脚本

**完成后**:
- [ ] 所有功能测试通过
- [ ] 性能数据收集完成
- [ ] 测试报告已生成
- [ ] 已决定是否升级到 Stage 2

---

## 📈 时间线

```
Week 1:
  Day 1-2: Phase 1 - API 验证
  Day 3-4: Phase 2 - Queue ID 机制
  Day 5:   Phase 3 - Framework (部分)

Week 2:
  Day 6-7: Phase 3 完成 + Phase 4 开始
  Day 8-9: Phase 4 - 测试和验证
  Day 10:  报告和文档
```

---

## 🎯 下一步

### 如果你是第一次接触：

1. ✅ 阅读本 README
2. ✅ 阅读 test_scenaria.md
3. ✅ 阅读 ARCH_Design_01 (至少前半部分)
4. → 决定是否开始实施

### 如果你准备好开始：

1. ✅ 进入 zhenaiter 容器
2. ✅ 测试 Queue ID 获取
3. ✅ 开始 Phase 1 - API 验证
4. → 按照 POC_Stage1_TODOLIST.md 执行

### 如果你遇到问题：

1. → 查看 ARCH_Design_03 (Queue ID)
2. → 查看 ARCH_Design_04 (HQD)
3. → 查看历史测试文档 (DOC_GPREEMPT/MI300_Testing/)

---

**最后更新**: 2026-02-03  
**维护者**: Zhehan

**状态**: 📋 **文档完成，可以开始实施！**

---

## 📖 附录：文档完整列表

```
DOC_POC_stage1/
├── README_START_HERE.md                          ← 本文档 ⭐ 从这开始
├── test_scenaria.md                              ← 测试场景
├── EXP_Design_01_MQD_HQD_模型关联性实验.md       ← ⭐⭐⭐⭐⭐ 必做实验！
├── ARCH_Design_01_POC_Stage1_实施方案.md          ← 完整方案
├── ARCH_Design_02_三种API技术对比.md              ← 技术选型
├── ARCH_Design_03_QueueID获取与环境配置.md        ← Queue ID
├── ARCH_Design_04_HQD信息获取完整指南.md          ← HQD 详解
└── POC_Stage1_TODOLIST.md                        ← 任务清单

总页数: ~3500 行
阅读时间: 1-2 小时（速读）/ 4-5 小时（精读）
```

**建议执行顺序**:
1. README_START_HERE.md (本文档) - 10 分钟
2. test_scenaria.md - 2 分钟
3. **EXP_Design_01** - 20 分钟阅读 + 1.5 小时实验 ⭐ **先做这个！**
4. 根据实验结果，选择阅读：
   - 如果 Queue ID 可预测 → 简化版实施（3 天）
   - 如果 Queue ID 不可预测 → 完整版实施（7 天）
5. ARCH_Design_01 - 30 分钟
6. ARCH_Design_02 - 20 分钟
7. POC_Stage1_TODOLIST.md - 20 分钟

**总计**: ~2 小时阅读 + 1.5 小时实验 = 3.5 小时

---

**🚀 准备好了吗？从阅读 test_scenaria.md 开始！**
