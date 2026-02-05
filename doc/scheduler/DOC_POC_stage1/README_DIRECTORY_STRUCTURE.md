# POC Stage 1 - 目录结构说明

**最后更新**: 2026-02-05

---

## 📁 目录组织

```
DOC_POC_stage1/
│
├── code/                               # 📦 所有代码文件 ⭐⭐⭐
│   │
│   ├── README.md                       # 代码目录详细说明
│   ├── Makefile                        # 编译脚本
│   │
│   ├── C++ POC工具 (推荐)
│   │   ├── kfd_queue_monitor.hpp      # Queue监控器头文件
│   │   ├── kfd_queue_monitor.cpp      # Queue监控器实现
│   │   ├── queue_monitor_main.cpp     # 监控工具主程序
│   │   └── kfd_preemption_poc.cpp     # 抢占POC工具
│   │
│   ├── C测试工具
│   │   ├── get_queue_info.c           # 快速Queue查询
│   │   ├── test_gpreempt_ioctl.c      # IOCTL测试
│   │   ├── preempt_queue_manual.c     # 手动抢占v1
│   │   └── preempt_queue_manual_v2.c  # 手动抢占v2
│   │
│   └── Shell测试脚本
│       ├── test_userspace_poc.sh      # ⭐ 一键测试
│       ├── exp01_with_api.sh          # ⭐ 实验1 (API)
│       ├── exp01_redesigned.sh        # 实验1 (重设计)
│       ├── exp01_queue_monitor_fixed.sh # 实验1 (修复版)
│       ├── host_queue_test.sh         # 宿主机测试
│       ├── host_queue_consistency_test.sh # 一致性测试
│       ├── quick_queue_test.sh        # 快速测试
│       ├── quick_queue_test_hip.sh    # HIP测试
│       ├── reliable_queue_test.sh     # 可靠测试
│       ├── test_queue_consistency.sh  # 一致性验证
│       └── fix_debugfs.sh             # 调试修复
│
└── 文档文件 (主目录)
    │
    ├── 00_INDEX_POC_TOOLS_AND_DOCS.md  # ⭐⭐⭐ 总索引（从这里开始）
    │
    ├── 用户指南
    │   ├── README_USERSPACE_POC.md     # ⭐⭐⭐ 完整使用指南 (618行)
    │   ├── QUICKSTART_CPP_POC.md       # ⭐ 5分钟快速开始
    │   ├── CPP_POC_FILES_SUMMARY.md    # 工具集总结
    │   ├── QUICK_START.md              # 快速开始 (早期)
    │   ├── README_QUEUE_TEST.md        # Queue测试说明
    │   └── EXP01_QUICK_START.md        # 实验1快速开始
    │
    ├── API文档
    │   ├── GET_QUEUE_SNAPSHOT_API_GUIDE.md  # ⭐ GET_QUEUE_SNAPSHOT API
    │   └── POC_SUSPEND_QUEUES_API_GUIDE.md  # Suspend/Resume API
    │
    ├── 架构设计
    │   ├── ARCH_Design_01_POC_Stage1_实施方案.md
    │   ├── ARCH_Design_02_三种API技术对比.md
    │   ├── ARCH_Design_03_QueueID获取与环境配置.md
    │   └── ARCH_Design_04_HQD信息获取完整指南.md
    │
    ├── 深度分析
    │   ├── New_DEEP_DIVE_MI308X_QUEUE_MECHANISMS.md
    │   ├── New_DESIGN_MAP_UNMAP_BASED_PREEMPTION.md
    │   ├── New_SW_QUEUE_HW_QUEUE_MAPPING_MECHANISM.md
    │   ├── New_IMPLEMENTATION_COMPARISON.md
    │   └── New_MAP_UNMAP_DETAILED_PROCESS.md
    │
    ├── 实验设计
    │   ├── EXP_01_QUEUE_USAGE_ANALYSIS.md
    │   └── EXP_Design_01_MQD_HQD_模型关联性实验.md
    │
    ├── 问题修复
    │   ├── CRITICAL_FIX_MQD_FORMAT.md
    │   ├── ISSUE_FIX_EXP01.md
    │   └── CHANGELOG_CPSCH_CORRECTION.md
    │
    ├── 内核机制
    │   ├── KERNEL_DOC_MQD_HQD_ANALYSIS.md
    │   ├── AQL_QUEUE_VS_MQD_RELATIONSHIP.md
    │   └── QUEUE_CREATION_TIMELINE.md
    │
    ├── 配置和硬件
    │   ├── KCQ_CONFIG_GUIDE.md
    │   └── MI308X_HARDWARE_INFO.md
    │
    └── 项目管理
        ├── POC_ROADMAP_WITH_EXPERIMENTS.md
        ├── POC_Stage1_TODOLIST.md
        └── CURRENT_STATUS_AND_NEXT_STEPS.md
```

---

## 🚀 快速导航

### 🎯 我想...

#### 1. 开始使用POC工具

```bash
# 步骤1: 进入代码目录
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 步骤2: 阅读快速开始
cd .. && cat QUICKSTART_CPP_POC.md

# 步骤3: 一键测试
cd code && ./test_userspace_poc.sh
```

**相关文档**:
- `QUICKSTART_CPP_POC.md` - 5分钟快速开始
- `code/README.md` - 代码目录说明
- `README_USERSPACE_POC.md` - 完整使用指南

---

#### 2. 编译和使用工具

```bash
# 进入代码目录
cd code/

# 编译
make clean && make all

# 使用
sudo ./queue_monitor <pid> 30 5
sudo ./kfd_preemption_poc <pid> 100
```

**相关文档**:
- `code/README.md` - 编译和使用说明
- `code/Makefile` - 编译配置

---

#### 3. 了解API

**阅读顺序**:
1. `GET_QUEUE_SNAPSHOT_API_GUIDE.md` ⭐ - 如何获取Queue ID
2. `POC_SUSPEND_QUEUES_API_GUIDE.md` - 如何Suspend/Resume

---

#### 4. 深入了解机制

**阅读顺序**:
1. `KERNEL_DOC_MQD_HQD_ANALYSIS.md` - MQD/HQD基础
2. `New_DEEP_DIVE_MI308X_QUEUE_MECHANISMS.md` - MI308X深入分析
3. `New_DESIGN_MAP_UNMAP_BASED_PREEMPTION.md` - 优化设计

---

#### 5. 运行实验

```bash
cd code/

# 实验1 (使用API，推荐)
./exp01_with_api.sh

# 实验1 (解析debugfs)
./exp01_redesigned.sh
```

**相关文档**:
- `EXP_01_QUEUE_USAGE_ANALYSIS.md` - 实验1设计
- `EXP01_QUICK_START.md` - 实验1快速开始

---

#### 6. 查找所有文档

**查看总索引**:
```bash
cat 00_INDEX_POC_TOOLS_AND_DOCS.md
```

这个文件包含所有文档和工具的完整索引。

---

## 📊 统计信息

```
总文件数: ~61

代码文件 (code/ 目录): 21 个
  - C++ 文件: 4 个 (*.cpp, *.hpp)
  - C 文件: 4 个 (*.c)
  - Shell 脚本: 11 个 (*.sh)
  - Makefile: 1 个
  - README: 1 个

文档文件 (主目录): ~40 个
  - 用户指南: 6 个
  - API 文档: 2 个
  - 架构设计: 4 个
  - 深度分析: 5 个
  - 实验设计: 2 个
  - 问题修复: 3 个
  - 内核机制: 3 个
  - 配置和硬件: 2 个
  - 项目管理: 3 个
  - 其他: ~10 个
```

---

## 🎓 文档层次

### 入门级 ⭐
适合第一次使用的用户

1. `00_INDEX_POC_TOOLS_AND_DOCS.md` - 总索引
2. `QUICKSTART_CPP_POC.md` - 快速开始
3. `code/README.md` - 代码说明

### 中级 ⭐⭐
适合日常开发

1. `README_USERSPACE_POC.md` - 完整使用指南
2. `GET_QUEUE_SNAPSHOT_API_GUIDE.md` - API详解
3. `code/` - 代码实现

### 高级 ⭐⭐⭐
适合深入研究

1. `New_DEEP_DIVE_MI308X_QUEUE_MECHANISMS.md` - 机制分析
2. `New_DESIGN_MAP_UNMAP_BASED_PREEMPTION.md` - 优化设计
3. `KERNEL_DOC_MQD_HQD_ANALYSIS.md` - 内核机制

---

## 💡 推荐阅读路径

### 路径1: 快速上手（15分钟）

1. `QUICKSTART_CPP_POC.md` (5分钟)
2. `cd code/ && ./test_userspace_poc.sh` (10分钟)

### 路径2: 深入了解（1小时）

1. `README_USERSPACE_POC.md` (20分钟)
2. `GET_QUEUE_SNAPSHOT_API_GUIDE.md` (20分钟)
3. `code/README.md` (10分钟)
4. 实际测试 (10分钟)

### 路径3: 全面掌握（4小时）

1. 快速上手 (15分钟)
2. 深入了解 (1小时)
3. 阅读架构设计文档 (1小时)
4. 阅读深度分析文档 (1小时)
5. 运行实验和POC测试 (1小时)

---

## 🔗 重要链接

**代码**: `code/`
- 一键测试: `code/test_userspace_poc.sh`
- Makefile: `code/Makefile`
- 代码说明: `code/README.md`

**文档**: 主目录
- 总索引: `00_INDEX_POC_TOOLS_AND_DOCS.md` ⭐⭐⭐
- 快速开始: `QUICKSTART_CPP_POC.md` ⭐
- 完整指南: `README_USERSPACE_POC.md` ⭐⭐⭐

---

## 📝 最近更新

### 2026-02-05: 目录重组

✅ **代码文件移至 `code/` 子目录**
- 所有C/C++源文件
- 所有Shell脚本
- Makefile
- 创建 `code/README.md`

✅ **文档保留在主目录**
- 更清晰的组织结构
- 更容易查找文档

✅ **更新索引**
- `00_INDEX_POC_TOOLS_AND_DOCS.md` - 更新路径
- `README_DIRECTORY_STRUCTURE.md` - 新增目录说明

---

**工作目录**: `/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1`

**代码目录**: `code/` ⭐

**开始**: `cd code/ && ./test_userspace_poc.sh`
