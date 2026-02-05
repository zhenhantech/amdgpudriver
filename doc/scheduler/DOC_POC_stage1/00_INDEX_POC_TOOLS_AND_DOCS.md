# POC Stage 1 - 工具和文档索引

**最后更新**: 2026-02-05  
**目录**: `/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1`

---

## 🎯 快速开始（推荐路径）

### 新用户 - 从这里开始 ⭐⭐⭐

1. **QUICKSTART_CPP_POC.md** - 5分钟快速开始C++版本POC
2. **README_USERSPACE_POC.md** - 完整的用户空间POC使用指南
3. **test_userspace_poc.sh** - 一键测试脚本

### 了解API

4. **GET_QUEUE_SNAPSHOT_API_GUIDE.md** - Queue ID获取API详解
5. **POC_SUSPEND_QUEUES_API_GUIDE.md** - Suspend/Resume API详解

---

## 📦 可执行工具（需编译）

### 代码目录

**所有代码文件已移至**: `code/` 子目录

### 编译所有工具

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
make clean && make all
```

### C++ POC工具（推荐）⭐

**位置**: `code/`

| 工具 | 源文件 | 用途 |
|------|--------|------|
| `queue_monitor` | `code/queue_monitor_main.cpp` + `code/kfd_queue_monitor.cpp` | Queue监控和统计分析 |
| `kfd_preemption_poc` | `code/kfd_preemption_poc.cpp` + `code/kfd_queue_monitor.cpp` | 抢占POC测试 |
| `get_queue_info` | `code/get_queue_info.c` | 快速Queue信息查询 |

**C++ 库文件**:
- `code/kfd_queue_monitor.hpp` - 头文件
- `code/kfd_queue_monitor.cpp` - 实现

### C 测试工具（早期版本）

**位置**: `code/`

| 工具 | 源文件 | 用途 |
|------|--------|------|
| `test_gpreempt_ioctl` | `code/test_gpreempt_ioctl.c` | 测试自定义IOCTL |
| `preempt_queue_manual` | `code/preempt_queue_manual.c` | 手动抢占测试v1 |
| `preempt_queue_manual_v2` | `code/preempt_queue_manual_v2.c` | 手动抢占测试v2 |

---

## 🔧 Shell测试脚本

**位置**: `code/`

### 一键测试脚本

- **code/test_userspace_poc.sh** ⭐⭐⭐ - 一键编译+测试C++ POC工具
- **code/test_queue_consistency.sh** - Queue一致性测试

### 实验脚本（Experiment 1）

- **code/exp01_with_api.sh** ⭐ - 使用API的实验1（推荐）
- **code/exp01_redesigned.sh** - 重新设计的实验1（解析debugfs）
- **code/exp01_queue_monitor_fixed.sh** - 修复版实验1

### Queue测试脚本

- **code/host_queue_test.sh** - 宿主机Queue测试
- **code/host_queue_consistency_test.sh** - 宿主机一致性测试
- **code/quick_queue_test.sh** - 快速Queue测试（PyTorch）
- **code/quick_queue_test_hip.sh** - 快速Queue测试（HIP）
- **code/reliable_queue_test.sh** - 可靠的Queue测试

### 调试脚本

- **code/fix_debugfs.sh** - Debugfs诊断和修复

---

## 📚 核心文档

### 1. 用户空间POC文档（最新）⭐⭐⭐

| 文档 | 内容 | 行数 |
|------|------|------|
| **README_USERSPACE_POC.md** | 完整的C++ POC使用指南 | 618 |
| **QUICKSTART_CPP_POC.md** | 5分钟快速开始 | 315 |
| **CPP_POC_FILES_SUMMARY.md** | C++ POC工具集总结 | ~330 |

### 2. API参考文档

| 文档 | 内容 | 行数 |
|------|------|------|
| **GET_QUEUE_SNAPSHOT_API_GUIDE.md** | GET_QUEUE_SNAPSHOT API详解 | 583 |
| **POC_SUSPEND_QUEUES_API_GUIDE.md** | Suspend/Resume API详解 | - |

### 3. 架构设计文档

| 文档 | 内容 |
|------|------|
| **ARCH_Design_01_POC_Stage1_实施方案.md** | POC Stage 1总体方案 |
| **ARCH_Design_02_三种API技术对比.md** | API技术对比 |
| **ARCH_Design_03_QueueID获取与环境配置.md** | Queue ID获取方法 |
| **ARCH_Design_04_HQD信息获取完整指南.md** | HQD信息获取 |

### 4. 深度分析文档

| 文档 | 内容 |
|------|------|
| **New_DEEP_DIVE_MI308X_QUEUE_MECHANISMS.md** | MI308X队列机制深入分析 |
| **New_DESIGN_MAP_UNMAP_BASED_PREEMPTION.md** | Map/Unmap优化抢占设计 |
| **New_SW_QUEUE_HW_QUEUE_MAPPING_MECHANISM.md** | 软硬件队列映射机制 |
| **New_IMPLEMENTATION_COMPARISON.md** | 实现方案对比 |
| **New_MAP_UNMAP_DETAILED_PROCESS.md** | Map/Unmap详细流程 |

### 5. 实验设计文档

| 文档 | 内容 |
|------|------|
| **EXP_01_QUEUE_USAGE_ANALYSIS.md** | 实验1：Queue使用分析 |
| **EXP_Design_01_MQD_HQD_模型关联性实验.md** | MQD/HQD关联性实验设计 |
| **EXP01_QUICK_START.md** | 实验1快速开始 |

### 6. 问题修复和调试文档

| 文档 | 内容 |
|------|------|
| **CRITICAL_FIX_MQD_FORMAT.md** | MQD格式理解的关键修正 |
| **ISSUE_FIX_EXP01.md** | 实验1问题修复 |
| **CHANGELOG_CPSCH_CORRECTION.md** | CPSCH概念修正 |

### 7. 内核机制文档

| 文档 | 内容 |
|------|------|
| **KERNEL_DOC_MQD_HQD_ANALYSIS.md** | 内核MQD/HQD机制分析 |
| **AQL_QUEUE_VS_MQD_RELATIONSHIP.md** | AQL Queue与MQD关系 |
| **QUEUE_CREATION_TIMELINE.md** | Queue创建时间线 |

### 8. 配置和硬件文档

| 文档 | 内容 |
|------|------|
| **KCQ_CONFIG_GUIDE.md** | KCQ配置指南 |
| **MI308X_HARDWARE_INFO.md** | MI308X硬件信息 |

### 9. 项目管理文档

| 文档 | 内容 |
|------|------|
| **POC_ROADMAP_WITH_EXPERIMENTS.md** | POC路线图 |
| **POC_Stage1_TODOLIST.md** | POC Stage 1任务列表 |
| **CURRENT_STATUS_AND_NEXT_STEPS.md** | 当前状态和下一步 |

### 10. 旧版快速开始文档

| 文档 | 内容 |
|------|------|
| **QUICK_START.md** | 早期快速开始（debugfs方式） |
| **QUICKSTART_实验立即开始.md** | 实验快速开始 |
| **README_QUEUE_TEST.md** | Queue测试说明 |
| **New_QUICK_START_GUIDE.md** | 新版快速开始 |
| **README_START_HERE.md** | 总体起点文档 |

---

## 🗂️ 文件统计

### 目录结构

```
DOC_POC_stage1/
├── code/                    # 所有代码文件 ⭐
│   ├── *.cpp, *.hpp        # C++ 源文件
│   ├── *.c                 # C 源文件
│   ├── *.sh                # Shell 脚本
│   ├── Makefile            # 编译脚本
│   └── README.md           # 代码目录说明
│
└── *.md                     # 文档文件

总文件数: ~61
```

### 按类型分类

```
代码文件 (code/ 目录):
  - C++ 源文件: 3 个 (*.cpp)
  - C++ 头文件: 1 个 (*.hpp)
  - C 源文件: 4 个 (*.c)
  - Shell 脚本: 11 个 (*.sh)
  - Makefile: 1 个
  - 代码说明: 1 个 (code/README.md)
  
文档文件 (主目录):
  - Markdown 文档: ~40 个 (*.md)
  
其他:
  - 分析脚本: analyze_*.py (如有)
```

### 按功能分类

```
用户空间POC工具: 10 个文件
  - C++ 库和工具: 4 个
  - C 工具: 4 个
  - Makefile: 1 个
  - 一键测试脚本: 1 个

实验脚本: 11 个文件

文档:
  - 用户指南: 3 个
  - API 文档: 2 个
  - 架构设计: 4 个
  - 深度分析: 5 个
  - 实验设计: 3 个
  - 调试和修复: 3 个
  - 内核机制: 3 个
  - 配置和硬件: 2 个
  - 项目管理: 3 个
  - 其他快速开始: 5 个
```

---

## 🚀 推荐工作流

### 第一次使用

1. 阅读 **QUICKSTART_CPP_POC.md**
2. 进入代码目录: `cd code/`
3. 运行 `./test_userspace_poc.sh`
4. 查看 **README_USERSPACE_POC.md** 了解详细用法

### 日常开发

1. 进入代码目录: `cd code/`
2. 编译工具: `make clean && make all`
3. 监控Queue: `sudo ./queue_monitor <pid> 30 5`
4. 测试抢占: `sudo ./kfd_preemption_poc <pid> 100`

### 深入研究

1. 阅读 **GET_QUEUE_SNAPSHOT_API_GUIDE.md** 了解API
2. 阅读 **New_DEEP_DIVE_MI308X_QUEUE_MECHANISMS.md** 了解机制
3. 阅读 **New_DESIGN_MAP_UNMAP_BASED_PREEMPTION.md** 了解优化方案

---

## 📝 最近更新（2026-02-05）

### 新增内容

✅ **C++ 用户空间POC工具集**
- `kfd_queue_monitor.hpp/cpp` - Queue监控库
- `queue_monitor_main.cpp` - 监控工具
- `kfd_preemption_poc.cpp` - 抢占POC工具
- `get_queue_info.c` - 快速查询工具

✅ **完整文档**
- `README_USERSPACE_POC.md` - 618行完整指南
- `QUICKSTART_CPP_POC.md` - 5分钟快速开始
- `CPP_POC_FILES_SUMMARY.md` - 工具集总结

✅ **测试脚本**
- `test_userspace_poc.sh` - 一键测试
- `exp01_with_api.sh` - 基于API的实验1

✅ **API文档**
- `GET_QUEUE_SNAPSHOT_API_GUIDE.md` - 完整的API使用指南

### 文件组织

所有最近20小时内创建的代码和文档已从  
`/mnt/md0/zhehan/code/flashinfer/dockercode/gpreempt_test`  
移动到  
`/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1`

---

## 🎯 核心成果

### 用户空间POC能力

✅ **完全在用户空间实施Queue抢占**
- 无需内核模块
- 使用KFD Debug Trap API
- 性能优异（Suspend ~400μs, Resume ~300μs）

✅ **完整的监控和分析工具**
- Queue使用情况监控
- 统计分析（稳定性、频率）
- POC代码自动生成

✅ **专业的C++实现**
- 现代C++17设计
- 易于扩展和维护
- 完整的API封装

---

## 💡 下一步建议

1. **测试工具**: 运行 `./test_userspace_poc.sh`
2. **实际应用**: 用真实AI模型测试
3. **性能优化**: 分析和优化Suspend/Resume延迟
4. **集成**: 将API集成到实际的GPU调度系统

---

## 📞 如何获取帮助

1. 查看对应的README文档
2. 运行测试脚本查看示例输出
3. 阅读API文档了解详细参数

---

**工作目录**: `/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1`

**代码目录**: `code/` ⭐

**快速开始**: `cd code/ && ./test_userspace_poc.sh`

**完整指南**: `README_USERSPACE_POC.md`

**代码说明**: `code/README.md`
