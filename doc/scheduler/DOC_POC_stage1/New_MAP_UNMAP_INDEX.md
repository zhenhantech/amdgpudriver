# Map/Unmap机制分析 - 文档索引

**日期**: 2026-02-03  
**主题**: 软件队列(MQD)到硬件队列(HQD)的Map/Unmap机制  
**GPU**: MI308X (GFX v9.4.3)

---

## 📚 文档导航

### 🎯 推荐阅读顺序

#### 第1步：快速了解（5-10分钟）
👉 **`MAP_UNMAP_VISUAL_GUIDE.md`**
- 图表和可视化说明
- 核心概念演示
- 时序图
- **最适合首次阅读**

#### 第2步：中文详细说明（15-20分钟）
👉 **`MAP_UNMAP_SUMMARY_CN.md`**
- 核心概念中文解释
- 触发时机详解
- DSV3.2测试数据解释
- 完整流程示例

#### 第3步：完整机制（30-40分钟）
👉 **`SW_QUEUE_HW_QUEUE_MAPPING_MECHANISM.md`**
- 队列创建的6个阶段详解
- MQD vs HQD深入分析
- 性能考量和优化
- 常见问题调试

#### 第4步：深入代码实现（40-60分钟）
👉 **`MAP_UNMAP_DETAILED_PROCESS.md`**
- 函数调用链完整分析
- Packet Manager机制
- Grace Period和Preemption
- HWS通信细节

---

## 📖 文档内容对照表

| 主题 | Visual Guide | Summary CN | Mapping Mechanism | Detailed Process |
|------|-------------|-----------|------------------|-----------------|
| **基础概念** |
| MQD vs HQD | ✓ 图表 | ✓ 详解 | ✓ 深入 | ✓ 代码 |
| Map定义 | ✓ 可视化 | ✓ 中文 | ✓ 完整 | ✓ 实现 |
| Unmap定义 | ✓ 可视化 | ✓ 中文 | ✓ 完整 | ✓ 实现 |
| **流程分析** |
| 创建队列 | ✓ 流程图 | ✓ 示例 | ✓ 6阶段 | ✓ 代码链 |
| Map操作 | ✓ 时序图 | ✓ 触发时机 | ✓ 详细流程 | ✓ 函数调用 |
| Unmap操作 | ✓ 时序图 | ✓ 触发时机 | ✓ 详细流程 | ✓ 函数调用 |
| 销毁队列 | ✓ 步骤图 | ✓ 示例 | ✓ 完整流程 | ✓ 代码 |
| **多XCC** |
| 1→4映射 | ✓ 图示 | ✓ 解释 | ✓ 代码证据 | ✓ load_mqd_v9_4_3 |
| 多XCC原因 | ✓ 说明 | ✓ 详解 | ✓ 设计理念 | ✓ 代码分析 |
| **HWS机制** |
| HIQ通信 | ✓ 链路图 | ✓ 简述 | ✓ 详解 | ✓ packet_manager |
| Runlist IB | ✓ 构建图 | ✓ 示例 | ✓ 内容说明 | ✓ pm_send_runlist |
| Packet提交 | ✓ 时序图 | － | ✓ 流程 | ✓ kq_submit_packet |
| **资源管理** |
| HQD分配 | ✓ 位图图 | ✓ 策略 | ✓ Round-robin | ✓ allocate_hqd |
| HQD释放 | ✓ 示例 | ✓ 流程 | ✓ 位图操作 | ✓ deallocate_hqd |
| 超额订阅 | ✓ 场景图 | ✓ 示例 | ✓ 设计理念 | － |
| **高级主题** |
| Grace Period | － | － | ✓ 简述 | ✓ 详解 |
| Preemption | － | － | ✓ 状态图 | ✓ 类型分析 |
| CWSR | － | － | ✓ 简述 | ✓ 详细机制 |
| **调试** |
| 常见问题 | － | － | ✓ 列表 | ✓ 详细排查 |
| 调试方法 | － | － | ✓ 命令 | ✓ 代码位置 |
| **代码** |
| 函数列表 | － | － | ✓ 表格 | ✓ 详细 |
| 代码位置 | － | － | ✓ 行号 | ✓ 完整路径 |

---

## 🎯 按需求选择文档

### 需求1: "我就想快速了解是怎么回事"
**推荐**: `MAP_UNMAP_VISUAL_GUIDE.md`
- 10分钟读完
- 图表直观
- 抓住核心要点

### 需求2: "我想理解为什么DSV3.2测试中MQD=80，HQD=288"
**推荐**: `MAP_UNMAP_SUMMARY_CN.md`
- 包含DSV3.2数据解释
- 中文详细说明
- 实际案例分析

### 需求3: "我要了解完整的队列生命周期"
**推荐**: `SW_QUEUE_HW_QUEUE_MAPPING_MECHANISM.md`
- 6个阶段完整分析
- 状态转换图
- 性能优化建议

### 需求4: "我要看代码实现细节"
**推荐**: `MAP_UNMAP_DETAILED_PROCESS.md`
- 函数调用链
- 代码位置和行号
- Packet Manager详解

### 需求5: "为什么1个队列在4个XCC都要加载？"
**推荐**: 所有文档都有覆盖
- Visual Guide: 图示说明
- Summary CN: 中文详解
- Mapping Mechanism: 设计理念
- Detailed Process: load_mqd_v9_4_3()代码

### 需求6: "Map/Unmap什么时候触发？"
**推荐**: `MAP_UNMAP_SUMMARY_CN.md` 的"触发时机"章节
- 5种Map触发场景
- 5种Unmap触发场景
- 实际代码位置

---

## 📊 关键代码快速索引

| 操作 | 函数名 | 文件 | 行号 | 详细说明在 |
|------|-------|------|------|----------|
| **队列创建** | `create_queue_cpsch` | kfd_device_queue_manager.c | 2050 | Mapping Mechanism |
| **HQD分配** | `allocate_hqd` | kfd_device_queue_manager.c | 777 | 所有文档 |
| **HQD释放** | `deallocate_hqd` | kfd_device_queue_manager.c | 811 | 所有文档 |
| **批量Map** | `map_queues_cpsch` | kfd_device_queue_manager.c | 2200 | Detailed Process |
| **批量Unmap** | `unmap_queues_cpsch` | kfd_device_queue_manager.c | 2353 | Detailed Process |
| **Unmap+Map** | `execute_queues_cpsch` | kfd_device_queue_manager.c | 2442 | Detailed Process |
| **Load MQD(单XCC)** | `load_mqd` | kfd_mqd_manager_v9.c | 278 | Mapping Mechanism |
| **Load MQD(多XCC)** | `load_mqd_v9_4_3` | kfd_mqd_manager_v9.c | 857 | 所有文档⭐ |
| **发送Runlist** | `pm_send_runlist` | kfd_packet_manager.c | 359 | Detailed Process |
| **发送Unmap** | `pm_send_unmap_queue` | kfd_packet_manager.c | 468 | Detailed Process |
| **队列销毁** | `destroy_queue_cpsch` | kfd_device_queue_manager.c | 2486 | Mapping Mechanism |

---

## 🔍 常见问题快速查找

### "为什么只看到30个队列？"
**文档**: `FINAL_ANSWER_30_QUEUES_MYSTERY.md`
- 因为sysfs显示的是per-XCC的MEC0用户队列数
- 实际每GPU有120个（30 × 4 XCC）

### "另外几个XCC为啥没有看到？"
**文档**: `WHY_ONLY_30_NOT_120_CRITICAL_FINDING.md`
- 因为cp_queue_bitmap初始化时只用了mec_bitmap[0]
- 可能是代码BUG

### "1个MQD对应几个HQD？"
**文档**: `MAP_UNMAP_SUMMARY_CN.md` - "MI308X的特殊性"
- MI308X: 1个MQD → 4个HQD（4个XCC）
- 代码: load_mqd_v9_4_3()遍历所有XCC

### "Map/Unmap什么时候发生？"
**文档**: `MAP_UNMAP_SUMMARY_CN.md` - "Map/Unmap的触发时机"
- Map: 队列创建/激活/恢复时
- Unmap: 队列销毁/deactivate/evict时

### "为什么需要Map/Unmap？"
**文档**: `MAP_UNMAP_SUMMARY_CN.md` - "为什么需要Map/Unmap"
- 硬件队列有限（120/GPU）
- 支持超额订阅
- 动态资源管理

### "Map/Unmap的开销是多少？"
**文档**: `SW_QUEUE_HW_QUEUE_MAPPING_MECHANISM.md` - "性能考量"
- Map: ~20-30 μs
- Unmap: ~20-25 μs

### "HQD是如何分配的？"
**文档**: `MAP_UNMAP_VISUAL_GUIDE.md` - "HQD位图管理"
- Round-robin轮询
- 位图管理
- 负载均衡

---

## 🎓 学习路径建议

### 路径A: 快速掌握（推荐给时间有限的人）
```
1. MAP_UNMAP_VISUAL_GUIDE.md (10分钟)
   └─ 看图表，理解核心概念
   
2. MAP_UNMAP_SUMMARY_CN.md 的"关键要点"部分 (5分钟)
   └─ 抓住4个关键发现
   
总计: 15分钟，掌握80%的内容 ✓
```

### 路径B: 全面理解（推荐给需要深入的人）
```
1. MAP_UNMAP_VISUAL_GUIDE.md (10分钟)
   └─ 建立整体框架
   
2. MAP_UNMAP_SUMMARY_CN.md (20分钟)
   └─ 理解中文详细说明
   
3. SW_QUEUE_HW_QUEUE_MAPPING_MECHANISM.md (40分钟)
   └─ 完整机制和优化
   
4. MAP_UNMAP_DETAILED_PROCESS.md (60分钟)
   └─ 代码实现细节
   
总计: 2.5小时，完全掌握 ✓
```

### 路径C: 代码审查（推荐给需要修改代码的人）
```
1. MAP_UNMAP_SUMMARY_CN.md (快速理解)
   └─ 建立上下文
   
2. MAP_UNMAP_DETAILED_PROCESS.md
   └─ 重点看"代码位置总结"表格
   
3. 直接查看源代码:
   ├─ kfd_device_queue_manager.c (2050, 777, 811, 2200, 2353)
   ├─ kfd_mqd_manager_v9.c (278, 857)
   └─ kfd_packet_manager.c (359, 468)
   
4. 使用ftrace跟踪实际执行
   
总计: 根据需要深入
```

---

## 🔑 核心要点速查

### 1分钟记住这些

```
✅ MQD = 软件队列（系统内存，数量不限）
✅ HQD = 硬件队列（GPU槽位，数量有限：30/XCC）

✅ Map = MQD加载到HQD（队列变active）
✅ Unmap = MQD从HQD卸载（队列变inactive）

✅ 批量操作（Runlist），不是逐个
✅ HWS自动管理，CPU只发packet
✅ MI308X: 1个MQD → 4个HQD（4个XCC）
✅ 支持超额订阅（队列数 > HQD数）
```

### 最重要的3个函数

```c
1. allocate_hqd()     ← 分配硬件队列槽位
2. load_mqd_v9_4_3()  ← 加载MQD到4个XCC的HQD
3. map_queues_cpsch() ← 批量Map所有active队列
```

### 最重要的1个发现

```
MI308X特性：

1个逻辑队列 = 4个物理HQD

为什么？
  - 4个XCC并行架构
  - 任务可能在任何XCC执行
  - 无法预知，所以全部加载
  - 最大化并行度 ⭐⭐⭐⭐⭐
```

---

## 🎨 关键图表索引

### 图表1: MQD→HQD映射关系
**文档**: `MAP_UNMAP_VISUAL_GUIDE.md` - "核心概念可视化"
```
系统内存 (MQD) → GPU Hardware (HQD)
           ↑
        Map操作
```

### 图表2: Map操作流程
**文档**: `MAP_UNMAP_VISUAL_GUIDE.md` - "创建Active队列"
```
分配MQD → 分配HQD → Map到硬件(4个XCC)
```

### 图表3: Unmap操作流程
**文档**: `MAP_UNMAP_VISUAL_GUIDE.md` - "销毁队列"
```
Unmap→HWS处理→释放HQD→释放MQD
```

### 图表4: 1个MQD→4个HQD
**文档**: `MAP_UNMAP_VISUAL_GUIDE.md` - "MI308X多XCC"
```
     MQD 1
       ├─→ XCC 0: HQD[1][3]
       ├─→ XCC 1: HQD[1][3]
       ├─→ XCC 2: HQD[1][3]
       └─→ XCC 3: HQD[1][3]
```

### 图表5: HWS通信链路
**文档**: `MAP_UNMAP_VISUAL_GUIDE.md` - "通信链路"
```
KFD → HIQ → HWS → HQD
```

### 图表6: Map时序图
**文档**: `MAP_UNMAP_VISUAL_GUIDE.md` - "Map操作时序"
```
KFD发送packet → HIQ → HWS处理 → 加载HQD → 更新Fence
```

---

## 📝 代码分析文件

### 相关的代码分析文档

1. **`CODE_ANALYSIS_30_QUEUES_SOURCE.md`**
   - cp_queue_bitmap初始化
   - 为什么是30个队列

2. **`FINAL_ANSWER_30_QUEUES_MYSTERY.md`**
   - 30个队列的完整解释
   - MEC1和MEC2的区别

3. **`WHY_ONLY_30_NOT_120_CRITICAL_FINDING.md`**
   - 为什么只有30而不是120
   - 可能的代码BUG

4. **`QUEUE_MAPPING_MECHANISM_CODE_ANALYSIS.md`**
   - 队列映射机制
   - 超过限制时的行为

---

## 🔗 与其他分析的关联

### Queue数量分析
```
30队列分析 ──┐
XCC/XCD分析 ─┤
             ├─→ Map/Unmap机制 ⭐
DSV3.2测试 ──┤
KCQ分析 ─────┘
```

**联系**：
- 理解30个队列的来源 → 理解Map的限制
- 理解4个XCC → 理解1→4映射
- DSV3.2的80 MQD → 理解实际Map过程
- KCQ占用2个 → 理解30=32-2

### 建议阅读顺序
```
1. FINAL_ANSWER_30_QUEUES_MYSTERY.md
   └─ 理解30个队列来源

2. XCC_XCD_AND_QUEUE_COUNT_CLARIFICATION.md
   └─ 理解XCC架构

3. MAP_UNMAP_VISUAL_GUIDE.md ⭐
   └─ 理解Map/Unmap机制

4. DSV3.2_CQ_HQ_ANALYSIS_REPORT.md
   └─ 看实际测试数据

总计: 完整的队列管理知识体系 ✓
```

---

## 💡 重要提示

### 1. 先看图表
```
建议：先看 MAP_UNMAP_VISUAL_GUIDE.md
  - 图表直观
  - 快速建立认知框架
  - 再看详细文字会更容易理解
```

### 2. 中英文对照
```
英文文档:
  - 代码注释和变量名都是英文
  - 更接近原始代码

中文文档:
  - 概念解释更详细
  - 更容易理解

建议: 两者结合阅读
```

### 3. 理论+实践
```
理论: 看文档理解机制
实践: 用ftrace跟踪实际过程

结合:
  1. 先看文档建立理论框架
  2. 再用ftrace验证实际行为
  3. 对照代码深入理解
```

---

## 🛠️ 验证和测试

### 验证Map/Unmap过程

```bash
# 1. 开启ftrace
echo 1 > /sys/kernel/debug/tracing/events/amdgpu/amdgpu_cs_ioctl/enable
echo 1 > /sys/kernel/debug/tracing/events/kfd/enable

# 2. 运行测试程序
python3 test_queue_create.py

# 3. 查看trace
cat /sys/kernel/debug/tracing/trace

# 应该看到:
# - kfd_ioctl: CREATE_QUEUE
# - allocate_hqd: pipe=X, queue=Y
# - load_mqd: xcc_id=0,1,2,3
# - map_queues_cpsch
```

### 监控HQD分配

```bash
# 实时监控
watch -n 1 'cat /sys/kernel/debug/kfd/hqds | head -50'

# 查看pipe分布
cat /sys/kernel/debug/kfd/hqds | grep "^[[:space:]]*[0-9]" | awk '{print $2}' | sort | uniq -c
# 应该看到队列均匀分布在4个Pipe上 ✓
```

---

## 📖 建议的学习计划

### Day 1: 概念理解（2小时）
```
上午:
  ✓ 阅读 MAP_UNMAP_VISUAL_GUIDE.md
  ✓ 阅读 MAP_UNMAP_SUMMARY_CN.md

下午:
  ✓ 运行简单的队列创建测试
  ✓ 观察MQD和HQD的变化
  ✓ 用ftrace看实际的map过程
```

### Day 2: 深入机制（3小时）
```
上午:
  ✓ 阅读 SW_QUEUE_HW_QUEUE_MAPPING_MECHANISM.md
  ✓ 理解6个阶段

下午:
  ✓ 查看相关代码
  ✓ 对照文档理解每个函数的作用
  ✓ 尝试修改num_kcq，观察变化
```

### Day 3: 代码实现（4小时）
```
上午:
  ✓ 阅读 MAP_UNMAP_DETAILED_PROCESS.md
  ✓ 跟踪函数调用链

下午:
  ✓ 阅读实际源代码
  ✓ 使用ftrace验证理解
  ✓ 尝试添加trace_printk调试
  ✓ 观察packet在HIQ中的传递
```

---

## 🎯 总结

### 4个文档，4个层次

```
1. Visual Guide    ← 可视化（图表）
2. Summary CN      ← 中文详解（文字）
3. Mapping Mech    ← 完整机制（系统）
4. Detailed Proc   ← 代码实现（底层）

从上到下，逐层深入 ✓
```

### 最重要的收获

```
理解了Map/Unmap机制 =
  ✓ 理解AMD GPU的队列管理
  ✓ 理解HWS的作用
  ✓ 理解多XCC架构的复杂性
  ✓ 理解为什么MQD≠HQD
  ✓ 理解动态资源管理的优势
```

---

**创建时间**: 2026-02-03  
**文档总数**: 4个核心文档  
**总字数**: ~15,000字  
**代码引用**: 20+处  
**图表数量**: 10+个  
**完整性**: ⭐⭐⭐⭐⭐
