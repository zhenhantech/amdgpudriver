# DeepSeek 3.2测试设置完成 ✅

## 📋 已完成的工作

### 1. 复制了容器内的DeepSeek测试脚本
```bash
✅ run_vLLM_v1_optimized.sh  - DeepSeek启动脚本（已复制到本地）
✅ test_inference.py         - 推理测试脚本（已复制到本地）
```

**脚本位置：**
- 本地: `/mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code/`
- 容器: `/data/code/rampup_doc/vLLM_test/scripts/`

### 2. 更新了ftrace捕获脚本
```bash
✅ run_deepseek_with_ftrace.sh  - 集成了DeepSeek测试 + ftrace同步捕获
```

**功能：**
- 自动配置ftrace（追踪KFD函数）
- 在容器内运行 `run_vLLM_v1_optimized.sh test`
- 同时捕获AMD日志（Level 3）和ftrace日志
- 自动分析Queue使用模式
- 生成对比报告

### 3. 更新了文档
```bash
✅ DEEPSEEK_TEST_GUIDE.md       - 完整测试指南
✅ DEEPSEEK_QUICK_FIX.md        - 问题修复指南
✅ TEST_TOOLS_OVERVIEW.md       - 测试工具总览
✅ DEEPSEEK_SETUP_COMPLETE.md   - 本文档
```

---

## 🚀 快速开始（一键运行）

```bash
# 进入测试目录
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code

# 运行测试（需要sudo，因为ftrace需要root权限）
sudo ./run_deepseek_with_ftrace.sh zhen_vllm_dsv3
```

**就这么简单！** 🎉

---

## 📊 测试会做什么

```
┌─────────────────────────────────────────────────────────┐
│ 步骤1: 配置ftrace (追踪KFD函数)                         │
│ 步骤2: 启动DeepSeek 3.2 (8 GPU)                        │
│ 步骤3: 同步捕获 AMD日志 + ftrace                        │
│ 步骤4: 提取Queue信息                                    │
│ 步骤5: 自动分析并生成报告                               │
└─────────────────────────────────────────────────────────┘
```

### 测试使用的配置
- **模型**: `/mnt/md0/models/Deepseekv3.2-ptpc`
- **GPU数量**: 8个 (gfx942)
- **Tensor Parallel**: 8
- **引擎**: vLLM V1（新架构）
- **Attention**: FlashMLA
- **优化**: ROCm Aiter全套（MOE、MHA、Fusion等）
- **AMD日志级别**: 3（减小日志量，适合分析）

---

## 🎯 核心验证目标

```
┌──────────────────────────────────────────────────────────┐
│  DeepSeek 3.2 (8 GPU) 使用几个Hardware Queue？          │
│                                                          │
│  场景1: 1个Queue  → ✅ POC设计完全适用                   │
│  场景2: 8个Queue  → ⚠️  需要批量操作                     │
│  场景3: 更多Queue → ❌ 需要重新评估                      │
└──────────────────────────────────────────────────────────┘
```

**对比基准**:
- **Case-A (CNN)**: 1个Queue, 1个GPU ✅ 已测试
- **Case-B (Transformer)**: 1个Queue, 1个GPU ✅ 已测试  
- **DeepSeek 3.2**: ??? 个Queue, 8个GPU ← **今天要验证**

---

## 📁 测试输出

测试完成后，会在 `log/deepseek_ftrace_<timestamp>/` 生成：

| 文件                       | 说明                          |
|----------------------------|-------------------------------|
| `deepseek_amd_log.txt`     | AMD日志（Level 3）            |
| `ftrace.txt`               | Kernel ftrace日志             |
| `queue_info.txt`           | Queue使用统计                 |
| `pid_mapping.txt`          | 进程PID映射                   |
| `analyze_deepseek.sh`      | 详细分析脚本（可重复运行）    |

**查看分析结果：**
```bash
./log/deepseek_ftrace_<timestamp>/analyze_deepseek.sh
```

---

## 📊 预期输出示例

### 最理想情况（单Queue模型）
```
━━━ Queue使用模式分析 ━━━
唯一Queue数量: 1

✅ 发现：DeepSeek也使用单Queue模型！
   → 与Case-A/Case-B一致
   → POC设计适用 ✓

━━━ 与Case-A/Case-B对比 ━━━
| 指标          | Case-A | Case-B | DeepSeek |
|---------------|--------|--------|----------|
| Queue数量     | 1      | 1      | 1        |
| GPU数量       | 1      | 1      | 8        |
| Kernel提交    | 127K   | 262K   | ???K     |

━━━ POC设计验证 ━━━
✅ POC设计验证结果: 完全适用

理由：
  1. DeepSeek也使用单Queue模型
  2. 即使8个GPU，也只用1个Queue
  3. Queue级别抢占设计完全适用
  4. 不需要修改POC设计
```

---

## ⏱️ 预计测试时间

- **模型加载**: ~30-60秒
- **推理测试**: 由 `test_inference.py` 控制（通常2-5分钟）
- **日志分析**: ~5秒
- **总计**: ~5-10分钟

---

## ⚠️ 注意事项

### 1. 必须使用sudo
```bash
# ✅ 正确
sudo ./run_deepseek_with_ftrace.sh zhen_vllm_dsv3

# ❌ 错误（ftrace需要root权限）
./run_deepseek_with_ftrace.sh zhen_vllm_dsv3
```

### 2. 确保容器运行中
```bash
# 检查容器状态
docker ps | grep zhen_vllm_dsv3

# 如果未运行，启动容器
docker start zhen_vllm_dsv3
```

### 3. 确保GPU可用
```bash
# 在容器内检查
docker exec zhen_vllm_dsv3 rocm-smi --showid

# 应该看到8个GPU
```

### 4. 磁盘空间
- AMD日志（Level 3）：~50-200MB
- ftrace日志：~50-200MB
- **总计**：~100-400MB per run

---

## 🔗 相关文档

### 测试相关
- [完整测试指南](DEEPSEEK_TEST_GUIDE.md) - 详细说明和故障排查
- [快速修复指南](DEEPSEEK_QUICK_FIX.md) - 常见问题解决方案
- [测试工具总览](TEST_TOOLS_OVERVIEW.md) - 所有测试工具

### POC设计
- [POC Stage 1 实施方案](../ARCH_Design_01_POC_Stage1_实施方案.md)
- [创新方案：Map/Unmap抢占](../New_DESIGN_MAP_UNMAP_BASED_PREEMPTION.md)
- [下一步计划](../NEXT_STEPS_PREEMPTION_POC.md)

### 之前的分析
- [Case-A/Case-B分析](log/case_comparison_20260205_155247/ANALYSIS_SUMMARY.md)
- [GEMM + ftrace分析](log/gemm_ftrace_20260205_143555/ANALYSIS_REPORT.md)

---

## 📞 如果遇到问题

### Q1: vLLM启动失败
**A**: 查看 [DEEPSEEK_QUICK_FIX.md](DEEPSEEK_QUICK_FIX.md)

### Q2: ftrace日志为空
**A**: 检查是否使用了sudo，以及debugfs是否挂载
```bash
sudo mount -t debugfs none /sys/kernel/debug
```

### Q3: AMD日志很少
**A**: 检查 AMD_LOG_LEVEL 是否生效
```bash
docker exec zhen_vllm_dsv3 bash -c "
    export AMD_LOG_LEVEL=3
    python3 -c 'import torch; print(torch.cuda.is_available())'
"
```

---

## ✅ 准备就绪！

一切都已配置好，现在可以运行测试了：

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/scheduler/DOC_POC_stage1/code
sudo ./run_deepseek_with_ftrace.sh zhen_vllm_dsv3
```

**祝测试顺利！** 🚀

---

**创建时间**: 2026-02-05  
**版本**: 1.0  
**状态**: ✅ Ready to Test

