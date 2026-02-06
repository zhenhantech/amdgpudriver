# DeepSeek 3.2 + ftrace 测试指南

## 📋 测试目的

在复杂AI模型（DeepSeek 3.2，使用8个GPU）下，验证我们的POC设计是否有问题。

**关键验证点：**
1. DeepSeek 3.2 使用几个Hardware Queue？
2. Queue使用模式是否与Case-A/Case-B（单Queue模型）一致？
3. 多GPU环境下的Queue分配策略如何？
4. POC的Queue级别抢占设计是否适用？

---

## 🚀 快速开始

### 1. 运行测试（需要sudo）

```bash
sudo ./run_deepseek_with_ftrace.sh <container_name>
```

**示例：**
```bash
# 运行DeepSeek 3.2测试
sudo ./run_deepseek_with_ftrace.sh zhen_vllm_dsv3
```

**说明：**
- 测试使用容器内的 `/data/code/rampup_doc/vLLM_test/scripts/run_vLLM_v1_optimized.sh test`
- 测试时长由 `test_inference.py` 自动控制，无需手动指定
- 自动应用所有vLLM优化配置（V1引擎、FlashMLA、ROCm Aiter等）

### 2. 查看分析结果

测试完成后，脚本会自动运行初步分析。如需重新分析：

```bash
./log/deepseek_ftrace_<timestamp>/analyze_deepseek.sh
```

---

## 📊 输出文件说明

测试会在 `log/deepseek_ftrace_<timestamp>/` 目录下生成以下文件：

| 文件                      | 说明                                      |
|---------------------------|-------------------------------------------|
| `deepseek_amd_log.txt`    | AMD日志（AMD_LOG_LEVEL=3）                |
| `ftrace.txt`              | Kernel ftrace日志（KFD相关函数）          |
| `queue_info.txt`          | 从AMD日志提取的Queue使用统计              |
| `pid_mapping.txt`         | 容器PID和进程PID映射                      |
| `analyze_deepseek.sh`     | 详细分析脚本（可重复运行）                |

---

## 🔍 关键分析指标

### 1. Queue数量
- **预期**：如果是单Queue模型（1个Queue），则POC设计完全适用
- **如果多Queue**：需要评估是否需要批量suspend/resume

### 2. 与Case-A/Case-B对比

| 指标              | Case-A (CNN) | Case-B (Transformer) | DeepSeek 3.2 |
|-------------------|--------------|----------------------|--------------|
| Queue数量         | 1            | 1                    | ?            |
| GPU使用           | 1            | 1                    | 8            |
| Kernel提交次数    | 127,099      | 261,809              | ?            |
| 测试时长          | 121秒        | 121秒                | 120秒        |

### 3. POC设计适用性判断

**场景1：DeepSeek也用单Queue**
- ✅ POC设计完全适用
- ✅ 即使8个GPU，也只用1个Queue
- ✅ Queue级别抢占设计不需要修改

**场景2：DeepSeek用多Queue（≤8个）**
- ⚠️  需要批量suspend/resume
- ✅ 创新方案（batch_unmap）更适合
- ✅ 性能提升更明显

**场景3：DeepSeek用大量Queue（>8个）**
- ❌ 需要重新评估POC设计
- ❌ 需要选择性抢占策略
- ❌ 可能需要优先级Queue识别

---

## 🛠️ 脚本工作流程

```
┌─────────────────────────────────────────────────────────────┐
│ 步骤1: 配置ftrace                                            │
│   - 清空之前的trace                                          │
│   - 设置buffer size (20MB，因为8 GPU)                       │
│   - 启用KFD + AMDGPU trace                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤2: 获取环境信息                                          │
│   - 获取容器PID                                              │
│   - 检查GPU可见性                                            │
│   - 记录PID映射                                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤3: 启动测试                                              │
│   - 启动ftrace                                               │
│   - 在容器内运行DeepSeek（AMD_LOG_LEVEL=3）                 │
│   - 同时捕获AMD日志和ftrace                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤4: 保存日志                                              │
│   - 停止ftrace                                               │
│   - 保存ftrace.txt                                           │
│   - 保存deepseek_amd_log.txt                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤5: 提取Queue信息                                         │
│   - 从AMD日志提取所有Queue地址                               │
│   - 统计Queue ID分布                                         │
│   - 统计Kernel提交次数                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤6: 快速分析                                              │
│   - 显示关键统计信息                                         │
│   - 生成分析脚本                                             │
│   - 运行初步分析                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 DeepSeek启动方式说明

脚本内置了3种DeepSeek启动方式，会自动检测：

### 方案1: Python测试脚本（推荐）
```bash
# 在容器的/workspace目录下创建 test_deepseek.py
python3 test_deepseek.py --duration 120
```

### 方案2: vLLM服务
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/deepseek-coder-33b-instruct \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9
```

### 方案3: 自定义启动命令
如需自定义，请修改脚本第118-157行。

---

## 🔧 修改AMD日志级别

当前使用 `AMD_LOG_LEVEL=3` 来减小日志量。如需更详细的日志，可以修改：

```bash
# 编辑脚本
vim run_deepseek_with_ftrace.sh

# 找到第126行
export AMD_LOG_LEVEL=3

# 修改为更高级别
export AMD_LOG_LEVEL=4  # 更详细
export AMD_LOG_LEVEL=5  # 最详细（日志会非常大）
```

**日志级别说明：**
- **Level 3**: Queue使用、Kernel提交（适合本次测试）
- **Level 4**: 增加Memory操作
- **Level 5**: 所有KFD交互（日志量大，不推荐）

---

## 📊 分析脚本功能

`analyze_deepseek.sh` 会分析以下内容：

### 1. Queue使用模式分析
- 列出所有唯一的Queue地址
- 统计总Queue数量
- 判断是单Queue还是多Queue模型

### 2. Kernel提交模式
- 统计Kernel提交总次数
- 显示前10个Kernel提交记录

### 3. GPU使用推测
- 从Queue地址推测GPU分布

### 4. ftrace关键事件
- Queue创建/销毁事件
- Map/Unmap事件

### 5. 与Case-A/Case-B对比
- 对比表格（Queue数量、Kernel提交次数）

### 6. POC设计验证结论
- 根据Queue数量判断POC设计适用性
- 给出具体建议

---

## ⚠️ 注意事项

### 1. 必须使用sudo
ftrace需要root权限访问 `/sys/kernel/debug/tracing/`

### 2. 确保容器正在运行
```bash
docker ps | grep zhen_vllm_dsv3
```

### 3. 确保8个GPU可见
```bash
docker exec zhen_vllm_dsv3 rocm-smi --showid
```

### 4. 磁盘空间
- AMD日志（Level 3）：~10-50MB（120秒）
- ftrace日志：~20-100MB（取决于活动）
- 总计：~30-150MB per run

### 5. 测试时长建议
- **快速测试**: 60秒（基本Queue信息）
- **标准测试**: 120秒（推荐，本测试默认值）
- **长时间测试**: 300秒+（观察Queue长时间行为）

---

## 🎯 预期结果

### 最佳情况：单Queue模型
```
✅ 发现：DeepSeek也使用单Queue模型！
   → 与Case-A/Case-B一致
   → POC设计适用 ✓
   
POC设计验证结果: 完全适用
  1. DeepSeek也使用单Queue模型
  2. 即使8个GPU，也只用1个Queue
  3. Queue级别抢占设计完全适用
  4. 不需要修改POC设计
```

### 次优情况：多Queue（≤8）
```
ℹ️  发现：DeepSeek使用 8 个Queue
   → 可能每个GPU一个Queue
   → POC设计需要支持多Queue
   
POC设计验证结果: 需要小幅调整
  1. DeepSeek使用多个Queue（8个）
  2. 需要批量suspend/resume
  3. 建议使用创新方案（batch_unmap）
  4. 性能提升更明显（批量操作）
```

### 需要重新评估：大量Queue
```
⚠️  发现：DeepSeek使用 16+ 个Queue
   → 超过GPU数量
   → 需要重新评估POC设计
   
POC设计验证结果: 需要重新评估
  1. Queue数量超预期
  2. 需要分析Queue用途
  3. 可能需要选择性抢占策略
```

---

## 📚 相关文档

- [POC Stage 1 实施方案](../ARCH_Design_01_POC_Stage1_实施方案.md)
- [创新方案：Map/Unmap抢占](../New_DESIGN_MAP_UNMAP_BASED_PREEMPTION.md)
- [Case-A/Case-B分析报告](./log/case_comparison_20260205_155247/ANALYSIS_SUMMARY.md)
- [下一步计划](../NEXT_STEPS_PREEMPTION_POC.md)

---

## 🤝 问题排查

### 问题1: 无法启动ftrace
```bash
❌ ftrace不可用! 请确保debugfs已挂载
```
**解决：**
```bash
sudo mount -t debugfs none /sys/kernel/debug
```

### 问题2: 容器PID获取失败
```bash
❌ 无法获取容器PID，容器可能未运行
```
**解决：**
```bash
# 检查容器状态
docker ps

# 启动容器
docker start zhen_vllm_dsv3
```

### 问题3: AMD日志为空或很少
**可能原因：**
- AMD_LOG_LEVEL未生效
- DeepSeek未实际运行GPU任务

**解决：**
- 检查容器内环境变量
- 手动运行DeepSeek推理测试

### 问题4: ftrace日志过大
**解决：**
- 减小buffer size（修改脚本第66行）
- 缩短测试时长

---

## ✅ 测试完成后

1. **查看结果**
   ```bash
   ./log/deepseek_ftrace_<timestamp>/analyze_deepseek.sh
   ```

2. **对比分析**
   - 与Case-A/Case-B的结果对比
   - 确认Queue使用模式

3. **更新POC设计**（如需要）
   - 单Queue → 继续当前设计
   - 多Queue → 考虑batch操作
   - 大量Queue → 重新评估

4. **记录发现**
   - 更新 `PROGRESS_UPDATE_<date>.md`
   - 记录关键发现和结论

---

**生成时间**: 2026-02-05  
**版本**: 1.0

