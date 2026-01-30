# Kernel 提交流程验证 - 快速开始

## 🚀 三步验证

### 第 1 步：编译测试程序

```bash
cd /mnt/md0/zhehan/code/coderampup/private_github/amdgpudriver/doc/kernel_submit/
hipcc -o test_kernel_trace test_kernel_trace.cpp
```

### 第 2 步：运行测试

```bash
# 基础测试（无需 root）
./test_kernel_trace

# 完整验证（需要 root）
sudo ./verify_kernel_flow.sh
```

### 第 3 步：查看结果

```bash
# 查看验证报告
cat trace_output/verification_report.txt

# 查看追踪文件
ls -lh trace_output/
```

---

## 📋 测试文件说明

| 文件 | 用途 |
|------|------|
| `test_kernel_trace.cpp` | HIP 测试程序（向量加法） |
| `verify_kernel_flow.sh` | 自动化验证脚本 |
| `VERIFICATION_GUIDE.md` | 详细验证指南 |
| `trace_output/` | 追踪输出目录（自动创建） |

---

## 🎯 验证内容

1. ✅ **GPU 和调度器信息**
   - GPU 型号和规格
   - MES/CPSCH 模式确认

2. ✅ **Kernel 提交流程**
   - HIP API 调用
   - HSA Runtime 交互
   - KFD 驱动处理
   - MES/CPSCH 调度

3. ✅ **文档对应关系**
   - KERNEL_TRACE_01: Application → HIP
   - KERNEL_TRACE_02: HSA Runtime
   - KERNEL_TRACE_03: KFD Driver
   - KERNEL_TRACE_04: MES/Hardware

---

## 🔍 快速验证命令

### 检查 GPU 信息
```bash
./test_kernel_trace | head -20
```

### 检查调度器模式
```bash
cat /sys/module/amdgpu/parameters/mes
# 输出: 0=CPSCH, 1=MES
```

### 使用 ROCprofiler 追踪
```bash
# rocprofv3 (推荐)
rocprofv3 --hip-api --hsa-api --kernel-trace \
    --output-file trace.csv ./test_kernel_trace

# 分析追踪
grep "hipLaunchKernel\|hsa_queue" trace.csv
```

### 使用 strace 追踪系统调用
```bash
strace -e openat,ioctl,mmap ./test_kernel_trace 2>&1 | grep kfd
```

---

## 📊 预期结果

### 成功运行输出

```
=== Kernel Submission Flow Test ===

[1] GPU Information:
  - Device Name: <你的 GPU 型号>
  - Compute Units: <CU 数量>

[2] Scheduler Mode:
  - MES enabled: <0 或 1>

[6] Launching Kernel:
  - Kernel execution time: <时间> us

[8] Verification:
  - ✅ All results correct!
```

### MES 模式 (MI300A/X, MI250X)
```
[2] Scheduler Mode:
  - MES enabled: 1
```

### CPSCH 模式 (MI308X, MI100)
```
[2] Scheduler Mode:
  - MES enabled: 0
```

---

## 🔧 故障排查

### hipcc 找不到
```bash
export PATH=/opt/rocm/bin:$PATH
```

### 权限问题
```bash
# 部分功能需要 root
sudo ./verify_kernel_flow.sh
```

### /dev/kfd 不存在
```bash
# 检查驱动
lsmod | grep amdgpu
sudo modprobe amdgpu
```

---

## 📚 详细文档

- **完整验证指南**: [VERIFICATION_GUIDE.md](./VERIFICATION_GUIDE.md)
- **Profiling 工具**: [ROCM_PROFILING_TOOLS_GUIDE.md](./ROCM_PROFILING_TOOLS_GUIDE.md)
- **文档索引**: [KERNEL_TRACE_INDEX.md](./KERNEL_TRACE_INDEX.md)

---

## 💡 提示

1. **第一次运行**：建议先运行基础测试 `./test_kernel_trace`
2. **需要详细追踪**：使用 `sudo ./verify_kernel_flow.sh`
3. **对比文档**：将追踪结果与文档流程图对比
4. **不同 GPU**：注意 MES/CPSCH 模式差异

---

**需要帮助？** 查看 [VERIFICATION_GUIDE.md](./VERIFICATION_GUIDE.md) 获取详细说明

