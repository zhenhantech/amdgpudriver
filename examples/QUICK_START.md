# 🚀 HIPIFY 真实案例 - 5分钟快速上手

## 📍 你在这里
```
/mnt/md0/zhehan/code/coderampup/examples/
├── hipify_reduction_demo.hip       ← 完整示例代码
├── run_reduction_demo.sh           ← 一键运行脚本 ⭐
├── README_reduction_demo.md        ← 详细文档
└── QUICK_START.md                  ← 你正在看的文档
```

## ⚡ 最快上手方式（3个命令）

```bash
cd /mnt/md0/zhehan/code/coderampup/examples

# 方式1: 一键运行（推荐）
./run_reduction_demo.sh

# 方式2: 手动编译运行
hipcc -O3 hipify_reduction_demo.hip -o demo && ./demo

# 方式3: 查看寄存器使用（学习重点）
hipcc -O3 --resource-usage hipify_reduction_demo.hip -o demo 2>&1 | grep "uses"
```

## 🎯 你将看到什么

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 HIPIFY 归约优化示例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔹 版本1: 未优化 (基础循环)
   执行时间: 1.234 ms      ← 基准性能
   带宽: 162.5 GB/s

🔹 版本2: 4倍循环展开
   执行时间: 0.895 ms      ← 27% 提升 ✅
   带宽: 224.3 GB/s

🔹 版本3: 16倍循环展开
   执行时间: 0.756 ms      ← 39% 提升 ✅✅
   带宽: 265.8 GB/s

📈 性能对比:
   版本2 vs 版本1: 1.38x 加速 (27.5% 提升)
   版本3 vs 版本1: 1.63x 加速 (38.7% 提升)
```

## 💡 5个核心学习点

### 1️⃣ 循环展开原理
```cpp
// ❌ 慢: 每次处理1个元素
for (int i = idx; i < N; i += stride) {
    sum += in[i];
}

// ✅ 快: 每次处理4个元素 (减少75%的循环开销)
for (int i = idx*4; i < N; i += stride*4) {
    sum += in[i] + in[i+1] + in[i+2] + in[i+3];
}
```

### 2️⃣ 寄存器使用
```bash
# 查看寄存器使用
hipcc --resource-usage xxx.hip 2>&1 | grep VGPR

# 输出:
# 版本1: 24 VGPRs  ← 低寄存器压力
# 版本2: 28 VGPRs  ← 中等
# 版本3: 36 VGPRs  ← 较高，但仍可接受
```

### 3️⃣ 性能权衡
| 展开倍数 | 寄存器 | 性能 | 建议 |
|---------|--------|------|------|
| 1x | ✅✅✅ | ⭐ | 寄存器紧张时 |
| 4x | ✅✅ | ⭐⭐⭐ | **默认推荐** ⭐ |
| 16x | ✅ | ⭐⭐⭐⭐ | 简单kernel |
| 32x+ | ❌ | ⭐⭐ | 可能寄存器溢出 |

### 4️⃣ 为什么HIPIFY不自动优化？
```
HIPIFY的定位:
  ┌──────────────────────┐
  │  CUDA代码  →  HIP代码 │  ← 只做"翻译"
  └──────────────────────┘
           ↓
  ❌ 不改变算法逻辑
  ❌ 不调整优化策略
  ❌ 不修改参数配置
  
你的工作:
  ┌──────────────────────┐
  │ HIP代码 → 优化的HIP代码│  ← 性能调优
  └──────────────────────┘
           ↓
  ✅ 针对AMD GPU架构优化
  ✅ 调整寄存器使用
  ✅ 利用Wavefront=64
```

### 5️⃣ 关键架构差异
```
NVIDIA GPU:
  Warp Size: 32 threads
  寄存器: 统一池

AMD GPU:
  Wavefront Size: 64 threads  ← 2倍！
  寄存器: VGPR + SGPR分离     ← 新概念

影响: 循环展开策略需要重新设计！
```

## 🧪 3个快速实验

### 实验1: 对比性能（2分钟）
```bash
./run_reduction_demo.sh
# 观察三个版本的性能差异
```

### 实验2: 查看寄存器（5分钟）
```bash
hipcc -O3 --resource-usage hipify_reduction_demo.hip -o demo 2>&1 | grep "uses"
# 记录每个kernel的VGPR使用量
```

### 实验3: 修改展开倍数（10分钟）
```cpp
// 编辑 hipify_reduction_demo.hip
// 尝试改成 8倍展开:
for(int i = idx*8; i < ARRAYSIZE-7; i += blockDim.x*gridDim.x*8) {
    sum += in[i] + in[i+1] + ... + in[i+7];
}
```

## 📚 下一步

### 如果你想深入学习
👉 阅读 `README_reduction_demo.md` - 完整教程

### 如果你想看更多案例
👉 查看 HIPIFY 学习计划文档:
```bash
/mnt/md0/zhehan/code/coderampup/doc/compiler/HIPIFY工具学习计划.md
```
- 第E.1节：寄存器优化详解（2399-2614行）
- 第E.2节：占用率调优详解（2616-2728行）

### 如果你想运行官方示例
```bash
# 克隆HIPIFY源码
git clone https://github.com/ROCm-Developer-Tools/HIPIFY.git
cd HIPIFY/tests/unit_tests/samples

# 编译运行
hipcc reduction.cu -o reduction
./reduction 52428800
```

### 如果你想系统学习
👉 按照学习计划的6个阶段逐步推进：
```
阶段0: 环境准备 (1天)
阶段1: hipify-perl (2-3天)  
阶段2: hipify-clang (4-5天)  ← 你现在这里
阶段3: 库映射 (3-4天)
阶段4: 手动优化 (5-7天)      ← 本示例属于这里
阶段5: 实战项目 (7-10天)
阶段6: 高级主题 (5-7天)
```

## 🆘 遇到问题？

### 编译错误
```bash
# 检查HIP环境
which hipcc
hipcc --version

# 如果未找到，设置环境变量
export PATH=/opt/rocm/bin:$PATH
```

### 运行错误
```bash
# 检查GPU
rocminfo | grep "Name:"

# 设置可见GPU
export HIP_VISIBLE_DEVICES=0
```

### 性能异常
```bash
# 确保使用优化编译
hipcc -O3 xxx.hip  # ← 必须有 -O3

# 查看GPU负载
rocm-smi
```

## 📊 性能目标

完成本示例后，你应该看到：
- ✅ 4倍展开：**20-30% 性能提升**
- ✅ 16倍展开：**30-40% 性能提升**
- ✅ 理解寄存器优化的原理和方法

## 🎓 学习成果检查

- [ ] 能够编译和运行示例
- [ ] 理解循环展开的优化原理
- [ ] 会使用 `--resource-usage` 查看寄存器
- [ ] 理解寄存器和性能的权衡关系
- [ ] 能够修改代码测试不同配置
- [ ] 知道为什么HIPIFY不做自动优化
- [ ] 理解AMD和NVIDIA GPU的架构差异

---

**开始你的性能优化之旅吧！🚀**

有问题随时查看完整文档或者提issue！

