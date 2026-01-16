# ROCm LLVM Pass 测试工具集

本目录包含用于测试和验证 ROCm LLVM 优化Pass的工具和示例。

## 快速开始

### 1. 运行综合测试套件

```bash
# 基础用法（使用默认GPU架构 gfx90a）
./rocm_pass_test_suite.sh

# 指定GPU架构
./rocm_pass_test_suite.sh gfx908   # MI100
./rocm_pass_test_suite.sh gfx90a   # MI200
./rocm_pass_test_suite.sh gfx1100  # RDNA3
```

### 2. 测试单个Pass

#### PromoteAlloca优化

```bash
# 编译并查看优化统计
hipcc -O3 --offload-arch=gfx90a \
      -mllvm -stats \
      test_promote_alloca.hip -o test_promote

# 查看详细日志
hipcc -O3 --offload-arch=gfx90a \
      -mllvm -debug-only=amdgpu-promote-alloca \
      test_promote_alloca.hip 2>&1 | less

# 运行测试
./test_promote
```

#### AtomicOptimizer优化

```bash
# DPP策略（默认，更快）
hipcc -O3 --offload-arch=gfx90a \
      -mllvm -amdgpu-atomic-optimizer-strategy=DPP \
      -mllvm -stats \
      test_atomic_optimizer.hip -o test_atomic_dpp

# Iterative策略
hipcc -O3 --offload-arch=gfx90a \
      -mllvm -amdgpu-atomic-optimizer-strategy=Iterative \
      -mllvm -stats \
      test_atomic_optimizer.hip -o test_atomic_iter

# 运行测试
./test_atomic_dpp
./test_atomic_iter
```

#### LoadStoreOptimizer优化

```bash
# 编译并查看优化统计
hipcc -O3 --offload-arch=gfx90a \
      -save-temps \
      -mllvm -stats \
      test_loadstore_optimizer.hip -o test_loadstore 2>&1 | \
      grep "load-store"

# 查看生成的汇编（检查向量化load）
cat test_loadstore_optimizer-hip-amdgcn-*.s | grep -i "load"

# 运行测试
./test_loadstore
```

#### Coalesced Memory Access 测试

```bash
# 编译并运行合并内存访问测试
hipcc -O3 --offload-arch=gfx90a \
      test_coalesced_memory.hip -o test_coalesced

# 运行（会测试矩阵转置和AoS vs SoA）
./test_coalesced

# 使用rocprof分析
rocprof --stats ./test_coalesced
cat results.stats.csv
```

#### comgr 基础功能测试

```bash
# 编译 comgr 测试程序（演示 JIT 编译）
gcc test_comgr_basic.c -o test_comgr \
    -I/opt/rocm/include \
    -L/opt/rocm/lib \
    -lamd_comgr

# 运行（会测试版本查询、ISA 查询、内核编译）
./test_comgr

# 查看详细说明
cat README_comgr.md
```

### 3. 对比优化级别

```bash
# 对比 -O0 到 -O3 的效果
./compare_opt_levels.sh test_promote_alloca.hip gfx90a
```

## 文件说明

| 文件 | 说明 |
|-----|------|
| `rocm_pass_test_suite.sh` | 综合测试套件（测试多个Pass） |
| `test_promote_alloca.hip` | PromoteAlloca优化测试代码 |
| `test_atomic_optimizer.hip` | AtomicOptimizer优化测试代码 |
| `test_loadstore_optimizer.hip` | LoadStoreOptimizer优化测试代码 |
| `test_coalesced_memory.hip` | 合并内存访问测试（矩阵转置、AoS vs SoA） |
| `test_comgr_basic.c` | comgr API 基础功能测试（JIT 编译示例） |
| `compare_opt_levels.sh` | 对比不同优化级别的效果 |
| `README_coalesced.md` | Coalesced Memory Access详细说明 |
| `README_comgr.md` | comgr 测试说明 |

## 常用命令参考

### 查看Pass统计

```bash
hipcc -O3 --offload-arch=gfx90a -mllvm -stats kernel.hip
```

### 查看Pass执行时间

```bash
hipcc -O3 --offload-arch=gfx90a -mllvm -time-passes kernel.hip 2>&1 | grep AMDGPU
```

### 保存中间文件

```bash
hipcc -O3 --offload-arch=gfx90a -save-temps kernel.hip
# 生成: .bc (LLVM IR), .ll (IR文本), .s (汇编)
```

### 查看优化后的LLVM IR

```bash
hipcc -O3 --offload-arch=gfx90a -save-temps kernel.hip
llvm-dis kernel-hip-*.bc -o optimized.ll
less optimized.ll
```

### 查看特定Pass的详细日志

```bash
# 可用的debug标签：
# - amdgpu-promote-alloca
# - amdgpu-load-store-opt
# - amdgpu-atomic-optimizer
# - si-whole-quad-mode
# - gcn-sched

hipcc -O3 --offload-arch=gfx90a \
      -mllvm -debug-only=amdgpu-promote-alloca \
      kernel.hip 2>&1 | less
```

### 禁用特定Pass（调试用）

```bash
# 禁用LoadStoreOptimizer
hipcc -O3 --offload-arch=gfx90a \
      -mllvm -amdgpu-enable-load-store-opt=false \
      kernel.hip

# 禁用Alloca向量化
hipcc -O3 --offload-arch=gfx90a \
      -mllvm -disable-promote-alloca-to-vector \
      kernel.hip
```

## GPU架构对照表

| GPU型号 | 架构代号 | --offload-arch |
|--------|---------|----------------|
| MI200系列 | CDNA2 | gfx90a |
| MI100 | CDNA1 | gfx908 |
| MI50/MI60 | GCN5 | gfx906 |
| RX 6800/6900 | RDNA2 | gfx1030 |
| RX 7900 | RDNA3 | gfx1100 |

## 运行时性能分析

```bash
# 编译kernel
hipcc -O3 --offload-arch=gfx90a kernel.hip -o test

# 基础profiling
rocprof --stats ./test

# 详细trace
rocprof --hip-trace --hsa-trace ./test

# 查看结果
cat results.stats.csv
```

## 故障排查

### 看不到优化统计

```bash
# 确保使用了 -mllvm -stats
export LLVM_ENABLE_STATS=1
hipcc -O3 --offload-arch=gfx90a -mllvm -stats kernel.hip
```

### GPU架构不匹配

```bash
# 查看系统GPU
rocminfo | grep "Name:"

# 使用正确的架构代号
hipcc --offload-arch=gfxXXX kernel.hip
```

### 权限问题

```bash
# 给脚本添加执行权限
chmod +x *.sh
```

## 参考文档

- [ROCm LLVM AMD特殊优化深度分析](../doc/compiler/ROCmLLVM_AMD特殊优化深度分析.md)
- [优化Pass实战指南](../doc/compiler/ROCmLLVM_优化Pass实战指南.md)
- [Linux测试快速上手](../doc/compiler/ROCmLLVM_Linux测试快速上手.md)

## 贡献

欢迎添加更多测试用例和优化示例！

