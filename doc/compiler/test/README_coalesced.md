# Coalesced Memory Access 测试说明

本测试演示了 AMD GPU 上合并内存访问的重要性。

## 快速运行

```bash
cd /mnt/md0/zhehan/code/coderampup/compiler/test

# 编译
hipcc -O3 --offload-arch=gfx90a test_coalesced_memory.hip -o test_coalesced

# 运行
./test_coalesced
```

## 测试内容

### 测试1: 矩阵转置
- **Naive版本**: 非合并写入（stride访问）
- **Optimized版本**: 使用共享内存实现合并访问
- **预期提升**: 5-20x（AMD GPU上更明显）

### 测试2: 粒子系统
- **AoS版本**: Array of Structures（非连续访问）
- **SoA版本**: Structure of Arrays（完美合并）
- **预期提升**: 2-5x

## 典型输出（MI200）

```
========================================
Coalesced Memory Access 测试
========================================

测试1: 矩阵转置 (4096x4096)
----------------------------------------
Naive (非合并):     2.45 ms, 137 GB/s
Optimized (合并):   0.52 ms, 646 GB/s
加速比:             4.7x
带宽提升:           4.7x

测试2: 粒子更新 (AoS vs SoA)
----------------------------------------
AoS (非合并):       0.85 ms
SoA (合并):         0.28 ms
加速比:             3.0x

========================================
测试完成！
========================================
```

## 使用 rocprof 分析

```bash
# 分析内存带宽
rocprof --stats ./test_coalesced

# 查看详细指标
cat results.stats.csv

# 关键指标：
# - MemUnitBusy: 应该在优化版本中更高
# - L2CacheHit: 应该在优化版本中更高
# - FetchSize/WriteSize: 优化版本应该更大（说明合并了）
```

## 相关文档

- [HIPIFY工具学习计划.md - 附录D](../doc/compiler/HIPIFY工具学习计划.md)
- [ROCmLLVM_AMD特殊优化深度分析.md](../doc/compiler/ROCmLLVM_AMD特殊优化深度分析.md)

## 学习要点

1. **Wavefront=64**: AMD GPU需要64个线程访问连续地址才能完全合并
2. **共享内存缓冲**: 用LDS避免非规则访问
3. **SoA优于AoS**: 数据结构布局对性能影响巨大
4. **向量化**: 使用float4等类型提高带宽

