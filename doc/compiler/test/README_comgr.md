# comgr 测试说明

## 快速运行

```bash
cd /mnt/md0/zhehan/code/coderampup/compiler/test

# 编译测试程序
gcc test_comgr_basic.c -o test_comgr \
    -I/opt/rocm/include \
    -L/opt/rocm/lib \
    -lamd_comgr

# 运行
./test_comgr
```

## 测试内容

这个测试程序演示了 comgr 的基础功能：

1. **版本查询**: 获取 comgr 库的版本号
2. **ISA 查询**: 列出所有支持的 GPU 架构
3. **编译测试**: 将 HIP 内核编译为 LLVM Bitcode

## 预期输出

```
========================================
comgr 基础功能测试
========================================

1. comgr 版本: 2.7

2. 支持的 ISA 数量: 30
   支持的架构列表:
   [0] amdgcn-amd-amdhsa--gfx900
   [1] amdgcn-amd-amdhsa--gfx902
   [2] amdgcn-amd-amdhsa--gfx906
   [3] amdgcn-amd-amdhsa--gfx908
   [4] amdgcn-amd-amdhsa--gfx90a
   [5] amdgcn-amd-amdhsa--gfx940
   [6] amdgcn-amd-amdhsa--gfx941
   [7] amdgcn-amd-amdhsa--gfx942
   ...

3. 编译测试内核:
   源码长度: 203 字节
   目标架构: amdgcn-amd-amdhsa--gfx900
   编译中...
   ✅ 编译成功！
   生成的 Bitcode 对象数量: 1
   Bitcode 大小: 4096 字节

========================================
测试完成！
========================================
```

## 故障排查

### 错误: 找不到 libcomgr.so

```bash
# 检查 ROCm 安装
ls /opt/rocm/lib/libamd_comgr.so

# 如果找不到，需要安装 ROCm
# Ubuntu/Debian:
sudo apt install rocm-comgr

# 或设置 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

### 错误: 编译失败

可能原因：
1. 目标架构不支持（尝试 gfx906 或 gfx908）
2. ROCm 版本过旧
3. device-libs 未正确安装

## 相关文档

- [ROCm_comgr深度解析.md](../../doc/compiler/ROCm_comgr深度解析.md) - comgr 详细文档
- [ROCm_LLVM编译器架构分析与学习计划.md](../../doc/compiler/ROCm_LLVM编译器架构分析与学习计划.md) - 整体架构

