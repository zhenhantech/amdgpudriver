# AMD ROCm 驱动分析文档

本目录包含了AMD ROCm驱动各个组件的详细分析文档。

## 📂 目录结构

```
doc/
├── README.md (本文件)
│
├── 📁 memory/ (内存管理相关)
│   ├── ROCm内存池详解.md
│   └── ROCm内存管理分层详解.md
│
├── 📁 kernel/ (Kernel调度相关)
│   ├── AQL定义详解.md
│   └── ROCm_Kernel_Dispatch流程详解.md
│
├── AMD_ROCM_架构分析.md (整体架构分析)
├── AMD_ROCM_架构图示.md (架构图示)
├── AMD_ROCM_详细调用流程.md (详细调用流程)
├── CLR目录结构详解.md (CLR组件结构)
├── HIP与CLR架构关系详解.md (HIP和CLR关系)
├── HIP目录结构详解.md (HIP组件结构)
├── ROCR-Runtime目录结构详解.md (ROCR Runtime结构)
├── 解决aqlprofile依赖问题.md (编译问题)
└── ROCm编译问题诊断.md (编译诊断)
```

## 📚 文档分类

### 🧠 内存管理 (memory/)

深入分析ROCm的内存管理机制，包括：
- **ROCm内存池详解.md**: 内存池的实现和优化
- **ROCm内存管理分层详解.md**: 从HIP到KFD的完整内存管理调用链

### 🚀 Kernel调度 (kernel/)

详细讲解GPU kernel的提交和执行流程：
- **AQL定义详解.md**: HSA标准的AQL packet格式定义
- **ROCm_Kernel_Dispatch流程详解.md**: Kernel从提交到执行的完整流程

### 🏗️ 整体架构

- **AMD_ROCM_架构分析.md**: ROCm软件栈的整体架构
- **AMD_ROCM_架构图示.md**: 架构图示和可视化
- **AMD_ROCM_详细调用流程.md**: 关键API的详细调用流程

### 📦 组件结构

- **HIP目录结构详解.md**: HIP API定义层的目录结构
- **CLR目录结构详解.md**: CLR实现层（HIP Runtime + OpenCL）
- **ROCR-Runtime目录结构详解.md**: HSA Runtime的目录结构
- **HIP与CLR架构关系详解.md**: HIP接口与CLR实现的关系

### 🔧 编译和问题诊断

- **解决aqlprofile依赖问题.md**: aqlprofile组件的依赖问题
- **ROCm编译问题诊断.md**: 常见编译问题和解决方案

## 🎯 快速导航

### 想了解整体架构？
👉 从 `AMD_ROCM_架构分析.md` 开始

### 想了解Kernel是如何提交的？
👉 查看 `kernel/ROCm_Kernel_Dispatch流程详解.md`

### 想了解内存是如何管理的？
👉 查看 `memory/ROCm内存管理分层详解.md`

### 想了解组件之间的关系？
👉 查看 `HIP与CLR架构关系详解.md`

### 遇到编译问题？
👉 查看 `ROCm编译问题诊断.md`

## 📖 阅读顺序建议

### 新手入门：
1. AMD_ROCM_架构分析.md
2. AMD_ROCM_架构图示.md
3. HIP与CLR架构关系详解.md

### 深入理解：
1. kernel/ROCm_Kernel_Dispatch流程详解.md
2. memory/ROCm内存管理分层详解.md
3. AMD_ROCM_详细调用流程.md

### 源码阅读：
1. HIP目录结构详解.md
2. CLR目录结构详解.md
3. ROCR-Runtime目录结构详解.md

## 🔗 相关链接

- ROCm官方文档: https://rocm.docs.amd.com/
- HSA Foundation: http://www.hsafoundation.com/
- AMD GPU开发者资源: https://developer.amd.com/

---

**最后更新**: 2024年11月

