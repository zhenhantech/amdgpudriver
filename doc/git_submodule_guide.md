# Git Submodule 使用指南

本文档介绍如何在项目中使用 Git Submodule 来管理外部依赖和第三方代码库。

## 目录
- [什么是 Git Submodule](#什么是-git-submodule)
- [添加 Submodule](#添加-submodule)
- [克隆包含 Submodule 的项目](#克隆包含-submodule-的项目)
- [更新 Submodule](#更新-submodule)
- [查看 Submodule 状态](#查看-submodule-状态)
- [删除 Submodule](#删除-submodule)
- [常见问题和最佳实践](#常见问题和最佳实践)

---

## 什么是 Git Submodule

Git Submodule 允许你将一个 Git 仓库作为另一个 Git 仓库的子目录。它能让你将外部项目集成到你的主项目中，同时保持它们的历史记录独立。

### 优点
- 保持外部依赖的独立版本控制
- 可以固定在特定的提交版本
- 便于管理多个相关项目

### 适用场景
- 引入第三方库或示例代码（如 ROCm examples）
- 管理共享的代码模块
- 追踪外部项目的特定版本

---

## 添加 Submodule

### 基本语法
```bash
git submodule add <repository-url> [path]
```

### 示例

#### 1. 添加 ROCm Systems 超级仓库（推荐）
```bash
cd /path/to/your/project
git submodule add https://github.com/ROCm/rocm-systems.git ROCm_keyDriver/rocm-systems
```

> **注意**：[rocm-systems](https://github.com/ROCm/rocm-systems) 是 ROCm 的超级仓库，整合了多个 ROCm 系统项目（包括 HIP、CLR、ROCr Runtime、rocProfiler 等）。推荐使用这个仓库来追踪 ROCm 生态系统。

#### 2. 添加 ROCm Examples（单独仓库）
```bash
git submodule add https://github.com/ROCm/rocm-examples.git ROCm_keyDriver/rocm-examples
```

#### 3. 添加 HIPIFY 工具
```bash
git submodule add https://github.com/ROCm/HIPIFY.git ROCm_keyDriver/HIPIFY
```

#### 4. 添加其他 ROCm 组件
```bash
# rocBLAS 库
git submodule add https://github.com/ROCm/rocBLAS.git Libraries/rocBLAS

# HIP 单独仓库（已包含在 rocm-systems 中）
git submodule add https://github.com/ROCm/HIP.git HIP
```

### 添加后的操作

添加 submodule 后，Git 会创建两个内容：
1. `.gitmodules` 文件：记录 submodule 的配置信息
2. Submodule 目录：包含子模块的代码

**必须提交这些变更：**
```bash
git add .gitmodules <submodule-path>
git commit -m "Add <submodule-name> as submodule"
git push
```

**示例：**
```bash
git add .gitmodules ROCm_keyDriver/rocm-systems
git commit -m "Add rocm-systems submodule"
git push
```

---

## 克隆包含 Submodule 的项目

当克隆一个包含 submodule 的项目时，需要额外的步骤来初始化和更新 submodule。

### 方法一：克隆后初始化（两步走）
```bash
# 1. 克隆主仓库
git clone <repository-url>
cd <repository>

# 2. 初始化并更新所有 submodule
git submodule update --init --recursive
```

### 方法二：克隆时自动初始化（一步到位）
```bash
git clone --recurse-submodules <repository-url>
```

**推荐使用方法二**，更简洁高效。

### 示例
```bash
# 克隆 amdgpudriver 项目并自动初始化所有 submodule
git clone --recurse-submodules https://github.com/your-org/amdgpudriver.git
```

---

## 更新 Submodule

### 查看当前 Submodule 状态
```bash
git submodule status
```

输出示例：
```
 9f014db6a42d9e893127ea57ad9cc27e9c7445b5 ROCm_keyDriver/rocm-systems (hip-version_7.3.53390-2596-g9f014db6a4)
```

### 更新到 Submodule 的最新提交

#### 更新单个 Submodule
```bash
# 方法1：进入 submodule 目录手动更新
cd ROCm_keyDriver/rocm-systems
git fetch
git checkout develop  # rocm-systems 的主分支是 develop
git pull origin develop
cd ../..

# 在父仓库中提交这个更新
git add ROCm_keyDriver/rocm-systems
git commit -m "Update rocm-systems to latest"
```

#### 更新所有 Submodule 到远程最新版本
```bash
# 更新所有 submodule 到它们各自追踪分支的最新提交
git submodule update --remote

# 查看更改
git status

# 提交更新
git add .
git commit -m "Update all submodules to latest"
```

#### 更新到特定提交或标签
```bash
cd ROCm_keyDriver/rocm-systems
git checkout <commit-hash-or-tag>
cd ../..

git add ROCm_keyDriver/rocm-systems
git commit -m "Update rocm-systems to version X.Y.Z"
```

### 同步主仓库中的 Submodule 更新

当其他人更新了 submodule，你需要同步：
```bash
# 拉取主仓库的更新
git pull

# 更新 submodule 到主仓库记录的提交
git submodule update --init --recursive
```

---

## 查看 Submodule 状态

### 查看所有 Submodule
```bash
git submodule status
```

### 查看 Submodule 配置
```bash
cat .gitmodules
```

输出示例：
```
[submodule "ROCm_keyDriver/rocm-systems"]
    path = ROCm_keyDriver/rocm-systems
    url = https://github.com/ROCm/rocm-systems.git
```

### 查看 Submodule 详细信息
```bash
git submodule foreach 'echo $name && git remote -v'
```

### 检查 Submodule 是否有未提交的更改
```bash
git submodule foreach 'git status'
```

---

## 删除 Submodule

删除 submodule 需要几个步骤：

### 完整删除流程
```bash
# 1. 取消 submodule 的初始化
git submodule deinit -f <submodule-path>

# 2. 从 Git 索引和工作树中删除
git rm -f <submodule-path>

# 3. 删除 .git/modules 中的 submodule 数据
rm -rf .git/modules/<submodule-path>

# 4. 提交删除操作
git commit -m "Remove <submodule-name> submodule"
```

### 示例：删除 rocm-systems
```bash
git submodule deinit -f ROCm_keyDriver/rocm-systems
git rm -f ROCm_keyDriver/rocm-systems
rm -rf .git/modules/ROCm_keyDriver/rocm-systems
git commit -m "Remove rocm-systems submodule"
git push
```

---

## 常见问题和最佳实践

### 1. Submodule 目录为空

**问题：**克隆项目后，submodule 目录是空的。

**解决方案：**
```bash
git submodule update --init --recursive
```

### 2. Submodule 指向错误的提交

**问题：**Submodule 不在预期的提交位置。

**解决方案：**
```bash
git submodule update --init --recursive
```

### 3. 修改 Submodule URL

**场景：**Submodule 的远程仓库地址改变了。

**解决方案：**
```bash
# 1. 编辑 .gitmodules 文件，修改 url
vim .gitmodules

# 2. 同步配置
git submodule sync

# 3. 更新 submodule
git submodule update --init --recursive

# 4. 提交更改
git add .gitmodules
git commit -m "Update submodule URL"
```

### 4. 在 Submodule 中进行开发

**不推荐**在 submodule 中直接开发，因为：
- Submodule 默认处于 "detached HEAD" 状态
- 容易造成提交丢失

**如果必须修改 submodule：**
```bash
cd <submodule-path>
git checkout -b my-feature-branch
# 进行修改
git add .
git commit -m "My changes"
git push origin my-feature-branch

# 在主仓库中更新引用
cd ../..
git add <submodule-path>
git commit -m "Update submodule with my changes"
```

### 5. 使用 Submodule 的分支而非固定提交

**配置 submodule 追踪特定分支：**
```bash
git config -f .gitmodules submodule.<submodule-path>.branch <branch-name>
git submodule update --remote
```

**示例：让 rocm-systems 追踪 develop 分支**
```bash
git config -f .gitmodules submodule.ROCm_keyDriver/rocm-systems.branch develop
git submodule update --remote ROCm_keyDriver/rocm-systems
git add .gitmodules ROCm_keyDriver/rocm-systems
git commit -m "Configure rocm-systems to track develop branch"
```

### 6. 批量操作所有 Submodule

```bash
# 在所有 submodule 中执行命令
git submodule foreach '<command>'

# 示例：查看所有 submodule 的状态
git submodule foreach 'git status'

# 示例：在所有 submodule 中拉取最新代码
git submodule foreach 'git pull origin main'
```

### 7. 浅克隆 Submodule（节省空间和时间）

```bash
# 浅克隆主仓库和 submodule
git clone --recurse-submodules --shallow-submodules --depth 1 <repository-url>
```

### 8. Submodule 意外丢失或损坏

**问题：**目录或 Git 信息丢失。

**解决方案：**
```bash
# 重新初始化和更新
git submodule update --init --recursive --force

# 如果还有问题，先清理再重新添加
git submodule deinit -f <submodule-path>
rm -rf <submodule-path>
rm -rf .git/modules/<submodule-path>
git submodule update --init --recursive
```

---

## 最佳实践

### ✅ 推荐做法

1. **明确提交 Submodule 更新**
   - 每次更新 submodule 后，在主仓库中提交
   - 提交信息要清楚说明更新的原因和版本

2. **使用 `--recurse-submodules` 克隆**
   - 确保一次性获取所有依赖

3. **定期同步 Submodule**
   - 定期检查和更新 submodule 到最新稳定版本

4. **文档化 Submodule 的作用**
   - 在 README 中说明各个 submodule 的用途

5. **固定在稳定版本**
   - 对于生产环境，将 submodule 固定在已测试的稳定版本

6. **提交前检查状态**
   - 使用 `git submodule status` 确认 submodule 状态
   - 确保 `.gitmodules` 文件已提交

### ❌ 避免的做法

1. **不要在 Submodule 中直接开发**
   - 容易造成提交丢失和混乱

2. **不要忽略 `.gitmodules` 文件**
   - 这个文件是 submodule 配置的关键

3. **不要忘记提交 Submodule 更新**
   - 更新后必须在主仓库中提交引用变更

4. **不要混淆 Submodule 和 Subtree**
   - 根据实际需求选择合适的工具

5. **不要使用 `rm -rf` 直接删除**
   - 必须使用正确的 Git 命令序列删除

---

## 快速参考

### 常用命令速查表

| 操作 | 命令 |
|------|------|
| 添加 submodule | `git submodule add <url> [path]` |
| 克隆包含 submodule 的项目 | `git clone --recurse-submodules <url>` |
| 初始化 submodule | `git submodule init` |
| 更新 submodule | `git submodule update` |
| 初始化并更新（递归） | `git submodule update --init --recursive` |
| 更新到远程最新 | `git submodule update --remote` |
| 查看状态 | `git submodule status` |
| 在所有 submodule 执行命令 | `git submodule foreach '<command>'` |
| 删除 submodule | `git submodule deinit -f <path> && git rm -f <path>` |
| 同步 URL | `git submodule sync` |
| 强制重新初始化 | `git submodule update --init --recursive --force` |

### 当前项目的 Submodule

本项目 (`amdgpudriver`) 当前包含以下 submodule：

| Submodule | 路径 | 仓库 URL | 说明 |
|-----------|------|----------|------|
| rocm-systems | ROCm_keyDriver/rocm-systems | https://github.com/ROCm/rocm-systems.git | ROCm 超级仓库，包含所有核心组件 |

#### ROCm Systems 包含的项目

`rocm-systems` 是一个超级仓库，整合了以下 ROCm 核心项目：

- **amdsmi**: AMD 系统管理接口
- **aqlprofile**: AQL 性能分析工具
- **clr**: 通用语言运行时（包含 HIP 和 OpenCL）
- **hip**: HIP 编程接口
- **hip-tests**: HIP 测试套件
- **rccl**: ROCm 通信库
- **rdc**: ROCm 数据中心工具
- **rocm-core**: ROCm 核心组件
- **rocminfo**: 系统信息工具
- **rocm-smi-lib**: 系统管理接口库
- **rocprofiler**: 性能分析器
- **rocprofiler-compute**: 计算性能分析
- **rocprofiler-register**: 寄存器分析
- **rocprofiler-sdk**: 性能分析 SDK
- **rocprofiler-systems**: 系统级性能分析
- **rocr-runtime**: ROCm 运行时
- **roctracer**: ROCm 追踪工具

通过这一个 submodule，即可访问整个 ROCm 生态系统的源代码。

---

## 相关资源

- [Git 官方文档 - Submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [GitHub 文档 - Working with submodules](https://github.blog/2016-02-01-working-with-submodules/)
- [Atlassian Git Submodule Tutorial](https://www.atlassian.com/git/tutorials/git-submodule)
- [ROCm Examples GitHub](https://github.com/ROCm/rocm-examples)

---

## 更新记录

- 2026-01-16: 切换到 rocm-systems 超级仓库，简化 submodule 管理
- 2026-01-16: 重建后更新，添加故障恢复章节
- 2026-01-15: 初始版本创建

