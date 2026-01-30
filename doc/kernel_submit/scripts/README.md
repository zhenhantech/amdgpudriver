# 验证和测试脚本

本目录包含用于验证 ROCm/AMDGPU 驱动和硬件配置的实用脚本。

## 📜 脚本列表

### 1. `partition_info.sh` - GPU 分区配置查看器

**功能**: 查看所有 GPU 的分区模式和硬件配置

**使用方法**:
```bash
# 普通用户权限（基础信息）
bash partition_info.sh

# Root 权限（完整信息，包括硬件单元数量）
sudo bash partition_info.sh
```

**输出信息**:
- 当前计算分区模式（SPX/DPX/TPX/QPX/CPX）
- 可用的分区模式
- 当前内存分区模式（NPS1/NPS4）
- XCC 实例数量
- SDMA 引擎数量
- 视频解码器数量
- JPEG 解码器数量
- Render 节点分布

**适用场景**:
- 验证 GPU 硬件配置
- 检查分区模式设置
- 调试多 GPU 环境
- 云环境资源分配验证

---

### 2. `enable_kfd_debug.sh` - 启用 KFD 调试日志

**功能**: 通过 Dynamic Debug 启用 KFD (Kernel Fusion Driver) 的调试日志

**使用方法**:
```bash
sudo bash enable_kfd_debug.sh
```

**启用的日志**:
- HQD (Hardware Queue Descriptor) 分配日志
- Queue 创建和销毁日志
- 其他 KFD 内部操作日志

**注意事项**:
- ⚠️ 需要 root 权限
- ⚠️ 需要内核支持 CONFIG_DYNAMIC_DEBUG
- ⚠️ 会产生大量日志，可能影响性能
- ⚠️ HIP/ROCm 会复用 Queue 池，需要系统重启或重新加载模块才能看到 HQD 分配日志

**查看日志**:
```bash
sudo dmesg -w                    # 实时查看
sudo dmesg | grep "hqd slot"     # 查看 HQD 分配
sudo dmesg | grep "kfd"          # 查看所有 KFD 日志
```

---

### 3. `disable_kfd_debug.sh` - 禁用 KFD 调试日志

**功能**: 禁用之前启用的 KFD 调试日志

**使用方法**:
```bash
sudo bash disable_kfd_debug.sh
```

**适用场景**:
- 调试完成后恢复正常日志级别
- 减少日志输出，提高性能

---

## 🔧 脚本开发指南

如果需要添加新的验证脚本：

1. **命名规范**: 使用小写字母和下划线，如 `verify_xxx.sh`
2. **添加注释**: 在脚本开头说明用途和使用方法
3. **设置权限**: `chmod +x script_name.sh`
4. **更新文档**: 在相关的 `.md` 文件中引用脚本

---

## 📚 相关文档

- [MEC_ARCHITECTURE.md](../MEC_ARCHITECTURE.md) - MEC 架构和 XCP 分区详解
- [VERIFICATION_GUIDE.md](../VERIFICATION_GUIDE.md) - 完整验证指南
- [KERNEL_TRACE_INDEX.md](../KERNEL_TRACE_INDEX.md) - 内核提交流程索引

---

**维护者**: ROCm Kernel Research Team  
**最后更新**: 2026-01-19

