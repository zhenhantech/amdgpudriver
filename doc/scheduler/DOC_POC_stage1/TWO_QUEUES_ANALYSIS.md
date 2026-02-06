# 深入分析：2个Queue的真相

**日期**: 2026-02-05

## 🎯 核心结论

**实际只有1个用户Queue在工作！**

- **qid=0**: 用户Compute Queue (ROCr创建)
- **qid=1**: 系统HIQ (Hardware Independent Queue)

## 📊 证据

### ROCr层面：只有1个Queue
```
Created SWq=0x7f3e4e98e000 to map on HWq=0x7f3c29c00000
- 只有1次 acquireQueue 调用
- 所有209次操作都使用同一个HWq地址
```

### KFD层面：2个Queue
```
创建: KFD_MQD_CREATE qid=0 type=COMPUTE active=1
销毁: KFD_MQD_DESTROY qid=1 was_active=0 (先销毁)
销毁: KFD_MQD_DESTROY qid=0 was_active=0 (后销毁)
```

## 💡 关键洞察

### qid=0 (用户Queue)
- 类型: COMPUTE
- 活跃: 是 (105次Kernel提交)
- ROCr映射: SWq → HWq=0x7f3c29c00000

### qid=1 (HIQ - 系统Queue)
- 类型: type=0 (系统)
- 活跃: 否 (was_active=0)
- 证据: 销毁时检查了 `kfd_hiq_mqd_doorbell_id`
- 用途: 系统管理、Packet Manager、特权操作

## 📈 使用统计

| 指标 | qid=0 | qid=1 |
|------|-------|-------|
| Kernel提交 | 105次 | 0次 |
| Active | 是 | 否 |
| 销毁顺序 | 第2个 | 第1个 |

## 🎯 对抢占设计的影响

1. **简化设计**: 每个用户进程只有1个Compute Queue
2. **HIQ透明**: 系统Queue不参与用户Kernel执行
3. **抢占目标**: 只需控制用户的qid=0

---

详细分析见完整日志和ftrace。
