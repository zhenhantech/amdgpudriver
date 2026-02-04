# POC Stage 1 实施 TODO List

**基于方案**: ARCH_Design_01_POC_Stage1_实施方案.md  
**创建日期**: 2026-02-03  
**预计时间**: 7-10天  
**当前进度**: 0/4 个阶段完成

---

## 📊 总体进度概览

- [ ] Phase 1: API 验证和封装 (2天)
- [ ] Phase 2: 队列识别机制 (2天)
- [ ] Phase 3: Test Framework 主程序 (2天)
- [ ] Phase 4: 测试和验证 (2-3天)

**完成度**: 0/4 (0%)

---

## 🔬 Phase 1: API 验证和封装 (2天)

**目标**: 验证 suspend_queues API 可用，并提供 C 库封装

### 1.1 API 可用性验证

**文件**: `test_api_availability.c` (新建)

- [ ] 测试 KFD_IOC_DBG_TRAP ioctl 是否存在
  ```c
  int fd = open("/dev/kfd", O_RDWR);
  struct kfd_ioctl_dbg_trap_args args = {0};
  args.op = KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT;
  int ret = ioctl(fd, AMDKFD_IOC_DBG_TRAP, &args);
  // 检查返回值
  ```

- [ ] 测试 suspend_queues 是否可用
  - [ ] 创建测试队列
  - [ ] 尝试调用 suspend
  - [ ] 检查错误码

- [ ] 测试 resume_queues 是否可用
  - [ ] 恢复测试队列
  - [ ] 验证队列继续执行

- [ ] 权限要求验证
  - [ ] 测试是否需要 root
  - [ ] 测试是否需要 CAP_SYS_ADMIN

### 1.2 C 库实现

**目录**: `poc_stage1/libgpreempt_poc/` (新建)

**文件**: `gpreempt_poc.h` (新建)

- [ ] 定义公共接口
  ```c
  // 初始化/清理
  int gpreempt_poc_init(void);
  void gpreempt_poc_cleanup(void);
  
  // 队列操作
  int gpreempt_suspend_queues(uint32_t *queue_ids, 
                             uint32_t num_queues,
                             uint32_t grace_period_us);
  
  int gpreempt_resume_queues(uint32_t *queue_ids,
                            uint32_t num_queues);
  
  // 队列查询
  typedef struct {
      uint32_t queue_id;
      uint32_t priority;
      uint32_t gpu_id;
      pid_t process_id;
      bool is_active;
      uint64_t queue_address;
  } gpreempt_queue_info_t;
  
  int gpreempt_get_all_queues(gpreempt_queue_info_t **queues,
                             uint32_t *num_queues);
  
  int gpreempt_find_queues_by_priority(uint32_t min_prio,
                                      uint32_t max_prio,
                                      gpreempt_queue_info_t **queues,
                                      uint32_t *num_queues);
  
  int gpreempt_find_queues_by_process(pid_t pid,
                                     gpreempt_queue_info_t **queues,
                                     uint32_t *num_queues);
  
  // 辅助函数
  void gpreempt_free_queue_info(gpreempt_queue_info_t *queues);
  ```

**文件**: `gpreempt_poc.c` (新建)

- [ ] 实现 gpreempt_poc_init()
  - [ ] 打开 /dev/kfd
  - [ ] 保存文件描述符
  - [ ] 错误处理

- [ ] 实现 gpreempt_poc_cleanup()
  - [ ] 关闭 /dev/kfd
  - [ ] 清理资源

- [ ] 实现 gpreempt_suspend_queues()
  - [ ] 构建 ioctl 参数
  - [ ] 调用 AMDKFD_IOC_DBG_TRAP
  - [ ] 错误处理和日志

- [ ] 实现 gpreempt_resume_queues()
  - [ ] 构建 ioctl 参数
  - [ ] 调用 AMDKFD_IOC_DBG_TRAP
  - [ ] 错误处理和日志

- [ ] 实现 gpreempt_get_all_queues()
  - [ ] 打开 /sys/kernel/debug/kfd/mqds
  - [ ] 解析 MQD 格式
  - [ ] 提取队列信息（ID, 优先级, 状态等）
  - [ ] 分配内存并返回

- [ ] 实现 gpreempt_find_queues_by_priority()
  - [ ] 调用 get_all_queues
  - [ ] 过滤指定优先级范围
  - [ ] 返回结果

- [ ] 实现 gpreempt_find_queues_by_process()
  - [ ] 解析 MQD 中的 process 信息
  - [ ] 按 pid 过滤

- [ ] 实现 gpreempt_free_queue_info()
  - [ ] 释放分配的内存

**文件**: `Makefile` (新建)

- [ ] 编译规则
  ```makefile
  CC = gcc
  CFLAGS = -Wall -Wextra -g -fPIC
  
  libgpreempt_poc.so: gpreempt_poc.o
      $(CC) -shared -o $@ $^
  
  gpreempt_poc.o: gpreempt_poc.c gpreempt_poc.h
      $(CC) $(CFLAGS) -c gpreempt_poc.c
  
  clean:
      rm -f *.o *.so
  
  install:
      cp libgpreempt_poc.so /usr/local/lib/
      cp gpreempt_poc.h /usr/local/include/
      ldconfig
  ```

### ✅ Phase 1 验证标准

- [ ] test_api_availability 成功运行
- [ ] suspend_queues 能正常暂停队列
- [ ] resume_queues 能正常恢复队列
- [ ] 库编译无错误和警告
- [ ] 能正确解析 MQD debugfs

---

## 🔍 Phase 2: 队列识别机制 (2天)

**目标**: 能可靠地识别 Online/Offline 队列

### 2.1 MQD Debugfs 解析

**文件**: `mqd_parser.c` (新建，集成到 libgpreempt_poc)

- [ ] 实现 MQD 格式解析
  ```c
  // MQD debugfs 格式示例:
  // "Compute queue on device 0001:01:00.0
  //     Queue ID: 1 (0x1)
  //     Address: 0x7f5a00000000
  //     Process: pid 12345 pasid 0x8001
  //     is active: yes
  //     priority: 2"
  
  typedef struct {
      char line[256];
      char *pos;
  } mqd_parser_t;
  
  int mqd_parse_queue_block(mqd_parser_t *parser,
                           gpreempt_queue_info_t *info);
  ```

- [ ] 提取关键字段
  - [ ] Queue ID (十进制和十六进制)
  - [ ] Process ID (pid)
  - [ ] Priority
  - [ ] is active 状态
  - [ ] Queue Address

- [ ] 错误处理
  - [ ] 格式变化容错
  - [ ] 缺失字段处理

### 2.2 队列分类策略

**策略 A: 按优先级分类** (最简单，推荐)

- [ ] 实现优先级阈值分类
  ```c
  #define ONLINE_PRIORITY_THRESHOLD  10
  
  bool is_online_queue(gpreempt_queue_info_t *q) {
      return q->priority >= ONLINE_PRIORITY_THRESHOLD;
  }
  ```

- [ ] 配置化阈值
  - [ ] 从环境变量读取
  - [ ] 从配置文件读取

**策略 B: 按进程 PID** (备选)

- [ ] 实现 PID 映射
  ```python
  # Python 侧
  online_pid = os.getpid()
  online_queues = gpreempt_find_queues_by_process(online_pid)
  ```

- [ ] 多进程支持
  - [ ] 维护 PID → 类型的映射表

**策略 C: 按队列地址范围** (备选)

- [ ] 预分配地址空间
  - [ ] Online 队列使用固定地址范围
  - [ ] Offline 队列使用另一地址范围

### 2.3 自动发现机制

- [ ] 定期扫描 MQD
  - [ ] 每秒扫描一次
  - [ ] 发现新队列自动分类

- [ ] 队列生命周期管理
  - [ ] 检测队列创建
  - [ ] 检测队列销毁
  - [ ] 更新内部队列列表

### ✅ Phase 2 验证标准

- [ ] 能正确识别 Online 队列
- [ ] 能正确识别 Offline 队列
- [ ] 队列分类准确率 100%
- [ ] 能处理队列动态创建/销毁
- [ ] MQD 解析鲁棒性测试通过

---

## 🎮 Phase 3: Test Framework 主程序 (2天)

**目标**: 实现完整的测试框架和监控逻辑

### 3.1 Python Framework 核心

**目录**: `poc_stage1/test_framework/` (新建)

**文件**: `gpreempt_scheduler.py` (新建)

- [ ] 实现 GPreemptScheduler 类
  ```python
  class GPreemptScheduler:
      def __init__(self, check_interval_ms=1):
          self.online_queues = []
          self.offline_queues = []
          self.online_task_pending = False
          self.monitor_thread = None
          self.running = True
          self.check_interval = check_interval_ms / 1000.0
      
      def start(self):
          """启动监控线程"""
          self.monitor_thread = threading.Thread(
              target=self._monitor_loop)
          self.monitor_thread.start()
      
      def stop(self):
          """停止监控"""
          self.running = False
          self.monitor_thread.join()
      
      def _monitor_loop(self):
          """监控主循环"""
          while self.running:
              time.sleep(self.check_interval)
              if self.online_task_pending:
                  self._handle_online_task()
  ```

- [ ] 实现队列注册
  - [ ] register_online_queue()
  - [ ] register_offline_queue()
  - [ ] unregister_queue()

- [ ] 实现抢占逻辑
  - [ ] _handle_online_task()
  - [ ] _suspend_offline_queues()
  - [ ] _resume_offline_queues()

- [ ] 实现完成检测
  - [ ] _wait_for_online_completion()
  - [ ] 方法1: 固定时间片
  - [ ] 方法2: 轮询队列状态（通过 rptr/wptr）

### 3.2 AI 模型包装

**文件**: `ai_model_wrapper.py` (新建)

- [ ] Online-AI 模型包装
  ```python
  class OnlineAIModel:
      def __init__(self, sched):
          self.sched = sched
          self.model = load_model("推理模型")
          self.queue_ids = []
      
      def inference(self, input_data):
          # 通知调度器
          self.sched.notify_online_task()
          
          # 执行推理
          result = self.model.forward(input_data)
          
          # 完成通知
          self.sched.online_task_complete()
          
          return result
  ```

- [ ] Offline-AI 模型包装
  ```python
  class OfflineAIModel:
      def __init__(self, sched):
          self.sched = sched
          self.model = load_model("训练模型")
          self.queue_ids = []
      
      def train_step(self, batch):
          # 训练一个 batch
          loss = self.model.train_step(batch)
          return loss
      
      def train_loop(self, epochs):
          # 持续训练循环
          for epoch in range(epochs):
              for batch in dataloader:
                  loss = self.train_step(batch)
  ```

### 3.3 测试主程序

**文件**: `test_priority_scheduling.py` (新建)

- [ ] 实现测试入口
  ```python
  def main():
      # 1. 初始化调度器
      sched = GPreemptScheduler(check_interval_ms=1)
      sched.start()
      
      # 2. 启动 Offline 模型
      offline = OfflineAIModel(sched)
      offline_thread = threading.Thread(
          target=offline.train_loop, args=(100,))
      offline_thread.start()
      
      # 等待队列创建
      time.sleep(2)
      
      # 3. 扫描并注册 Offline 队列
      offline_queues = scan_queues(min_prio=0, max_prio=5)
      for q in offline_queues:
          sched.register_offline_queue(q.queue_id, q.priority)
      
      # 4. 启动 Online 模型
      online = OnlineAIModel(sched)
      
      # 等待队列创建
      time.sleep(1)
      
      # 5. 注册 Online 队列
      online_queues = scan_queues(min_prio=10, max_prio=15)
      for q in online_queues:
          sched.register_online_queue(q.queue_id, q.priority)
      
      # 6. 模拟 Online 请求
      for i in range(20):
          print(f"\n=== Online 请求 #{i+1} ===")
          latency = online.inference(test_input)
          print(f"延迟: {latency:.2f} ms")
          time.sleep(0.5)  # 每 500ms 一个请求
      
      # 7. 清理
      sched.stop()
      offline_thread.join()
  ```

- [ ] 实现辅助函数
  - [ ] scan_queues()
  - [ ] measure_latency()
  - [ ] log_statistics()

### 3.4 日志和统计

- [ ] 实现统计收集
  ```python
  class Statistics:
      def __init__(self):
          self.online_count = 0
          self.suspend_count = 0
          self.resume_count = 0
          self.suspend_latencies = []
          self.resume_latencies = []
          self.online_latencies = []
      
      def record_suspend(self, latency_ms):
          self.suspend_count += 1
          self.suspend_latencies.append(latency_ms)
      
      def print_summary(self):
          print(f"\n=== 统计摘要 ===")
          print(f"Online 任务: {self.online_count}")
          print(f"Suspend 次数: {self.suspend_count}")
          print(f"Resume 次数: {self.resume_count}")
          print(f"平均 Suspend 延迟: {np.mean(self.suspend_latencies):.2f} ms")
          print(f"平均 Resume 延迟: {np.mean(self.resume_latencies):.2f} ms")
          print(f"平均 Online 延迟: {np.mean(self.online_latencies):.2f} ms")
  ```

- [ ] 实现日志输出
  - [ ] 时间戳
  - [ ] 事件类型
  - [ ] 队列状态
  - [ ] 延迟数据

### ✅ Phase 3 验证标准

- [ ] Test Framework 能正常启动
- [ ] 能正确识别和注册队列
- [ ] 监控线程正常工作
- [ ] 能触发 suspend/resume
- [ ] 统计数据正确收集
- [ ] 日志输出完整

---

## 🧪 Phase 4: 测试和验证 (2-3天)

**目标**: 全面测试和性能验证

### 4.1 功能测试

**Test Case 1: 基本抢占测试** (`test_basic_preemption.py`)

- [ ] 测试场景
  - [ ] 启动 Offline 模型（持续训练）
  - [ ] 等待稳定（10秒）
  - [ ] 触发 Online 任务
  - [ ] 验证 Offline 被暂停
  - [ ] 验证 Online 正确执行
  - [ ] 验证 Offline 恢复

- [ ] 验证点
  - [ ] Offline 队列从 active 变为 inactive
  - [ ] Online 任务延迟 < 50ms
  - [ ] Offline 恢复后继续执行（无数据丢失）
  - [ ] 无内核错误或崩溃

**Test Case 2: 频繁抢占测试** (`test_frequent_preemption.py`)

- [ ] 测试场景
  - [ ] Offline 持续运行
  - [ ] Online 每 100ms 提交一次
  - [ ] 持续 5 分钟

- [ ] 验证点
  - [ ] 所有 Online 任务成功执行
  - [ ] Offline 吞吐量下降 < 20%
  - [ ] 无错误或崩溃
  - [ ] 内存无泄漏

**Test Case 3: 边界条件测试** (`test_edge_cases.py`)

- [ ] 空队列暂停
  - [ ] suspend 不存在的队列
  - [ ] 验证错误处理

- [ ] 重复 suspend
  - [ ] suspend 已经暂停的队列
  - [ ] 验证幂等性

- [ ] 重复 resume
  - [ ] resume 已经运行的队列
  - [ ] 验证幂等性

- [ ] 并发操作
  - [ ] 同时 suspend 多个队列
  - [ ] 验证原子性

### 4.2 性能测试

**延迟测试** (`test_latency.py`)

- [ ] 测量 suspend_queues 延迟
  ```python
  start = time.time()
  gpreempt_suspend_queues(queue_ids, num_queues, 1000)
  end = time.time()
  suspend_latency = (end - start) * 1000  # ms
  ```
  - [ ] 目标: < 5ms
  - [ ] 重复 100 次，计算平均值和标准差

- [ ] 测量 resume_queues 延迟
  - [ ] 目标: < 5ms
  - [ ] 重复 100 次测量

- [ ] 测量 Online 端到端延迟
  - [ ] 从任务提交到完成
  - [ ] 目标: < 50ms
  - [ ] 包含抢占开销

**吞吐量测试** (`test_throughput.py`)

- [ ] Baseline: Offline 单独运行
  - [ ] 运行 5 分钟
  - [ ] 记录处理的 batch 数

- [ ] With Preemption: Offline + Online 混合
  - [ ] Online 每秒 2 次请求
  - [ ] 运行 5 分钟
  - [ ] 记录 Offline 处理的 batch 数

- [ ] 计算吞吐量损失
  ```python
  throughput_loss = (baseline_throughput - mixed_throughput) / baseline_throughput * 100
  # 目标: < 20%
  ```

### 4.3 稳定性测试

**长时间运行测试** (`test_stability.py`)

- [ ] 运行 1 小时
  - [ ] Offline 持续训练
  - [ ] Online 随机间隔请求（1-10秒）
  - [ ] 监控系统资源（CPU, 内存）

- [ ] 验证点
  - [ ] 无崩溃
  - [ ] 无 dmesg 错误
  - [ ] 内存使用稳定（无泄漏）
  - [ ] 所有任务正确完成

**压力测试** (`test_stress.py`)

- [ ] 高频 Online 请求
  - [ ] 每 10ms 一个请求
  - [ ] 持续 10 分钟

- [ ] 多 Offline 队列
  - [ ] 创建 10 个 Offline 队列
  - [ ] 全部需要暂停和恢复

### 4.4 结果分析

- [ ] 生成测试报告
  - [ ] 功能测试结果表
  - [ ] 性能测试数据图表
  - [ ] 延迟分布直方图

- [ ] 性能分析
  - [ ] 识别性能瓶颈
  - [ ] 与目标对比
  - [ ] 优化建议

- [ ] 问题清单
  - [ ] 发现的 bug
  - [ ] 限制和风险
  - [ ] 待改进项

### ✅ Phase 4 验证标准

**功能验证**
- [ ] 所有功能测试 100% 通过
- [ ] 所有边界测试通过
- [ ] 无未处理的异常

**性能验证**
- [ ] Online 延迟 < 50ms (可接受)
- [ ] Online 延迟 < 10ms (理想)
- [ ] Suspend 延迟 < 5ms
- [ ] Resume 延迟 < 5ms
- [ ] Offline 吞吐量损失 < 20%

**稳定性验证**
- [ ] 1 小时长时间运行无错误
- [ ] 高频测试无崩溃
- [ ] 无内存泄漏
- [ ] 系统资源使用正常

---

## 📂 文件结构

```
poc_stage1/
├── libgpreempt_poc/          # C 库
│   ├── gpreempt_poc.h
│   ├── gpreempt_poc.c
│   ├── mqd_parser.c
│   ├── Makefile
│   └── README.md
│
├── test_framework/           # Python 测试框架
│   ├── gpreempt_scheduler.py
│   ├── ai_model_wrapper.py
│   ├── test_priority_scheduling.py
│   └── requirements.txt
│
├── tests/                    # 测试用例
│   ├── test_basic_preemption.py
│   ├── test_frequent_preemption.py
│   ├── test_edge_cases.py
│   ├── test_latency.py
│   ├── test_throughput.py
│   ├── test_stability.py
│   └── test_stress.py
│
├── tools/                    # 辅助工具
│   ├── test_api_availability.c
│   ├── scan_queues.py
│   └── visualize_results.py
│
├── docs/                     # 文档
│   ├── ARCH_Design_01_POC_Stage1_实施方案.md
│   ├── ARCH_Design_02_三种API技术对比.md
│   ├── POC_Stage1_TODOLIST.md  (本文档)
│   └── test_scenaria.md
│
└── results/                  # 测试结果
    ├── functional_tests/
    ├── performance_tests/
    └── reports/
```

---

## 📊 里程碑

### Milestone 1: API 验证 (完成 Phase 1)
- [ ] API 可用性确认
- [ ] C 库编译成功
- [ ] 基本功能测试通过

### Milestone 2: 框架完成 (完成 Phase 2-3)
- [ ] 队列识别机制工作
- [ ] Test Framework 运行
- [ ] 能触发抢占

### Milestone 3: 验证成功 (完成 Phase 4)
- [ ] 所有测试通过
- [ ] 性能达标
- [ ] 报告完成

---

## 🐛 风险和应对

### 风险 1: suspend_queues 延迟太高

**症状**: Online 延迟 > 50ms

**原因分析**:
- ioctl 系统调用开销
- suspend_queues 内部逻辑复杂
- DQM 层额外处理

**应对方案**:
- → 升级到 POC Stage 2 (CWSR 直接使用)
- → 绕过 debugfs trap 接口
- → 预期延迟降低到 ~100μs

### 风险 2: 队列识别不可靠

**症状**: 无法准确识别 Online/Offline 队列

**原因分析**:
- MQD debugfs 格式不稳定
- 优先级信息缺失
- 进程信息不准确

**应对方案**:
- 方案 A: 使用环境变量标记
- 方案 B: 修改 HIP Runtime 添加标记
- 方案 C: 使用专门的队列创建 API

### 风险 3: 频繁抢占导致不稳定

**症状**: 系统崩溃或驱动错误

**原因分析**:
- suspend_queues 不是为高频使用设计的
- 可能存在竞态条件
- 资源泄漏

**应对方案**:
- 降低抢占频率
- 添加错误恢复机制
- 升级到 Stage 2 或 Stage 3

---

## 📈 进度跟踪

**开始日期**: 2026-02-03  
**预计完成日期**: 2026-02-13 (10 个工作日)

| Phase | 状态 | 开始日期 | 完成日期 | 实际用时 |
|-------|------|---------|---------|----------|
| Phase 1 | ⏸️ 未开始 | - | - | - |
| Phase 2 | ⏸️ 未开始 | - | - | - |
| Phase 3 | ⏸️ 未开始 | - | - | - |
| Phase 4 | ⏸️ 未开始 | - | - | - |

**总体进度**: 0% (0/4 phases)

---

## 📚 参考资料

### KFD 源码

- `kfd_chardev.c:3310-3321` - suspend/resume_queues 实现
- `kfd_device_queue_manager.c` - DQM 层接口
- `include/uapi/linux/kfd_ioctl.h` - ioctl 定义

### 相关文档

- `ARCH_Design_01_POC_Stage1_实施方案.md` - 整体方案
- `ARCH_Design_02_三种API技术对比.md` - API 对比
- `../DOC_GPREEMPT/TODOLIST.md` - 完整实施计划
- `../DOC_GPREEMPT/CWSR_API_USAGE_REFERENCE.md` - CWSR 参考

---

## ➡️ Stage 2 预研

如果 Stage 1 成功但性能不满足（延迟 > 10ms），准备：

- [ ] 研究 CWSR API 直接使用的可行性
- [ ] 设计新的 ioctl 接口
- [ ] 评估内核修改的工作量
- [ ] 准备 Stage 2 实施计划

---

**最后更新**: 2026-02-03  
**维护者**: Zhehan

**下一步**: 开始 Phase 1 - API 验证和封装 🚀
