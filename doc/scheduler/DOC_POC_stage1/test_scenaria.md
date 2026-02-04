
测试场景:
1. Online-AI-model: 具有高优先级，有实时性要求
2. Offline-AI-model: 低优先级，没实时性要求

我们需要设计一个测试框架(Test_framework)
Test_framework具有如下功能，可以明确知道Offline-AI-model和Online-AI-model的任务请求，也知道Online-AI-model或者offline-AI-model所用的Hardware-Queue或者Software-queue（我们需要实验确认这点），当Online-AI-model任务来时，我们需要suspend 所用的Hardware-Queue (Software-queue), 当Online-AI-model运行完(或者时间片到了)后，再resume Hardware-Queue (Software-queue).

