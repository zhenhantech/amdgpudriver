#!/usr/bin/env python3
"""
Phase 4 Test: Dual Model Priority Scheduling (Intensive Configuration)
测试双模型优先级调度和 latency 保证 - 高负载配置

场景:
  - 高优先级任务: ResNet-18 在线推理 (20 reqs/sec, 50ms 间隔)
  - 低优先级任务: ResNet-50 批处理 (batch=1024, 连续运行)

目标:
  - 验证高负载下的优先级调度
  - 验证大 batch size 场景
  - 更长测试时间 (3 分钟)
"""

import torch
import torchvision.models as models
import multiprocessing as mp
import time
import numpy as np
import json
import sys
import os

def high_priority_worker(duration, queue):
    """
    高优先级任务: ResNet-18 在线推理
    
    模拟在线推理服务:
      - 20 reqs/sec (每 50ms 一个请求)
      - 每个请求处理 batch=1
      - 记录每个请求的延迟
    """
    try:
        print(f"[HIGH] Starting high priority task (ResNet-18)")
        print(f"[HIGH] Target: 20 reqs/sec (50ms interval), Duration: {duration}s")
        
        # 加载模型
        model = models.resnet18(weights=None).cuda()
        model.eval()
        
        # 准备输入
        x = torch.randn(1, 3, 224, 224).cuda()
        
        # Warmup
        print(f"[HIGH] Warming up...")
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()
        
        print(f"[HIGH] Warmup completed, starting test...")
        
        latencies = []
        start_time = time.time()
        request_count = 0
        
        while (time.time() - start_time) < duration:
            req_start = time.time()
            
            # 推理
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()
            
            latency = (time.time() - req_start) * 1000  # ms
            latencies.append(latency)
            request_count += 1
            
            # 20 reqs/sec = 50ms 间隔
            elapsed = time.time() - req_start
            sleep_time = max(0, 0.05 - elapsed)  # 50ms = 0.05s
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        elapsed = time.time() - start_time
        
        # 统计
        latencies = np.array(latencies)
        result = {
            'task': 'high_priority',
            'model': 'ResNet-18',
            'config': '20 reqs/sec (50ms interval)',
            'requests': len(latencies),
            'duration': elapsed,
            'throughput_rps': len(latencies) / elapsed,
            'latency_avg_ms': float(np.mean(latencies)),
            'latency_p50_ms': float(np.percentile(latencies, 50)),
            'latency_p95_ms': float(np.percentile(latencies, 95)),
            'latency_p99_ms': float(np.percentile(latencies, 99)),
            'latency_max_ms': float(np.max(latencies))
        }
        
        print(f"\n[HIGH] Results:")
        print(f"  Requests: {result['requests']}")
        print(f"  Throughput: {result['throughput_rps']:.2f} req/s")
        print(f"  Latency Avg: {result['latency_avg_ms']:.2f} ms")
        print(f"  Latency P50: {result['latency_p50_ms']:.2f} ms")
        print(f"  Latency P95: {result['latency_p95_ms']:.2f} ms")
        print(f"  Latency P99: {result['latency_p99_ms']:.2f} ms")
        print(f"  Latency Max: {result['latency_max_ms']:.2f} ms")
        
        queue.put(result)
        
    except Exception as e:
        print(f"[HIGH] Error: {e}")
        import traceback
        traceback.print_exc()
        queue.put({'error': str(e)})

def low_priority_worker(duration, queue):
    """
    低优先级任务: ResNet-50 批处理
    
    模拟批处理任务:
      - 连续运行，不间断
      - 使用超大 batch size (1024)
      - 记录总吞吐量
    """
    try:
        print(f"[LOW] Starting low priority task (ResNet-50)")
        print(f"[LOW] Mode: Continuous inference, Batch=1024, Duration: {duration}s")
        
        # 加载模型
        model = models.resnet50(weights=None).cuda()
        model.eval()
        
        # 准备输入（超大 batch）
        batch_size = 1024
        print(f"[LOW] Allocating batch size {batch_size}...")
        x = torch.randn(batch_size, 3, 224, 224).cuda()
        print(f"[LOW] Batch allocated successfully")
        
        # Warmup
        print(f"[LOW] Warming up...")
        for _ in range(5):  # 减少 warmup 次数（batch 太大）
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()
        
        print(f"[LOW] Warmup completed, starting test...")
        
        count = 0
        start_time = time.time()
        
        while (time.time() - start_time) < duration:
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()
            count += 1
            
            # 每 10 次迭代报告一次进度
            if count % 10 == 0:
                elapsed = time.time() - start_time
                throughput = count / elapsed
                print(f"[LOW] Progress: {count} iterations, {throughput:.2f} iter/s")
        
        elapsed = time.time() - start_time
        
        # 统计
        result = {
            'task': 'low_priority',
            'model': 'ResNet-50',
            'config': f'batch={batch_size}, continuous',
            'batch_size': batch_size,
            'iterations': count,
            'duration': elapsed,
            'throughput_ips': count / elapsed,
            'images_per_sec': (count * batch_size) / elapsed
        }
        
        print(f"\n[LOW] Results:")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Batch size: {result['batch_size']}")
        print(f"  Throughput: {result['throughput_ips']:.2f} iter/s")
        print(f"  Images/sec: {result['images_per_sec']:.1f}")
        
        queue.put(result)
        
    except Exception as e:
        print(f"[LOW] Error: {e}")
        import traceback
        traceback.print_exc()
        queue.put({'error': str(e)})

def run_test(duration=180, output_file=None):
    """
    运行双模型测试（高负载配置）
    """
    print("=" * 70)
    print("Phase 4: Dual Model Priority Scheduling Test (Intensive)")
    print("=" * 70)
    print(f"\nTest configuration:")
    print(f"  Duration: {duration}s (3 minutes)")
    print(f"  High priority: ResNet-18, 20 reqs/sec (50ms interval)")
    print(f"  Low priority:  ResNet-50, batch=1024, continuous")
    print()
    
    # 检查 XSched
    if 'LD_PRELOAD' in os.environ and 'libshimhip.so' in os.environ['LD_PRELOAD']:
        mode = "XSched"
        print("✅ Running with XSched")
        print(f"   LD_PRELOAD: {os.environ['LD_PRELOAD']}")
    else:
        mode = "Native"
        print("ℹ️  Running with Native scheduler (no XSched)")
    
    print("\n⚠️  WARNING: This is an intensive test!")
    print("   - High request rate (20 req/s)")
    print("   - Large batch size (1024)")
    print("   - Long duration (3 minutes)")
    print("   - May require significant GPU memory")
    print()
    
    print("Starting workers...")
    print("-" * 70)
    
    # 创建队列
    high_queue = mp.Queue()
    low_queue = mp.Queue()
    
    # 创建进程
    high_proc = mp.Process(target=high_priority_worker, args=(duration, high_queue))
    low_proc = mp.Process(target=low_priority_worker, args=(duration, low_queue))
    
    # 启动进程
    high_proc.start()
    time.sleep(1)  # 稍微错开启动
    low_proc.start()
    
    # 等待完成
    high_proc.join()
    low_proc.join()
    
    # 获取结果
    high_result = high_queue.get()
    low_result = low_queue.get()
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    # 整合结果
    results = {
        'test_name': 'phase4_dual_model_intensive',
        'mode': mode,
        'duration': duration,
        'high_priority': high_result,
        'low_priority': low_result,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 打印汇总
    if 'error' not in high_result:
        print(f"\nHigh Priority (ResNet-18, 20 req/s):")
        print(f"  Total Requests: {high_result['requests']}")
        print(f"  P99 Latency: {high_result['latency_p99_ms']:.2f} ms")
        print(f"  Throughput:  {high_result['throughput_rps']:.2f} req/s")
    
    if 'error' not in low_result:
        print(f"\nLow Priority (ResNet-50, batch=1024):")
        print(f"  Total Iterations: {low_result['iterations']}")
        print(f"  Throughput:  {low_result['throughput_ips']:.2f} iter/s")
        print(f"  Images/sec:  {low_result['images_per_sec']:.1f}")
    
    # 保存结果
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {output_file}")
    
    print("=" * 70)
    
    return results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 4 Dual Model Test (Intensive)')
    parser.add_argument('--duration', type=int, default=180,
                       help='Test duration in seconds (default: 180 = 3 minutes)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path')
    
    args = parser.parse_args()
    
    try:
        results = run_test(duration=args.duration, output_file=args.output)
        
        # 简单判断
        if 'error' not in results['high_priority']:
            high_p99 = results['high_priority']['latency_p99_ms']
            if high_p99 < 100:
                print(f"\n✅ High priority P99 ({high_p99:.2f}ms) looks reasonable")
            else:
                print(f"\n⚠️  High priority P99 ({high_p99:.2f}ms) seems high")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
