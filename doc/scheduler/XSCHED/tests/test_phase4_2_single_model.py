#!/usr/bin/env python3
"""
Phase 4 Test 2: Single Model Baseline Performance
测试单个模型在 Baseline 和 XSched 下的性能
"""
import torch
import torchvision.models as models
import time
import json
import sys
import numpy as np
from datetime import datetime

def run_single_model_test(model_name, batch_size, num_iterations, device='cuda:0'):
    """运行单模型测试"""
    print(f"\n{'='*70}")
    print(f"Testing {model_name} (batch={batch_size}, iterations={num_iterations})")
    print(f"{'='*70}")
    
    # 加载模型
    print(f"[1/3] Loading model...")
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=False).to(device).eval()
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=False).to(device).eval()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"  ✅ Model loaded on {device}")
    
    # 准备输入
    input_tensor = torch.randn(batch_size, 3, 224, 224, device=device)
    
    # 预热
    print(f"[2/3] Warming up...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_tensor)
    torch.cuda.synchronize()
    print(f"  ✅ Warmup complete")
    
    # 测试
    print(f"[3/3] Running {num_iterations} iterations...")
    latencies = []
    
    with torch.no_grad():
        for i in range(num_iterations):
            start = time.time()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_iterations}")
    
    # 计算统计
    latencies = np.array(latencies)
    results = {
        'model': model_name,
        'batch_size': batch_size,
        'iterations': num_iterations,
        'device': device,
        'latency_avg': float(np.mean(latencies)),
        'latency_std': float(np.std(latencies)),
        'latency_min': float(np.min(latencies)),
        'latency_p50': float(np.percentile(latencies, 50)),
        'latency_p95': float(np.percentile(latencies, 95)),
        'latency_p99': float(np.percentile(latencies, 99)),
        'latency_max': float(np.max(latencies)),
        'throughput': 1000.0 / np.mean(latencies),  # iter/s
    }
    
    print(f"\n{'='*70}")
    print(f"Results for {model_name}:")
    print(f"  Throughput: {results['throughput']:.2f} iter/s")
    print(f"  Latency Avg: {results['latency_avg']:.2f} ms")
    print(f"  Latency P50: {results['latency_p50']:.2f} ms")
    print(f"  Latency P95: {results['latency_p95']:.2f} ms")
    print(f"  Latency P99: {results['latency_p99']:.2f} ms")
    print(f"  Latency Max: {results['latency_max']:.2f} ms")
    print(f"{'='*70}")
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Phase 4 Test 2: Single Model Test')
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'resnet50'],
                        help='Model to test')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations')
    parser.add_argument('--output', required=True, help='Output JSON file')
    args = parser.parse_args()
    
    print("="*70)
    print("Phase 4 Test 2: Single Model Performance")
    print("="*70)
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Batch Size: {args.batch}")
    print(f"  Iterations: {args.iterations}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    
    # 运行测试
    results = run_single_model_test(args.model, args.batch, args.iterations)
    results['timestamp'] = datetime.now().isoformat()
    results['torch_version'] = torch.__version__
    
    # 保存结果
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {args.output}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
