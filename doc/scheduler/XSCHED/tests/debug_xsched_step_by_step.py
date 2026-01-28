#!/usr/bin/env python3
"""
XSched Debug Script - Step by Step
渐进式测试 XSched，从简单到复杂
"""
import torch
import sys
import os

def test_step_1_basic_tensor():
    """Step 1: 测试基础 tensor 操作"""
    print("\n" + "="*70)
    print("Step 1: Basic Tensor Operations")
    print("="*70)
    
    try:
        print("[1.1] Creating CPU tensor...")
        a = torch.randn(100, 100)
        print(f"  ✅ CPU tensor created: {a.shape}")
        
        print("[1.2] Moving to GPU...")
        a_gpu = a.to('cuda:0')
        print(f"  ✅ GPU tensor created: {a_gpu.device}")
        
        print("[1.3] Simple addition...")
        b_gpu = torch.randn(100, 100, device='cuda:0')
        c_gpu = a_gpu + b_gpu
        print(f"  ✅ Addition completed: {c_gpu.shape}")
        
        print("[1.4] Synchronize...")
        torch.cuda.synchronize()
        print(f"  ✅ Synchronization successful")
        
        print("\n✅ Step 1 PASSED: Basic tensor operations work")
        return True
    except Exception as e:
        print(f"\n❌ Step 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step_2_matmul():
    """Step 2: 测试矩阵乘法"""
    print("\n" + "="*70)
    print("Step 2: Matrix Multiplication")
    print("="*70)
    
    try:
        print("[2.1] Creating matrices...")
        a = torch.randn(256, 256, device='cuda:0')
        b = torch.randn(256, 256, device='cuda:0')
        print(f"  ✅ Matrices created")
        
        print("[2.2] Matrix multiplication...")
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        print(f"  ✅ Matmul completed: {c.shape}")
        
        print("[2.3] Multiple matmuls...")
        for i in range(5):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        print(f"  ✅ 5 matmuls completed")
        
        print("\n✅ Step 2 PASSED: Matrix multiplication works")
        return True
    except Exception as e:
        print(f"\n❌ Step 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step_3_conv2d():
    """Step 3: 测试卷积操作（MIOpen）"""
    print("\n" + "="*70)
    print("Step 3: Convolution (MIOpen)")
    print("="*70)
    
    try:
        print("[3.1] Creating conv layer...")
        conv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1).to('cuda:0').eval()
        print(f"  ✅ Conv layer created")
        
        print("[3.2] Creating input tensor...")
        x = torch.randn(1, 3, 224, 224, device='cuda:0')
        print(f"  ✅ Input created: {x.shape}")
        
        print("[3.3] Forward pass (small batch)...")
        with torch.no_grad():
            y = conv(x)
        torch.cuda.synchronize()
        print(f"  ✅ Conv forward completed: {y.shape}")
        
        print("[3.4] Multiple forward passes...")
        with torch.no_grad():
            for i in range(3):
                y = conv(x)
        torch.cuda.synchronize()
        print(f"  ✅ 3 conv passes completed")
        
        print("\n✅ Step 3 PASSED: Convolution (MIOpen) works")
        return True
    except Exception as e:
        print(f"\n❌ Step 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step_4_simple_model():
    """Step 4: 测试简单模型"""
    print("\n" + "="*70)
    print("Step 4: Simple Model (Few Layers)")
    print("="*70)
    
    try:
        print("[4.1] Creating simple model...")
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 10)
        ).to('cuda:0').eval()
        print(f"  ✅ Model created")
        
        print("[4.2] Forward pass...")
        x = torch.randn(1, 3, 32, 32, device='cuda:0')
        with torch.no_grad():
            y = model(x)
        torch.cuda.synchronize()
        print(f"  ✅ Forward completed: {y.shape}")
        
        print("\n✅ Step 4 PASSED: Simple model works")
        return True
    except Exception as e:
        print(f"\n❌ Step 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step_5_resnet():
    """Step 5: 测试 ResNet"""
    print("\n" + "="*70)
    print("Step 5: ResNet Model")
    print("="*70)
    
    try:
        import torchvision.models as models
        
        print("[5.1] Loading ResNet-18...")
        model = models.resnet18(pretrained=False).to('cuda:0').eval()
        print(f"  ✅ ResNet-18 loaded")
        
        print("[5.2] Forward pass (batch=1)...")
        x = torch.randn(1, 3, 224, 224, device='cuda:0')
        with torch.no_grad():
            y = model(x)
        torch.cuda.synchronize()
        print(f"  ✅ Forward completed: {y.shape}")
        
        print("[5.3] Forward pass (batch=8)...")
        x = torch.randn(8, 3, 224, 224, device='cuda:0')
        with torch.no_grad():
            y = model(x)
        torch.cuda.synchronize()
        print(f"  ✅ Forward completed: {y.shape}")
        
        print("\n✅ Step 5 PASSED: ResNet works")
        return True
    except Exception as e:
        print(f"\n❌ Step 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*70)
    print("XSched Debug - Progressive Testing")
    print("="*70)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"LD_PRELOAD: {os.environ.get('LD_PRELOAD', 'Not set')}")
    
    # Check XSched
    xsched_loaded = 'libshimhip.so' in os.environ.get('LD_PRELOAD', '')
    if xsched_loaded:
        print("✅ XSched is loaded")
    else:
        print("⚠️  XSched is NOT loaded (running baseline)")
    
    # Run tests
    results = {}
    
    results['step1'] = test_step_1_basic_tensor()
    if not results['step1']:
        print("\n⚠️  Stopping at Step 1")
        sys.exit(1)
    
    results['step2'] = test_step_2_matmul()
    if not results['step2']:
        print("\n⚠️  Stopping at Step 2")
        sys.exit(1)
    
    results['step3'] = test_step_3_conv2d()
    if not results['step3']:
        print("\n⚠️  Stopping at Step 3")
        sys.exit(1)
    
    results['step4'] = test_step_4_simple_model()
    if not results['step4']:
        print("\n⚠️  Stopping at Step 4")
        sys.exit(1)
    
    results['step5'] = test_step_5_resnet()
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    for step, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{step}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests PASSED!")
        return 0
    else:
        print("\n⚠️  Some tests FAILED")
        return 1

if __name__ == '__main__':
    sys.exit(main())
