import torch
import torch.nn as nn
import psutil
import GPUtil
import time
import os

def check_pytorch_hardware():
    print("=== PyTorch Hardware Diagnostic ===\n")
    
    # Check PyTorch version and CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
        # Check current GPU
        current_device = torch.cuda.current_device()
        print(f"Current GPU: {current_device}")
        
        # Check GPU memory usage
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU memory allocated: {allocated:.2f} GB")
        print(f"GPU memory cached: {cached:.2f} GB")
        
    else:
        print("CUDA is not available. PyTorch will use CPU.")
    
    # Check system memory
    print(f"\n=== System Memory ===")
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / 1024**3:.1f} GB")
    print(f"Available RAM: {memory.available / 1024**3:.1f} GB")
    print(f"RAM usage: {memory.percent:.1f}%")
    
    # Check if GPUtil is available for more detailed GPU info
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        print(f"\n=== Detailed GPU Information ===")
        for gpu in gpus:
            print(f"GPU {gpu.id}: {gpu.name}")
            print(f"  Memory: {gpu.memoryTotal} MB total, {gpu.memoryUsed} MB used, {gpu.memoryFree} MB free")
            print(f"  GPU utilization: {gpu.load*100:.1f}%")
            print(f"  Temperature: {gpu.temperature}°C")
    except ImportError:
        print("\nGPUtil not available. Install with: pip install GPUtil")
    
    return torch.cuda.is_available()

def test_gpu_performance():
    """Test GPU performance with a simple tensor operation"""
    print(f"\n=== GPU Performance Test ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test")
        return
    
    # Create a large tensor on GPU
    device = torch.device('cuda')
    size = 1000
    
    print(f"Creating {size}x{size} tensor on GPU...")
    start_time = time.time()
    x = torch.randn(size, size, device=device)
    y = torch.randn(size, size, device=device)
    creation_time = time.time() - start_time
    print(f"Tensor creation time: {creation_time:.4f} seconds")
    
    # Test matrix multiplication
    print("Performing matrix multiplication...")
    start_time = time.time()
    z = torch.mm(x, y)
    torch.cuda.synchronize()  # Wait for GPU to finish
    computation_time = time.time() - start_time
    print(f"Matrix multiplication time: {computation_time:.4f} seconds")
    
    # Check memory usage after operation
    allocated = torch.cuda.memory_allocated() / 1024**3
    cached = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU memory after test: {allocated:.2f} GB allocated, {cached:.2f} GB cached")
    
    # Clear GPU memory
    del x, y, z
    torch.cuda.empty_cache()
    print("GPU memory cleared")

def check_training_script_optimization():
    """Check if the training script can be optimized"""
    print(f"\n=== Training Script Optimization Suggestions ===")
    
    # Check if cuDNN is enabled
    if torch.cuda.is_available():
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        print(f"cuDNN benchmark mode: {torch.backends.cudnn.benchmark}")
        
        # Suggest optimizations
        if not torch.backends.cudnn.benchmark:
            print("Suggestion: Enable cuDNN benchmark mode for faster training")
            print("  Add: torch.backends.cudnn.benchmark = True")
        
        if not torch.backends.cudnn.deterministic:
            print("Suggestion: Consider deterministic mode for reproducible results")
            print("  Add: torch.backends.cudnn.deterministic = True")
    
    # Check batch size optimization
    print("\nBatch size optimization:")
    print("- Current batch size: 32")
    print("- Try increasing batch size to utilize more GPU memory")
    print("- Monitor GPU memory usage and adjust accordingly")
    
    # Check data loading optimization
    print("\nData loading optimization:")
    print("- Consider using num_workers > 0 in DataLoader")
    print("- Use pin_memory=True for faster GPU transfer")
    print("- Example: DataLoader(..., num_workers=4, pin_memory=True)")

if __name__ == "__main__":
    cuda_available = check_pytorch_hardware()
    test_gpu_performance()
    check_training_script_optimization()
    
    print(f"\n=== Summary ===")
    if cuda_available:
        print("✅ PyTorch is using GPU")
        print("💡 Consider the optimization suggestions above")
    else:
        print("❌ PyTorch is not using GPU")
        print("💡 Install CUDA and cuDNN to enable GPU acceleration") 