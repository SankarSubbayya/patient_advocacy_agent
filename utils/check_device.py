#!/usr/bin/env python
"""
Check available compute devices (GPU/CPU).
"""

def check_devices():
    """Check what compute devices are available."""
    print("\n" + "="*80)
    print("Device Availability Check")
    print("="*80)
    
    try:
        import torch
    except ImportError:
        print("\n✗ PyTorch not installed")
        print("Install with: uv pip install torch")
        return
    
    print(f"\nPyTorch Version: {torch.__version__}")
    
    # Check CPU
    print("\n1. CPU")
    print("-" * 60)
    print(f"   ✓ CPU is always available")
    
    # Check CUDA (NVIDIA GPU)
    print("\n2. CUDA (NVIDIA GPU)")
    print("-" * 60)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"   ✓ CUDA is available")
        print(f"   Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"     Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print(f"   ✗ CUDA not available")
    
    # Check MPS (Apple Silicon)
    print("\n3. MPS (Apple Silicon)")
    print("-" * 60)
    try:
        mps_available = torch.backends.mps.is_available()
        if mps_available:
            print(f"   ✓ MPS is available (Apple M1/M2/M3)")
        else:
            print(f"   ✗ MPS not available")
    except AttributeError:
        print(f"   ✗ MPS not supported in this PyTorch version")
    
    # Determine which device will be used
    print("\n" + "="*80)
    print("Selected Device")
    print("="*80)
    
    if torch.backends.mps.is_available():
        selected = "mps"
        print(f"✓ Will use: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        selected = "cuda"
        print(f"✓ Will use: CUDA (NVIDIA GPU)")
    else:
        selected = "cpu"
        print(f"✓ Will use: CPU")
    
    print(f"\nDevice string: '{selected}'")
    
    # Test tensor creation
    print("\n" + "="*80)
    print("Device Test")
    print("="*80)
    
    try:
        device = torch.device(selected)
        test_tensor = torch.randn(100, 100).to(device)
        result = test_tensor @ test_tensor.T
        print(f"✓ Successfully created and computed tensor on {selected}")
        print(f"  Tensor shape: {result.shape}")
        print(f"  Tensor device: {result.device}")
    except Exception as e:
        print(f"✗ Failed to use {selected}: {e}")
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    if selected == "mps":
        print("✓ You're using Apple Silicon GPU acceleration!")
    elif selected == "cuda":
        print("✓ You're using NVIDIA GPU acceleration!")
    else:
        print("⚠ You're using CPU only (training will be slower)")
        print("\nTo enable GPU:")
        print("- For NVIDIA: Install CUDA toolkit and GPU-enabled PyTorch")
        print("- For Apple: Ensure you have PyTorch with MPS support")

if __name__ == "__main__":
    check_devices()


