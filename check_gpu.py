#!/usr/bin/env python
"""GPU kontrol scripti"""

import torch
import os

print("="*70)
print("GPU KONTROL")
print("="*70)

print(f"\nPyTorch Sürümü: {torch.__version__}")
print(f"CUDA Sürümü (PyTorch): {torch.version.cuda}")
print(f"CUDA Kullanılabilir: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"\n✅ GPU BULUNDU!")
    print(f"GPU Sayısı: {torch.cuda.device_count()}")
    print(f"Aktif GPU: {torch.cuda.current_device()}")
    print(f"GPU Adı: {torch.cuda.get_device_name(0)}")
    
    # GPU Properties
    props = torch.cuda.get_device_properties(0)
    print(f"GPU Memory: {props.total_memory / 1e9:.2f} GB")
    print(f"CUDA Capability: {props.major}.{props.minor}")
else:
    print(f"\n❌ GPU BULUNAMADI!")
    print("\nSebepleri kontrol et:")
    print("  1. NVIDIA GPU sürücüsü yüklü mü? (nvidia-smi komutunu çalıştır)")
    print("  2. PyTorch CUDA sürümü yüklü mü?")
    print("  3. GPU ile CUDA sürümü uyumlu mu?")
    
    print("\n💡 Çözüm:")
    print("  - CPU-only PyTorch: pip install torch torchvision torchaudio")
    print("  - GPU PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("    (cu118 yerine CUDA sürümünüzü kullanın: cu121, cu124 vb.)")

print("\n" + "="*70)
