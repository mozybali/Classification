"""Görüntü dosyalarını incele"""
import zipfile
import numpy as np
import io

# ZIP dosyasını aç
z = zipfile.ZipFile('NeAR_dataset/ALAN/ALAN.zip')

# .npy dosyalarını bul
npy_files = [f for f in z.namelist() if f.endswith('.npy')]

print(f"Toplam .npy dosya: {len(npy_files)}")
print(f"\nİlk 5 dosya incelemeleri:")

# İlk 5 dosyayı incele
for i, file in enumerate(npy_files[:5], 1):
    data = np.load(io.BytesIO(z.read(file)))
    print(f"\n{i}. {file}:")
    print(f"   Shape: {data.shape}")
    print(f"   Dtype: {data.dtype}")
    print(f"   Min: {data.min():.4f}")
    print(f"   Max: {data.max():.4f}")
    print(f"   Mean: {data.mean():.4f}")
    print(f"   Std: {data.std():.4f}")
    print(f"   Unique values: {len(np.unique(data))}")

z.close()
