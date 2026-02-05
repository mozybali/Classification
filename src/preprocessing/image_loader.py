"""
Görüntü Yükleme ve Veri Seti Yönetimi
ZIP arşivinden 3D görüntü yükleme işlemleri
"""

import numpy as np
import zipfile
import io
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd


class ImageLoader:
    """ZIP arşivinden 3D görüntü yükleme"""
    
    def __init__(self, zip_path: str, cache_in_memory: bool = False):
        """
        Args:
            zip_path: ZIP dosyası yolu
            cache_in_memory: Görüntüleri bellekte önbelleğe al
        """
        self.zip_path = Path(zip_path)
        self.cache_in_memory = cache_in_memory
        self.cache = {} if cache_in_memory else None
        # Lazy-open zip to keep the object picklable for DataLoader workers.
        self._zip_file = None

        # Dosya listesini al (kisa sureli ac, sonra kapat)
        self.available_files = self._build_index()
        
        
        
        print(f"✓ ImageLoader başlatıldı: {len(self.available_files)} görüntü bulundu")
    

    def _build_index(self) -> Dict[str, str]:
        """Build index of .npy files inside the zip."""
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            return {
                Path(f).stem: f for f in zf.namelist()
                if f.endswith('.npy')
            }

    def _get_zip_file(self) -> zipfile.ZipFile:
        """Process-local lazy zip handle."""
        if self._zip_file is None:
            self._zip_file = zipfile.ZipFile(self.zip_path, 'r')
        return self._zip_file

    def load_image(self, roi_id: str) -> np.ndarray:
        """
        Tek bir görüntü yükle
        
        Args:
            roi_id: ROI kimliği (örn: ZS000_L)
        
        Returns:
            np.ndarray: 3D görüntü array'i
        """
        # Önbellekte var mı kontrol et
        if self.cache_in_memory and roi_id in self.cache:
            return self.cache[roi_id].copy()
        
        # Dosya yolunu bul
        if roi_id not in self.available_files:
            raise ValueError(f"ROI {roi_id} bulunamadı!")
        
        file_path = self.available_files[roi_id]
        
        with self._get_zip_file().open(file_path) as f:
            image = np.load(io.BytesIO(f.read()))
        
        # Önbelleğe al
        if self.cache_in_memory:
            self.cache[roi_id] = image.copy()
        
        return image
    
    def load_batch(self, roi_ids: list) -> np.ndarray:
        """
        Birden fazla görüntü yükle
        
        Args:
            roi_ids: ROI kimlik listesi
        
        Returns:
            np.ndarray: Batch halinde görüntüler [N, H, W, D]
        """
        images = [self.load_image(roi_id) for roi_id in roi_ids]
        return np.stack(images, axis=0)
    
    def get_image_info(self, roi_id: str) -> Dict:
        """Görüntü hakkında bilgi al"""
        image = self.load_image(roi_id)
        return {
            'roi_id': roi_id,
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min': float(image.min()),
            'max': float(image.max()),
            'mean': float(image.mean()),
            'std': float(image.std()),
            'non_zero_ratio': float((image != 0).sum() / image.size)
        }
    
    def preload_all(self):
        """Tüm görüntüleri belleğe yükle (dikkatli kullanın!)"""
        if not self.cache_in_memory:
            print("⚠️  Cache modu aktif değil!")
            return
        
        print(f"📦 {len(self.available_files)} görüntü yükleniyor...")
        for i, roi_id in enumerate(self.available_files.keys(), 1):
            self.load_image(roi_id)
            if i % 100 == 0:
                print(f"  {i}/{len(self.available_files)} yüklendi...")
        
        print(f"✓ Tüm görüntüler belleğe yüklendi!")
    
    def __del__(self):
        if hasattr(self, '_zip_file') and self._zip_file is not None:
            self._zip_file.close()

    def __getstate__(self):
        """Pickle support: drop zip handle and cache."""
        state = self.__dict__.copy()
        state['_zip_file'] = None
        if self.cache_in_memory:
            state['cache'] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._zip_file = None

    def __len__(self):
        return len(self.available_files)
    
    def __repr__(self):
        return f"ImageLoader(zip_path={self.zip_path}, num_images={len(self)})"


class DatasetStatistics:
    """Veri seti istatistikleri hesaplama"""
    
    def __init__(self, image_loader: ImageLoader, info_csv: str):
        """
        Args:
            image_loader: ImageLoader instance
            info_csv: info.csv dosyası yolu
        """
        self.image_loader = image_loader
        self.df = pd.read_csv(info_csv)
    
    def compute_statistics(self, sample_size: Optional[int] = None) -> Dict:
        """
        Veri seti istatistiklerini hesapla
        
        Args:
            sample_size: Örnekleme yapılacaksa kaç görüntü kullanılacak
        
        Returns:
            Dict: İstatistikler
        """
        roi_ids = self.df['ROI_id'].tolist()
        
        if sample_size and sample_size < len(roi_ids):
            roi_ids = np.random.choice(roi_ids, sample_size, replace=False).tolist()
            print(f"📊 {sample_size} görüntü üzerinden istatistik hesaplanıyor...")
        else:
            print(f"📊 Tüm {len(roi_ids)} görüntü üzerinden istatistik hesaplanıyor...")
        
        # İstatistikleri topla
        means = []
        stds = []
        non_zero_ratios = []
        
        for i, roi_id in enumerate(roi_ids, 1):
            try:
                image = self.image_loader.load_image(roi_id)
                means.append(image.mean())
                stds.append(image.std())
                non_zero_ratios.append((image != 0).sum() / image.size)
                
                if i % 100 == 0:
                    print(f"  {i}/{len(roi_ids)} işlendi...")
            except Exception as e:
                print(f"⚠️  {roi_id} yüklenemedi: {e}")
        
        stats = {
            'global_mean': float(np.mean(means)),
            'global_std': float(np.mean(stds)),
            'mean_non_zero_ratio': float(np.mean(non_zero_ratios)),
            'min_mean': float(np.min(means)),
            'max_mean': float(np.max(means)),
            'min_std': float(np.min(stds)),
            'max_std': float(np.max(stds)),
            'sample_size': len(roi_ids)
        }
        
        print("\n✓ İstatistikler hesaplandı:")
        for key, value in stats.items():
            print(f"  {key}: {value:.6f}")
        
        return stats
    
    def compute_class_statistics(self) -> Dict:
        """Sınıf bazında istatistikler"""
        stats = {}
        
        for label in [False, True]:
            label_name = "Anomaly" if label else "Normal"
            roi_ids = self.df[self.df['ROI_anomaly'] == label]['ROI_id'].tolist()
            
            print(f"\n📊 {label_name} sınıfı istatistikleri ({len(roi_ids)} görüntü):")
            
            means = []
            stds = []
            
            for roi_id in roi_ids[:min(100, len(roi_ids))]:  # İlk 100 örnek
                try:
                    image = self.image_loader.load_image(roi_id)
                    means.append(image.mean())
                    stds.append(image.std())
                except:
                    pass
            
            stats[label_name] = {
                'mean': float(np.mean(means)),
                'std': float(np.mean(stds)),
                'count': len(roi_ids)
            }
            
            print(f"  Mean: {stats[label_name]['mean']:.6f}")
            print(f"  Std: {stats[label_name]['std']:.6f}")
        
        return stats
    
    def analyze_image_quality(self, roi_id: str) -> Dict:
        """Tek bir görüntünün kalite analizini yap"""
        image = self.image_loader.load_image(roi_id)
        
        # Temel metrikler
        metrics = {
            'roi_id': roi_id,
            'shape': image.shape,
            'dtype': str(image.dtype),
            'total_voxels': int(image.size),
            'non_zero_voxels': int((image != 0).sum()),
            'zero_voxels': int((image == 0).sum()),
            'non_zero_ratio': float((image != 0).sum() / image.size),
            'mean': float(image.mean()),
            'std': float(image.std()),
            'min': float(image.min()),
            'max': float(image.max()),
        }
        
        # Bağlantılı bileşen analizi (3D)
        if image.dtype == bool or np.array_equal(image, image.astype(bool)):
            from scipy import ndimage
            labeled, num_components = ndimage.label(image)
            metrics['num_connected_components'] = int(num_components)
            
            if num_components > 0:
                # En büyük bileşen boyutu
                component_sizes = [
                    (labeled == i).sum() 
                    for i in range(1, num_components + 1)
                ]
                metrics['largest_component_size'] = int(max(component_sizes))
                metrics['largest_component_ratio'] = float(
                    max(component_sizes) / metrics['non_zero_voxels']
                    if metrics['non_zero_voxels'] > 0 else 0
                )
        
        return metrics


def save_statistics_report(stats: Dict, output_path: str):
    """İstatistikleri dosyaya kaydet"""
    import json
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ İstatistikler kaydedildi: {output_path}")
