"""
Veri Önişleme Modülü
NeAR Dataset için veri yükleme, normalizasyon ve augmentation işlemleri
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import zipfile

# PyTorch imports (optional)
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch bulunamadı. Dataset sınıfı kullanılamayacak.")

# Kendi modüllerimiz
from .image_loader import ImageLoader
from .data_splitter import DataSplitter
from .class_balancer import ClassBalancer
from .augmentation_manager import AugmentationManager

def resolve_csv_path(config: Dict) -> Path:
    """Resolve CSV path, optionally using a processed dataset."""
    dataset_cfg = config.get('dataset', {})
    dataset_path = Path(dataset_cfg.get('path', ''))
    csv_file = dataset_cfg.get('csv_file', 'info.csv')

    if dataset_cfg.get('use_processed', False):
        processed_csv = dataset_cfg.get('processed_csv')
        if processed_csv:
            proc_path = Path(processed_csv)
            if not proc_path.is_absolute():
                # First try project root relative
                proc_path = Path(processed_csv)
                if not proc_path.exists():
                    # Fallback: relative to dataset path
                    proc_path = dataset_path / processed_csv
            if proc_path.exists():
                print(f"✅ Processed CSV kullanılıyor: {proc_path}")
                return proc_path
            else:
                print(f"⚠️  Processed CSV bulunamadı, orijinale dönülüyor: {processed_csv}")

    return dataset_path / csv_file


class ALANDataset(Dataset if TORCH_AVAILABLE else object):
    """ALAN dataset için PyTorch Dataset sınıfı"""
    
    def __init__(self, 
                 csv_path: str,
                 zip_path: str,
                 subset: str = 'train',
                 transform=None,
                 load_images: bool = True,
                 cache_in_memory: bool = False):
        """
        Args:
            csv_path: info.csv dosyasının yolu
            zip_path: ALAN.zip dosyasının yolu
            subset: 'train', 'test', veya 'dev'
            transform: Veri augmentation transformları
            load_images: Görüntüleri yükle (False ise sadece metadata)
            cache_in_memory: Görüntüleri bellekte önbelleğe al
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch gerekli! pip install torch")
        
        self.csv_path = Path(csv_path)
        self.zip_path = Path(zip_path)
        self.subset = self._map_subset(subset)
        self.transform = transform
        self.load_images = load_images
        
        # CSV'yi yükle ve filtrele
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['subset'] == self.subset].reset_index(drop=True)
        
        # Görüntü yükleyici
        if self.load_images:
            self.image_loader = ImageLoader(str(zip_path), cache_in_memory=cache_in_memory)
        else:
            self.image_loader = None
        
        print(f"✓ {subset} seti yüklendi: {len(self.df)} örnek")
        print(f"  - Normal: {(~self.df['ROI_anomaly']).sum()}")
        print(f"  - Anomali: {self.df['ROI_anomaly'].sum()}")
        if self.load_images:
            print(f"  - Görüntü yükleme: Aktif")
            if cache_in_memory:
                print(f"  - Önbellek: Aktif (bellek kullanımı yüksek olabilir)")
    
    def _map_subset(self, subset: str) -> str:
        """
        Subset ismini CSV formatına çevirir
        
        NeAR Dataset'te subset isimleri CSV'de 'ZS-train', 'ZS-test', 'ZS-dev' 
        formatında saklanıyor. Bu metod kullanıcı dostu 'train', 'test', 'dev' 
        isimlerini CSV formatına çevirir.
        
        Args:
            subset: Kullanıcı tarafından verilen subset ismi ('train', 'test', 'dev', 'val')
            
        Returns:
            CSV formatında subset ismi ('ZS-train', 'ZS-test', 'ZS-dev')
            
        Example:
            >>> _map_subset('train')
            'ZS-train'
            >>> _map_subset('val')  # val, dev için alias
            'ZS-dev'
        """
        mapping = {
            'train': 'ZS-train',
            'test': 'ZS-test',
            'dev': 'ZS-dev',
            'val': 'ZS-dev'  # Alias: 'val' ve 'dev' aynı subset'e map olur
        }
        return mapping.get(subset, subset)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Tek bir örnek döndürür
        
        Returns:
            Dict: {
                'roi_id': ROI kimliği,
                'label': Anomali etiketi (0=Normal, 1=Anomali),
                'image': Görüntü tensörü (torch.Tensor, shape: [1, 128, 128, 128]),
                'laterality': 'L' veya 'R'
            }
        """
        row = self.df.iloc[idx]
        
        sample = {
            'roi_id': row['ROI_id'],
            'label': int(row['ROI_anomaly']),  # TRUE->1, FALSE->0
            'laterality': row['ROI_id'][-1],  # L veya R
            'subset': row['subset']
        }
        
        # Görüntü yükle
        if self.load_images and self.image_loader is not None:
            try:
                image = self.image_loader.load_image(row['ROI_id'])
                
                # Transform uygula
                if self.transform:
                    image = self.transform(image)
                else:
                    # En azından float'a çevir
                    image = image.astype(np.float32)
                
                # Kanal boyutu ekle [H,W,D] -> [1,H,W,D]
                if image.ndim == 3:
                    image = np.expand_dims(image, axis=0)
                
                # Torch tensor'a çevir
                sample['image'] = torch.from_numpy(image).float()
                
            except Exception as e:
                print(f"⚠️  Görüntü yüklenemedi ({row['ROI_id']}): {e}")
                # Boş tensor döndür
                sample['image'] = torch.zeros(1, 128, 128, 128, dtype=torch.float32)
        
        return sample


class DataPreprocessor:
    """Veri önişleme pipeline'ı"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Konfigürasyon dictionary'si
        """
        self.config = config
        self.dataset_path = Path(config['dataset']['path'])
        self.csv_path = resolve_csv_path(config)
        self.zip_path = self.dataset_path / config['dataset']['zip_file']
    
    def load_data(self) -> pd.DataFrame:
        """CSV dosyasını yükler"""
        df = pd.read_csv(self.csv_path)
        print(f"✓ Dataset yüklendi: {len(df)} ROI")
        return df
    
    def analyze_class_imbalance(self, df: pd.DataFrame) -> Dict:
        """Sınıf dengesizliğini analiz eder"""
        total = len(df)
        anomaly_count = df['ROI_anomaly'].sum()
        normal_count = total - anomaly_count
        
        imbalance_ratio = normal_count / anomaly_count if anomaly_count > 0 else float('inf')
        
        analysis = {
            'total_samples': total,
            'normal_count': normal_count,
            'anomaly_count': anomaly_count,
            'imbalance_ratio': imbalance_ratio,
            'class_weights': {
                0: total / (2 * normal_count),  # Normal için weight
                1: total / (2 * anomaly_count)   # Anomali için weight
            }
        }
        
        print(f"\n⚖️  SINIF DENGESİZLİĞİ ANALİZİ:")
        print(f"  • Normal: {normal_count} (%{normal_count/total*100:.1f})")
        print(f"  • Anomali: {anomaly_count} (%{anomaly_count/total*100:.1f})")
        print(f"  • Dengesizlik Oranı: 1:{imbalance_ratio:.2f}")
        print(f"  • Önerilen Class Weights:")
        print(f"    - Normal (0): {analysis['class_weights'][0]:.3f}")
        print(f"    - Anomali (1): {analysis['class_weights'][1]:.3f}")
        
        return analysis
    
    def extract_zip_if_needed(self):
        """ZIP dosyalarını gerekirse extract eder"""
        zip_file = self.dataset_path / 'ALAN.zip'
        
        if zip_file.exists():
            extract_dir = self.dataset_path / 'ALAN_extracted'
            if not extract_dir.exists():
                print(f"📦 ZIP dosyası extract ediliyor...")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                print(f"✓ Extract tamamlandı: {extract_dir}")
            else:
                print(f"✓ Veriler zaten extract edilmiş: {extract_dir}")
            return extract_dir
        else:
            print(f"⚠️  ALAN.zip bulunamadı: {zip_file}")
            return None
    
    def _normalize_subset_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize subset values to expected ZS-* names when needed."""
        if 'subset' not in df.columns:
            return df
        mapping = {
            'train': 'ZS-train',
            'val': 'ZS-dev',
            'dev': 'ZS-dev',
            'test': 'ZS-test'
        }
        df = df.copy()
        df['subset'] = df['subset'].astype(str).map(lambda x: mapping.get(x, x))
        return df

    def _get_subset_labels(self, df: pd.DataFrame) -> Tuple[str, str, str]:
        """Return (train_label, val_label, test_label) based on subset values."""
        values = set(df['subset'].astype(str).unique()) if 'subset' in df.columns else set()
        train_label = 'ZS-train' if 'ZS-train' in values else 'train'
        val_label = 'ZS-dev' if 'ZS-dev' in values else ('dev' if 'dev' in values else 'val')
        test_label = 'ZS-test' if 'ZS-test' in values else 'test'
        return train_label, val_label, test_label

    def _apply_data_splitting(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        split_cfg = self.config.get('dataset', {}).get('data_splitting', {})
        if not split_cfg.get('enabled', False):
            return df, False

        method = str(split_cfg.get('method', 'existing')).lower()
        splitter = DataSplitter(random_state=split_cfg.get('random_state', 42), verbose=True)

        if method in ['existing', 'existing_column', 'existing_split']:
            col = split_cfg.get('existing_split_column', 'subset')
            if col != 'subset' and col in df.columns:
                df = df.copy()
                df['subset'] = df[col]
            df = self._normalize_subset_values(df)
            return df, False

        if method == 'simple':
            splits = splitter.split_simple(
                df,
                train_ratio=split_cfg.get('train_ratio', 0.7),
                val_ratio=split_cfg.get('val_ratio', 0.15),
                test_ratio=split_cfg.get('test_ratio', 0.15)
            )
        elif method == 'stratified':
            splits = splitter.split_stratified(
                df,
                stratify_column=split_cfg.get('stratify_column', 'ROI_anomaly'),
                train_ratio=split_cfg.get('train_ratio', 0.7),
                val_ratio=split_cfg.get('val_ratio', 0.15),
                test_ratio=split_cfg.get('test_ratio', 0.15)
            )
        elif method in ['patient', 'patient_level']:
            splits = splitter.split_by_patient(
                df,
                patient_id_column=split_cfg.get('patient_id_column', 'ROI_id'),
                train_ratio=split_cfg.get('train_ratio', 0.7),
                val_ratio=split_cfg.get('val_ratio', 0.15),
                test_ratio=split_cfg.get('test_ratio', 0.15),
                stratify_column=split_cfg.get('stratify_column')
            )
        else:
            print(f"Unknown split method '{method}', skipping split.")
            return df, False

        subset_map = {'train': 'ZS-train', 'val': 'ZS-dev', 'test': 'ZS-test'}
        for split_name, split_df in splits.items():
            split_df = split_df.copy()
            split_df['subset'] = subset_map.get(split_name, split_name)
            splits[split_name] = split_df

        combined = pd.concat([splits['train'], splits['val'], splits['test']], ignore_index=True)

        if split_cfg.get('save_splits', False):
            output_dir = split_cfg.get('splits_output_dir', 'outputs/splits')
            splitter.save_splits(splits, output_dir)

        return combined, True

    def _apply_class_balancing(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        bal_cfg = self.config.get('preprocessing', {}).get('class_balancing', {})
        if not bal_cfg.get('enabled', False):
            return df, False

        method = str(bal_cfg.get('method', 'none')).lower()
        if method in ['none', 'null', 'false', '']:
            return df, False

        label_column = bal_cfg.get('label_column', 'ROI_anomaly')
        if label_column not in df.columns:
            print(f"Class balancing skipped: missing label column '{label_column}'.")
            return df, False

        train_label, _, _ = self._get_subset_labels(df)
        train_df = df[df['subset'] == train_label].copy()
        if train_df.empty:
            print("Class balancing skipped: train subset not found.")
            return df, False

        balancer = ClassBalancer(verbose=True)
        if method in ['oversample', 'over', 'random_oversample']:
            strategy = bal_cfg.get('target_ratio', 'auto')
            if isinstance(strategy, (int, float)) and strategy <= 0:
                strategy = 'auto'
            train_bal = balancer.random_oversample(train_df, label_column, strategy=strategy)
        elif method in ['undersample', 'under', 'random_undersample']:
            train_bal = balancer.random_undersample(train_df, label_column)
        elif method == 'smote':
            k = int(bal_cfg.get('smote_k_neighbors', 5))
            train_bal = balancer.smote_balance(train_df, label_column, k_neighbors=k)
        elif method == 'adasyn':
            train_bal = balancer.adasyn_balance(train_df, label_column)
        elif method in ['smote_tomek', 'smote_enn']:
            train_bal = balancer.combined_sampling(train_df, label_column, method=method)
        else:
            print(f"Unknown balancing method '{method}', skipping.")
            return df, False

        train_bal['subset'] = train_label
        rest_df = df[df['subset'] != train_label]
        combined = pd.concat([train_bal, rest_df], ignore_index=True)

        if bal_cfg.get('save_balanced_data', False):
            output_path = Path(bal_cfg.get('balanced_data_path', 'outputs/balanced_data.csv'))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(output_path, index=False)

        return combined, True

    def _apply_augmentation_strategy(self, df: pd.DataFrame, imbalance_info: Dict) -> bool:
        strat_cfg = self.config.get('preprocessing', {}).get('augmentation_strategy', {})
        if not strat_cfg:
            return False

        auto_adjust = strat_cfg.get('auto_adjust', False)
        level = strat_cfg.get('level')
        manager = AugmentationManager(verbose=True)

        if auto_adjust:
            level = manager.recommend_augmentation_level(
                dataset_size=len(df),
                imbalance_ratio=imbalance_info.get('imbalance_ratio', 1.0)
            )
            print(f"Auto augmentation level selected: {level}")

        if not level:
            return False

        preset = manager.get_preset_config(level)
        if not preset:
            return False

        pre_cfg = self.config.setdefault('preprocessing', {})
        aug_cfg = pre_cfg.get('augmentation', {})
        aug_cfg.update(preset)
        aug_cfg['enabled'] = True
        pre_cfg['augmentation'] = aug_cfg

        # Sync medical knobs if provided by preset
        if 'adaptive_crop' in preset or 'mask_processing' in preset:
            med_cfg = pre_cfg.get('medical', {})
            if 'adaptive_crop' in preset:
                med_cfg['adaptive_crop'] = preset['adaptive_crop']
            if 'mask_processing' in preset:
                med_cfg['mask_processing'] = preset['mask_processing']
            pre_cfg['medical'] = med_cfg

        return True

    def _persist_processed_csv(self, df: pd.DataFrame) -> Path:
        dataset_cfg = self.config.get('dataset', {})
        processed_csv = dataset_cfg.get('processed_csv', 'outputs/processed_data.csv')
        output_path = Path(processed_csv)
        if not output_path.is_absolute():
            output_path = Path(processed_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        dataset_cfg['use_processed'] = True
        dataset_cfg['processed_csv'] = str(output_path)
        self.config['dataset'] = dataset_cfg
        self.csv_path = output_path

        print(f"Processed CSV saved: {output_path}")
        return output_path

    def get_dataloaders(self, 
                       batch_size: int = 32,
                       num_workers: int = 4) -> Dict[str, DataLoader]:
        """PyTorch DataLoader'lar olusturur (config-driven pipeline)"""
        from copy import deepcopy
        from .dataloader_factory import DataLoaderFactory

        cfg = deepcopy(self.config)
        cfg.setdefault('training', {})
        cfg['training']['batch_size'] = batch_size
        cfg['training']['num_workers'] = num_workers

        loaders = DataLoaderFactory.create_dataloaders(
            cfg,
            model_type='classifier',
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False
        )

        # Backward-compat: 'dev' and 'val' aliases
        if 'val' in loaders and 'dev' not in loaders:
            loaders['dev'] = loaders['val']
        if 'dev' in loaders and 'val' not in loaders:
            loaders['val'] = loaders['dev']

        return loaders

    def create_dataloaders(self,
                          batch_size: int = 32,
                          num_workers: int = 4):
        """Backward-compatible helper that returns (train, val, test) loaders."""
        loaders = self.get_dataloaders(batch_size=batch_size, num_workers=num_workers)
        return loaders['train'], loaders['dev'], loaders['test']

    def prepare_for_training(self) -> Dict:
        """Egitim icin tum hazirliklari yapar"""
        print()
        print("="*60)
        print("VERI ONISLEME BASLADI")
        print("="*60)

        # 1. Veriyi yukle
        df = self.load_data()
        df = self._normalize_subset_values(df)

        # 2. Opsiyonel data splitting uygula
        df, split_modified = self._apply_data_splitting(df)

        # 3. Sinif dengesizligini analiz et (train subset uzerinden)
        train_label, _, _ = self._get_subset_labels(df)
        imbalance_df = df[df['subset'] == train_label] if 'subset' in df.columns else df
        if imbalance_df.empty:
            imbalance_df = df
        imbalance_info = self.analyze_class_imbalance(imbalance_df)

        # 4. Opsiyonel class balancing uygula (train subset)
        df, balance_modified = self._apply_class_balancing(df)
        modified = split_modified or balance_modified

        # 5. Augmentation stratejisini uygula (config guncelle)
        self._apply_augmentation_strategy(df, imbalance_info)

        # 6. Islenmis CSV'yi kaydet (gerekirse) ve path'i guncelle
        if modified or self.config.get('dataset', {}).get('use_processed', False):
            self._persist_processed_csv(df)

        # 7. ZIP dosyasini extract et
        extract_dir = self.extract_zip_if_needed()

        # 8. DataLoader'lari olustur
        print()
        print("DataLoader'lar olusturuluyor...")
        training_cfg = self.config.get('training', {})
        batch_size = training_cfg.get('batch_size', self.config.get('batch_size', 32))
        num_workers = training_cfg.get('num_workers', self.config.get('num_workers', 4))
        dataloaders = self.get_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers
        )

        print()
        print("VERI ONISLEME TAMAMLANDI")
        print("="*60)
        print()

        return {
            'dataloaders': dataloaders,
            'class_weights': imbalance_info['class_weights'],
            'extract_dir': extract_dir,
            'dataframe': df
        }

def main():
    """Ana çalıştırma fonksiyonu"""
    config = {
        'dataset_path': 'NeAR_dataset/ALAN',
        'batch_size': 32,
        'num_workers': 4
    }
    
    preprocessor = DataPreprocessor(config)
    results = preprocessor.prepare_for_training()
    
    # Test et
    print("\n🧪 DataLoader Test:")
    train_loader = results['dataloaders']['train']
    batch = next(iter(train_loader))
    print(f"  Batch keys: {batch.keys()}")
    print(f"  Batch size: {len(batch['roi_id'])}")
    print(f"  Label distribution: {batch['label'].sum().item()} anomali")


if __name__ == "__main__":
    main()
