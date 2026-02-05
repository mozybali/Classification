"""
DataLoader Factory - DataLoader Oluşturma ve Yönetim
Tekrarlayan DataLoader kodlarını merkezileştirir
"""

from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .preprocess import ALANDataset, resolve_csv_path
from .pipeline_builder import create_preprocessing_pipeline


class DataLoaderFactory:
    """DataLoader oluşturma factory'si"""
    
    @staticmethod
    def create_dataloaders(
        config: Dict,
        model_type: str = 'classifier',
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
        persistent_workers: bool = False
    ) -> Dict[str, DataLoader]:
        """
        Train, val, test DataLoader'ları oluştur
        
        Args:
            config: Full config dictionary
            model_type: Model tipi ('classifier', 'siamese', 'autoencoder')
            num_workers: Worker sayısı (None ise config'den alınır)
            pin_memory: Pin memory kullan (GPU için)
            persistent_workers: Worker'ları persistent tut
        
        Returns:
            Dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch gerekli!")
        
        # Config'den parametreleri al
        dataset_config = config['dataset']
        training_config = config['training']
        preprocessing_config = config.get('preprocessing', {})
        
        dataset_path = Path(dataset_config['path'])
        csv_path = resolve_csv_path(config)
        zip_path = dataset_path / dataset_config['zip_file']
        
        batch_size = training_config.get('batch_size', 32)
        if num_workers is None:
            num_workers = training_config.get('num_workers', 0)  # Windows: 0 as default
        
        # Pipeline'ları oluştur
        print("\n🔧 Preprocessing Pipeline Oluşturuluyor...")
        pipelines = create_preprocessing_pipeline(
            config, 
            model_type=model_type,
            verbose=False
        )
        
        # Dataset'leri oluştur
        print("\n📦 Dataset'ler Yükleniyor...")
        
        train_dataset = ALANDataset(
            csv_path=str(csv_path),
            zip_path=str(zip_path),
            subset='train',
            transform=pipelines['train'],
            load_images=preprocessing_config.get('load_images', True),
            cache_in_memory=preprocessing_config.get('cache_in_memory', False)
        )
        
        val_dataset = ALANDataset(
            csv_path=str(csv_path),
            zip_path=str(zip_path),
            subset='dev',  # dev = validation
            transform=pipelines['val'],
            load_images=preprocessing_config.get('load_images', True),
            cache_in_memory=preprocessing_config.get('cache_in_memory', False)
        )
        
        test_dataset = ALANDataset(
            csv_path=str(csv_path),
            zip_path=str(zip_path),
            subset='test',
            transform=pipelines['test'],
            load_images=preprocessing_config.get('load_images', True),
            cache_in_memory=preprocessing_config.get('cache_in_memory', False)
        )
        
        # DataLoader'ları oluştur
        print("\n🔄 DataLoader'lar Oluşturuluyor...")
        
        use_weighted_sampler = training_config.get('use_weighted_sampler', False)
        train_sampler = None
        train_shuffle = True

        if use_weighted_sampler:
            try:
                from torch.utils.data import WeightedRandomSampler
                labels = train_dataset.df['ROI_anomaly'].astype(int).values
                if len(np.unique(labels)) > 1:
                    total = len(labels)
                    count_0 = (labels == 0).sum()
                    count_1 = (labels == 1).sum()
                    w0 = total / (2 * count_0) if count_0 > 0 else 1.0
                    w1 = total / (2 * count_1) if count_1 > 0 else 1.0
                    sample_weights = np.array([w1 if l == 1 else w0 for l in labels], dtype=np.float64)
                    train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
                    train_shuffle = False
                    print("Weighted sampler aktif (train)")
                else:
                    print("Weighted sampler devre disi (tek sinif)")
            except Exception as e:
                print(f"Weighted sampler kurulamad?: {e}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            drop_last=True  # Son incomplete batch'i at
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            drop_last=False
        )
        
        # Özet
        print("\n✅ DataLoader'lar Hazır:")
        print(f"  • Train: {len(train_loader)} batch ({len(train_dataset)} örnek)")
        print(f"  • Val:   {len(val_loader)} batch ({len(val_dataset)} örnek)")
        print(f"  • Test:  {len(test_loader)} batch ({len(test_dataset)} örnek)")
        print(f"  • Batch size: {batch_size}")
        print(f"  • Workers: {num_workers}")
        print(f"  • Pin memory: {pin_memory}")
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    
    @staticmethod
    def create_single_dataloader(
        config: Dict,
        subset: str = 'train',
        model_type: str = 'classifier',
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        num_workers: Optional[int] = None,
        **kwargs
    ) -> DataLoader:
        """
        Tek bir DataLoader oluştur
        
        Args:
            config: Full config dictionary
            subset: 'train', 'val', 'dev', veya 'test'
            model_type: Model tipi
            batch_size: Batch size (None ise config'den)
            shuffle: Shuffle kullan (None ise train için True)
            num_workers: Worker sayısı
            **kwargs: DataLoader'a geçilecek ek parametreler
        
        Returns:
            DataLoader
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch gerekli!")
        
        # Config'den parametreleri al
        dataset_config = config['dataset']
        training_config = config['training']
        preprocessing_config = config.get('preprocessing', {})
        
        dataset_path = Path(dataset_config['path'])
        csv_path = resolve_csv_path(config)
        zip_path = dataset_path / dataset_config['zip_file']
        
        if batch_size is None:
            batch_size = training_config.get('batch_size', 32)
        
        if num_workers is None:
            num_workers = training_config.get('num_workers', 4)
        
        if shuffle is None:
            shuffle = (subset == 'train')
        
        # Pipeline oluştur
        pipelines = create_preprocessing_pipeline(
            config, 
            model_type=model_type,
            verbose=False
        )
        
        # Subset mapping
        subset_map = {'val': 'dev', 'validation': 'dev'}
        subset = subset_map.get(subset, subset)
        
        # Pipeline seç
        mode = 'train' if subset == 'train' else 'val'
        transform = pipelines[mode]
        
        # Dataset oluştur
        dataset = ALANDataset(
            csv_path=str(csv_path),
            zip_path=str(zip_path),
            subset=subset,
            transform=transform,
            load_images=preprocessing_config.get('load_images', True),
            cache_in_memory=preprocessing_config.get('cache_in_memory', False)
        )
        
        # DataLoader oluştur
        sampler = None
        if subset == 'train' and training_config.get('use_weighted_sampler', False):
            try:
                from torch.utils.data import WeightedRandomSampler
                labels = dataset.df['ROI_anomaly'].astype(int).values
                if len(np.unique(labels)) > 1:
                    total = len(labels)
                    count_0 = (labels == 0).sum()
                    count_1 = (labels == 1).sum()
                    w0 = total / (2 * count_0) if count_0 > 0 else 1.0
                    w1 = total / (2 * count_1) if count_1 > 0 else 1.0
                    sample_weights = np.array([w1 if l == 1 else w0 for l in labels], dtype=np.float64)
                    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
                    shuffle = False
                    print("Weighted sampler aktif (train)")
                else:
                    print("Weighted sampler devre disi (tek sinif)")
            except Exception as e:
                print(f"Weighted sampler kurulamad?: {e}")

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            **kwargs
        )
        
        print(f"\n✅ {subset.upper()} DataLoader Hazır:")
        print(f"  • Batch count: {len(loader)}")
        print(f"  • Sample count: {len(dataset)}")
        print(f"  • Batch size: {batch_size}")
        
        return loader
    
    @staticmethod
    def get_dataloader_stats(loader: DataLoader) -> Dict:
        """
        DataLoader istatistiklerini döndür
        
        Returns:
            Dict: İstatistikler
        """
        dataset = loader.dataset
        
        stats = {
            'num_batches': len(loader),
            'num_samples': len(dataset),
            'batch_size': loader.batch_size,
            'num_workers': loader.num_workers,
            'pin_memory': loader.pin_memory,
            'shuffle': hasattr(loader.sampler, 'shuffle') and loader.sampler.shuffle,
        }
        
        # Dataset stats (varsa)
        if hasattr(dataset, 'df'):
            df = dataset.df
            stats['num_normal'] = (~df['ROI_anomaly']).sum()
            stats['num_anomaly'] = df['ROI_anomaly'].sum()
            stats['anomaly_ratio'] = stats['num_anomaly'] / len(df)
        
        return stats


def get_dataloaders(config: Dict, 
                   model_type: str = 'classifier',
                   **kwargs) -> Dict[str, DataLoader]:
    """
    Convenience function - DataLoader'ları oluştur
    
    Args:
        config: Full config dictionary
        model_type: Model tipi
        **kwargs: DataLoaderFactory.create_dataloaders'a geçilecek parametreler
    
    Returns:
        Dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    
    Example:
        >>> from src.utils.helpers import load_config
        >>> from src.preprocessing.dataloader_factory import get_dataloaders
        >>> 
        >>> config = load_config('configs/config.yaml')
        >>> loaders = get_dataloaders(config, model_type='classifier')
        >>> 
        >>> for batch in loaders['train']:
        >>>     images = batch['image']  # [B, 1, 128, 128, 128]
        >>>     labels = batch['label']  # [B]
    """
    return DataLoaderFactory.create_dataloaders(config, model_type=model_type, **kwargs)


def create_dataloaders(config: Dict,
                       model_type: str = 'classifier',
                       **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Backward-compatible helper that returns (train, val, test) loaders.
    """
    loaders = DataLoaderFactory.create_dataloaders(config, model_type=model_type, **kwargs)
    return loaders['train'], loaders['val'], loaders['test']
