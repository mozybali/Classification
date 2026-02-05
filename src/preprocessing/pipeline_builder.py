"""
Pipeline Builder - Config-driven Transform Pipeline Oluşturma
Transform'ları config dosyasından otomatik olarak oluşturur
"""

from typing import Dict, List, Optional
from .image_transforms import (
    Compose, ToFloat, Normalize, AddChannel,
    RandomFlip3D, RandomRotation3D, RandomShift3D,
    RandomZoom3D, ElasticDeformation, RandomNoise,
    PadOrCrop3D
)
from .medical_transforms import (
    MedicalIntensityNormalization,
    AdaptiveROICrop,
    ResampleToSpacing,
    BinaryMaskProcessor
)


class PipelineBuilder:
    """Config'den transform pipeline'ı oluşturur"""
    
    # Transform registry
    TRANSFORM_REGISTRY = {
        'to_float': ToFloat,
        'normalize': Normalize,
        'add_channel': AddChannel,
        'random_flip': RandomFlip3D,
        'random_rotation': RandomRotation3D,
        'random_shift': RandomShift3D,
        'random_zoom': RandomZoom3D,
        'elastic_deformation': ElasticDeformation,
        'random_noise': RandomNoise
    }
    
    @classmethod
    def from_config(cls, config: Dict, mode: str = 'train') -> Compose:
        """
        Config'den transform pipeline oluştur
        
        Args:
            config: Preprocessing config dictionary
            mode: 'train', 'val', 'test' veya 'inference'
        
        Returns:
            Compose: Transform pipeline
        
        Example config:
            preprocessing:
              normalize: true
              mean: 0.002
              std: 0.045
              augmentation:
                enabled: true
                random_flip:
                  enabled: true
                  p: 0.5
                  axes: [0, 1, 2]
        """
        transforms = []
        
        # Temel dönüşümler (her zaman)
        transforms.append(ToFloat())

        # Medical-specific preprocessing (optional)
        medical_cfg = config.get('medical', {})
        if medical_cfg:
            intensity_cfg = medical_cfg.get('intensity_normalization', {})
            if intensity_cfg.get('enabled', False):
                transforms.append(
                    MedicalIntensityNormalization(
                        method=intensity_cfg.get('method', 'minmax'),
                        percentile_range=tuple(intensity_cfg.get('percentile_range', [1.0, 99.0])),
                        clip_output=bool(intensity_cfg.get('clip_output', True))
                    )
                )

            spacing_cfg = medical_cfg.get('spacing_normalization', {})
            if spacing_cfg.get('enabled', False):
                transforms.append(
                    ResampleToSpacing(
                        target_spacing=tuple(spacing_cfg.get('target_spacing', [1.0, 1.0, 1.0])),
                        interpolation_order=int(spacing_cfg.get('interpolation_order', 1))
                    )
                )

            crop_cfg = medical_cfg.get('adaptive_crop', {})
            if crop_cfg.get('enabled', False):
                transforms.append(
                    AdaptiveROICrop(
                        margin=crop_cfg.get('margin', 10),
                        min_size=crop_cfg.get('min_size', 64),
                        center_crop=bool(crop_cfg.get('center_crop', False))
                    )
                )

            mask_cfg = medical_cfg.get('mask_processing', {})
            if mask_cfg.get('enabled', False):
                transforms.append(
                    BinaryMaskProcessor(
                        fill_holes=bool(mask_cfg.get('fill_holes', True)),
                        min_component_size=int(mask_cfg.get('min_component_size', 100)),
                        morphology=mask_cfg.get('morphology', 'closing'),
                        structure_size=int(mask_cfg.get('structure_size', 2))
                    )
                )

        # Ensure fixed volume size if target_size provided
        target_size = config.get('target_size')
        if target_size:
            transforms.append(PadOrCrop3D(target_shape=tuple(target_size)))
        
        # Normalizasyon
        if config.get('normalize', False):
            mean = config.get('mean', 0.0)
            std = config.get('std', 1.0)
            transforms.append(Normalize(mean=mean, std=std))
        
        # Augmentasyon (sadece train için)
        if mode == 'train' and config.get('augmentation', {}).get('enabled', False):
            aug_config = config['augmentation']
            
            # Her transform'u config'den ekle
            if aug_config.get('random_flip', {}).get('enabled', False):
                flip_cfg = aug_config['random_flip']
                transforms.append(RandomFlip3D(
                    p=flip_cfg.get('p', 0.5),
                    axes=flip_cfg.get('axes', [0, 1, 2])
                ))
            
            if aug_config.get('random_rotation', {}).get('enabled', False):
                rot_cfg = aug_config['random_rotation']
                transforms.append(RandomRotation3D(
                    p=rot_cfg.get('p', 0.5),
                    angle_range=tuple(rot_cfg.get('angle_range', [-15, 15])),
                    axes=tuple(rot_cfg.get('axes', [0, 1]))
                ))
            
            if aug_config.get('random_shift', {}).get('enabled', False):
                shift_cfg = aug_config['random_shift']
                transforms.append(RandomShift3D(
                    p=shift_cfg.get('p', 0.5),
                    max_shift=shift_cfg.get('max_shift', 10)
                ))
            
            if aug_config.get('random_zoom', {}).get('enabled', False):
                zoom_cfg = aug_config['random_zoom']
                transforms.append(RandomZoom3D(
                    p=zoom_cfg.get('p', 0.3),
                    zoom_range=tuple(zoom_cfg.get('zoom_range', [0.9, 1.1]))
                ))
            
            if aug_config.get('elastic_deformation', {}).get('enabled', False):
                elastic_cfg = aug_config['elastic_deformation']
                transforms.append(ElasticDeformation(
                    p=elastic_cfg.get('p', 0.3),
                    alpha=elastic_cfg.get('alpha', 10),
                    sigma=elastic_cfg.get('sigma', 4)
                ))
            
            if aug_config.get('random_noise', {}).get('enabled', False):
                noise_cfg = aug_config['random_noise']
                transforms.append(RandomNoise(
                    p=noise_cfg.get('p', 0.2),
                    noise_std=noise_cfg.get('noise_std', 0.01)
                ))
        
        return Compose(transforms)
    
    @classmethod
    def build_train_pipeline(cls, config: Dict) -> Compose:
        """Eğitim pipeline'ı oluştur"""
        return cls.from_config(config, mode='train')
    
    @classmethod
    def build_val_pipeline(cls, config: Dict) -> Compose:
        """Validasyon pipeline'ı oluştur"""
        return cls.from_config(config, mode='val')
    
    @classmethod
    def build_test_pipeline(cls, config: Dict) -> Compose:
        """Test pipeline'ı oluştur"""
        return cls.from_config(config, mode='test')
    
    @classmethod
    def build_inference_pipeline(cls, config: Dict) -> Compose:
        """Inference pipeline'ı oluştur"""
        return cls.from_config(config, mode='inference')
    
    @classmethod
    def validate_config(cls, config: Dict) -> bool:
        """
        Config'in geçerli olup olmadığını kontrol et
        
        Returns:
            bool: Config geçerli mi?
        """
        required_keys = ['normalize']
        for key in required_keys:
            if key not in config:
                print(f"⚠️  Eksik config anahtarı: {key}")
                return False
        
        # Normalizasyon parametreleri
        if config.get('normalize', False):
            if 'mean' not in config or 'std' not in config:
                print("⚠️  Normalizasyon aktif ama mean/std eksik!")
                return False
        
        # Augmentasyon kontrolü
        if 'augmentation' in config:
            aug = config['augmentation']
            if aug.get('enabled', False):
                # En az bir transform enabled olmalı
                has_transform = any([
                    aug.get(t, {}).get('enabled', False)
                    for t in ['random_flip', 'random_rotation', 'random_shift',
                             'random_zoom', 'elastic_deformation', 'random_noise']
                ])
                if not has_transform:
                    print("⚠️  Augmentasyon aktif ama hiçbir transform enabled değil!")
                    return False
        
        return True
    
    @classmethod
    def print_pipeline_summary(cls, pipeline: Compose, name: str = "Pipeline"):
        """Pipeline özetini yazdır"""
        print(f"\n{'='*70}")
        print(f"{name} Özeti")
        print(f"{'='*70}")
        print(f"Toplam Transform Sayısı: {len(pipeline.transforms)}")
        print("\nTransform'lar:")
        for i, transform in enumerate(pipeline.transforms, 1):
            print(f"  {i}. {transform}")
        print(f"{'='*70}\n")


class PreprocessingStrategy:
    """Farklı model tipleri için preprocessing stratejileri"""
    
    @staticmethod
    def get_strategy(model_type: str, config: Dict) -> Dict[str, Compose]:
        """
        Model tipine göre preprocessing stratejisi döndür
        
        Args:
            model_type: 'classifier', 'siamese', 'autoencoder'
            config: Preprocessing config
        
        Returns:
            Dict: {'train': Compose, 'val': Compose, 'test': Compose}
        """
        if model_type == 'classifier':
            return {
                'train': PipelineBuilder.build_train_pipeline(config),
                'val': PipelineBuilder.build_val_pipeline(config),
                'test': PipelineBuilder.build_test_pipeline(config)
            }
        
        elif model_type == 'siamese':
            # Siamese için özel augmentasyon (iki görüntü de aynı transform)
            return {
                'train': PipelineBuilder.build_train_pipeline(config),
                'val': PipelineBuilder.build_val_pipeline(config),
                'test': PipelineBuilder.build_test_pipeline(config)
            }
        
        elif model_type == 'autoencoder':
            # Autoencoder için hafif augmentasyon
            # Config'i kopyala ve hafiflet
            light_config = config.copy()
            if 'augmentation' in light_config:
                aug = light_config['augmentation']
                # Tüm p değerlerini düşür
                for key in ['random_flip', 'random_rotation', 'random_shift']:
                    if key in aug and aug[key].get('enabled', False):
                        aug[key]['p'] = 0.3
            
            return {
                'train': PipelineBuilder.from_config(light_config, mode='train'),
                'val': PipelineBuilder.build_val_pipeline(config),
                'test': PipelineBuilder.build_test_pipeline(config)
            }
        
        else:
            raise ValueError(f"Bilinmeyen model tipi: {model_type}")


def create_preprocessing_pipeline(config: Dict, 
                                  model_type: str = 'classifier',
                                  verbose: bool = True) -> Dict[str, Compose]:
    """
    Preprocessing pipeline oluştur (High-level API)
    
    Args:
        config: Full config dictionary
        model_type: Model tipi
        verbose: Pipeline özetini yazdır
    
    Returns:
        Dict: {'train': Compose, 'val': Compose, 'test': Compose}
    
    Example:
        >>> config = load_config('configs/config.yaml')
        >>> pipelines = create_preprocessing_pipeline(config, model_type='classifier')
        >>> train_transform = pipelines['train']
    """
    # Config'den preprocessing bölümünü al
    preprocessing_config = config.get('preprocessing', {}).copy()
    dataset_cfg = config.get('dataset', {})
    if 'target_size' not in preprocessing_config and dataset_cfg.get('image_size'):
        preprocessing_config['target_size'] = dataset_cfg.get('image_size')
    # Augmentation strategy (dataset stats yoksa level'a fallback)
    strat_cfg = preprocessing_config.get('augmentation_strategy', {})
    if strat_cfg:
        try:
            from .augmentation_manager import AugmentationManager
            level = strat_cfg.get('level')
            if strat_cfg.get('auto_adjust', False):
                level = level or 'normal'
            if level:
                preset = AugmentationManager(verbose=False).get_preset_config(level)
                if preset:
                    aug_cfg = preprocessing_config.get('augmentation', {})
                    aug_cfg.update(preset)
                    aug_cfg['enabled'] = True
                    preprocessing_config['augmentation'] = aug_cfg

                    if 'adaptive_crop' in preset or 'mask_processing' in preset:
                        med_cfg = preprocessing_config.get('medical', {})
                        if 'adaptive_crop' in preset:
                            med_cfg['adaptive_crop'] = preset['adaptive_crop']
                        if 'mask_processing' in preset:
                            med_cfg['mask_processing'] = preset['mask_processing']
                        preprocessing_config['medical'] = med_cfg
        except Exception:
            pass

    
    # Validate
    if not PipelineBuilder.validate_config(preprocessing_config):
        raise ValueError("Geçersiz preprocessing config!")
    
    # Strategy kullanarak pipeline'ları oluştur
    pipelines = PreprocessingStrategy.get_strategy(model_type, preprocessing_config)
    
    # Özet yazdır
    if verbose:
        print("\n🔧 Preprocessing Pipeline'ları Oluşturuldu")
        print(f"Model Tipi: {model_type}")
        PipelineBuilder.print_pipeline_summary(pipelines['train'], "Training Pipeline")
        PipelineBuilder.print_pipeline_summary(pipelines['val'], "Validation Pipeline")
    
    return pipelines
