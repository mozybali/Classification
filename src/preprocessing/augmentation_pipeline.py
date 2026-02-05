"""
Augmentation Pipeline Documentation and Validation
Data augmentation sırasını ve stratejisini açıklayan ve doğrulayan modül
"""

import numpy as np
from typing import List, Dict, Callable, Optional
import torch


class AugmentationPipeline:
    """
    Augmentation pipeline manager
    
    Sıra önemlidir:
    1. Spatial transforms (flip, rotate, shift)
    2. Intensity transforms (zoom, elastic deformation)
    3. Noise addition
    
    Neden bu sıra?
    - Spatial transforms önce gelir (data geometry'yi değiştirir)
    - Intensity transforms sonra (data values'ını değiştirir)
    - Noise en sona (minimal preprocessing etkisi)
    """
    
    VALID_STAGES = ['spatial', 'intensity', 'noise']
    
    def __init__(self, transforms_dict: Dict[str, List[Callable]]):
        """
        Args:
            transforms_dict: {
                'spatial': [flip, rotate, shift, ...],
                'intensity': [zoom, elastic, ...],
                'noise': [gaussian_noise, ...]
            }
        """
        self.transforms = {}
        
        # Validate ve organize et
        for stage in self.VALID_STAGES:
            if stage in transforms_dict:
                self.transforms[stage] = transforms_dict[stage]
                print(f"  Stage '{stage}': {len(self.transforms[stage])} transforms")
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Image'e pipeline'ı uygula (sıra garantili)
        
        Args:
            image: [H, W, D] array
            
        Returns:
            Augmented image
        """
        result = image.copy()
        
        # Stage 1: Spatial transforms
        if 'spatial' in self.transforms:
            for transform in self.transforms['spatial']:
                if np.random.rand() < 0.5:  # 50% probability
                    result = transform(result)
        
        # Stage 2: Intensity transforms
        if 'intensity' in self.transforms:
            for transform in self.transforms['intensity']:
                if np.random.rand() < 0.3:  # 30% probability
                    result = transform(result)
        
        # Stage 3: Noise addition
        if 'noise' in self.transforms:
            for transform in self.transforms['noise']:
                if np.random.rand() < 0.2:  # 20% probability
                    result = transform(result)
        
        return result
    
    def __repr__(self) -> str:
        """Pipeline bilgisini print et"""
        info = "Augmentation Pipeline:\n"
        for stage in self.VALID_STAGES:
            if stage in self.transforms:
                info += f"  {stage.upper()}:\n"
                for t in self.transforms[stage]:
                    info += f"    - {t.__class__.__name__}\n"
        return info


class AugmentationValidator:
    """Augmentation config'ini validate et"""
    
    @staticmethod
    def validate_config(config: Dict) -> bool:
        """
        Config valid mi?
        
        Kontrol noktaları:
        - Probability'ler [0, 1] arasında
        - Angle ranges mantıklı
        - Alpha/sigma values positive
        """
        aug_config = config.get('preprocessing', {}).get('augmentation', {})
        
        validation_results = {
            'valid': True,
            'warnings': []
        }
        
        # Flip probability
        flip_p = aug_config.get('random_flip', {}).get('p', 0.5)
        if not (0 <= flip_p <= 1):
            validation_results['valid'] = False
            validation_results['warnings'].append(
                f"❌ random_flip.p must be [0,1], got {flip_p}"
            )
        
        # Rotation angles
        rot_range = aug_config.get('random_rotation', {}).get('angle_range', [-15, 15])
        if len(rot_range) != 2 or rot_range[0] >= rot_range[1]:
            validation_results['warnings'].append(
                f"⚠️  random_rotation.angle_range unusual: {rot_range}"
            )
        
        # Elastic deformation
        alpha = aug_config.get('elastic_deformation', {}).get('alpha', 10)
        sigma = aug_config.get('elastic_deformation', {}).get('sigma', 4)
        if alpha <= 0 or sigma <= 0:
            validation_results['valid'] = False
            validation_results['warnings'].append(
                f"❌ elastic_deformation: alpha and sigma must be > 0"
            )
        
        # Zoom range
        zoom_range = aug_config.get('random_zoom', {}).get('zoom_range', [0.9, 1.1])
        if not (0 < zoom_range[0] < zoom_range[1]):
            validation_results['warnings'].append(
                f"⚠️  random_zoom.zoom_range unusual: {zoom_range}"
            )
        
        return validation_results


class AugmentationPreset:
    """Önceden ayarlanmış augmentation profilleri"""
    
    PRESETS = {
        'light': {
            'random_flip': {'p': 0.3, 'axes': [0, 1, 2]},
            'random_rotation': {'p': 0.2, 'angle_range': [-10, 10]},
            'random_shift': {'p': 0.2, 'max_shift': 5},
        },
        'normal': {
            'random_flip': {'p': 0.5, 'axes': [0, 1, 2]},
            'random_rotation': {'p': 0.5, 'angle_range': [-15, 15]},
            'random_shift': {'p': 0.5, 'max_shift': 10},
            'random_zoom': {'p': 0.3, 'zoom_range': [0.9, 1.1]},
            'elastic_deformation': {'p': 0.3, 'alpha': 10, 'sigma': 4},
        },
        'heavy': {
            'random_flip': {'p': 0.8, 'axes': [0, 1, 2]},
            'random_rotation': {'p': 0.8, 'angle_range': [-25, 25]},
            'random_shift': {'p': 0.8, 'max_shift': 15},
            'random_zoom': {'p': 0.5, 'zoom_range': [0.8, 1.2]},
            'elastic_deformation': {'p': 0.5, 'alpha': 15, 'sigma': 5},
            'random_noise': {'p': 0.3, 'noise_std': 0.02},
        },
        'medical_kidney': {
            # Medical imaging best practices
            'random_flip': {'p': 0.5, 'axes': [0, 1, 2]},
            'random_rotation': {'p': 0.5, 'angle_range': [-20, 20]},
            'elastic_deformation': {'p': 0.4, 'alpha': 8, 'sigma': 3},
            'random_shift': {'p': 0.4, 'max_shift': 8},
            'random_zoom': {'p': 0.2, 'zoom_range': [0.95, 1.05]},
            # Kidney için moderate augmentation
        }
    }
    
    @classmethod
    def get_preset(cls, preset_name: str) -> Dict:
        """Preset'i al"""
        if preset_name not in cls.PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. "
                f"Valid: {list(cls.PRESETS.keys())}"
            )
        return cls.PRESETS[preset_name]
    
    @classmethod
    def list_presets(cls) -> List[str]:
        """Tüm presets'i listele"""
        return list(cls.PRESETS.keys())


# Documentation
AUGMENTATION_GUIDE = """
🎨 DATA AUGMENTATION PIPELINE GUIDE
===================================

STAGE 1: SPATIAL TRANSFORMS (Geometry değiştirir)
-------------------------------------------------
✓ Random Flip (axes: 0, 1, 2)
  - Aynı anatomik görünüm oluşturur
  - Flip probability: 0.3-0.8

✓ Random Rotation (angle_range)
  - Lateral rotation simülasyonu
  - Range: [-25, 25] degrees

✓ Random Shift (max_shift)
  - Position variation
  - Medical imaging için safe

✓ Random Zoom (zoom_range)
  - Scale variation
  - Range: [0.8, 1.2]

STAGE 2: INTENSITY TRANSFORMS (Values değiştirir)
--------------------------------------------------
✓ Elastic Deformation (alpha, sigma)
  - Kidney deformation simülasyonu
  - Medical imaging için uygun

✓ Intensity Normalization
  - Preprocesse integrated

STAGE 3: NOISE ADDITION (Realism ekler)
---------------------------------------
✓ Gaussian Noise (noise_std)
  - Scanner noise simülasyonu
  - std: 0.01-0.05

APPLICATION ORDER (Önemli!)
---------------------------
1. Spatial (flip → rotate → shift → zoom)
2. Intensity (elastic deformation)
3. Noise (gaussian noise)

WHY THIS ORDER?
- Spatial transforms'ı ilk yapmak geometry'yi koru
- Intensity transforms'ı sonra uygulamak value changes'ı isit yapabilir
- Noise'u en sona eklemek minimal preprocessing etkisi sağlar

PROBABILITY SETTINGS
-------------------
Light mode:   30% spatial, 20% intensity, 0% noise
Normal mode:  50% spatial, 30% intensity, 20% noise
Heavy mode:   80% spatial, 50% intensity, 30% noise

MEDICAL BEST PRACTICES
----------------------
- Kidney için 'medical_kidney' preset kullan
- Flip probability <= 0.5 (anatomy constraint)
- Rotation angle_range <= ±25 degrees
- Zoom range <= [0.95, 1.05] (small changes)
- Elastic deformation alpha <= 15 (small deformations)
"""

if __name__ == '__main__':
    # Presets'i listele
    print(f"Available presets: {AugmentationPreset.list_presets()}")
    
    # Preset'i al
    normal_config = AugmentationPreset.get_preset('normal')
    print(f"\nNormal augmentation config:\n{normal_config}")
    
    # Validation
    validator = AugmentationValidator()
    result = validator.validate_config({'preprocessing': {'augmentation': normal_config}})
    print(f"\nValidation result: {result}")
    
    print("\n" + "="*70)
    print(AUGMENTATION_GUIDE)
