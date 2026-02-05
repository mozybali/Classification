"""
Tıbbi Görüntü İşleme için Özel Transform'lar
Medical imaging standartlarına uygun preprocessing transforms
Özellikle binary kidney segmentation masks için optimize edilmiş
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, Optional, Union
from .image_transforms import BaseTransform


class MedicalIntensityNormalization(BaseTransform):
    """
    Medical imaging için intensity normalization
    
    Özellikler:
    - Z-score normalization (outlier-robust)
    - Min-max normalization (percentile-based)
    - Percentile clipping ile outlier handling
    
    Binary mask'ler için değil, intensity image'lar için kullanılır.
    NeAR dataset'inde binary mask var, bu transform ileride intensity
    image'lar eklenirse kullanılır.
    
    Args:
        method: 'z-score' veya 'minmax'
        percentile_range: Outlier filtreleme için percentile aralığı (lower, upper)
        clip_output: Output'u [0, 1] aralığına clip et
    """
    
    def __init__(
        self, 
        method: str = 'z-score',
        percentile_range: Tuple[float, float] = (1.0, 99.0),
        clip_output: bool = False
    ):
        if method not in ['z-score', 'minmax']:
            raise ValueError(f"Method must be 'z-score' or 'minmax', got {method}")
        
        if not (0 <= percentile_range[0] < percentile_range[1] <= 100):
            raise ValueError(f"Invalid percentile_range: {percentile_range}")
        
        self.method = method
        self.percentile_range = percentile_range
        self.clip_output = clip_output
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Intensity normalization uygula
        
        Args:
            image: 3D numpy array (H, W, D)
            
        Returns:
            Normalized 3D array
        """
        # Binary mask check - binary mask'lere uygulama
        if image.dtype == bool or set(np.unique(image)) <= {0, 1}:
            return image.astype(np.float32)
        
        # Percentile-based outlier filtreleme
        lower_percentile, upper_percentile = self.percentile_range
        lower_bound = np.percentile(image, lower_percentile)
        upper_bound = np.percentile(image, upper_percentile)
        
        # Outlier'ları çıkar ve istatistikleri hesapla
        mask = (image >= lower_bound) & (image <= upper_bound)
        valid_pixels = image[mask]
        
        if len(valid_pixels) == 0:
            # Fallback: tüm image'ı kullan
            valid_pixels = image.flatten()
        
        if self.method == 'z-score':
            # Z-score normalization
            mean = valid_pixels.mean()
            std = valid_pixels.std()
            
            if std < 1e-8:  # Constant image
                normalized = image - mean
            else:
                normalized = (image - mean) / std
        
        elif self.method == 'minmax':
            # Min-max normalization (percentile-based)
            normalized = (image - lower_bound) / (upper_bound - lower_bound + 1e-8)
        
        # Output clipping
        if self.clip_output:
            normalized = np.clip(normalized, 0.0, 1.0)
        
        return normalized.astype(np.float32)
    
    def __repr__(self):
        return (f"MedicalIntensityNormalization(method='{self.method}', "
                f"percentile_range={self.percentile_range}, "
                f"clip_output={self.clip_output})")


class AdaptiveROICrop(BaseTransform):
    """
    Non-zero bölgeyi otomatik tespit edip kırp (Adaptive ROI cropping)
    
    Özellikler:
    - Automatic bounding box detection (non-zero region)
    - Configurable margin (padding)
    - Memory efficiency (gereksiz zero-padding'i kaldırır)
    - Multi-channel support
    
    NeAR ALAN dataset için kritik: 128x128x128 volume'lerde
    kidney sadece küçük bir bölgede olabilir. Bu transform
    gereksiz boş alanı kaldırarak memory ve computation tasarrufu sağlar.
    
    Args:
        margin: ROI etrafına eklenecek margin (voxel cinsinden)
        min_size: Minimum crop size (her eksen için)
        center_crop: Eğer True ise, ROI'yi center'a alacak şekilde crop yapar
    """
    
    def __init__(
        self,
        margin: Union[int, Tuple[int, int, int]] = 10,
        min_size: Union[int, Tuple[int, int, int]] = 32,
        center_crop: bool = False
    ):
        # Margin'i tuple'a çevir
        if isinstance(margin, int):
            self.margin = (margin, margin, margin)
        else:
            self.margin = margin
        
        # Min size'ı tuple'a çevir
        if isinstance(min_size, int):
            self.min_size = (min_size, min_size, min_size)
        else:
            self.min_size = min_size
        
        self.center_crop = center_crop
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Adaptive ROI cropping uygula
        
        Args:
            image: 3D numpy array (H, W, D) veya 4D (C, H, W, D)
            
        Returns:
            Cropped 3D/4D array
        """
        is_4d = image.ndim == 4
        
        if is_4d:
            # Multi-channel: (C, H, W, D)
            # Non-zero detection için tüm channel'ları birleştir
            detection_volume = image.max(axis=0)  # (H, W, D)
        else:
            # Single channel: (H, W, D)
            detection_volume = image
        
        # Non-zero bounding box tespit et
        non_zero_indices = np.nonzero(detection_volume)
        
        if len(non_zero_indices[0]) == 0:
            # Tamamen zero volume - crop yapma
            return image
        
        # Her eksen için min/max
        bbox = []
        for axis_idx in range(3):
            axis_min = non_zero_indices[axis_idx].min()
            axis_max = non_zero_indices[axis_idx].max()
            
            # Margin ekle
            axis_min = max(0, axis_min - self.margin[axis_idx])
            axis_max = min(detection_volume.shape[axis_idx] - 1, 
                          axis_max + self.margin[axis_idx])
            
            # Min size check
            current_size = axis_max - axis_min + 1
            if current_size < self.min_size[axis_idx]:
                # Center'ı koru, size'ı artır
                center = (axis_min + axis_max) // 2
                half_size = self.min_size[axis_idx] // 2
                
                axis_min = max(0, center - half_size)
                axis_max = min(detection_volume.shape[axis_idx] - 1,
                              center + half_size)
            
            bbox.append((axis_min, axis_max + 1))  # +1 for slicing
        
        # Center crop adjustment
        if self.center_crop:
            # ROI'yi volume center'ına yakınlaştır
            for axis_idx in range(3):
                volume_center = detection_volume.shape[axis_idx] // 2
                roi_center = (bbox[axis_idx][0] + bbox[axis_idx][1]) // 2
                
                shift = volume_center - roi_center
                
                # Shift uygula (boundaries check)
                new_min = bbox[axis_idx][0] + shift
                new_max = bbox[axis_idx][1] + shift
                
                if new_min >= 0 and new_max <= detection_volume.shape[axis_idx]:
                    bbox[axis_idx] = (new_min, new_max)
        
        # Crop uygula
        if is_4d:
            cropped = image[:, 
                          bbox[0][0]:bbox[0][1],
                          bbox[1][0]:bbox[1][1],
                          bbox[2][0]:bbox[2][1]]
        else:
            cropped = image[bbox[0][0]:bbox[0][1],
                          bbox[1][0]:bbox[1][1],
                          bbox[2][0]:bbox[2][1]]
        
        return cropped
    
    def __repr__(self):
        return (f"AdaptiveROICrop(margin={self.margin}, "
                f"min_size={self.min_size}, "
                f"center_crop={self.center_crop})")


class ResampleToSpacing(BaseTransform):
    """
    Anisotropic spacing'i normalize et (Physical coordinate handling)
    
    Medical imaging'de voxel'ler genelde isotropic değildir:
    - Axial plane: 0.5mm x 0.5mm
    - Slice thickness: 3mm
    
    Bu transform spacing'i normalize eder (örn: 1mm x 1mm x 1mm)
    
    NeAR dataset için: Dataset eğer spacing bilgisi sağlıyorsa kullanılır.
    Şu an .npy format spacing bilgisi içermiyor, ileride DICOM/NIfTI
    desteği eklenirse kullanılır.
    
    Args:
        target_spacing: Hedef spacing (mm cinsinden), (x, y, z)
        interpolation_order: Scipy interpolation order
            - 0: Nearest neighbor (masks için)
            - 1: Linear (intensity images için)
            - 3: Cubic (smooth intensity images için)
    """
    
    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        interpolation_order: int = 1
    ):
        self.target_spacing = target_spacing
        self.interpolation_order = interpolation_order
    
    def __call__(
        self, 
        image: np.ndarray, 
        current_spacing: Optional[Tuple[float, float, float]] = None
    ) -> np.ndarray:
        """
        Spacing normalization uygula
        
        Args:
            image: 3D numpy array (H, W, D)
            current_spacing: Mevcut spacing (x, y, z). None ise transform uygulanmaz.
            
        Returns:
            Resampled 3D array
        """
        if current_spacing is None:
            # Spacing bilgisi yok, transform uygulama
            return image
        
        # Zoom faktörlerini hesapla
        zoom_factors = [
            curr / target 
            for curr, target in zip(current_spacing, self.target_spacing)
        ]
        
        # Zoom faktörleri ~1.0 ise (tolerance: 1%), skip
        if all(abs(z - 1.0) < 0.01 for z in zoom_factors):
            return image
        
        # Resampling uygula
        resampled = ndimage.zoom(
            image, 
            zoom_factors, 
            order=self.interpolation_order,
            mode='nearest'
        )
        
        return resampled.astype(image.dtype)
    
    def __repr__(self):
        return (f"ResampleToSpacing(target_spacing={self.target_spacing}, "
                f"interpolation_order={self.interpolation_order})")


class BinaryMaskProcessor(BaseTransform):
    """
    Binary segmentation mask'ler için özel işlemler
    
    Özellikler:
    - Hole filling (internal holes)
    - Small component removal
    - Morphological operations (erosion, dilation)
    - Connected component analysis
    
    NeAR ALAN dataset'i için kritik: Binary kidney masks'lerde
    noise removal ve post-processing.
    
    Args:
        fill_holes: Internal hole'ları doldur
        min_component_size: Bu boyuttan küçük component'ları kaldır (voxel sayısı)
        morphology: Morphological operation ('none', 'erosion', 'dilation', 'opening', 'closing')
        structure_size: Morphological structuring element boyutu
    """
    
    def __init__(
        self,
        fill_holes: bool = True,
        min_component_size: int = 100,
        morphology: str = 'none',
        structure_size: int = 3
    ):
        if morphology not in ['none', 'erosion', 'dilation', 'opening', 'closing']:
            raise ValueError(f"Invalid morphology: {morphology}")
        
        self.fill_holes = fill_holes
        self.min_component_size = min_component_size
        self.morphology = morphology
        self.structure_size = structure_size
    
    def __call__(self, mask: np.ndarray) -> np.ndarray:
        """
        Binary mask processing uygula
        
        Args:
            mask: 3D binary mask (H, W, D)
            
        Returns:
            Processed binary mask
        """
        # Boolean'a çevir
        if mask.dtype != bool:
            mask = mask > 0.5
        
        processed = mask.copy()
        
        # 1. Hole filling
        if self.fill_holes:
            processed = ndimage.binary_fill_holes(processed)
        
        # 2. Small component removal
        if self.min_component_size > 0:
            labeled, num_features = ndimage.label(processed)
            
            # Her component'ın boyutunu hesapla
            component_sizes = np.bincount(labeled.ravel())
            
            # Küçük component'ları kaldır
            small_components = component_sizes < self.min_component_size
            small_components[0] = False  # Background'u koru
            
            processed[small_components[labeled]] = False
        
        # 3. Morphological operations
        if self.morphology != 'none':
            structure = ndimage.generate_binary_structure(3, 1)
            
            # Structuring element'i büyüt
            if self.structure_size > 1:
                structure = ndimage.iterate_structure(
                    structure, 
                    self.structure_size - 1
                )
            
            if self.morphology == 'erosion':
                processed = ndimage.binary_erosion(processed, structure=structure)
            elif self.morphology == 'dilation':
                processed = ndimage.binary_dilation(processed, structure=structure)
            elif self.morphology == 'opening':
                processed = ndimage.binary_opening(processed, structure=structure)
            elif self.morphology == 'closing':
                processed = ndimage.binary_closing(processed, structure=structure)
        
        return processed.astype(np.float32)
    
    def __repr__(self):
        return (f"BinaryMaskProcessor(fill_holes={self.fill_holes}, "
                f"min_component_size={self.min_component_size}, "
                f"morphology='{self.morphology}', "
                f"structure_size={self.structure_size})")


# Medical preprocessing pipeline presets
def get_medical_kidney_pipeline(
    normalize_intensity: bool = True,
    adaptive_crop: bool = True,
    mask_processing: bool = True,
    augmentation: bool = True
):
    """
    NeAR ALAN kidney segmentation dataset için önceden tanımlanmış pipeline
    
    Args:
        normalize_intensity: Intensity normalization ekle (intensity image varsa)
        adaptive_crop: Adaptive ROI cropping ekle
        mask_processing: Binary mask post-processing ekle
        augmentation: Data augmentation ekle (training için)
    
    Returns:
        Transform listesi
    """
    from .image_transforms import (
        ToFloat, RandomFlip3D, RandomRotation3D, 
        ElasticDeformation, Compose
    )
    
    transforms = []
    
    # 1. Type conversion
    transforms.append(ToFloat())
    
    # 2. Intensity normalization (intensity image varsa)
    if normalize_intensity:
        transforms.append(
            MedicalIntensityNormalization(
                method='minmax',
                percentile_range=(1.0, 99.0),
                clip_output=True
            )
        )
    
    # 3. Adaptive ROI cropping
    if adaptive_crop:
        transforms.append(
            AdaptiveROICrop(
                margin=10,
                min_size=64,
                center_crop=False
            )
        )
    
    # 4. Binary mask processing
    if mask_processing:
        transforms.append(
            BinaryMaskProcessor(
                fill_holes=True,
                min_component_size=100,
                morphology='closing',
                structure_size=2
            )
        )
    
    # 5. Data augmentation (training için)
    if augmentation:
        transforms.extend([
            RandomFlip3D(p=0.5, axes=[0, 1, 2]),
            RandomRotation3D(
                p=0.5,
                angle_range=(-15, 15),
                axes=(0, 1)
            ),
            ElasticDeformation(
                p=0.3,
                alpha=100,
                sigma=10
            )
        ])
    
    return Compose(transforms)


# Export
__all__ = [
    'MedicalIntensityNormalization',
    'AdaptiveROICrop',
    'ResampleToSpacing',
    'BinaryMaskProcessor',
    'get_medical_kidney_pipeline'
]
