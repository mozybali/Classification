"""
3D Medikal Görüntü Dönüşümleri ve Augmentasyon
3D binary mask'ler için özel augmentasyon teknikleri
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, Optional, List
import random


class BaseTransform:
    """Tüm transform'lar için base class"""
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Transform'u uygula"""
        raise NotImplementedError
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"


class ToFloat(BaseTransform):
    """Boolean/int array'i float32'ye çevir"""
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32)


class Normalize(BaseTransform):
    """Görüntüyü normalize et"""
    
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.std != 0:
            return (image - self.mean) / self.std
        return image - self.mean
    
    def __repr__(self):
        return f"Normalize(mean={self.mean}, std={self.std})"


class RandomFlip3D(BaseTransform):
    """3D görüntüde rastgele flip uygula"""
    
    def __init__(self, p: float = 0.5, axes: List[int] = None):
        """
        Args:
            p: Flip uygulama olasılığı
            axes: Flip uygulanacak eksenler (None ise tüm eksenler)
        """
        self.p = p
        self.axes = axes if axes is not None else [0, 1, 2]
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            # Rastgele bir eksen seç
            axis = random.choice(self.axes)
            image = np.flip(image, axis=axis).copy()
        return image
    
    def __repr__(self):
        return f"RandomFlip3D(p={self.p}, axes={self.axes})"


class RandomRotation3D(BaseTransform):
    """3D görüntüde rastgele rotasyon"""
    
    def __init__(self, p: float = 0.5, angle_range: Tuple[float, float] = (-15, 15), 
                 axes: Tuple[int, int] = (0, 1)):
        """
        Args:
            p: Rotasyon uygulama olasılığı
            angle_range: Açı aralığı (derece)
            axes: Rotasyon düzlemi (0,1), (0,2) veya (1,2)
        """
        self.p = p
        self.angle_range = angle_range
        self.axes = axes
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            angle = random.uniform(*self.angle_range)
            image = ndimage.rotate(image, angle, axes=self.axes, 
                                  reshape=False, order=1, mode='nearest')
        return image
    
    def __repr__(self):
        return f"RandomRotation3D(p={self.p}, angle_range={self.angle_range}, axes={self.axes})"


class RandomShift3D(BaseTransform):
    """3D görüntüde rastgele kaydırma (translation)"""
    
    def __init__(self, p: float = 0.5, max_shift: int = 10):
        """
        Args:
            p: Shift uygulama olasılığı
            max_shift: Maksimum kaydırma piksel sayısı
        """
        self.p = p
        self.max_shift = max_shift
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            shifts = [random.randint(-self.max_shift, self.max_shift) for _ in range(3)]
            image = ndimage.shift(image, shifts, order=1, mode='nearest')
        return image
    
    def __repr__(self):
        return f"RandomShift3D(p={self.p}, max_shift={self.max_shift})"


class RandomZoom3D(BaseTransform):
    """3D görüntüde rastgele zoom (scale)"""
    
    def __init__(self, p: float = 0.5, zoom_range: Tuple[float, float] = (0.9, 1.1)):
        """
        Args:
            p: Zoom uygulama olasılığı
            zoom_range: Zoom faktörü aralığı
        """
        self.p = p
        self.zoom_range = zoom_range
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            zoom_factor = random.uniform(*self.zoom_range)
            image = ndimage.zoom(image, zoom_factor, order=1, mode='nearest')
            
            # Orijinal boyuta geri getir
            if image.shape != (128, 128, 128):
                # Crop veya pad
                image = self._resize_to_original(image, (128, 128, 128))
        
        return image
    
    def _resize_to_original(self, image: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Görüntüyü hedef boyuta getir (crop veya pad)"""
        current_shape = image.shape
        result = np.zeros(target_shape, dtype=image.dtype)
        
        # Her eksen için hesapla
        slices_src = []
        slices_dst = []
        
        for i in range(3):
            if current_shape[i] > target_shape[i]:
                # Crop
                start = (current_shape[i] - target_shape[i]) // 2
                slices_src.append(slice(start, start + target_shape[i]))
                slices_dst.append(slice(None))
            else:
                # Pad
                start = (target_shape[i] - current_shape[i]) // 2
                slices_src.append(slice(None))
                slices_dst.append(slice(start, start + current_shape[i]))
        
        result[tuple(slices_dst)] = image[tuple(slices_src)]
        return result
    
    def __repr__(self):
        return f"RandomZoom3D(p={self.p}, zoom_range={self.zoom_range})"


class RandomNoise(BaseTransform):
    """Rastgele gürültü ekle (Gaussian noise)"""
    
    def __init__(self, p: float = 0.5, noise_std: float = 0.01):
        """
        Args:
            p: Gürültü ekleme olasılığı
            noise_std: Gürültü standart sapması
        """
        self.p = p
        self.noise_std = noise_std
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            noise = np.random.normal(0, self.noise_std, image.shape)
            image = image + noise.astype(image.dtype)
            # Clip to [0, 1] for binary masks
            image = np.clip(image, 0, 1)
        return image
    
    def __repr__(self):
        return f"RandomNoise(p={self.p}, noise_std={self.noise_std})"


class PadOrCrop3D(BaseTransform):
    """Pad or center-crop 3D volume to a fixed target shape."""

    def __init__(self, target_shape: Tuple[int, int, int] = (128, 128, 128)):
        self.target_shape = tuple(int(x) for x in target_shape)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 4:
            channels = image.shape[0]
            result = np.zeros((channels, *self.target_shape), dtype=image.dtype)
            for c in range(channels):
                result[c] = self._pad_or_crop(image[c], self.target_shape)
            return result

        return self._pad_or_crop(image, self.target_shape)

    def _pad_or_crop(self, image: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        result = np.zeros(target_shape, dtype=image.dtype)
        slices_src = []
        slices_dst = []

        for i in range(3):
            if image.shape[i] > target_shape[i]:
                start = (image.shape[i] - target_shape[i]) // 2
                slices_src.append(slice(start, start + target_shape[i]))
                slices_dst.append(slice(None))
            else:
                start = (target_shape[i] - image.shape[i]) // 2
                slices_src.append(slice(None))
                slices_dst.append(slice(start, start + image.shape[i]))

        result[tuple(slices_dst)] = image[tuple(slices_src)]
        return result

    def __repr__(self):
        return f"PadOrCrop3D(target_shape={self.target_shape})"


class ElasticDeformation(BaseTransform):
    """Elastik deformasyon (özellikle medikal görüntüler için)"""
    
    def __init__(self, p: float = 0.3, alpha: float = 10, sigma: float = 4):
        """
        Args:
            p: Uygulama olasılığı
            alpha: Deformasyon gücü
            sigma: Gaussian smoothing parametresi
        """
        self.p = p
        self.alpha = alpha
        self.sigma = sigma
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            shape = image.shape
            
            # Rastgele deformasyon alanları oluştur
            dx = ndimage.gaussian_filter(
                (np.random.rand(*shape) * 2 - 1), 
                self.sigma, mode="constant", cval=0
            ) * self.alpha
            
            dy = ndimage.gaussian_filter(
                (np.random.rand(*shape) * 2 - 1), 
                self.sigma, mode="constant", cval=0
            ) * self.alpha
            
            dz = ndimage.gaussian_filter(
                (np.random.rand(*shape) * 2 - 1), 
                self.sigma, mode="constant", cval=0
            ) * self.alpha
            
            # Grid koordinatları
            x, y, z = np.meshgrid(
                np.arange(shape[0]), 
                np.arange(shape[1]), 
                np.arange(shape[2]), 
                indexing='ij'
            )
            
            # Deformasyonu uygula
            indices = [
                np.clip(x + dx, 0, shape[0] - 1).flatten(),
                np.clip(y + dy, 0, shape[1] - 1).flatten(),
                np.clip(z + dz, 0, shape[2] - 1).flatten()
            ]
            
            image = ndimage.map_coordinates(
                image, indices, order=1, mode='nearest'
            ).reshape(shape)
        
        return image
    
    def __repr__(self):
        return f"ElasticDeformation(p={self.p}, alpha={self.alpha}, sigma={self.sigma})"


class AddChannel(BaseTransform):
    """Kanal boyutu ekle [H,W,D] -> [1,H,W,D]"""
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return np.expand_dims(image, axis=0)


class Compose:
    """Birden fazla transform'u sırayla uygula"""
    
    def __init__(self, transforms: List[BaseTransform]):
        self.transforms = transforms
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            image = transform(image)
        return image
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


# Önceden tanımlı augmentasyon pipeline'ları
def get_training_transforms(light: bool = False, normalize: bool = False, mean: float = 0.0, std: float = 1.0) -> Compose:
    """
    Eğitim için augmentasyon pipeline'ı
    
    Args:
        light: Hafif augmentasyon (daha az agresif)
        normalize: Normalizasyon uygula (binary mask için False önerilir)
        mean: Normalizasyon mean değeri
        std: Normalizasyon std değeri
    
    Returns:
        Compose: Transform pipeline'ı
    """
    # NOT: Binary mask dataset için normalize=False olmalı (zaten 0/1 değerleri)
    if light:
        transforms = [
            ToFloat(),
            RandomFlip3D(p=0.5, axes=[0, 1, 2]),
            RandomRotation3D(p=0.3, angle_range=(-10, 10)),
            RandomShift3D(p=0.3, max_shift=5),
        ]
    else:
        transforms = [
            ToFloat(),
            RandomFlip3D(p=0.5, axes=[0, 1, 2]),
            RandomRotation3D(p=0.5, angle_range=(-15, 15)),
            RandomShift3D(p=0.5, max_shift=10),
            RandomZoom3D(p=0.3, zoom_range=(0.9, 1.1)),
            ElasticDeformation(p=0.3, alpha=10, sigma=4),
            RandomNoise(p=0.2, noise_std=0.01),
        ]
    
    # Opsiyonel normalizasyon ekle
    if normalize:
        transforms.append(Normalize(mean=mean, std=std))
    
    return Compose(transforms)


def get_validation_transforms(normalize: bool = False, mean: float = 0.0, std: float = 1.0) -> Compose:
    """
    Validasyon/Test için transform pipeline'ı (sadece temel dönüşümler)
    
    Args:
        normalize: Normalizasyon uygula (binary mask için False önerilir)
        mean: Normalizasyon mean değeri
        std: Normalizasyon std değeri
    
    Returns:
        Compose: Transform pipeline'ı
    """
    transforms = [
        ToFloat(),
    ]
    
    # Opsiyonel normalizasyon ekle
    if normalize:
        transforms.append(Normalize(mean=mean, std=std))
    
    return Compose(transforms)


def get_inference_transforms() -> Compose:
    """
    Inference için transform pipeline'ı
    
    Returns:
        Compose: Transform pipeline'ı
    """
    return get_validation_transforms()
