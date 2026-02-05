"""
Preprocessing Module - 3D Medical Image Processing
Modular and config-driven preprocessing for NeAR ALAN dataset
"""

# Core classes
from .image_loader import ImageLoader, DatasetStatistics, save_statistics_report
from .preprocess import ALANDataset, DataPreprocessor

# Transform builders
from .pipeline_builder import (
    PipelineBuilder,
    PreprocessingStrategy,
    create_preprocessing_pipeline
)

# DataLoader factory
from .dataloader_factory import DataLoaderFactory, get_dataloaders

# NaN handling and data splitting
from .nan_handler import NaNHandler, quick_nan_check, clean_dataset
from .data_splitter import DataSplitter, quick_split

# Class balancing and augmentation management
from .class_balancer import ClassBalancer, quick_balance_check
from .augmentation_manager import AugmentationManager

# Individual transforms (eğer ihtiyaç duyulursa)
from .image_transforms import (
    # Base
    BaseTransform,
    Compose,
    # Basic transforms
    ToFloat,
    Normalize,
    AddChannel,
    # Augmentations
    RandomFlip3D,
    RandomRotation3D,
    RandomShift3D,
    RandomZoom3D,
    ElasticDeformation,
    RandomNoise,
    # Prebuilt pipelines
    get_training_transforms,
    get_validation_transforms,
    get_inference_transforms
)

# Medical transforms
from .medical_transforms import (
    MedicalIntensityNormalization,
    AdaptiveROICrop,
    ResampleToSpacing,
    BinaryMaskProcessor,
    get_medical_kidney_pipeline
)

# Public API
__all__ = [
    # Core
    'ImageLoader',
    'DatasetStatistics',
    'ALANDataset',
    'DataPreprocessor',
    
    # Builders
    'PipelineBuilder',
    'PreprocessingStrategy',
    'create_preprocessing_pipeline',
    
    # Factory
    'DataLoaderFactory',
    'get_dataloaders',
    
    # NaN handling
    'NaNHandler',
    'quick_nan_check',
    'clean_dataset',
    
    # Data splitting
    'DataSplitter',
    'quick_split',
    
    # Class balancing
    'ClassBalancer',
    'quick_balance_check',
    
    # Augmentation management
    'AugmentationManager',
    
    # Transforms
    'BaseTransform',
    'Compose',
    'ToFloat',
    'Normalize',
    'AddChannel',
    'RandomFlip3D',
    'RandomRotation3D',
    'RandomShift3D',
    'RandomZoom3D',
    'ElasticDeformation',
    'RandomNoise',
    'get_training_transforms',
    'get_validation_transforms',
    'get_inference_transforms',
    
    # Medical transforms
    'MedicalIntensityNormalization',
    'AdaptiveROICrop',
    'ResampleToSpacing',
    'BinaryMaskProcessor',
    'get_medical_kidney_pipeline',
    
    # Utils
    'save_statistics_report',
]

# Version
__version__ = '1.0.0'
