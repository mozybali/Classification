"""
Config Validation with Pydantic
Tüm konfigürasyonları validate et
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from pathlib import Path
import yaml


class NaNHandlingConfig(BaseModel):
    """NaN işleme konfigürasyonu"""
    enabled: bool = Field(default=False)
    method: str = Field(default="fill_mean")
    fill_value: float = Field(default=0.0)
    columns: Optional[List[str]] = Field(default=None)
    report_path: str = Field(default="outputs/nan_report.json")
    
    @validator('method')
    def validate_method(cls, v):
        valid = ['remove', 'fill_value', 'fill_mean', 'fill_median', 'fill_mode', 'fill_forward', 'fill_backward']
        if v not in valid:
            raise ValueError(f"method {valid}'den biri olmalı")
        return v


class DataSplittingConfig(BaseModel):
    """Veri bölme konfigürasyonu"""
    enabled: bool = Field(default=True)
    method: str = Field(default="stratified")
    train_ratio: float = Field(default=0.7, gt=0, lt=1)
    val_ratio: float = Field(default=0.15, gt=0, lt=1)
    test_ratio: float = Field(default=0.15, gt=0, lt=1)
    stratify_column: str = Field(default="ROI_anomaly")
    patient_id_column: str = Field(default="ROI_id")
    existing_split_column: str = Field(default="subset")
    random_state: int = Field(default=42)
    save_splits: bool = Field(default=True)
    splits_output_dir: str = Field(default="outputs/splits")
    
    @validator('method')
    def validate_method(cls, v):
        valid = ['simple', 'stratified', 'patient', 'existing']
        if v not in valid:
            raise ValueError(f"method {valid}'den biri olmalı")
        return v
    
    @validator('train_ratio', 'val_ratio', 'test_ratio')
    def validate_ratios(cls, v):
        if not (0 < v < 1):
            raise ValueError("Ratio değerleri 0 ile 1 arasında olmalı")
        return v


class DatasetConfig(BaseModel):
    """Dataset konfigürasyonu"""
    path: str = Field(..., description="Dataset yolu")
    csv_file: str = Field(default="info.csv", description="CSV dosyası")
    zip_file: str = Field(default="ALAN.zip", description="ZIP dosyası")
    image_size: List[int] = Field(default=[128, 128, 128], description="Image boyutu")
    channels: int = Field(default=1, description="Channel sayısı")
    nan_handling: Optional[NaNHandlingConfig] = Field(default=None)
    data_splitting: Optional[DataSplittingConfig] = Field(default=None)
    
    @validator('image_size')
    def validate_image_size(cls, v):
        if len(v) != 3 or any(x <= 0 for x in v):
            raise ValueError("image_size [H, W, D] formatında positive olmalı")
        return v
    
    @validator('channels')
    def validate_channels(cls, v):
        if v <= 0:
            raise ValueError("channels > 0 olmalı")
        return v
    
    @validator('path')
    def validate_path(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Dataset path bulunamadı: {v}")
        return v


class AugmentationConfig(BaseModel):
    """Augmentation konfigürasyonu"""
    enabled: bool = Field(default=True)
    mode: str = Field(default="normal")
    
    @validator('mode')
    def validate_mode(cls, v):
        valid = ['light', 'normal', 'heavy']
        if v not in valid:
            raise ValueError(f"mode {valid}'den biri olmalı")
        return v


class TrainingConfig(BaseModel):
    """Training konfigürasyonu"""
    epochs: int = Field(default=100, ge=1, le=10000, description="Epoch sayısı")
    batch_size: int = Field(default=32, ge=1, le=1024, description="Batch size")
    learning_rate: float = Field(default=1e-4, gt=0, description="Learning rate")
    num_workers: int = Field(default=0, ge=0, le=32, description="Worker sayısı")  # Windows: default 0
    loss_type: str = Field(default='cross_entropy', description="Loss tipi")
    optimizer: str = Field(default='adam', description="Optimizer tipi")
    use_amp: bool = Field(default=True, description="Mixed precision training")
    
    @validator('loss_type')
    def validate_loss_type(cls, v):
        valid = ['cross_entropy', 'focal', 'weighted_cross_entropy', 'bce', 'bce_with_logits']
        if v not in valid:
            raise ValueError(f"loss_type {valid}'den biri olmalı")
        return v
    
    @validator('optimizer')
    def validate_optimizer(cls, v):
        valid = ['adam', 'adamw', 'sgd', 'adadelta', 'adagrad']
        if v not in valid:
            raise ValueError(f"optimizer {valid}'den biri olmalı")
        return v


class ModelConfig(BaseModel):
    """Model konfigürasyonu"""
    model_type: str = Field(default='resnet3d', description="Model tipi")
    num_classes: int = Field(default=2, ge=2, description="Sınıf sayısı")
    in_channels: int = Field(default=1, ge=1, description="Input channel sayısı")
    base_filters: int = Field(default=32, ge=16, description="Base filter sayısı")
    dropout: float = Field(default=0.5, ge=0, le=1, description="Dropout rate")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        valid = ['cnn3d_simple', 'resnet3d', 'densenet3d', 'gcn', 'gat', 'graphsage']
        if v not in valid:
            raise ValueError(f"model_type {valid}'den biri olmalı")
        return v


class ProjectConfig(BaseModel):
    """Tüm proje konfigürasyonu"""
    dataset: DatasetConfig
    training: TrainingConfig
    model: ModelConfig
    preprocessing: Optional[Dict[str, Any]] = Field(default=None)
    seed: int = Field(default=42, ge=0, description="Random seed")
    device: str = Field(default='cuda', description="Device tipi")
    use_amp: bool = Field(default=True, description="Mixed precision training")
    
    @validator('device')
    def validate_device(cls, v):
        if v not in ['cuda', 'cpu']:
            raise ValueError(f"device 'cuda' veya 'cpu' olmalı")
        return v
    
    class Config:
        """Pydantic config"""
        arbitrary_types_allowed = True
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ProjectConfig':
        """YAML dosyasından load et"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        try:
            return cls(**config_dict)
        except ValueError as e:
            raise ValueError(f"Config validation hatası: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Dict'e dönüştür"""
        return self.dict()
    
    def validate_and_report(self) -> str:
        """Validation raporu oluştur"""
        report = "✅ Config Validation Passed\n\n"
        report += f"📊 Dataset Configuration:\n"
        report += f"  - Path: {self.dataset.path}\n"
        report += f"  - CSV: {self.dataset.csv_file}\n"
        report += f"  - Image Size: {self.dataset.image_size}\n"
        report += f"  - Channels: {self.dataset.channels}\n\n"
        
        report += f"🤖 Model Configuration:\n"
        report += f"  - Type: {self.model.model_type}\n"
        report += f"  - Classes: {self.model.num_classes}\n"
        report += f"  - Base Filters: {self.model.base_filters}\n"
        report += f"  - Dropout: {self.model.dropout}\n\n"
        
        report += f"🏋️ Training Configuration:\n"
        report += f"  - Epochs: {self.training.epochs}\n"
        report += f"  - Batch Size: {self.training.batch_size}\n"
        report += f"  - Learning Rate: {self.training.learning_rate}\n"
        report += f"  - Optimizer: {self.training.optimizer}\n"
        report += f"  - Loss: {self.training.loss_type}\n"
        report += f"  - Mixed Precision: {self.training.use_amp}\n\n"
        
        report += f"⚙️ General Configuration:\n"
        report += f"  - Device: {self.device}\n"
        report += f"  - Random Seed: {self.seed}\n"
        
        return report


# Example usage
if __name__ == '__main__':
    try:
        config = ProjectConfig.from_yaml('configs/config.yaml')
        print(config.validate_and_report())
    except Exception as e:
        print(f"❌ Config Error: {e}")
        import traceback
        traceback.print_exc()

