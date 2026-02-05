"""
Model Factory - Config'den model oluşturma
Modüler model seçimi ve instantiation
"""

import torch
from typing import Dict, Optional
from .base_model import BaseModel
from .cnn_models import CNN3DSimple, ResNet3D, DenseNet3D
from .gnn_models import GCNClassifier, GATClassifier, GraphSAGEClassifier


class ModelFactory:
    """Config'den model oluşturan factory class"""
    
    # Model registry
    MODEL_REGISTRY = {
        # CNN Models
        'cnn3d_simple': CNN3DSimple,
        'resnet3d': ResNet3D,
        'densenet3d': DenseNet3D,
        
        # GNN Models
        'gcn': GCNClassifier,
        'gat': GATClassifier,
        'graphsage': GraphSAGEClassifier,
    }
    
    @classmethod
    def create_model(cls, config: Dict) -> BaseModel:
        """
        Config'den model oluşturur
        
        Args:
            config: Model konfigürasyonu
                Required keys:
                - model_type: Model tipi (örn: 'cnn3d_simple', 'gcn')
                - num_classes: Sınıf sayısı
                
        Returns:
            BaseModel: Oluşturulan model
            
        Example config:
            model:
              model_type: "resnet3d"
              num_classes: 2
              in_channels: 1
              base_filters: 32
              dropout: 0.5
        """
        model_type = config.get('model_type', 'cnn3d_simple').lower()
        
        if model_type not in cls.MODEL_REGISTRY:
            available = ', '.join(cls.MODEL_REGISTRY.keys())
            raise ValueError(
                f"Bilinmeyen model tipi: {model_type}\n"
                f"Mevcut modeller: {available}"
            )
        
        model_class = cls.MODEL_REGISTRY[model_type]
        model = model_class(config)
        
        print(f"✓ Model oluşturuldu: {model_type}")
        
        return model
    
    @classmethod
    def list_available_models(cls) -> list:
        """Mevcut model tiplerini listeler"""
        return list(cls.MODEL_REGISTRY.keys())
    
    @classmethod
    def get_model_info(cls, model_type: str) -> str:
        """Model hakkında bilgi döndürür"""
        if model_type not in cls.MODEL_REGISTRY:
            return f"Model bulunamadı: {model_type}"
        
        model_class = cls.MODEL_REGISTRY[model_type]
        return f"{model_type}: {model_class.__doc__}"
    
    @classmethod
    def register_model(cls, name: str, model_class):
        """Yeni model tipi kaydet (extensibility için)"""
        cls.MODEL_REGISTRY[name] = model_class
        print(f"✓ Yeni model kaydedildi: {name}")


def load_model_from_checkpoint(checkpoint_path: str, config: Dict, device: str = 'cuda') -> BaseModel:
    """
    Checkpoint'ten model yükler
    
    Args:
        checkpoint_path: Checkpoint dosya yolu
        config: Model konfigürasyonu
        device: 'cuda' veya 'cpu'
        
    Returns:
        Yüklenmiş model
    """
    # Model oluştur
    model = ModelFactory.create_model(config)
    
    # Checkpoint yükle
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # State dict yükle
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"✓ Model checkpoint'ten yüklendi: {checkpoint_path}")
    
    return model


def get_model_config_template(model_type: str) -> Dict:
    """
    Model tipi için örnek config döndürür
    
    Args:
        model_type: Model tipi
        
    Returns:
        Örnek config dictionary
    """
    templates = {
        'cnn3d_simple': {
            'model_type': 'cnn3d_simple',
            'num_classes': 2,
            'in_channels': 1,
            'base_filters': 16,
            'dropout': 0.5
        },
        'resnet3d': {
            'model_type': 'resnet3d',
            'num_classes': 2,
            'in_channels': 1,
            'base_filters': 32,
            'num_blocks': [2, 2, 2, 2],
            'dropout': 0.5
        },
        'densenet3d': {
            'model_type': 'densenet3d',
            'num_classes': 2,
            'in_channels': 1,
            'growth_rate': 16,
            'num_layers': [4, 4, 4, 4],
            'dropout': 0.5
        },
        'gcn': {
            'model_type': 'gcn',
            'num_classes': 2,
            'node_features': 4,
            'hidden_channels': 64,
            'num_layers': 3,
            'dropout': 0.5
        },
        'gat': {
            'model_type': 'gat',
            'num_classes': 2,
            'node_features': 4,
            'hidden_channels': 64,
            'num_layers': 3,
            'num_heads': 4,
            'dropout': 0.5
        },
        'graphsage': {
            'model_type': 'graphsage',
            'num_classes': 2,
            'node_features': 4,
            'hidden_channels': 64,
            'num_layers': 3,
            'dropout': 0.5
        }
    }
    
    return templates.get(model_type, {})
