"""
Base Model - Tüm modeller için temel sınıf
Modüler model mimarisi için abstract base class
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple


class BaseModel(nn.Module, ABC):
    """Tüm modeller için base class"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Model konfigürasyonu
        """
        super(BaseModel, self).__init__()
        self.config = config
        self.num_classes = config.get('num_classes', 2)
        self.model_name = self.__class__.__name__
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        pass
    
    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feature extraction
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        pass
    
    def get_num_parameters(self) -> Tuple[int, int]:
        """
        Model parametre sayısını döndürür
        
        Returns:
            (total_params, trainable_params)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def freeze_backbone(self):
        """Backbone'u dondurur (transfer learning için)"""
        if hasattr(self, 'backbone'):
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"✓ {self.model_name}: Backbone dondu")
    
    def unfreeze_backbone(self):
        """Backbone'u çözülür"""
        if hasattr(self, 'backbone'):
            for param in self.backbone.parameters():
                param.requires_grad = True
            print(f"✓ {self.model_name}: Backbone çözüldü")
    
    def print_model_info(self):
        """Model bilgilerini yazdırır"""
        total, trainable = self.get_num_parameters()
        print(f"\n{'='*70}")
        print(f"MODEL: {self.model_name}")
        print(f"{'='*70}")
        print(f"Total Parameters: {total:,}")
        print(f"Trainable Parameters: {trainable:,}")
        print(f"Non-trainable Parameters: {total - trainable:,}")
        print(f"Memory (approx): {total * 4 / 1024 / 1024:.2f} MB")
        print(f"{'='*70}\n")


class BaseClassifier(BaseModel):
    """Classification modelleri için base class"""
    
    def __init__(self, config: Dict):
        super(BaseClassifier, self).__init__(config)
        self.dropout = config.get('dropout', 0.5)
    
    def create_classifier_head(self, in_features: int, hidden_dims: list = None) -> nn.Module:
        """
        Classification head oluşturur
        
        Args:
            in_features: Input feature dimension
            hidden_dims: Hidden layer dimensions (None ise default kullanılır)
            
        Returns:
            nn.Module: Classifier head
        """
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        layers = []
        current_dim = in_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout)
            ])
            current_dim = hidden_dim
        
        # Final layer
        layers.append(nn.Linear(current_dim, self.num_classes))
        
        return nn.Sequential(*layers)


class BaseGraphModel(BaseModel):
    """Graph Neural Network modelleri için base class"""
    
    def __init__(self, config: Dict):
        super(BaseGraphModel, self).__init__(config)
        self.node_features = config.get('node_features', 64)
        self.edge_features = config.get('edge_features', 32)
    
    @abstractmethod
    def build_graph(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        3D volume'den graph oluşturur
        
        Args:
            x: 3D volume tensor (B, C, D, H, W)
            
        Returns:
            (node_features, edge_index)
        """
        pass
