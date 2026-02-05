"""
CNN Models - 3D Convolutional Neural Networks
Kidney anomaly detection için 3D CNN mimarileri
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .base_model import BaseClassifier


class CNN3DSimple(BaseClassifier):
    """Basit 3D CNN modeli"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Model konfigürasyonu
                - in_channels: Input kanal sayısı (default: 1)
                - num_classes: Çıkış sınıf sayısı
                - base_filters: İlk conv layer'daki filter sayısı (default: 16)
        """
        super(CNN3DSimple, self).__init__(config)
        
        in_channels = config.get('in_channels', 1)
        base_filters = config.get('base_filters', 16)
        
        # 3D Convolutional Backbone
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv3d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 128 -> 64
            
            # Block 2
            nn.Conv3d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 64 -> 32
            
            # Block 3
            nn.Conv3d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 32 -> 16
            
            # Block 4
            nn.Conv3d(base_filters * 4, base_filters * 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))  # Global pooling
        )
        
        # Feature dimension
        self.feature_dim = base_filters * 8
        
        # Classifier head
        self.classifier = self.create_classifier_head(
            self.feature_dim,
            hidden_dims=[256, 128]
        )
        
        self.print_model_info()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, D, H, W) - 3D volume
            
        Returns:
            Output logits (B, num_classes)
        """
        features = self.extract_features(x)
        output = self.classifier(features)
        return output
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Feature extraction"""
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        return features


class ResNet3D(BaseClassifier):
    """3D ResNet - Residual connections ile"""
    
    def __init__(self, config: Dict):
        super(ResNet3D, self).__init__(config)
        
        in_channels = config.get('in_channels', 1)
        base_filters = config.get('base_filters', 32)
        num_blocks = config.get('num_blocks', [2, 2, 2, 2])
        
        self.in_channels = base_filters
        
        # Initial convolution
        self.conv1 = nn.Conv3d(in_channels, base_filters, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(base_filters, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(base_filters * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(base_filters * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(base_filters * 8, num_blocks[3], stride=2)
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Feature dimension
        self.feature_dim = base_filters * 8
        
        # Classifier
        self.classifier = self.create_classifier_head(
            self.feature_dim,
            hidden_dims=[512, 256]
        )
        
        self.print_model_info()
    
    def _make_layer(self, out_channels: int, num_blocks: int, stride: int = 1):
        """Residual layer oluşturur"""
        layers = []
        
        # First block (stride ile downsampling)
        layers.append(ResidualBlock3D(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        output = self.classifier(features)
        return output
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ResidualBlock3D(nn.Module):
    """3D Residual Block"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class DenseNet3D(BaseClassifier):
    """3D DenseNet - Dense connections ile"""
    
    def __init__(self, config: Dict):
        super(DenseNet3D, self).__init__(config)
        
        in_channels = config.get('in_channels', 1)
        growth_rate = config.get('growth_rate', 16)
        num_layers = config.get('num_layers', [4, 4, 4, 4])
        
        # Initial convolution
        num_init_features = growth_rate * 2
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dense blocks
        num_features = num_init_features
        for i, num_layer in enumerate(num_layers):
            block = DenseBlock3D(num_features, growth_rate, num_layer)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layer * growth_rate
            
            if i != len(num_layers) - 1:
                trans = TransitionLayer3D(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('avgpool', nn.AdaptiveAvgPool3d((1, 1, 1)))
        
        self.feature_dim = num_features
        
        # Classifier
        self.classifier = self.create_classifier_head(
            self.feature_dim,
            hidden_dims=[512, 256]
        )
        
        self.print_model_info()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        output = self.classifier(features)
        return output
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return features


class DenseBlock3D(nn.Module):
    """3D Dense Block"""
    
    def __init__(self, num_input_features: int, growth_rate: int, num_layers: int):
        super(DenseBlock3D, self).__init__()
        
        for i in range(num_layers):
            layer = DenseLayer3D(num_input_features + i * growth_rate, growth_rate)
            self.add_module(f'denselayer{i+1}', layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for name, layer in self.named_children():
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class DenseLayer3D(nn.Module):
    """3D Dense Layer"""
    
    def __init__(self, num_input_features: int, growth_rate: int):
        super(DenseLayer3D, self).__init__()
        
        self.norm1 = nn.BatchNorm3d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(num_input_features, growth_rate * 4, kernel_size=1, bias=False)
        
        self.norm2 = nn.BatchNorm3d(growth_rate * 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(growth_rate * 4, growth_rate, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        return out


class TransitionLayer3D(nn.Module):
    """3D Transition Layer (downsampling)"""
    
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayer3D, self).__init__()
        
        self.norm = nn.BatchNorm3d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(self.relu(self.norm(x)))
        out = self.pool(out)
        return out
