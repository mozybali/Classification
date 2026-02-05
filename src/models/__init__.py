"""
Model Modülü - Modular Architecture
"""

# Base models
from .base_model import BaseModel, BaseClassifier, BaseGraphModel

# CNN models
from .cnn_models import CNN3DSimple, ResNet3D, DenseNet3D

# GNN models
try:
    from .gnn_models import GCNClassifier, GATClassifier, GraphSAGEClassifier
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False

# Factory
from .model_factory import ModelFactory, load_model_from_checkpoint, get_model_config_template

__all__ = [
    # Base
    'BaseModel',
    'BaseClassifier',
    'BaseGraphModel',
    
    # CNN
    'CNN3DSimple',
    'ResNet3D',
    'DenseNet3D',
    
    # Factory
    'ModelFactory',
    'load_model_from_checkpoint',
    'get_model_config_template',
]

if GNN_AVAILABLE:
    __all__.extend(['GCNClassifier', 'GATClassifier', 'GraphSAGEClassifier'])

