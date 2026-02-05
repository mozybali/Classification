"""
Integration Tests - Model ve Training Tests
Modeller, training, evaluation ve config validation testleri
"""

import pytest
import torch
import numpy as np
from pathlib import Path


class TestModelCreation:
    """Model oluşturma testleri"""
    
    @pytest.mark.model
    def test_create_cnn_model(self):
        """CNN3D Simple model oluşturulabilmeli"""
        from src.models import ModelFactory
        
        config = {
            'model_type': 'cnn3d_simple',
            'num_classes': 2,
            'in_channels': 1,
            'base_filters': 16
        }
        
        model = ModelFactory.create_model(config)
        assert model is not None
        assert model.num_classes == 2
    
    @pytest.mark.model
    def test_create_resnet_model(self):
        """ResNet3D model oluşturulabilmeli"""
        from src.models import ModelFactory
        
        config = {
            'model_type': 'resnet3d',
            'num_classes': 2,
            'in_channels': 1,
            'base_filters': 32
        }
        
        model = ModelFactory.create_model(config)
        assert model is not None
        assert model.model_name == 'ResNet3D'
    
    @pytest.mark.model
    def test_create_densenet_model(self):
        """DenseNet3D model oluşturulabilmeli"""
        from src.models import ModelFactory
        
        config = {
            'model_type': 'densenet3d',
            'num_classes': 2,
            'in_channels': 1,
            'base_filters': 16
        }
        
        model = ModelFactory.create_model(config)
        assert model is not None
        assert model.model_name == 'DenseNet3D'
    
    @pytest.mark.model
    def test_model_forward_pass(self):
        """Model forward pass başarıyla çalışmalı"""
        from src.models import ModelFactory
        
        config = {
            'model_type': 'cnn3d_simple',
            'num_classes': 2,
            'in_channels': 1
        }
        
        model = ModelFactory.create_model(config)
        
        # Dummy input
        x = torch.randn(2, 1, 128, 128, 128)
        output = model(x)
        
        assert output.shape == torch.Size([2, 2])  # [batch_size, num_classes]
    
    @pytest.mark.model
    def test_model_to_device(self):
        """Model device'a taşınabilmeli"""
        from src.models import ModelFactory
        
        config = {'model_type': 'cnn3d_simple', 'num_classes': 2}
        model = ModelFactory.create_model(config)
        
        device = 'cpu'  # Always use CPU for tests
        model = model.to(device)
        
        # Check device placement
        assert next(model.parameters()).device.type == device
    
    @pytest.mark.model
    def test_invalid_model_type(self):
        """Invalid model tipi başarısız olmalı"""
        from src.models import ModelFactory
        
        config = {'model_type': 'invalid_model', 'num_classes': 2}
        
        with pytest.raises((ValueError, KeyError)):
            model = ModelFactory.create_model(config)


class TestTraining:
    """Training testleri"""
    
    @pytest.mark.training
    @pytest.mark.slow
    def test_training_epoch(self):
        """Training epoch başarıyla tamamlanmalı"""
        from src.models import ModelFactory
        from src.training import ModularTrainer
        from torch.utils.data import DataLoader, TensorDataset
        
        # Dummy data
        images = torch.randn(16, 1, 128, 128, 128)
        labels = torch.randint(0, 2, (16,))
        dataset = TensorDataset(images, labels)
        
        # Custom collate function (to match expected format)
        def collate_fn(batch):
            images = torch.stack([x[0] for x in batch])
            labels = torch.tensor([x[1] for x in batch])
            return {'image': images, 'label': labels}
        
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
        
        # Model
        model_config = {'model_type': 'cnn3d_simple', 'num_classes': 2}
        model = ModelFactory.create_model(model_config)
        
        # Training config
        training_config = {
            'epochs': 1,
            'learning_rate': 1e-4,
            'optimizer': 'adam',
            'loss_type': 'cross_entropy',
            'use_amp': False
        }
        
        # Trainer
        trainer = ModularTrainer(
            model=model,
            train_loader=loader,
            val_loader=None,
            config=training_config,
            device='cpu'
        )
        
        # Train
        trainer.train_epoch()
        
        # Check that training updated parameters
        assert trainer.train_history['loss'][-1] > 0


class TestEvaluation:
    """Evaluation testleri"""
    
    @pytest.mark.slow
    def test_evaluation_metrics(self):
        """Evaluation metrikleri hesaplanabilmeli"""
        from src.training.evaluator import ModelEvaluator
        from src.models import ModelFactory
        from torch.utils.data import DataLoader, TensorDataset
        
        # Dummy data
        images = torch.randn(16, 1, 128, 128, 128)
        labels = torch.randint(0, 2, (16,))
        dataset = TensorDataset(images, labels)
        
        def collate_fn(batch):
            images = torch.stack([x[0] for x in batch])
            labels = torch.tensor([x[1] for x in batch])
            return {'image': images, 'label': labels}
        
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
        
        # Model
        model_config = {'model_type': 'cnn3d_simple', 'num_classes': 2}
        model = ModelFactory.create_model(model_config)
        
        # Evaluate
        evaluator = ModelEvaluator(model, device='cpu')
        results = evaluator.evaluate(loader, save_dir=None)
        
        # Check metrics
        metrics = results['metrics']
        assert 'accuracy' in metrics
        assert 'precision_weighted' in metrics
        assert 'recall_weighted' in metrics
        assert 'f1_weighted' in metrics
        assert 'auc_roc' in metrics


class TestConfigValidation:
    """Config validation testleri"""
    
    @pytest.mark.unit
    def test_valid_config(self):
        """Valid config geçmeli"""
        from src.utils.config_validation import ProjectConfig, DatasetConfig, TrainingConfig, ModelConfig
        
        config = ProjectConfig(
            dataset=DatasetConfig(path='NeAR_dataset/ALAN'),
            training=TrainingConfig(epochs=10, batch_size=32),
            model=ModelConfig(model_type='cnn3d_simple'),
            seed=42,
            device='cpu'
        )
        
        assert config is not None
        assert config.training.epochs == 10
        assert config.model.model_type == 'cnn3d_simple'
    
    @pytest.mark.unit
    def test_config_from_yaml(self):
        """YAML'dan config yüklenmeli"""
        from src.utils.config_validation import ProjectConfig
        
        config = ProjectConfig.from_yaml('configs/config.yaml')
        
        assert config is not None
        assert config.dataset.path == 'NeAR_dataset/ALAN'
        assert config.training.epochs > 0
        assert config.model.num_classes == 2
    
    @pytest.mark.unit
    def test_invalid_batch_size(self):
        """Invalid batch size başarısız olmalı"""
        from src.utils.config_validation import TrainingConfig
        
        with pytest.raises(ValueError):
            config = TrainingConfig(batch_size=2048)  # Too large
    
    @pytest.mark.unit
    def test_invalid_epochs(self):
        """Invalid epochs başarısız olmalı"""
        from src.utils.config_validation import TrainingConfig
        
        with pytest.raises(ValueError):
            config = TrainingConfig(epochs=0)  # Must be >= 1
    
    @pytest.mark.unit
    def test_invalid_learning_rate(self):
        """Invalid learning rate başarısız olmalı"""
        from src.utils.config_validation import TrainingConfig
        
        with pytest.raises(ValueError):
            config = TrainingConfig(learning_rate=-0.001)  # Must be > 0
    
    @pytest.mark.unit
    def test_invalid_device(self):
        """Invalid device başarısız olmalı"""
        from src.utils.config_validation import ProjectConfig, DatasetConfig, TrainingConfig, ModelConfig
        
        with pytest.raises(ValueError):
            config = ProjectConfig(
                dataset=DatasetConfig(path='NeAR_dataset/ALAN'),
                training=TrainingConfig(),
                model=ModelConfig(),
                device='gpu'  # Invalid
            )
    
    @pytest.mark.unit
    def test_invalid_loss_type(self):
        """Invalid loss type başarısız olmalı"""
        from src.utils.config_validation import TrainingConfig
        
        with pytest.raises(ValueError):
            config = TrainingConfig(loss_type='invalid_loss')
    
    @pytest.mark.unit
    def test_invalid_optimizer(self):
        """Invalid optimizer başarısız olmalı"""
        from src.utils.config_validation import TrainingConfig
        
        with pytest.raises(ValueError):
            config = TrainingConfig(optimizer='invalid_optimizer')
    
    @pytest.mark.unit
    def test_invalid_model_type(self):
        """Invalid model type başarısız olmalı"""
        from src.utils.config_validation import ModelConfig
        
        with pytest.raises(ValueError):
            config = ModelConfig(model_type='invalid_model')
    
    @pytest.mark.unit
    def test_config_to_dict(self):
        """Config dict'e dönüştürülebilmeli"""
        from src.utils.config_validation import ProjectConfig, DatasetConfig, TrainingConfig, ModelConfig
        
        config = ProjectConfig(
            dataset=DatasetConfig(path='NeAR_dataset/ALAN'),
            training=TrainingConfig(),
            model=ModelConfig(),
            device='cpu'
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'dataset' in config_dict
        assert 'training' in config_dict
        assert 'model' in config_dict


class TestHelpers:
    """Helper function testleri"""
    
    @pytest.mark.unit
    def test_load_config(self):
        """Config helper load etmeli"""
        from src.utils.helpers import load_config
        
        config = load_config('configs/config.yaml')
        
        assert config is not None
        assert 'dataset' in config
        assert 'training' in config
        assert 'model' in config
    
    @pytest.mark.unit
    def test_set_seed(self):
        """Seed ayarlanabilmeli"""
        from src.utils.helpers import set_seed
        
        set_seed(42)
        
        # Create two random tensors with same seed
        x1 = torch.randn(10)
        
        set_seed(42)
        x2 = torch.randn(10)
        
        assert torch.allclose(x1, x2)
    
    @pytest.mark.unit
    def test_get_device(self):
        """Device seçilebilmeli"""
        from src.utils.helpers import get_device
        
        device_cpu = get_device(prefer_cuda=False)
        assert device_cpu == 'cpu'
        
        device = get_device(prefer_cuda=True)
        assert device in ['cuda', 'cpu']  # cpu if cuda not available


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'not slow'])
