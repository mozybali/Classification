"""
Training Runner - Modular training pipeline'ı başlatır
Config-driven training için ana script
"""

import sys
import yaml
import torch
from pathlib import Path

# Ana dizini path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ModelFactory
from src.training import ModularTrainer
from src.preprocessing.preprocess import DataPreprocessor


def run_training(config_path: str = 'configs/config.yaml'):
    """
    Training'i başlatır
    
    Args:
        config_path: Config dosyası yolu
    """
    # Config yükle
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("="*70)
    print("MODULAR TRAINING PIPELINE")
    print("="*70)
    print(f"Config: {config_path}\n")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Model oluştur
    model_config = config['model']
    print(f"\n📦 Model oluşturuluyor: {model_config['model_type']}")
    model = ModelFactory.create_model(model_config)
    
    # Data loaders oluştur
    print(f"\n📊 Data loaders oluşturuluyor...")
    preprocessor = DataPreprocessor(config)
    
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    
    dataloaders = preprocessor.get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders.get('dev')  # Validation loader
    
    # Trainer oluştur
    print(f"\n🎯 Trainer oluşturuluyor...")
    trainer = ModularTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        device=device
    )
    
    # Training başlat
    num_epochs = config['training']['epochs']
    trainer.train(num_epochs)
    
    print(f"\n✅ Training tamamlandı!")
    print(f"Checkpoints: {trainer.save_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Modular Training Pipeline')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Config dosyası yolu')
    parser.add_argument('--model', type=str, default=None,
                       help='Model tipi (config override)')
    
    args = parser.parse_args()
    
    # Config override (eğer command line'dan model belirtilmişse)
    if args.model:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        config['model']['model_type'] = args.model
        
        # Temp config kaydet
        temp_config = 'configs/temp_config.yaml'
        with open(temp_config, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        run_training(temp_config)
    else:
        run_training(args.config)
