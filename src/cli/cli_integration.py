"""
CLI Integration - Subprocess calls'ı direct imports ile değiştir
"""

import sys
from pathlib import Path

# Ana dizini path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.error_handling import log_exceptions, ProjectLogger


logger = ProjectLogger().get_logger('cli_integration')


@log_exceptions('training')
def run_training_direct(config_path: str = 'configs/config.yaml'):
    """Training'i doğrudan çalıştır (subprocess değil)"""
    from src.models import ModelFactory
    from src.training import ModularTrainer
    from src.preprocessing.preprocess import DataPreprocessor
    from src.utils.helpers import load_config
    import torch
    
    logger.info(f"Training başlatılıyor: {config_path}")
    
    # Config yükle
    config = load_config(config_path)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    # Model
    model_config = config['model']
    logger.info(f"Model oluşturuluyor: {model_config['model_type']}")
    model = ModelFactory.create_model(model_config)
    
    # Data loaders
    logger.info("Data loaders oluşturuluyor...")
    preprocessor = DataPreprocessor(config)
    dataloaders = preprocessor.get_dataloaders(
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    
    # Trainer
    trainer = ModularTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders.get('dev'),
        config=config['training'],
        device=device
    )
    
    # Training
    num_epochs = config['training']['epochs']
    trainer.train(num_epochs)
    
    logger.info("Training tamamlandı!")
    return trainer


@log_exceptions('evaluation')
def run_evaluation_direct(
    checkpoint_path: str,
    config_path: str = 'configs/config.yaml',
    test_set: str = 'test'
):
    """Evaluation'ı doğrudan çalıştır"""
    from src.models import ModelFactory, load_model_from_checkpoint
    from src.preprocessing.preprocess import DataPreprocessor
    from src.training.evaluator import ModelEvaluator
    from src.utils.helpers import load_config
    import torch
    
    logger.info(f"Evaluation başlatılıyor: {checkpoint_path}")
    
    # Config
    config = load_config(config_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model yükle
    model, metadata = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config['model'],
        device=device
    )
    logger.info(f"Model yüklendi. Epoch: {metadata.get('epoch', 'unknown')}")
    
    # Data loader
    preprocessor = DataPreprocessor(config)
    dataloaders = preprocessor.get_dataloaders(
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    
    test_loader = dataloaders.get(test_set)
    if test_loader is None:
        raise ValueError(f"Test set '{test_set}' bulunamadı")
    
    # Evaluate
    evaluator = ModelEvaluator(model, device)

    eval_cfg = config.get('evaluation', {})
    thr_cfg = eval_cfg.get('threshold', {})
    threshold_strategy = thr_cfg.get('strategy', 'fixed')
    threshold = float(thr_cfg.get('default', 0.5))
    beta = float(thr_cfg.get('beta', 2.0))
    min_precision = thr_cfg.get('min_precision')

    selection_set = thr_cfg.get('selection_set')
    if not selection_set:
        if test_set == 'test' and 'dev' in dataloaders:
            selection_set = 'dev'
        else:
            selection_set = test_set
    if selection_set in ['val', 'validation']:
        selection_set = 'dev'
    threshold_loader = dataloaders.get(selection_set)
    if threshold_loader is None:
        selection_set = None

    results = evaluator.evaluate(
        test_loader=test_loader,
        save_dir=f'outputs/evaluation_{test_set}',
        threshold_strategy=threshold_strategy,
        threshold=threshold,
        beta=beta,
        min_precision=min_precision,
        threshold_loader=threshold_loader,
        threshold_selection=selection_set
    )
    
    logger.info("Evaluation tamamlandı!")
    return results


@log_exceptions('hyperparameter_optimization')
def run_hpo_direct(config_path: str = 'configs/config.yaml', n_trials: int = 50):
    """Hiperparametre optimizasyonu doğrudan çalıştır"""
    from src.training.hyperparameter_optimizer import HyperparameterOptimizer
    from src.utils.helpers import load_config
    import torch
    
    logger.info(f"HPO başlatılıyor: {n_trials} trials")
    
    config = load_config(config_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    optimizer = HyperparameterOptimizer(config, device=device)
    study = optimizer.optimize(n_trials=n_trials, sampler='bayesian')
    
    logger.info(f"Best params: {study.best_params}")
    return study


@log_exceptions('cross_validation')
def run_cv_direct(config_path: str = 'configs/config.yaml', n_splits: int = 5):
    """Cross-validation doğrudan çalıştır"""
    from src.training.cross_validator_fixed import CrossValidator
    from src.preprocessing.preprocess import DataPreprocessor
    from src.models import ModelFactory
    from src.utils.helpers import load_config
    import torch
    
    logger.info(f"Cross-validation başlatılıyor: {n_splits}-fold")
    
    config = load_config(config_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Dataset
    preprocessor = DataPreprocessor(config)
    from src.preprocessing.preprocess import ALANDataset
    dataset = ALANDataset(
        csv_path=str(Path(config['dataset']['path']) / config['dataset']['csv_file']),
        zip_path=str(Path(config['dataset']['path']) / config['dataset']['zip_file']),
        subset='train',
        load_images=True
    )
    
    # CV
    cv = CrossValidator(n_splits=n_splits)
    results = cv.train_and_evaluate(
        dataset=dataset,
        model_class=ModelFactory,
        model_config=config['model'],
        training_config=config['training'],
        device=device
    )
    
    logger.info(f"CV Mean F1: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
    return results


# NEW: Updated main.py integration
def get_menu_action_updated(choice: str, config: dict):
    """
    Updated menu action - subprocess çağrılarını doğrudan import'lara çevir
    """
    
    actions = {
        # Training
        '7': lambda: run_training_direct('configs/config.yaml'),
        
        # Evaluation
        '8': lambda: run_evaluation_direct(
            input("Checkpoint path: ").strip(),
            config_path='configs/config.yaml'
        ),
        
        # HPO
        'H': lambda: run_hpo_direct(n_trials=50),
        
        # Cross-validation
        'CV': lambda: run_cv_direct(n_splits=5),
    }
    
    action = actions.get(choice)
    if action:
        try:
            result = action()
            logger.info(f"Action '{choice}' başarıyla tamamlandı")
            return result
        except Exception as e:
            logger.error(f"Action '{choice}' hatası: {e}")
            raise
    else:
        logger.warning(f"Unknown action: {choice}")
        return None


if __name__ == '__main__':
    print("✅ CLI Integration setup başarıyla!")
    print("Now use run_training_direct(), run_evaluation_direct(), etc.")
