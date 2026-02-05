"""
Evaluation Runner - Main script for model evaluation
Trained modelleri test etmek için
"""

import argparse
import sys
import yaml
import torch
from pathlib import Path

# Ana dizini path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))


def _configure_console() -> None:
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


_configure_console()

from src.models import ModelFactory, load_model_from_checkpoint
from src.preprocessing.preprocess import DataPreprocessor
from src.training.evaluator import ModelEvaluator
from src.utils.visualization import TrainingVisualizer, create_evaluation_report


def run_evaluation(
    checkpoint_path: str,
    config_path: str = 'configs/config.yaml',
    save_dir: str = 'evaluation_results',
    test_set: str = 'test'
):
    """
    Model evaluation runner
    
    Args:
        checkpoint_path: Model checkpoint path
        config_path: Config file path
        save_dir: Results save directory
        test_set: Test set name ('test' or 'dev')
    """
    print(f"\n{'='*70}")
    print("MODEL EVALUATION RUNNER")
    print(f"{'='*70}\n")
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test Set: {test_set}")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Load model
    print("📦 Loading model...")
    model = load_model_from_checkpoint(checkpoint_path, config['model'], device)
    
    # Prepare test data
    print("\n📊 Preparing test data...")
    preprocessor = DataPreprocessor(config)
    loaders = preprocessor.get_dataloaders(
        batch_size=config['training'].get('batch_size', 8),
        num_workers=config['training'].get('num_workers', 4)
    )
    
    test_loader = loaders.get(test_set)
    if test_loader is None:
        print(f"❌ Test set '{test_set}' not found!")
        return
    
    # Evaluate
    print(f"\n🔍 Running evaluation...")
    evaluator = ModelEvaluator(model, device)

    eval_cfg = config.get('evaluation', {})
    thr_cfg = eval_cfg.get('threshold', {})
    threshold_strategy = thr_cfg.get('strategy', 'fixed')
    threshold = float(thr_cfg.get('default', 0.5))
    beta = float(thr_cfg.get('beta', 2.0))
    min_precision = thr_cfg.get('min_precision')

    selection_set = thr_cfg.get('selection_set')
    if not selection_set:
        if test_set == 'test' and 'dev' in loaders:
            selection_set = 'dev'
        else:
            selection_set = test_set
    if selection_set in ['val', 'validation']:
        selection_set = 'dev'
    threshold_loader = loaders.get(selection_set)
    if threshold_loader is None:
        selection_set = None

    results = evaluator.evaluate(
        test_loader,
        save_dir=save_dir,
        threshold_strategy=threshold_strategy,
        threshold=threshold,
        beta=beta,
        min_precision=min_precision,
        threshold_loader=threshold_loader,
        threshold_selection=selection_set
    )
    
    # Create comprehensive report
    print(f"\n📄 Creating comprehensive report...")
    report_path = Path(save_dir) / 'evaluation_report.pdf'
    create_evaluation_report(save_dir, str(report_path))
    
    print(f"\n✅ Evaluation completed!")
    print(f"   Results saved to: {save_dir}")
    print(f"   Report: {report_path}")
    
    return results


def compare_models(
    checkpoints: dict,
    config_path: str = 'configs/config.yaml',
    save_dir: str = 'comparison_results',
    test_set: str = 'test'
):
    """
    Multiple models karşılaştır
    
    Args:
        checkpoints: Dict of {model_name: checkpoint_path}
        config_path: Config file
        save_dir: Results directory
        test_set: Test set name
    """
    print(f"\n{'='*70}")
    print("MODEL COMPARISON RUNNER")
    print(f"{'='*70}\n")
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load models
    models = {}
    print("📦 Loading models...")
    for name, checkpoint in checkpoints.items():
        print(f"  - {name}: {checkpoint}")
        model = load_model_from_checkpoint(checkpoint, config['model'], device)
        models[name] = model
    
    # Prepare test data
    print("\n📊 Preparing test data...")
    preprocessor = DataPreprocessor(config)
    loaders = preprocessor.get_dataloaders(
        batch_size=config['training'].get('batch_size', 8),
        num_workers=config['training'].get('num_workers', 4)
    )
    
    test_loader = loaders.get(test_set)
    if test_loader is None:
        print(f"❌ Test set '{test_set}' not found!")
        return
    
    # Compare models
    print(f"\n🔍 Comparing models...")
    evaluator = ModelEvaluator(models[list(models.keys())[0]], device)
    comparison_results = evaluator.compare_models(models, test_loader, save_dir)
    
    print(f"\n✅ Comparison completed!")
    print(f"   Results saved to: {save_dir}")
    
    return comparison_results


def main():
    parser = argparse.ArgumentParser(description='Model Evaluation Runner')
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate single model')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                            help='Model checkpoint path')
    eval_parser.add_argument('--config', type=str, default='configs/config.yaml',
                            help='Config file path')
    eval_parser.add_argument('--save-dir', type=str, default='evaluation_results',
                            help='Results save directory')
    eval_parser.add_argument('--test-set', type=str, default='test',
                            choices=['test', 'dev'],
                            help='Test set to use')
    
    # Compare command
    comp_parser = subparsers.add_parser('compare', help='Compare multiple models')
    comp_parser.add_argument('--checkpoints', nargs='+', required=True,
                            help='Model checkpoint paths (format: name=path)')
    comp_parser.add_argument('--config', type=str, default='configs/config.yaml',
                            help='Config file path')
    comp_parser.add_argument('--save-dir', type=str, default='comparison_results',
                            help='Results save directory')
    comp_parser.add_argument('--test-set', type=str, default='test',
                            choices=['test', 'dev'],
                            help='Test set to use')
    
    args = parser.parse_args()
    
    if args.command == 'evaluate':
        run_evaluation(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            save_dir=args.save_dir,
            test_set=args.test_set
        )
    
    elif args.command == 'compare':
        # Parse checkpoints
        checkpoints = {}
        for cp in args.checkpoints:
            if '=' in cp:
                name, path = cp.split('=', 1)
                checkpoints[name] = path
            else:
                # Auto-name
                name = Path(cp).stem
                checkpoints[name] = cp
        
        compare_models(
            checkpoints=checkpoints,
            config_path=args.config,
            save_dir=args.save_dir,
            test_set=args.test_set
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
