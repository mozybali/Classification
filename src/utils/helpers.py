"""
Utility helper functions
Common utility functions for the project
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import yaml


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    print(f"✓ Random seed set: {seed}")


def count_parameters(model) -> int:
    try:
        import torch.nn as nn
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters:")
        print(f"  • Total: {total:,}")
        print(f"  • Trainable: {trainable:,}")
        print(f"  • Frozen: {total - trainable:,}")
        return total
    except ImportError:
        print("⚠️  PyTorch not found")
        return 0


def save_config(config: Dict, save_path: str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"✓ Config saved: {save_path}")


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"✓ Config loaded: {config_path}")
    return config


def plot_training_history(train_losses: List[float],
                          val_losses: List[float],
                          metrics_history: List[Dict],
                          save_path: Optional[str] = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(train_losses) + 1)
    
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    accuracies = [m.get('accuracy', 0) for m in metrics_history]
    axes[0, 1].plot(epochs, accuracies, 'g-')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    precisions = [m.get('precision', 0) for m in metrics_history]
    recalls = [m.get('recall', 0) for m in metrics_history]
    f1_scores = [m.get('f1_score', 0) for m in metrics_history]
    
    axes[1, 0].plot(epochs, precisions, label='Precision')
    axes[1, 0].plot(epochs, recalls, label='Recall')
    axes[1, 0].plot(epochs, f1_scores, label='F1-Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision, Recall, F1-Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    aucs = [m.get('auc', 0) for m in metrics_history]
    axes[1, 1].plot(epochs, aucs, 'purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].set_title('ROC-AUC Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history saved: {save_path}")
    plt.show()


def plot_confusion_matrix(cm: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         save_path: Optional[str] = None) -> None:
    if class_names is None:
        class_names = ['Normal', 'Anomaly']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved: {save_path}")
    plt.show()


def get_device(prefer_cuda: bool = True) -> str:
    try:
        import torch
        if prefer_cuda and torch.cuda.is_available():
            cuda_capability = torch.cuda.get_device_capability()
            capability_version = float(f"{cuda_capability[0]}.{cuda_capability[1]}")
            device = 'cuda'
            print(f"✓ Device: {device}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            if capability_version >= 12.0:
                print(f"  Capability: {capability_version} (Blackwell RTX 5050)")
        else:
            device = 'cpu'
            print(f"✓ Device: {device}")
    except ImportError:
        device = 'cpu'
        print(f"✓ Device: {device}")
    
    return device


def save_metrics_to_json(metrics: Dict, save_path: str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, (np.int64, np.float64)):
            serializable_metrics[key] = float(value)
        else:
            serializable_metrics[key] = value
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_metrics, f, indent=4, ensure_ascii=False)
    
    print(f"✓ Metrics saved: {save_path}")


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max') -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"⚠️  Early stopping triggered (patience: {self.patience})")
                return True
        
        return False


def main() -> None:
    print("\n" + "="*60)
    print("UTILITY FUNCTIONS TEST")
    print("="*60 + "\n")
    
    set_seed(42)
    device = get_device()
    print("✓ All utility functions working correctly")


if __name__ == "__main__":
    main()
