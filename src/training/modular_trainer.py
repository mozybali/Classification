"""
Modular Trainer - Esnek ve genişletilebilir training pipeline
CNN ve GNN modelleri için unified training interface
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import amp
from pathlib import Path
import time
from typing import Dict, Optional, Callable
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)
import json


class ModularTrainer:
    """Modüler trainer - tüm model tipleri için"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Dict,
        device: str = 'cuda'
    ):
        """
        Args:
            model: Training için model
            train_loader: Training dataloader
            val_loader: Validation dataloader (opsiyonel)
            config: Training konfigürasyonu
            device: 'cuda' veya 'cpu'
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # RTX 5050 sm_120 kontrolü ile device seçimi
        if device == 'cuda' and torch.cuda.is_available():
            cuda_capability = torch.cuda.get_device_capability()
            capability_version = float(f"{cuda_capability[0]}.{cuda_capability[1]}")
            
            # sm_120 (12.0) veya daha yeni - arch_list'te destekleniyorsa kullan
            if capability_version >= 12.0:
                arch_list = torch.cuda.get_arch_list()
                # Format: sm_120 not sm_1200 (major=12, minor=0 -> sm_120)
                if cuda_capability[1] == 0:
                    sm_arch = f"sm_{cuda_capability[0]}{cuda_capability[1]}"
                else:
                    sm_arch = f"sm_{cuda_capability[0]}{cuda_capability[1]:02d}"
                
                if sm_arch in arch_list:
                    self.device = device
                else:
                    import warnings
                    warnings.warn(
                        f"⚠️  GPU {sm_arch} not supported, using CPU",
                        UserWarning
                    )
                    self.device = 'cpu'
            else:
                self.device = device
        else:
            self.device = 'cpu'
        
        # Model'i device'a taşı
        self.model.to(self.device)
        
        # Training components
        self.optimizer = self._create_optimizer()
        self.criterion = self._create_criterion()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision - sadece desteklenen GPU'larda
        self.use_amp = config.get('use_amp', True) and self.device == 'cuda'
        self.scaler = amp.GradScaler('cuda') if self.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_metric = float("-inf")
        self.train_history = {'loss': [], 'acc': []}
        self.val_history = {'loss': [], 'acc': [], 'precision': [], 'recall': [], 'f1': [], 'f_beta': [], 'auc': []}
        # Backward-compat aliases used by main.py/helpers.plot_training_history
        self.train_losses = self.train_history['loss']
        self.val_losses = self.val_history['loss']
        self.metrics_history = []
        
        # Early stopping
        self.early_stopping_patience = config.get('early_stopping', {}).get('patience', 15)
        self.early_stopping_counter = 0
        self.min_delta = config.get('early_stopping', {}).get('min_delta', 0.001)
        self.best_metric_name = config.get('best_metric', 'f1')
        self.best_metric_beta = config.get('best_metric_beta', 2.0)
        
        # Save directory
        self.save_dir = Path(config.get('save_dir', 'checkpoints'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self._print_trainer_info()
    
    def _print_trainer_info(self):
        """Trainer bilgilerini yazdır"""
        print(f"\n{'='*70}")
        print(f"MODULAR TRAINER")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Optimizer: {type(self.optimizer).__name__}")
        print(f"Scheduler: {type(self.scheduler).__name__ if self.scheduler else 'None'}")
        print(f"Criterion: {type(self.criterion).__name__}")
        print(f"Save Directory: {self.save_dir}")
        print(f"Early Stopping: {self.early_stopping_patience} epochs")
        print(f"{'='*70}\n")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Optimizer oluştur"""
        optimizer_type = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, 
                           momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Bilinmeyen optimizer: {optimizer_type}")
    
    def _create_criterion(self) -> nn.Module:
        """Loss function oluştur"""
        loss_type = self.config.get('loss_type', 'cross_entropy').lower()
        
        if loss_type == 'cross_entropy':
            class_weights = self.config.get('class_weights')
            if class_weights:
                weights = torch.tensor(list(class_weights.values()), dtype=torch.float32)
                weights = weights.to(self.device)
                return nn.CrossEntropyLoss(weight=weights)
            return nn.CrossEntropyLoss()
        
        elif loss_type == 'focal':
            return FocalLoss(alpha=self.config.get('focal_alpha', 1.0),
                           gamma=self.config.get('focal_gamma', 2.0))
        
        elif loss_type == 'bce':
            return nn.BCEWithLogitsLoss()
        
        else:
            raise ValueError(f"Bilinmeyen loss type: {loss_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Learning rate scheduler oluştur"""
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type is None or scheduler_type == 'none':
            return None
        
        if scheduler_type == 'step':
            step_size = self.config.get('step_size', 10)
            gamma = self.config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        
        elif scheduler_type == 'cosine':
            epochs = self.config.get('epochs', 100)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=5, factor=0.5
            )
        
        else:
            raise ValueError(f"Bilinmeyen scheduler: {scheduler_type}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Bir epoch training"""
        self.model.train()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass (with mixed precision)
            if self.use_amp:
                with amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Metrics
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Calculate metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return {'loss': epoch_loss, 'acc': epoch_acc}
    
    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """Validation"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            if self.use_amp:
                with amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Metrics
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Positive class probability
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Calculate metrics
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        # Precision, recall, F1
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='binary', zero_division=0, pos_label=1
            )
        except Exception:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='weighted', zero_division=0
            )

        # F-beta (recall-weighted)
        beta = float(self.best_metric_beta) if self.best_metric_beta is not None else 2.0
        denom = (beta ** 2 * precision + recall)
        f_beta = (1 + beta ** 2) * precision * recall / denom if denom > 0 else 0.0
        
        # AUC
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        return {
            'loss': epoch_loss,
            'acc': epoch_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f_beta': f_beta,
            'auc': auc
        }
    
    def train(self, num_epochs: int):
        """Complete training loop"""
        print(f"\n🚀 Training başlıyor: {num_epochs} epochs\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.train_history['loss'].append(train_metrics['loss'])
            self.train_history['acc'].append(train_metrics['acc'])
            
            # Validate
            val_metrics = self.validate_epoch()
            if val_metrics:
                self.val_history['loss'].append(val_metrics['loss'])
                self.val_history['acc'].append(val_metrics['acc'])
                self.val_history['precision'].append(val_metrics['precision'])
                self.val_history['recall'].append(val_metrics['recall'])
                self.val_history['f1'].append(val_metrics['f1'])
                self.val_history['f_beta'].append(val_metrics.get('f_beta', 0.0))
                self.val_history['auc'].append(val_metrics['auc'])
                self.metrics_history.append({
                    'accuracy': val_metrics['acc'],
                    'precision': val_metrics['precision'],
                    'recall': val_metrics['recall'],
                    'f1_score': val_metrics['f1'],
                    'auc': val_metrics['auc']
                })
            
            # Learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('acc', train_metrics['acc']))
                else:
                    self.scheduler.step()
            
            # Print metrics
            self._print_epoch_metrics(train_metrics, val_metrics)
            
            # Save checkpoint
            if val_metrics:
                metric_value = self._get_metric_value(val_metrics)
            else:
                metric_value = train_metrics.get('acc', 0.0)

            if val_metrics and metric_value > self.best_val_metric:
                self.best_val_metric = metric_value
                self.save_checkpoint('best_model.pth')
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
            
            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"\n⏹️  Early stopping at epoch {epoch + 1}")
                break
        
        print(f"\n✅ Training tamamlandı!")
        self.save_training_history()
    
    def _print_epoch_metrics(self, train_metrics: Dict, val_metrics: Dict):
        """Epoch metrikleri yazdır"""
        print(f"\nEpoch {self.current_epoch + 1}:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}")
        
        if val_metrics:
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['acc']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, "
                  f"AUC: {val_metrics['auc']:.4f}")
    
    def _get_metric_value(self, val_metrics: Dict) -> float:
        """Select metric value for best checkpoint selection."""
        name = (self.best_metric_name or 'f1').lower()
        mapping = {
            'acc': 'acc',
            'accuracy': 'acc',
            'auc': 'auc',
            'auc_roc': 'auc',
            'f1': 'f1',
            'precision': 'precision',
            'recall': 'recall',
            'f_beta': 'f_beta'
        }
        key = mapping.get(name, name)
        return float(val_metrics.get(key, val_metrics.get('f1', 0.0)))


    def save_checkpoint(self, filename: str):
        """Checkpoint kaydet"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        print(f"  💾 Checkpoint kaydedildi: {save_path}")
    
    def save_training_history(self):
        """Training history'yi JSON olarak kaydet"""
        history = {
            'train': self.train_history,
            'val': self.val_history
        }
        
        save_path = self.save_dir / 'training_history.json'
        with open(save_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"  📊 Training history kaydedildi: {save_path}")


class FocalLoss(nn.Module):
    """Focal Loss - class imbalance için"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
