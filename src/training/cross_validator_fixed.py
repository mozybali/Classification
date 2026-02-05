"""
Cross-Validator - Tamamlanmış K-Fold Cross-Validation
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import pandas as pd


class CrossValidator:
    """K-Fold Cross-Validation sistemi"""
    
    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        stratified: bool = True
    ):
        """
        Args:
            n_splits: Fold sayısı
            random_state: Random seed
            stratified: Sınıf dağılımını koru
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.stratified = stratified
        
        if stratified:
            self.splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state
            )
        else:
            from sklearn.model_selection import KFold
            self.splitter = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state
            )
    
    def get_splits(
        self,
        dataset,
        labels: Optional[np.ndarray] = None
    ) -> List[Tuple[DataLoader, DataLoader]]:
        """
        Dataset'i fold'lara böl
        
        Args:
            dataset: PyTorch Dataset
            labels: Sınıf etikerleri (stratified için gerekli)
            
        Returns:
            List of (train_loader, val_loader) tuples
        """
        n_samples = len(dataset)
        indices = np.arange(n_samples)
        
        # Labels'ı al (stratified için)
        if labels is None and self.stratified:
            try:
                # Varsayılan: dataset['label'] field'ından al
                labels = np.array([
                    dataset[i]['label'] for i in range(len(dataset))
                ])
            except:
                print("⚠️  Labels alınamadı, non-stratified CV kullan")
                self.stratified = False
        
        splits = []
        for fold_idx, (train_idx, val_idx) in enumerate(
            self.splitter.split(indices, labels if self.stratified else None)
        ):
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            splits.append((train_subset, val_subset))
            print(f"Fold {fold_idx + 1}: Train={len(train_idx)}, Val={len(val_idx)}")
        
        return splits
    
    def train_and_evaluate(
        self,
        dataset,
        model_class,
        model_config: Dict,
        training_config: Dict,
        labels: Optional[np.ndarray] = None,
        device: str = 'cuda'
    ) -> Dict:
        """
        Tüm fold'larda train ve evaluate et
        
        Args:
            dataset: Full dataset
            model_class: Model sınıfı
            model_config: Model konfigürasyonu
            training_config: Training konfigürasyonu
            labels: Sınıf etikerleri
            device: Training device
            
        Returns:
            Cross-validation sonuçları
        """
        from src.models import ModelFactory
        from src.training import ModularTrainer
        from src.training.evaluator import ModelEvaluator
        
        splits = self.get_splits(dataset, labels)
        cv_results = {
            'fold_scores': [],
            'fold_metrics': [],
            'mean_score': 0.0,
            'std_score': 0.0
        }
        
        print(f"\n{'='*70}")
        print(f"CROSS-VALIDATION ({self.n_splits}-Fold)")
        print(f"{'='*70}\n")
        
        for fold_idx, (train_subset, val_subset) in enumerate(splits):
            print(f"\n🔄 FOLD {fold_idx + 1}/{self.n_splits}")
            print(f"{'='*70}")
            
            # DataLoaders
            train_loader = DataLoader(
                train_subset,
                batch_size=training_config['batch_size'],
                num_workers=training_config['num_workers'],
                shuffle=True
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=training_config['batch_size'],
                num_workers=training_config['num_workers']
            )
            
            # Model
            model = ModelFactory.create_model(model_config)
            
            # Trainer
            trainer = ModularTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=training_config,
                device=device
            )
            
            # Train
            epochs = training_config.get('cv_epochs', 20)
            trainer.train(epochs)
            
            # Evaluate
            evaluator = ModelEvaluator(model, device)
            results = evaluator.evaluate(val_loader, save_dir=None)
            
            fold_f1 = results['metrics'].get('f1_weighted', 0)
            cv_results['fold_scores'].append(fold_f1)
            cv_results['fold_metrics'].append(results['metrics'])
            
            print(f"  F1-Score: {fold_f1:.4f}")
        
        # Summary
        cv_results['mean_score'] = np.mean(cv_results['fold_scores'])
        cv_results['std_score'] = np.std(cv_results['fold_scores'])
        
        print(f"\n{'='*70}")
        print(f"CV RESULTS")
        print(f"{'='*70}")
        print(f"Mean F1-Score: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        print(f"Fold Scores: {[f'{s:.4f}' for s in cv_results['fold_scores']]}")
        
        return cv_results


class SimpleCV:
    """Simplified CV helper (backward compat)"""
    
    @staticmethod
    def quick_cv(
        dataset,
        model_class,
        model_config: Dict,
        training_config: Dict,
        n_splits: int = 5,
        device: str = 'cuda'
    ) -> Dict:
        """Quick cross-validation"""
        validator = CrossValidator(n_splits=n_splits)
        
        return validator.train_and_evaluate(
            dataset=dataset,
            model_class=model_class,
            model_config=model_config,
            training_config=training_config,
            device=device
        )


if __name__ == '__main__':
    print("✅ Cross-Validator setup başarıyla!")
