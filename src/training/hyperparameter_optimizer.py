"""
Hiperparametre Optimizasyon Modülü
Grid Search ve Bayesian Optimization (Optuna) destekli
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import itertools
from tqdm import tqdm

from .modular_trainer import ModularTrainer
from ..models.model_factory import ModelFactory


class HyperparameterOptimizer:
    """Hiperparametre optimizasyon sınıfı"""
    
    def __init__(self,
                 base_config: Dict,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = 'cuda'):
        """
        Args:
            base_config: Temel konfigürasyon
            train_loader: Eğitim veri yükleyici
            val_loader: Validasyon veri yükleyici
            device: 'cuda' veya 'cpu'
        """
        self.base_config = base_config.copy()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.results = []
        self.best_params = None
        self.best_score = 0.0
        
        # Sonuç kaydetme dizini
        self.output_dir = Path("outputs/hyperparameter_optimization")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Hiperparametre Optimizer oluşturuldu")
        print(f"  - Device: {self.device}")
        print(f"  - Output: {self.output_dir}")
    
    def grid_search(self,
                   param_grid: Dict[str, List[Any]],
                   metric: str = 'accuracy',
                   num_epochs: int = 10,
                   save_all_models: bool = False) -> Dict:
        """
        Grid Search ile hiperparametre optimizasyonu
        
        Args:
            param_grid: Aranacak parametre grid'i
                Örnek: {
                    'learning_rate': [1e-3, 1e-4, 1e-5],
                    'batch_size': [16, 32],
                    'optimizer': ['adam', 'adamw']
                }
            metric: Optimize edilecek metrik ('accuracy', 'f1', 'auc')
            num_epochs: Her kombinasyon için epoch sayısı
            save_all_models: Tüm modelleri kaydet (çok yer kaplar!)
        
        Returns:
            En iyi parametreler ve sonuçlar
        """
        print("\n" + "="*70)
        print("GRID SEARCH HİPERPARAMETRE OPTİMİZASYONU")
        print("="*70)
        
        # Grid kombinasyonlarını oluştur
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        total_trials = len(combinations)
        print(f"\n📊 Toplam deneme sayısı: {total_trials}")
        print(f"📈 Her deneme: {num_epochs} epoch")
        print(f"🎯 Optimize edilecek metrik: {metric}")
        print(f"\n🔍 Aranacak parametreler:")
        for name, values in param_grid.items():
            print(f"  - {name}: {values}")
        
        # Tüm kombinasyonları dene
        print(f"\n{'='*70}")
        print("GRID SEARCH BAŞLADI")
        print(f"{'='*70}\n")
        
        for idx, combination in enumerate(combinations, 1):
            # Parametreleri hazırla
            params = dict(zip(param_names, combination))
            
            print(f"\n{'─'*70}")
            print(f"Deneme {idx}/{total_trials}")
            print(f"{'─'*70}")
            print("Parametreler:")
            for key, value in params.items():
                print(f"  {key}: {value}")
            print()
            
            # Modeli eğit ve değerlendir
            score, metrics = self._train_and_evaluate(
                params=params,
                num_epochs=num_epochs,
                metric=metric,
                trial_id=f"grid_{idx}",
                save_model=save_all_models
            )
            
            # Sonucu kaydet
            result = {
                'trial_id': idx,
                'params': params,
                'score': score,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            self.results.append(result)
            
            # En iyi sonucu güncelle
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                print(f"\n🎉 YENİ EN İYİ SONUÇ! {metric}: {score:.4f}")
            
            print(f"Mevcut Skor: {score:.4f} (En İyi: {self.best_score:.4f})")
        
        # Sonuçları kaydet
        self._save_results(method='grid_search')
        
        # Özet rapor
        self._print_summary()
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results
        }
    
    def bayesian_search(self,
                       param_distributions: Dict[str, Dict],
                       n_trials: int = 50,
                       metric: str = 'accuracy',
                       num_epochs: int = 10,
                       timeout: Optional[int] = None,
                       n_jobs: int = 1) -> Dict:
        """
        Bayesian Optimization (Optuna) ile hiperparametre optimizasyonu
        
        Args:
            param_distributions: Parametre dağılımları
                Örnek: {
                    'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
                    'batch_size': {'type': 'categorical', 'choices': [16, 32, 64]},
                    'dropout': {'type': 'float', 'low': 0.1, 'high': 0.5}
                }
            n_trials: Deneme sayısı
            metric: Optimize edilecek metrik
            num_epochs: Her deneme için epoch sayısı
            timeout: Maksimum süre (saniye)
            n_jobs: Paralel iş sayısı (1 = sıralı)
        
        Returns:
            En iyi parametreler ve sonuçlar
        """
        print("\n" + "="*70)
        print("BAYESIAN OPTIMIZATION (OPTUNA) HİPERPARAMETRE OPTİMİZASYONU")
        print("="*70)
        
        print(f"\n📊 Toplam deneme sayısı: {n_trials}")
        print(f"📈 Her deneme: {num_epochs} epoch")
        print(f"🎯 Optimize edilecek metrik: {metric}")
        print(f"⏱️  Timeout: {timeout if timeout else 'Yok'} saniye")
        print(f"\n🔍 Aranacak parametreler:")
        for name, dist in param_distributions.items():
            if dist['type'] == 'categorical':
                print(f"  - {name}: {dist['choices']}")
            elif dist['type'] == 'float':
                log_str = " (log scale)" if dist.get('log', False) else ""
                print(f"  - {name}: [{dist['low']}, {dist['high']}]{log_str}")
            elif dist['type'] == 'int':
                print(f"  - {name}: [{dist['low']}, {dist['high']}]")
        
        # Optuna study oluştur
        study_name = f"hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=TPESampler(seed=self.base_config.get('seed', 42)),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
        
        # Objective function
        def objective(trial: optuna.Trial) -> float:
            # Parametreleri sample et
            params = {}
            for name, dist in param_distributions.items():
                if dist['type'] == 'float':
                    params[name] = trial.suggest_float(
                        name, dist['low'], dist['high'],
                        log=dist.get('log', False)
                    )
                elif dist['type'] == 'int':
                    params[name] = trial.suggest_int(
                        name, dist['low'], dist['high']
                    )
                elif dist['type'] == 'categorical':
                    params[name] = trial.suggest_categorical(
                        name, dist['choices']
                    )
            
            # Modeli eğit ve değerlendir
            score, metrics = self._train_and_evaluate(
                params=params,
                num_epochs=num_epochs,
                metric=metric,
                trial_id=f"optuna_{trial.number}",
                trial=trial
            )
            
            # Sonucu kaydet
            result = {
                'trial_id': trial.number,
                'params': params,
                'score': score,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            self.results.append(result)
            
            return score
        
        # Optimizasyon çalıştır
        print(f"\n{'='*70}")
        print("BAYESIAN SEARCH BAŞLADI")
        print(f"{'='*70}\n")
        
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        # En iyi sonuçları al
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # Sonuçları kaydet
        self._save_results(method='bayesian_search')
        self._save_optuna_study(study)
        
        # Özet rapor
        self._print_summary()
        
        # Optuna görselleştirmeleri
        self._create_optuna_visualizations(study)
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'study': study,
            'all_results': self.results
        }
    
    def _train_and_evaluate(self,
                           params: Dict,
                           num_epochs: int,
                           trial_id: str,
                           metric: str = 'accuracy',
                           trial: Optional[optuna.Trial] = None,
                           save_model: bool = False) -> tuple:
        """
        Train and evaluate with given params.

        Returns:
            (best_score, best_metrics) tuple
        """
        from copy import deepcopy

        # Deep copy to avoid cross-trial mutation
        config = deepcopy(self.base_config)
        config.setdefault('training', {})
        config.setdefault('model', {})

        # Apply params into config
        for key, value in params.items():
            if key in ['learning_rate', 'weight_decay', 'optimizer', 'scheduler', 'momentum']:
                config['training'][key] = value
            elif key in ['batch_size', 'num_workers']:
                config['training'][key] = value
            elif key in ['dropout', 'base_filters']:
                config['model'][key] = value
            elif key == 'model_type':
                config['model']['model_type'] = value
            else:
                config['training'][key] = value

        config['training']['epochs'] = num_epochs

        # Recreate loaders if batch_size/num_workers changed
        train_loader = self.train_loader
        val_loader = self.val_loader
        if 'batch_size' in params or 'num_workers' in params:
            try:
                from ..preprocessing.dataloader_factory import DataLoaderFactory
                loaders = DataLoaderFactory.create_dataloaders(config)
                train_loader = loaders['train']
                val_loader = loaders['val']
            except Exception as e:
                print(f"⚠️  DataLoader recreation failed: {e}")

        # Build model
        try:
            model = ModelFactory.create_model(config['model'])
        except Exception as e:
            print(f"❌ Model creation failed: {e}")
            return 0.0, {}

        # Build trainer
        trainer = ModularTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config['training'],
            device=self.device
        )

        def get_score(metric_name: str, val_metrics: Dict, train_metrics: Dict) -> float:
            if not val_metrics:
                return float(train_metrics.get('acc', 0.0))
            name = metric_name.lower()
            if name in ['accuracy', 'acc']:
                return float(val_metrics.get('acc', 0.0))
            if name == 'f1':
                return float(val_metrics.get('f1', 0.0))
            if name == 'auc':
                return float(val_metrics.get('auc', 0.0))
            return float(val_metrics.get(name, 0.0))

        best_score = float('-inf')
        best_metrics: Dict = {}

        for epoch in range(num_epochs):
            trainer.current_epoch = epoch
            train_metrics = trainer.train_epoch()
            val_metrics = trainer.validate_epoch()

            score = get_score(metric, val_metrics, train_metrics)
            if score > best_score:
                best_score = score
                best_metrics = val_metrics or {}

            # Scheduler step
            if trainer.scheduler:
                if isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    trainer.scheduler.step(val_metrics.get('acc', train_metrics.get('acc', 0.0)))
                else:
                    trainer.scheduler.step()

            # Optuna pruning
            if trial is not None:
                trial.report(score, epoch)
                if trial.should_prune():
                    print(f"  ✂️ Trial pruned at epoch {epoch + 1}")
                    raise optuna.TrialPruned()

            # Progress
            if (epoch + 1) % max(1, num_epochs // 5) == 0:
                val_loss = val_metrics.get('loss', 0.0) if val_metrics else 0.0
                print(
                    f"  Epoch {epoch + 1}/{num_epochs} - "
                    f"Loss: {train_metrics.get('loss', 0.0):.4f}/{val_loss:.4f} - "
                    f"Score: {score:.4f}"
                )

        # Save model (optional)
        if save_model:
            model_path = self.output_dir / f"model_{trial_id}.pth"
            torch.save(model.state_dict(), model_path)

        return best_score, best_metrics

    def _save_results(self, method: str):
        """Sonuçları JSON dosyasına kaydet"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.output_dir / f"{method}_results_{timestamp}.json"
        
        results_data = {
            'method': method,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results,
            'timestamp': timestamp
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Sonuçlar kaydedildi: {results_file}")
    
    def _save_optuna_study(self, study: optuna.Study):
        """Optuna study'yi kaydet"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        study_file = self.output_dir / f"optuna_study_{timestamp}.pkl"
        
        import pickle
        with open(study_file, 'wb') as f:
            pickle.dump(study, f)
        
        print(f"💾 Optuna study kaydedildi: {study_file}")
    
    def _create_optuna_visualizations(self, study: optuna.Study):
        """Optuna görselleştirmelerini oluştur"""
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_slice
            )
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Optimization history
            fig1 = plot_optimization_history(study)
            fig1.write_html(
                str(self.output_dir / f"optimization_history_{timestamp}.html")
            )
            
            # Parameter importances
            fig2 = plot_param_importances(study)
            fig2.write_html(
                str(self.output_dir / f"param_importances_{timestamp}.html")
            )
            
            # Slice plot
            fig3 = plot_slice(study)
            fig3.write_html(
                str(self.output_dir / f"param_slice_{timestamp}.html")
            )
            
            print(f"📊 Görselleştirmeler oluşturuldu: {self.output_dir}")
            
        except Exception as e:
            print(f"⚠️  Görselleştirme hatası: {e}")
    
    def _print_summary(self):
        """Özet rapor yazdır"""
        print("\n" + "="*70)
        print("OPTİMİZASYON ÖZETİ")
        print("="*70)
        
        print(f"\n✅ Toplam deneme sayısı: {len(self.results)}")
        print(f"🏆 En iyi skor: {self.best_score:.4f}")
        print(f"\n📋 En iyi parametreler:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        
        # Top 5 sonuç
        print(f"\n🥇 Top 5 Sonuç:")
        sorted_results = sorted(self.results, key=lambda x: x['score'], reverse=True)
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"\n{i}. Skor: {result['score']:.4f}")
            print(f"   Parametreler: {result['params']}")
        
        print("\n" + "="*70)


def run_interactive_optimization():
    """İnteraktif hiperparametre optimizasyon menüsü"""
    from ..utils.helpers import load_config
    
    print("\n" + "="*70)
    print("HİPERPARAMETRE OPTİMİZASYONU")
    print("="*70)
    
    # Config yükle
    try:
        config = load_config()
        print("✓ Config yüklendi")
    except:
        print("❌ Config yüklenemedi, varsayılan ayarlar kullanılacak")
        config = {}
    
    # Optimizasyon yöntemi seç
    print("\n📊 Optimizasyon Yöntemini Seçin:")
    print("1. Grid Search (Tüm kombinasyonları dene)")
    print("2. Bayesian Search (Akıllı arama - Optuna)")
    print("0. Geri")
    
    choice = input("\nSeçiminiz (0-2): ").strip()
    
    if choice == '0':
        return
    
    # Optimize edilecek metrikleri seç
    print("\n🎯 Optimize Edilecek Metrik:")
    print("1. Accuracy")
    print("2. F1 Score")
    print("3. AUC")
    
    metric_choice = input("\nSeçiminiz (1-3): ").strip()
    metric_map = {'1': 'accuracy', '2': 'f1', '3': 'auc'}
    metric = metric_map.get(metric_choice, 'accuracy')
    
    # Epoch sayısı
    num_epochs = int(input("\n📈 Her deneme için epoch sayısı (örn: 10): ").strip() or "10")
    
    if choice == '1':
        # Grid Search
        print("\n" + "="*70)
        print("GRID SEARCH AYARLARI")
        print("="*70)
        
        # Parametreleri topla
        param_grid = {}
        
        # Learning rate
        print("\n📌 Learning Rate:")
        print("   Örnek: 0.001,0.0001,0.00001")
        lr_input = input("   Değerler (virgülle ayırın): ").strip()
        if lr_input:
            param_grid['learning_rate'] = [float(x) for x in lr_input.split(',')]
        
        # Batch size
        print("\n📌 Batch Size:")
        print("   Örnek: 16,32,64")
        bs_input = input("   Değerler (virgülle ayırın): ").strip()
        if bs_input:
            param_grid['batch_size'] = [int(x) for x in bs_input.split(',')]
        
        # Optimizer
        print("\n📌 Optimizer:")
        print("   Örnek: adam,adamw,sgd")
        opt_input = input("   Değerler (virgülle ayırın): ").strip()
        if opt_input:
            param_grid['optimizer'] = [x.strip() for x in opt_input.split(',')]
        
        # Dropout
        print("\n📌 Dropout:")
        print("   Örnek: 0.3,0.5,0.7")
        drop_input = input("   Değerler (virgülle ayırın, boş bırakılabilir): ").strip()
        if drop_input:
            param_grid['dropout'] = [float(x) for x in drop_input.split(',')]
        
        # Özet göster
        print("\n📋 Grid Search Özeti:")
        total_combinations = 1
        for key, values in param_grid.items():
            print(f"  {key}: {values}")
            total_combinations *= len(values)
        
        print(f"\n⚠️  Toplam {total_combinations} kombinasyon denenecek!")
        print(f"   Tahmini süre: ~{total_combinations * num_epochs * 2} dakika")
        
        confirm = input("\nDevam edilsin mi? (e/h): ").strip().lower()
        if confirm != 'e':
            print("İptal edildi.")
            return
        
        print("\n🚀 Grid Search başlatılıyor...")
        # TODO: Gerçek dataloaders ile çalıştır
        print("⚠️  Not: Bu özellik train_loader ve val_loader gerektirir.")
        print("   Lütfen run_training.py veya ana menüden eğitim başlatın.")
        
    elif choice == '2':
        # Bayesian Search
        print("\n" + "="*70)
        print("BAYESIAN SEARCH (OPTUNA) AYARLARI")
        print("="*70)
        
        n_trials = int(input("\n📊 Deneme sayısı (örn: 50): ").strip() or "50")
        timeout = input("⏱️  Maksimum süre (saniye, boş=sınırsız): ").strip()
        timeout = int(timeout) if timeout else None
        
        # Parametre dağılımları
        param_distributions = {}
        
        print("\n📌 Learning Rate:")
        print("   Aralık (log scale): örn: 0.00001-0.01")
        lr_min = float(input("   Min değer: ").strip() or "1e-5")
        lr_max = float(input("   Max değer: ").strip() or "1e-2")
        param_distributions['learning_rate'] = {
            'type': 'float',
            'low': lr_min,
            'high': lr_max,
            'log': True
        }
        
        print("\n📌 Batch Size:")
        print("   Seçenekler: örn: 16,32,64")
        bs_input = input("   Değerler (virgülle ayırın): ").strip() or "16,32,64"
        param_distributions['batch_size'] = {
            'type': 'categorical',
            'choices': [int(x) for x in bs_input.split(',')]
        }
        
        print("\n📌 Optimizer:")
        print("   Seçenekler: örn: adam,adamw")
        opt_input = input("   Değerler (virgülle ayırın): ").strip() or "adam,adamw"
        param_distributions['optimizer'] = {
            'type': 'categorical',
            'choices': [x.strip() for x in opt_input.split(',')]
        }
        
        print("\n📌 Dropout:")
        dr_min = float(input("   Min değer (örn: 0.1): ").strip() or "0.1")
        dr_max = float(input("   Max değer (örn: 0.7): ").strip() or "0.7")
        param_distributions['dropout'] = {
            'type': 'float',
            'low': dr_min,
            'high': dr_max
        }
        
        # Özet göster
        print("\n📋 Bayesian Search Özeti:")
        print(f"  Deneme sayısı: {n_trials}")
        print(f"  Timeout: {timeout if timeout else 'Yok'} saniye")
        print(f"  Optimize edilecek metrik: {metric}")
        print(f"\n  Parametreler:")
        for key, dist in param_distributions.items():
            if dist['type'] == 'categorical':
                print(f"    {key}: {dist['choices']}")
            elif dist['type'] == 'float':
                print(f"    {key}: [{dist['low']}, {dist['high']}]")
        
        print(f"\n   Tahmini süre: ~{n_trials * num_epochs * 2} dakika")
        
        confirm = input("\nDevam edilsin mi? (e/h): ").strip().lower()
        if confirm != 'e':
            print("İptal edildi.")
            return
        
        print("\n🚀 Bayesian Search başlatılıyor...")
        print("⚠️  Not: Bu özellik train_loader ve val_loader gerektirir.")
        print("   Lütfen run_training.py veya ana menüden eğitim başlatın.")
    
    else:
        print("❌ Geçersiz seçim!")


if __name__ == "__main__":
    run_interactive_optimization()
