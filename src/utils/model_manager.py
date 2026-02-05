"""
Model Manager - Model ve Performans Kaydetme Sistemi
Eğitilen modelleri, performans metriklerini ve grafikleri organize bir şekilde kaydeder
"""

import torch
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shutil


class ModelManager:
    """
    Model ve performans yönetim sistemi
    Her model için ayrı dizin oluşturur ve tüm bilgileri organize eder
    """
    
    def __init__(self, base_dir: str = "outputs/trained_models"):
        """
        Args:
            base_dir: Modellerin kaydedileceği ana dizin
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def create_model_directory(
        self, 
        model_name: str,
        timestamp: Optional[str] = None
    ) -> Path:
        """
        Yeni model için dizin oluştur
        
        Args:
            model_name: Model adı (örn: 'resnet3d', 'cnn3d_simple')
            timestamp: Zaman damgası (None ise otomatik oluşturulur)
            
        Returns:
            Model dizin path'i
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_dir = self.base_dir / f"{model_name}_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Alt dizinler oluştur
        (model_dir / "checkpoints").mkdir(exist_ok=True)
        (model_dir / "plots").mkdir(exist_ok=True)
        (model_dir / "metrics").mkdir(exist_ok=True)
        (model_dir / "logs").mkdir(exist_ok=True)
        
        return model_dir
    
    def save_model_checkpoint(
        self,
        model_dir: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict,
        config: Dict,
        is_best: bool = False,
        scheduler: Optional[Any] = None
    ):
        """
        Model checkpoint'ini kaydet
        
        Args:
            model_dir: Model dizini
            model: PyTorch model
            optimizer: Optimizer
            epoch: Epoch sayısı
            metrics: Performans metrikleri
            config: Model ve training konfigürasyonu
            is_best: En iyi model mi?
            scheduler: Learning rate scheduler
        """
        checkpoint_dir = model_dir / "checkpoints"
        
        # Checkpoint hazırla
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Son checkpoint
        last_path = checkpoint_dir / "last_checkpoint.pth"
        torch.save(checkpoint, last_path)
        print(f"✅ Last checkpoint saved: {last_path}")
        
        # En iyi model
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"🏆 Best model saved: {best_path}")
        
        # Epoch checkpoint (her 10 epoch'ta bir)
        if epoch % 10 == 0:
            epoch_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
            torch.save(checkpoint, epoch_path)
            print(f"💾 Epoch {epoch} checkpoint saved: {epoch_path}")
    
    def save_training_metrics(
        self,
        model_dir: Path,
        train_history: Dict,
        val_history: Dict,
        test_metrics: Optional[Dict] = None
    ):
        """
        Training metriklerini kaydet
        
        Args:
            model_dir: Model dizini
            train_history: Training history dict
            val_history: Validation history dict
            test_metrics: Test metrikleri (opsiyonel)
        """
        metrics_dir = model_dir / "metrics"
        
        # Training history
        history = {
            'train': train_history,
            'validation': val_history,
            'timestamp': datetime.now().isoformat()
        }
        
        if test_metrics:
            history['test'] = test_metrics
        
        # JSON olarak kaydet
        history_path = metrics_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"✅ Training history saved: {history_path}")
        
        # En iyi metrikleri ayrı kaydet
        best_metrics = self._extract_best_metrics(val_history, test_metrics)
        best_path = metrics_dir / "best_metrics.json"
        with open(best_path, 'w') as f:
            json.dump(best_metrics, f, indent=2)
        print(f"✅ Best metrics saved: {best_path}")
        
        # CSV olarak da kaydet (Excel'de açmak için)
        self._save_metrics_csv(metrics_dir, train_history, val_history)
    
    def _extract_best_metrics(
        self,
        val_history: Dict,
        test_metrics: Optional[Dict] = None
    ) -> Dict:
        """En iyi metrikleri çıkar"""
        
        best = {
            'validation': {
                'best_epoch': int(np.argmin(val_history['loss'])) + 1,
                'best_loss': float(min(val_history['loss'])),
                'best_accuracy': float(max(val_history['acc'])),
            }
        }
        
        if 'f1' in val_history:
            best['validation']['best_f1'] = float(max(val_history['f1']))
        if 'auc' in val_history:
            best['validation']['best_auc'] = float(max(val_history['auc']))
        
        if test_metrics:
            best['test'] = test_metrics
        
        return best
    
    def _save_metrics_csv(
        self,
        metrics_dir: Path,
        train_history: Dict,
        val_history: Dict
    ):
        """Metrikleri CSV formatında kaydet"""
        import pandas as pd
        
        # DataFrame oluştur
        epochs = range(1, len(train_history['loss']) + 1)
        data = {
            'epoch': epochs,
            'train_loss': train_history['loss'],
            'train_acc': train_history['acc'],
            'val_loss': val_history['loss'],
            'val_acc': val_history['acc'],
        }
        
        if 'f1' in val_history:
            data['val_f1'] = val_history['f1']
        if 'auc' in val_history:
            data['val_auc'] = val_history['auc']
        
        df = pd.DataFrame(data)
        csv_path = metrics_dir / "training_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Metrics CSV saved: {csv_path}")
    
    def save_training_plots(
        self,
        model_dir: Path,
        train_history: Dict,
        val_history: Dict
    ):
        """
        Training grafikleri kaydet
        
        Args:
            model_dir: Model dizini
            train_history: Training history
            val_history: Validation history
        """
        plots_dir = model_dir / "plots"
        
        # 1. Loss ve Accuracy grafikleri
        self._plot_training_curves(
            train_history, 
            val_history, 
            plots_dir / "training_curves.png"
        )
        
        # 2. Ayrı ayrı grafikler
        self._plot_loss_curve(
            train_history, 
            val_history, 
            plots_dir / "loss_curve.png"
        )
        
        self._plot_accuracy_curve(
            train_history, 
            val_history, 
            plots_dir / "accuracy_curve.png"
        )
        
        # 3. F1 ve AUC (varsa)
        if 'f1' in val_history:
            self._plot_metric_curve(
                val_history['f1'], 
                'F1-Score', 
                plots_dir / "f1_curve.png"
            )
        
        if 'auc' in val_history:
            self._plot_metric_curve(
                val_history['auc'], 
                'AUC-ROC', 
                plots_dir / "auc_curve.png"
            )
    
    def _plot_training_curves(
        self,
        train_history: Dict,
        val_history: Dict,
        save_path: Path
    ):
        """Tüm training eğrileri (2x2 grid)"""
        
        epochs = range(1, len(train_history['loss']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(epochs, train_history['loss'], 'b-', linewidth=2, label='Train')
        axes[0, 0].plot(epochs, val_history['loss'], 'r-', linewidth=2, label='Validation')
        axes[0, 0].set_xlabel('Epoch', fontsize=11)
        axes[0, 0].set_ylabel('Loss', fontsize=11)
        axes[0, 0].set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, train_history['acc'], 'b-', linewidth=2, label='Train')
        axes[0, 1].plot(epochs, val_history['acc'], 'r-', linewidth=2, label='Validation')
        axes[0, 1].set_xlabel('Epoch', fontsize=11)
        axes[0, 1].set_ylabel('Accuracy', fontsize=11)
        axes[0, 1].set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1-Score
        if 'f1' in val_history:
            axes[1, 0].plot(epochs, val_history['f1'], 'g-', linewidth=2)
            axes[1, 0].set_xlabel('Epoch', fontsize=11)
            axes[1, 0].set_ylabel('F1-Score', fontsize=11)
            axes[1, 0].set_title('Validation F1-Score', fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'F1-Score Not Available', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # AUC-ROC
        if 'auc' in val_history:
            axes[1, 1].plot(epochs, val_history['auc'], 'm-', linewidth=2)
            axes[1, 1].set_xlabel('Epoch', fontsize=11)
            axes[1, 1].set_ylabel('AUC-ROC', fontsize=11)
            axes[1, 1].set_title('Validation AUC-ROC', fontsize=12, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'AUC-ROC Not Available', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Training curves saved: {save_path}")
    
    def _plot_loss_curve(
        self,
        train_history: Dict,
        val_history: Dict,
        save_path: Path
    ):
        """Sadece loss grafiği"""
        
        epochs = range(1, len(train_history['loss']) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_history['loss'], 'b-', linewidth=2.5, 
                marker='o', markersize=4, label='Train Loss')
        plt.plot(epochs, val_history['loss'], 'r-', linewidth=2.5, 
                marker='s', markersize=4, label='Validation Loss')
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Loss', fontsize=12, fontweight='bold')
        plt.title('Loss Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Loss curve saved: {save_path}")
    
    def _plot_accuracy_curve(
        self,
        train_history: Dict,
        val_history: Dict,
        save_path: Path
    ):
        """Sadece accuracy grafiği"""
        
        epochs = range(1, len(train_history['acc']) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_history['acc'], 'b-', linewidth=2.5, 
                marker='o', markersize=4, label='Train Accuracy')
        plt.plot(epochs, val_history['acc'], 'r-', linewidth=2.5, 
                marker='s', markersize=4, label='Validation Accuracy')
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        plt.title('Accuracy Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Accuracy curve saved: {save_path}")
    
    def _plot_metric_curve(
        self,
        metric_values: list,
        metric_name: str,
        save_path: Path
    ):
        """Tek bir metrik grafiği"""
        
        epochs = range(1, len(metric_values) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, metric_values, 'g-', linewidth=2.5, 
                marker='D', markersize=4, label=metric_name)
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel(metric_name, fontsize=12, fontweight='bold')
        plt.title(f'{metric_name} Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ {metric_name} curve saved: {save_path}")
    
    def save_roc_curve(
        self,
        model_dir: Path,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        model_name: Optional[str] = None
    ):
        """
        ROC eğrisini kaydet
        
        Args:
            model_dir: Model dizini
            fpr: False positive rate
            tpr: True positive rate
            auc_score: AUC skoru
            model_name: Model adı
        """
        plots_dir = model_dir / "plots"
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, 'b-', linewidth=3, 
                label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
        plt.xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
        
        title = 'ROC Curve'
        if model_name:
            title += f' - {model_name}'
        plt.title(title, fontsize=14, fontweight='bold')
        
        plt.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        roc_path = plots_dir / "roc_curve.png"
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ ROC curve saved: {roc_path}")
        
        # ROC verilerini de JSON olarak kaydet
        roc_data = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(auc_score),
            'timestamp': datetime.now().isoformat()
        }
        
        roc_json_path = model_dir / "metrics" / "roc_data.json"
        with open(roc_json_path, 'w') as f:
            json.dump(roc_data, f, indent=2)
        print(f"✅ ROC data saved: {roc_json_path}")
    
    def save_confusion_matrix(
        self,
        model_dir: Path,
        cm: np.ndarray,
        class_names: list = None,
        normalize: bool = False
    ):
        """
        Confusion matrix'i kaydet
        
        Args:
            model_dir: Model dizini
            cm: Confusion matrix
            class_names: Sınıf isimleri
            normalize: Normalize edilsin mi
        """
        plots_dir = model_dir / "plots"
        
        if class_names is None:
            class_names = ['Normal', 'Anomaly']
        
        # Plot confusion matrix
        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
            cm_to_plot = cm_normalized
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
            cm_to_plot = cm
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_to_plot, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold')
        
        cm_path = plots_dir / f"confusion_matrix{'_normalized' if normalize else ''}.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Confusion matrix saved: {cm_path}")
        
        # Her iki versiyonu da kaydet
        if not normalize:
            self.save_confusion_matrix(model_dir, cm, class_names, normalize=True)
    
    def save_config(
        self,
        model_dir: Path,
        config: Dict
    ):
        """
        Konfigürasyonu kaydet
        
        Args:
            model_dir: Model dizini
            config: Konfigürasyon dict
        """
        config_path = model_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Config saved: {config_path}")
    
    def save_model_summary(
        self,
        model_dir: Path,
        model: torch.nn.Module,
        input_size: tuple = (1, 1, 128, 128, 128)
    ):
        """
        Model summary kaydet
        
        Args:
            model_dir: Model dizini
            model: PyTorch model
            input_size: Input boyutu
        """
        summary_path = model_dir / "model_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write(f"Model: {model.__class__.__name__}\n")
            f.write("="*70 + "\n\n")
            f.write(str(model))
            f.write("\n\n" + "="*70 + "\n")
            
            # Parametre sayısı
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            f.write(f"\nTotal Parameters: {total_params:,}\n")
            f.write(f"Trainable Parameters: {trainable_params:,}\n")
            f.write(f"Non-trainable Parameters: {total_params - trainable_params:,}\n")
        
        print(f"✅ Model summary saved: {summary_path}")
    
    def create_model_report(
        self,
        model_dir: Path,
        model_name: str,
        train_history: Dict,
        val_history: Dict,
        test_metrics: Optional[Dict] = None,
        training_time: Optional[float] = None
    ):
        """
        Kapsamlı model raporu oluştur
        
        Args:
            model_dir: Model dizini
            model_name: Model adı
            train_history: Training history
            val_history: Validation history
            test_metrics: Test metrikleri
            training_time: Training süresi (saniye)
        """
        report_path = model_dir / "MODEL_REPORT.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Model Raporu: {model_name}\n\n")
            f.write(f"**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # Training info
            f.write("## Training Bilgileri\n\n")
            f.write(f"- **Toplam Epoch:** {len(train_history['loss'])}\n")
            if training_time:
                hours = training_time // 3600
                minutes = (training_time % 3600) // 60
                f.write(f"- **Training Süresi:** {int(hours)}h {int(minutes)}m\n")
            f.write("\n")
            
            # Best validation metrics
            f.write("## En İyi Validation Metrikleri\n\n")
            best_epoch = int(np.argmin(val_history['loss'])) + 1
            f.write(f"- **En İyi Epoch:** {best_epoch}\n")
            f.write(f"- **En Düşük Loss:** {min(val_history['loss']):.4f}\n")
            f.write(f"- **En Yüksek Accuracy:** {max(val_history['acc']):.4f}\n")
            
            if 'f1' in val_history:
                f.write(f"- **En Yüksek F1:** {max(val_history['f1']):.4f}\n")
            if 'auc' in val_history:
                f.write(f"- **En Yüksek AUC:** {max(val_history['auc']):.4f}\n")
            f.write("\n")
            
            # Test metrics
            if test_metrics:
                f.write("## Test Metrikleri\n\n")
                f.write(f"- **Accuracy:** {test_metrics.get('accuracy', 0):.4f}\n")
                f.write(f"- **Precision:** {test_metrics.get('precision_weighted', 0):.4f}\n")
                f.write(f"- **Recall:** {test_metrics.get('recall_weighted', 0):.4f}\n")
                f.write(f"- **F1-Score:** {test_metrics.get('f1_weighted', 0):.4f}\n")
                f.write(f"- **AUC-ROC:** {test_metrics.get('auc_roc', 0):.4f}\n")
                f.write(f"- **MCC:** {test_metrics.get('mcc', 0):.4f}\n")
                f.write("\n")
            
            # Dosyalar
            f.write("## Kaydedilen Dosyalar\n\n")
            f.write("### Checkpoints\n")
            f.write("- `checkpoints/best_model.pth` - En iyi model\n")
            f.write("- `checkpoints/last_checkpoint.pth` - Son checkpoint\n\n")
            
            f.write("### Metrikler\n")
            f.write("- `metrics/training_history.json` - Training history\n")
            f.write("- `metrics/best_metrics.json` - En iyi metrikler\n")
            f.write("- `metrics/training_metrics.csv` - CSV formatında metrikler\n\n")
            
            f.write("### Grafikler\n")
            f.write("- `plots/training_curves.png` - Tüm training eğrileri\n")
            f.write("- `plots/loss_curve.png` - Loss eğrisi\n")
            f.write("- `plots/accuracy_curve.png` - Accuracy eğrisi\n")
            f.write("- `plots/roc_curve.png` - ROC eğrisi\n")
            f.write("- `plots/confusion_matrix.png` - Confusion matrix\n\n")
        
        print(f"✅ Model report created: {report_path}")
    
    def list_trained_models(self) -> list:
        """Eğitilmiş modelleri listele"""
        
        models = []
        for model_dir in self.base_dir.iterdir():
            if model_dir.is_dir():
                best_metrics_path = model_dir / "metrics" / "best_metrics.json"
                if best_metrics_path.exists():
                    with open(best_metrics_path, 'r') as f:
                        metrics = json.load(f)
                    
                    models.append({
                        'name': model_dir.name,
                        'path': str(model_dir),
                        'metrics': metrics
                    })
        
        return models
    
    def compare_models(
        self,
        output_dir: Optional[str] = None
    ):
        """
        Tüm eğitilmiş modelleri karşılaştır
        
        Args:
            output_dir: Karşılaştırma sonuçlarının kaydedileceği dizin
        """
        if output_dir is None:
            output_dir = self.base_dir / "model_comparison"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Modelleri listele
        models = self.list_trained_models()
        
        if not models:
            print("❌ Hiç eğitilmiş model bulunamadı!")
            return
        
        print(f"\n{'='*70}")
        print(f"MODEL KARŞILAŞTIRMASI - {len(models)} model bulundu")
        print(f"{'='*70}\n")
        
        # Karşılaştırma tablosu
        comparison_data = {}
        for model in models:
            model_name = model['name']
            metrics = model['metrics'].get('validation', {})
            comparison_data[model_name] = metrics
        
        # JSON olarak kaydet
        comparison_path = output_dir / "model_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print(f"✅ Comparison data saved: {comparison_path}")
        
        # Karşılaştırma grafiği
        self._plot_model_comparison(comparison_data, output_dir)
        
        # Karşılaştırma tablosu yazdır
        self._print_comparison_table(comparison_data)
    
    def _plot_model_comparison(
        self,
        comparison_data: Dict,
        output_dir: Path
    ):
        """Model karşılaştırma grafikleri"""
        
        models = list(comparison_data.keys())
        metrics_to_plot = ['best_accuracy', 'best_f1', 'best_auc', 'best_loss']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            values = []
            valid_models = []
            
            for model in models:
                if metric in comparison_data[model]:
                    values.append(comparison_data[model][metric])
                    valid_models.append(model)
            
            if values:
                # Kısa model isimleri için
                short_names = [m.split('_')[0] for m in valid_models]
                
                bars = axes[idx].bar(range(len(values)), values, 
                                   color='skyblue', edgecolor='navy', alpha=0.7)
                axes[idx].set_xticks(range(len(values)))
                axes[idx].set_xticklabels(short_names, rotation=45, ha='right')
                axes[idx].set_ylabel(metric.replace('_', ' ').title())
                axes[idx].set_title(metric.replace('_', ' ').title())
                axes[idx].grid(True, alpha=0.3, axis='y')
                
                # Bar üzerine değerleri yaz
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                                 f'{value:.3f}',
                                 ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        comparison_plot_path = output_dir / "model_comparison.png"
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Comparison plot saved: {comparison_plot_path}")
    
    def _print_comparison_table(self, comparison_data: Dict):
        """Karşılaştırma tablosunu yazdır"""
        
        print(f"\n{'Model':<30} {'Accuracy':<12} {'F1':<12} {'AUC':<12} {'Loss':<12}")
        print("-" * 78)
        
        for model_name, metrics in comparison_data.items():
            acc = metrics.get('best_accuracy', 0)
            f1 = metrics.get('best_f1', 0)
            auc = metrics.get('best_auc', 0)
            loss = metrics.get('best_loss', 0)
            
            print(f"{model_name:<30} {acc:<12.4f} {f1:<12.4f} {auc:<12.4f} {loss:<12.4f}")
        
        print("=" * 78 + "\n")


# Kolay kullanım için yardımcı fonksiyonlar
def create_model_manager(base_dir: str = "outputs/trained_models") -> ModelManager:
    """Model manager oluştur"""
    return ModelManager(base_dir)


def save_complete_model_results(
    model_manager: ModelManager,
    model_dir: Path,
    model: torch.nn.Module,
    model_name: str,
    optimizer: torch.optim.Optimizer,
    train_history: Dict,
    val_history: Dict,
    test_metrics: Dict,
    config: Dict,
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    cm: np.ndarray,
    training_time: Optional[float] = None
):
    """
    Tüm model sonuçlarını kaydet (tek seferde)
    
    Args:
        model_manager: ModelManager instance
        model_dir: Model dizini
        model: PyTorch model
        model_name: Model adı
        optimizer: Optimizer
        train_history: Training history
        val_history: Validation history
        test_metrics: Test metrikleri
        config: Konfigürasyon
        fpr: False positive rate
        tpr: True positive rate
        auc_score: AUC skoru
        cm: Confusion matrix
        training_time: Training süresi
    """
    print(f"\n{'='*70}")
    print(f"MODEL SONUÇLARI KAYDEDİLİYOR: {model_name}")
    print(f"{'='*70}\n")
    
    # Model checkpoint
    model_manager.save_model_checkpoint(
        model_dir, model, optimizer,
        epoch=len(train_history['loss']),
        metrics=val_history,
        config=config,
        is_best=True
    )
    
    # Metrikler
    model_manager.save_training_metrics(
        model_dir, train_history, val_history, test_metrics
    )
    
    # Grafikler
    model_manager.save_training_plots(
        model_dir, train_history, val_history
    )
    
    # ROC curve
    model_manager.save_roc_curve(
        model_dir, fpr, tpr, auc_score, model_name
    )
    
    # Confusion matrix
    model_manager.save_confusion_matrix(
        model_dir, cm
    )
    
    # Config
    model_manager.save_config(model_dir, config)
    
    # Model summary
    model_manager.save_model_summary(model_dir, model)
    
    # Model raporu
    model_manager.create_model_report(
        model_dir, model_name, train_history, val_history,
        test_metrics, training_time
    )
    
    print(f"\n{'='*70}")
    print(f"✅ TÜM SONUÇLAR KAYDEDİLDİ: {model_dir}")
    print(f"{'='*70}\n")
