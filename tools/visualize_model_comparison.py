"""
Model Comparison Visualizer - ROC AUC ve Skor Grafikleri
Multiple modellerin karşılaştırmalı görselleştirilmesi
"""

import sys
from pathlib import Path

# Ana dizini path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
import torch
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from src.models import load_model_from_checkpoint
from src.preprocessing.preprocess import DataPreprocessor

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


class ModelComparisonVisualizer:
    """Multiple model karşılaştırmalı görselleştirme"""
    
    def __init__(self, config_path: str = 'configs/config.yaml'):
        """
        Args:
            config_path: Config dosya yolu
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_results = {}
        
    def evaluate_model(self, model_path: str, model_name: str, test_loader):
        """
        Tek bir modeli değerlendir ve ROC verilerini topla
        
        Args:
            model_path: Model checkpoint path
            model_name: Model ismi
            test_loader: Test dataloader
            
        Returns:
            Dict: Model evaluation results
        """
        print(f"\n📦 Loading {model_name}...")
        model = load_model_from_checkpoint(model_path, self.config['model'], self.device)
        model.eval()
        
        all_labels = []
        all_probs = []
        
        print(f"🔍 Evaluating {model_name}...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Positive class prob
        
        # Convert to numpy
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        # Calculate predictions
        all_preds = (all_probs >= 0.5).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        results = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'labels': all_labels,
            'probabilities': all_probs
        }
        
        print(f"✅ {model_name} - AUC: {roc_auc:.4f}, Accuracy: {accuracy:.4f}")
        
        return results
    
    def evaluate_multiple_models(self, model_configs: Dict[str, str], test_loader):
        """
        Multiple modelleri değerlendir
        
        Args:
            model_configs: {model_name: checkpoint_path}
            test_loader: Test dataloader
        """
        print(f"\n{'='*70}")
        print("MULTIPLE MODEL EVALUATION")
        print(f"{'='*70}\n")
        
        for model_name, checkpoint_path in model_configs.items():
            try:
                results = self.evaluate_model(checkpoint_path, model_name, test_loader)
                self.model_results[model_name] = results
            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {e}")
                continue
        
        print(f"\n✅ Evaluated {len(self.model_results)} models successfully\n")
    
    def plot_roc_curves(self, save_path: str = 'outputs/roc_comparison.png'):
        """
        Tüm modeller için ROC curve çiz
        
        Args:
            save_path: Grafik kayıt yolu
        """
        plt.figure(figsize=(12, 9))
        
        # Color palette
        colors = sns.color_palette("husl", len(self.model_results))
        
        for idx, (model_name, results) in enumerate(self.model_results.items()):
            plt.plot(
                results['fpr'], 
                results['tpr'], 
                color=colors[idx],
                linewidth=2.5, 
                label=f"{model_name} (AUC = {results['auc']:.4f})"
            )
        
        # Random classifier line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        
        plt.xlabel('False Positive Rate (FPR)', fontsize=13, fontweight='bold')
        plt.ylabel('True Positive Rate (TPR)', fontsize=13, fontweight='bold')
        plt.title('ROC Curve Comparison - Multiple Models', fontsize=15, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        # Add shaded area for best performance
        plt.fill_between([0, 0, 1], [0, 1, 1], alpha=0.1, color='green', 
                        label='_nolegend_')
        
        plt.tight_layout()
        
        # Save plot
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curves saved: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_auc_scores_bar(self, save_path: str = 'outputs/auc_scores_comparison.png'):
        """
        AUC skorlarını bar chart olarak göster
        
        Args:
            save_path: Grafik kayıt yolu
        """
        models = list(self.model_results.keys())
        auc_scores = [self.model_results[m]['auc'] for m in models]
        
        # Sort by AUC score
        sorted_indices = np.argsort(auc_scores)[::-1]
        models = [models[i] for i in sorted_indices]
        auc_scores = [auc_scores[i] for i in sorted_indices]
        
        plt.figure(figsize=(12, 7))
        
        # Create color gradient
        colors = plt.cm.RdYlGn(np.linspace(0.4, 0.9, len(models)))
        
        bars = plt.barh(models, auc_scores, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, score in zip(bars, auc_scores):
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.4f}',
                    ha='left', va='center', fontsize=11, fontweight='bold')
        
        plt.xlabel('AUC Score', fontsize=13, fontweight='bold')
        plt.ylabel('Model', fontsize=13, fontweight='bold')
        plt.title('AUC-ROC Score Comparison', fontsize=15, fontweight='bold')
        plt.xlim([0, 1.1])
        plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, 
                   label='Random Classifier')
        plt.grid(True, alpha=0.3, axis='x')
        plt.legend(fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ AUC scores bar chart saved: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_metrics_comparison(self, save_path: str = 'outputs/metrics_comparison.png'):
        """
        Tüm metrikleri karşılaştırmalı göster
        
        Args:
            save_path: Grafik kayıt yolu
        """
        models = list(self.model_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        # Prepare data
        data = {metric: [self.model_results[m][metric] for m in models] 
                for metric in metrics}
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        colors = sns.color_palette("Set2", len(models))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = data[metric]
            
            bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=1.2)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{value:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_ylabel('Score', fontsize=11, fontweight='bold')
            ax.set_title(metric.replace('_', ' ').title(), fontsize=13, fontweight='bold')
            ax.set_ylim([0, 1.1])
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add reference line at 0.5
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, linewidth=1)
        
        # Remove extra subplot
        fig.delaxes(axes[5])
        
        plt.suptitle('Performance Metrics Comparison - All Models', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        # Save plot
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Metrics comparison saved: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_model_ranking(self, save_path: str = 'outputs/model_ranking.png'):
        """
        Model sıralaması - radar chart ve genel performans
        
        Args:
            save_path: Grafik kayıt yolu
        """
        models = list(self.model_results.keys())
        
        # Calculate overall score (weighted average of metrics)
        weights = {'accuracy': 0.2, 'precision': 0.2, 'recall': 0.2, 'f1_score': 0.2, 'auc': 0.2}
        
        overall_scores = []
        for model in models:
            score = sum(self.model_results[model][metric] * weight 
                       for metric, weight in weights.items())
            overall_scores.append(score)
        
        # Sort by overall score
        sorted_indices = np.argsort(overall_scores)[::-1]
        models_sorted = [models[i] for i in sorted_indices]
        scores_sorted = [overall_scores[i] for i in sorted_indices]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Overall ranking bar chart
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models_sorted)))
        bars = ax1.barh(models_sorted, scores_sorted, color=colors, 
                       edgecolor='black', linewidth=1.5)
        
        for bar, score in zip(bars, scores_sorted):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.4f}',
                    ha='left', va='center', fontsize=11, fontweight='bold')
        
        ax1.set_xlabel('Overall Score (Weighted Average)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Model', fontsize=12, fontweight='bold')
        ax1.set_title('Model Ranking by Overall Performance', fontsize=14, fontweight='bold')
        ax1.set_xlim([0, 1.1])
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Top 3 models - detailed metrics
        top_3_models = models_sorted[:min(3, len(models_sorted))]
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, model in enumerate(top_3_models):
            values = [self.model_results[model][m] for m in metrics]
            offset = width * (i - 1)
            ax2.bar(x + offset, values, width, label=model, 
                   edgecolor='black', linewidth=1)
        
        ax2.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax2.set_title('Top 3 Models - Detailed Metrics', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.replace('_', '\n') for m in metrics], fontsize=10)
        ax2.set_ylim([0, 1.1])
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Model ranking saved: {save_path}")
        
        plt.show()
        plt.close()
    
    def create_summary_report(self, save_path: str = 'outputs/model_comparison_summary.txt'):
        """
        Özet rapor oluştur
        
        Args:
            save_path: Rapor kayıt yolu
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("MODEL COMPARISON SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Overall statistics
            f.write("📊 OVERALL STATISTICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Total Models Evaluated: {len(self.model_results)}\n\n")
            
            # Detailed model results
            f.write("📈 DETAILED MODEL RESULTS\n")
            f.write("-"*70 + "\n\n")
            
            for model_name, results in self.model_results.items():
                f.write(f"Model: {model_name}\n")
                f.write(f"  AUC-ROC:    {results['auc']:.4f}\n")
                f.write(f"  Accuracy:   {results['accuracy']:.4f}\n")
                f.write(f"  Precision:  {results['precision']:.4f}\n")
                f.write(f"  Recall:     {results['recall']:.4f}\n")
                f.write(f"  F1-Score:   {results['f1_score']:.4f}\n")
                f.write("\n")
            
            # Best model per metric
            f.write("🏆 BEST MODELS PER METRIC\n")
            f.write("-"*70 + "\n")
            
            metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1_score']
            for metric in metrics:
                best_model = max(self.model_results.items(), 
                               key=lambda x: x[1][metric])
                f.write(f"{metric.upper():12s}: {best_model[0]:20s} "
                       f"({best_model[1][metric]:.4f})\n")
        
        print(f"✅ Summary report saved: {save_path}")
    
    def visualize_all(self, output_dir: str = 'outputs'):
        """
        Tüm grafikleri oluştur
        
        Args:
            output_dir: Çıktı dizini
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print("GENERATING ALL VISUALIZATIONS")
        print(f"{'='*70}\n")
        
        # 1. ROC Curves
        self.plot_roc_curves(output_dir / 'roc_comparison.png')
        
        # 2. AUC Scores Bar Chart
        self.plot_auc_scores_bar(output_dir / 'auc_scores_comparison.png')
        
        # 3. Metrics Comparison
        self.plot_metrics_comparison(output_dir / 'metrics_comparison.png')
        
        # 4. Model Ranking
        self.plot_model_ranking(output_dir / 'model_ranking.png')
        
        # 5. Summary Report
        self.create_summary_report(output_dir / 'model_comparison_summary.txt')
        
        print(f"\n{'='*70}")
        print(f"✅ All visualizations saved to: {output_dir}")
        print(f"{'='*70}\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Model Comparison Visualizer')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Config file path')
    parser.add_argument('--output-dir', type=str, default='outputs/model_comparison',
                       help='Output directory for visualizations')
    parser.add_argument('--models', type=str, nargs='+', required=False,
                       help='Model checkpoint paths')
    parser.add_argument('--model-names', type=str, nargs='+', required=False,
                       help='Model names (must match number of models)')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ModelComparisonVisualizer(args.config)
    
    # Load test data
    print("\n📦 Loading test data...")
    preprocessor = DataPreprocessor(visualizer.config)
    _, _, test_loader = preprocessor.get_dataloaders()
    print(f"✅ Test loader ready: {len(test_loader)} batches\n")
    
    # If models are provided, evaluate them
    if args.models and args.model_names:
        if len(args.models) != len(args.model_names):
            raise ValueError("Number of models must match number of model names")
        
        model_configs = dict(zip(args.model_names, args.models))
        visualizer.evaluate_multiple_models(model_configs, test_loader)
    else:
        # Example: Auto-detect models from checkpoints directory
        print("⚠️  No models specified. Please provide --models and --model-names")
        print("\nExample usage:")
        print("python visualize_model_comparison.py \\")
        print("  --models checkpoints/model1.pth checkpoints/model2.pth \\")
        print("  --model-names 'ResNet50' 'VGG16' \\")
        print("  --output-dir outputs/comparison")
        return
    
    # Generate all visualizations
    if len(visualizer.model_results) > 0:
        visualizer.visualize_all(args.output_dir)
    else:
        print("❌ No models evaluated successfully")


if __name__ == '__main__':
    main()
