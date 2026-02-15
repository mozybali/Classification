"""
Visualization Tools - Training & Evaluation Plotting
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json


# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class TrainingVisualizer:
    """Training history visualization"""
    
    @staticmethod
    def plot_training_history(
        train_history: Dict,
        val_history: Dict,
        save_path: Optional[str] = None
    ):
        """
        Training history plots oluştur
        
        Args:
            train_history: {'loss': [...], 'acc': [...]}
            val_history: {'loss': [...], 'acc': [...], 'f1': [...], 'auc': [...]}
            save_path: Plot'u kaydet
        """
        epochs = range(1, len(train_history['loss']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(epochs, train_history['loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, val_history['loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, train_history['acc'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, val_history['acc'], 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training & Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1-Score plot
        axes[1, 0].plot(epochs, val_history['f1'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_title('Validation F1-Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # AUC plot
        axes[1, 1].plot(epochs, val_history['auc'], 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC-ROC')
        axes[1, 1].set_title('Validation AUC-ROC')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Training history plot saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray,
        class_names: List[str] = None,
        normalize: bool = False,
        save_path: Optional[str] = None,
        show: bool = True,
        close: bool = True,
        return_fig: bool = False
    ):
        """
        Confusion matrix plot

        Args:
            cm: Confusion matrix
            class_names: Class names
            normalize: Normalize edilsin mi
            save_path: Save path
        """
        if class_names is None:
            class_names = ['Normal', 'Anomaly']

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
            ax=ax
        )
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"??? Confusion matrix saved: {save_path}")
        elif show:
            plt.show()

        if return_fig:
            return fig
        if close:
            plt.close(fig)

    def plot_roc_curves(
        roc_data: Dict,
        save_path: Optional[str] = None
    ):
        """
        ROC curves için multiple models
        
        Args:
            roc_data: {model_name: {'fpr': [...], 'tpr': [...], 'auc': 0.xx}}
            save_path: Save path
        """
        plt.figure(figsize=(10, 8))
        
        # Enhanced color palette
        colors = sns.color_palette("husl", len(roc_data))
        
        for idx, (model_name, data) in enumerate(roc_data.items()):
            fpr = data['fpr']
            tpr = data['tpr']
            auc = data.get('auc', 0.0)
            plt.plot(fpr, tpr, color=colors[idx], linewidth=2.5, 
                    label=f"{model_name} (AUC={auc:.4f})")
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10, frameon=True, shadow=True)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ ROC curves saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_metrics_comparison(
        model_metrics: Dict[str, Dict],
        metrics_to_plot: List[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Multiple models için metrics karşılaştırması
        
        Args:
            model_metrics: {model_name: {metric_name: value}}
            metrics_to_plot: Plot edilecek metrikler
            save_path: Save path
        """
        if metrics_to_plot is None:
            metrics_to_plot = ['accuracy', 'precision_weighted', 'recall_weighted', 
                             'f1_weighted', 'auc_roc']
        
        models = list(model_metrics.keys())
        n_metrics = len(metrics_to_plot)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics_to_plot):
            values = [model_metrics[m].get(metric, 0.0) for m in models]
            
            bars = axes[idx].bar(models, values, color='skyblue', edgecolor='navy', alpha=0.7)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{value:.3f}',
                             ha='center', va='bottom', fontsize=9)
            
            axes[idx].set_ylabel(metric.replace('_', ' ').title())
            axes[idx].set_ylim([0, 1.1])
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Metrics comparison saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_per_class_metrics(
        per_class_metrics: Dict[str, Dict],
        save_path: Optional[str] = None,
        show: bool = True,
        close: bool = True,
        return_fig: bool = False
    ):
        """
        Per-class metrics bar plot

        Args:
            per_class_metrics: {class_name: {metric: value}}
            save_path: Save path
        """
        classes = list(per_class_metrics.keys())
        metrics = ['precision', 'recall', 'f1']

        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, metric in enumerate(metrics):
            values = [per_class_metrics[c][metric] for c in classes]
            offset = width * (i - 1)
            bars = ax.bar(x + offset, values, width, label=metric.capitalize())

            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}',
                       ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"??? Per-class metrics saved: {save_path}")
        elif show:
            plt.show()

        if return_fig:
            return fig
        if close:
            plt.close(fig)

class ErrorAnalysisVisualizer:
    """Error analysis ve misclassification visualization"""
    
    @staticmethod
    def plot_error_distribution(
        predictions: Dict,
        save_path: Optional[str] = None,
        show: bool = True,
        close: bool = True,
        return_fig: bool = False
    ):
        """
        Error distribution plot

        Args:
            predictions: {'labels': [...], 'predictions': [...], 'probabilities': [...]}
            save_path: Save path
        """
        labels = np.array(predictions['labels'])
        preds = np.array(predictions['predictions'])
        probs = np.array(predictions['probabilities'])

        # Correct ve incorrect predictions
        correct_mask = (labels == preds)
        incorrect_mask = ~correct_mask

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Probability distribution for correct predictions
        axes[0].hist(probs[correct_mask], bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0].set_xlabel('Prediction Probability (Positive Class)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Probability Distribution - Correct Predictions')
        axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold=0.5')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Probability distribution for incorrect predictions
        axes[1].hist(probs[incorrect_mask], bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[1].set_xlabel('Prediction Probability (Positive Class)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Probability Distribution - Incorrect Predictions')
        axes[1].axvline(0.5, color='blue', linestyle='--', linewidth=2, label='Threshold=0.5')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"??? Error distribution saved: {save_path}")
        elif show:
            plt.show()

        if return_fig:
            return fig
        if close:
            plt.close(fig)

    def plot_threshold_analysis(
        labels: np.ndarray,
        probabilities: np.ndarray,
        save_path: Optional[str] = None,
        show: bool = True,
        close: bool = True,
        return_fig: bool = False
    ):
        """
        Threshold analysis - optimal threshold bulmak icin

        Args:
            labels: True labels
            probabilities: Prediction probabilities
            save_path: Save path
        """
        from sklearn.metrics import precision_recall_curve, f1_score

        thresholds = np.linspace(0, 1, 100)
        precisions = []
        recalls = []
        f1_scores = []

        for thresh in thresholds:
            preds = (probabilities >= thresh).astype(int)
            if len(np.unique(preds)) > 1:
                from sklearn.metrics import precision_score, recall_score
                precisions.append(precision_score(labels, preds, zero_division=0))
                recalls.append(recall_score(labels, preds, zero_division=0))
                f1_scores.append(f1_score(labels, preds, zero_division=0))
            else:
                precisions.append(0)
                recalls.append(0)
                f1_scores.append(0)

        # Find optimal threshold (max F1)
        optimal_idx = int(np.argmax(f1_scores))
        optimal_threshold = thresholds[optimal_idx]
        max_f1 = f1_scores[optimal_idx]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
        ax.plot(thresholds, recalls, 'g-', label='Recall', linewidth=2)
        ax.plot(thresholds, f1_scores, 'r-', label='F1-Score', linewidth=2)
        ax.axvline(optimal_threshold, color='orange', linestyle='--', linewidth=2,
                   label=f'Optimal Threshold={optimal_threshold:.3f}')
        ax.axvline(0.5, color='gray', linestyle=':', linewidth=1, label='Default=0.5')

        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title(f'Threshold Analysis (Optimal F1={max_f1:.4f} @ Threshold={optimal_threshold:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"??? Threshold analysis saved: {save_path}")
        elif show:
            plt.show()

        if return_fig:
            return fig, optimal_threshold, max_f1
        if close:
            plt.close(fig)

        return optimal_threshold, max_f1


def create_evaluation_report(
    results_dir: str,
    output_path: str = 'evaluation_report.pdf'
):
    """
    Comprehensive evaluation report olustur (tum plot'lari birlestir)

    Args:
        results_dir: Results directory
        output_path: PDF output path
    """
    from matplotlib.backends.backend_pdf import PdfPages

    results_dir = Path(results_dir)

    # Load results
    with open(results_dir / 'evaluation_results_full.json', 'r') as f:
        results = json.load(f)

    with PdfPages(output_path) as pdf:
        # Page 1: Confusion Matrix
        fig = TrainingVisualizer.plot_confusion_matrix(
            np.array(results['confusion_matrix']),
            return_fig=True,
            show=False,
            close=False
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: ROC Curve
        fpr = results['roc_curve']['fpr']
        tpr = results['roc_curve']['tpr']
        auc = results['metrics']['auc_roc']

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC={auc:.4f}')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Precision-Recall Curve (if available)
        pr_curve = results.get('pr_curve', {})
        pr_precision = pr_curve.get('precision')
        pr_recall = pr_curve.get('recall')
        if pr_precision and pr_recall:
            ap = results['metrics'].get('pr_auc', 0.0)
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(pr_recall, pr_precision, 'g-', linewidth=2, label=f'AP={ap:.4f}')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig)
            plt.close(fig)

        # Page 4: Per-class metrics
        fig = TrainingVisualizer.plot_per_class_metrics(
            results['per_class_metrics'],
            return_fig=True,
            show=False,
            close=False
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 5: Error distribution
        fig = ErrorAnalysisVisualizer.plot_error_distribution(
            results['predictions'],
            return_fig=True,
            show=False,
            close=False
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Page 6: Threshold analysis (use evaluator output if present)
        threshold_curves = results.get('threshold_analysis', {}).get('curves', {})
        if threshold_curves and threshold_curves.get('thresholds'):
            thr_values = np.array(threshold_curves['thresholds'])
            thr_precision = np.array(threshold_curves.get('precision', []))
            thr_recall = np.array(threshold_curves.get('recall', []))
            thr_f1 = np.array(threshold_curves.get('f1', []))
            thr_f2 = np.array(threshold_curves.get('f2', []))
            selected_thr = results.get('threshold', {}).get('value', 0.5)

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(thr_values, thr_precision, linewidth=2, label='Precision')
            ax.plot(thr_values, thr_recall, linewidth=2, label='Recall')
            ax.plot(thr_values, thr_f1, linewidth=2, label='F1')
            if thr_f2.size > 0:
                ax.plot(thr_values, thr_f2, linewidth=2, label='F2')
            ax.axvline(selected_thr, color='black', linestyle='--', linewidth=1, label=f'Selected={selected_thr:.3f}')
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Score')
            ax.set_title('Threshold Trade-off Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            labels = np.array(results['predictions']['labels'])
            probs = np.array(results['predictions']['probabilities'])
            fig, _opt_thr, _max_f1 = ErrorAnalysisVisualizer.plot_threshold_analysis(
                labels,
                probs,
                return_fig=True,
                show=False,
                close=False
            )
        pdf.savefig(fig)
        plt.close(fig)

    print(f"âœ… Comprehensive report saved: {output_path}")
