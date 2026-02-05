"""
Model Evaluator - Test & Evaluation Module
Trained modellerin detayli degerlendirilmesi icin
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, matthews_corrcoef
)
import json
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Model degerlendirme ve test sinifi"""

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Args:
            model: Degerlendirilecek model
            device: 'cuda' veya 'cpu'
        """
        self.model = model
        # Match trainer device selection (fallback to CPU if GPU unsupported)
        if device == 'cuda' and torch.cuda.is_available():
            try:
                cuda_capability = torch.cuda.get_device_capability()
                capability_version = float(f"{cuda_capability[0]}.{cuda_capability[1]}")
                if capability_version >= 12.0:
                    arch_list = torch.cuda.get_arch_list()
                    if cuda_capability[1] == 0:
                        sm_arch = f"sm_{cuda_capability[0]}{cuda_capability[1]}"
                    else:
                        sm_arch = f"sm_{cuda_capability[0]}{cuda_capability[1]:02d}"
                    if sm_arch in arch_list:
                        self.device = device
                    else:
                        self.device = 'cpu'
                else:
                    self.device = device
            except Exception:
                self.device = 'cpu'
        else:
            self.device = 'cpu'
        self.model.to(self.device)
        self.model.eval()

        self.results = {}

    @torch.no_grad()
    def _collect_outputs(self, loader: DataLoader, desc: str = "Evaluating"):
        """Collect labels, probabilities, and roi_ids from a loader."""
        all_labels = []
        all_probs = []
        all_roi_ids = []

        for batch in tqdm(loader, desc=desc):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            roi_ids = batch.get('roi_id', ['unknown'] * len(labels))

            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_roi_ids.extend(roi_ids)

        return np.array(all_labels), np.array(all_probs), all_roi_ids

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        save_dir: Optional[str] = None,
        threshold_strategy: str = 'fixed',
        threshold: float = 0.5,
        beta: float = 2.0,
        min_precision: Optional[float] = None,
        threshold_loader: Optional[DataLoader] = None,
        threshold_selection: Optional[str] = None
    ) -> Dict:
        """
        Model'i test set'te degerlendir

        Args:
            test_loader: Test dataloader
            save_dir: Sonuclarin kaydedilecegi dizin
            threshold_strategy: 'fixed', 'f_beta', 'recall_at_precision'
            threshold: Sabit threshold (fixed icin)
            beta: F-beta icin beta degeri
            min_precision: recall_at_precision veya f_beta icin minimum precision
            threshold_loader: Threshold secimi icin kullanilacak loader (opsiyonel)
            threshold_selection: Threshold secimi icin set etiketi (opsiyonel)

        Returns:
            Dict: Comprehensive evaluation results
        """
        print(f"\n{'='*70}")
        print("MODEL EVALUATION")
        print(f"{'='*70}\n")

        # Threshold selection data
        thr_loader = threshold_loader or test_loader
        thr_labels, thr_probs, _ = self._collect_outputs(thr_loader, desc="Selecting threshold")

        threshold_value = self._select_threshold(
            labels=thr_labels,
            probs=thr_probs,
            strategy=threshold_strategy,
            default=threshold,
            beta=beta,
            min_precision=min_precision
        )

        # Evaluation data (test)
        all_labels, all_probs, all_roi_ids = self._collect_outputs(test_loader, desc="Evaluating")
        all_preds = (all_probs >= threshold_value).astype(int)

        print("\nMetrics hesaplanıyor...")
        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)

        print("\nSinif bazinda metrikler...")
        per_class_metrics = self._calculate_per_class_metrics(all_labels, all_preds)

        print("\nConfusion matrix olusturuluyor...")
        cm = confusion_matrix(all_labels, all_preds)

        print("\nROC curve hesaplanıyor...")
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

        self.results = {
            'metrics': metrics,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            },
            'predictions': {
                'roi_ids': all_roi_ids,
                'labels': all_labels.tolist(),
                'predictions': all_preds.tolist(),
                'probabilities': all_probs.tolist()
            },
            'threshold': {
                'strategy': threshold_strategy,
                'value': float(threshold_value),
                'beta': float(beta),
                'min_precision': None if min_precision is None else float(min_precision),
                'default': float(threshold),
                'selection_set': threshold_selection or ('test' if threshold_loader is None else 'custom')
            }
        }

        self._print_results(metrics, per_class_metrics, cm)

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self._save_results(save_dir)
            self._save_plots(save_dir, cm, fpr, tpr)

        return self.results

    def _select_threshold(
        self,
        labels: np.ndarray,
        probs: np.ndarray,
        strategy: str = 'fixed',
        default: float = 0.5,
        beta: float = 2.0,
        min_precision: Optional[float] = None
    ) -> float:
        """Select decision threshold based on strategy."""
        strategy = (strategy or 'fixed').lower()
        if strategy in ['fixed', 'default']:
            return float(default)

        thresholds = np.linspace(0.0, 1.0, 101)
        best_thr = float(default)
        best_score = -1.0

        for thr in thresholds:
            preds = (probs >= thr).astype(int)
            precision, recall, _f1, _ = precision_recall_fscore_support(
                labels, preds, average='binary', zero_division=0, pos_label=1
            )

            if min_precision is not None and precision < min_precision:
                continue

            if strategy in ['f_beta', 'fbeta']:
                denom = (beta ** 2 * precision + recall)
                score = (1 + beta ** 2) * precision * recall / denom if denom > 0 else 0.0
            elif strategy in ['recall_at_precision', 'recall']:
                score = recall
            else:
                return float(default)

            if score > best_score:
                best_score = score
                best_thr = float(thr)

        return best_thr

    def _calculate_metrics(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        probs: np.ndarray
    ) -> Dict:
        """Overall metrics hesapla"""

        accuracy = accuracy_score(labels, preds)

        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0
        )

        precision_b, recall_b, f1_b, _ = precision_recall_fscore_support(
            labels, preds, average='binary', zero_division=0
        )

        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            auc = 0.0

        mcc = matthews_corrcoef(labels, preds)

        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision_w),
            'recall_weighted': float(recall_w),
            'f1_weighted': float(f1_w),
            'precision_binary': float(precision_b),
            'recall_binary': float(recall_b),
            'f1_binary': float(f1_b),
            'auc_roc': float(auc),
            'mcc': float(mcc),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }

    def _calculate_per_class_metrics(
        self,
        labels: np.ndarray,
        preds: np.ndarray
    ) -> Dict:
        """Per-class metrics"""

        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0
        )

        per_class = {}
        class_names = ['Normal', 'Anomaly']

        for i, name in enumerate(class_names):
            per_class[name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }

        return per_class

    def _print_results(
        self,
        metrics: Dict,
        per_class_metrics: Dict,
        cm: np.ndarray
    ):
        """Sonuclari yazdir"""

        print(f"\n{'='*70}")
        print("EVALUATION RESULTS")
        print(f"{'='*70}\n")

        if 'threshold' in self.results:
            thr = self.results['threshold']
            print(f"\nDecision Threshold: {thr.get('value', 0.5):.3f} (strategy={thr.get('strategy')})")

        print("Overall Metrics:")
        print(f"  Accuracy:        {metrics['accuracy']:.4f}")
        print(f"  Precision (w):   {metrics['precision_weighted']:.4f}")
        print(f"  Recall (w):      {metrics['recall_weighted']:.4f}")
        print(f"  F1-Score (w):    {metrics['f1_weighted']:.4f}")
        print(f"  AUC-ROC:         {metrics['auc_roc']:.4f}")
        print(f"  MCC:             {metrics['mcc']:.4f}")
        print(f"  Sensitivity:     {metrics['sensitivity']:.4f}")
        print(f"  Specificity:     {metrics['specificity']:.4f}")

        print("\nBinary Classification (Anomaly Detection):")
        print(f"  Precision:       {metrics['precision_binary']:.4f}")
        print(f"  Recall:          {metrics['recall_binary']:.4f}")
        print(f"  F1-Score:        {metrics['f1_binary']:.4f}")

        print("\nConfusion Matrix:")
        print(f"  TP (True Positive):  {metrics['tp']}")
        print(f"  TN (True Negative):  {metrics['tn']}")
        print(f"  FP (False Positive): {metrics['fp']}")
        print(f"  FN (False Negative): {metrics['fn']}")

        print("\nPer-Class Metrics:")
        for class_name, class_metrics in per_class_metrics.items():
            print(f"\n  {class_name}:")
            print(f"    Precision:  {class_metrics['precision']:.4f}")
            print(f"    Recall:     {class_metrics['recall']:.4f}")
            print(f"    F1-Score:   {class_metrics['f1']:.4f}")
            print(f"    Support:    {class_metrics['support']}")

        print(f"\n{'='*70}\n")

    def _save_results(self, save_dir: Path):
        """Sonuclari JSON olarak kaydet"""

        metrics_path = save_dir / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'metrics': self.results['metrics'],
                'per_class_metrics': self.results['per_class_metrics'],
                'threshold': self.results.get('threshold')
            }, f, indent=2)
        print(f"Metrics saved: {metrics_path}")

        full_results_path = save_dir / 'evaluation_results_full.json'
        with open(full_results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Full results saved: {full_results_path}")

        labels = np.array(self.results['predictions']['labels'])
        preds = np.array(self.results['predictions']['predictions'])
        report = classification_report(
            labels, preds,
            target_names=['Normal', 'Anomaly'],
            digits=4
        )
        report_path = save_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Classification report saved: {report_path}")

    def _save_plots(
        self,
        save_dir: Path,
        cm: np.ndarray,
        fpr: np.ndarray,
        tpr: np.ndarray
    ):
        """Visualization plots kaydet"""

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly']
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        cm_path = save_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix plot saved: {cm_path}")

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f"AUC = {self.results['metrics']['auc_roc']:.4f}")
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        roc_path = save_dir / 'roc_curve.png'
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved: {roc_path}")

    def compare_models(
        self,
        models: Dict[str, nn.Module],
        test_loader: DataLoader,
        save_dir: Optional[str] = None
    ) -> Dict:
        """Birden fazla modeli karsilastir"""
        print(f"\n{'='*70}")
        print("MODEL COMPARISON")
        print(f"{'='*70}\n")

        comparison_results = {}

        for model_name, model in models.items():
            print(f"\nEvaluating: {model_name}")
            print("-" * 70)

            original_model = self.model
            self.model = model
            self.model.to(self.device)
            self.model.eval()

            results = self.evaluate(test_loader, save_dir=None)
            comparison_results[model_name] = results['metrics']

            self.model = original_model

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            comparison_path = save_dir / 'model_comparison.json'
            with open(comparison_path, 'w') as f:
                json.dump(comparison_results, f, indent=2)
            print(f"Comparison results saved: {comparison_path}")

        return comparison_results
