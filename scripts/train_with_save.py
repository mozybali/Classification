"""
Model Training & Evaluation with Complete Saving System
Eğitim sırasında ve sonrasında tüm metrikleri ve grafikleri kaydeder
"""

import yaml
import torch
import numpy as np
from pathlib import Path
import time
from sklearn.metrics import roc_curve, confusion_matrix

from src.models import ModelFactory
from src.preprocessing.preprocess import DataPreprocessor
from src.training.modular_trainer import ModularTrainer
from src.training.evaluator import ModelEvaluator
from src.utils.model_manager import ModelManager, save_complete_model_results


def train_and_save_model(config_path: str = 'configs/config.yaml'):
    """
    Model eğit ve tüm sonuçları kaydet
    
    Args:
        config_path: Config dosyası yolu
    """
    # Config yükle
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*70)
    print("MODEL TRAINING WITH COMPLETE SAVING SYSTEM")
    print("="*70 + "\n")
    
    # Model Manager oluştur
    model_manager = ModelManager(base_dir="outputs/trained_models")
    
    # Model adı
    model_name = config['model']['model_type']
    model_dir = model_manager.create_model_directory(model_name)
    
    print(f"📁 Model dizini oluşturuldu: {model_dir}\n")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {device}\n")
    
    # Data loaders oluştur
    print("📊 Data loaders hazırlanıyor...")
    preprocessor = DataPreprocessor(config)
    train_loader, val_loader, test_loader = preprocessor.create_dataloaders()
    print(f"✅ Data loaders hazır")
    print(f"   Train samples: {len(train_loader.dataset)}")
    print(f"   Val samples: {len(val_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}\n")
    
    # Model oluştur
    print(f"🧠 Model oluşturuluyor: {model_name}")
    model = ModelFactory.create_model(config['model'])
    model.to(device)
    
    # Parametre sayısı
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Toplam parametre: {total_params:,}\n")
    
    # Training config
    training_config = config['training']
    training_config['save_dir'] = str(model_dir / 'checkpoints')
    
    # Trainer oluştur
    print("🚀 Training başlatılıyor...\n")
    trainer = ModularTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device
    )
    
    # Training başlat (süre ölç)
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    print(f"\n✅ Training tamamlandı! Süre: {training_time/60:.2f} dakika\n")
    
    # Training history al
    train_history = trainer.train_history
    val_history = trainer.val_history
    
    # Test evaluation
    print("📊 Test seti değerlendiriliyor...\n")
    evaluator = ModelEvaluator(model, device)
    
    # Test predictions
    all_labels = []
    all_preds = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Test metrics hesapla
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        roc_auc_score, matthews_corrcoef
    )
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    auc_score = roc_auc_score(all_labels, all_probs)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    test_metrics = {
        'accuracy': float(accuracy),
        'precision_weighted': float(precision),
        'recall_weighted': float(recall),
        'f1_weighted': float(f1),
        'auc_roc': float(auc_score),
        'mcc': float(mcc)
    }
    
    print("📈 Test Metrikleri:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   AUC-ROC: {auc_score:.4f}")
    print(f"   MCC: {mcc:.4f}\n")
    
    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # TÜM SONUÇLARI KAYDET
    print("💾 Tüm sonuçlar kaydediliyor...\n")
    save_complete_model_results(
        model_manager=model_manager,
        model_dir=model_dir,
        model=model,
        model_name=model_name,
        optimizer=trainer.optimizer,
        train_history=train_history,
        val_history=val_history,
        test_metrics=test_metrics,
        config=config,
        fpr=fpr,
        tpr=tpr,
        auc_score=auc_score,
        cm=cm,
        training_time=training_time
    )
    
    print("\n" + "="*70)
    print("✅ İŞLEM TAMAMLANDI!")
    print("="*70)
    print(f"\n📂 Model sonuçları: {model_dir}")
    print(f"\n📊 İçerik:")
    print(f"   • checkpoints/best_model.pth - En iyi model")
    print(f"   • metrics/training_history.json - Training metrikleri")
    print(f"   • metrics/best_metrics.json - En iyi metrikler")
    print(f"   • metrics/training_metrics.csv - CSV formatında")
    print(f"   • plots/training_curves.png - Training grafikleri")
    print(f"   • plots/loss_curve.png - Loss eğrisi")
    print(f"   • plots/accuracy_curve.png - Accuracy eğrisi")
    print(f"   • plots/roc_curve.png - ROC eğrisi")
    print(f"   • plots/confusion_matrix.png - Confusion matrix")
    print(f"   • MODEL_REPORT.md - Detaylı rapor")
    print(f"\n" + "="*70 + "\n")
    
    return model_dir, test_metrics


def compare_all_models():
    """
    Eğitilmiş tüm modelleri karşılaştır
    """
    print("\n" + "="*70)
    print("MODEL KARŞILAŞTIRMASI")
    print("="*70 + "\n")
    
    model_manager = ModelManager(base_dir="outputs/trained_models")
    model_manager.compare_models()
    
    print("\n✅ Karşılaştırma tamamlandı!")
    print("📂 Sonuçlar: outputs/trained_models/model_comparison/\n")


def load_model_and_evaluate(model_dir: str, test_loader):
    """
    Kaydedilmiş modeli yükle ve değerlendir
    
    Args:
        model_dir: Model dizini
        test_loader: Test dataloader
    """
    from src.models import load_model_from_checkpoint
    
    model_dir = Path(model_dir)
    checkpoint_path = model_dir / "checkpoints" / "best_model.pth"
    
    if not checkpoint_path.exists():
        print(f"❌ Model bulunamadı: {checkpoint_path}")
        return None
    
    print(f"\n📦 Model yükleniyor: {checkpoint_path}\n")
    
    # Config'i yükle
    config_path = model_dir / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Model yükle
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = ModelFactory.create_model(config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✅ Model yüklendi\n")
    
    # Evaluate
    evaluator = ModelEvaluator(model, device)
    results = evaluator.evaluate(test_loader, save_dir=str(model_dir / "evaluation"))
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Training & Evaluation System")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Config file path')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all trained models')
    parser.add_argument('--load', type=str, default=None,
                       help='Load and evaluate a specific model directory')
    
    args = parser.parse_args()
    
    if args.compare:
        # Tüm modelleri karşılaştır
        compare_all_models()
    elif args.load:
        # Belirli bir modeli yükle ve değerlendir
        # Test loader gerekli (örnek olarak)
        print("⚠️  Test loader gerekli. Lütfen scripti düzenleyin.")
    else:
        # Yeni model eğit
        train_and_save_model(args.config)
