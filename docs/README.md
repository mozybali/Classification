# 🏥 Medical Image Analysis System

**3D Tıbbi Görüntü Analizi ve Derin Öğrenme Tabanlı Anomali Tespit Sistemi**

Bu proje, 3D tıbbi görüntüler üzerinde derin öğrenme algoritmaları kullanarak anomali tespiti yapmak için geliştirilmiş kapsamlı bir sistemdir. Modüler mimari, çoklu model desteği ve detaylı değerlendirme araçlarıyla akademik araştırma ve uygulamalar için tasarlanmıştır.

---

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Sistem Gereksinimleri](#-sistem-gereksinimleri)
- [Kurulum](#-kurulum)
- [Proje Yapısı](#-proje-yapısı)
- [Hızlı Başlangıç](#-hızlı-başlangıç)
- [Kullanım](#-kullanım)
- [Konfigürasyon](#-konfigürasyon)
- [Desteklenen Modeller](#-desteklenen-modeller)
- [Değerlendirme Metrikleri](#-değerlendirme-metrikleri)
- [Geliştirme](#-geliştirme)

---

## 🌟 Özellikler

### 🧠 Derin Öğrenme Modelleri

**CNN (Convolutional Neural Networks) Modelleri:**
- **CNN3DSimple**: Temel 3D konvolüsyonel model
- **ResNet3D**: Residual bağlantılı derin 3D ağ
- **DenseNet3D**: Yoğun bağlantılı 3D mimari

**GNN (Graph Neural Networks) Modelleri:**
- **GCN (Graph Convolutional Network)**: Graf konvolüsyon tabanlı sınıflandırma
- **GAT (Graph Attention Network)**: Dikkat mekanizmalı graf öğrenme
- **GraphSAGE**: Örnek tabanlı graf öğrenme

### 🔄 Gelişmiş Veri İşleme

**Veri Önişleme:**
- 3D binary mask işleme (128×128×128)
- Otomatik NaN (eksik veri) kontrolü ve düzeltme
- Çoklu veri bölme stratejileri (simple, stratified, patient-level)
- Normalize edilebilir veri akışı

**Veri Artırma (Augmentation):**
- 3D uzamsal transformasyonlar (flip, rotation, zoom)
- Elastik deformasyon (medikal görüntü realitesi için)
- Yoğunluk normalizasyonu
- ROI (Region of Interest) kırpma
- Üç seviye augmentation: light, normal, heavy

**Sınıf Dengeleme:**
- Oversampling (SMOTE, Random)
- Undersampling (Random, Tomek Links, Edited NN)
- Class weights otomasyonu

### 🎯 Eğitim Sistemi

- **Modüler yapı**: Kolay özelleştirme ve genişletme
- **Mixed Precision Training**: Bellek optimizasyonu (AMP)
- **Çoklu optimizer desteği**: Adam, AdamW, SGD
- **Learning rate schedulers**: Cosine, Step, Plateau, Exponential
- **Early stopping**: Aşırı öğrenmeyi önleme
- **Checkpoint yönetimi**: Otomatik model kaydetme
- **TensorBoard entegrasyonu**: Gerçek zamanlı görselleştirme
- **Gradient clipping**: Eğitim stabilitesi

### 📊 Değerlendirme ve Analiz

**Metrikler:**
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC, AUC-PR
- Matthews Correlation Coefficient (MCC)
- Specificity, NPV (Negative Predictive Value)
- Confusion Matrix

**Analiz Araçları:**
- K-Fold cross-validation
- Model karşılaştırma sistemi
- Detaylı istatistiksel raporlar
- PDF rapor otomasyonu
- Eğitim eğrileri görselleştirme
- ROC ve PR curve grafikleri

**Veri Analizi:**
- İstatistiksel dataset keşfi
- Pattern ve outlier tespiti
- İnteraktif dashboard (Streamlit desteği)
- Korelasyon ve dağılım analizleri

### 🎨 Görselleştirme

- Training/validation loss ve accuracy grafikleri
- Confusion matrix heatmap
- ROC curves ve AUC hesaplama
- Precision-Recall curves
- Model karşılaştırma grafikleri
- Dataset istatistikleri görselleştirme

### 💾 Model ve Performans Kayıt Sistemi

**Otomatik Kaydetme Özellikleri:**
- **Model Checkpoints**: Best model, last checkpoint, periodic saves
- **Training Metrics**: JSON, CSV formatlarında kayıt
- **Grafik Kaydetme**: 
  - Training curves (loss, accuracy, F1, AUC)
  - ROC curve (AUC skoru ile birlikte) - **Her model için ayrı**
  - Confusion matrix (normal ve normalized)
- **Organize Dizin Yapısı**: Her model için timestamp'li klasör
- **Karşılaştırma Sistemi**: Tüm modellerin metriklerini yan yana görüntüleme
- **Detaylı Raporlar**: Markdown ve JSON formatlarında

**Kayıt Yapısı:**
```
outputs/trained_models/
├── resnet3d_20260125_120000/
│   ├── checkpoints/
│   │   ├── best_model.pth
│   │   └── last_checkpoint.pth
│   ├── metrics/
│   │   ├── training_history.json
│   │   ├── best_metrics.json
│   │   ├── training_metrics.csv
│   │   └── roc_data.json
│   ├── plots/
│   │   ├── training_curves.png
│   │   ├── loss_curve.png
│   │   ├── accuracy_curve.png
│   │   ├── f1_curve.png
│   │   ├── auc_curve.png
│   │   ├── roc_curve.png (AUC ile)
│   │   ├── confusion_matrix.png
│   │   └── confusion_matrix_normalized.png
│   ├── config.yaml
│   ├── model_summary.txt
│   └── MODEL_REPORT.md
└── model_comparison/
    ├── model_comparison.json
    └── model_comparison.png
```

---

## 💻 Sistem Gereksinimleri

### Minimum Gereksinimler
- **İşletim Sistemi**: Windows 10/11, Linux, macOS
- **Python**: 3.8 veya üzeri
- **RAM**: En az 8 GB (16 GB önerilir)
- **Depolama**: 5 GB boş alan

### GPU Desteği (Opsiyonel ama Önerilir)
- **NVIDIA GPU** (CUDA 11.0+ destekli)
- **VRAM**: En az 4 GB (8+ GB önerilir)

> **⚠️ RTX 5050 Kullanıcıları İçin Önemli Not:**
> 
> RTX 5050 GPU'lar (sm_120 CUDA capability) henüz PyTorch tarafından desteklenmemektedir.
> Bu GPU ile sistem otomatik olarak CPU moduna geçecektir. Training CPU'da daha yavaş
> olacaktır ancak tamamen işlevseldir. PyTorch'un sm_120 desteği eklemesi beklenmektedir.
> 
> Detaylar için: [FIX_RTX5050_CUDA.md](FIX_RTX5050_CUDA.md)

---

## 🚀 Kurulum

### 1. Repository'yi Klonlama

```bash
git clone <repository-url>
cd Tez
```

### 2. Sanal Ortam Oluşturma (Önerilir)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

### 3. Bağımlılıkları Yükleme

```bash
pip install -r requirements.txt
```

**PyTorch Kurulumu (GPU desteği için):**

```bash
# CUDA 12.1 (Güncel)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU Only
pip install torch torchvision torchaudio
```

### 4. Dataset Hazırlama

Dataset'inizi `NeAR_dataset/` klasörüne yerleştirin:

```
NeAR_dataset/
├── ALAN/
│   ├── info.csv          # Metadata dosyası
│   └── *.npy             # 3D görüntü dosyaları (128×128×128)
```

**`info.csv` formatı:**
```csv
ROI_id,ROI_file,ROI_anomaly,subset
001,data_001.npy,0,train
002,data_002.npy,1,train
...
```

---

## 📁 Proje Yapısı

```
Tez/
│
├── 📂 NeAR_dataset/              # Dataset klasörü
│   ├── ALAN/                     # Ana dataset
│   │   ├── info.csv              # Metadata
│   │   └── *.npy                 # 3D binary mask dosyaları
│   └── synthetic_data/           # Test verileri
│
├── 📂 src/                       # Kaynak kod modülleri
│   ├── data_analysis/            # Veri analizi modülleri
│   │   ├── explore_data.py       # Dataset keşfi
│   │   ├── detailed_analysis.py  # İstatistiksel analiz
│   │   └── interactive_dashboard.py  # Streamlit dashboard
│   │
│   ├── preprocessing/            # Veri önişleme
│   │   ├── preprocess.py         # Ana preprocessor
│   │   ├── image_loader.py       # Görüntü yükleme
│   │   ├── image_transforms.py   # 3D augmentation
│   │   ├── medical_transforms.py # Medikal transformlar
│   │   ├── nan_handler.py        # Eksik veri yönetimi
│   │   ├── data_splitter.py      # Train/val/test bölme
│   │   ├── class_balancer.py     # Sınıf dengeleme
│   │   ├── pipeline_builder.py   # Pipeline oluşturma
│   │   └── dataloader_factory.py # DataLoader factory
│   │
│   ├── models/                   # Model mimarileri
│   │   ├── base_model.py         # Abstract base class
│   │   ├── cnn_models.py         # CNN mimarileri
│   │   ├── gnn_models.py         # GNN mimarileri
│   │   └── model_factory.py      # Model factory pattern
│   │
│   ├── training/                 # Eğitim ve değerlendirme
│   │   ├── modular_trainer.py    # Modüler trainer
│   │   ├── evaluator.py          # Test evaluation
│   │   ├── cross_validator.py    # K-Fold CV
│   │   └── train.py              # Legacy trainer
│   │
│   └── utils/                    # Yardımcı araçlar
│       ├── helpers.py            # Genel fonksiyonlar
│       ├── visualization.py      # Görselleştirme
│       ├── model_manager.py      # Model kaydetme/yükleme
│       └── image_processing_utils.py  # Görüntü utils
│
├── 📂 cli/                       # Komut satırı arayüzleri
│   ├── run_training.py           # Eğitim başlatma
│   ├── run_evaluation.py         # Model değerlendirme
│   ├── run_hyperparameter_optimization.py  # Hiperparametre optimizasyonu
│   ├── class_balance_menu.py     # Sınıf dengeleme menüsü
│   └── data_preprocessing_menu.py  # Veri önişleme menüsü
│
├── 📂 tools/                     # Yardımcı araçlar
│   ├── analyze_dataset.py        # Dataset analizi
│   ├── analyze.py                # Hızlı analiz
│   ├── inspect_images.py         # Görüntü inceleme
│   ├── visualize_model_comparison.py  # Model karşılaştırma
│   └── verify_consistency.py     # Tutarlılık kontrolü
│
├── 📂 scripts/                   # Test ve kurulum scriptleri
│   ├── test_cpu_training.py      # CPU eğitim testi
│   ├── test_cuda.py              # CUDA testi
│   ├── test_setup.py             # Kurulum testi
│   ├── check_sm120_support.py    # RTX 5050 kontrol
│   ├── fix_rtx5050.bat           # RTX 5050 düzeltme
│   ├── install_cuda_pytorch.bat  # CUDA kurulum
│   ├── use_cpu_fallback.py       # CPU fallback
│   └── train_with_save.py        # Model kayıtlı eğitim
│
├── 📂 docs/                      # Dokümantasyon
│   ├── README.md                 # Bu dosya
│   ├── TRAINING_GUIDE.py         # Eğitim rehberi
│   └── QUICK_START_MODEL_MANAGER.py  # Model manager rehberi
│
├── 📂 configs/                   # Konfigürasyon dosyaları
│   └── config.yaml               # Ana config
│
├── 📂 notebooks/                 # Jupyter notebook'lar
│   └── 01_data_exploration.ipynb
│
├── 📂 outputs/                   # Çıktılar
│   ├── trained_models/           # Kaydedilen modeller
│   ├── plots/                    # Grafikler
│   ├── splits/                   # Veri split kayıtları
│   └── reports/                  # Raporlar
│
├── 📄 main.py                    # Ana menü ve GUI
├── 📄 requirements.txt           # Python bağımlılıkları
└── 📄 config.yaml                # Konfigürasyon
```

---

## ⚡ Hızlı Başlangıç

### İnteraktif Menü ile Çalıştırma

```bash
python main.py
```

Ana menü seçenekleri:
- **[1]** Temel Veri Analizi
- **[2]** Detaylı İstatistiksel Analiz
- **[3]** Örnek Veri Görüntüleme
- **[4]** Görüntü İstatistikleri
- **[5]** Transform Testleri
- **[6]** Veri Ön İşleme
- **[7]** Model Eğitimi
- **[8]** Model Değerlendirme
- **[9]** Tüm Pipeline (Analiz + Eğitim)

### Komut Satırı Kullanımı

**1. Dataset Analizi:**
```bash
python tools/analyze_dataset.py
```

**2. Model Eğitimi:**
```bash
python cli/run_training.py --config configs/config.yaml
```

**3. Model Değerlendirme:**
```bash
python cli/run_evaluation.py --checkpoint outputs/trained_models/resnet3d_*/checkpoints/best_model.pth
```

**4. Hiperparametre Optimizasyonu:**
```bash
python cli/run_hyperparameter_optimization.py
```

**5. Model Karşılaştırma Görselleştirme:**
```bash
python tools/visualize_model_comparison.py
```

**6. Sınıf Dengeleme Menüsü:**
```bash
python cli/class_balance_menu.py
```

**7. Veri Önişleme Menüsü:**
```bash
python cli/data_preprocessing_menu.py
```

---

## 🛠️ Kullanım

### 1. Model Eğitimi ve Otomatik Kaydetme

Yeni model manager sistemi ile tüm sonuçlar otomatik kaydedilir:

```python
from train_with_save import train_and_save_model

# Model eğit ve tüm sonuçları kaydet
model_dir, test_metrics = train_and_save_model('configs/config.yaml')

# Kaydedilen içerik:
# - checkpoints/best_model.pth - En iyi model
# - metrics/training_history.json - Tüm metrikler
# - metrics/training_metrics.csv - CSV formatında
# - plots/training_curves.png - Training grafikleri
# - plots/roc_curve.png - ROC eğrisi (AUC ile)
# - plots/confusion_matrix.png - Confusion matrix
# - MODEL_REPORT.md - Detaylı rapor
```

### 2. Veri Analizi

Dataset'inizi anlamak için analiz araçlarını kullanın:

```python
from src.data_analysis.explore_data import DatasetExplorer

explorer = DatasetExplorer('NeAR_dataset/ALAN')
explorer.analyze()
explorer.visualize_distributions()
```

### 3. Veri Önişleme

```python
from src.preprocessing.preprocess import DataPreprocessor

preprocessor = DataPreprocessor('configs/config.yaml')
train_loader, val_loader, test_loader = preprocessor.create_dataloaders()
```

### 4. Model Oluşturma

```python
from src.models import ModelFactory

config = {
    'model_type': 'resnet3d',
    'num_classes': 2,
    'in_channels': 1,
    'base_filters': 32,
    'dropout': 0.5
}

model = ModelFactory.create_model(config)
```

### 5. Model Eğitimi (Manuel)

```python
from src.training import ModularTrainer

trainer = ModularTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=training_config
)

trainer.train()
```

### 6. Model Değerlendirme ve Kaydetme

```python
from src.training import ModelEvaluator
from src.utils.model_manager import ModelManager

# Model manager oluştur
model_manager = ModelManager(base_dir="outputs/trained_models")
model_dir = model_manager.create_model_directory("resnet3d")

# Evaluate et
evaluator = ModelEvaluator(model, device='cuda')
results = evaluator.evaluate(test_loader, save_dir=str(model_dir / "evaluation"))

# ROC ve confusion matrix kaydet
from sklearn.metrics import roc_curve, confusion_matrix
import numpy as np

# Test predictions
all_labels, all_probs = [], []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        images = batch['image'].to('cuda')
        labels = batch['label']
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# ROC curve kaydet
fpr, tpr, _ = roc_curve(all_labels, all_probs)
auc_score = results['metrics']['auc_roc']
model_manager.save_roc_curve(model_dir, fpr, tpr, auc_score, "ResNet3D")

# Confusion matrix kaydet
cm = confusion_matrix(all_labels, all_probs > 0.5)
model_manager.save_confusion_matrix(model_dir, cm)

print(f"✅ Tüm sonuçlar kaydedildi: {model_dir}")
```

### 7. Tüm Modelleri Karşılaştırma

```python
from src.utils.model_manager import ModelManager

model_manager = ModelManager()
model_manager.compare_models()

# Çıktı:
# - outputs/trained_models/model_comparison/model_comparison.json
# - outputs/trained_models/model_comparison/model_comparison.png
# - Her modelin AUC, Accuracy, F1 gibi metrikleri yan yana
```

---

## ⚙️ Konfigürasyon

Tüm proje ayarları `configs/config.yaml` dosyasında merkezi olarak yönetilir.

### Temel Konfigürasyon Bölümleri

#### 1. Dataset Ayarları

```yaml
dataset:
  path: "NeAR_dataset/ALAN"
  csv_file: "info.csv"
  image_size: [128, 128, 128]
  channels: 1
  
  nan_handling:
    enabled: true
    method: "fill_mean"  # remove, fill_value, fill_mean, fill_median
  
  data_splitting:
    enabled: true
    method: "stratified"  # simple, stratified, patient
    train_ratio: 0.7
    val_ratio: 0.15
    test_ratio: 0.15
```

#### 2. Preprocessing

```yaml
preprocessing:
  normalize: true
  mean: 0.5
  std: 0.5
  
  augmentation:
    enabled: true
    mode: "normal"  # light, normal, heavy
    
    transforms:
      random_flip: {p: 0.5, axes: [0, 1, 2]}
      random_rotation: {p: 0.5, max_angle: 15}
      random_zoom: {p: 0.3, zoom_range: [0.9, 1.1]}
      elastic_deformation: {p: 0.2, alpha: 50, sigma: 5}
```

#### 3. Model Ayarları

```yaml
model:
  model_type: "resnet3d"  # cnn3d_simple, resnet3d, densenet3d, gcn, gat, graphsage
  num_classes: 2
  in_channels: 1
  base_filters: 32
  dropout: 0.5
```

#### 4. Training Ayarları

```yaml
training:
  epochs: 100
  batch_size: 8
  learning_rate: 0.001
  
  optimizer:
    type: "adam"  # adam, adamw, sgd
    weight_decay: 0.0001
  
  scheduler:
    type: "cosine"  # cosine, step, plateau, exponential
    patience: 10
  
  early_stopping:
    enabled: true
    patience: 15
    min_delta: 0.001
  
  mixed_precision: true
  gradient_clip: 1.0
```

---

## 🤖 Desteklenen Modeller

### CNN Modelleri

#### 1. CNN3DSimple
```yaml
model:
  model_type: "cnn3d_simple"
  base_filters: 32
  num_layers: 4
```
- Basit 3D konvolüsyonel bloklar
- Hızlı eğitim ve çıkarım
- Küçük dataset'ler için uygun

#### 2. ResNet3D
```yaml
model:
  model_type: "resnet3d"
  base_filters: 32
  num_blocks: [2, 2, 2, 2]
```
- Residual bağlantılar (skip connections)
- Derin ağlar için gradient flow
- Yüksek performans

#### 3. DenseNet3D
```yaml
model:
  model_type: "densenet3d"
  growth_rate: 32
  num_layers: [6, 12, 24, 16]
```
- Yoğun bağlantılar
- Feature reuse
- Parametre verimliliği

### GNN Modelleri

#### 1. GCN (Graph Convolutional Network)
```yaml
model:
  model_type: "gcn"
  hidden_dims: [64, 128]
  num_layers: 2
```
- Graf konvolüsyon operatörleri
- Komşu node bilgisi kullanımı

#### 2. GAT (Graph Attention Network)
```yaml
model:
  model_type: "gat"
  hidden_dims: [64, 128]
  num_heads: 4
```
- Attention mekanizması
- Adaptif node importance

#### 3. GraphSAGE
```yaml
model:
  model_type: "graphsage"
  hidden_dims: [64, 128]
  aggregator: "mean"  # mean, max, lstm
```
- Minibatch training
- Scalable graf öğrenme

---

## 📊 Değerlendirme Metrikleri

### Sınıflandırma Metrikleri

| Metrik | Açıklama | Kullanım |
|--------|----------|----------|
| **Accuracy** | Doğru tahmin oranı | Genel performans |
| **Precision** | Pozitif tahminlerde doğruluk | Yanlış pozitif kontrolü |
| **Recall** | Gerçek pozitifleri bulma | Yanlış negatif kontrolü |
| **F1-Score** | Precision-Recall dengesi | Genel metrik |
| **AUC-ROC** | ROC eğris (Otomatik Kaydetme ile)

```python
from train_with_save import train_and_save_model

# Tek satırda tüm pipeline - eğitim, test ve kaydetme
model_dir, test_metrics = train_and_save_model('configs/config.yaml')

# Otomatik olarak kaydedilir:
# ✅ Model checkpoint (best_model.pth)
# ✅ Training history (JSON + CSV)
# ✅ Training curves (loss, accuracy, F1, AUC)
# ✅ ROC curve (AUC ile birlikte)
# ✅ Confusion matrix (normal ve normalized)
# ✅ Model summary ve konfigürasyon
# ✅ Detaylı rapor (MODEL_REPORT.md)

print(f"Model kaydedildi: {model_dir}")
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}"tion-Ready
- Checkpoint yönetimi
- Error handling
- Logging sistemi
- Otomatik raporlama

### 3. Akademik Standartlar
- Cross-validation desteği
- Detaylı metrikler
- Reproducibility (random seed kontrolü)
- Comprehensive documentation
Çoklu Model Eğitimi ve Karşılaştırma

```python
from train_with_save import train_and_save_model, compare_all_models
import yaml

# Farklı modelleri eğit
models_to_train = ['cnn3d_simple', 'resnet3d', 'densenet3d']

for model_type in models_to_train:
    print(f"\n{'='*70}")
    print(f"Training: {model_type}")
    print(f"{'='*70}\n")
    
    # Config'i güncelle
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['model']['model_type'] = model_type
    
    # Temporary config kaydet
    with open('configs/temp_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Eğit ve kaydet
    model_dir, metrics = train_and_save_model('configs/temp_config.yaml')
    
    print(f"\n✅ {model_type} tamamlandı!")
    print(f"   Test AUC: {metrics['auc_roc']:.4f}")

# Tüm modelleri karşılaştır
print("\n" + "="*70)
print("TÜM MODELLERİ KARŞILAŞTIR")
print("="*70 + "\n")

compare_all_models()

# Çıktı:
# outputs/trained_models/
#   ├── cnn3d_simple_20260125_120000/
#   │   ├── plots/roc_curve.png (AUC ile)
#   │   ├── metrics/best_metrics.json
#   │   └── ...
#   ├── resnet3d_20260125_130000/
#   │   ├── plots/roc_curve.png (AUC ile)
#   │   ├── metrics/best_metrics.json
#   │   └── ...
#   ├── densenet3d_20260125_140000/
#   │   ├── plots/roc_curve.png (AUC ile)
#   │   ├── metrics/best_metrics.json
#   │   └── ...
#   └── model_comparison/
#       ├── model_comparison.json
#       └── model_comparison.png (Tüm modellerin karşılaştırması
# 1. Config yükle
with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

# 2. Data hazırla
preprocessor = DataPreprocessor(config)
train_loader, val_loader, test_loader = preprocessor.create_dataloaders()

# 3. Model oluştur
model = ModelFactory.create_model(config['model'])

# 4. Eğit
trainer = ModularTrainer(model, train_loader, val_loader, config['training'])
trainer.train()

# 5. Değerlendir
from src.training import Evaluator
evaluator = Evaluator(model, test_loader)
metrics = evaluator.evaluate()
print(metrics)
```

### Model Karşılaştırma

```python
models_to_compare = ['cnn3d_simple', 'resnet3d', 'densenet3d']
results = {}

for model_type in models_to_compare:
    config['model']['model_type'] = model_type
    model = ModelFactory.create_model(config['model'])
    
    trainer = ModularTrainer(model, train_loader, val_loader, config['training'])
    trainer.train()
    
    evaluator = Evaluator(model, test_loader)
    results[model_type] = evaluator.evaluate()

# Karşılaştırma grafiği
from src.utils.visualization import plot_model_comparison
plot_model_comparison(results, save_path='outputs/comparison.png')
```

---

## 🐛 Hata Ayıklama

### Sık Karşılaşılan Sorunlar

**1. CUDA Out of Memory**
```yaml
training:
  batch_size: 4  # Batch size'ı küçült
  mixed_precision: true  # AMP kullan
```

**2. Overfitting**
```yaml
model:
  dropout: 0.5  # Dropout artır

training:
  early_stopping:
    enabled: true
    patience: 10

preprocessing:
  augmentation:
    mode: "heavy"  # Augmentation artır
```

**3. Düşük Performans**
- Learning rate'i ayarlayın
- Model mimarisini değiştirin
- Daha fazla data augmentation kullanın
- Class balancing uygulayın

---

## 🔧 Geliştirme

### Yeni Model Ekleme

```python
# src/models/my_model.py
from .base_model import BaseModel
import torch.nn as nn

class MyCustomModel(BaseModel):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        # Model tanımı
        
    def forward(self, x):
        # Forward pass
        return x

# src/models/model_factory.py içinde kaydet:
MODEL_REGISTRY = {
    'my_custom_model': MyCustomModel,
    # ... diğer modeller
}
```

### Yeni Transform Ekleme

```python
# src/preprocessing/custom_transforms.py
class MyTransform:
    def __init__(self, param1, param2):
        self.param1 = param1
        
    def __call__(self, image):
        # Transform işlemi
        return transformed_image
```

### Testing

```bash
# Unit testler çalıştır
pytest src/preprocessing/test_transforms.py
pytest src/preprocessing/test_medical_transforms.py
```

---

## 📝 Best Practices

1. **Config Kullanımı**: Her zaman config.yaml üzerinden ayar yapın
2. **Seed Setting**: Reproducibility için seed sabitleyın
3. **Validation**: Her epoch'ta validation yapın
4. **Checkpointing**: Düzenli model kaydetme
5. **Logging**: TensorBoard ile takip edin
6. **Data Augmentation**: Overfitting'i önleyin
7. **Cross-Validation**: Son değerlendirmede kullanın

---

## 📄 Lisans

Bu proje akademik ve araştırma amaçlı geliştirilmiştir.

---

## 🤝 Katkıda Bulunma

Proje geliştirmeye katkıda bulunmak için:

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

---

## 📧 İletişim

Proje hakkında sorularınız için lütfen issue açın veya iletişime geçin.

---

## 🙏 Teşekkürler

Bu proje aşağıdaki açık kaynak projeleri kullanmaktadır:
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

**Son Güncelleme**: 2026

**Python Version**: 3.8+

**PyTorch Version**: 2.0+
