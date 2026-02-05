"""
NeAR Dataset - Böbrek Anomali Tespiti Projesi
Ana çalıştırma scripti
"""

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import os
import json


def _configure_console() -> None:
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


_configure_console()

# Modülleri import et
from src.data_analysis.explore_data import DatasetExplorer
from src.utils.helpers import (
    load_config, 
    save_config, 
    set_seed, 
    get_device,
    count_parameters,
    plot_training_history
)


def show_menu():
    """İnteraktif menü göster"""
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*70)
    print("    NeAR DATASET - BÖBREK ANOMALİ TESPİTİ PROJESİ")
    print("    🎯 Merkezi Yönetim Paneli")
    print("="*70)
    print("\n📊 VERİ ANALİZİ VE KEŞİF:")
    print("  [1] Temel Veri Analizi")
    print("  [2] Detaylı Veri Analizi (İstatistiksel Testler)")
    print("  [3] Örnek Veri Görüntüle")
    print("  [A] Detaylı Dataset Analizi (tools/analyze_dataset.py)")
    print("\n🖼️  GÖRÜNTÜ İŞLEME:")
    print("  [4] Görüntü İstatistikleri Hesapla")
    print("  [5] Görüntü Transform Testleri")
    print("\n🏥 TİBBİ GÖRÜNTÜ İŞLEME:")
    print("  [M] Medical Transform Testleri")
    print("\n🔧 VERİ ÖNİŞLEME VE DENGELEME:")
    print("  [6] Temel Veri Ön İşleme")
    print("  [B] Veri Önişleme Menüsü (NaN handling, splitting)")
    print("  [C] Sınıf Dengeleme Menüsü (class balance, augmentation)")
    print("\n🤖 MODEL YÖNETİMİ:")
    print("  [7] Model Eğitimi")
    print("  [8] Model Değerlendirme")
    print("  [D] Model Karşılaştırma ve Görselleştirme")
    print("\n🎯 HİPERPARAMETRE OPTİMİZASYONU:")
    print("  [H] Hiperparametre Optimizasyonu (Grid/Bayesian Search)")
    print("\n🧪 TEST VE KURULUM:")
    print("  [T] Sistem Test ve Kurulum Kontrolü")
    print("\n⚡ HIZLI İŞLEMLER:")
    print("  [9] Tüm Pipeline (Analiz + Eğitim)")
    print("\n  [0] Çıkış")
    print("="*70)
    
    choice = input("\n👉 Seçiminizi yapın: ").strip().upper()
    return choice


def get_menu_action(choice, config):
    """Menü seçimine göre aksiyon döndür"""
    actions = {
        '1': ('analyze', False),
        '2': ('analyze', True),
        '3': ('display', False),
        'A': ('detailed_dataset_analysis', False),
        '4': ('image_stats', False),
        '5': ('transform_test', False),
        'M': ('medical_test', False),
        '6': ('preprocess', False),
        'B': ('preprocessing_menu', False),
        'C': ('class_balance_menu', False),
        '7': ('train', False),
        '8': ('evaluate', False),
        'D': ('model_comparison', False),
        'H': ('hyperparameter_optimization', False),
        'T': ('test_setup', False),
        '9': ('all', True),
        '0': ('exit', False)
    }
    return actions.get(choice, (None, None))


def display_sample_data(config):
    """Örnek verileri görüntüle"""
    import pandas as pd
    
    print("\n" + "="*70)
    print("ÖRNEK VERİ GÖRÜNTÜLENİYOR")
    print("="*70)
    
    from src.preprocessing.preprocess import resolve_csv_path
    dataset_path = resolve_csv_path(config)
    
    try:
        # Veri setini yükle
        df = pd.read_csv(dataset_path)
        
        print(f"\n✓ Veri seti yüklendi: {len(df)} kayıt\n")
        
        # ROI_id'den ek bilgiler çıkar
        df['patient_id'] = df['ROI_id'].str[:5]  # İlk 5 karakter hasta ID
        df['laterality'] = df['ROI_id'].str[-1]  # Son karakter (L/R)
        df['label'] = df['ROI_anomaly'].astype(int)  # Boolean to int
        
        # Sütun bilgileri
        print("📋 SÜTUN BİLGİLERİ:")
        print("="*70)
        print(f"  • ROI_id: Böbrek ROI kimliği (hasta_id + laterality)")
        print(f"  • subset: Veri seti bölümü (train/test/dev)")
        print(f"  • ROI_anomaly: Anomali durumu (True/False)")
        print(f"  • patient_id: Hasta kimliği (çıkarıldı)")
        print(f"  • laterality: Böbrek tarafı - L: Sol, R: Sağ (çıkarıldı)")
        print(f"  • label: Anomali etiketi - 0: Normal, 1: Anomali (çıkarıldı)")
        print()
        
        # İlk 10 kayıt
        print("\n📊 İLK 10 KAYIT:")
        print("="*70)
        display_df = df[['ROI_id', 'patient_id', 'laterality', 'subset', 'ROI_anomaly', 'label']].head(10)
        print(display_df.to_string(index=True))
        print()
        
        # Anomalili kayıtlar
        anomaly_samples = df[df['ROI_anomaly'] == True][['ROI_id', 'patient_id', 'laterality', 'subset']].head(10)
        print("\n🔴 ANOMALİ ÖRNEK KAYITLAR (İlk 10):")
        print("="*70)
        print(anomaly_samples.to_string(index=True))
        print()
        
        # Normal kayıtlar
        normal_samples = df[df['ROI_anomaly'] == False][['ROI_id', 'patient_id', 'laterality', 'subset']].head(10)
        print("\n🟢 NORMAL ÖRNEK KAYITLAR (İlk 10):")
        print("="*70)
        print(normal_samples.to_string(index=True))
        print()
        
        # Rastgele 5 kayıt
        print("\n🔀 RASTGELE 5 KAYIT:")
        print("="*70)
        random_samples = df[['ROI_id', 'patient_id', 'laterality', 'subset', 'ROI_anomaly']].sample(5, random_state=42)
        print(random_samples.to_string())
        print()
        
        # Temel istatistikler
        print("\n📈 TEMEL İSTATİSTİKLER:")
        print("="*70)
        
        # Anomali dağılımı
        print(f"\n🎯 Anomali Dağılımı:")
        anomaly_counts = df['ROI_anomaly'].value_counts()
        print(f"  • Normal (False): {anomaly_counts.get(False, 0)} (%{anomaly_counts.get(False, 0)/len(df)*100:.2f})")
        print(f"  • Anomali (True): {anomaly_counts.get(True, 0)} (%{anomaly_counts.get(True, 0)/len(df)*100:.2f})")
        
        # Subset dağılımı
        print(f"\n📦 Subset Dağılımı:")
        for subset, count in df['subset'].value_counts().items():
            anomaly_in_subset = df[(df['subset'] == subset) & (df['ROI_anomaly'] == True)].shape[0]
            print(f"  • {subset}: {count} ROI ({anomaly_in_subset} anomali, %{anomaly_in_subset/count*100:.2f})")
        
        # Laterality dağılımı
        print(f"\n🔄 Laterality Dağılımı:")
        for lat, count in df['laterality'].value_counts().items():
            anomaly_in_lat = df[(df['laterality'] == lat) & (df['ROI_anomaly'] == True)].shape[0]
            lat_name = "Sol" if lat == 'L' else "Sağ"
            print(f"  • {lat_name} ({lat}): {count} ROI ({anomaly_in_lat} anomali, %{anomaly_in_lat/count*100:.2f})")
        
        # Hasta istatistikleri
        print(f"\n👤 Hasta İstatistikleri:")
        n_patients = df['patient_id'].nunique()
        print(f"  • Toplam hasta sayısı: {n_patients}")
        print(f"  • Hasta başına ortalama ROI: {len(df) / n_patients:.2f}")
        
        # Her iki böbrekte anomali olan hastalar
        patient_anomaly = df.groupby('patient_id')['ROI_anomaly'].sum()
        both_anomaly = (patient_anomaly == 2).sum()
        one_anomaly = (patient_anomaly == 1).sum()
        no_anomaly = (patient_anomaly == 0).sum()
        print(f"  • Her iki böbrek normal: {no_anomaly}")
        print(f"  • Tek böbrek anomalili: {one_anomaly}")
        print(f"  • Her iki böbrek anomalili: {both_anomaly}")
        
        # Eksik değer kontrolü
        print("\n\n🔍 EKSİK DEĞER KONTROLÜ:")
        print("="*70)
        missing = df[['ROI_id', 'subset', 'ROI_anomaly']].isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("✓ Eksik değer bulunmuyor!")
        
    except Exception as e:
        print(f"\n❌ Hata: {str(e)}")


def run_medical_transform_test(config):
    """
    Medical transform'ları test et ve sonuçları göster
    """
    print("\n" + "="*70)
    print("🏥 TİBBİ TRANSFORM TESTLERİ")
    print("="*70)
    
    try:
        from src.preprocessing.medical_transforms import (
            MedicalIntensityNormalization,
            AdaptiveROICrop,
            BinaryMaskProcessor,
            get_medical_kidney_pipeline
        )
        import numpy as np
        
        # Synthetic test mask oluştur
        print("\n🧪 Test mask'i oluşturuluyor...")
        mask = np.zeros((128, 128, 128), dtype=np.float32)
        mask[40:80, 40:80, 40:80] = 1  # Basit küp şeklinde kidney
        
        # Random noise ekle
        noise = np.random.rand(128, 128, 128) < 0.01
        mask[noise] = 1
        
        print(f"  Original: Shape={mask.shape}, Volume={mask.sum():.0f} voxels")
        
        # Test 1: Adaptive ROI Crop
        print("\n🔬 Test 1: Adaptive ROI Cropping")
        print("  - Non-zero bölgeyi otomatik tespit eder")
        print("  - Gereksiz padding'i kaldırır (memory optimization)")
        
        crop = AdaptiveROICrop(margin=10, min_size=32)
        cropped = crop(mask)
        
        memory_saved = (1 - cropped.size / mask.size) * 100
        print(f"  Result: Shape={cropped.shape}, Memory saved={memory_saved:.1f}%")
        
        # Test 2: Binary Mask Processing
        print("\n🧹 Test 2: Binary Mask Post-processing")
        print("  - Noise removal (küçük component'lar)")
        print("  - Hole filling")
        print("  - Morphological operations")
        
        processor = BinaryMaskProcessor(
            fill_holes=True,
            min_component_size=1000,
            morphology='closing'
        )
        
        cleaned = processor(mask)
        noise_removed = mask.sum() - cleaned.sum()
        print(f"  Result: Noise removed={noise_removed:.0f} voxels")
        
        # Test 3: Medical Intensity Normalization
        print("\n📊 Test 3: Medical Intensity Normalization")
        print("  - Z-score / Min-max normalization")
        print("  - Percentile-based outlier filtering")
        print("  - Binary mask'ler için otomatik skip")
        
        normalizer = MedicalIntensityNormalization(
            method='minmax',
            percentile_range=(1, 99),
            clip_output=True
        )
        
        # Intensity image simülasyonu
        intensity_img = np.random.randn(64, 64, 64) * 100 + 500
        normalized = normalizer(intensity_img)
        
        print(f"  Original: Min={intensity_img.min():.1f}, Max={intensity_img.max():.1f}")
        print(f"  Normalized: Min={normalized.min():.2f}, Max={normalized.max():.2f}")
        
        # Test 4: Complete Pipeline
        print("\n🔧 Test 4: Complete Medical Pipeline")
        print("  - ToFloat + AdaptiveCrop + MaskProcessing + Augmentation")
        
        pipeline = get_medical_kidney_pipeline(
            normalize_intensity=False,
            adaptive_crop=True,
            mask_processing=True,
            augmentation=False
        )
        
        processed = pipeline(mask)
        print(f"  Result: {mask.shape} → {processed.shape}")
        print(f"  Volume: {mask.sum():.0f} → {processed.sum():.0f} voxels")
        
        print("\n✅ Tüm testler başarıyla tamamlandı!")
        print("\n💡 Detaylı bilgi için src/preprocessing/test_medical_transforms.py dosyasını inceleyebilirsiniz.")
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()

def analyze_data(config, detailed=False):
    """Veri seti analizi yap"""
    print("\n" + "="*70)
    print("VERİ SETİ ANALİZİ")
    print("="*70)
    
    from src.preprocessing.preprocess import resolve_csv_path
    dataset_path = resolve_csv_path(config)
    
    if detailed:
        # Detaylı analiz
        from src.data_analysis.detailed_analysis import DetailedAnalyzer
        
        print("\n🔍 Detaylı analiz modu aktif...")
        analyzer = DetailedAnalyzer(str(dataset_path))
        
        # Kapsamlı rapor
        print("\n" + "="*70)
        print("KAPSAMLI RAPOR OLUŞTURULUYOR")
        print("="*70)
        
        # Rapor oluştur
        plot_dir = Path(config['logging']['plot_dir'])
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        report = analyzer.generate_report(save_path=str(plot_dir.parent / 'detailed_analysis_report.txt'))
        print(report)
        
        # Görselleştirme
        print("\n📊 Kapsamlı görselleştirmeler oluşturuluyor...")
        analyzer.plot_comprehensive_analysis(save_dir=str(plot_dir))
        
        # İstatistikler
        print("\n" + "="*70)
        print("İSTATİSTİKSEL TESTLER")
        print("="*70)
        
        subset_comp = analyzer.compare_subsets()
        print(f"\n📊 Subset Karşılaştırması:")
        print(f"  Chi-Square: {subset_comp['chi_square_test']['chi2']:.4f}")
        print(f"  P-value: {subset_comp['chi_square_test']['p_value']:.4f}")
        print(f"  Anlamlı farklılık: {'Evet ✓' if subset_comp['chi_square_test']['significant'] else 'Hayır ✗'}")
        
        lat_comp = analyzer.compare_laterality()
        print(f"\n🔄 Laterality Karşılaştırması:")
        print(f"  Sol anomali oranı: %{lat_comp['anomaly_rates']['left']*100:.2f}")
        print(f"  Sağ anomali oranı: %{lat_comp['anomaly_rates']['right']*100:.2f}")
        print(f"  Fark: %{lat_comp['anomaly_rates']['difference']*100:.2f}")
        
        # Class weights
        weights = analyzer.get_class_weights('balanced')
        print(f"\n⚖️ Önerilen Class Weights:")
        print(f"  Normal (0): {weights['normal']:.4f}")
        print(f"  Anomaly (1): {weights['anomaly']:.4f}")
        print(f"  Ratio: 1:{weights['ratio']:.2f}")
        
        # İlginç hastalar
        interesting = analyzer.find_interesting_patients()
        print(f"\n🔍 İlginç Hasta Profilleri:")
        print(f"  Her iki böbrek anomalili: {len(interesting['both_anomaly'])} hasta")
        print(f"  Sadece sol anomalili: {len(interesting['left_only'])} hasta")
        print(f"  Sadece sağ anomalili: {len(interesting['right_only'])} hasta")
        
    else:
        # Basit analiz
        explorer = DatasetExplorer(str(dataset_path))
        
        # Analiz ve raporlama
        explorer.print_summary()
        
        # Görselleştirme
        plot_dir = Path(config['logging']['plot_dir'])
        plot_dir.mkdir(parents=True, exist_ok=True)
        explorer.visualize_distribution(save_path=str(plot_dir / 'data_analysis.png'))
        
        print("\n💡 Daha detaylı analiz için:")
        print("   python main.py --mode analyze --detailed")
        print("   veya tools/analyze_dataset.py'yi kullanabilirsiniz")
    
    print("\n✅ Veri analizi tamamlandı!")


def preprocess_data(config):
    """Veri önişleme"""
    print("\n" + "="*70)
    print("VERİ ÖNİŞLEME")
    print("="*70)
    
    # Lazy import
    from src.preprocessing.preprocess import DataPreprocessor
    
    # Tüm config'i geç (DataPreprocessor bunu bekliyor)
    preprocessor = DataPreprocessor(config)
    results = preprocessor.prepare_for_training()
    
    print("\n✅ Veri önişleme tamamlandı!")
    return results

def run_hpo_if_enabled(config, preprocessed_data, device):
    # Run hyperparameter optimization if enabled and return best params
    hpo_cfg = config.get('hpo', {})
    if not hpo_cfg.get('enabled', False):
        return None

    from src.training.hyperparameter_optimizer import HyperparameterOptimizer

    print("\n" + "="*70)
    print("HYPERPARAMETER OPTIMIZATION (AUTO)")
    print("="*70)

    # Build base config for HPO
    hpo_base = deepcopy(config)
    hpo_base.setdefault('training', {})
    hpo_base.setdefault('model', {})

    # Inject class weights into training config (if enabled)
    use_class_weights = (
        config.get('training', {}).get('use_class_weights', True)
        and config.get('class_weights', {}).get('use_in_loss', True)
    )
    if use_class_weights:
        class_weights = (
            preprocessed_data.get('class_weights')
            if config.get('class_weights', {}).get('auto', False)
            else config.get('class_weights', {}).get('manual')
        )
        if class_weights:
            hpo_base['training']['class_weights'] = class_weights
    # Prepare optimizer
    train_loader = preprocessed_data['dataloaders']['train']
    val_loader = preprocessed_data['dataloaders']['dev']
    optimizer = HyperparameterOptimizer(
        base_config=hpo_base,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    method = hpo_cfg.get('method', 'bayesian').lower()
    metric = hpo_cfg.get('metric', 'f1')
    num_epochs = int(hpo_cfg.get('num_epochs', 10))

    best_params = None

    if method in ['bayesian', 'optuna']:
        param_distributions = hpo_cfg.get('param_distributions') or {}
        n_trials = int(hpo_cfg.get('n_trials', 30))
        timeout = hpo_cfg.get('timeout')
        n_jobs = int(hpo_cfg.get('n_jobs', 1))
        results = optimizer.bayesian_search(
            param_distributions=param_distributions,
            n_trials=n_trials,
            metric=metric,
            num_epochs=num_epochs,
            timeout=timeout,
            n_jobs=n_jobs
        )
        best_params = results.get('best_params')

    elif method in ['grid', 'grid_search']:
        param_grid = hpo_cfg.get('param_grid') or {}
        results = optimizer.grid_search(
            param_grid=param_grid,
            metric=metric,
            num_epochs=num_epochs,
            save_all_models=False
        )
        best_params = results.get('best_params')

    else:
        print(f"Warning: unknown HPO method '{method}', skipping HPO.")
        return None

    if best_params:
        print("\nBest params from HPO:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

        # Optionally save best params
        if hpo_cfg.get('save_best_params', True):
            output_dir = Path(hpo_cfg.get('output_dir', 'outputs/hyperparameter_optimization'))
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / 'best_params.json', 'w', encoding='utf-8') as f:
                json.dump(best_params, f, indent=2)

    return best_params



def train_model(config, preprocessed_data=None):
    """Model eğitimi"""
    print("\n" + "="*70)
    print("MODEL EĞİTİMİ")
    print("="*70)
    
    # Lazy imports
    from src.models.model_factory import ModelFactory
    from src.training.modular_trainer import ModularTrainer
    
    # Seed ayarla
    set_seed(config['seed'])
    
    # Device seç
    device = get_device(prefer_cuda=(config['device'] == 'cuda'))
    
    # Önişleme (eğer yapılmamışsa)
    if preprocessed_data is None:
        preprocessed_data = preprocess_data(config)

    # Optional HPO before training
    best_params = run_hpo_if_enabled(config, preprocessed_data, device)
    if best_params:
        # Apply best params to config
        for key, value in best_params.items():
            if key in ['learning_rate', 'weight_decay', 'optimizer', 'scheduler', 'momentum']:
                config['training'][key] = value
            elif key in ['batch_size', 'num_workers']:
                config['training'][key] = value
            elif key in ['dropout', 'base_filters', 'model_type']:
                config['model'][key] = value
            else:
                config['training'][key] = value

        # If batch_size/num_workers changed, rebuild dataloaders
        if any(k in best_params for k in ['batch_size', 'num_workers']):
            print("\nRebuilding dataloaders with best batch_size/num_workers...")
            preprocessed_data = preprocess_data(config)
    
    # Model oluştur
    print("\n📦 Model oluşturuluyor...")
    model = ModelFactory.create_model(config['model'])
    model = model.to(device)
    
    # Parametre sayısı
    count_parameters(model)
    
    # Training config hazirla
    training_config = {
        **config['training'],
        'save_dir': config['training']['save_dir']
    }
    use_class_weights = (
        config.get('training', {}).get('use_class_weights', True)
        and config.get('class_weights', {}).get('use_in_loss', True)
    )
    if use_class_weights:
        training_config['class_weights'] = (
            preprocessed_data.get('class_weights')
            if config.get('class_weights', {}).get('auto', False)
            else config.get('class_weights', {}).get('manual')
        )
    
    # Trainer oluştur
    trainer = ModularTrainer(
        model=model,
        train_loader=preprocessed_data['dataloaders']['train'],
        val_loader=preprocessed_data['dataloaders']['dev'],
        config=training_config,
        device=device
    )
    
    # Eğitimi başlat
    trainer.train(num_epochs=config['training']['epochs'])
    # Auto evaluation after training (optional)
    run_auto_evaluation(config, preprocessed_data, device, trainer=trainer)

    
    # Training history'yi görselleştir
    if config['logging']['save_plots']:
        plot_dir = Path(config['logging']['plot_dir'])
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_training_history(
            trainer.train_losses,
            trainer.val_losses,
            trainer.metrics_history,
            save_path=str(plot_dir / 'training_history.png')
        )
    
    print("\n✅ Model eğitimi tamamlandı!")
    return trainer

def run_auto_evaluation(config, preprocessed_data, device, trainer=None):
    """Run evaluation automatically after training if enabled."""
    eval_cfg = config.get('evaluation', {})
    if not eval_cfg.get('auto', False):
        return

    test_set = eval_cfg.get('test_set', 'test')
    if test_set in ['val', 'validation']:
        test_set = 'dev'

    # Pick checkpoint
    checkpoint_name = eval_cfg.get('checkpoint_name', 'best_model.pth')
    checkpoint_path = eval_cfg.get('checkpoint_path')
    if checkpoint_path is None:
        checkpoint_path = str(Path(config['training'].get('save_dir', 'checkpoints')) / checkpoint_name)

    # Prepare dataloader
    loaders = preprocessed_data.get('dataloaders') if preprocessed_data else None
    if not loaders or test_set not in loaders:
        try:
            from src.preprocessing.preprocess import DataPreprocessor
            preprocessor = DataPreprocessor(config)
            loaders = preprocessor.get_dataloaders(
                batch_size=config['training'].get('batch_size', 8),
                num_workers=config['training'].get('num_workers', 4)
            )
        except Exception as e:
            print(f"Warning: auto evaluation skipped (dataloader error): {e}")
            return

    test_loader = loaders.get(test_set)
    if test_loader is None:
        print(f"Warning: auto evaluation skipped (test set '{test_set}' not found).")
        return

    from src.training.evaluator import ModelEvaluator
    from src.utils.visualization import create_evaluation_report

    # Load model
    model = None
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            from src.models.model_factory import load_model_from_checkpoint
            model = load_model_from_checkpoint(checkpoint_path, config['model'], device)
            print(f"Auto evaluation using checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Warning: checkpoint load failed, using in-memory model. Error: {e}")
            model = None

    if model is None and trainer is not None:
        model = trainer.model

    if model is None:
        print("Warning: auto evaluation skipped (no model available).")
        return

    save_dir = Path(eval_cfg.get('output_dir', f"outputs/evaluation_{test_set}"))
    evaluator = ModelEvaluator(model, device)

    thr_cfg = eval_cfg.get('threshold', {})
    threshold_strategy = thr_cfg.get('strategy', 'fixed')
    threshold = float(thr_cfg.get('default', 0.5))
    beta = float(thr_cfg.get('beta', 2.0))
    min_precision = thr_cfg.get('min_precision')

    selection_set = thr_cfg.get('selection_set')
    if not selection_set:
        if test_set == 'test' and loaders and 'dev' in loaders:
            selection_set = 'dev'
        else:
            selection_set = test_set
    if selection_set in ['val', 'validation']:
        selection_set = 'dev'
    threshold_loader = loaders.get(selection_set) if loaders else None
    if threshold_loader is None:
        selection_set = None

    results = evaluator.evaluate(
        test_loader,
        save_dir=str(save_dir),
        threshold_strategy=threshold_strategy,
        threshold=threshold,
        beta=beta,
        min_precision=min_precision,
        threshold_loader=threshold_loader,
        threshold_selection=selection_set
    )

    # Report
    try:
        report_path = save_dir / 'evaluation_report.pdf'
        create_evaluation_report(str(save_dir), str(report_path))
        print(f"Auto evaluation report created: {report_path}")
    except Exception as e:
        print(f"Warning: report creation failed: {e}")

    return results



def evaluate_model(config, model_path):
    """Model evaluation"""
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)

    from src.models.model_factory import load_model_from_checkpoint
    from src.preprocessing.preprocess import DataPreprocessor
    from src.training.evaluator import ModelEvaluator
    from src.utils.visualization import create_evaluation_report

    # Model path check
    model_path = model_path.strip()
    if not model_path:
        print("Error: model path is empty.")
        return None

    if not Path(model_path).exists():
        print(f"Error: model file not found: {model_path}")
        return None

    # Device
    device = get_device(prefer_cuda=(config.get('device', 'cuda') == 'cuda'))

    # Test set selection
    test_set = input("\nTest set (test/dev) [test]: ").strip().lower() or 'test'
    if test_set in ['val', 'validation']:
        test_set = 'dev'
    if test_set not in ['test', 'dev']:
        print("Warning: invalid choice, using 'test'.")
        test_set = 'test'

    # Load model
    print("\nLoading model...")
    model = load_model_from_checkpoint(model_path, config['model'], device)

    # Prepare dataloader
    print("\nPreparing test dataloader...")
    preprocessor = DataPreprocessor(config)
    loaders = preprocessor.get_dataloaders(
        batch_size=config['training'].get('batch_size', 8),
        num_workers=config['training'].get('num_workers', 4)
    )

    test_loader = loaders.get(test_set)
    if test_loader is None:
        print(f"Error: test set '{test_set}' not found.")
        return None

    # Evaluate
    save_dir = Path(f"outputs/evaluation_{test_set}")
    evaluator = ModelEvaluator(model, device)

    eval_cfg = config.get('evaluation', {})
    thr_cfg = eval_cfg.get('threshold', {})
    threshold_strategy = thr_cfg.get('strategy', 'fixed')
    threshold = float(thr_cfg.get('default', 0.5))
    beta = float(thr_cfg.get('beta', 2.0))
    min_precision = thr_cfg.get('min_precision')

    selection_set = thr_cfg.get('selection_set')
    if not selection_set:
        if test_set == 'test' and loaders and 'dev' in loaders:
            selection_set = 'dev'
        else:
            selection_set = test_set
    if selection_set in ['val', 'validation']:
        selection_set = 'dev'
    threshold_loader = loaders.get(selection_set) if loaders else None
    if threshold_loader is None:
        selection_set = None

    results = evaluator.evaluate(
        test_loader,
        save_dir=str(save_dir),
        threshold_strategy=threshold_strategy,
        threshold=threshold,
        beta=beta,
        min_precision=min_precision,
        threshold_loader=threshold_loader,
        threshold_selection=selection_set
    )

    # Report
    try:
        report_path = save_dir / 'evaluation_report.pdf'
        create_evaluation_report(str(save_dir), str(report_path))
        print(f"\nReport created: {report_path}")
    except Exception as e:
        print(f"Warning: report creation failed: {e}")

    print("\nModel evaluation completed.")
    return results


def run_hyperparameter_optimization():
    """Run hyperparameter optimization menu"""
    print("\n" + "="*70)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*70)

    try:
        from cli.run_hyperparameter_optimization import main as hpo_main
        hpo_main()
    except Exception as e:
        print(f"\nError: hyperparameter optimization menu failed: {e}")
        print("\nManual run:")
        print("   python cli/run_hyperparameter_optimization.py")


def run_detailed_dataset_analysis():
    """Detaylı dataset analizi (tools/analyze_dataset.py)"""
    print("\n" + "="*70)
    print("DETAYLI DATASET ANALİZİ")
    print("="*70)
    
    try:
        from tools.analyze_dataset import analyze_samples
        analyze_samples()
    except Exception as e:
        print(f"\n❌ Analiz çalıştırılamadı: {e}")
        print("\n💡 Manuel çalıştırma:")
        print("   python tools/analyze_dataset.py")


def run_preprocessing_menu():
    """Veri önişleme menüsünü çalıştır"""
    print("\n" + "="*70)
    print("VERİ ÖNİŞLEME MENÜSÜ")
    print("="*70)
    
    try:
        from cli.data_preprocessing_menu import DataPreprocessingMenu
        menu = DataPreprocessingMenu()
        menu.main()
    except Exception as e:
        print(f"\n❌ Menü çalıştırılamadı: {e}")
        print("\n💡 Manuel çalıştırma:")
        print("   python cli/data_preprocessing_menu.py")


def run_class_balance_menu():
    """Sınıf dengeleme menüsünü çalıştır"""
    print("\n" + "="*70)
    print("SINIF DENGELEME MENÜSÜ")
    print("="*70)
    
    try:
        from cli.class_balance_menu import ClassBalanceMenu
        menu = ClassBalanceMenu()
        menu.main()
    except Exception as e:
        print(f"\n❌ Menü çalıştırılamadı: {e}")
        print("\n💡 Manuel çalıştırma:")
        print("   python cli/class_balance_menu.py")


def run_model_comparison():
    """Model karşılaştırma görselleştirmesini çalıştır"""
    print("\n" + "="*70)
    print("MODEL KARŞILAŞTIRMA VE GÖRSELLEŞTİRME")
    print("="*70)
    
    try:
        from tools.visualize_model_comparison import main as visualize_main
        visualize_main()
    except Exception as e:
        print(f"\n❌ Görselleştirme çalıştırılamadı: {e}")
        print("\n💡 Manuel çalıştırma:")
        print("   python tools/visualize_model_comparison.py")


def run_test_setup():
    """Sistem test ve kurulum kontrolü"""
    print("\n" + "="*70)
    print("SİSTEM TEST VE KURULUM KONTROLÜ")
    print("="*70)
    
    try:
        from scripts.test_setup import run_all_tests
        run_all_tests()
    except Exception as e:
        print(f"\n❌ Test çalıştırılamadı: {e}")
        print("\n💡 Manuel çalıştırma:")
        print("   python scripts/test_setup.py")


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(
        description='NeAR Dataset - Böbrek Anomali Tespiti',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Kullanım Örnekleri:
  Veri analizi:           python main.py --mode analyze
  Detaylı veri analizi:   python main.py --mode analyze --detailed
  Model eğitimi:          python main.py --mode train
  Tüm pipeline:           python main.py --mode all
  
Jupyter Notebook ile detaylı inceleme:
  jupyter notebook notebooks/01_data_exploration.ipynb
        """
    )
    
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Config dosyası yolu')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['analyze', 'preprocess', 'train', 'evaluate', 'all'],
                        help='Çalıştırma modu (belirtilmezse menü gösterilir)')
    parser.add_argument('--detailed', action='store_true',
                        help='Detaylı analiz modu (sadece analyze ile kullanılır)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Değerlendirme için model yolu')
    
    args = parser.parse_args()
    
    # Config yükle
    print(f"\n📄 Config yükleniyor: {args.config}")
    config = load_config(args.config)
    
    # Eğer mode belirtilmemişse menü göster
    if args.mode is None:
        while True:
            choice = show_menu()
            
            if choice == '0':
                print("\n👋 Çıkılıyor...\n")
                break
            
            action, detailed = get_menu_action(choice, config)
            
            if action is None:
                print("\n❌ Geçersiz seçim! Lütfen 0-9 arası bir sayı girin.")
                input("\nDevam etmek için Enter'a basın...")
                continue
            
            # Seçime göre işlemi çalıştır
            try:
                if action == 'analyze':
                    analyze_data(config, detailed=detailed)
                elif action == 'display':
                    display_sample_data(config)
                elif action == 'detailed_dataset_analysis':
                    run_detailed_dataset_analysis()
                elif action == 'image_stats':
                    from src.utils.image_processing_utils import compute_image_statistics
                    compute_image_statistics(config)
                elif action == 'transform_test':
                    from src.utils.image_processing_utils import test_image_transforms
                    test_image_transforms(config)
                elif action == 'medical_test':
                    run_medical_transform_test(config)
                elif action == 'preprocess':
                    preprocess_data(config)
                elif action == 'preprocessing_menu':
                    run_preprocessing_menu()
                elif action == 'class_balance_menu':
                    run_class_balance_menu()
                elif action == 'train':
                    preprocessed_data = preprocess_data(config)
                    train_model(config, preprocessed_data)
                elif action == 'evaluate':
                    model_path = input("\n📂 Model dosyası yolu: ").strip()
                    if not model_path:
                        print("❌ Model yolu belirtilmedi!")
                    else:
                        evaluate_model(config, model_path)
                elif action == 'model_comparison':
                    run_model_comparison()
                elif action == 'hyperparameter_optimization':
                    run_hyperparameter_optimization()
                elif action == 'test_setup':
                    run_test_setup()
                elif action == 'all':
                    analyze_data(config, detailed=True)
                    preprocessed_data = preprocess_data(config)
                    train_model(config, preprocessed_data)
                
                print("\n" + "="*70)
                print("✅ İŞLEM TAMAMLANDI")
                print("="*70)
                input("\nDevam etmek için Enter'a basın...")
                
            except KeyboardInterrupt:
                print("\n\n⚠️  İşlem kullanıcı tarafından iptal edildi.")
                input("\nDevam etmek için Enter'a basın...")
            except Exception as e:
                print(f"\n❌ Hata oluştu: {str(e)}")
                input("\nDevam etmek için Enter'a basın...")
        
        return
    
    # Komut satırı argümanı ile çalıştırma (eski davranış)
    if args.mode == 'analyze':
        analyze_data(config, detailed=args.detailed)
    
    elif args.mode == 'preprocess':
        preprocess_data(config)
    
    elif args.mode == 'train':
        preprocessed_data = preprocess_data(config)
        train_model(config, preprocessed_data)
    
    elif args.mode == 'evaluate':
        if args.model_path is None:
            print("❌ --model-path argümanı gerekli!")
            sys.exit(1)
        evaluate_model(config, args.model_path)
    
    elif args.mode == 'all':
        # Tüm pipeline'ı çalıştır
        analyze_data(config, detailed=True)
        preprocessed_data = preprocess_data(config)
        train_model(config, preprocessed_data)
    
    print("\n" + "="*70)
    print("✅ İŞLEM TAMAMLANDI")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
