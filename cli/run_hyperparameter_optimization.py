"""
Hiperparametre Optimizasyon Arayüzü
Grid Search ve Bayesian Optimization için kullanıcı dostu menü
"""

import torch
from pathlib import Path
import sys

# Proje dizinini path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))


def _configure_console() -> None:
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


_configure_console()

from src.training.hyperparameter_optimizer import HyperparameterOptimizer, run_interactive_optimization
from src.utils.helpers import load_config
from src.preprocessing.dataloader_factory import create_dataloaders


def main():
    """Ana menü"""
    print("\n" + "="*80)
    print(" "*20 + "HİPERPARAMETRE OPTİMİZASYONU")
    print("="*80)
    
    # Config yükle
    config_path = 'configs/config.yaml'
    try:
        config = load_config(config_path)
        print("\n✓ Config başarıyla yüklendi")
    except Exception as e:
        print(f"\n❌ Config yüklenemedi: {e}")
        print("⚠️  Varsayılan ayarlar kullanılacak")
        config = {
            'dataset': {
                'path': 'NeAR_dataset/ALAN',
                'csv_file': 'info.csv',
                'zip_file': 'ALAN.zip'
            },
            'preprocessing': {
                'normalize': False,
                'mean': 0.0,
                'std': 1.0,
                'augmentation': {'enabled': False}
            },
            'training': {
                'batch_size': 32,
                'num_workers': 0,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'optimizer': 'adam'
            },
            'model': {
                'model_type': 'resnet3d',
                'num_classes': 2,
                'in_channels': 1
            },
            'seed': 42
        }
    
    # Device kontrolü
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ Device: {device}")
    
    if device == 'cpu':
        print("⚠️  GPU bulunamadı, CPU kullanılacak (yavaş olabilir)")
    
    # Ana menü
    while True:
        print("\n" + "─"*80)
        print("ANA MENÜ")
        print("─"*80)
        print("\n1. 🔍 Grid Search Optimizasyonu")
        print("2. 🎯 Bayesian Optimization (Optuna)")
        print("3. 📊 Hızlı Test (Az parametre, hızlı sonuç)")
        print("4. ⚙️  Özel Ayarlar")
        print("5. 📖 Yardım ve Örnekler")
        print("0. ❌ Çıkış")
        
        choice = input("\n👉 Seçiminiz (0-5): ").strip()
        
        if choice == '0':
            print("\n👋 Çıkılıyor...")
            break
        
        elif choice == '1':
            grid_search_menu(config, device)
        
        elif choice == '2':
            bayesian_search_menu(config, device)
        
        elif choice == '3':
            quick_test_menu(config, device)
        
        elif choice == '4':
            custom_settings_menu(config, device)
        
        elif choice == '5':
            show_help()
        
        else:
            print("\n❌ Geçersiz seçim! Lütfen 0-5 arası bir sayı girin.")


def grid_search_menu(config: dict, device: str):
    """Grid Search menüsü"""
    print("\n" + "="*80)
    print("GRID SEARCH HİPERPARAMETRE OPTİMİZASYONU")
    print("="*80)
    
    print("\n📝 Grid Search tüm parametre kombinasyonlarını sistematik olarak dener.")
    print("   Az sayıda parametre seçmek daha hızlı sonuç verir.\n")
    
    # Parametreleri topla
    param_grid = {}
    
    # Learning Rate
    print("─"*80)
    print("📌 Learning Rate (Öğrenme Hızı)")
    print("   Önerilen: 0.001, 0.0001, 0.00001")
    lr_input = input("   Değerler (virgülle ayırın): ").strip()
    if lr_input:
        try:
            param_grid['learning_rate'] = [float(x.strip()) for x in lr_input.split(',')]
        except:
            print("   ⚠️  Geçersiz format, varsayılan kullanılacak: [0.001, 0.0001]")
            param_grid['learning_rate'] = [0.001, 0.0001]
    else:
        param_grid['learning_rate'] = [0.0001]
    
    # Batch Size
    print("\n─"*80)
    print("📌 Batch Size")
    print("   Önerilen: 16, 32, 64 (GPU belleğinize göre)")
    bs_input = input("   Değerler (virgülle ayırın): ").strip()
    if bs_input:
        try:
            param_grid['batch_size'] = [int(x.strip()) for x in bs_input.split(',')]
        except:
            print("   ⚠️  Geçersiz format, varsayılan kullanılacak: [32]")
            param_grid['batch_size'] = [32]
    else:
        param_grid['batch_size'] = [32]
    
    # Optimizer
    print("\n─"*80)
    print("📌 Optimizer")
    print("   Seçenekler: adam, adamw, sgd")
    opt_input = input("   Değerler (virgülle ayırın): ").strip()
    if opt_input:
        param_grid['optimizer'] = [x.strip().lower() for x in opt_input.split(',')]
    else:
        param_grid['optimizer'] = ['adam']
    
    # Dropout
    print("\n─"*80)
    print("📌 Dropout (Overfitting önleme)")
    print("   Önerilen: 0.3, 0.5")
    drop_input = input("   Değerler (virgülle ayırın, boş=atla): ").strip()
    if drop_input:
        try:
            param_grid['dropout'] = [float(x.strip()) for x in drop_input.split(',')]
        except:
            print("   ⚠️  Geçersiz format, atlandı")
    
    # Epoch sayısı
    print("\n─"*80)
    print("📌 Epoch Sayısı (Her kombinasyon için)")
    print("   Önerilen: 10-20 (hızlı test), 50+ (gerçek eğitim)")
    num_epochs = input("   Epoch sayısı: ").strip()
    try:
        num_epochs = int(num_epochs)
    except:
        num_epochs = 10
        print(f"   ⚠️  Geçersiz format, varsayılan kullanılacak: {num_epochs}")
    
    # Metrik seçimi
    print("\n─"*80)
    print("📌 Optimize Edilecek Metrik")
    print("   1. Accuracy (Doğruluk)")
    print("   2. F1 Score (Dengeli metrik)")
    print("   3. AUC (ROC eğrisi altı alan)")
    metric_choice = input("   Seçiminiz (1-3): ").strip()
    metric_map = {'1': 'accuracy', '2': 'f1', '3': 'auc'}
    metric = metric_map.get(metric_choice, 'accuracy')
    
    # Özet göster
    print("\n" + "="*80)
    print("ÖZET")
    print("="*80)
    total_combinations = 1
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
        total_combinations *= len(values)
    
    print(f"\n  📊 Toplam kombinasyon: {total_combinations}")
    print(f"  📈 Her kombinasyon: {num_epochs} epoch")
    print(f"  🎯 Metrik: {metric}")
    print(f"  ⏱️  Tahmini süre: ~{total_combinations * num_epochs * 2} dakika")
    
    # Onay
    print("\n─"*80)
    confirm = input("🚀 Grid Search başlatılsın mı? (e/h): ").strip().lower()
    
    if confirm != 'e':
        print("❌ İptal edildi.")
        return
    
    # Dataloaders hazırla
    print("\n📦 Veri yükleniyor...")
    try:
        train_loader, val_loader, _ = create_dataloaders(config)
        print("✓ Veri başarıyla yüklendi")
    except Exception as e:
        print(f"❌ Veri yükleme hatası: {e}")
        print("⚠️  Lütfen dataset yolunu ve config ayarlarını kontrol edin.")
        return
    
    # Optimizer oluştur ve çalıştır
    print("\n🔍 Grid Search başlatılıyor...\n")
    
    optimizer = HyperparameterOptimizer(
        base_config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    results = optimizer.grid_search(
        param_grid=param_grid,
        metric=metric,
        num_epochs=num_epochs,
        save_all_models=False
    )
    
    print("\n✅ Grid Search tamamlandı!")
    print(f"🏆 En iyi skor: {results['best_score']:.4f}")
    print(f"📋 En iyi parametreler:")
    for key, value in results['best_params'].items():
        print(f"   {key}: {value}")


def bayesian_search_menu(config: dict, device: str):
    """Bayesian Search menüsü"""
    print("\n" + "="*80)
    print("BAYESIAN OPTIMIZATION (OPTUNA)")
    print("="*80)
    
    print("\n📝 Bayesian Optimization parametreleri akıllıca arar (Grid Search'ten hızlı).")
    print("   Daha fazla deneme sayısı = daha iyi sonuç\n")
    
    # Deneme sayısı
    print("─"*80)
    print("📌 Deneme Sayısı")
    print("   Önerilen: 20-50 (hızlı), 100+ (kapsamlı)")
    n_trials = input("   Deneme sayısı: ").strip()
    try:
        n_trials = int(n_trials)
    except:
        n_trials = 30
        print(f"   ⚠️  Geçersiz format, varsayılan kullanılacak: {n_trials}")
    
    # Timeout
    print("\n─"*80)
    print("📌 Maksimum Süre (saniye)")
    print("   Boş bırakırsanız tüm denemeler yapılır")
    timeout = input("   Timeout (saniye): ").strip()
    timeout = int(timeout) if timeout else None
    
    # Epoch sayısı
    print("\n─"*80)
    print("📌 Epoch Sayısı (Her deneme için)")
    print("   Önerilen: 10-20 (hızlı test), 50+ (gerçek eğitim)")
    num_epochs = input("   Epoch sayısı: ").strip()
    try:
        num_epochs = int(num_epochs)
    except:
        num_epochs = 10
        print(f"   ⚠️  Geçersiz format, varsayılan kullanılacak: {num_epochs}")
    
    # Parametre dağılımları
    param_distributions = {}
    
    # Learning Rate
    print("\n─"*80)
    print("📌 Learning Rate Aralığı (logaritmik ölçek)")
    print("   Önerilen: 0.00001 - 0.01")
    lr_min = input("   Min değer: ").strip()
    lr_max = input("   Max değer: ").strip()
    try:
        lr_min = float(lr_min) if lr_min else 1e-5
        lr_max = float(lr_max) if lr_max else 1e-2
    except:
        lr_min, lr_max = 1e-5, 1e-2
        print(f"   ⚠️  Geçersiz format, varsayılan: [{lr_min}, {lr_max}]")
    
    param_distributions['learning_rate'] = {
        'type': 'float',
        'low': lr_min,
        'high': lr_max,
        'log': True
    }
    
    # Batch Size
    print("\n─"*80)
    print("📌 Batch Size Seçenekleri")
    print("   Önerilen: 16, 32, 64")
    bs_input = input("   Değerler (virgülle ayırın): ").strip()
    if bs_input:
        try:
            batch_sizes = [int(x.strip()) for x in bs_input.split(',')]
        except:
            batch_sizes = [16, 32, 64]
            print(f"   ⚠️  Geçersiz format, varsayılan: {batch_sizes}")
    else:
        batch_sizes = [16, 32, 64]
    
    param_distributions['batch_size'] = {
        'type': 'categorical',
        'choices': batch_sizes
    }
    
    # Optimizer
    print("\n─"*80)
    print("📌 Optimizer Seçenekleri")
    print("   Önerilen: adam, adamw")
    opt_input = input("   Değerler (virgülle ayırın): ").strip()
    if opt_input:
        optimizers = [x.strip().lower() for x in opt_input.split(',')]
    else:
        optimizers = ['adam', 'adamw']
    
    param_distributions['optimizer'] = {
        'type': 'categorical',
        'choices': optimizers
    }
    
    # Dropout
    print("\n─"*80)
    print("📌 Dropout Aralığı")
    print("   Önerilen: 0.1 - 0.7")
    dr_min = input("   Min değer: ").strip()
    dr_max = input("   Max değer: ").strip()
    try:
        dr_min = float(dr_min) if dr_min else 0.1
        dr_max = float(dr_max) if dr_max else 0.7
    except:
        dr_min, dr_max = 0.1, 0.7
        print(f"   ⚠️  Geçersiz format, varsayılan: [{dr_min}, {dr_max}]")
    
    param_distributions['dropout'] = {
        'type': 'float',
        'low': dr_min,
        'high': dr_max
    }
    
    # Metrik seçimi
    print("\n─"*80)
    print("📌 Optimize Edilecek Metrik")
    print("   1. Accuracy (Doğruluk)")
    print("   2. F1 Score (Dengeli metrik)")
    print("   3. AUC (ROC eğrisi altı alan)")
    metric_choice = input("   Seçiminiz (1-3): ").strip()
    metric_map = {'1': 'accuracy', '2': 'f1', '3': 'auc'}
    metric = metric_map.get(metric_choice, 'accuracy')
    
    # Özet göster
    print("\n" + "="*80)
    print("ÖZET")
    print("="*80)
    print(f"  📊 Deneme sayısı: {n_trials}")
    print(f"  ⏱️  Timeout: {timeout if timeout else 'Yok'} saniye")
    print(f"  📈 Her deneme: {num_epochs} epoch")
    print(f"  🎯 Metrik: {metric}")
    print(f"\n  Aranacak Parametreler:")
    for key, dist in param_distributions.items():
        if dist['type'] == 'categorical':
            print(f"    {key}: {dist['choices']}")
        elif dist['type'] == 'float':
            log_str = " (log)" if dist.get('log') else ""
            print(f"    {key}: [{dist['low']}, {dist['high']}]{log_str}")
    
    print(f"\n  ⏱️  Tahmini süre: ~{n_trials * num_epochs * 2} dakika")
    
    # Onay
    print("\n─"*80)
    confirm = input("🚀 Bayesian Optimization başlatılsın mı? (e/h): ").strip().lower()
    
    if confirm != 'e':
        print("❌ İptal edildi.")
        return
    
    # Dataloaders hazırla
    print("\n📦 Veri yükleniyor...")
    try:
        train_loader, val_loader, _ = create_dataloaders(config)
        print("✓ Veri başarıyla yüklendi")
    except Exception as e:
        print(f"❌ Veri yükleme hatası: {e}")
        print("⚠️  Lütfen dataset yolunu ve config ayarlarını kontrol edin.")
        return
    
    # Optimizer oluştur ve çalıştır
    print("\n🎯 Bayesian Optimization başlatılıyor...\n")
    
    optimizer = HyperparameterOptimizer(
        base_config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    results = optimizer.bayesian_search(
        param_distributions=param_distributions,
        n_trials=n_trials,
        metric=metric,
        num_epochs=num_epochs,
        timeout=timeout,
        n_jobs=1
    )
    
    print("\n✅ Bayesian Optimization tamamlandı!")
    print(f"🏆 En iyi skor: {results['best_score']:.4f}")
    print(f"📋 En iyi parametreler:")
    for key, value in results['best_params'].items():
        print(f"   {key}: {value}")


def quick_test_menu(config: dict, device: str):
    """Hızlı test menüsü"""
    print("\n" + "="*80)
    print("HIZLI TEST")
    print("="*80)
    
    print("\n📝 Hızlı test modunda az parametre ve az epoch kullanılır.")
    print("   Sistemi test etmek ve hızlı sonuç almak için uygundur.\n")
    
    print("─"*80)
    print("🎯 Hangi yöntemi denemek istersiniz?")
    print("   1. Grid Search (3-5 kombinasyon)")
    print("   2. Bayesian Search (10 deneme)")
    
    method = input("\n👉 Seçiminiz (1-2): ").strip()
    
    if method == '1':
        # Hızlı Grid Search
        param_grid = {
            'learning_rate': [1e-3, 1e-4],
            'optimizer': ['adam'],
            'batch_size': [32]
        }
        num_epochs = 5
        metric = 'accuracy'
        
        print("\n📋 Hızlı Grid Search Ayarları:")
        print(f"  Learning Rate: {param_grid['learning_rate']}")
        print(f"  Optimizer: {param_grid['optimizer']}")
        print(f"  Batch Size: {param_grid['batch_size']}")
        print(f"  Epoch: {num_epochs}")
        print(f"  Toplam: 2 kombinasyon × {num_epochs} epoch = ~10 dakika")
        
        confirm = input("\n🚀 Başlat? (e/h): ").strip().lower()
        if confirm != 'e':
            return
        
        # Veri yükle ve çalıştır
        try:
            train_loader, val_loader, _ = create_dataloaders(config)
            
            optimizer = HyperparameterOptimizer(
                base_config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device
            )
            
            results = optimizer.grid_search(
                param_grid=param_grid,
                metric=metric,
                num_epochs=num_epochs
            )
            
            print("\n✅ Hızlı test tamamlandı!")
            
        except Exception as e:
            print(f"\n❌ Hata: {e}")
    
    elif method == '2':
        # Hızlı Bayesian Search
        param_distributions = {
            'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
            'batch_size': {'type': 'categorical', 'choices': [32]},
            'optimizer': {'type': 'categorical', 'choices': ['adam', 'adamw']},
            'dropout': {'type': 'float', 'low': 0.3, 'high': 0.6}
        }
        n_trials = 10
        num_epochs = 5
        metric = 'accuracy'
        
        print("\n📋 Hızlı Bayesian Search Ayarları:")
        print(f"  Deneme sayısı: {n_trials}")
        print(f"  Epoch: {num_epochs}")
        print(f"  Tahmini süre: ~{n_trials * num_epochs * 2} dakika")
        
        confirm = input("\n🚀 Başlat? (e/h): ").strip().lower()
        if confirm != 'e':
            return
        
        # Veri yükle ve çalıştır
        try:
            train_loader, val_loader, _ = create_dataloaders(config)
            
            optimizer = HyperparameterOptimizer(
                base_config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device
            )
            
            results = optimizer.bayesian_search(
                param_distributions=param_distributions,
                n_trials=n_trials,
                metric=metric,
                num_epochs=num_epochs
            )
            
            print("\n✅ Hızlı test tamamlandı!")
            
        except Exception as e:
            print(f"\n❌ Hata: {e}")


def custom_settings_menu(config: dict, device: str):
    """Özel ayarlar menüsü"""
    print("\n" + "="*80)
    print("ÖZEL AYARLAR")
    print("="*80)
    
    print("\n⚙️  Config dosyasını düzenleyerek ayarları değiştirebilirsiniz:")
    print(f"   📄 configs/config.yaml")
    print("\n   Model, eğitim ve dataset ayarları bu dosyada bulunur.")
    print("   Düzenledikten sonra optimizasyonu tekrar çalıştırın.")
    
    input("\n↵ Devam etmek için Enter'a basın...")


def show_help():
    """Yardım ve örnekler"""
    print("\n" + "="*80)
    print("YARDIM VE ÖRNEKLER")
    print("="*80)
    
    print("\n📚 HİPERPARAMETRE OPTİMİZASYONU NEDİR?")
    print("─"*80)
    print("""
Hiperparametre optimizasyonu, model performansını artırmak için en iyi
parametre kombinasyonunu bulmaktır. Optimize edebileceğiniz parametreler:

  • Learning Rate: Modelin ne hızda öğreneceği
  • Batch Size: Her adımda kaç örnek kullanılacağı
  • Optimizer: Gradient descent algoritması (Adam, AdamW, SGD)
  • Dropout: Overfitting'i önlemek için nöron kapatma oranı
  • Model Architecture: Katman sayısı, filtre sayısı vb.
""")
    
    print("\n🔍 GRID SEARCH vs BAYESIAN SEARCH")
    print("─"*80)
    print("""
Grid Search:
  ✓ Tüm kombinasyonları sistematik olarak dener
  ✓ Basit ve anlaşılır
  ✗ Çok parametre ile yavaş olabilir
  → Az parametre için ideal

Bayesian Search (Optuna):
  ✓ Akıllıca arama yapar (önceki denemelerden öğrenir)
  ✓ Daha az denemede iyi sonuç bulur
  ✓ Çok parametre için uygun
  → Genel olarak önerilen yöntem
""")
    
    print("\n💡 ÖNERLER")
    print("─"*80)
    print("""
1. İlk deneme için Hızlı Test kullanın (5-10 dakika)
2. Sonuçlar iyiyse, daha fazla epoch ve deneme ile tekrarlayın
3. Learning rate genellikle en önemli parametredir
4. GPU varsa daha büyük batch size kullanın (16, 32, 64)
5. Sonuçları outputs/hyperparameter_optimization/ dizininde bulabilirsiniz
""")
    
    print("\n📊 ÖRNEK ÇIKTI")
    print("─"*80)
    print("""
Optimizasyon sonunda şunları elde edersiniz:
  • En iyi parametreler (JSON formatında)
  • Tüm denemelerin sonuçları
  • Görselleştirmeler (Optuna için HTML dosyaları)
  • Parameter importance analizi
  • Optimization history grafiği
""")
    
    input("\n↵ Ana menüye dönmek için Enter'a basın...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Program kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n\n❌ Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()
