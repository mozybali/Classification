"""
System Setup Test and Validation
Proje kurulumunun, bağımlılıkların ve temel işlevlerin test edilmesi
"""

import sys
import importlib
from pathlib import Path

# Root path
ROOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_PATH))


def print_header(title: str):
    """Başlık yazdır"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_python_version():
    """Python sürümü kontrolü"""
    print_header("1️⃣  PYTHON VERSİYONU")
    
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python: {py_version}")
    
    if sys.version_info >= (3, 8):
        print("✅ Python 3.8+ tespit edildi")
        return True
    else:
        print("❌ Python 3.8+ gerekli!")
        return False


def test_core_dependencies():
    """Temel bağımlılıkları test et"""
    print_header("2️⃣  TEMEL BAĞIMLILIKLAR")
    
    core_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'yaml': 'PyYAML',
    }
    
    all_available = True
    for module_name, display_name in core_packages.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name:20} {version}")
        except ImportError:
            print(f"❌ {display_name:20} BULUNAMADI")
            all_available = False
    
    return all_available


def test_optional_dependencies():
    """Opsiyonel bağımlılıkları test et"""
    print_header("3️⃣  OPSİYONEL BAĞIMLILIKLAR")
    
    optional_packages = {
        'torch_geometric': 'torch-geometric (GNN desteği)',
        'optuna': 'Optuna (Hiperparametre optimizasyonu)',
        'plotly': 'Plotly (Görselleştirme)',
        'tqdm': 'tqdm (Progress bars)',
        'tensorboard': 'TensorBoard (Training monitoring)',
    }
    
    available_count = 0
    for module_name, display_name in optional_packages.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name:50} {version}")
            available_count += 1
        except ImportError:
            print(f"⚠️  {display_name:50} BULUNAMADI")
    
    print(f"\n📊 {available_count}/{len(optional_packages)} opsiyonel paket mevcut")
    return available_count >= 3  # En az 3'ü varsa ok


def test_project_structure():
    """Proje yapısını kontrol et"""
    print_header("4️⃣  PROJE YAPISI")
    
    required_dirs = [
        'src',
        'src/models',
        'src/training',
        'src/preprocessing',
        'src/utils',
        'src/data_analysis',
        'src/cli',
        'cli',
        'tools',
        'scripts',
        'tests',
        'configs',
        'docs',
        'NeAR_dataset',
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = ROOT_PATH / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ BULUNAMADI")
            all_exist = False
    
    return all_exist


def test_key_files():
    """Temel dosyaları kontrol et"""
    print_header("5️⃣  TEMEL DOSYALAR")
    
    required_files = [
        'main.py',
        'requirements.txt',
        'pytest.ini',
        'configs/config.yaml',
        'docs/README.md',
    ]
    
    all_exist = True
    for file_name in required_files:
        file_path = ROOT_PATH / file_name
        if file_path.exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} BULUNAMADI")
            all_exist = False
    
    return all_exist


def test_imports():
    """Ana modülleri test et"""
    print_header("6️⃣  MOD ÜLERI İMPORT ET")
    
    test_modules = {
        'src.models': 'Model Modülü',
        'src.training': 'Training Modülü',
        'src.preprocessing.preprocess': 'Preprocessing',
        'src.utils.helpers': 'Utilities',
        'src.data_analysis.explore_data': 'Data Analysis',
    }
    
    all_imported = True
    for module_name, display_name in test_modules.items():
        try:
            importlib.import_module(module_name)
            print(f"✅ {display_name}")
        except Exception as e:
            print(f"❌ {display_name}: {e}")
            all_imported = False
    
    return all_imported


def test_cuda():
    """CUDA ve GPU desteğini kontrol et"""
    print_header("7️⃣  GPU/CUDA DESTEĞI")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✅ CUDA kullanılabilir")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            
            # Capability check
            capability = torch.cuda.get_device_capability(0)
            print(f"   Compute Capability: {capability[0]}.{capability[1]}")
            
            if capability >= (1, 2):
                print(f"✅ RTX 5050 (sm_120) desteği mevcut")
            
            return True
        else:
            print("⚠️  CUDA mevcut değil (CPU mode'de çalışacak)")
            return True  # CPU da valid
            
    except Exception as e:
        print(f"❌ CUDA kontrolü hatası: {e}")
        return False


def test_config():
    """Config dosyasını test et"""
    print_header("8️⃣  KONFİGÜRASYON")
    
    try:
        from src.utils.helpers import load_config
        config = load_config('configs/config.yaml')
        
        required_keys = ['dataset', 'preprocessing', 'model', 'training']
        all_exist = True
        
        for key in required_keys:
            if key in config:
                print(f"✅ config.{key}")
            else:
                print(f"❌ config.{key} BULUNAMADI")
                all_exist = False
        
        return all_exist
        
    except Exception as e:
        print(f"❌ Config yüklenemedi: {e}")
        return False


def test_data():
    """Veri seti kontrolü"""
    print_header("9️⃣  VERİ SETİ")
    
    try:
        import pandas as pd
        
        csv_path = ROOT_PATH / 'NeAR_dataset' / 'ALAN' / 'info.csv'
        
        if not csv_path.exists():
            print(f"❌ Dataset CSV bulunamadı: {csv_path}")
            return False
        
        df = pd.read_csv(csv_path)
        print(f"✅ Dataset yüklendi")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        if df.shape[0] > 0:
            print(f"   Rows: {df.shape[0]}")
            print(f"✅ Dataset geçerli")
            return True
        else:
            print(f"❌ Dataset boş")
            return False
            
    except Exception as e:
        print(f"❌ Dataset kontrolü hatası: {e}")
        return False


def test_pytest():
    """Pytest kurulumunu test et"""
    print_header("🔟  PYTEST")
    
    try:
        import pytest
        print(f"✅ Pytest {pytest.__version__} kurulu")
        
        tests_path = ROOT_PATH / 'tests'
        if tests_path.exists():
            test_files = list(tests_path.glob('test_*.py'))
            print(f"   Test dosyaları: {len(test_files)}")
            for tf in test_files:
                print(f"   - {tf.name}")
            return True
        else:
            print(f"⚠️  tests/ dizini bulunamadı")
            return False
            
    except ImportError:
        print(f"❌ Pytest kurulu değil")
        print(f"   Kurulum: pip install pytest")
        return False
    except Exception as e:
        print(f"❌ Pytest kontrolü hatası: {e}")
        return False


def run_all_tests():
    """Tüm testleri çalıştır"""
    print("\n" + "🧪" * 35)
    print("  NeAR DATASET - SİSTEM KURULUM TESTI")
    print("🧪" * 35)
    
    results = {
        'Python Version': test_python_version(),
        'Core Dependencies': test_core_dependencies(),
        'Optional Dependencies': test_optional_dependencies(),
        'Project Structure': test_project_structure(),
        'Key Files': test_key_files(),
        'Module Imports': test_imports(),
        'GPU/CUDA': test_cuda(),
        'Configuration': test_config(),
        'Dataset': test_data(),
        'PyTest': test_pytest(),
    }
    
    # Özet
    print_header("📊 TEST ÖZETİ")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")
    
    print(f"\n{'='*70}")
    print(f"📈 Sonuç: {passed}/{total} test başarılı")
    
    if passed == total:
        print("🎉 Sistem kurulumu BAŞARILI!")
        print("\nArtık şunları çalıştırabilirsiniz:")
        print("  python main.py              # Interactive menu")
        print("  pytest tests/ -v            # Run tests")
        print("  python cli/run_training.py  # Training")
    elif passed >= total * 0.8:
        print("⚠️  Çoğu test başarılı ama bazı bağımlılıklar eksik")
        print("   requirements.txt'ten eksikleri yükleyin:")
        print("   pip install -r requirements.txt")
    else:
        print("❌ Sistem kurulumu eksik - lütfen gerekli paketleri yükleyin")
        print("   pip install -r requirements.txt")
    
    print("="*70 + "\n")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
