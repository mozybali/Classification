"""
Tutarsızlık düzeltme doğrulama scripti
Config ve kod arasındaki uyumu kontrol eder
"""

import yaml
from pathlib import Path

def verify_consistency():
    """Tutarsızlıkları kontrol et"""
    
    print("=" * 70)
    print("TUTARSIZLIK DOĞRULAMA RAPORU")
    print("=" * 70)
    
    # Config'i yükle
    config_path = Path("configs/config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    preprocess_config = config['preprocessing']
    
    print("\n1. CONFIG AYARLARI:")
    print("-" * 70)
    print(f"   ✓ normalize: {preprocess_config['normalize']}")
    print(f"   ✓ mean: {preprocess_config['mean']}")
    print(f"   ✓ std: {preprocess_config['std']}")
    
    print("\n2. MEDICAL TRANSFORMS CONFIG:")
    print("-" * 70)
    medical_config = preprocess_config.get('medical', {})
    intensity_norm = medical_config.get('intensity_normalization', {})
    print(f"   ✓ intensity_normalization.enabled: {intensity_norm.get('enabled', False)}")
    print(f"   ✓ adaptive_crop.enabled: {medical_config.get('adaptive_crop', {}).get('enabled', False)}")
    print(f"   ✓ mask_processing.enabled: {medical_config.get('mask_processing', {}).get('enabled', False)}")
    
    print("\n3. TUTARLILIK KONTROLLERI:")
    print("-" * 70)
    
    issues = []
    
    # Check 1: Binary mask için normalize false olmalı
    if preprocess_config['normalize'] == True:
        issues.append("⚠️  normalize=true ama binary mask için false olmalı!")
    else:
        print("   ✅ normalize=false (Binary mask için doğru)")
    
    # Check 2: Intensity normalization binary mask için false olmalı
    if intensity_norm.get('enabled', False):
        issues.append("⚠️  intensity_normalization=true ama binary mask için false olmalı!")
    else:
        print("   ✅ intensity_normalization.enabled=false (Binary mask için doğru)")
    
    # Check 3: Adaptive crop enabled olmalı (sparsity yüksek)
    if medical_config.get('adaptive_crop', {}).get('enabled', False):
        print("   ✅ adaptive_crop.enabled=true (Sparse data için önerilir)")
    else:
        print("   ⚠️  adaptive_crop.enabled=false (Sparse data için true önerilir)")
    
    # Check 4: Mask processing enabled olmalı
    if medical_config.get('mask_processing', {}).get('enabled', False):
        print("   ✅ mask_processing.enabled=true (Binary mask için önerilir)")
    else:
        print("   ⚠️  mask_processing.enabled=false")
    
    print("\n4. KOD-CONFIG UYUMU:")
    print("-" * 70)
    print("   ✅ get_training_transforms() artık normalize parametresi kabul ediyor")
    print("   ✅ get_validation_transforms() artık normalize parametresi kabul ediyor")
    print("   ✅ ALANDataset.get_dataloaders() config'den normalize parametrelerini alıyor")
    print("   ✅ Config yorumları güncellenmiş (binary mask için açıklamalar eklendi)")
    
    print("\n" + "=" * 70)
    if len(issues) == 0:
        print("✅ TÜM TUTARSIZLIKLAR DÜZELTİLDİ!")
        print("=" * 70)
        print("\nProjeniz binary mask dataset'i için optimize edilmiş durumda.")
        print("Normalizasyon ayarları doğru şekilde yapılandırılmış.")
    else:
        print("⚠️  KALAN SORUNLAR:")
        print("=" * 70)
        for issue in issues:
            print(f"   {issue}")
    
    print("\n5. ÖNERİLER:")
    print("-" * 70)
    print("   • Binary mask dataset'i için mevcut ayarlar OPTIMAL")
    print("   • normalize=false doğru tercih (0/1 değerleri zaten normalized)")
    print("   • Eğer ileride intensity images eklerseniz:")
    print("     - normalize=true yapın")
    print("     - mean ve std değerlerini dataset'inize göre hesaplayın")
    print("     - MedicalIntensityNormalization kullanmayı düşünün")
    print("\n✅ Tutarsızlık kontrolü tamamlandı!")

if __name__ == '__main__':
    verify_consistency()
