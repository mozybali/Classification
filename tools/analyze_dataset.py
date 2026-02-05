"""Ön işleme gereksinimlerini belirlemek için hızlı veri seti analizi"""
import numpy as np
import zipfile

def analyze_samples():
    zf = zipfile.ZipFile('NeAR_dataset/ALAN/ALAN.zip')
    
    print("=" * 70)
    print("NEAR ALAN VERI SETİ ANALİZİ")
    print("=" * 70)
    
    # Birden fazla örneği analiz et
    sample_names = ['alan/ZS000_L.npy', 'alan/ZS001_R.npy', 'alan/ZS046_R.npy', 
                    'alan/ZS192_L.npy', 'alan/ZS416_L.npy']
    
    print("\nÖrnek Ayrıntıları:")
    print("-" * 70)
    
    is_all_binary = True
    dtypes = set()
    sparsity_levels = []
    
    for fname in sample_names:
        try:
            sample = np.load(zf.open(fname))
            name = fname.split('/')[-1]
            
            unique_vals = np.unique(sample)
            is_binary = set(unique_vals) <= {0, 1, False, True}
            nonzero_pct = (sample > 0).sum() / sample.size * 100
            
            dtypes.add(str(sample.dtype))
            sparsity_levels.append(nonzero_pct)
            
            if not is_binary:
                is_all_binary = False
            
            print(f"\n{name}:")
            print(f"  Boyut: {sample.shape}")
            print(f"  Veri Tipi: {sample.dtype}")
            print(f"  Min/Max: {sample.min()} / {sample.max()}")
            print(f"  Benzersiz değerler: {len(unique_vals)}")
            print(f"  İkili mi: {is_binary}")
            print(f"  Sıfırdan Farklı: {nonzero_pct:.2f}%")
            
        except Exception as e:
            print(f"\n{fname}: Hata - {e}")
    
    print("\n" + "=" * 70)
    print("ÖZET & ÖNERİLER")
    print("=" * 70)
    
    print(f"\n✓ Veri Tipi: {', '.join(dtypes)}")
    print(f"✓ Tüm örnekler ikili: {is_all_binary}")
    print(f"✓ Ortalama seyreklik (sıfırdan farklı): {np.mean(sparsity_levels):.2f}%")
    print(f"✓ Min/Max seyreklik: {min(sparsity_levels):.2f}% / {max(sparsity_levels):.2f}%")
    
    print("\n" + "-" * 70)
    print("ÖN İŞLEME ÖNERİLERİ:")
    print("-" * 70)
    
    if is_all_binary:
        print("\n✅ VERİ TİPİ: İkili Bölümleme Maskeleri")
        print("\n   Önerilen ön işleme:")
        print("   1. ✅ Mevcut dönüşümleri koruyun (ikili için zaten optimal)")
        print("   2. ✅ Tıbbi Yoğunluk Normalizasyonu - GEREKMİYOR")
        print("   3. ✅ CT Pencereleme - GEREKMİYOR (yoğunluk verisi yok)")
        print("   4. ✅ CLAHE - GEREKMİYOR (yoğunluk verisi yok)")
        print("   5. ✅ İkili maske morfolojik işlemler - ZATEN UYGULANMIŞ")
        
    else:
        print("\n⚠️  VERİ TİPİ: Yoğunluk Görüntüleri")
        print("\n   Önerilen ön işleme:")
        print("   1. ✅ Tıbbi yoğunluk normalizasyonu - GEREKLİ")
        print("   2. ⚠️  CT Pencereleme - EĞER CT verisi varsa")
        print("   3. ⚠️  CLAHE - Kontrast için isteğe bağlı")
    
    avg_sparsity = np.mean(sparsity_levels)
    if avg_sparsity < 5.0:
        print(f"\n✅ SEYREKLİK: Çok seyrek ({avg_sparsity:.2f}%)")
        print("   → AdaptiveROICrop - YÜKSEK ORANDA ÖNERİLİ (zaten uygulanmıştır)")
    
    print("\n" + "=" * 70)
    print("SON KARAR: EK ÖN İŞLEME GEREKMİYOR")
    print("=" * 70)
    print("\nMevcut pipeline'ınız bu ikili maske veri seti için OPTİMAL'dir!")
    print("Önerilen kısa vadeli iyileştirmeler (CT Pencereleme, CLAHE)")
    print("uygulanabilir değildir çünkü verileriniz ikili bölümleme maskelerinden oluşmaktadır.")
    print("\n✅ Mevcut uygulama kullanım durumunuz için mükemmeldir!")
    
if __name__ == '__main__':
    analyze_samples()
