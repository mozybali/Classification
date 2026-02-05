"""
Sınıf Dengeleme ve Veri Arttırma Menü Sistemi
Dengesizlik analizi ve augmentation yönetimi için interaktif menü
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Dict
import yaml

# Src modüllerini import et
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing.class_balancer import ClassBalancer
from src.preprocessing.augmentation_manager import AugmentationManager


class ClassBalanceMenu:
    """Sınıf dengeleme ve augmentation için interaktif menü sistemi"""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.csv_path: Optional[str] = None
        self.label_column: str = "ROI_anomaly"
        self.balancer = ClassBalancer(verbose=True)
        self.aug_manager = AugmentationManager(verbose=True)
        self.balance_report: Optional[Dict] = None
        self.augmentation_config: Optional[Dict] = None
        self.modified = False
        
    def clear_screen(self):
        """Ekranı temizle"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self, title: str):
        """Başlık yazdır"""
        print("\n" + "="*70)
        print(f"  {title}")
        print("="*70 + "\n")
    
    def wait_for_enter(self):
        """Kullanıcıdan Enter bekle"""
        input("\nDevam etmek için Enter'a basın...")
    
    def load_dataset(self):
        """Veri seti yükle"""
        self.clear_screen()
        self.print_header("📁 VERİ SETİ YÜKLEME")
        
        default_path = "NeAR_dataset/ALAN/info.csv"
        
        print(f"Varsayılan: {default_path}")
        csv_path = input("CSV dosya yolu (Enter = varsayılan): ").strip()
        
        if not csv_path:
            csv_path = default_path
        
        try:
            self.df = pd.read_csv(csv_path)
            self.csv_path = csv_path
            self.modified = False
            
            print(f"\n✅ Veri seti başarıyla yüklendi!")
            print(f"   Dosya: {csv_path}")
            print(f"   Satır sayısı: {len(self.df)}")
            print(f"   Kolon sayısı: {len(self.df.columns)}")
            print(f"   Kolonlar: {', '.join(self.df.columns.tolist())}")
            
            # Etiket kolonu seç
            if 'ROI_anomaly' in self.df.columns:
                self.label_column = 'ROI_anomaly'
            else:
                print(f"\nMevcut kolonlar: {', '.join(self.df.columns.tolist())}")
                label_col = input("Etiket kolonu adı: ").strip()
                if label_col in self.df.columns:
                    self.label_column = label_col
                else:
                    print(f"⚠️  Kolon bulunamadı: {label_col}")
            
        except Exception as e:
            print(f"\n❌ HATA: {e}")
        
        self.wait_for_enter()
    
    def analyze_balance(self):
        """Sınıf dengesini analiz et"""
        self.clear_screen()
        self.print_header("📊 SINIF DENGESİ ANALİZİ")
        
        if self.df is None:
            print("⚠️  Önce veri seti yüklemelisiniz!")
            self.wait_for_enter()
            return
        
        try:
            # Analiz
            self.balance_report = self.balancer.analyze_class_distribution(self.df, self.label_column)
            
            # Öneriler
            print("\n💡 ÖNERİLER:")
            print("-" * 70)
            recommendations = self.balancer.recommend_strategy(
                self.balance_report['imbalance_metrics']['imbalance_ratio']
            )
            for rec in recommendations:
                print(rec)
            print("-" * 70)
            
            # Class weights
            print("\n⚖️  SINIF AĞIRLIKLARI (Loss function için):")
            print("-" * 70)
            weights = self.balancer.calculate_class_weights(self.df, self.label_column)
            
        except Exception as e:
            print(f"\n❌ HATA: {e}")
        
        self.wait_for_enter()
    
    def visualize_distribution(self):
        """Sınıf dağılımını görselleştir"""
        self.clear_screen()
        self.print_header("📈 SINIF DAĞILIMI GÖRSELLEŞTİRME")
        
        if self.df is None:
            print("⚠️  Önce veri seti yüklemelisiniz!")
            self.wait_for_enter()
            return
        
        save = input("Görselleştirmeyi kaydetmek istiyor musunuz? (e/h): ").strip().lower()
        
        output_path = None
        if save == 'e':
            default_path = "outputs/class_distribution.png"
            output_path = input(f"Dosya yolu (Enter = {default_path}): ").strip() or default_path
        
        try:
            self.balancer.visualize_distribution(self.df, self.label_column, output_path)
            if output_path is None:
                print("\n⚠️  Grafik penceresi kapatıldı.")
        except Exception as e:
            print(f"\n❌ HATA: {e}")
        
        self.wait_for_enter()
    
    def balance_menu(self):
        """Dengeleme menüsü"""
        while True:
            self.clear_screen()
            self.print_header("⚖️  SINIF DENGELEME MENÜSÜ")
            
            if self.df is None:
                print("⚠️  Önce veri seti yüklemelisiniz!")
                self.wait_for_enter()
                return
            
            if self.balance_report:
                ratio = self.balance_report['imbalance_metrics']['imbalance_ratio']
                print(f"📊 Mevcut Dengesizlik Oranı: {ratio:.2f}:1")
                print()
            
            print("1. Random Oversampling (Azınlık sınıfını çoğalt)")
            print("2. Random Undersampling (Çoğunluk sınıfını azalt)")
            print("3. SMOTE (Synthetic Minority Oversampling)")
            print("4. ADASYN (Adaptive Synthetic Sampling)")
            print("5. SMOTE + Tomek Links (Kombine)")
            print("6. SMOTE + ENN (Kombine)")
            print("7. Sınıf Ağırlıklarını Hesapla ve Kaydet")
            print("0. Ana Menüye Dön")
            
            choice = input("\nSeçiminiz: ").strip()
            
            if choice == '1':
                self.apply_random_oversample()
            elif choice == '2':
                self.apply_random_undersample()
            elif choice == '3':
                self.apply_smote()
            elif choice == '4':
                self.apply_adasyn()
            elif choice == '5':
                self.apply_combined('smote_tomek')
            elif choice == '6':
                self.apply_combined('smote_enn')
            elif choice == '7':
                self.save_class_weights()
            elif choice == '0':
                break
            else:
                print("\n❌ Geçersiz seçim!")
                self.wait_for_enter()
    
    def apply_random_oversample(self):
        """Random oversampling uygula"""
        self.clear_screen()
        self.print_header("🔄 RANDOM OVERSAMPLING")
        
        print("Azınlık sınıfı rastgele kopyalanarak çoğaltılacak.")
        confirm = input("\nDevam edilsin mi? (e/h): ").strip().lower()
        
        if confirm == 'e':
            self.df = self.balancer.random_oversample(self.df, self.label_column)
            self.modified = True
            self.balance_report = None  # Yeniden analiz gerekli
            print("\n✅ İşlem tamamlandı!")
        else:
            print("\n❌ İşlem iptal edildi.")
        
        self.wait_for_enter()
    
    def apply_random_undersample(self):
        """Random undersampling uygula"""
        self.clear_screen()
        self.print_header("✂️  RANDOM UNDERSAMPLING")
        
        print("⚠️  Çoğunluk sınıfından örnekler SİLİNECEK!")
        print("Bu işlem veri kaybına neden olur.")
        confirm = input("\nEmin misiniz? (e/h): ").strip().lower()
        
        if confirm == 'e':
            self.df = self.balancer.random_undersample(self.df, self.label_column)
            self.modified = True
            self.balance_report = None
            print("\n✅ İşlem tamamlandı!")
        else:
            print("\n❌ İşlem iptal edildi.")
        
        self.wait_for_enter()
    
    def apply_smote(self):
        """SMOTE uygula"""
        self.clear_screen()
        self.print_header("🧬 SMOTE - Sentetik Örnek Üretimi")
        
        print("SMOTE, mevcut örnekleri kullanarak yeni sentetik örnekler üretir.")
        print("Bu, veri setini genişletir ve dengeyi sağlar.")
        
        k_neighbors = input("\nk_neighbors değeri (Enter = 5): ").strip()
        k_neighbors = int(k_neighbors) if k_neighbors else 5
        
        confirm = input("Devam edilsin mi? (e/h): ").strip().lower()
        
        if confirm == 'e':
            self.df = self.balancer.smote_balance(self.df, self.label_column, k_neighbors)
            self.modified = True
            self.balance_report = None
            print("\n✅ İşlem tamamlandı!")
        else:
            print("\n❌ İşlem iptal edildi.")
        
        self.wait_for_enter()
    
    def apply_adasyn(self):
        """ADASYN uygula"""
        self.clear_screen()
        self.print_header("🧬 ADASYN - Adaptif Sentetik Örnekleme")
        
        print("ADASYN, SMOTE'un gelişmiş versiyonudur.")
        print("Zor öğrenilen bölgelere daha fazla odaklanır.")
        
        confirm = input("\nDevam edilsin mi? (e/h): ").strip().lower()
        
        if confirm == 'e':
            self.df = self.balancer.adasyn_balance(self.df, self.label_column)
            self.modified = True
            self.balance_report = None
            print("\n✅ İşlem tamamlandı!")
        else:
            print("\n❌ İşlem iptal edildi.")
        
        self.wait_for_enter()
    
    def apply_combined(self, method: str):
        """Kombine sampling uygula"""
        self.clear_screen()
        method_name = "SMOTE + Tomek Links" if method == 'smote_tomek' else "SMOTE + ENN"
        self.print_header(f"🔗 {method_name}")
        
        print(f"{method_name}: Oversampling ve undersampling kombinasyonu")
        print("En iyi sonuçları verebilir ancak daha yavaştır.")
        
        confirm = input("\nDevam edilsin mi? (e/h): ").strip().lower()
        
        if confirm == 'e':
            self.df = self.balancer.combined_sampling(self.df, self.label_column, method)
            self.modified = True
            self.balance_report = None
            print("\n✅ İşlem tamamlandı!")
        else:
            print("\n❌ İşlem iptal edildi.")
        
        self.wait_for_enter()
    
    def save_class_weights(self):
        """Class weights'i hesapla ve kaydet"""
        self.clear_screen()
        self.print_header("💾 SINIF AĞIRLIKLARINI KAYDETME")
        
        weights = self.balancer.calculate_class_weights(self.df, self.label_column)
        
        default_path = "outputs/class_weights.yaml"
        output_path = input(f"\nKayıt yolu (Enter = {default_path}): ").strip() or default_path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # YAML formatında kaydet
        weights_config = {
            'class_weights': {
                'auto': False,
                'manual': weights
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(weights_config, f, default_flow_style=False)
        
        print(f"\n✅ Sınıf ağırlıkları kaydedildi: {output_path}")
        print("\nBu dosyayı config.yaml'a ekleyebilir veya training'de kullanabilirsiniz.")
        
        self.wait_for_enter()
    
    def augmentation_menu(self):
        """Augmentation menüsü"""
        while True:
            self.clear_screen()
            self.print_header("🎨 VERİ ARTTIRMA (AUGMENTATION) MENÜSÜ")
            
            if self.augmentation_config:
                print(f"📊 Mevcut Config: {self.augmentation_config.get('mode', 'custom')}")
                print()
            
            print("1. Mevcut Augmentation Config'i Analiz Et")
            print("2. Preset Config Seç (Light/Normal/Heavy)")
            print("3. Özel Config Oluştur (İnteraktif)")
            print("4. Augmentation Önerileri Al")
            print("5. Config'i Kaydet")
            print("6. Config'i Yükle")
            print("7. İki Config'i Karşılaştır")
            print("0. Ana Menüye Dön")
            
            choice = input("\nSeçiminiz: ").strip()
            
            if choice == '1':
                self.analyze_augmentation()
            elif choice == '2':
                self.select_preset_config()
            elif choice == '3':
                self.create_custom_config()
            elif choice == '4':
                self.get_augmentation_recommendations()
            elif choice == '5':
                self.save_augmentation_config()
            elif choice == '6':
                self.load_augmentation_config()
            elif choice == '7':
                self.compare_augmentation_configs()
            elif choice == '0':
                break
            else:
                print("\n❌ Geçersiz seçim!")
                self.wait_for_enter()
    
    def analyze_augmentation(self):
        """Mevcut augmentation config'i analiz et"""
        self.clear_screen()
        self.print_header("🔍 AUGMENTATION ANALİZİ")
        
        if self.augmentation_config is None:
            print("⚠️  Henüz config yüklenmedi. Varsayılan config kullanılacak.")
            self.augmentation_config = self.aug_manager.get_preset_config('normal')
        
        self.aug_manager.analyze_augmentation_config(self.augmentation_config)
        
        self.wait_for_enter()
    
    def select_preset_config(self):
        """Preset config seç"""
        self.clear_screen()
        self.print_header("📦 PRESET CONFIG SEÇME")
        
        print("Mevcut Preset'ler:")
        print("  1. Light      - Hafif augmentation (>5000 örnek için)")
        print("  2. Normal     - Orta seviye augmentation (1000-5000 örnek)")
        print("  3. Heavy      - Yoğun augmentation (<1000 örnek)")
        print("  4. Medical Kidney - Böbrek görüntüleme özel")
        
        choice = input("\nSeçiminiz (1-4): ").strip()
        
        presets = {'1': 'light', '2': 'normal', '3': 'heavy', '4': 'medical_kidney'}
        
        if choice in presets:
            level = presets[choice]
            self.augmentation_config = self.aug_manager.get_preset_config(level)
            print(f"\n✅ '{level}' config yüklendi!")
            
            # Analiz göster
            self.aug_manager.analyze_augmentation_config(self.augmentation_config)
        else:
            print("\n❌ Geçersiz seçim!")
        
        self.wait_for_enter()
    
    def create_custom_config(self):
        """Özel config oluştur"""
        self.clear_screen()
        self.print_header("🛠️  ÖZEL AUGMENTATION CONFIG OLUŞTURMA")
        
        print("Her transform için aktif/pasif durumunu belirleyeceksiniz.\n")
        
        self.augmentation_config = self.aug_manager.create_custom_config()
        
        print("\n✅ Özel config oluşturuldu!")
        self.aug_manager.analyze_augmentation_config(self.augmentation_config)
        
        self.wait_for_enter()
    
    def get_augmentation_recommendations(self):
        """Augmentation önerileri al"""
        self.clear_screen()
        self.print_header("💡 AUGMENTATION ÖNERİLERİ")
        
        if self.df is None or self.balance_report is None:
            print("⚠️  Önce veri setini yükleyip analiz edin!")
            self.wait_for_enter()
            return
        
        dataset_size = len(self.df)
        imbalance_ratio = self.balance_report['imbalance_metrics']['imbalance_ratio']
        minority_count = self.balance_report['imbalance_metrics']['minority_count']
        
        recommendations = self.aug_manager.get_recommendations(
            dataset_size=dataset_size,
            imbalance_ratio=imbalance_ratio,
            minority_count=minority_count
        )
        
        print(f"\n📊 Dataset Özellikleri:")
        print(f"   Boyut: {dataset_size}")
        print(f"   Dengesizlik: {imbalance_ratio:.2f}:1")
        print(f"   Azınlık Sınıfı: {minority_count} örnek")
        
        print(f"\n🎯 Önerilen Augmentation Seviyesi: {recommendations['augmentation_level'].upper()}")
        
        print("\n📋 Stratejik Öneriler:")
        print("-" * 70)
        for strategy in recommendations['strategies']:
            print(strategy)
        
        if recommendations['warnings']:
            print("\n⚠️  Uyarılar:")
            for warning in recommendations['warnings']:
                print(f"   {warning}")
        
        print("-" * 70)
        
        # Otomatik olarak önerilen config'i yükle
        load_rec = input("\nÖnerilen config'i yüklemek ister misiniz? (e/h): ").strip().lower()
        if load_rec == 'e':
            self.augmentation_config = recommendations['recommended_config']
            print("\n✅ Önerilen config yüklendi!")
        
        self.wait_for_enter()
    
    def save_augmentation_config(self):
        """Augmentation config'i kaydet"""
        self.clear_screen()
        self.print_header("💾 AUGMENTATION CONFIG KAYDETME")
        
        if self.augmentation_config is None:
            print("⚠️  Henüz config oluşturulmadı!")
            self.wait_for_enter()
            return
        
        default_path = "configs/augmentation_config.json"
        output_path = input(f"Kayıt yolu (Enter = {default_path}): ").strip() or default_path
        
        self.aug_manager.save_config(self.augmentation_config, output_path)
        
        self.wait_for_enter()
    
    def load_augmentation_config(self):
        """Augmentation config'i yükle"""
        self.clear_screen()
        self.print_header("📂 AUGMENTATION CONFIG YÜKLEME")
        
        config_path = input("Config dosya yolu: ").strip()
        
        try:
            self.augmentation_config = self.aug_manager.load_config(config_path)
            self.aug_manager.analyze_augmentation_config(self.augmentation_config)
        except Exception as e:
            print(f"\n❌ HATA: {e}")
        
        self.wait_for_enter()
    
    def compare_augmentation_configs(self):
        """İki config'i karşılaştır"""
        self.clear_screen()
        self.print_header("⚖️  CONFIG KARŞILAŞTIRMA")
        
        print("İlk config dosyası:")
        config1_path = input("  Dosya yolu: ").strip()
        
        print("\nİkinci config dosyası:")
        config2_path = input("  Dosya yolu: ").strip()
        
        try:
            config1 = self.aug_manager.load_config(config1_path)
            config2 = self.aug_manager.load_config(config2_path)
            
            comparison = self.aug_manager.compare_configs(config1, config2)
            
            print("\n" + "="*70)
            print("📊 KARŞILAŞTIRMA SONUÇLARI")
            print("="*70)
            print(f"Config 1 Agresiflik: {comparison['config1_aggressiveness']:.1f}/100")
            print(f"Config 2 Agresiflik: {comparison['config2_aggressiveness']:.1f}/100")
            
            if comparison['differences']:
                print("\n🔄 Farklılıklar:")
                for diff in comparison['differences']:
                    print(f"   {diff['transform']:25s}: Config1={diff['config1']}, Config2={diff['config2']}")
            
            if comparison['similarities']:
                print(f"\n✅ Ortak Aktif Transformlar: {', '.join(comparison['similarities'])}")
            
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ HATA: {e}")
        
        self.wait_for_enter()
    
    def save_current_data(self):
        """Dengelenmiş veriyi kaydet"""
        self.clear_screen()
        self.print_header("💾 VERİ KAYDETME")
        
        if self.df is None:
            print("⚠️  Veri seti yüklenmemiş!")
            self.wait_for_enter()
            return
        
        default_path = "outputs/balanced_data.csv"
        output_path = input(f"Çıktı dosyası (Enter = {default_path}): ").strip() or default_path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        self.modified = False
        
        print(f"\n✅ Dengelenmiş veri kaydedildi: {output_path}")
        self.wait_for_enter()
    
    def show_summary(self):
        """Özet bilgi göster"""
        self.clear_screen()
        self.print_header("ℹ️  ÖZET BİLGİ")
        
        if self.df is None:
            print("⚠️  Veri seti yüklenmemiş!")
            self.wait_for_enter()
            return
        
        print(f"📁 Dosya: {self.csv_path}")
        print(f"🏷️  Etiket Kolonu: {self.label_column}")
        print(f"📊 Toplam Örnek: {len(self.df)}")
        print(f"{'✏️  Değiştirildi' if self.modified else '✅ Kaydedildi'}")
        
        if self.balance_report:
            print(f"\n⚖️  Dengesizlik Oranı: {self.balance_report['imbalance_metrics']['imbalance_ratio']:.2f}:1")
            print(f"   Dengeli: {'✅ Evet' if self.balance_report['imbalance_metrics']['is_balanced'] else '⚠️  Hayır'}")
        
        if self.augmentation_config:
            print(f"\n🎨 Augmentation Config: {self.augmentation_config.get('mode', 'custom')}")
        
        self.wait_for_enter()
    
    def main(self):
        """Ana menü"""
        while True:
            self.clear_screen()
            self.print_header("⚖️  SINIF DENGELEME VE VERİ ARTTIRMA SİSTEMİ")
            
            if self.df is not None:
                print(f"📊 Yüklü Veri: {self.csv_path}")
                print(f"   Satır: {len(self.df)} | Etiket: {self.label_column}")
                if self.modified:
                    print("   ⚠️  Değişiklikler kaydedilmedi!")
                print()
            
            print("1. Veri Seti Yükle")
            print("2. Sınıf Dağılımını Analiz Et")
            print("3. Sınıf Dağılımını Görselleştir")
            print("4. Sınıf Dengeleme İşlemleri")
            print("5. Augmentation Yönetimi")
            print("6. Dengelenmiş Veriyi Kaydet")
            print("7. Özet Bilgi Göster")
            print("0. Çıkış")
            
            choice = input("\nSeçiminiz: ").strip()
            
            if choice == '1':
                self.load_dataset()
            elif choice == '2':
                self.analyze_balance()
            elif choice == '3':
                self.visualize_distribution()
            elif choice == '4':
                self.balance_menu()
            elif choice == '5':
                self.augmentation_menu()
            elif choice == '6':
                self.save_current_data()
            elif choice == '7':
                self.show_summary()
            elif choice == '0':
                if self.modified:
                    confirm = input("\n⚠️  Kaydedilmemiş değişiklikler var! Çıkmak istediğinizden emin misiniz? (e/h): ").strip().lower()
                    if confirm == 'e':
                        print("\n👋 Görüşmek üzere!")
                        break
                else:
                    print("\n👋 Görüşmek üzere!")
                    break
            else:
                print("\n❌ Geçersiz seçim!")
                self.wait_for_enter()


    def main_menu(self):
        """Backward compatibility alias for main"""
        self.main()


def main():
    """Ana fonksiyon"""
    menu = ClassBalanceMenu()
    menu.main()


if __name__ == "__main__":
    main()
