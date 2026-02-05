"""
Veri Önişleme Menü Sistemi
NaN işleme ve veri bölme işlemleri için interaktif menü
"""

import sys
import os
from pathlib import Path
import pandas as pd
from typing import Optional, Dict

# Src modüllerini import et
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing.nan_handler import NaNHandler
from src.preprocessing.data_splitter import DataSplitter


class DataPreprocessingMenu:
    """Veri önişleme için interaktif menü sistemi"""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.csv_path: Optional[str] = None
        self.nan_handler = NaNHandler(verbose=True)
        self.data_splitter = DataSplitter(verbose=True)
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
        
        # Varsayılan path
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
            
        except Exception as e:
            print(f"\n❌ HATA: {e}")
        
        self.wait_for_enter()
    
    def nan_menu(self):
        """NaN işleme menüsü"""
        while True:
            self.clear_screen()
            self.print_header("🔍 NaN DEĞERLERİ İŞLEME MENÜSÜ")
            
            if self.df is None:
                print("⚠️  Önce veri seti yüklemelisiniz!")
                self.wait_for_enter()
                return
            
            print("1. NaN Değerlerini Analiz Et")
            print("2. NaN İçeren Satırları Sil")
            print("3. NaN Değerlerini Sabit Değerle Doldur")
            print("4. NaN Değerlerini Ortalama ile Doldur")
            print("5. NaN Değerlerini Medyan ile Doldur")
            print("6. NaN Değerlerini Mod ile Doldur")
            print("7. NaN Değerlerini Forward Fill ile Doldur")
            print("8. NaN Değerlerini Backward Fill ile Doldur")
            print("9. NaN Raporunu Kaydet")
            print("0. Ana Menüye Dön")
            
            choice = input("\nSeçiminiz: ").strip()
            
            if choice == '1':
                self.analyze_nan()
            elif choice == '2':
                self.remove_nan_rows()
            elif choice == '3':
                self.fill_nan_with_value()
            elif choice == '4':
                self.fill_nan_with_mean()
            elif choice == '5':
                self.fill_nan_with_median()
            elif choice == '6':
                self.fill_nan_with_mode()
            elif choice == '7':
                self.fill_nan_forward()
            elif choice == '8':
                self.fill_nan_backward()
            elif choice == '9':
                self.save_nan_report()
            elif choice == '0':
                break
            else:
                print("\n❌ Geçersiz seçim!")
                self.wait_for_enter()
    
    def analyze_nan(self):
        """NaN analizi"""
        self.clear_screen()
        self.print_header("📊 NaN ANALİZİ")
        self.nan_handler.analyze_nan(self.df)
        self.wait_for_enter()
    
    def remove_nan_rows(self):
        """NaN içeren satırları sil"""
        self.clear_screen()
        self.print_header("🗑️ NaN SATIRLARINI SİLME")
        
        print("Hangi kolonlardaki NaN'ları silmek istiyorsunuz?")
        print("(Boş bırakırsanız tüm kolonlar kontrol edilir)")
        print(f"Mevcut kolonlar: {', '.join(self.df.columns.tolist())}")
        
        cols_input = input("\nKolonlar (virgülle ayırın): ").strip()
        columns = [c.strip() for c in cols_input.split(',')] if cols_input else None
        
        confirm = input(f"\nEmin misiniz? (e/h): ").strip().lower()
        if confirm == 'e':
            self.df = self.nan_handler.remove_rows_with_nan(self.df, columns)
            self.modified = True
            print("\n✅ İşlem tamamlandı!")
        else:
            print("\n❌ İşlem iptal edildi.")
        
        self.wait_for_enter()
    
    def fill_nan_with_value(self):
        """NaN'ları sabit değerle doldur"""
        self.clear_screen()
        self.print_header("✏️ NaN'LARI SABİT DEĞERLE DOLDURMA")
        
        value = input("Doldurma değeri: ").strip()
        
        # Tip dönüşümü
        try:
            if '.' in value:
                value = float(value)
            else:
                try:
                    value = int(value)
                except:
                    pass  # String olarak kalacak
        except:
            pass
        
        print(f"\nMevcut kolonlar: {', '.join(self.df.columns.tolist())}")
        cols_input = input("Kolonlar (virgülle ayırın, boş = hepsi): ").strip()
        columns = [c.strip() for c in cols_input.split(',')] if cols_input else None
        
        self.df = self.nan_handler.fill_nan_with_value(self.df, value, columns)
        self.modified = True
        print("\n✅ İşlem tamamlandı!")
        self.wait_for_enter()
    
    def fill_nan_with_mean(self):
        """NaN'ları ortalama ile doldur"""
        self.clear_screen()
        self.print_header("📊 NaN'LARI ORTALAMA İLE DOLDURMA")
        
        print(f"Numerik kolonlar: {', '.join(self.df.select_dtypes(include=['number']).columns.tolist())}")
        cols_input = input("Kolonlar (virgülle ayırın, boş = tüm numerik): ").strip()
        columns = [c.strip() for c in cols_input.split(',')] if cols_input else None
        
        self.df = self.nan_handler.fill_nan_with_mean(self.df, columns)
        self.modified = True
        print("\n✅ İşlem tamamlandı!")
        self.wait_for_enter()
    
    def fill_nan_with_median(self):
        """NaN'ları medyan ile doldur"""
        self.clear_screen()
        self.print_header("📊 NaN'LARI MEDYAN İLE DOLDURMA")
        
        print(f"Numerik kolonlar: {', '.join(self.df.select_dtypes(include=['number']).columns.tolist())}")
        cols_input = input("Kolonlar (virgülle ayırın, boş = tüm numerik): ").strip()
        columns = [c.strip() for c in cols_input.split(',')] if cols_input else None
        
        self.df = self.nan_handler.fill_nan_with_median(self.df, columns)
        self.modified = True
        print("\n✅ İşlem tamamlandı!")
        self.wait_for_enter()
    
    def fill_nan_with_mode(self):
        """NaN'ları mod ile doldur"""
        self.clear_screen()
        self.print_header("📊 NaN'LARI MOD İLE DOLDURMA")
        
        print(f"Mevcut kolonlar: {', '.join(self.df.columns.tolist())}")
        cols_input = input("Kolonlar (virgülle ayırın, boş = hepsi): ").strip()
        columns = [c.strip() for c in cols_input.split(',')] if cols_input else None
        
        self.df = self.nan_handler.fill_nan_with_mode(self.df, columns)
        self.modified = True
        print("\n✅ İşlem tamamlandı!")
        self.wait_for_enter()
    
    def fill_nan_forward(self):
        """Forward fill"""
        self.clear_screen()
        self.print_header("⏩ FORWARD FILL")
        
        print(f"Mevcut kolonlar: {', '.join(self.df.columns.tolist())}")
        cols_input = input("Kolonlar (virgülle ayırın, boş = hepsi): ").strip()
        columns = [c.strip() for c in cols_input.split(',')] if cols_input else None
        
        self.df = self.nan_handler.fill_nan_forward(self.df, columns)
        self.modified = True
        print("\n✅ İşlem tamamlandı!")
        self.wait_for_enter()
    
    def fill_nan_backward(self):
        """Backward fill"""
        self.clear_screen()
        self.print_header("⏪ BACKWARD FILL")
        
        print(f"Mevcut kolonlar: {', '.join(self.df.columns.tolist())}")
        cols_input = input("Kolonlar (virgülle ayırın, boş = hepsi): ").strip()
        columns = [c.strip() for c in cols_input.split(',')] if cols_input else None
        
        self.df = self.nan_handler.fill_nan_backward(self.df, columns)
        self.modified = True
        print("\n✅ İşlem tamamlandı!")
        self.wait_for_enter()
    
    def save_nan_report(self):
        """NaN raporunu kaydet"""
        self.clear_screen()
        self.print_header("💾 NaN RAPORU KAYDETME")
        
        default_path = "outputs/nan_report.json"
        output_path = input(f"Çıktı dosyası (Enter = {default_path}): ").strip()
        
        if not output_path:
            output_path = default_path
        
        self.nan_handler.save_report(output_path)
        self.wait_for_enter()
    
    def split_menu(self):
        """Veri bölme menüsü"""
        while True:
            self.clear_screen()
            self.print_header("✂️ VERİ BÖLME MENÜSÜ")
            
            if self.df is None:
                print("⚠️  Önce veri seti yüklemelisiniz!")
                self.wait_for_enter()
                return
            
            print("1. Basit Rastgele Bölme")
            print("2. Stratified Bölme (Sınıf Dengeli)")
            print("3. Patient-Level Bölme (Data Leakage Önleyici)")
            print("4. Mevcut Subset Kolonuna Göre Bölme")
            print("5. Split'leri Ayrı CSV Dosyalarına Kaydet")
            print("6. Split Bilgisini Yeni Kolon Olarak Ekle")
            print("0. Ana Menüye Dön")
            
            choice = input("\nSeçiminiz: ").strip()
            
            if choice == '1':
                self.split_simple()
            elif choice == '2':
                self.split_stratified()
            elif choice == '3':
                self.split_patient_level()
            elif choice == '4':
                self.split_existing()
            elif choice == '5':
                self.save_splits()
            elif choice == '6':
                self.save_split_column()
            elif choice == '0':
                break
            else:
                print("\n❌ Geçersiz seçim!")
                self.wait_for_enter()
    
    def get_split_ratios(self) -> tuple:
        """Kullanıcıdan split oranlarını al"""
        print("\nBölme oranlarını girin:")
        train_ratio = float(input("  Training oranı (0-1, varsayılan 0.7): ").strip() or "0.7")
        val_ratio = float(input("  Validation oranı (0-1, varsayılan 0.15): ").strip() or "0.15")
        test_ratio = float(input("  Test oranı (0-1, varsayılan 0.15): ").strip() or "0.15")
        
        # Kontrol
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.01:
            print(f"\n⚠️  Uyarı: Oranlar toplamı {total:.2f} (1.0 olmalı). Normalize ediliyor...")
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total
        
        return train_ratio, val_ratio, test_ratio
    
    def split_simple(self):
        """Basit rastgele bölme"""
        self.clear_screen()
        self.print_header("🎲 BASİT RASTGELE BÖLME")
        
        train_ratio, val_ratio, test_ratio = self.get_split_ratios()
        
        splits = self.data_splitter.split_simple(self.df, train_ratio, val_ratio, test_ratio)
        self.current_splits = splits
        
        print("\n✅ Veri seti başarıyla bölündü!")
        self.wait_for_enter()
    
    def split_stratified(self):
        """Stratified bölme"""
        self.clear_screen()
        self.print_header("⚖️ STRATIFIED BÖLME (Sınıf Dengeli)")
        
        print(f"Mevcut kolonlar: {', '.join(self.df.columns.tolist())}")
        stratify_col = input("\nStratification kolonu (varsayılan: ROI_anomaly): ").strip() or "ROI_anomaly"
        
        if stratify_col not in self.df.columns:
            print(f"\n❌ Hata: '{stratify_col}' kolonu bulunamadı!")
            self.wait_for_enter()
            return
        
        train_ratio, val_ratio, test_ratio = self.get_split_ratios()
        
        splits = self.data_splitter.split_stratified(self.df, stratify_col, train_ratio, val_ratio, test_ratio)
        self.current_splits = splits
        
        print("\n✅ Veri seti başarıyla bölündü!")
        self.wait_for_enter()
    
    def split_patient_level(self):
        """Patient-level bölme"""
        self.clear_screen()
        self.print_header("👤 PATIENT-LEVEL BÖLME")
        
        print("Bu bölme yöntemi aynı hastanın verilerinin aynı sette kalmasını sağlar.")
        print("Medical imaging için önemli: data leakage'ı önler!\n")
        
        print(f"Mevcut kolonlar: {', '.join(self.df.columns.tolist())}")
        patient_col = input("\nHasta ID kolonu (varsayılan: ROI_id): ").strip() or "ROI_id"
        
        stratify = input("Stratification kullanılsın mı? (e/h): ").strip().lower()
        stratify_col = None
        if stratify == 'e':
            stratify_col = input("Stratification kolonu (varsayılan: ROI_anomaly): ").strip() or "ROI_anomaly"
        
        train_ratio, val_ratio, test_ratio = self.get_split_ratios()
        
        splits = self.data_splitter.split_by_patient(
            self.df, patient_col, train_ratio, val_ratio, test_ratio, stratify_col
        )
        self.current_splits = splits
        
        print("\n✅ Veri seti başarıyla bölündü!")
        self.wait_for_enter()
    
    def split_existing(self):
        """Mevcut subset kolonuna göre bölme"""
        self.clear_screen()
        self.print_header("📋 MEVCUT SUBSET KOLONUNA GÖRE BÖLME")
        
        print(f"Mevcut kolonlar: {', '.join(self.df.columns.tolist())}")
        split_col = input("\nSubset kolonu (varsayılan: subset): ").strip() or "subset"
        
        if split_col not in self.df.columns:
            print(f"\n❌ Hata: '{split_col}' kolonu bulunamadı!")
            self.wait_for_enter()
            return
        
        splits = self.data_splitter.split_by_existing_column(self.df, split_col)
        self.current_splits = splits
        
        print("\n✅ Veri seti başarıyla bölündü!")
        self.wait_for_enter()
    
    def save_splits(self):
        """Split'leri kaydet"""
        self.clear_screen()
        self.print_header("💾 SPLIT'LERİ KAYDETME")
        
        if not hasattr(self, 'current_splits'):
            print("⚠️  Önce veri setini bölmelisiniz!")
            self.wait_for_enter()
            return
        
        default_dir = "outputs/splits"
        output_dir = input(f"Çıktı dizini (Enter = {default_dir}): ").strip() or default_dir
        prefix = input("Dosya prefix (Enter = boş): ").strip()
        
        self.data_splitter.save_splits(self.current_splits, output_dir, prefix)
        print("\n✅ Split'ler başarıyla kaydedildi!")
        self.wait_for_enter()
    
    def save_split_column(self):
        """Split bilgisini kolon olarak ekle"""
        self.clear_screen()
        self.print_header("💾 SPLIT BİLGİSİNİ KOLON OLARAK EKLEME")
        
        if not hasattr(self, 'current_splits'):
            print("⚠️  Önce veri setini bölmelisiniz!")
            self.wait_for_enter()
            return
        
        default_path = "outputs/data_with_splits.csv"
        output_path = input(f"Çıktı dosyası (Enter = {default_path}): ").strip() or default_path
        column_name = input("Kolon ismi (varsayılan: split): ").strip() or "split"
        
        self.data_splitter.save_split_column(self.df, self.current_splits, output_path, column_name)
        print("\n✅ Dosya başarıyla kaydedildi!")
        self.wait_for_enter()
    
    def save_current_data(self):
        """Mevcut veriyi kaydet"""
        self.clear_screen()
        self.print_header("💾 VERİ KAYDETME")
        
        if self.df is None:
            print("⚠️  Veri seti yüklenmemiş!")
            self.wait_for_enter()
            return
        
        default_path = "outputs/processed_data.csv"
        output_path = input(f"Çıktı dosyası (Enter = {default_path}): ").strip() or default_path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        self.modified = False
        
        print(f"\n✅ Veri başarıyla kaydedildi: {output_path}")
        self.wait_for_enter()
    
    def show_data_info(self):
        """Veri bilgilerini göster"""
        self.clear_screen()
        self.print_header("ℹ️ VERİ SETİ BİLGİLERİ")
        
        if self.df is None:
            print("⚠️  Veri seti yüklenmemiş!")
            self.wait_for_enter()
            return
        
        print(f"Dosya: {self.csv_path}")
        print(f"Değiştirildi: {'Evet' if self.modified else 'Hayır'}")
        print(f"\nBoyut: {self.df.shape}")
        print(f"Satır: {len(self.df)}")
        print(f"Kolon: {len(self.df.columns)}")
        print(f"\nKolonlar: {', '.join(self.df.columns.tolist())}")
        print(f"\nVeri Tipleri:")
        print(self.df.dtypes)
        print(f"\nİlk 5 satır:")
        print(self.df.head())
        
        self.wait_for_enter()
    
    def main(self):
        """Ana menü"""
        while True:
            self.clear_screen()
            self.print_header("🔬 VERİ ÖNİŞLEME MENÜ SİSTEMİ")
            
            if self.df is not None:
                print(f"📊 Yüklü Veri: {self.csv_path}")
                print(f"   Satır: {len(self.df)} | Kolon: {len(self.df.columns)}")
                if self.modified:
                    print("   ⚠️  Değişiklikler kaydedilmedi!")
                print()
            
            print("1. Veri Seti Yükle")
            print("2. NaN Değerleri İşleme")
            print("3. Veri Setini Bölme (Train/Val/Test)")
            print("4. Veri Setini Kaydet")
            print("5. Veri Seti Bilgilerini Göster")
            print("0. Çıkış")
            
            choice = input("\nSeçiminiz: ").strip()
            
            if choice == '1':
                self.load_dataset()
            elif choice == '2':
                self.nan_menu()
            elif choice == '3':
                self.split_menu()
            elif choice == '4':
                self.save_current_data()
            elif choice == '5':
                self.show_data_info()
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
    menu = DataPreprocessingMenu()
    menu.main()


if __name__ == "__main__":
    main()
