"""
NaN Handler - Eksik Veri İşleme Modülü
Veri setindeki NaN ve eksik değerleri tespit etme ve işleme
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json


class NaNHandler:
    """NaN ve eksik değerleri işlemek için modüler sınıf"""
    
    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: Detaylı çıktı göster
        """
        self.verbose = verbose
        self.nan_report = {}
        
    def analyze_nan(self, df: pd.DataFrame) -> Dict:
        """
        DataFrame'deki NaN değerleri analiz eder
        
        Args:
            df: Analiz edilecek DataFrame
            
        Returns:
            Dict: NaN analiz raporu
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns_with_nan': {},
            'rows_with_nan': 0,
            'total_nan_values': 0,
            'nan_percentage': 0.0
        }
        
        # Her kolon için NaN sayısı
        for col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                report['columns_with_nan'][col] = {
                    'count': int(nan_count),
                    'percentage': float(nan_count / len(df) * 100)
                }
                report['total_nan_values'] += nan_count
        
        # En az bir NaN içeren satır sayısı
        report['rows_with_nan'] = int(df.isna().any(axis=1).sum())
        
        # Genel NaN yüzdesi
        total_cells = len(df) * len(df.columns)
        report['nan_percentage'] = float(report['total_nan_values'] / total_cells * 100) if total_cells > 0 else 0.0
        
        self.nan_report = report
        
        if self.verbose:
            self._print_nan_report(report)
        
        return report
    
    def _print_nan_report(self, report: Dict):
        """NaN raporunu yazdırır"""
        print("\n" + "="*60)
        print("📊 NaN DEĞERLERİ ANALİZ RAPORU")
        print("="*60)
        print(f"Toplam Satır: {report['total_rows']}")
        print(f"Toplam Kolon: {report['total_columns']}")
        print(f"NaN İçeren Satır: {report['rows_with_nan']}")
        print(f"Toplam NaN Değer: {report['total_nan_values']}")
        print(f"Genel NaN Oranı: {report['nan_percentage']:.2f}%")
        
        if report['columns_with_nan']:
            print("\n📋 Kolonlara Göre NaN Dağılımı:")
            print("-" * 60)
            for col, info in report['columns_with_nan'].items():
                print(f"  {col:20s}: {info['count']:5d} NaN ({info['percentage']:5.2f}%)")
        else:
            print("\n✅ Veri setinde NaN değer bulunamadı!")
        print("="*60 + "\n")
    
    def remove_rows_with_nan(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        NaN içeren satırları siler
        
        Args:
            df: DataFrame
            columns: Kontrol edilecek kolonlar (None ise tüm kolonlar)
            
        Returns:
            Temizlenmiş DataFrame
        """
        original_len = len(df)
        
        if columns:
            df_clean = df.dropna(subset=columns)
        else:
            df_clean = df.dropna()
        
        removed_count = original_len - len(df_clean)
        
        if self.verbose:
            print(f"🗑️  {removed_count} satır silindi (NaN içeren)")
            print(f"   Kalan satır sayısı: {len(df_clean)}")
        
        return df_clean
    
    def fill_nan_with_value(self, df: pd.DataFrame, value: Union[int, float, str], 
                           columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        NaN değerleri belirtilen değerle doldurur
        
        Args:
            df: DataFrame
            value: Doldurma değeri
            columns: Doldurulacak kolonlar (None ise tüm kolonlar)
            
        Returns:
            Doldurulmuş DataFrame
        """
        df_filled = df.copy()
        
        if columns:
            for col in columns:
                if col in df.columns:
                    filled_count = df_filled[col].isna().sum()
                    df_filled[col].fillna(value, inplace=True)
                    if self.verbose and filled_count > 0:
                        print(f"✓ '{col}' kolonunda {filled_count} NaN değer '{value}' ile dolduruldu")
        else:
            filled_count = df_filled.isna().sum().sum()
            df_filled.fillna(value, inplace=True)
            if self.verbose and filled_count > 0:
                print(f"✓ Toplam {filled_count} NaN değer '{value}' ile dolduruldu")
        
        return df_filled
    
    def fill_nan_with_mean(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        NaN değerleri ortalama ile doldurur (numerik kolonlar için)
        
        Args:
            df: DataFrame
            columns: Doldurulacak kolonlar (None ise tüm numerik kolonlar)
            
        Returns:
            Doldurulmuş DataFrame
        """
        df_filled = df.copy()
        
        # Numerik kolonları belirle
        if columns:
            numeric_cols = [col for col in columns if col in df.select_dtypes(include=[np.number]).columns]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            nan_count = df_filled[col].isna().sum()
            if nan_count > 0:
                mean_value = df_filled[col].mean()
                df_filled[col].fillna(mean_value, inplace=True)
                if self.verbose:
                    print(f"✓ '{col}' kolonunda {nan_count} NaN değer ortalama ({mean_value:.4f}) ile dolduruldu")
        
        return df_filled
    
    def fill_nan_with_median(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        NaN değerleri medyan ile doldurur (numerik kolonlar için)
        
        Args:
            df: DataFrame
            columns: Doldurulacak kolonlar (None ise tüm numerik kolonlar)
            
        Returns:
            Doldurulmuş DataFrame
        """
        df_filled = df.copy()
        
        # Numerik kolonları belirle
        if columns:
            numeric_cols = [col for col in columns if col in df.select_dtypes(include=[np.number]).columns]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            nan_count = df_filled[col].isna().sum()
            if nan_count > 0:
                median_value = df_filled[col].median()
                df_filled[col].fillna(median_value, inplace=True)
                if self.verbose:
                    print(f"✓ '{col}' kolonunda {nan_count} NaN değer medyan ({median_value:.4f}) ile dolduruldu")
        
        return df_filled
    
    def fill_nan_with_mode(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        NaN değerleri mod (en sık görülen değer) ile doldurur
        
        Args:
            df: DataFrame
            columns: Doldurulacak kolonlar (None ise tüm kolonlar)
            
        Returns:
            Doldurulmuş DataFrame
        """
        df_filled = df.copy()
        
        cols_to_fill = columns if columns else df.columns.tolist()
        
        for col in cols_to_fill:
            if col in df.columns:
                nan_count = df_filled[col].isna().sum()
                if nan_count > 0:
                    mode_values = df_filled[col].mode()
                    if len(mode_values) > 0:
                        mode_value = mode_values[0]
                        df_filled[col].fillna(mode_value, inplace=True)
                        if self.verbose:
                            print(f"✓ '{col}' kolonunda {nan_count} NaN değer mod ({mode_value}) ile dolduruldu")
        
        return df_filled
    
    def fill_nan_forward(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        NaN değerleri forward fill ile doldurur (önceki değeri kullan)
        
        Args:
            df: DataFrame
            columns: Doldurulacak kolonlar (None ise tüm kolonlar)
            
        Returns:
            Doldurulmuş DataFrame
        """
        df_filled = df.copy()
        
        if columns:
            df_filled[columns] = df_filled[columns].fillna(method='ffill')
        else:
            df_filled = df_filled.fillna(method='ffill')
        
        if self.verbose:
            remaining_nan = df_filled.isna().sum().sum()
            print(f"✓ Forward fill uygulandı (Kalan NaN: {remaining_nan})")
        
        return df_filled
    
    def fill_nan_backward(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        NaN değerleri backward fill ile doldurur (sonraki değeri kullan)
        
        Args:
            df: DataFrame
            columns: Doldurulacak kolonlar (None ise tüm kolonlar)
            
        Returns:
            Doldurulmuş DataFrame
        """
        df_filled = df.copy()
        
        if columns:
            df_filled[columns] = df_filled[columns].fillna(method='bfill')
        else:
            df_filled = df_filled.fillna(method='bfill')
        
        if self.verbose:
            remaining_nan = df_filled.isna().sum().sum()
            print(f"✓ Backward fill uygulandı (Kalan NaN: {remaining_nan})")
        
        return df_filled
    
    def save_report(self, output_path: str):
        """NaN raporunu JSON dosyasına kaydeder"""
        if not self.nan_report:
            print("⚠️  Henüz analiz yapılmadı! Önce analyze_nan() çalıştırın.")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.nan_report, f, indent=2, ensure_ascii=False)
        
        print(f"✓ NaN raporu kaydedildi: {output_path}")


def quick_nan_check(csv_path: str) -> Dict:
    """
    Hızlı NaN kontrolü için yardımcı fonksiyon
    
    Args:
        csv_path: CSV dosya yolu
        
    Returns:
        NaN analiz raporu
    """
    df = pd.read_csv(csv_path)
    handler = NaNHandler(verbose=True)
    return handler.analyze_nan(df)


def clean_dataset(csv_path: str, method: str = 'remove', output_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    Veri setini temizlemek için hızlı fonksiyon
    
    Args:
        csv_path: CSV dosya yolu
        method: Temizleme metodu ('remove', 'fill_value', 'fill_mean', 'fill_median', 'fill_mode')
        output_path: Temizlenmiş veriyi kaydetme yolu (opsiyonel)
        **kwargs: Method'a özgü parametreler
        
    Returns:
        Temizlenmiş DataFrame
    """
    df = pd.read_csv(csv_path)
    handler = NaNHandler(verbose=True)
    
    # İlk analiz
    handler.analyze_nan(df)
    
    # Temizleme yöntemi uygula
    if method == 'remove':
        df_clean = handler.remove_rows_with_nan(df, columns=kwargs.get('columns'))
    elif method == 'fill_value':
        df_clean = handler.fill_nan_with_value(df, value=kwargs.get('value', 0), columns=kwargs.get('columns'))
    elif method == 'fill_mean':
        df_clean = handler.fill_nan_with_mean(df, columns=kwargs.get('columns'))
    elif method == 'fill_median':
        df_clean = handler.fill_nan_with_median(df, columns=kwargs.get('columns'))
    elif method == 'fill_mode':
        df_clean = handler.fill_nan_with_mode(df, columns=kwargs.get('columns'))
    elif method == 'fill_forward':
        df_clean = handler.fill_nan_forward(df, columns=kwargs.get('columns'))
    elif method == 'fill_backward':
        df_clean = handler.fill_nan_backward(df, columns=kwargs.get('columns'))
    else:
        raise ValueError(f"Geçersiz method: {method}")
    
    # Sonuç analizi
    print("\n" + "="*60)
    print("🔍 TEMİZLEME SONRASI ANALİZ")
    print("="*60)
    handler.analyze_nan(df_clean)
    
    # Kaydetme
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        print(f"✓ Temizlenmiş veri kaydedildi: {output_path}")
    
    return df_clean


if __name__ == "__main__":
    # Test için örnek kullanım
    print("NaN Handler Modülü - Test")
    print("-" * 60)
    
    # Örnek veri oluştur
    test_data = {
        'id': [1, 2, 3, 4, 5],
        'value1': [10, np.nan, 30, 40, np.nan],
        'value2': [1.5, 2.5, np.nan, 4.5, 5.5],
        'category': ['A', 'B', np.nan, 'A', 'B']
    }
    df_test = pd.DataFrame(test_data)
    
    # Analiz
    handler = NaNHandler()
    handler.analyze_nan(df_test)
    
    # Ortalama ile doldur
    print("\n--- Ortalama ile Doldurma ---")
    df_filled = handler.fill_nan_with_mean(df_test)
    print(df_filled)
