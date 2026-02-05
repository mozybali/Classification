"""
Data Splitter - Veri Bölme Modülü
Train/Validation/Test setlerine ayırma işlemleri
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import json


class DataSplitter:
    """Veri setini train/val/test olarak bölen modüler sınıf"""
    
    def __init__(self, random_state: int = 42, verbose: bool = True):
        """
        Args:
            random_state: Tekrarlanabilirlik için seed
            verbose: Detaylı çıktı göster
        """
        self.random_state = random_state
        self.verbose = verbose
        self.split_report = {}
        
    def split_simple(self, 
                     df: pd.DataFrame,
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
        """
        Basit rastgele bölme (stratification olmadan)
        
        Args:
            df: Bölünecek DataFrame
            train_ratio: Training seti oranı
            val_ratio: Validation seti oranı  
            test_ratio: Test seti oranı
            
        Returns:
            Dict: {'train': DataFrame, 'val': DataFrame, 'test': DataFrame}
        """
        # Oranları kontrol et
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Oranlar toplamı 1.0 olmalı!"
        
        # İlk bölme: train + (val+test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=(1 - train_ratio),
            random_state=self.random_state
        )
        
        # İkinci bölme: val + test
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_ratio,
            random_state=self.random_state
        )
        
        splits = {
            'train': train_df.reset_index(drop=True),
            'val': val_df.reset_index(drop=True),
            'test': test_df.reset_index(drop=True)
        }
        
        if self.verbose:
            self._print_split_info(splits, method="Simple Random Split")
        
        self.split_report = self._generate_report(splits)
        return splits
    
    def split_stratified(self,
                        df: pd.DataFrame,
                        stratify_column: str,
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
        """
        Stratified split - Sınıf dengesi korunarak bölme
        
        Args:
            df: Bölünecek DataFrame
            stratify_column: Stratification için kullanılacak kolon (örn: 'ROI_anomaly')
            train_ratio: Training seti oranı
            val_ratio: Validation seti oranı
            test_ratio: Test seti oranı
            
        Returns:
            Dict: {'train': DataFrame, 'val': DataFrame, 'test': DataFrame}
        """
        # Oranları kontrol et
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Oranlar toplamı 1.0 olmalı!"
        assert stratify_column in df.columns, f"Kolon bulunamadı: {stratify_column}"
        
        # İlk bölme: train + (val+test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(1 - train_ratio),
            stratify=df[stratify_column],
            random_state=self.random_state
        )
        
        # İkinci bölme: val + test
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_ratio,
            stratify=temp_df[stratify_column],
            random_state=self.random_state
        )
        
        splits = {
            'train': train_df.reset_index(drop=True),
            'val': val_df.reset_index(drop=True),
            'test': test_df.reset_index(drop=True)
        }
        
        if self.verbose:
            self._print_split_info(splits, method="Stratified Split", stratify_col=stratify_column)
        
        self.split_report = self._generate_report(splits, stratify_column)
        return splits
    
    def split_by_patient(self,
                        df: pd.DataFrame,
                        patient_id_column: str,
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15,
                        stratify_column: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Patient-level split - Aynı hastanın verileri aynı sette kalır
        Medical imaging için önemli: data leakage önler
        
        Args:
            df: Bölünecek DataFrame
            patient_id_column: Hasta ID kolonu (örn: ROI_id'den çıkarılacak)
            train_ratio: Training seti oranı
            val_ratio: Validation seti oranı
            test_ratio: Test seti oranı
            stratify_column: Opsiyonel stratification kolonu
            
        Returns:
            Dict: {'train': DataFrame, 'val': DataFrame, 'test': DataFrame}
        """
        # Oranları kontrol et
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Oranlar toplamı 1.0 olmalı!"
        
        # Hasta ID'lerini çıkar (örn: ZS000_L -> ZS000)
        if patient_id_column in df.columns:
            patient_ids = df[patient_id_column].unique()
        else:
            # ROI_id'den hasta ID'sini çıkar (son 2 karakter: _L veya _R)
            df_temp = df.copy()
            df_temp['_patient_id'] = df_temp['ROI_id'].str.rsplit('_', n=1).str[0]
            patient_ids = df_temp['_patient_id'].unique()
            patient_id_column = '_patient_id'
            df = df_temp
        
        # Hasta seviyesinde bölme
        if stratify_column:
            # Stratified patient split
            # Her hasta için dominant label'ı bul
            patient_labels = df.groupby(patient_id_column)[stratify_column].agg(
                lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
            )
            
            train_patients, temp_patients = train_test_split(
                patient_ids,
                test_size=(1 - train_ratio),
                stratify=patient_labels.loc[patient_ids],
                random_state=self.random_state
            )
            
            relative_test_ratio = test_ratio / (val_ratio + test_ratio)
            val_patients, test_patients = train_test_split(
                temp_patients,
                test_size=relative_test_ratio,
                stratify=patient_labels.loc[temp_patients],
                random_state=self.random_state
            )
        else:
            # Simple patient split
            train_patients, temp_patients = train_test_split(
                patient_ids,
                test_size=(1 - train_ratio),
                random_state=self.random_state
            )
            
            relative_test_ratio = test_ratio / (val_ratio + test_ratio)
            val_patients, test_patients = train_test_split(
                temp_patients,
                test_size=relative_test_ratio,
                random_state=self.random_state
            )
        
        # DataFrame'leri filtrele
        splits = {
            'train': df[df[patient_id_column].isin(train_patients)].reset_index(drop=True),
            'val': df[df[patient_id_column].isin(val_patients)].reset_index(drop=True),
            'test': df[df[patient_id_column].isin(test_patients)].reset_index(drop=True)
        }
        
        # Geçici patient_id kolonunu kaldır
        if '_patient_id' in splits['train'].columns:
            for split_name in splits:
                splits[split_name] = splits[split_name].drop(columns=['_patient_id'])
        
        if self.verbose:
            self._print_split_info(splits, method="Patient-Level Split", patient_col=patient_id_column)
        
        self.split_report = self._generate_report(splits, stratify_column)
        return splits
    
    def split_by_existing_column(self, df: pd.DataFrame, split_column: str = 'subset') -> Dict[str, pd.DataFrame]:
        """
        Mevcut bir kolona göre bölme (örn: 'subset' kolonu zaten train/test içeriyor)
        
        Args:
            df: Bölünecek DataFrame
            split_column: Split bilgisi içeren kolon
            
        Returns:
            Dict: {'train': DataFrame, 'val': DataFrame, 'test': DataFrame}
        """
        assert split_column in df.columns, f"Kolon bulunamadı: {split_column}"
        
        splits = {}
        unique_subsets = df[split_column].unique()
        
        # Subset isimlerini standartlaştır
        mapping = {
            'ZS-train': 'train',
            'ZS-test': 'test',
            'ZS-dev': 'val',
            'train': 'train',
            'test': 'test',
            'val': 'val',
            'dev': 'val'
        }
        
        for subset in unique_subsets:
            standardized_name = mapping.get(subset, subset)
            splits[standardized_name] = df[df[split_column] == subset].reset_index(drop=True)
        
        # Eğer val yoksa, train'den ayır
        if 'val' not in splits and 'train' in splits:
            train_df = splits['train']
            new_train, val_df = train_test_split(
                train_df,
                test_size=0.15,
                random_state=self.random_state
            )
            splits['train'] = new_train.reset_index(drop=True)
            splits['val'] = val_df.reset_index(drop=True)
        
        if self.verbose:
            self._print_split_info(splits, method="Existing Column Split")
        
        self.split_report = self._generate_report(splits)
        return splits
    
    def _print_split_info(self, splits: Dict, method: str = "Split", **kwargs):
        """Split bilgilerini yazdırır"""
        print("\n" + "="*60)
        print(f"📊 VERİ BÖLME RAPORU - {method}")
        print("="*60)
        
        total_samples = sum(len(df) for df in splits.values())
        
        for split_name, split_df in splits.items():
            split_size = len(split_df)
            split_percentage = (split_size / total_samples * 100) if total_samples > 0 else 0
            print(f"\n{split_name.upper():10s}: {split_size:5d} örnekler ({split_percentage:5.2f}%)")
            
            # Sınıf dağılımı varsa göster
            if 'ROI_anomaly' in split_df.columns:
                anomaly_count = split_df['ROI_anomaly'].sum()
                normal_count = len(split_df) - anomaly_count
                anomaly_ratio = (anomaly_count / len(split_df) * 100) if len(split_df) > 0 else 0
                print(f"           Normal: {normal_count:5d} ({100-anomaly_ratio:5.2f}%)")
                print(f"           Anomali: {anomaly_count:5d} ({anomaly_ratio:5.2f}%)")
        
        print("\n" + "="*60 + "\n")
    
    def _generate_report(self, splits: Dict, stratify_column: Optional[str] = None) -> Dict:
        """Detaylı rapor oluşturur"""
        report = {
            'total_samples': sum(len(df) for df in splits.values()),
            'splits': {}
        }
        
        for split_name, split_df in splits.items():
            split_info = {
                'size': len(split_df),
                'percentage': (len(split_df) / report['total_samples'] * 100) if report['total_samples'] > 0 else 0
            }
            
            # Stratification kolonunu analiz et
            if stratify_column and stratify_column in split_df.columns:
                value_counts = split_df[stratify_column].value_counts().to_dict()
                split_info['class_distribution'] = {
                    str(k): int(v) for k, v in value_counts.items()
                }
            
            report['splits'][split_name] = split_info
        
        return report
    
    def save_splits(self, splits: Dict, output_dir: str, prefix: str = ''):
        """
        Split'leri ayrı CSV dosyalarına kaydeder
        
        Args:
            splits: Split dictionary
            output_dir: Çıktı dizini
            prefix: Dosya ismi prefix (örn: 'cleaned_')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_df in splits.items():
            output_path = output_dir / f"{prefix}{split_name}.csv"
            split_df.to_csv(output_path, index=False)
            if self.verbose:
                print(f"✓ {split_name} seti kaydedildi: {output_path}")
        
        # Raporu da kaydet
        if self.split_report:
            report_path = output_dir / f"{prefix}split_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(self.split_report, f, indent=2, ensure_ascii=False)
            if self.verbose:
                print(f"✓ Split raporu kaydedildi: {report_path}")
    
    def save_split_column(self, df: pd.DataFrame, splits: Dict, output_path: str, column_name: str = 'split'):
        """
        Split bilgisini yeni bir kolon olarak ekleyip kaydeder
        
        Args:
            df: Orijinal DataFrame
            splits: Split dictionary
            output_path: Çıktı dosya yolu
            column_name: Yeni kolon ismi
        """
        df_with_split = df.copy()
        df_with_split[column_name] = None
        
        for split_name, split_df in splits.items():
            # Index'lere göre eşleştir
            indices = split_df.index.tolist()
            df_with_split.loc[indices, column_name] = split_name
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_with_split.to_csv(output_path, index=False)
        
        if self.verbose:
            print(f"✓ Split kolonu eklenmiş veri kaydedildi: {output_path}")


def quick_split(csv_path: str, 
                method: str = 'stratified',
                train_ratio: float = 0.7,
                val_ratio: float = 0.15,
                test_ratio: float = 0.15,
                output_dir: Optional[str] = None,
                **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Hızlı veri bölme için yardımcı fonksiyon
    
    Args:
        csv_path: CSV dosya yolu
        method: Bölme metodu ('simple', 'stratified', 'patient', 'existing')
        train_ratio: Training oranı
        val_ratio: Validation oranı
        test_ratio: Test oranı
        output_dir: Çıktı dizini (opsiyonel)
        **kwargs: Method'a özgü parametreler
        
    Returns:
        Split dictionary
    """
    df = pd.read_csv(csv_path)
    splitter = DataSplitter(verbose=True)
    
    if method == 'simple':
        splits = splitter.split_simple(df, train_ratio, val_ratio, test_ratio)
    elif method == 'stratified':
        stratify_col = kwargs.get('stratify_column', 'ROI_anomaly')
        splits = splitter.split_stratified(df, stratify_col, train_ratio, val_ratio, test_ratio)
    elif method == 'patient':
        patient_col = kwargs.get('patient_id_column', 'ROI_id')
        stratify_col = kwargs.get('stratify_column', None)
        splits = splitter.split_by_patient(df, patient_col, train_ratio, val_ratio, test_ratio, stratify_col)
    elif method == 'existing':
        split_col = kwargs.get('split_column', 'subset')
        splits = splitter.split_by_existing_column(df, split_col)
    else:
        raise ValueError(f"Geçersiz method: {method}")
    
    # Kaydetme
    if output_dir:
        splitter.save_splits(splits, output_dir)
    
    return splits


if __name__ == "__main__":
    # Test için örnek kullanım
    print("Data Splitter Modülü - Test")
    print("-" * 60)
    
    # Örnek veri oluştur
    test_data = {
        'ROI_id': [f'ZS{i:03d}_L' for i in range(100)] + [f'ZS{i:03d}_R' for i in range(100)],
        'ROI_anomaly': [i % 3 == 0 for i in range(200)],  # ~33% anomali
        'subset': ['ZS-train'] * 140 + ['ZS-test'] * 60
    }
    df_test = pd.DataFrame(test_data)
    
    splitter = DataSplitter()
    
    # Stratified split test
    print("\n--- Stratified Split Test ---")
    splits = splitter.split_stratified(df_test, 'ROI_anomaly', 0.7, 0.15, 0.15)
    
    # Patient-level split test
    print("\n--- Patient-Level Split Test ---")
    splits = splitter.split_by_patient(df_test, 'ROI_id', 0.7, 0.15, 0.15, stratify_column='ROI_anomaly')
