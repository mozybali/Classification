"""
Class Balancer - Sınıf Dengesizliği Analizi ve Dengeleme
Imbalanced dataset'ler için oversampling, undersampling ve SMOTE
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import Counter
import json
import warnings

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
    from imblearn.combine import SMOTETomek, SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    warnings.warn("⚠️  imbalanced-learn bulunamadı. Bazı özellikler kullanılamayacak. pip install imbalanced-learn")


class ClassBalancer:
    """Sınıf dengesizliğini analiz eden ve dengeleyen modüler sınıf"""
    
    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: Detaylı çıktı göster
        """
        self.verbose = verbose
        self.balance_report = {}
        
    def analyze_class_distribution(self, df: pd.DataFrame, label_column: str) -> Dict:
        """
        Sınıf dağılımını detaylı analiz eder
        
        Args:
            df: Analiz edilecek DataFrame
            label_column: Sınıf etiketi kolonu
            
        Returns:
            Dict: Analiz raporu
        """
        if label_column not in df.columns:
            raise ValueError(f"Kolon bulunamadı: {label_column}")
        
        # Sınıf sayıları
        class_counts = df[label_column].value_counts().to_dict()
        total_samples = len(df)
        
        # İstatistikler
        report = {
            'total_samples': total_samples,
            'num_classes': len(class_counts),
            'class_distribution': {},
            'imbalance_metrics': {}
        }
        
        # Her sınıf için detaylı bilgi
        for class_label, count in class_counts.items():
            report['class_distribution'][str(class_label)] = {
                'count': int(count),
                'percentage': float(count / total_samples * 100),
                'ratio': float(count / total_samples)
            }
        
        # Dengesizlik metrikleri
        counts_array = np.array(list(class_counts.values()))
        max_count = counts_array.max()
        min_count = counts_array.min()
        
        report['imbalance_metrics'] = {
            'imbalance_ratio': float(max_count / min_count),
            'majority_class': str(df[label_column].value_counts().index[0]),
            'minority_class': str(df[label_column].value_counts().index[-1]),
            'majority_count': int(max_count),
            'minority_count': int(min_count),
            'is_balanced': bool(max_count / min_count < 1.5),
            'balance_score': float(min_count / max_count * 100)  # 0-100, 100 = perfect balance
        }
        
        self.balance_report = report
        
        if self.verbose:
            self._print_distribution_report(report)
        
        return report
    
    def _print_distribution_report(self, report: Dict):
        """Dağılım raporunu yazdırır"""
        print("\n" + "="*70)
        print("📊 SINIF DAĞILIMI ANALİZ RAPORU")
        print("="*70)
        print(f"Toplam Örnek: {report['total_samples']}")
        print(f"Sınıf Sayısı: {report['num_classes']}")
        print(f"\nDengelilik Skoru: {report['imbalance_metrics']['balance_score']:.1f}%")
        print(f"Dengesizlik Oranı: {report['imbalance_metrics']['imbalance_ratio']:.2f}:1")
        
        # Durum değerlendirmesi
        if report['imbalance_metrics']['is_balanced']:
            print("✅ Veri seti dengeli (< 1.5:1)")
        else:
            print("⚠️  Veri seti dengesiz!")
            if report['imbalance_metrics']['imbalance_ratio'] > 10:
                print("   🔴 ÇOK YÜKSEK dengesizlik! (> 10:1)")
            elif report['imbalance_metrics']['imbalance_ratio'] > 5:
                print("   🟠 YÜKSEK dengesizlik (> 5:1)")
            else:
                print("   🟡 ORTA seviye dengesizlik")
        
        print("\n📋 Sınıf Dağılımı:")
        print("-" * 70)
        for class_label, info in report['class_distribution'].items():
            bar_length = int(info['percentage'] / 2)
            bar = "█" * bar_length
            print(f"  Sınıf {class_label:10s}: {info['count']:6d} örnekler ({info['percentage']:5.2f}%) {bar}")
        
        print("\n📈 Dengesizlik Detayları:")
        print("-" * 70)
        print(f"  Çoğunluk Sınıfı : {report['imbalance_metrics']['majority_class']} ({report['imbalance_metrics']['majority_count']} örnek)")
        print(f"  Azınlık Sınıfı  : {report['imbalance_metrics']['minority_class']} ({report['imbalance_metrics']['minority_count']} örnek)")
        
        print("="*70 + "\n")
    
    def recommend_strategy(self, imbalance_ratio: float) -> List[str]:
        """
        Dengesizlik oranına göre strateji önerileri
        
        Args:
            imbalance_ratio: Dengesizlik oranı
            
        Returns:
            Önerilen stratejiler listesi
        """
        recommendations = []
        
        if imbalance_ratio < 1.5:
            recommendations.append("✅ Veri setiniz zaten dengeli. İşlem gerekmeyebilir.")
        elif imbalance_ratio < 3:
            recommendations.append("🟢 Hafif dengesizlik: Class weights kullanımı önerilir")
            recommendations.append("   Alternatif: Hafif oversampling")
        elif imbalance_ratio < 5:
            recommendations.append("🟡 Orta dengesizlik:")
            recommendations.append("   1. SMOTE (öncelikli)")
            recommendations.append("   2. Random Oversampling")
            recommendations.append("   3. Class weights + augmentation")
        elif imbalance_ratio < 10:
            recommendations.append("🟠 Yüksek dengesizlik:")
            recommendations.append("   1. SMOTE + Tomek Links (kombine)")
            recommendations.append("   2. ADASYN")
            recommendations.append("   3. Oversampling + Augmentation")
        else:
            recommendations.append("🔴 Çok yüksek dengesizlik:")
            recommendations.append("   1. Veri toplama/ekleme önerilir")
            recommendations.append("   2. SMOTE + Heavy Augmentation")
            recommendations.append("   3. Ensemble methods")
            recommendations.append("   4. Focal Loss kullanımı")
        
        return recommendations
    
    def calculate_class_weights(self, df: pd.DataFrame, label_column: str) -> Dict:
        """
        Sınıf ağırlıklarını hesaplar (PyTorch/TensorFlow için)
        
        Args:
            df: DataFrame
            label_column: Sınıf etiketi kolonu
            
        Returns:
            Dict: Sınıf ağırlıkları
        """
        class_counts = df[label_column].value_counts().to_dict()
        total_samples = len(df)
        num_classes = len(class_counts)
        
        # Balanced class weights: n_samples / (n_classes * n_samples_per_class)
        class_weights = {}
        for class_label, count in class_counts.items():
            weight = total_samples / (num_classes * count)
            class_weights[str(class_label)] = float(weight)
        
        if self.verbose:
            print("\n" + "="*70)
            print("⚖️  SINIF AĞIRLIKLARI")
            print("="*70)
            for class_label, weight in class_weights.items():
                print(f"  Sınıf {class_label}: {weight:.4f}")
            print("="*70 + "\n")
        
        return class_weights
    
    def random_oversample(self, df: pd.DataFrame, label_column: str, 
                         strategy: str = 'auto') -> pd.DataFrame:
        """
        Random oversampling - Azınlık sınıfını rastgele çoğaltır
        
        Args:
            df: DataFrame
            label_column: Sınıf etiketi kolonu
            strategy: 'auto' (majority ile eşitle), 'minority' (2x), veya dict
            
        Returns:
            Dengelenmiş DataFrame
        """
        if not IMBLEARN_AVAILABLE:
            # Basit implementasyon
            return self._simple_oversample(df, label_column)
        
        # Özellikleri ve etiketleri ayır
        X = df.drop(columns=[label_column])
        y = df[label_column]
        
        # Sadece numerik kolonları kullan (SMOTE için)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            # Numerik kolon yoksa basit yöntemi kullan
            return self._simple_oversample(df, label_column)
        
        X_numeric = X[numeric_cols]
        
        # Random oversampling
        ros = RandomOverSampler(sampling_strategy=strategy, random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X_numeric, y)
        
        # DataFrame'e geri dönüştür
        df_resampled = pd.DataFrame(X_resampled, columns=numeric_cols)
        df_resampled[label_column] = y_resampled
        
        # Non-numeric kolonları ekle (ilk değerleri tekrarla)
        for col in non_numeric_cols:
            df_resampled[col] = X[col].iloc[0]
        
        if self.verbose:
            original_dist = df[label_column].value_counts().to_dict()
            new_dist = df_resampled[label_column].value_counts().to_dict()
            print(f"\n✅ Random Oversampling tamamlandı!")
            print(f"   Önceki: {original_dist}")
            print(f"   Sonrası: {new_dist}")
            print(f"   Toplam: {len(df)} → {len(df_resampled)} örnek\n")
        
        return df_resampled
    
    def _simple_oversample(self, df: pd.DataFrame, label_column: str) -> pd.DataFrame:
        """Basit oversampling (imbalanced-learn olmadan)"""
        class_counts = df[label_column].value_counts()
        max_count = class_counts.max()
        
        dfs = []
        for class_label in class_counts.index:
            class_df = df[df[label_column] == class_label]
            count = len(class_df)
            
            if count < max_count:
                # Eksik miktarı tamamla
                n_samples = max_count - count
                sampled = class_df.sample(n=n_samples, replace=True, random_state=42)
                dfs.append(pd.concat([class_df, sampled], ignore_index=True))
            else:
                dfs.append(class_df)
        
        df_balanced = pd.concat(dfs, ignore_index=True)
        
        if self.verbose:
            print(f"\n✅ Simple Oversampling tamamlandı!")
            print(f"   Toplam: {len(df)} → {len(df_balanced)} örnek\n")
        
        return df_balanced
    
    def random_undersample(self, df: pd.DataFrame, label_column: str,
                          strategy: str = 'auto') -> pd.DataFrame:
        """
        Random undersampling - Çoğunluk sınıfını azaltır
        
        Args:
            df: DataFrame
            label_column: Sınıf etiketi kolonu
            strategy: 'auto' (minority ile eşitle) veya dict
            
        Returns:
            Dengelenmiş DataFrame
        """
        class_counts = df[label_column].value_counts()
        min_count = class_counts.min()
        
        dfs = []
        for class_label in class_counts.index:
            class_df = df[df[label_column] == class_label]
            
            if len(class_df) > min_count:
                # Minority boyutuna indir
                sampled = class_df.sample(n=min_count, random_state=42)
                dfs.append(sampled)
            else:
                dfs.append(class_df)
        
        df_balanced = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        
        if self.verbose:
            print(f"\n✅ Random Undersampling tamamlandı!")
            print(f"   Toplam: {len(df)} → {len(df_balanced)} örnek")
            print(f"   ⚠️  {len(df) - len(df_balanced)} örnek silindi\n")
        
        return df_balanced
    
    def smote_balance(self, df: pd.DataFrame, label_column: str,
                     k_neighbors: int = 5) -> pd.DataFrame:
        """
        SMOTE (Synthetic Minority Over-sampling Technique)
        Yapay örnekler üreterek azınlık sınıfını dengeler
        
        Args:
            df: DataFrame
            label_column: Sınıf etiketi kolonu
            k_neighbors: SMOTE k_neighbors parametresi
            
        Returns:
            Dengelenmiş DataFrame
        """
        if not IMBLEARN_AVAILABLE:
            print("⚠️  SMOTE için imbalanced-learn gerekli: pip install imbalanced-learn")
            return self._simple_oversample(df, label_column)
        
        # Özellikleri ve etiketleri ayır
        X = df.drop(columns=[label_column])
        y = df[label_column]
        
        # Sadece numerik kolonları kullan
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            print("⚠️  SMOTE için en az 2 numerik özellik gerekli. Basit oversampling kullanılıyor.")
            return self._simple_oversample(df, label_column)
        
        X_numeric = X[numeric_cols]
        
        # Minority class boyutunu kontrol et
        min_class_count = y.value_counts().min()
        k_neighbors = min(k_neighbors, min_class_count - 1)
        
        try:
            # SMOTE uygula
            smote = SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_numeric, y)
            
            # DataFrame'e geri dönüştür
            df_resampled = pd.DataFrame(X_resampled, columns=numeric_cols)
            df_resampled[label_column] = y_resampled
            
            # Non-numeric kolonları ekle
            for col in non_numeric_cols:
                df_resampled[col] = X[col].iloc[0]
            
            if self.verbose:
                print(f"\n✅ SMOTE tamamlandı!")
                print(f"   Toplam: {len(df)} → {len(df_resampled)} örnek")
                print(f"   {len(df_resampled) - len(df)} yapay örnek üretildi\n")
            
            return df_resampled
            
        except Exception as e:
            print(f"⚠️  SMOTE hatası: {e}")
            print("   Basit oversampling kullanılıyor...")
            return self._simple_oversample(df, label_column)
    
    def adasyn_balance(self, df: pd.DataFrame, label_column: str) -> pd.DataFrame:
        """
        ADASYN (Adaptive Synthetic Sampling)
        SMOTE'un geliştirilmiş versiyonu
        
        Args:
            df: DataFrame
            label_column: Sınıf etiketi kolonu
            
        Returns:
            Dengelenmiş DataFrame
        """
        if not IMBLEARN_AVAILABLE:
            return self._simple_oversample(df, label_column)
        
        X = df.drop(columns=[label_column])
        y = df[label_column]
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return self._simple_oversample(df, label_column)
        
        X_numeric = X[numeric_cols]
        
        try:
            adasyn = ADASYN(sampling_strategy='auto', random_state=42)
            X_resampled, y_resampled = adasyn.fit_resample(X_numeric, y)
            
            df_resampled = pd.DataFrame(X_resampled, columns=numeric_cols)
            df_resampled[label_column] = y_resampled
            
            if self.verbose:
                print(f"\n✅ ADASYN tamamlandı!")
                print(f"   Toplam: {len(df)} → {len(df_resampled)} örnek\n")
            
            return df_resampled
            
        except Exception as e:
            print(f"⚠️  ADASYN hatası: {e}. SMOTE kullanılıyor...")
            return self.smote_balance(df, label_column)
    
    def combined_sampling(self, df: pd.DataFrame, label_column: str,
                         method: str = 'smote_tomek') -> pd.DataFrame:
        """
        Kombine sampling yöntemleri
        
        Args:
            df: DataFrame
            label_column: Sınıf etiketi kolonu
            method: 'smote_tomek' veya 'smote_enn'
            
        Returns:
            Dengelenmiş DataFrame
        """
        if not IMBLEARN_AVAILABLE:
            return self.smote_balance(df, label_column)
        
        X = df.drop(columns=[label_column])
        y = df[label_column]
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return self._simple_oversample(df, label_column)
        
        X_numeric = X[numeric_cols]
        
        try:
            if method == 'smote_tomek':
                sampler = SMOTETomek(random_state=42)
            else:  # smote_enn
                sampler = SMOTEENN(random_state=42)
            
            X_resampled, y_resampled = sampler.fit_resample(X_numeric, y)
            
            df_resampled = pd.DataFrame(X_resampled, columns=numeric_cols)
            df_resampled[label_column] = y_resampled
            
            if self.verbose:
                print(f"\n✅ {method.upper()} tamamlandı!")
                print(f"   Toplam: {len(df)} → {len(df_resampled)} örnek\n")
            
            return df_resampled
            
        except Exception as e:
            print(f"⚠️  {method} hatası: {e}. SMOTE kullanılıyor...")
            return self.smote_balance(df, label_column)
    
    def save_report(self, output_path: str):
        """Dengeleme raporunu kaydet"""
        if not self.balance_report:
            print("⚠️  Henüz analiz yapılmadı! Önce analyze_class_distribution() çalıştırın.")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.balance_report, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Dengeleme raporu kaydedildi: {output_path}")
    
    def visualize_distribution(self, df: pd.DataFrame, label_column: str,
                             output_path: Optional[str] = None):
        """
        Sınıf dağılımını görselleştirir
        
        Args:
            df: DataFrame
            label_column: Sınıf etiketi kolonu
            output_path: Grafik kaydetme yolu (opsiyonel)
        """
        try:
            import matplotlib.pyplot as plt
            
            class_counts = df[label_column].value_counts()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Bar chart
            class_counts.plot(kind='bar', ax=ax1, color=['#2ecc71', '#e74c3c'])
            ax1.set_title('Sınıf Dağılımı', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Sınıf', fontsize=12)
            ax1.set_ylabel('Örnek Sayısı', fontsize=12)
            ax1.grid(axis='y', alpha=0.3)
            
            # Pie chart
            class_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%', 
                            colors=['#2ecc71', '#e74c3c'], startangle=90)
            ax2.set_title('Sınıf Oranları', fontsize=14, fontweight='bold')
            ax2.set_ylabel('')
            
            plt.tight_layout()
            
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"✓ Görselleştirme kaydedildi: {output_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            print("⚠️  Görselleştirme için matplotlib gerekli: pip install matplotlib")


def quick_balance_check(csv_path: str, label_column: str = 'ROI_anomaly') -> Dict:
    """
    Hızlı dengesizlik kontrolü
    
    Args:
        csv_path: CSV dosya yolu
        label_column: Sınıf etiketi kolonu
        
    Returns:
        Analiz raporu
    """
    df = pd.read_csv(csv_path)
    balancer = ClassBalancer(verbose=True)
    report = balancer.analyze_class_distribution(df, label_column)
    
    # Öneriler
    print("\n💡 ÖNERİLER:")
    print("-" * 70)
    recommendations = balancer.recommend_strategy(report['imbalance_metrics']['imbalance_ratio'])
    for rec in recommendations:
        print(rec)
    print("-" * 70 + "\n")
    
    return report


if __name__ == "__main__":
    # Test için örnek kullanım
    print("Class Balancer Modülü - Test")
    print("-" * 70)
    
    # Örnek dengesiz veri
    test_data = {
        'feature1': np.random.randn(200),
        'feature2': np.random.randn(200),
        'label': [0] * 150 + [1] * 50  # 3:1 dengesizlik
    }
    df_test = pd.DataFrame(test_data)
    
    balancer = ClassBalancer()
    
    # Analiz
    report = balancer.analyze_class_distribution(df_test, 'label')
    
    # Öneriler
    recommendations = balancer.recommend_strategy(report['imbalance_metrics']['imbalance_ratio'])
    print("\n💡 Öneriler:")
    for rec in recommendations:
        print(rec)
    
    # Class weights
    weights = balancer.calculate_class_weights(df_test, 'label')
