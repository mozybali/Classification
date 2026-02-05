"""
Detaylı Veri Analizi Modülü
NeAR Dataset için ileri seviye analiz araçları
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from scipy import stats
from collections import Counter


class DetailedAnalyzer:
    """Detaylı veri analizi sınıfı"""
    
    def __init__(self, csv_path: str):
        """
        Args:
            csv_path: info.csv dosyasının yolu
        """
        self.csv_path = Path(csv_path)
        self.df = None
        self.load_and_preprocess()
    
    def load_and_preprocess(self):
        """Veriyi yükle ve ön işle"""
        self.df = pd.read_csv(self.csv_path)
        
        # Ek kolonlar ekle
        self.df['laterality'] = self.df['ROI_id'].str.extract(r'_([LR])$')[0]
        self.df['patient_id'] = self.df['ROI_id'].str.extract(r'(ZS\d+)')[0]
        self.df['patient_number'] = self.df['patient_id'].str.extract(r'ZS(\d+)')[0].astype(int)
        
        print(f"✓ Veri yüklendi ve işlendi: {len(self.df)} ROI, {self.df['patient_id'].nunique()} hasta")
    
    def get_comprehensive_statistics(self) -> Dict:
        """Kapsamlı istatistikler"""
        stats_dict = {
            'basic': {
                'total_rois': len(self.df),
                'total_patients': self.df['patient_id'].nunique(),
                'anomaly_count': self.df['ROI_anomaly'].sum(),
                'normal_count': (~self.df['ROI_anomaly']).sum(),
                'anomaly_ratio': self.df['ROI_anomaly'].mean(),
            },
            'subset': {},
            'laterality': {},
            'patient_level': {}
        }
        
        # Subset istatistikleri
        for subset in self.df['subset'].unique():
            subset_df = self.df[self.df['subset'] == subset]
            stats_dict['subset'][subset] = {
                'total': len(subset_df),
                'anomaly': subset_df['ROI_anomaly'].sum(),
                'anomaly_ratio': subset_df['ROI_anomaly'].mean()
            }
        
        # Laterality istatistikleri
        for side in ['L', 'R']:
            side_df = self.df[self.df['laterality'] == side]
            stats_dict['laterality'][side] = {
                'total': len(side_df),
                'anomaly': side_df['ROI_anomaly'].sum(),
                'anomaly_ratio': side_df['ROI_anomaly'].mean()
            }
        
        # Hasta seviyesi istatistikleri
        patient_anomaly_count = self.df.groupby('patient_id')['ROI_anomaly'].sum()
        stats_dict['patient_level'] = {
            'both_normal': (patient_anomaly_count == 0).sum(),
            'one_anomaly': (patient_anomaly_count == 1).sum(),
            'both_anomaly': (patient_anomaly_count == 2).sum()
        }
        
        return stats_dict
    
    def analyze_patient_patterns(self) -> pd.DataFrame:
        """Hasta anomali pattern analizi"""
        # Her hasta için sol ve sağ durumu
        patient_patterns = []
        
        for patient_id in self.df['patient_id'].unique():
            patient_df = self.df[self.df['patient_id'] == patient_id]
            
            if len(patient_df) == 2:
                left_anomaly = patient_df[patient_df['laterality'] == 'L']['ROI_anomaly'].iloc[0]
                right_anomaly = patient_df[patient_df['laterality'] == 'R']['ROI_anomaly'].iloc[0]
                subset = patient_df['subset'].iloc[0]
                
                # Pattern belirle
                if not left_anomaly and not right_anomaly:
                    pattern = 'Both Normal'
                elif left_anomaly and right_anomaly:
                    pattern = 'Both Anomaly'
                elif left_anomaly:
                    pattern = 'Left Only'
                else:
                    pattern = 'Right Only'
                
                patient_patterns.append({
                    'patient_id': patient_id,
                    'pattern': pattern,
                    'subset': subset,
                    'left_anomaly': left_anomaly,
                    'right_anomaly': right_anomaly
                })
        
        pattern_df = pd.DataFrame(patient_patterns)
        return pattern_df
    
    def compare_subsets(self) -> Dict:
        """Subset'ler arasında karşılaştırma"""
        comparison = {}
        
        # Chi-square test for independence
        contingency_table = pd.crosstab(self.df['subset'], self.df['ROI_anomaly'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        comparison['chi_square_test'] = {
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # Subset arası anomali oranı karşılaştırması
        subset_anomaly_rates = self.df.groupby('subset')['ROI_anomaly'].mean()
        comparison['anomaly_rates'] = subset_anomaly_rates.to_dict()
        
        return comparison
    
    def compare_laterality(self) -> Dict:
        """Sol ve sağ böbrek karşılaştırması"""
        comparison = {}
        
        # Chi-square test
        contingency_table = pd.crosstab(self.df['laterality'], self.df['ROI_anomaly'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        comparison['chi_square_test'] = {
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # Anomali oranı karşılaştırması
        left_rate = self.df[self.df['laterality'] == 'L']['ROI_anomaly'].mean()
        right_rate = self.df[self.df['laterality'] == 'R']['ROI_anomaly'].mean()
        
        comparison['anomaly_rates'] = {
            'left': left_rate,
            'right': right_rate,
            'difference': abs(left_rate - right_rate)
        }
        
        return comparison
    
    def get_class_weights(self, method: str = 'balanced') -> Dict:
        """Class weights hesaplama"""
        total = len(self.df)
        normal_count = (~self.df['ROI_anomaly']).sum()
        anomaly_count = self.df['ROI_anomaly'].sum()
        
        if method == 'balanced':
            weight_normal = total / (2 * normal_count)
            weight_anomaly = total / (2 * anomaly_count)
        elif method == 'inverse_freq':
            weight_normal = 1.0
            weight_anomaly = normal_count / anomaly_count
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'normal': weight_normal,
            'anomaly': weight_anomaly,
            'ratio': weight_anomaly / weight_normal
        }
    
    def analyze_anomaly_distribution(self) -> Dict:
        """Anomali dağılımı detaylı analizi"""
        analysis = {}
        
        # Genel dağılım
        analysis['overall'] = {
            'count': self.df['ROI_anomaly'].sum(),
            'percentage': self.df['ROI_anomaly'].mean() * 100
        }
        
        # Subset bazında
        analysis['by_subset'] = {}
        for subset in self.df['subset'].unique():
            subset_df = self.df[self.df['subset'] == subset]
            analysis['by_subset'][subset] = {
                'count': subset_df['ROI_anomaly'].sum(),
                'percentage': subset_df['ROI_anomaly'].mean() * 100,
                'total': len(subset_df)
            }
        
        # Laterality bazında
        analysis['by_laterality'] = {}
        for side in ['L', 'R']:
            side_df = self.df[self.df['laterality'] == side]
            analysis['by_laterality'][side] = {
                'count': side_df['ROI_anomaly'].sum(),
                'percentage': side_df['ROI_anomaly'].mean() * 100,
                'total': len(side_df)
            }
        
        # Subset x Laterality
        analysis['by_subset_laterality'] = {}
        for subset in self.df['subset'].unique():
            analysis['by_subset_laterality'][subset] = {}
            for side in ['L', 'R']:
                filtered = self.df[(self.df['subset'] == subset) & (self.df['laterality'] == side)]
                if len(filtered) > 0:
                    analysis['by_subset_laterality'][subset][side] = {
                        'count': filtered['ROI_anomaly'].sum(),
                        'percentage': filtered['ROI_anomaly'].mean() * 100,
                        'total': len(filtered)
                    }
        
        return analysis
    
    def find_interesting_patients(self) -> Dict:
        """İlginç hasta profilleri bul"""
        interesting = {
            'both_anomaly': [],
            'left_only': [],
            'right_only': [],
            'high_patient_numbers_with_anomaly': []
        }
        
        pattern_df = self.analyze_patient_patterns()
        
        # Her iki böbrek anomalili
        both_anomaly = pattern_df[pattern_df['pattern'] == 'Both Anomaly']
        interesting['both_anomaly'] = both_anomaly['patient_id'].tolist()
        
        # Sadece sol
        left_only = pattern_df[pattern_df['pattern'] == 'Left Only']
        interesting['left_only'] = left_only['patient_id'].tolist()
        
        # Sadece sağ
        right_only = pattern_df[pattern_df['pattern'] == 'Right Only']
        interesting['right_only'] = right_only['patient_id'].tolist()
        
        # Yüksek hasta numaralı anomaliler (son hastalar)
        high_number_patients = self.df[
            (self.df['patient_number'] > 700) & 
            (self.df['ROI_anomaly'] == True)
        ]['patient_id'].unique()
        interesting['high_patient_numbers_with_anomaly'] = high_number_patients.tolist()
        
        return interesting
    
    def generate_report(self, save_path: str = None) -> str:
        """Detaylı rapor oluştur"""
        report_lines = []
        
        report_lines.append("="*80)
        report_lines.append("NeAR DATASET - DETAYLI ANALİZ RAPORU")
        report_lines.append("="*80)
        
        # Temel istatistikler
        stats = self.get_comprehensive_statistics()
        report_lines.append("\n📊 TEMEL İSTATİSTİKLER:")
        report_lines.append(f"  • Toplam ROI: {stats['basic']['total_rois']}")
        report_lines.append(f"  • Toplam Hasta: {stats['basic']['total_patients']}")
        report_lines.append(f"  • Anomalili ROI: {stats['basic']['anomaly_count']} (%{stats['basic']['anomaly_ratio']*100:.2f})")
        report_lines.append(f"  • Normal ROI: {stats['basic']['normal_count']} (%{(1-stats['basic']['anomaly_ratio'])*100:.2f})")
        
        # Subset analizi
        report_lines.append("\n📈 SUBSET ANALİZİ:")
        for subset, data in stats['subset'].items():
            report_lines.append(f"  {subset}:")
            report_lines.append(f"    - Toplam: {data['total']} ROI")
            report_lines.append(f"    - Anomali: {data['anomaly']} (%{data['anomaly_ratio']*100:.2f})")
        
        # Laterality analizi
        report_lines.append("\n🔄 LATERALITY ANALİZİ:")
        for side, data in stats['laterality'].items():
            side_name = "Sol (L)" if side == "L" else "Sağ (R)"
            report_lines.append(f"  {side_name}:")
            report_lines.append(f"    - Toplam: {data['total']} ROI")
            report_lines.append(f"    - Anomali: {data['anomaly']} (%{data['anomaly_ratio']*100:.2f})")
        
        # Hasta seviyesi
        report_lines.append("\n👤 HASTA SEVİYESİ ANALİZ:")
        report_lines.append(f"  • Her iki böbrek normal: {stats['patient_level']['both_normal']} hasta")
        report_lines.append(f"  • Tek böbrek anomalili: {stats['patient_level']['one_anomaly']} hasta")
        report_lines.append(f"  • Her iki böbrek anomalili: {stats['patient_level']['both_anomaly']} hasta")
        
        # Subset karşılaştırması
        subset_comp = self.compare_subsets()
        report_lines.append("\n📊 SUBSET KARŞILAŞTIRMASI:")
        report_lines.append(f"  Chi-Square Test:")
        report_lines.append(f"    - Chi2: {subset_comp['chi_square_test']['chi2']:.4f}")
        report_lines.append(f"    - P-value: {subset_comp['chi_square_test']['p_value']:.4f}")
        report_lines.append(f"    - Anlamlı farklılık: {'Evet' if subset_comp['chi_square_test']['significant'] else 'Hayır'}")
        
        # Laterality karşılaştırması
        lat_comp = self.compare_laterality()
        report_lines.append("\n🔄 LATERALITY KARŞILAŞTIRMASI:")
        report_lines.append(f"  Sol anomali oranı: %{lat_comp['anomaly_rates']['left']*100:.2f}")
        report_lines.append(f"  Sağ anomali oranı: %{lat_comp['anomaly_rates']['right']*100:.2f}")
        report_lines.append(f"  Fark: %{lat_comp['anomaly_rates']['difference']*100:.2f}")
        report_lines.append(f"  Chi-Square P-value: {lat_comp['chi_square_test']['p_value']:.4f}")
        
        # Class weights
        weights = self.get_class_weights('balanced')
        report_lines.append("\n⚖️ CLASS WEIGHTS (Balanced):")
        report_lines.append(f"  • Normal: {weights['normal']:.4f}")
        report_lines.append(f"  • Anomaly: {weights['anomaly']:.4f}")
        report_lines.append(f"  • Ratio: 1:{weights['ratio']:.2f}")
        
        # İlginç hastalar
        interesting = self.find_interesting_patients()
        report_lines.append("\n🔍 İLGİNÇ HASTA PROFİLLERİ:")
        report_lines.append(f"  • Her iki böbrek anomalili: {len(interesting['both_anomaly'])} hasta")
        report_lines.append(f"  • Sadece sol anomalili: {len(interesting['left_only'])} hasta")
        report_lines.append(f"  • Sadece sağ anomalili: {len(interesting['right_only'])} hasta")
        
        report_lines.append("\n" + "="*80)
        
        report = "\n".join(report_lines)
        
        # Kaydet
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✓ Rapor kaydedildi: {save_path}")
        
        return report
    
    def plot_comprehensive_analysis(self, save_dir: str = None):
        """Kapsamlı görselleştirme"""
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Genel overview
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1.1. Genel anomali dağılımı
        ax1 = fig.add_subplot(gs[0, 0])
        anomaly_counts = self.df['ROI_anomaly'].value_counts()
        ax1.pie(anomaly_counts, labels=['Normal', 'Anomaly'], autopct='%1.1f%%',
               colors=['#2ecc71', '#e74c3c'], startangle=90)
        ax1.set_title('Genel Anomali Dağılımı', fontweight='bold')
        
        # 1.2. Subset dağılımı
        ax2 = fig.add_subplot(gs[0, 1])
        subset_data = self.df.groupby(['subset', 'ROI_anomaly']).size().unstack(fill_value=0)
        subset_data.plot(kind='bar', ax=ax2, color=['#2ecc71', '#e74c3c'], stacked=True)
        ax2.set_title('Subset Dağılımı', fontweight='bold')
        ax2.set_xlabel('Subset')
        ax2.set_ylabel('Sayı')
        ax2.legend(['Normal', 'Anomaly'])
        ax2.tick_params(axis='x', rotation=45)
        
        # 1.3. Laterality dağılımı
        ax3 = fig.add_subplot(gs[0, 2])
        lat_data = self.df.groupby(['laterality', 'ROI_anomaly']).size().unstack(fill_value=0)
        lat_data.plot(kind='bar', ax=ax3, color=['#2ecc71', '#e74c3c'])
        ax3.set_title('Sol vs Sağ Böbrek', fontweight='bold')
        ax3.set_xlabel('Taraf')
        ax3.set_ylabel('Sayı')
        ax3.legend(['Normal', 'Anomaly'])
        ax3.tick_params(axis='x', rotation=0)
        
        # 1.4. Hasta seviyesi analiz
        ax4 = fig.add_subplot(gs[1, 0])
        patient_anomaly_count = self.df.groupby('patient_id')['ROI_anomaly'].sum()
        both_normal = (patient_anomaly_count == 0).sum()
        one_anomaly = (patient_anomaly_count == 1).sum()
        both_anomaly = (patient_anomaly_count == 2).sum()
        
        categories = ['Her İki\nNormal', 'Tek\nAnomali', 'Her İki\nAnomali']
        values = [both_normal, one_anomaly, both_anomaly]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        ax4.bar(categories, values, color=colors)
        ax4.set_title('Hasta Seviyesi Analiz', fontweight='bold')
        ax4.set_ylabel('Hasta Sayısı')
        for i, v in enumerate(values):
            ax4.text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')
        
        # 1.5. Heatmap - Subset x Laterality
        ax5 = fig.add_subplot(gs[1, 1])
        heatmap_data = self.df.groupby(['subset', 'laterality'])['ROI_anomaly'].mean() * 100
        heatmap_pivot = heatmap_data.unstack()
        sns.heatmap(heatmap_pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax5,
                   cbar_kws={'label': 'Anomali %'})
        ax5.set_title('Anomali Yoğunluğu Haritası', fontweight='bold')
        
        # 1.6. Pattern analizi
        ax6 = fig.add_subplot(gs[1, 2])
        pattern_df = self.analyze_patient_patterns()
        pattern_counts = pattern_df['pattern'].value_counts()
        pattern_counts.plot(kind='barh', ax=ax6, color=['#2ecc71', '#e74c3c', '#3498db', '#9b59b6'])
        ax6.set_title('Hasta Pattern Dağılımı', fontweight='bold')
        ax6.set_xlabel('Hasta Sayısı')
        
        # 1.7. Anomali oranı trend (hasta numarasına göre)
        ax7 = fig.add_subplot(gs[2, :])
        patient_bins = pd.cut(self.df['patient_number'], bins=20)
        trend_data = self.df.groupby(patient_bins)['ROI_anomaly'].mean() * 100
        trend_data.plot(kind='line', ax=ax7, marker='o', color='#e74c3c', linewidth=2)
        ax7.set_title('Anomali Oranı Trendi (Hasta Numarasına Göre)', fontweight='bold')
        ax7.set_xlabel('Hasta Numarası Aralığı')
        ax7.set_ylabel('Anomali Oranı (%)')
        ax7.grid(True, alpha=0.3)
        ax7.axhline(y=self.df['ROI_anomaly'].mean()*100, color='blue', linestyle='--',
                   label=f'Genel Ortalama: {self.df["ROI_anomaly"].mean()*100:.1f}%')
        ax7.legend()
        
        plt.suptitle('NeAR Dataset - Kapsamlı Analiz', fontsize=16, fontweight='bold', y=0.995)
        
        if save_dir:
            plt.savefig(save_dir / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
            print(f"✓ Görsel kaydedildi: {save_dir / 'comprehensive_analysis.png'}")
        
        plt.show()


def main():
    """Ana çalıştırma fonksiyonu"""
    # Analyzer oluştur
    analyzer = DetailedAnalyzer('NeAR_dataset/ALAN/info.csv')
    
    # Rapor oluştur
    report = analyzer.generate_report(save_path='outputs/detailed_analysis_report.txt')
    print(report)
    
    # Görselleştirme
    analyzer.plot_comprehensive_analysis(save_dir='outputs/plots')
    
    # Ek analizler
    print("\n" + "="*80)
    print("ANOMALI DAĞILIMI DETAYLI ANALİZ")
    print("="*80)
    
    dist_analysis = analyzer.analyze_anomaly_distribution()
    print(f"\nGenel: {dist_analysis['overall']['count']} anomali (%{dist_analysis['overall']['percentage']:.2f})")
    
    print("\nSubset Bazında:")
    for subset, data in dist_analysis['by_subset'].items():
        print(f"  {subset}: {data['count']}/{data['total']} (%{data['percentage']:.2f})")
    
    print("\nLaterality Bazında:")
    for side, data in dist_analysis['by_laterality'].items():
        side_name = "Sol" if side == "L" else "Sağ"
        print(f"  {side_name}: {data['count']}/{data['total']} (%{data['percentage']:.2f})")


if __name__ == "__main__":
    main()
