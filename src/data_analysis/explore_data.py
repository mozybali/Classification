"""
Veri Seti Analiz Modülü
NeAR Dataset ALAN veri setini inceler ve istatistikler çıkarır.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple


class DatasetExplorer:
    """ALAN dataset'ini analiz eden sınıf"""
    
    def __init__(self, dataset_path: str):
        """
        Args:
            dataset_path: NeAR_dataset/ALAN/info.csv dosyasının yolu
        """
        self.dataset_path = Path(dataset_path)
        self.df = None
        self.load_data()
    
    def load_data(self):
        """CSV dosyasını yükler"""
        self.df = pd.read_csv(self.dataset_path)
        print(f"✓ Veri seti yüklendi: {len(self.df)} ROI")
    
    def get_basic_statistics(self) -> Dict:
        """Temel istatistikleri döndürür"""
        stats = {
            'total_rois': len(self.df),
            'total_patients': len(self.df['ROI_id'].str.split('_').str[0].unique()),
            'anomaly_count': self.df['ROI_anomaly'].sum(),
            'normal_count': len(self.df) - self.df['ROI_anomaly'].sum(),
            'anomaly_ratio': self.df['ROI_anomaly'].sum() / len(self.df)
        }
        return stats
    
    def get_subset_distribution(self) -> pd.DataFrame:
        """Train/Test/Dev dağılımını döndürür"""
        subset_dist = self.df.groupby(['subset', 'ROI_anomaly']).size().unstack(fill_value=0)
        subset_dist.columns = ['Normal', 'Anomaly']
        subset_dist['Total'] = subset_dist.sum(axis=1)
        subset_dist['Anomaly_Ratio'] = subset_dist['Anomaly'] / subset_dist['Total']
        return subset_dist
    
    def get_laterality_analysis(self) -> Dict:
        """Sol/Sağ böbrek anomali analizi"""
        self.df['laterality'] = self.df['ROI_id'].str.extract(r'_([LR])$')[0]
        
        laterality_stats = {}
        for side in ['L', 'R']:
            side_df = self.df[self.df['laterality'] == side]
            laterality_stats[side] = {
                'total': len(side_df),
                'anomaly': side_df['ROI_anomaly'].sum(),
                'normal': len(side_df) - side_df['ROI_anomaly'].sum(),
                'anomaly_ratio': side_df['ROI_anomaly'].sum() / len(side_df)
            }
        
        return laterality_stats
    
    def get_patient_level_analysis(self) -> Dict:
        """Hasta seviyesinde anomali analizi"""
        self.df['patient_id'] = self.df['ROI_id'].str.extract(r'(ZS\d+)')[0]
        
        # Her hasta için anomali sayısı
        patient_anomalies = self.df.groupby('patient_id')['ROI_anomaly'].sum()
        
        patient_stats = {
            'both_normal': (patient_anomalies == 0).sum(),
            'one_anomaly': (patient_anomalies == 1).sum(),
            'both_anomaly': (patient_anomalies == 2).sum()
        }
        
        return patient_stats
    
    def visualize_distribution(self, save_path: str = None):
        """Veri dağılımını görselleştirir"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Anomali dağılımı
        anomaly_counts = self.df['ROI_anomaly'].value_counts()
        axes[0, 0].pie(anomaly_counts, labels=['Normal', 'Anomaly'], autopct='%1.1f%%',
                       colors=['#2ecc71', '#e74c3c'], startangle=90)
        axes[0, 0].set_title('Overall Anomaly Distribution')
        
        # 2. Subset dağılımı
        subset_dist = self.get_subset_distribution()
        subset_dist[['Normal', 'Anomaly']].plot(kind='bar', ax=axes[0, 1], color=['#2ecc71', '#e74c3c'])
        axes[0, 1].set_title('Anomaly Distribution by Subset')
        axes[0, 1].set_xlabel('Subset')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend(title='Class')
        axes[0, 1].tick_params(axis='x', rotation=0)
        
        # 3. Laterality analizi
        if 'laterality' not in self.df.columns:
            self.df['laterality'] = self.df['ROI_id'].str.extract(r'_([LR])$')[0]
        
        lat_data = self.df.groupby(['laterality', 'ROI_anomaly']).size().unstack(fill_value=0)
        lat_data.plot(kind='bar', ax=axes[1, 0], color=['#2ecc71', '#e74c3c'])
        axes[1, 0].set_title('Anomaly Distribution by Laterality')
        axes[1, 0].set_xlabel('Side (L=Left, R=Right)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend(title='Anomaly', labels=['Normal', 'Anomaly'])
        axes[1, 0].tick_params(axis='x', rotation=0)
        
        # 4. Hasta seviyesi analizi
        patient_stats = self.get_patient_level_analysis()
        axes[1, 1].bar(patient_stats.keys(), patient_stats.values(), color=['#3498db', '#f39c12', '#e74c3c'])
        axes[1, 1].set_title('Patient-Level Anomaly Analysis')
        axes[1, 1].set_xlabel('Anomaly Status')
        axes[1, 1].set_ylabel('Patient Count')
        axes[1, 1].tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Görselleştirme kaydedildi: {save_path}")
        
        plt.show()
    
    def print_summary(self):
        """Detaylı özet rapor yazdırır"""
        print("\n" + "="*60)
        print("NeAR DATASET - ALAN VERİ SETİ ANALİZİ")
        print("="*60)
        
        # Temel istatistikler
        stats = self.get_basic_statistics()
        print(f"\n📊 TEMEL İSTATİSTİKLER:")
        print(f"  • Toplam ROI Sayısı: {stats['total_rois']}")
        print(f"  • Toplam Hasta Sayısı: {stats['total_patients']}")
        print(f"  • Anomalili ROI: {stats['anomaly_count']} (%{stats['anomaly_ratio']*100:.2f})")
        print(f"  • Normal ROI: {stats['normal_count']} (%{(1-stats['anomaly_ratio'])*100:.2f})")
        
        # Subset dağılımı
        print(f"\n📈 SUBSET DAĞILIMI:")
        subset_dist = self.get_subset_distribution()
        print(subset_dist.to_string())
        
        # Laterality analizi
        print(f"\n🔄 LATERAL ANALİZ (Sol/Sağ Böbrek):")
        lat_stats = self.get_laterality_analysis()
        for side, stats in lat_stats.items():
            side_name = "Sol" if side == "L" else "Sağ"
            print(f"  {side_name} Böbrek:")
            print(f"    - Toplam: {stats['total']}")
            print(f"    - Anomali: {stats['anomaly']} (%{stats['anomaly_ratio']*100:.2f})")
            print(f"    - Normal: {stats['normal']}")
        
        # Hasta seviyesi analizi
        print(f"\n👤 HASTA SEVİYESİ ANALİZ:")
        patient_stats = self.get_patient_level_analysis()
        print(f"  • Her iki böbrek normal: {patient_stats['both_normal']} hasta")
        print(f"  • Tek böbrek anomalili: {patient_stats['one_anomaly']} hasta")
        print(f"  • Her iki böbrek anomalili: {patient_stats['both_anomaly']} hasta")
        
        print("\n" + "="*60 + "\n")


def main():
    """Ana çalıştırma fonksiyonu"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NeAR Dataset Analizi')
    parser.add_argument('--dataset', type=str, default='NeAR_dataset/ALAN/info.csv',
                       help='Dataset CSV yolu')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Çıktı dizini')
    parser.add_argument('--detailed', action='store_true',
                       help='Detaylı analiz (detailed_analysis.py kullanılır)')
    
    args = parser.parse_args()
    
    if args.detailed:
        print("Detaylı analiz için detailed_analysis.py kullanın veya Jupyter notebook açın:")
        print("  python src/data_analysis/detailed_analysis.py")
        print("  jupyter notebook notebooks/01_data_exploration.ipynb")
        return
    
    # Dataset yolunu belirt
    dataset_path = args.dataset
    
    # Explorer oluştur
    explorer = DatasetExplorer(dataset_path)
    
    # Özet rapor
    explorer.print_summary()
    
    # Görselleştirme
    from pathlib import Path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    explorer.visualize_distribution(save_path=str(output_dir / "data_analysis.png"))
    
    print(f"\n💡 Daha detaylı analiz için:")
    print(f"   python src/data_analysis/detailed_analysis.py")
    print(f"   jupyter notebook notebooks/01_data_exploration.ipynb")


if __name__ == "__main__":
    main()
