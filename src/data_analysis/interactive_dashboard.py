"""
İnteraktif Veri Analiz Dashboard'u
Streamlit tabanlı (opsiyonel)
"""

# NOT: Bu dosya opsiyoneldir. Streamlit kurulu değilse çalışmayacaktır.
# Kurulum: pip install streamlit

try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    import sys
    
    # Parent directory ekle
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.data_analysis.detailed_analysis import DetailedAnalyzer
    
    st.set_page_config(page_title="NeAR Dataset Analizi", layout="wide", page_icon="🔬")
    
    # Başlık
    st.title("🔬 NeAR Dataset - İnteraktif Veri Analizi")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("⚙️ Ayarlar")
    dataset_path = st.sidebar.text_input(
        "Dataset Yolu", 
        value="NeAR_dataset/ALAN/info.csv"
    )
    
    # Veri yükle
    @st.cache_data
    def load_data(path):
        try:
            analyzer = DetailedAnalyzer(path)
            return analyzer
        except Exception as e:
            st.error(f"Veri yüklenirken hata: {e}")
            return None
    
    analyzer = load_data(dataset_path)
    
    if analyzer is None:
        st.stop()
    
    # Ana içerik
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Genel Bakış", 
        "🔍 Detaylı İstatistikler",
        "📈 Görselleştirmeler",
        "👤 Hasta Analizi",
        "💾 Rapor"
    ])
    
    # Tab 1: Genel Bakış
    with tab1:
        st.header("📊 Genel Bakış")
        
        stats = analyzer.get_comprehensive_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Toplam ROI", stats['basic']['total_rois'])
        with col2:
            st.metric("Toplam Hasta", stats['basic']['total_patients'])
        with col3:
            st.metric("Anomalili ROI", stats['basic']['anomaly_count'])
        with col4:
            st.metric("Anomali Oranı", f"%{stats['basic']['anomaly_ratio']*100:.2f}")
        
        st.markdown("---")
        
        # Veri önizleme
        st.subheader("Veri Önizlemesi")
        st.dataframe(analyzer.df.head(20))
        
        # Temel bilgiler
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Subset Dağılımı")
            subset_counts = analyzer.df['subset'].value_counts()
            st.bar_chart(subset_counts)
        
        with col2:
            st.subheader("Anomali Dağılımı")
            anomaly_counts = analyzer.df['ROI_anomaly'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(anomaly_counts, labels=['Normal', 'Anomaly'], autopct='%1.1f%%',
                   colors=['#2ecc71', '#e74c3c'])
            st.pyplot(fig)
    
    # Tab 2: Detaylı İstatistikler
    with tab2:
        st.header("🔍 Detaylı İstatistikler")
        
        # Subset analizi
        st.subheader("Subset Bazında Analiz")
        subset_stats = []
        for subset, data in stats['subset'].items():
            subset_stats.append({
                'Subset': subset,
                'Toplam': data['total'],
                'Anomali': data['anomaly'],
                'Anomali Oranı (%)': f"{data['anomaly_ratio']*100:.2f}"
            })
        st.table(pd.DataFrame(subset_stats))
        
        # Laterality analizi
        st.subheader("Laterality Analizi")
        lat_stats = []
        for side, data in stats['laterality'].items():
            side_name = "Sol (L)" if side == "L" else "Sağ (R)"
            lat_stats.append({
                'Taraf': side_name,
                'Toplam': data['total'],
                'Anomali': data['anomaly'],
                'Anomali Oranı (%)': f"{data['anomaly_ratio']*100:.2f}"
            })
        st.table(pd.DataFrame(lat_stats))
        
        # Karşılaştırma testleri
        st.subheader("İstatistiksel Testler")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Subset Karşılaştırması**")
            subset_comp = analyzer.compare_subsets()
            st.write(f"Chi-Square: {subset_comp['chi_square_test']['chi2']:.4f}")
            st.write(f"P-value: {subset_comp['chi_square_test']['p_value']:.4f}")
            if subset_comp['chi_square_test']['significant']:
                st.success("✓ Anlamlı farklılık var")
            else:
                st.info("- Anlamlı farklılık yok")
        
        with col2:
            st.markdown("**Laterality Karşılaştırması**")
            lat_comp = analyzer.compare_laterality()
            st.write(f"Sol: %{lat_comp['anomaly_rates']['left']*100:.2f}")
            st.write(f"Sağ: %{lat_comp['anomaly_rates']['right']*100:.2f}")
            st.write(f"Fark: %{lat_comp['anomaly_rates']['difference']*100:.2f}")
    
    # Tab 3: Görselleştirmeler
    with tab3:
        st.header("📈 Görselleştirmeler")
        
        vis_type = st.selectbox(
            "Görselleştirme Tipi",
            ["Anomali Dağılımı", "Subset Analizi", "Laterality Analizi", "Heatmap"]
        )
        
        if vis_type == "Anomali Dağılımı":
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Pie chart
            anomaly_counts = analyzer.df['ROI_anomaly'].value_counts()
            axes[0].pie(anomaly_counts, labels=['Normal', 'Anomaly'], autopct='%1.1f%%',
                       colors=['#2ecc71', '#e74c3c'])
            axes[0].set_title('Genel Dağılım')
            
            # Bar chart
            anomaly_counts.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'])
            axes[1].set_title('Sayılar')
            axes[1].set_xticklabels(['Normal', 'Anomaly'], rotation=0)
            
            st.pyplot(fig)
        
        elif vis_type == "Subset Analizi":
            subset_data = analyzer.df.groupby(['subset', 'ROI_anomaly']).size().unstack(fill_value=0)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            subset_data.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], stacked=True)
            ax.set_title('Subset Bazında Anomali Dağılımı')
            ax.set_xlabel('Subset')
            ax.set_ylabel('ROI Sayısı')
            ax.legend(['Normal', 'Anomaly'])
            plt.xticks(rotation=45)
            
            st.pyplot(fig)
        
        elif vis_type == "Laterality Analizi":
            lat_data = analyzer.df.groupby(['laterality', 'ROI_anomaly']).size().unstack(fill_value=0)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            lat_data.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
            ax.set_title('Sol vs Sağ Böbrek Anomali Dağılımı')
            ax.set_xlabel('Taraf')
            ax.set_ylabel('ROI Sayısı')
            ax.legend(['Normal', 'Anomaly'])
            plt.xticks(rotation=0)
            
            st.pyplot(fig)
        
        elif vis_type == "Heatmap":
            heatmap_data = analyzer.df.groupby(['subset', 'laterality'])['ROI_anomaly'].mean() * 100
            heatmap_pivot = heatmap_data.unstack()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(heatmap_pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                       cbar_kws={'label': 'Anomali Oranı (%)'})
            ax.set_title('Subset x Laterality Anomali Yoğunluğu')
            
            st.pyplot(fig)
    
    # Tab 4: Hasta Analizi
    with tab4:
        st.header("👤 Hasta Seviyesi Analiz")
        
        # Pattern analizi
        pattern_df = analyzer.analyze_patient_patterns()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            both_normal = len(pattern_df[pattern_df['pattern'] == 'Both Normal'])
            st.metric("Her İki Böbrek Normal", both_normal)
        
        with col2:
            one_anomaly = len(pattern_df[pattern_df['pattern'].isin(['Left Only', 'Right Only'])])
            st.metric("Tek Taraf Anomali", one_anomaly)
        
        with col3:
            both_anomaly = len(pattern_df[pattern_df['pattern'] == 'Both Anomaly'])
            st.metric("Her İki Böbrek Anomali", both_anomaly)
        
        # Pattern dağılımı
        st.subheader("Hasta Pattern Dağılımı")
        pattern_counts = pattern_df['pattern'].value_counts()
        st.bar_chart(pattern_counts)
        
        # İlginç hastalar
        st.subheader("İlginç Hasta Profilleri")
        interesting = analyzer.find_interesting_patients()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Her İki Böbrek Anomalili Hastalar:**")
            st.write(f"Toplam: {len(interesting['both_anomaly'])} hasta")
            if st.checkbox("Hastaları göster (Her iki anomali)"):
                st.write(interesting['both_anomaly'][:20])
        
        with col2:
            st.markdown("**Tek Taraf Anomalili Hastalar:**")
            st.write(f"Sadece Sol: {len(interesting['left_only'])} hasta")
            st.write(f"Sadece Sağ: {len(interesting['right_only'])} hasta")
    
    # Tab 5: Rapor
    with tab5:
        st.header("💾 Detaylı Rapor")
        
        if st.button("Rapor Oluştur"):
            with st.spinner("Rapor oluşturuluyor..."):
                report = analyzer.generate_report()
                st.text_area("Rapor", report, height=600)
                
                # İndirme butonu
                st.download_button(
                    label="📥 Raporu İndir",
                    data=report,
                    file_name="near_dataset_analysis_report.txt",
                    mime="text/plain"
                )

except ImportError:
    print("⚠️  Streamlit kurulu değil!")
    print("Kurulum için: pip install streamlit")
    print("Çalıştırma: streamlit run src/data_analysis/interactive_dashboard.py")
