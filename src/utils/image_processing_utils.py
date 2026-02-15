"""
Main.py için görüntü işleme fonksiyonları
"""

from pathlib import Path
import sys


def compute_image_statistics(config):
    """Görüntü istatistiklerini hesapla"""
    print("\n" + "="*70)
    print("GÖRÜNTÜ İSTATİSTİKLERİ HESAPLANIYOR")
    print("="*70)
    
    from src.preprocessing.image_loader import ImageLoader, DatasetStatistics, save_statistics_report
    from src.preprocessing.dataloader_factory import resolve_image_source_path
    
    dataset_path = Path(config['dataset']['path'])
    image_source_path = resolve_image_source_path(config['dataset'])
    csv_path = dataset_path / config['dataset']['csv_file']
    
    # Image loader
    print("\n📦 Görüntü yükleyici başlatılıyor...")
    loader = ImageLoader(str(image_source_path), cache_in_memory=False)
    
    # Statistics calculator
    stats_calc = DatasetStatistics(loader, str(csv_path))
    
    # Kullanıcıdan örnekleme boyutu al
    print("\n💡 İpucu: Tüm görüntüler için hesaplama uzun sürebilir.")
    try:
        sample_input = input("Kaç görüntü üzerinden hesaplansın? (boş=tümü, örn: 100): ").strip()
    except EOFError:
        sample_input = "100"
        print("100 (varsayılan)")
    
    sample_size = None
    if sample_input:
        try:
            sample_size = int(sample_input)
        except:
            print("⚠️  Geçersiz sayı, tüm görüntüler kullanılacak")
    
    # Global istatistikler
    print("\n📊 Global istatistikler hesaplanıyor...")
    global_stats = stats_calc.compute_statistics(sample_size=sample_size)
    
    # Sınıf bazında istatistikler
    print("\n📊 Sınıf bazında istatistikler hesaplanıyor...")
    class_stats = stats_calc.compute_class_statistics()
    
    # Sonuçları birleştir
    all_stats = {
        'global': global_stats,
        'by_class': class_stats
    }
    
    # Kaydet
    output_dir = Path(config['logging']['plot_dir']).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    save_statistics_report(all_stats, str(output_dir / 'image_statistics.json'))
    
    print("\n✅ İstatistikler hesaplandı ve kaydedildi!")


def test_image_transforms(config):
    """Run a lightweight transform sanity check."""
    print("\n" + "="*70)
    print("IMAGE TRANSFORM TESTS")
    print("="*70)

    from pathlib import Path
    import numpy as np
    import pandas as pd

    from src.preprocessing.image_loader import ImageLoader
    from src.preprocessing.pipeline_builder import create_preprocessing_pipeline
    from src.preprocessing.dataloader_factory import resolve_image_source_path

    dataset_path = Path(config['dataset']['path'])
    csv_path = dataset_path / config['dataset']['csv_file']
    image_source_path = resolve_image_source_path(config['dataset'])

    if not csv_path.exists() or not image_source_path.exists():
        print("Dataset files not found.")
        print(f"CSV: {csv_path}")
        print(f"Image source: {image_source_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Dataset CSV is empty.")
        return

    # Build pipelines
    print("\nBuilding preprocessing pipelines...")
    pipelines = create_preprocessing_pipeline(config, model_type='classifier', verbose=True)

    # Pick samples
    sample_count = min(3, len(df))
    sample_ids = df['ROI_id'].sample(sample_count, random_state=42).tolist()

    loader = ImageLoader(str(image_source_path), cache_in_memory=False)

    def summarize(name: str, arr: np.ndarray):
        arr = np.asarray(arr)
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")

    print("\nApplying transforms on sample images...")
    for roi_id in sample_ids:
        try:
            image = loader.load_image(roi_id)
            print(f"\nROI: {roi_id}")
            summarize('original', image)

            train_img = pipelines['train'](image)
            summarize('train', train_img)

            val_img = pipelines['val'](image)
            summarize('val', val_img)
        except Exception as e:
            print(f"  Error processing {roi_id}: {e}")

    print("\nTransform test completed.")
