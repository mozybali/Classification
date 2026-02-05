"""
Augmentation Manager - Veri Arttırma Yönetimi
3D Medical Image Augmentation stratejileri ve pipeline kontrolü
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
import json
from collections import defaultdict


class AugmentationManager:
    """Veri arttırma stratejilerini yöneten ve analiz eden sınıf"""
    
    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: Detaylı çıktı göster
        """
        self.verbose = verbose
        self.augmentation_config = {}
        self.applied_augmentations = []
        
    def get_augmentation_strategies(self) -> Dict:
        """
        Kullanılabilir augmentation stratejilerini döndürür
        
        Returns:
            Dict: Strateji bilgileri
        """
        strategies = {
            'geometric': {
                'name': 'Geometrik Transformlar',
                'transforms': {
                    'random_flip': {
                        'name': 'Rastgele Çevirme (Flip)',
                        'params': {'p': 0.5, 'axes': [0, 1, 2]},
                        'description': 'Görüntüyü rastgele eksenlerde çevirir',
                        'recommended_for': 'Tüm durumlar'
                    },
                    'random_rotation': {
                        'name': 'Rastgele Döndürme (Rotation)',
                        'params': {'p': 0.5, 'angle_range': [-15, 15], 'axes': [0, 1]},
                        'description': 'Görüntüyü rastgele açılarda döndürür',
                        'recommended_for': 'Yönelim bağımsız objeler'
                    },
                    'random_shift': {
                        'name': 'Rastgele Kaydırma (Shift)',
                        'params': {'p': 0.5, 'max_shift': 10},
                        'description': 'Görüntüyü rastgele yönlerde kaydırır',
                        'recommended_for': 'Pozisyon varyasyonu'
                    },
                    'random_zoom': {
                        'name': 'Rastgele Yakınlaştırma (Zoom)',
                        'params': {'p': 0.3, 'zoom_range': [0.9, 1.1]},
                        'description': 'Görüntüyü rastgele büyütür/küçültür',
                        'recommended_for': 'Boyut varyasyonu'
                    }
                }
            },
            'intensity': {
                'name': 'Intensity Transformlar',
                'transforms': {
                    'random_noise': {
                        'name': 'Rastgele Gürültü (Noise)',
                        'params': {'p': 0.2, 'noise_std': 0.01},
                        'description': 'Gaussian gürültü ekler',
                        'recommended_for': 'Gürültüye dayanıklılık'
                    },
                    'random_brightness': {
                        'name': 'Parlaklık Değişimi',
                        'params': {'p': 0.3, 'brightness_range': [0.8, 1.2]},
                        'description': 'Parlaklık seviyesini değiştirir',
                        'recommended_for': 'Intensity varyasyonu'
                    },
                    'random_contrast': {
                        'name': 'Kontrast Değişimi',
                        'params': {'p': 0.3, 'contrast_range': [0.8, 1.2]},
                        'description': 'Kontrast seviyesini değiştirir',
                        'recommended_for': 'Kontrast varyasyonu'
                    }
                }
            },
            'deformation': {
                'name': 'Deformasyon Transformlar',
                'transforms': {
                    'elastic_deformation': {
                        'name': 'Elastik Deformasyon',
                        'params': {'p': 0.3, 'alpha': 10, 'sigma': 4},
                        'description': 'Elastik distorsiyon uygular',
                        'recommended_for': 'Organ deformasyonları'
                    }
                }
            },
            'medical': {
                'name': 'Medical-Specific Transformlar',
                'transforms': {
                    'adaptive_crop': {
                        'name': 'Adaptif ROI Crop',
                        'params': {'enabled': True, 'margin': 10},
                        'description': 'ROI etrafında adaptif kırpma',
                        'recommended_for': 'Medical imaging'
                    },
                    'mask_processing': {
                        'name': 'Mask Post-Processing',
                        'params': {'fill_holes': True, 'min_component_size': 100},
                        'description': 'Binary mask temizleme',
                        'recommended_for': 'Segmentation masks'
                    }
                }
            }
        }
        
        return strategies
    
    def recommend_augmentation_level(self, dataset_size: int, imbalance_ratio: float) -> str:
        """
        Dataset boyutu ve dengesizlik oranına göre augmentation seviyesi öner
        
        Args:
            dataset_size: Dataset boyutu
            imbalance_ratio: Dengesizlik oranı
            
        Returns:
            Önerilen seviye ('light', 'normal', 'heavy')
        """
        if dataset_size > 5000:
            if imbalance_ratio < 2:
                return 'light'
            elif imbalance_ratio < 5:
                return 'normal'
            else:
                return 'heavy'
        elif dataset_size > 1000:
            if imbalance_ratio < 2:
                return 'normal'
            else:
                return 'heavy'
        else:
            return 'heavy'
    
    def get_preset_config(self, level: str = 'normal') -> Dict:
        """
        Hazır augmentation konfigürasyonları
        
        Args:
            level: 'light', 'normal', 'heavy'
            
        Returns:
            Augmentation config
        """
        presets = {
            'light': {
                'enabled': True,
                'mode': 'light',
                'random_flip': {'enabled': True, 'p': 0.3, 'axes': [0, 1]},
                'random_rotation': {'enabled': True, 'p': 0.3, 'angle_range': [-10, 10]},
                'random_shift': {'enabled': False},
                'random_zoom': {'enabled': False},
                'elastic_deformation': {'enabled': False},
                'random_noise': {'enabled': True, 'p': 0.1, 'noise_std': 0.005}
            },
            'normal': {
                'enabled': True,
                'mode': 'normal',
                'random_flip': {'enabled': True, 'p': 0.5, 'axes': [0, 1, 2]},
                'random_rotation': {'enabled': True, 'p': 0.5, 'angle_range': [-15, 15]},
                'random_shift': {'enabled': True, 'p': 0.5, 'max_shift': 10},
                'random_zoom': {'enabled': True, 'p': 0.3, 'zoom_range': [0.9, 1.1]},
                'elastic_deformation': {'enabled': True, 'p': 0.3, 'alpha': 10, 'sigma': 4},
                'random_noise': {'enabled': True, 'p': 0.2, 'noise_std': 0.01}
            },
            'heavy': {
                'enabled': True,
                'mode': 'heavy',
                'random_flip': {'enabled': True, 'p': 0.7, 'axes': [0, 1, 2]},
                'random_rotation': {'enabled': True, 'p': 0.7, 'angle_range': [-20, 20]},
                'random_shift': {'enabled': True, 'p': 0.7, 'max_shift': 15},
                'random_zoom': {'enabled': True, 'p': 0.5, 'zoom_range': [0.8, 1.2]},
                'elastic_deformation': {'enabled': True, 'p': 0.5, 'alpha': 15, 'sigma': 5},
                'random_noise': {'enabled': True, 'p': 0.3, 'noise_std': 0.02},
                'random_brightness': {'enabled': True, 'p': 0.3, 'brightness_range': [0.8, 1.2]},
                'random_contrast': {'enabled': True, 'p': 0.3, 'contrast_range': [0.8, 1.2]}
            },
            'medical_kidney': {
                'enabled': True,
                'mode': 'medical_kidney',
                'random_flip': {'enabled': True, 'p': 0.5, 'axes': [0, 1, 2]},
                'random_rotation': {'enabled': True, 'p': 0.4, 'angle_range': [-10, 10]},
                'random_shift': {'enabled': True, 'p': 0.4, 'max_shift': 8},
                'random_zoom': {'enabled': True, 'p': 0.3, 'zoom_range': [0.95, 1.05]},
                'elastic_deformation': {'enabled': True, 'p': 0.4, 'alpha': 8, 'sigma': 4},
                'random_noise': {'enabled': True, 'p': 0.2, 'noise_std': 0.01},
                'adaptive_crop': {'enabled': True, 'margin': 10},
                'mask_processing': {'enabled': True, 'fill_holes': True}
            }
        }
        
        return presets.get(level, presets['normal'])
    
    def create_custom_config(self) -> Dict:
        """İnteraktif olarak özel config oluştur"""
        config = {'enabled': True, 'mode': 'custom'}
        
        strategies = self.get_augmentation_strategies()
        
        for category_key, category_info in strategies.items():
            print(f"\n{'='*70}")
            print(f"📦 {category_info['name']}")
            print('='*70)
            
            for transform_key, transform_info in category_info['transforms'].items():
                print(f"\n🔧 {transform_info['name']}")
                print(f"   📝 {transform_info['description']}")
                print(f"   💡 Önerilir: {transform_info['recommended_for']}")
                
                enable = input(f"   Aktif et? (e/h, varsayılan: e): ").strip().lower()
                
                if enable != 'h':
                    config[transform_key] = {'enabled': True}
                    config[transform_key].update(transform_info['params'])
                else:
                    config[transform_key] = {'enabled': False}
        
        return config
    
    def estimate_augmented_size(self, original_size: int, augmentation_factor: float = 1.0) -> int:
        """
        Augmentation sonrası dataset boyutunu tahmin eder
        
        Args:
            original_size: Orijinal dataset boyutu
            augmentation_factor: Çarpan (1.0 = değişiklik yok, 2.0 = 2x büyüklük)
            
        Returns:
            Tahmini boyut
        """
        return int(original_size * (1 + augmentation_factor))
    
    def calculate_augmentation_factor(self, minority_count: int, majority_count: int,
                                     target_balance: float = 1.0) -> float:
        """
        İstenen dengeye ulaşmak için gereken augmentation faktörünü hesaplar
        
        Args:
            minority_count: Azınlık sınıfı örnek sayısı
            majority_count: Çoğunluk sınıfı örnek sayısı
            target_balance: Hedef denge oranı (1.0 = perfect balance)
            
        Returns:
            Augmentation faktörü
        """
        target_minority = int(majority_count * target_balance)
        needed_samples = max(0, target_minority - minority_count)
        factor = needed_samples / minority_count if minority_count > 0 else 0
        
        if self.verbose:
            print(f"\n📊 AUGMENTATION FAKTÖRÜ HESAPLAMA")
            print(f"   Mevcut azınlık: {minority_count}")
            print(f"   Hedef azınlık: {target_minority}")
            print(f"   Gerekli ek örnek: {needed_samples}")
            print(f"   Augmentation faktörü: {factor:.2f}x\n")
        
        return factor
    
    def analyze_augmentation_config(self, config: Dict) -> Dict:
        """
        Augmentation konfigürasyonunu analiz eder
        
        Args:
            config: Augmentation config
            
        Returns:
            Analiz raporu
        """
        report = {
            'enabled': config.get('enabled', True),
            'mode': config.get('mode', 'unknown'),
            'active_transforms': [],
            'inactive_transforms': [],
            'total_probability': 0.0,
            'aggressiveness_score': 0.0
        }
        
        aggressiveness_weights = {
            'random_flip': 1,
            'random_rotation': 2,
            'random_shift': 2,
            'random_zoom': 3,
            'elastic_deformation': 4,
            'random_noise': 2,
            'random_brightness': 2,
            'random_contrast': 2
        }
        
        total_weight = 0
        
        for key, value in config.items():
            if isinstance(value, dict) and 'enabled' in value:
                if value.get('enabled', False):
                    prob = value.get('p', 0.5)
                    report['active_transforms'].append({
                        'name': key,
                        'probability': prob,
                        'params': value
                    })
                    report['total_probability'] += prob
                    
                    # Aggressiveness hesapla
                    weight = aggressiveness_weights.get(key, 1)
                    total_weight += weight * prob
                else:
                    report['inactive_transforms'].append(key)
        
        # Aggressiveness score (0-100)
        max_possible = sum(aggressiveness_weights.values())
        report['aggressiveness_score'] = (total_weight / max_possible * 100) if max_possible > 0 else 0
        
        if self.verbose:
            self._print_augmentation_report(report)
        
        return report
    
    def _print_augmentation_report(self, report: Dict):
        """Augmentation raporunu yazdır"""
        print("\n" + "="*70)
        print("🎨 AUGMENTATION KONFİGÜRASYONU ANALİZİ")
        print("="*70)
        print(f"Durum: {'✅ Aktif' if report['enabled'] else '❌ Devre dışı'}")
        print(f"Mod: {report['mode']}")
        print(f"Agresiflik Skoru: {report['aggressiveness_score']:.1f}/100")
        
        if report['aggressiveness_score'] < 30:
            print("   🟢 Hafif augmentation (Conservative)")
        elif report['aggressiveness_score'] < 60:
            print("   🟡 Orta augmentation (Balanced)")
        else:
            print("   🔴 Yoğun augmentation (Aggressive)")
        
        print(f"\nAktif Transform Sayısı: {len(report['active_transforms'])}")
        if report['active_transforms']:
            print("\n📋 Aktif Transformlar:")
            for transform in report['active_transforms']:
                print(f"   • {transform['name']:25s} (p={transform['probability']:.2f})")
        
        if report['inactive_transforms']:
            print(f"\n❌ Devre Dışı Transformlar: {', '.join(report['inactive_transforms'])}")
        
        print("="*70 + "\n")
    
    def get_recommendations(self, dataset_size: int, imbalance_ratio: float,
                           minority_count: int) -> Dict:
        """
        Kapsamlı augmentation önerileri
        
        Args:
            dataset_size: Dataset boyutu
            imbalance_ratio: Dengesizlik oranı
            minority_count: Azınlık sınıfı sayısı
            
        Returns:
            Öneriler dictionary
        """
        recommendations = {
            'augmentation_level': self.recommend_augmentation_level(dataset_size, imbalance_ratio),
            'recommended_config': None,
            'strategies': [],
            'warnings': []
        }
        
        # Seviye önerisi
        level = recommendations['augmentation_level']
        recommendations['recommended_config'] = self.get_preset_config(level)
        
        # Stratejik öneriler
        if dataset_size < 500:
            recommendations['strategies'].append("🔴 ÇOK KÜÇÜK dataset! Yoğun augmentation ve dış veri kaynakları gerekli")
            recommendations['warnings'].append("Overfitting riski çok yüksek!")
        elif dataset_size < 1000:
            recommendations['strategies'].append("🟠 Küçük dataset. Heavy augmentation önerilir")
        elif dataset_size < 5000:
            recommendations['strategies'].append("🟡 Orta boyut dataset. Normal augmentation yeterli")
        else:
            recommendations['strategies'].append("🟢 Yeterli veri. Hafif augmentation yeterli olabilir")
        
        if imbalance_ratio > 5:
            recommendations['strategies'].append("⚖️ Azınlık sınıfına özel yoğun augmentation uygulayın")
            recommendations['strategies'].append(f"   Hedef: {minority_count} → {int(minority_count * imbalance_ratio)} örnek")
        
        # Medical imaging özel öneriler
        recommendations['strategies'].append("🏥 Medical Imaging İçin:")
        recommendations['strategies'].append("   • Anatomik yapı korumalı transformlar kullanın")
        recommendations['strategies'].append("   • Elastic deformation organ deformasyonlarını simüle eder")
        recommendations['strategies'].append("   • Aşırı rotation/zoom'dan kaçının")
        
        return recommendations
    
    def save_config(self, config: Dict, output_path: str):
        """Augmentation config'i kaydet"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"✓ Augmentation config kaydedildi: {output_path}")
    
    def load_config(self, config_path: str) -> Dict:
        """Augmentation config'i yükle"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if self.verbose:
            print(f"✓ Augmentation config yüklendi: {config_path}")
        
        return config
    
    def compare_configs(self, config1: Dict, config2: Dict) -> Dict:
        """İki augmentation config'i karşılaştır"""
        comparison = {
            'config1_aggressiveness': 0.0,
            'config2_aggressiveness': 0.0,
            'differences': [],
            'similarities': []
        }
        
        report1 = self.analyze_augmentation_config(config1)
        report2 = self.analyze_augmentation_config(config2)
        
        comparison['config1_aggressiveness'] = report1['aggressiveness_score']
        comparison['config2_aggressiveness'] = report2['aggressiveness_score']
        
        # Farkları bul
        all_keys = set(list(config1.keys()) + list(config2.keys()))
        
        for key in all_keys:
            if key in ['enabled', 'mode']:
                continue
                
            val1 = config1.get(key, {}).get('enabled', False)
            val2 = config2.get(key, {}).get('enabled', False)
            
            if val1 != val2:
                comparison['differences'].append({
                    'transform': key,
                    'config1': val1,
                    'config2': val2
                })
            elif val1 and val2:
                comparison['similarities'].append(key)
        
        return comparison


if __name__ == "__main__":
    # Test
    manager = AugmentationManager(verbose=True)
    
    # Preset config
    print("=== NORMAL PRESET ===")
    config_normal = manager.get_preset_config('normal')
    manager.analyze_augmentation_config(config_normal)
    
    # Öneriler
    print("\n=== ÖNERİLER ===")
    recommendations = manager.get_recommendations(
        dataset_size=800,
        imbalance_ratio=4.5,
        minority_count=150
    )
    
    print(f"Önerilen Seviye: {recommendations['augmentation_level']}")
    for strategy in recommendations['strategies']:
        print(strategy)
