"""
Complete feature extraction pipeline for ECG classification
"""
import time
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from config.feature_config import FeatureExtractionConfig, FEATURE_EXTRACTION_PRESETS
from config.settings import CACHE_DIR, DATA_DIR
from models.feature_extraction.feature_extractor import ECGFeatureExtractor
from models.feature_extraction.feature_selection import FeatureSelector
from models.feature_extraction.visualization import FeatureVisualizer


class FeatureExtractionPipeline:
    """Complete pipeline for ECG feature extraction and selection"""
    
    def __init__(self, config: Optional[FeatureExtractionConfig] = None):
        self.config = config or FEATURE_EXTRACTION_PRESETS['standard']
        self.extractor = ECGFeatureExtractor(self.config)
        self.selector = FeatureSelector(self.config)
        self.results = {}
        
    def run(self,
            X_preprocessed: np.ndarray,
            y_encoded: np.ndarray,
            label_encoder: Any,
            use_cache: bool = True,
            visualize: bool = True) -> Dict[str, Any]:
        """
        Run complete feature extraction pipeline
        
        Args:
            X_preprocessed: Preprocessed ECG signals
            y_encoded: Encoded labels
            label_encoder: Label encoder object
            use_cache: Whether to use cached results
            visualize: Whether to create visualizations
            
        Returns:
            Dictionary with all extraction results
        """
        print("STARTING ECG FEATURE EXTRACTION PIPELINE")
        print("=" * 70)
        
        start_time = time.time()
        self._print_configuration()
        
        # Initialize tracking
        processing_stages = {}
        
        # Step 1: Extract all features
        print("\n" + "=" * 50)
        print("STEP 1: FEATURE EXTRACTION")
        print("=" * 50)
        
        stage_start = time.time()
        feature_df = self.extractor.extract_all_features(
            X_preprocessed,
            use_cache=use_cache,
            cache_dir=CACHE_DIR
        )
        processing_stages['Feature Extraction'] = {'time': time.time() - stage_start}
        
        # Step 2: Feature quality validation
        print("\n" + "=" * 50)
        print("STEP 2: FEATURE QUALITY VALIDATION")
        print("=" * 50)
        
        stage_start = time.time()
        feature_df, quality_issues = self._validate_feature_quality(feature_df)
        processing_stages['Quality Validation'] = {'time': time.time() - stage_start}
        
        # Step 3: Remove correlated features
        print("\n" + "=" * 50)
        print("STEP 3: CORRELATION ANALYSIS")
        print("=" * 50)
        
        stage_start = time.time()
        feature_df_clean, dropped_correlated = self.selector.remove_correlated_features(feature_df)
        processing_stages['Correlation Analysis'] = {'time': time.time() - stage_start}
        
        # Step 4: Feature importance analysis
        print("\n" + "=" * 50)
        print("STEP 4: FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        
        stage_start = time.time()
        feature_importance = self.selector.analyze_feature_importance(
            feature_df_clean, 
            y_encoded,
            methods=['f_classif', 'mutual_info']
        )
        processing_stages['Importance Analysis'] = {'time': time.time() - stage_start}
        
        # Step 5: Feature selection
        print("\n" + "=" * 50)
        print("STEP 5: FEATURE SELECTION")
        print("=" * 50)
        
        stage_start = time.time()
        feature_df_selected = self.selector.select_features(
            feature_df_clean,
            feature_importance,
            method='top_k'
        )
        processing_stages['Feature Selection'] = {'time': time.time() - stage_start}
        
        # Step 6: PCA analysis
        print("\n" + "=" * 50)
        print("STEP 6: PCA ANALYSIS")
        print("=" * 50)
        
        stage_start = time.time()
        n_components = min(50, feature_df_selected.shape[1])
        X_pca, pca, scaler = self.selector.perform_pca_analysis(
            feature_df_selected,
            n_components=n_components
        )
        processing_stages['PCA Analysis'] = {'time': time.time() - stage_start}
        
        # Compile results first (before visualization so statistics are available)
        total_time = time.time() - start_time
        
        self.results = {
            'X_features': feature_df_selected.values,
            'X_pca': X_pca,
            'feature_names': feature_df_selected.columns.tolist(),
            'feature_importance': feature_importance,
            'y_encoded': y_encoded,
            'label_encoder': label_encoder,
            'pca_transformer': pca,
            'scaler': scaler,
            'config': self.config,
            'statistics': {
                'total_features': len(feature_df.columns),
                'selected_features': len(feature_df_selected.columns),
                'reduction_percentage': (1 - len(feature_df_selected.columns) / len(feature_df.columns)) * 100,
                'quality_issues': quality_issues,
                'dropped_correlated': len(dropped_correlated),
                'memory_usage_mb': feature_df_selected.memory_usage(deep=True).sum() / (1024**2),
                'processing_time': total_time,
                'processing_stages': processing_stages,
                'top_feature_name': feature_importance.iloc[0]['feature'],
                'top_feature_score': feature_importance.iloc[0]['combined_score'],
                'feature_categories': self._count_feature_categories(feature_df_selected.columns.tolist()),
                'valid_signals_percentage': 100.0,  # Already filtered in preprocessing
                'non_zero_variance_percentage': 100.0,  # Already filtered
                'uncorrelated_percentage': (len(feature_df_selected.columns) / len(feature_df.columns)) * 100,
                'selected_percentage': (len(feature_df_selected.columns) / len(feature_df_clean.columns)) * 100
            }
        }
        
        # Step 7: Create visualizations (after results are compiled)
        if visualize:
            print("\n" + "=" * 50)
            print("STEP 7: CREATING VISUALIZATIONS")
            print("=" * 50)
            
            stage_start = time.time()
            self._create_visualizations(
                feature_df_selected,
                y_encoded,
                feature_importance,
                X_pca,
                pca,
                label_encoder
            )
            # Update processing time to include visualization
            visualization_time = time.time() - stage_start
            processing_stages['Visualization'] = {'time': visualization_time}
            self.results['statistics']['processing_stages'] = processing_stages
            self.results['statistics']['processing_time'] = time.time() - start_time
        
        # Save results
        if use_cache:
            self._save_results()
        
        # Create report
        report = self.selector.create_feature_report(DATA_DIR / 'results' / 'features')
        
        self._print_summary()
        
        return self.results
    
    def _print_configuration(self):
        """Print pipeline configuration"""
        print(f"\n📋 Configuration:")
        print(f"   - Sampling rate: {self.config.sampling_rate} Hz")
        print(f"   - R-peak detection: height={self.config.r_peak_height}, distance={self.config.r_peak_distance}")
        print(f"   - Feature selection: top {self.config.feature_selection_k} features")
        print(f"   - Correlation threshold: {self.config.correlation_threshold}")
        print(f"   - Wavelet scales: {len(self.config.wavelet_scales)}")
    
    def _validate_feature_quality(self, feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Validate feature quality and identify issues"""
        issues = []
        
        # Check for infinite values
        inf_features = feature_df.columns[np.isinf(feature_df).any()].tolist()
        if inf_features:
            issues.append(f"Infinite values found in {len(inf_features)} features")
            feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
            feature_df = feature_df.fillna(feature_df.median())
        
        # Check for constant features
        constant_features = feature_df.columns[feature_df.nunique() == 1].tolist()
        if constant_features:
            issues.append(f"Constant features found: {len(constant_features)}")
            feature_df = feature_df.drop(columns=constant_features)
        
        # Check for highly skewed features
        numeric_cols = feature_df.select_dtypes(include=np.number).columns
        skewness = feature_df[numeric_cols].skew()
        highly_skewed = skewness[np.abs(skewness) > 5].index.tolist()
        if highly_skewed:
            issues.append(f"Highly skewed features: {len(highly_skewed)}")
        
        if issues:
            print("⚠️  Feature quality issues found:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ No major feature quality issues found")
        
        return feature_df, issues
    
    def _create_visualizations(self,
                             feature_df: pd.DataFrame,
                             y_encoded: np.ndarray,
                             feature_importance: pd.DataFrame,
                             X_pca: np.ndarray,
                             pca: Any,
                             label_encoder: Any) -> None:
        """Create all visualizations"""
        viz_dir = DATA_DIR / 'visualizations' / 'features'
        visualizer = FeatureVisualizer(viz_dir)
        
        # Feature importance plots
        visualizer.plot_feature_importance(feature_importance)
        
        # Correlation matrix
        top_features = feature_importance.head(30)['feature'].tolist()
        visualizer.plot_correlation_matrix(feature_df, top_features)
        
        # Feature distributions by class
        top_features_dist = feature_importance.head(6)['feature'].tolist()
        visualizer.plot_feature_distributions(
            feature_df,
            y_encoded,
            top_features_dist,
            class_names=label_encoder.classes_.tolist()
        )
        
        # PCA analysis
        visualizer.plot_pca_analysis(
            X_pca,
            y_encoded,
            pca.explained_variance_ratio_,
            class_names=label_encoder.classes_.tolist()
        )
        
        # Feature heatmap
        top_features_heatmap = feature_importance.head(20)['feature'].tolist()
        visualizer.plot_feature_heatmap(
            feature_df,
            y_encoded,
            top_features_heatmap
        )
        
        # Summary plot
        visualizer.create_summary_plot(self.results['statistics'])
        
        print(f"✅ Visualizations saved to: {viz_dir}")
    
    def _count_feature_categories(self, feature_names: List[str]) -> Dict[str, int]:
        """Count features by category"""
        categories = {
            'temporal': 0,
            'morphological': 0,
            'frequency': 0,
            'st_segment': 0,
            'wavelet': 0,
            'other': 0
        }
        
        feature_keywords = {
            'temporal': ['mean', 'std', 'var', 'median', 'min', 'max', 'range', 'rms'],
            'morphological': ['r_peak', 'rr_', 'heart_rate', 'qrs_', 'pnn', 'rmssd'],
            'frequency': ['power', 'freq', 'spectral', 'hrv_'],
            'st_segment': ['st_elevation', 'st_depression', 'st_slope'],
            'wavelet': ['cwt_', 'dwt_', 'wavelet_']
        }
        
        for feature in feature_names:
            feature_lower = feature.lower()
            categorized = False
            
            for category, keywords in feature_keywords.items():
                if any(keyword in feature_lower for keyword in keywords):
                    categories[category] += 1
                    categorized = True
                    break
            
            if not categorized:
                categories['other'] += 1
        
        return categories
    
    def _save_results(self):
        """Save extraction results"""
        output_dir = DATA_DIR / 'processed'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        output_file = output_dir / 'feature_extraction_results.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save individual components
        np.save(output_dir / 'X_features.npy', self.results['X_features'])
        np.save(output_dir / 'X_pca.npy', self.results['X_pca'])
        
        # Save feature names
        with open(output_dir / 'feature_names.txt', 'w') as f:
            for name in self.results['feature_names']:
                f.write(f"{name}\n")
        
        # Save transformers
        with open(output_dir / 'feature_transformers.pkl', 'wb') as f:
            pickle.dump({
                'pca': self.results['pca_transformer'],
                'scaler': self.results['scaler']
            }, f)
        
        print(f"\n📁 Results saved to: {output_dir}")
    
    def _print_summary(self):
        """Print extraction summary"""
        stats = self.results['statistics']
        
        print("\n" + "=" * 70)
        print("🎉 FEATURE EXTRACTION COMPLETE!")
        print(f"⏱️  Total time: {stats['processing_time']:.2f} seconds")
        print(f"\n📊 Results:")
        print(f"   - Original features: {stats['total_features']:,}")
        print(f"   - Selected features: {stats['selected_features']:,}")
        print(f"   - Reduction: {stats['reduction_percentage']:.1f}%")
        print(f"   - Memory usage: {stats['memory_usage_mb']:.2f} MB")
        print(f"\n🏆 Top feature: {stats['top_feature_name']} (score: {stats['top_feature_score']:.3f})")
        print(f"\n📊 Feature categories:")
        for category, count in stats['feature_categories'].items():
            print(f"   - {category}: {count}")
        print("\n🔄 Ready for Phase 4: Model Training!")


# Convenience function for backwards compatibility
def run_phase3_feature_extraction(X_preprocessed, y_encoded, label_encoder, 
                                target_conditions, drive_path=None,
                                sampling_rate=100):
    """Backwards compatible function for Phase 3"""
    
    # Create config
    config = FeatureExtractionConfig(sampling_rate=sampling_rate)
    
    # Run pipeline
    pipeline = FeatureExtractionPipeline(config)
    results = pipeline.run(X_preprocessed, y_encoded, label_encoder)
    
    # Return in format similar to original
    return results
