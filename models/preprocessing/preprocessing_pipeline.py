"""
Complete preprocessing pipeline for ECG classification
"""
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from config.preprocessing_config import PreprocessingConfig, PREPROCESSING_PRESETS
from config.settings import TARGET_CONDITIONS, CACHE_DIR, DATA_DIR
from models.preprocessing.signal_quality import SignalQualityAssessor
from models.preprocessing.signal_processor import SignalProcessor
from models.preprocessing.signal_normalizer import SignalNormalizer
from models.preprocessing.label_processor import LabelProcessor


class PreprocessingPipeline:
    """Complete preprocessing pipeline for ECG data"""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PREPROCESSING_PRESETS['standard']
        self.quality_assessor = SignalQualityAssessor(self.config)
        self.signal_processor = SignalProcessor(self.config)
        self.normalizer = SignalNormalizer(self.config)
        self.label_processor = LabelProcessor(TARGET_CONDITIONS)
        
        # Results storage
        self.results = {}
        
    def run(self, 
            X: np.ndarray,
            labels: List[List[str]],
            ids: List[str],
            use_cache: bool = True,
            visualize: bool = True) -> Dict:
        """
        Run complete preprocessing pipeline
        
        Args:
            X: Raw ECG signals
            labels: Diagnostic labels
            ids: Record IDs
            use_cache: Whether to use cached results
            visualize: Whether to create visualizations
            
        Returns:
            Dictionary with all preprocessing results
        """
        print("STARTING ECG PREPROCESSING PIPELINE")
        print("=" * 60)
        
        start_time = time.time()
        
        # Print configuration
        self._print_configuration()
        
        # Step 1: Quality Assessment
        X_valid, labels_valid, ids_valid, quality_reports = self.quality_assessor.filter_valid_signals(
            X, labels, ids
        )
        
        # Step 2: Signal Processing (filtering, artifact removal)
        X_processed, processing_info = self.signal_processor.process_batch(
            X_valid, 
            use_cache=use_cache,
            cache_dir=CACHE_DIR
        )
        
        # Step 3: Normalization
        X_normalized, norm_params = self.normalizer.normalize(X_processed)
        
        # Step 4: Label Processing
        y_encoded, labels_processed, valid_label_indices = self.label_processor.process_labels(
            labels_valid
        )
        
        # Step 5: Final alignment
        X_final = X_normalized[valid_label_indices]
        ids_final = [ids_valid[i] for i in valid_label_indices]
        processing_info_final = [processing_info[i] for i in valid_label_indices]
        
        # Step 6: Create visualizations
        if visualize:
            self._create_visualizations(X_valid, X_final, labels_processed, ids_final)
        
        # Compile results
        self.results = {
            'X_preprocessed': X_final,
            'y_encoded': y_encoded,
            'labels_processed': labels_processed,
            'ids_final': ids_final,
            'label_info': self.label_processor.get_label_info(),
            'normalization_params': norm_params,
            'quality_reports': quality_reports,
            'processing_info': processing_info_final,
            'config': self.config,
            'statistics': self._calculate_statistics(X, X_final, quality_reports, processing_info)
        }
        
        # Save results
        if use_cache:
            self._save_results()
        
        end_time = time.time()
        self._print_summary(end_time - start_time)
        
        return self.results
    
    def _print_configuration(self):
        """Print preprocessing configuration"""
        print(f"\n📋 Configuration:")
        print(f"   - Sampling rate: {self.config.sampling_rate} Hz")
        print(f"   - Target length: {self.config.target_length} samples")
        print(f"   - Filters: {self.config.highpass_freq}-{self.config.lowpass_freq} Hz")
        print(f"   - Notch filter: {self.config.notch_freq} Hz")
        print(f"   - Normalization: {self.config.normalization_method}")
        print(f"   - Quality threshold: max {self.config.max_missing_leads} bad leads")
    
    def _calculate_statistics(self, X_original, X_final, quality_reports, processing_info):
        """Calculate comprehensive statistics"""
        quality_stats = self.quality_assessor.get_quality_statistics(quality_reports)
        processing_stats = self.signal_processor.get_processing_statistics(processing_info)
        
        stats = {
            'original_samples': len(X_original),
            'final_samples': len(X_final),
            'reduction_rate': (len(X_original) - len(X_final)) / len(X_original),
            'quality_stats': quality_stats,
            'processing_stats': processing_stats,
            'data_shape': {
                'original': X_original.shape,
                'final': X_final.shape
            },
            'memory_usage': {
                'original_gb': X_original.nbytes / (1024**3),
                'final_gb': X_final.nbytes / (1024**3)
            }
        }
        
        return stats
    
    def _create_visualizations(self, X_raw, X_processed, labels, ids):
        """Create preprocessing visualizations"""
        print("\n=== CREATING VISUALIZATIONS ===")
        
        viz_dir = DATA_DIR / 'visualizations' / 'preprocessing'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Before/After comparison
        self._plot_signal_comparison(X_raw, X_processed, viz_dir)
        
        # 2. Signal statistics
        self._plot_signal_statistics(X_processed, viz_dir)
        
        # 3. Label distribution
        self._plot_label_distribution(labels, viz_dir)
        
        print(f"✅ Visualizations saved to: {viz_dir}")
    
    def _plot_signal_comparison(self, X_raw, X_processed, viz_dir):
        """Plot before/after signal comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ECG Preprocessing Results', fontsize=16)
        
        # Select a random sample
        sample_idx = np.random.randint(0, min(len(X_raw), len(X_processed)))
        
        # Plot first 2 leads
        for lead in range(min(2, X_raw.shape[2])):
            # Raw signal
            axes[0, lead].plot(X_raw[sample_idx, :, lead], 'b-', alpha=0.7)
            axes[0, lead].set_title(f'Lead {lead+1} - Raw Signal')
            axes[0, lead].set_ylabel('Amplitude (mV)')
            axes[0, lead].grid(True, alpha=0.3)
            
            # Processed signal
            axes[1, lead].plot(X_processed[sample_idx, :, lead], 'r-', alpha=0.7)
            axes[1, lead].set_title(f'Lead {lead+1} - Processed Signal')
            axes[1, lead].set_xlabel('Time (samples)')
            axes[1, lead].set_ylabel('Amplitude (normalized)')
            axes[1, lead].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'signal_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_signal_statistics(self, X_processed, viz_dir):
        """Plot signal statistics"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Amplitude distribution
        all_amplitudes = X_processed.flatten()
        axes[0].hist(all_amplitudes, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_title('Amplitude Distribution')
        axes[0].set_xlabel('Normalized Amplitude')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # Lead-wise statistics
        lead_means = np.mean(X_processed, axis=(0, 1))
        lead_stds = np.std(X_processed, axis=(0, 1))
        
        lead_indices = range(len(lead_means))
        axes[1].bar(lead_indices, lead_means, alpha=0.7)
        axes[1].set_title('Mean Amplitude per Lead')
        axes[1].set_xlabel('Lead Index')
        axes[1].set_ylabel('Mean Amplitude')
        axes[1].set_xticks(lead_indices)
        axes[1].grid(True, alpha=0.3)
        
        axes[2].bar(lead_indices, lead_stds, alpha=0.7, color='orange')
        axes[2].set_title('Standard Deviation per Lead')
        axes[2].set_xlabel('Lead Index')
        axes[2].set_ylabel('Standard Deviation')
        axes[2].set_xticks(lead_indices)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'signal_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_label_distribution(self, labels, viz_dir):
        """Plot label distribution"""
        label_counts = Counter(labels)
        
        plt.figure(figsize=(10, 6))
        conditions = list(label_counts.keys())
        counts = list(label_counts.values())
        
        bars = plt.bar(conditions, counts, alpha=0.7)
        
        # Color bars by frequency
        max_count = max(counts)
        for bar, count in zip(bars, counts):
            bar.set_color(plt.cm.viridis(count / max_count))
        
        plt.xlabel('Condition')
        plt.ylabel('Count')
        plt.title('Label Distribution After Preprocessing')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'label_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self):
        """Save preprocessing results"""
        output_file = DATA_DIR / 'processed' / 'preprocessing_results.pkl'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save individual components for easier access
        np.save(DATA_DIR / 'processed' / 'X_preprocessed.npy', self.results['X_preprocessed'])
        np.save(DATA_DIR / 'processed' / 'y_encoded.npy', self.results['y_encoded'])
        
        # Save label encoder
        self.label_processor.save(DATA_DIR / 'processed' / 'label_processor.pkl')
        
        # Save normalization parameters
        self.normalizer.save_parameters(DATA_DIR / 'processed' / 'normalization_params.pkl')
        
        print(f"\n📁 Results saved to: {DATA_DIR / 'processed'}")
    
    def _print_summary(self, elapsed_time):
        """Print preprocessing summary"""
        stats = self.results['statistics']
        
        print("\n" + "=" * 60)
        print("🎉 PREPROCESSING COMPLETE!")
        print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
        print(f"\n📊 Final dataset:")
        print(f"   - Shape: {stats['data_shape']['final']}")
        print(f"   - Samples: {stats['final_samples']:,} (from {stats['original_samples']:,})")
        print(f"   - Reduction: {stats['reduction_rate']*100:.1f}%")
        print(f"   - Memory: {stats['memory_usage']['final_gb']:.2f} GB")
        print(f"   - Classes: {len(self.label_processor.label_encoder.classes_)}")
        
        print(f"\n✅ Results available in: self.results")
        print(f"   - X_preprocessed: Preprocessed signals")
        print(f"   - y_encoded: Encoded labels")
        print(f"   - label_info: Label encoding information")
        print(f"   - statistics: Comprehensive statistics")
        print("\n🔄 Ready for Phase 3: Feature Engineering!")


# Convenience function for backwards compatibility
def run_phase2_preprocessing(X, labels, ids, target_conditions, 
                           drive_path=None, sampling_rate=100, 
                           normalization_method='z-score'):
    """Backwards compatible function for Phase 2"""
    
    # Create config
    config = PreprocessingConfig(
        sampling_rate=sampling_rate,
        normalization_method=normalization_method
    )
    
    # Run pipeline
    pipeline = PreprocessingPipeline(config)
    results = pipeline.run(X, labels, ids)
    
    # Return in original format
    return (
        results['X_preprocessed'],
        results['y_encoded'],
        results['labels_processed'],
        results['ids_final'],
        results['label_info']['encoder'],
        results['label_info']['class_weights'],
        config
    )