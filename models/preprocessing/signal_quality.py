"""
Signal quality assessment for ECG data
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
from tqdm import tqdm

from config.preprocessing_config import PreprocessingConfig


class SignalQualityAssessor:
    """Assess quality of ECG signals"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
    def assess_signal(self, signal: np.ndarray) -> Dict:
        """
        Comprehensive signal quality assessment
        
        Args:
            signal: ECG signal array (time_points, leads)
            
        Returns:
            dict: Quality metrics and flags
        """
        quality_metrics = {
            'length': signal.shape[0],
            'leads': signal.shape[1] if signal.ndim > 1 else 1,
            'is_valid': True,
            'issues': [],
            'lead_quality': []
        }
        
        # Check signal dimensions
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        
        # Check signal length
        if signal.shape[0] < self.config.min_signal_length:
            quality_metrics['is_valid'] = False
            quality_metrics['issues'].append(
                f"Signal too short: {signal.shape[0]} < {self.config.min_signal_length}"
            )
        
        # Assess each lead
        missing_leads = 0
        for lead_idx in range(signal.shape[1]):
            lead_signal = signal[:, lead_idx]
            lead_quality = self._assess_lead_quality(lead_signal, lead_idx)
            quality_metrics['lead_quality'].append(lead_quality)
            
            if not lead_quality['is_valid']:
                missing_leads += 1
        
        quality_metrics['missing_leads'] = missing_leads
        
        # Check if too many bad leads
        if missing_leads > self.config.max_missing_leads:
            quality_metrics['is_valid'] = False
            quality_metrics['issues'].append(
                f"Too many bad leads: {missing_leads} > {self.config.max_missing_leads}"
            )
        
        # Overall signal statistics
        quality_metrics['mean_amplitude'] = float(np.mean(np.abs(signal)))
        quality_metrics['std_amplitude'] = float(np.std(signal))
        quality_metrics['snr_estimate'] = float(
            np.mean(np.abs(signal)) / (np.std(signal) + 1e-8)
        )
        
        return quality_metrics
    
    def _assess_lead_quality(self, lead_signal: np.ndarray, lead_idx: int) -> Dict:
        """Assess quality of a single lead"""
        lead_quality = {
            'lead_idx': lead_idx,
            'is_valid': True,
            'issues': []
        }
        
        # Check for flat line (constant values)
        std_dev = np.std(lead_signal)
        if std_dev < self.config.min_lead_std:
            lead_quality['is_valid'] = False
            lead_quality['issues'].append(f"Flat line (std={std_dev:.6f})")
        
        # Check for excessive noise
        elif std_dev > self.config.max_lead_std:
            lead_quality['issues'].append(f"Excessive noise (std={std_dev:.2f})")
        
        # Check for saturation
        signal_range = np.max(lead_signal) - np.min(lead_signal)
        if signal_range < self.config.min_signal_range:
            lead_quality['is_valid'] = False
            lead_quality['issues'].append(f"Possible saturation (range={signal_range:.3f})")
        
        # Check for clipping
        max_val = np.max(np.abs(lead_signal))
        if max_val > 0:
            saturation_count = np.sum(
                np.abs(lead_signal) > (max_val * self.config.saturation_threshold)
            )
            saturation_ratio = saturation_count / len(lead_signal)
            if saturation_ratio > 0.01:  # More than 1% of samples at max
                lead_quality['issues'].append(
                    f"Possible clipping ({saturation_ratio*100:.1f}% samples)"
                )
        
        # Additional metrics
        lead_quality['std'] = float(std_dev)
        lead_quality['range'] = float(signal_range)
        lead_quality['mean'] = float(np.mean(lead_signal))
        
        return lead_quality
    
    def filter_valid_signals(self, 
                           X: np.ndarray, 
                           labels: List, 
                           ids: List,
                           verbose: bool = True) -> Tuple[np.ndarray, List, List, List[Dict]]:
        """
        Filter out invalid signals based on quality assessment
        
        Returns:
            Tuple of (X_valid, labels_valid, ids_valid, quality_reports)
        """
        if verbose:
            print("=== SIGNAL QUALITY ASSESSMENT ===")
        
        valid_indices = []
        quality_reports = []
        
        # Use tqdm for progress bar
        iterator = enumerate(zip(X, labels, ids))
        if verbose:
            iterator = tqdm(
                iterator, 
                total=len(X), 
                desc="Assessing quality"
            )
        
        for i, (signal, label, record_id) in iterator:
            quality = self.assess_signal(signal)
            quality['record_id'] = record_id
            quality_reports.append(quality)
            
            if quality['is_valid']:
                valid_indices.append(i)
        
        # Filter datasets
        X_valid = X[valid_indices]
        labels_valid = [labels[i] for i in valid_indices]
        ids_valid = [ids[i] for i in valid_indices]
        
        if verbose:
            self._print_quality_summary(X, X_valid, quality_reports)
        
        return X_valid, labels_valid, ids_valid, quality_reports
    
    def _print_quality_summary(self, X_original, X_valid, quality_reports):
        """Print summary of quality assessment"""
        print(f"\n✅ Quality assessment complete:")
        print(f"   - Original records: {len(X_original):,}")
        print(f"   - Valid records: {len(X_valid):,}")
        print(f"   - Rejected: {len(X_original) - len(X_valid):,} "
              f"({(len(X_original) - len(X_valid))/len(X_original)*100:.1f}%)")
        
        # Collect all issues
        all_issues = []
        for report in quality_reports:
            all_issues.extend(report['issues'])
            for lead_quality in report.get('lead_quality', []):
                all_issues.extend(
                    [f"Lead {lead_quality['lead_idx']}: {issue}" 
                     for issue in lead_quality['issues']]
                )
        
        if all_issues:
            issue_counts = Counter(all_issues)
            print(f"\n📊 Common quality issues:")
            for issue, count in issue_counts.most_common(5):
                print(f"   - {issue}: {count} occurrences")
    
    def get_quality_statistics(self, quality_reports: List[Dict]) -> Dict:
        """Generate statistics from quality reports"""
        valid_count = sum(1 for r in quality_reports if r['is_valid'])
        total_count = len(quality_reports)
        
        stats = {
            'total_records': total_count,
            'valid_records': valid_count,
            'invalid_records': total_count - valid_count,
            'validity_rate': valid_count / total_count if total_count > 0 else 0,
            'mean_snr': np.mean([r['snr_estimate'] for r in quality_reports]),
            'mean_amplitude': np.mean([r['mean_amplitude'] for r in quality_reports]),
            'issue_summary': Counter()
        }
        
        # Count issues
        for report in quality_reports:
            for issue in report['issues']:
                stats['issue_summary'][issue] += 1
        
        return stats