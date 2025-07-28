"""
Feature visualization utilities
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder

plt.style.use('seaborn-v0_8')


class FeatureVisualizer:
    """Create visualizations for feature analysis"""
    
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_feature_importance(self, 
                              importance_df: pd.DataFrame,
                              top_n: int = 20) -> None:
        """Plot feature importance scores"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Top N features by combined score
        top_features = importance_df.head(top_n)
        
        ax = axes[0, 0]
        ax.barh(range(len(top_features)), top_features['combined_score'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'], fontsize=8)
        ax.set_xlabel('Combined Importance Score')
        ax.set_title(f'Top {top_n} Most Important Features')
        ax.invert_yaxis()
        
        # Feature score comparison
        ax = axes[0, 1]
        if 'f_score_norm' in importance_df.columns and 'mutual_info_norm' in importance_df.columns:
            ax.scatter(importance_df['f_score_norm'], 
                      importance_df['mutual_info_norm'],
                      alpha=0.6, s=30)
            ax.set_xlabel('F-Score (Normalized)')
            ax.set_ylabel('Mutual Information (Normalized)')
            ax.set_title('Feature Selection Score Comparison')
            ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        
        # Feature categories distribution
        ax = axes[1, 0]
        feature_categories = self._categorize_features(importance_df['feature'].tolist())
        category_counts = pd.Series(feature_categories).value_counts()
        
        wedges, texts, autotexts = ax.pie(category_counts.values, 
                                          labels=category_counts.index,
                                          autopct='%1.1f%%',
                                          startangle=90)
        ax.set_title('Feature Distribution by Category')
        
        # Importance score distribution
        ax = axes[1, 1]
        ax.hist(importance_df['combined_score'], bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Combined Importance Score')
        ax.set_ylabel('Number of Features')
        ax.set_title('Distribution of Feature Importance Scores')
        ax.axvline(importance_df['combined_score'].mean(), 
                  color='red', linestyle='--', 
                  label=f'Mean: {importance_df["combined_score"].mean():.3f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'feature_importance_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_matrix(self, 
                              feature_df: pd.DataFrame,
                              feature_names: Optional[List[str]] = None,
                              max_features: int = 30) -> None:
        """Plot correlation matrix of features"""
        # Select features to plot
        if feature_names is None:
            feature_names = feature_df.columns[:max_features]
        else:
            feature_names = feature_names[:max_features]
        
        # Calculate correlation matrix
        corr_matrix = feature_df[feature_names].corr()
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=False,
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8})
        
        plt.title(f'Feature Correlation Matrix (Top {len(feature_names)} Features)')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'feature_correlation_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_distributions(self,
                                 feature_df: pd.DataFrame,
                                 y_encoded: np.ndarray,
                                 feature_names: List[str],
                                 class_names: Optional[List[str]] = None) -> None:
        """Plot feature distributions by class"""
        n_features = min(6, len(feature_names))
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(np.unique(y_encoded)))]
        
        for i, feature in enumerate(feature_names[:n_features]):
            ax = axes[i]
            
            # Create box plot for each class
            data_by_class = []
            for class_idx in sorted(np.unique(y_encoded)):
                class_mask = y_encoded == class_idx
                if class_mask.sum() > 0:
                    data_by_class.append(feature_df[feature][class_mask].values)
            
            bp = ax.boxplot(data_by_class, 
                           labels=[class_names[idx] for idx in sorted(np.unique(y_encoded))],
                           patch_artist=True)
            
            # Color boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(data_by_class)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_title(f'{feature}')
            ax.set_xlabel('Class')
            ax.set_ylabel('Feature Value')
            ax.tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Feature Distributions by Class', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'feature_class_distributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_pca_analysis(self,
                         X_pca: np.ndarray,
                         y_encoded: np.ndarray,
                         explained_variance_ratio: np.ndarray,
                         class_names: Optional[List[str]] = None) -> None:
        """Plot PCA analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(np.unique(y_encoded)))]
        
        # Explained variance
        ax = axes[0, 0]
        n_components = len(explained_variance_ratio)
        ax.plot(range(1, n_components + 1), explained_variance_ratio, 'bo-', markersize=4)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('Explained Variance by Component')
        ax.grid(True, alpha=0.3)
        
        # Cumulative explained variance
        ax = axes[0, 1]
        cumulative_variance = np.cumsum(explained_variance_ratio)
        ax.plot(range(1, n_components + 1), cumulative_variance, 'ro-', markersize=4)
        ax.axhline(y=0.95, color='k', linestyle='--', alpha=0.5, label='95% Variance')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.set_title('Cumulative Explained Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2D PCA projection
        ax = axes[1, 0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(y_encoded))))
        
        for i, class_name in enumerate(class_names):
            mask = y_encoded == i
            if mask.sum() > 0:
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                          c=[colors[i]], alpha=0.6, s=30, label=class_name)
        
        ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.3f})')
        ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.3f})')
        ax.set_title('2D PCA Projection')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3D PCA projection
        ax = fig.add_subplot(224, projection='3d')
        
        for i, class_name in enumerate(class_names):
            mask = y_encoded == i
            if mask.sum() > 0:
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                          c=[colors[i]], alpha=0.6, s=30, label=class_name)
        
        ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.3f})')
        ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.3f})')
        ax.set_zlabel(f'PC3 ({explained_variance_ratio[2]:.3f})')
        ax.set_title('3D PCA Projection')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'pca_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_heatmap(self,
                           feature_df: pd.DataFrame,
                           y_encoded: np.ndarray,
                           top_features: List[str],
                           max_samples: int = 100) -> None:
        """Plot heatmap of top features"""
        # Sample data if too large
        if len(feature_df) > max_samples:
            sample_idx = np.random.choice(len(feature_df), max_samples, replace=False)
            feature_df_sample = feature_df.iloc[sample_idx]
            y_sample = y_encoded[sample_idx]
        else:
            feature_df_sample = feature_df
            y_sample = y_encoded
        
        # Sort by class
        sort_idx = np.argsort(y_sample)
        feature_matrix = feature_df_sample[top_features[:20]].iloc[sort_idx].values.T
        y_sorted = y_sample[sort_idx]
        
        # Normalize features for visualization
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(feature_matrix.T).T
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        im = ax.imshow(feature_matrix, aspect='auto', cmap='RdBu_r', 
                      interpolation='nearest', vmin=-3, vmax=3)
        
        # Add class boundaries
        class_boundaries = []
        for class_idx in range(len(np.unique(y_sorted))):
            boundary = np.where(y_sorted == class_idx)[0][-1]
            if boundary < len(y_sorted) - 1:
                class_boundaries.append(boundary + 0.5)
        
        for boundary in class_boundaries:
            ax.axvline(x=boundary, color='black', linewidth=2)
        
        # Labels
        ax.set_yticks(range(len(top_features[:20])))
        ax.set_yticklabels(top_features[:20], fontsize=8)
        ax.set_xlabel('Samples (sorted by class)')
        ax.set_title('Top Features Heatmap')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Standardized Feature Value')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'feature_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _categorize_features(self, feature_names: List[str]) -> List[str]:
        """Categorize features by type"""
        categories = []
        
        feature_keywords = {
            'Temporal': ['mean', 'std', 'var', 'median', 'min', 'max', 'range', 
                        'rms', 'skewness', 'kurtosis'],
            'Morphological': ['r_peak', 'rr_', 'heart_rate', 'qrs_', 'pnn', 'rmssd'],
            'Frequency': ['power', 'freq', 'spectral', 'hrv_'],
            'ST-Segment': ['st_elevation', 'st_depression', 'st_slope'],
            'Wavelet': ['cwt_', 'dwt_', 'wavelet_'],
            'Other': []
        }
        
        for feature in feature_names:
            feature_lower = feature.lower()
            categorized = False
            
            for category, keywords in feature_keywords.items():
                if category != 'Other' and any(keyword in feature_lower for keyword in keywords):
                    categories.append(category)
                    categorized = True
                    break
            
            if not categorized:
                categories.append('Other')
        
        return categories
    
    def create_summary_plot(self,
                          feature_stats: Dict,
                          save_name: str = 'feature_extraction_summary.png') -> None:
        """Create summary visualization of feature extraction"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Feature categories pie chart
        ax = axes[0, 0]
        if 'feature_categories' in feature_stats:
            categories = feature_stats['feature_categories']
            ax.pie(categories.values(), labels=categories.keys(), autopct='%1.1f%%')
            ax.set_title('Features by Category')
        
        # Feature extraction statistics
        ax = axes[0, 1]
        stats_text = f"""Feature Extraction Summary
        
Total Features: {feature_stats.get('total_features', 0):,}
Selected Features: {feature_stats.get('selected_features', 0):,}
Reduction: {feature_stats.get('reduction_percentage', 0):.1f}%

Memory Usage: {feature_stats.get('memory_usage_mb', 0):.2f} MB
Processing Time: {feature_stats.get('processing_time', 0):.2f}s

Top Feature: {feature_stats.get('top_feature_name', 'N/A')}
Score: {feature_stats.get('top_feature_score', 0):.3f}"""
        
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax.axis('off')
        
        # Processing stages timeline
        ax = axes[1, 0]
        if 'processing_stages' in feature_stats:
            stages = feature_stats['processing_stages']
            y_pos = np.arange(len(stages))
            times = [s['time'] for s in stages.values()]
            names = list(stages.keys())
            
            ax.barh(y_pos, times)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names)
            ax.set_xlabel('Processing Time (seconds)')
            ax.set_title('Processing Time by Stage')
        
        # Feature quality metrics
        ax = axes[1, 1]
        quality_metrics = {
            'Valid Signals': feature_stats.get('valid_signals_percentage', 100),
            'Non-zero Variance': feature_stats.get('non_zero_variance_percentage', 100),
            'Uncorrelated': feature_stats.get('uncorrelated_percentage', 100),
            'Selected': feature_stats.get('selected_percentage', 100)
        }
        
        x = np.arange(len(quality_metrics))
        ax.bar(x, quality_metrics.values())
        ax.set_xticks(x)
        ax.set_xticklabels(quality_metrics.keys(), rotation=45, ha='right')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Feature Quality Metrics')
        ax.set_ylim(0, 105)
        
        # Add value labels on bars
        for i, v in enumerate(quality_metrics.values()):
            ax.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()