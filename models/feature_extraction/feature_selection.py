"""
Feature selection and analysis utilities
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import pickle

from config.feature_config import FeatureExtractionConfig


class FeatureSelector:
    """Feature selection and importance analysis"""
    
    def __init__(self, config: FeatureExtractionConfig):
        self.config = config
        self.feature_importance = None
        self.selected_features = None
        self.selection_method = None
    
    def analyze_feature_importance(self, 
                                 X: pd.DataFrame, 
                                 y: np.ndarray,
                                 methods: List[str] = ['f_classif', 'mutual_info']) -> pd.DataFrame:
        """
        Analyze feature importance using multiple methods
        
        Args:
            X: Feature DataFrame
            y: Target labels
            methods: List of methods to use
            
        Returns:
            DataFrame with feature importance scores
        """
        print("=== FEATURE IMPORTANCE ANALYSIS ===")
        
        importance_scores = pd.DataFrame({'feature': X.columns})
        
        # F-statistic
        if 'f_classif' in methods:
            selector_f = SelectKBest(score_func=f_classif, k='all')
            selector_f.fit(X, y)
            importance_scores['f_score'] = selector_f.scores_
            importance_scores['f_pvalue'] = selector_f.pvalues_
        
        # Mutual information
        if 'mutual_info' in methods:
            selector_mi = SelectKBest(score_func=mutual_info_classif, k='all')
            selector_mi.fit(X, y)
            importance_scores['mutual_info'] = selector_mi.scores_
        
        # Chi-squared (for non-negative features)
        if 'chi2' in methods and (X >= 0).all().all():
            selector_chi2 = SelectKBest(score_func=chi2, k='all')
            selector_chi2.fit(X, y)
            importance_scores['chi2_score'] = selector_chi2.scores_
        
        # Normalize scores
        for col in importance_scores.columns:
            if col != 'feature' and col != 'f_pvalue':
                min_val = importance_scores[col].min()
                max_val = importance_scores[col].max()
                if max_val > min_val:
                    importance_scores[f'{col}_norm'] = (
                        (importance_scores[col] - min_val) / (max_val - min_val)
                    )
                else:
                    importance_scores[f'{col}_norm'] = 0.5
        
        # Calculate combined score
        norm_cols = [col for col in importance_scores.columns if col.endswith('_norm')]
        if norm_cols:
            importance_scores['combined_score'] = importance_scores[norm_cols].mean(axis=1)
        else:
            importance_scores['combined_score'] = 0.5
        
        # Sort by combined score
        importance_scores = importance_scores.sort_values('combined_score', ascending=False)
        
        self.feature_importance = importance_scores
        
        print(f"✅ Feature importance analysis complete:")
        print(f"   - Total features analyzed: {len(X.columns)}")
        print(f"   - Methods used: {methods}")
        
        return importance_scores
    
    def select_features(self, 
                       X: pd.DataFrame,
                       importance_scores: Optional[pd.DataFrame] = None,
                       method: str = 'top_k',
                       k: Optional[int] = None) -> pd.DataFrame:
        """
        Select features based on importance scores
        
        Args:
            X: Feature DataFrame
            importance_scores: Pre-computed importance scores
            method: Selection method ('top_k', 'percentile', 'elbow')
            k: Number of features to select
            
        Returns:
            Selected features DataFrame
        """
        if importance_scores is None:
            importance_scores = self.feature_importance
        
        if importance_scores is None:
            raise ValueError("No feature importance scores available")
        
        k = k or self.config.feature_selection_k
        
        if method == 'top_k':
            selected_feature_names = importance_scores.head(k)['feature'].tolist()
        
        elif method == 'percentile':
            threshold = np.percentile(importance_scores['combined_score'], 100 - k)
            selected_feature_names = importance_scores[
                importance_scores['combined_score'] >= threshold
            ]['feature'].tolist()
        
        elif method == 'elbow':
            # Find elbow point in importance scores
            scores = importance_scores['combined_score'].values
            diffs = np.diff(scores)
            diffs2 = np.diff(diffs)
            
            # Find point of maximum curvature
            if len(diffs2) > 0:
                elbow_idx = np.argmax(np.abs(diffs2)) + 2
                selected_feature_names = importance_scores.head(elbow_idx)['feature'].tolist()
            else:
                selected_feature_names = importance_scores.head(k)['feature'].tolist()
        
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        self.selected_features = selected_feature_names
        self.selection_method = method
        
        X_selected = X[selected_feature_names]
        
        print(f"\n✅ Feature selection complete:")
        print(f"   - Method: {method}")
        print(f"   - Selected features: {len(selected_feature_names)}")
        print(f"   - Reduction: {(1 - len(selected_feature_names)/len(X.columns))*100:.1f}%")
        
        return X_selected
    
    def remove_correlated_features(self, 
                                 X: pd.DataFrame,
                                 threshold: Optional[float] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove highly correlated features
        
        Args:
            X: Feature DataFrame
            threshold: Correlation threshold
            
        Returns:
            Tuple of (cleaned DataFrame, removed features)
        """
        threshold = threshold or self.config.correlation_threshold
        
        print(f"\n🔍 Removing highly correlated features (threshold: {threshold})")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find features to remove
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = []
        for column in upper_tri.columns:
            if any(upper_tri[column] > threshold):
                # Keep the feature with higher importance if available
                if self.feature_importance is not None:
                    correlated_features = upper_tri.index[upper_tri[column] > threshold].tolist()
                    correlated_features.append(column)
                    
                    # Get importance scores
                    importance_dict = dict(zip(
                        self.feature_importance['feature'],
                        self.feature_importance['combined_score']
                    ))
                    
                    # Keep the most important feature
                    feature_scores = [(f, importance_dict.get(f, 0)) for f in correlated_features]
                    feature_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # Drop all except the most important
                    for f, _ in feature_scores[1:]:
                        if f not in to_drop:
                            to_drop.append(f)
                else:
                    # No importance scores, just drop the second feature
                    if column not in to_drop:
                        to_drop.append(column)
        
        print(f"   - Found {len(to_drop)} correlated features to remove")
        
        X_clean = X.drop(columns=to_drop)
        
        print(f"   - Remaining features: {len(X_clean.columns)}")
        
        return X_clean, to_drop
    
    def perform_pca_analysis(self, 
                           X: pd.DataFrame,
                           n_components: Optional[int] = None) -> Tuple[np.ndarray, PCA, StandardScaler]:
        """
        Perform PCA analysis
        
        Args:
            X: Feature DataFrame
            n_components: Number of components
            
        Returns:
            Tuple of (transformed data, PCA object, scaler)
        """
        print("\n=== PCA ANALYSIS ===")
        
        n_components = n_components or min(50, X.shape[1])
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate metrics
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Find components for 95% variance
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        print(f"✅ PCA Analysis Results:")
        print(f"   - Components: {n_components}")
        print(f"   - Explained variance (first 10): {explained_variance_ratio[:10].sum():.3f}")
        print(f"   - Cumulative variance: {cumulative_variance[-1]:.3f}")
        print(f"   - Components for 95% variance: {n_components_95}")
        
        # Get feature contributions
        feature_contributions = pd.DataFrame(
            abs(pca.components_[:10]).T,
            columns=[f'PC{i+1}' for i in range(10)],
            index=X.columns
        )
        
        # Find top contributing features for each PC
        self.pca_top_features = {}
        for i in range(min(10, n_components)):
            top_features = feature_contributions[f'PC{i+1}'].nlargest(5)
            self.pca_top_features[f'PC{i+1}'] = top_features.to_dict()
        
        return X_pca, pca, scaler
    
    def create_feature_report(self, save_dir: Path) -> Dict:
        """Create comprehensive feature analysis report"""
        report = {
            'total_features': len(self.feature_importance) if self.feature_importance is not None else 0,
            'selected_features': len(self.selected_features) if self.selected_features else 0,
            'selection_method': self.selection_method,
            'top_10_features': [],
            'feature_categories': {},
            'pca_top_features': self.pca_top_features if hasattr(self, 'pca_top_features') else {}
        }
        
        if self.feature_importance is not None:
            # Top 10 features
            for _, row in self.feature_importance.head(10).iterrows():
                report['top_10_features'].append({
                    'name': row['feature'],
                    'score': float(row['combined_score'])
                })
            
            # Feature categories
            feature_groups = {
                'temporal': ['mean', 'std', 'var', 'median', 'min', 'max'],
                'morphological': ['r_peak', 'rr_', 'heart_rate', 'qrs_'],
                'frequency': ['power', 'freq', 'spectral', 'hrv_'],
                'st_segment': ['st_elevation', 'st_depression', 'st_slope'],
                'wavelet': ['cwt_', 'dwt_', 'wavelet_']
            }
            
            for category, keywords in feature_groups.items():
                count = sum(1 for feature in self.feature_importance['feature']
                          if any(keyword in feature.lower() for keyword in keywords))
                report['feature_categories'][category] = count
        
        # Save report
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_dir / 'feature_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        if self.feature_importance is not None:
            self.feature_importance.to_csv(save_dir / 'feature_importance.csv', index=False)
        
        return report