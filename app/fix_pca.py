# fix_pca.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys

def safe_pca_analysis(X: np.ndarray, desired_components: int = 50) -> dict:
    """Perform PCA with automatic component adjustment"""
    
    n_samples, n_features = X.shape
    
    # Calculate maximum possible components
    max_components = min(n_samples - 1, n_features)
    
    # Adjust components if necessary
    n_components = min(desired_components, max_components)
    
    if n_components < desired_components:
        print(f"📊 PCA: Adjusting components from {desired_components} to {n_components}")
        print(f"   (Data has {n_features} features and {n_samples} samples)")
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Find components needed for 95% variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    
    results = {
        'X_pca': X_pca,
        'n_components': n_components,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'n_components_95': n_components_95,
        'pca_model': pca,
        'scaler': scaler
    }
    
    print(f"✅ PCA completed successfully!")
    print(f"   Components used: {n_components}")
    print(f"   Variance explained: {cumulative_variance[-1]:.2%}")
    print(f"   Components for 95% variance: {n_components_95}")
    
    return results

# Test the fix
if __name__ == "__main__":
    print("Testing PCA fix...")
    
    # Create small test data (like your current test)
    X_test = np.random.randn(100, 20)  # 100 samples, 20 features
    
    # This would have failed before
    results = safe_pca_analysis(X_test, desired_components=50)
    
    print("\n✅ PCA fix working correctly!")