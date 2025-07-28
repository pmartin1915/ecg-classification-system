"""
Label processing utilities for ECG classification
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import pickle
from pathlib import Path


class LabelProcessor:
    """Process and encode ECG diagnostic labels"""
    
    def __init__(self, target_conditions: List[str]):
        self.target_conditions = target_conditions
        self.label_encoder = LabelEncoder()
        self.class_weights = None
        self.label_statistics = {}
        
    def process_labels(self, 
                      labels: List[List[str]], 
                      prioritize: bool = True) -> Tuple[np.ndarray, List[str], List[int]]:
        """
        Process multi-label data into single labels
        
        Args:
            labels: List of label lists for each sample
            prioritize: Whether to prioritize conditions by order in target_conditions
            
        Returns:
            Tuple of (encoded_labels, processed_labels, valid_indices)
        """
        print("=== LABEL PROCESSING ===")
        
        # Create priority mapping
        label_priority = {condition: i for i, condition in enumerate(self.target_conditions)}
        
        processed_labels = []
        valid_indices = []
        label_stats = Counter()
        multi_label_count = 0
        
        for idx, label_list in enumerate(labels):
            # Filter to target conditions only
            available_conditions = [cond for cond in label_list if cond in self.target_conditions]
            
            if not available_conditions:
                continue  # Skip records without target conditions
            
            if len(available_conditions) > 1:
                multi_label_count += 1
            
            if prioritize and len(available_conditions) > 1:
                # Select highest priority condition
                selected_condition = min(available_conditions, 
                                       key=lambda x: label_priority[x])
            else:
                # Take first available condition
                selected_condition = available_conditions[0]
            
            processed_labels.append(selected_condition)
            valid_indices.append(idx)
            label_stats[selected_condition] += 1
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(processed_labels)
        
        # Calculate class weights
        self.class_weights = self._calculate_class_weights(y_encoded)
        
        # Store statistics
        self.label_statistics = {
            'total_samples': len(labels),
            'valid_samples': len(processed_labels),
            'removed_samples': len(labels) - len(processed_labels),
            'multi_label_samples': multi_label_count,
            'label_distribution': dict(label_stats),
            'classes': self.label_encoder.classes_.tolist()
        }
        
        self._print_label_summary()
        
        return y_encoded, processed_labels, valid_indices
    
    def _calculate_class_weights(self, y_encoded: np.ndarray) -> Dict[int, float]:
        """Calculate class weights for imbalanced data"""
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_encoded),
            y=y_encoded
        )
        
        return {i: weight for i, weight in enumerate(class_weights)}
    
    def _print_label_summary(self):
        """Print summary of label processing"""
        stats = self.label_statistics
        
        print(f"\n✅ Label processing complete:")
        print(f"   - Total samples: {stats['total_samples']:,}")
        print(f"   - Valid samples: {stats['valid_samples']:,}")
        print(f"   - Removed: {stats['removed_samples']:,} "
              f"({stats['removed_samples']/stats['total_samples']*100:.1f}%)")
        print(f"   - Multi-label samples: {stats['multi_label_samples']:,}")
        
        print(f"\n📊 Label distribution:")
        for condition in self.target_conditions:
            count = stats['label_distribution'].get(condition, 0)
            percentage = (count / stats['valid_samples']) * 100 if stats['valid_samples'] > 0 else 0
            print(f"   - {condition}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n⚖️  Class weights for balancing:")
        for i, (class_name, weight) in enumerate(zip(self.label_encoder.classes_, 
                                                    self.class_weights.values())):
            print(f"   - {class_name}: {weight:.3f}")
    
    def encode_labels(self, labels: List[str]) -> np.ndarray:
        """Encode string labels to integers"""
        return self.label_encoder.transform(labels)
    
    def decode_labels(self, encoded_labels: np.ndarray) -> List[str]:
        """Decode integer labels to strings"""
        return self.label_encoder.inverse_transform(encoded_labels).tolist()
    
    def get_label_info(self) -> Dict:
        """Get comprehensive label information"""
        return {
            'encoder': self.label_encoder,
            'class_weights': self.class_weights,
            'target_conditions': self.target_conditions,
            'statistics': self.label_statistics
        }
    
    def save(self, filepath: Path):
        """Save label processor state"""
        state = {
            'target_conditions': self.target_conditions,
            'label_encoder': self.label_encoder,
            'class_weights': self.class_weights,
            'label_statistics': self.label_statistics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, filepath: Path):
        """Load label processor state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.target_conditions = state['target_conditions']
        self.label_encoder = state['label_encoder']
        self.class_weights = state['class_weights']
        self.label_statistics = state['label_statistics']