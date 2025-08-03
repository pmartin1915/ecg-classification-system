"""
Enhanced ECG Classification Component for Clinical Interface
Supports both legacy and new MI-enhanced models with real-time prediction
"""
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import time
import warnings
warnings.filterwarnings('ignore')

from config.settings import MODELS_DIR, TARGET_CONDITIONS


class ECGClassifier:
    """
    Production-ready ECG classifier with MI-enhanced model support
    """
    
    def __init__(self, model_type: str = "combined_mi_enhanced"):
        """
        Initialize classifier with specified model type
        
        Args:
            model_type: "combined_mi_enhanced", "ptbxl_only", or "auto"
        """
        self.model_type = model_type
        self.model = None
        self.feature_transformers = None
        self.label_encoder = None
        self.feature_names = None
        self.model_info = {}
        self.confidence_thresholds = {
            'MI': 0.7,      # High threshold for MI (critical condition)
            'NORM': 0.6,    # Standard threshold for normal
            'STTC': 0.65,   # Moderate threshold for ST/T changes
            'CD': 0.65,     # Moderate threshold for conduction disorders
            'HYP': 0.65     # Moderate threshold for hypertrophy
        }
        
        # Cache for loaded models (singleton pattern)
        self._model_cache = {}
        
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load the trained ECG classification model
        
        Args:
            model_path: Path to specific model file, or auto-detect
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            if model_path is None:
                model_path = self._find_best_model()
            
            if not model_path:
                st.error("❌ No trained model found. Please train a model first.")
                return False
            
            # Check cache first
            cache_key = str(model_path)
            if cache_key in self._model_cache:
                cached_data = self._model_cache[cache_key]
                self.model = cached_data['model']
                self.feature_transformers = cached_data.get('transformers')
                self.label_encoder = cached_data.get('label_encoder')
                self.feature_names = cached_data.get('feature_names')
                self.model_info = cached_data.get('info', {})
                return True
            
            # Load model components
            model_path = Path(model_path)
            
            # Load main model
            if model_path.suffix in ['.pkl', '.pickle']:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
            else:
                model_data = joblib.load(model_path)
            
            # Handle different model formats
            if isinstance(model_data, dict):
                self.model = model_data.get('model', model_data.get('best_model'))
                self.feature_transformers = model_data.get('feature_transformers')
                self.label_encoder = model_data.get('label_encoder')
                self.feature_names = model_data.get('feature_names')
                self.model_info = model_data.get('model_info', {})
            else:
                # Simple model object
                self.model = model_data
                self.model_info = {'type': 'simple_model'}
            
            # Try to load auxiliary files
            model_dir = model_path.parent
            self._load_auxiliary_files(model_dir)
            
            # Cache the loaded model
            self._model_cache[cache_key] = {
                'model': self.model,
                'transformers': self.feature_transformers,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
                'info': self.model_info
            }
            
            # Update model info
            self.model_info.update({
                'loaded_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_path': str(model_path),
                'has_mi_enhancement': 'combined' in str(model_path).lower() or 'mi' in str(model_path).lower()
            })
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return False
    
    def _find_best_model(self) -> Optional[str]:
        """Find the best available model based on type preference"""
        model_dir = MODELS_DIR / "trained_models"
        
        if not model_dir.exists():
            return None
        
        # Priority order for model selection
        if self.model_type == "combined_mi_enhanced":
            search_patterns = [
                "*combined*mi*.pkl",
                "*combined*.pkl", 
                "*mi_enhanced*.pkl",
                "*phase4*.pkl",
                "*.pkl"
            ]
        elif self.model_type == "ptbxl_only":
            search_patterns = [
                "*ptbxl*.pkl",
                "*phase4*.pkl", 
                "*.pkl"
            ]
        else:  # auto
            search_patterns = [
                "*combined*mi*.pkl",
                "*combined*.pkl",
                "*mi_enhanced*.pkl",
                "*phase4*.pkl",
                "*.pkl"
            ]
        
        for pattern in search_patterns:
            models = list(model_dir.glob(pattern))
            if models:
                # Return the most recent model
                latest_model = max(models, key=lambda p: p.stat().st_mtime)
                return str(latest_model)
        
        return None
    
    def _load_auxiliary_files(self, model_dir: Path):
        """Load auxiliary files like feature transformers and encoders"""
        # Try to load feature transformers
        transformer_files = [
            "feature_transformers.pkl",
            "transformers.pkl", 
            "preprocessing.pkl"
        ]
        
        for tf_file in transformer_files:
            tf_path = model_dir / tf_file
            if tf_path.exists():
                try:
                    with open(tf_path, 'rb') as f:
                        self.feature_transformers = pickle.load(f)
                    break
                except:
                    continue
        
        # Try to load label encoder
        encoder_files = [
            "label_encoder.pkl",
            "encoder.pkl"
        ]
        
        for enc_file in encoder_files:
            enc_path = model_dir / enc_file
            if enc_path.exists():
                try:
                    with open(enc_path, 'rb') as f:
                        self.label_encoder = pickle.load(f)
                    break
                except:
                    continue
        
        # Try to load feature names
        feature_files = [
            "feature_names.txt",
            "features.txt"
        ]
        
        for feat_file in feature_files:
            feat_path = model_dir / feat_file
            if feat_path.exists():
                try:
                    with open(feat_path, 'r') as f:
                        self.feature_names = [line.strip() for line in f.readlines()]
                    break
                except:
                    continue
    
    def predict(self, X: np.ndarray, return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Perform ECG classification prediction
        
        Args:
            X: Feature matrix or preprocessed ECG signals
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with predictions, probabilities, and confidence metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Ensure X is 2D
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            # Get base predictions
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)
                predictions = self.model.predict(X)
            else:
                predictions = self.model.predict(X)
                probabilities = None
            
            # Convert predictions to labels if we have label encoder
            if self.label_encoder is not None:
                predicted_labels = self.label_encoder.inverse_transform(predictions)
            else:
                predicted_labels = predictions
            
            # Calculate confidence and risk assessment
            results = []
            for i in range(len(predictions)):
                pred_label = predicted_labels[i] if hasattr(predicted_labels, '__getitem__') else predicted_labels
                
                result = {
                    'prediction': pred_label,
                    'prediction_index': predictions[i] if hasattr(predictions, '__getitem__') else predictions,
                    'confidence_score': 0.0,
                    'risk_level': 'Unknown',
                    'clinical_priority': 'Standard',
                    'probabilities': {},
                    'confidence_assessment': 'Low'
                }
                
                if probabilities is not None:
                    # Get probabilities for this sample
                    sample_probs = probabilities[i] if probabilities.ndim > 1 else probabilities
                    
                    # Map probabilities to condition names
                    if self.label_encoder is not None:
                        condition_names = self.label_encoder.classes_
                    else:
                        condition_names = TARGET_CONDITIONS
                    
                    for j, condition in enumerate(condition_names):
                        if j < len(sample_probs):
                            result['probabilities'][condition] = float(sample_probs[j])
                    
                    # Calculate confidence score
                    max_prob = np.max(sample_probs)
                    result['confidence_score'] = float(max_prob)
                    
                    # Assess confidence level
                    threshold = self.confidence_thresholds.get(pred_label, 0.65)
                    if max_prob >= threshold:
                        result['confidence_assessment'] = 'High'
                    elif max_prob >= 0.5:
                        result['confidence_assessment'] = 'Moderate'
                    else:
                        result['confidence_assessment'] = 'Low'
                
                # Assess clinical risk and priority
                result.update(self._assess_clinical_risk(pred_label, result['confidence_score']))
                
                results.append(result)
            
            prediction_time = time.time() - start_time
            
            # Return summary for single prediction or list for batch
            if len(results) == 1:
                results[0]['prediction_time_ms'] = prediction_time * 1000
                results[0]['model_info'] = self.model_info
                return results[0]
            else:
                return {
                    'predictions': results,
                    'batch_size': len(results),
                    'total_prediction_time_ms': prediction_time * 1000,
                    'avg_prediction_time_ms': (prediction_time * 1000) / len(results),
                    'model_info': self.model_info
                }
                
        except Exception as e:
            return {
                'error': str(e),
                'prediction': 'Error',
                'confidence_score': 0.0,
                'risk_level': 'Unknown'
            }
    
    def _assess_clinical_risk(self, condition: str, confidence: float) -> Dict[str, str]:
        """Assess clinical risk level and priority based on condition and confidence"""
        
        # Base risk assessment by condition
        risk_mapping = {
            'MI': 'Critical',       # Myocardial Infarction - immediate attention
            'STTC': 'High',        # ST/T Changes - concerning
            'CD': 'Moderate',      # Conduction Disorders - monitor
            'HYP': 'Moderate',     # Hypertrophy - follow-up needed
            'NORM': 'Low'          # Normal - routine
        }
        
        priority_mapping = {
            'MI': 'Critical - Immediate',
            'STTC': 'High - Urgent',
            'CD': 'Moderate - Monitor',
            'HYP': 'Moderate - Follow-up',
            'NORM': 'Standard'
        }
        
        base_risk = risk_mapping.get(condition, 'Moderate')
        base_priority = priority_mapping.get(condition, 'Standard')
        
        # Adjust based on confidence
        if confidence < 0.6:
            # Low confidence - recommend review
            if base_risk == 'Critical':
                priority = 'Critical - Requires Review'
            else:
                priority = f"{base_priority} - Review Recommended"
        else:
            priority = base_priority
        
        return {
            'risk_level': base_risk,
            'clinical_priority': priority
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        if self.model is None:
            return {'status': 'No model loaded'}
        
        info = self.model_info.copy()
        info.update({
            'model_type': self.model_type,
            'target_conditions': TARGET_CONDITIONS,
            'confidence_thresholds': self.confidence_thresholds,
            'has_feature_transformers': self.feature_transformers is not None,
            'has_label_encoder': self.label_encoder is not None,
            'feature_count': len(self.feature_names) if self.feature_names else 'Unknown'
        })
        
        return info


def display_classification_interface():
    """
    Streamlit interface for ECG classification
    """
    st.header("🫀 ECG Classification")
    
    # Model selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_type = st.selectbox(
            "Select Model Type",
            ["combined_mi_enhanced", "ptbxl_only", "auto"],
            help="Combined MI Enhanced provides better MI detection"
        )
    
    with col2:
        if st.button("🔄 Reload Model", help="Reload the classification model"):
            if 'classifier' in st.session_state:
                del st.session_state.classifier
    
    # Initialize classifier
    if 'classifier' not in st.session_state or st.session_state.get('model_type') != model_type:
        with st.spinner("Loading classification model..."):
            st.session_state.classifier = ECGClassifier(model_type=model_type)
            st.session_state.model_type = model_type
            
            if st.session_state.classifier.load_model():
                st.success(f"✅ Model loaded successfully!")
                
                # Display model info
                model_info = st.session_state.classifier.get_model_info()
                
                with st.expander("📊 Model Information", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Model Type", model_info.get('model_type', 'Unknown'))
                        st.metric("Features", model_info.get('feature_count', 'Unknown'))
                    
                    with col2:
                        st.metric("MI Enhanced", "Yes" if model_info.get('has_mi_enhancement') else "No")
                        st.metric("Loaded At", model_info.get('loaded_at', 'Unknown'))
                    
                    with col3:
                        if model_info.get('has_mi_enhancement'):
                            st.success("🎯 Enhanced MI Detection Active")
                        else:
                            st.warning("⚠️ Standard Model (Limited MI Detection)")
            else:
                st.error("❌ Failed to load model. Please check if models are available.")
                return
    
    classifier = st.session_state.classifier
    
    # Classification interface
    st.subheader("📈 Real-time Classification")
    
    # Input method selection
    input_method = st.radio(
        "Input Method",
        ["Upload ECG File", "Generate Synthetic ECG", "Manual Feature Input"],
        horizontal=True
    )
    
    if input_method == "Upload ECG File":
        uploaded_file = st.file_uploader(
            "Upload ECG Data",
            type=['csv', 'txt', 'npy'],
            help="Upload preprocessed ECG signal or feature data"
        )
        
        if uploaded_file:
            # Process uploaded file
            try:
                if uploaded_file.name.endswith('.npy'):
                    data = np.load(uploaded_file)
                else:
                    data = pd.read_csv(uploaded_file).values
                
                st.success(f"✅ Data loaded: {data.shape}")
                
                if st.button("🔍 Classify ECG"):
                    with st.spinner("Analyzing ECG..."):
                        result = classifier.predict(data)
                        display_classification_results(result)
                        
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
    
    elif input_method == "Generate Synthetic ECG":
        st.info("🧪 Synthetic ECG generation for testing")
        
        col1, col2 = st.columns(2)
        with col1:
            condition = st.selectbox("Target Condition", TARGET_CONDITIONS)
        with col2:
            noise_level = st.slider("Noise Level", 0.0, 0.5, 0.1)
        
        if st.button("🎲 Generate & Classify"):
            with st.spinner("Generating synthetic ECG..."):
                # Generate synthetic data (placeholder for now)
                synthetic_data = generate_synthetic_ecg(condition, noise_level)
                result = classifier.predict(synthetic_data)
                
                st.success("✅ Synthetic ECG generated and classified")
                display_classification_results(result)
    
    elif input_method == "Manual Feature Input":
        st.info("🔧 Manual feature input for testing")
        
        # Create input fields for key features
        with st.form("manual_features"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Temporal Features")
                hr = st.number_input("Heart Rate (bpm)", 50, 200, 75)
                rr_std = st.number_input("RR Std Dev", 0.0, 1.0, 0.1)
                
            with col2:
                st.subheader("Frequency Features")
                lf_power = st.number_input("LF Power", 0.0, 1.0, 0.3)
                hf_power = st.number_input("HF Power", 0.0, 1.0, 0.4)
                
            with col3:
                st.subheader("Morphology Features")
                qt_interval = st.number_input("QT Interval (ms)", 300, 500, 400)
                st_elevation = st.number_input("ST Elevation", -0.5, 0.5, 0.0)
            
            if st.form_submit_button("🔍 Classify Features"):
                # Create feature vector
                features = np.array([[hr, rr_std, lf_power, hf_power, qt_interval, st_elevation]])
                
                # Pad or truncate to expected feature count if needed
                expected_features = classifier.feature_names
                if expected_features and len(expected_features) != features.shape[1]:
                    # Pad with zeros or use mean values
                    if features.shape[1] < len(expected_features):
                        padding = np.zeros((1, len(expected_features) - features.shape[1]))
                        features = np.hstack([features, padding])
                    else:
                        features = features[:, :len(expected_features)]
                
                result = classifier.predict(features)
                display_classification_results(result)


def display_classification_results(result: Dict[str, Any]):
    """Display classification results in a clinical format"""
    
    if 'error' in result:
        st.error(f"❌ Classification Error: {result['error']}")
        return
    
    # Main prediction display
    prediction = result['prediction']
    confidence = result['confidence_score']
    risk_level = result['risk_level']
    
    # Color coding based on condition severity
    color_map = {
        'MI': '🔴',
        'STTC': '🟡', 
        'CD': '🟠',
        'HYP': '🟠',
        'NORM': '🟢'
    }
    
    icon = color_map.get(prediction, '⚪')
    
    st.markdown("---")
    st.subheader("🎯 Classification Results")
    
    # Primary result
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🫀 Predicted Condition",
            f"{icon} {prediction}",
            help=f"Confidence: {confidence:.1%}"
        )
    
    with col2:
        confidence_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
        st.metric(
            "🎯 Confidence",
            f"{confidence_color} {confidence:.1%}",
            help=result.get('confidence_assessment', 'Unknown')
        )
    
    with col3:
        risk_color = "🔴" if risk_level == "Critical" else "🟡" if risk_level == "High" else "🟢"
        st.metric(
            "⚠️ Risk Level", 
            f"{risk_color} {risk_level}",
            help=result.get('clinical_priority', 'Standard')
        )
    
    # Clinical interpretation
    if prediction == 'MI':
        st.error(f"🚨 **CRITICAL**: Myocardial Infarction detected (Confidence: {confidence:.1%})")
        st.error("⚡ **IMMEDIATE ACTION REQUIRED**: Contact emergency services")
    elif prediction == 'STTC':
        st.warning(f"⚠️ **HIGH PRIORITY**: ST/T Changes detected (Confidence: {confidence:.1%})")
        st.warning("📞 **URGENT**: Cardiology consultation recommended")
    elif prediction in ['CD', 'HYP']:
        st.info(f"📊 **MODERATE**: {prediction} detected (Confidence: {confidence:.1%})")
        st.info("📋 **FOLLOW-UP**: Monitor and schedule appropriate care")
    else:  # NORM
        st.success(f"✅ **NORMAL**: No significant abnormalities detected (Confidence: {confidence:.1%})")
    
    # Detailed probabilities
    if 'probabilities' in result and result['probabilities']:
        with st.expander("📊 Detailed Probability Breakdown", expanded=False):
            probs_df = pd.DataFrame([
                {
                    'Condition': condition,
                    'Probability': f"{prob:.1%}",
                    'Risk Level': 'Critical' if condition == 'MI' else 'High' if condition == 'STTC' else 'Moderate' if condition in ['CD', 'HYP'] else 'Low'
                }
                for condition, prob in result['probabilities'].items()
            ]).sort_values('Probability', ascending=False)
            
            st.dataframe(probs_df, use_container_width=True)
    
    # Performance metrics
    if 'prediction_time_ms' in result:
        st.caption(f"⏱️ Analysis completed in {result['prediction_time_ms']:.1f}ms")


def generate_synthetic_ecg(condition: str = 'NORM', noise_level: float = 0.1) -> np.ndarray:
    """Generate synthetic ECG features for testing"""
    np.random.seed(42)
    
    # Base feature patterns for different conditions
    feature_patterns = {
        'NORM': [75, 0.05, 0.3, 0.4, 400, 0.0, 0.5, 0.8],
        'MI': [85, 0.15, 0.2, 0.3, 420, 0.2, 0.3, 0.6],
        'STTC': [80, 0.10, 0.25, 0.35, 410, 0.1, 0.4, 0.7],
        'CD': [90, 0.20, 0.35, 0.25, 450, 0.0, 0.6, 0.5],
        'HYP': [70, 0.08, 0.4, 0.5, 380, -0.1, 0.7, 0.9]
    }
    
    base_features = feature_patterns.get(condition, feature_patterns['NORM'])
    
    # Add noise
    features = np.array(base_features) + np.random.normal(0, noise_level, len(base_features))
    
    # Extend to common feature count (50 features)
    while len(features) < 50:
        features = np.append(features, np.random.normal(0.5, 0.1))
    
    return features.reshape(1, -1)


if __name__ == "__main__":
    display_classification_interface()