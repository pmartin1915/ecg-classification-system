"""
Model Integration Utilities for Enhanced MI Detection
Handles transition from Phase 4 training to Phase 5 clinical deployment
"""
import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import shutil

from config.settings import MODELS_DIR, DATA_DIR
from app.utils.dataset_manager import run_combined_dataset_loading


class ModelIntegrationManager:
    """
    Manages integration of newly trained MI-enhanced models into the clinical app
    """
    
    def __init__(self):
        self.models_dir = MODELS_DIR / "trained_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def save_phase4_model_for_deployment(self, 
                                       training_results: Dict[str, Any],
                                       model_name: str = "combined_mi_enhanced_model",
                                       include_metadata: bool = True) -> str:
        """
        Save Phase 4 training results in deployment-ready format
        
        Args:
            training_results: Results from Phase 4 training pipeline
            model_name: Name for the saved model
            include_metadata: Whether to include training metadata
            
        Returns:
            str: Path to saved model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_path = self.models_dir / model_filename
        
        # Extract components from training results
        best_model = training_results.get('best_model')
        model_name_from_results = training_results.get('best_model_name', 'unknown')
        
        # Create comprehensive model package
        model_package = {
            'model': best_model,
            'model_name': model_name_from_results,
            'feature_transformers': training_results.get('feature_transformers'),
            'label_encoder': training_results.get('label_encoder'),
            'feature_names': training_results.get('feature_names', []),
            'model_info': {
                'training_date': timestamp,
                'model_type': 'combined_mi_enhanced',
                'dataset_source': 'PTB-XL + ECG_Arrhythmia',
                'mi_enhancement': True,
                'phase': 'Phase4_to_Phase5_Integration'
            }
        }
        
        if include_metadata:
            model_package['training_metadata'] = {
                'test_results': training_results.get('test_results', {}),
                'validation_results': training_results.get('validation_results', {}),
                'training_config': training_results.get('config', {}),
                'feature_importance': training_results.get('feature_importance', {}),
                'dataset_stats': training_results.get('dataset_stats', {})
            }
        
        # Save the model package
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        # Save auxiliary files for easier loading
        self._save_auxiliary_files(model_package, model_path.stem)
        
        return str(model_path)
    
    def _save_auxiliary_files(self, model_package: Dict[str, Any], base_name: str):
        """Save auxiliary model files separately"""
        
        # Save feature transformers
        if model_package.get('feature_transformers'):
            transformer_path = self.models_dir / f"{base_name}_feature_transformers.pkl"
            with open(transformer_path, 'wb') as f:
                pickle.dump(model_package['feature_transformers'], f)
        
        # Save label encoder
        if model_package.get('label_encoder'):
            encoder_path = self.models_dir / f"{base_name}_label_encoder.pkl"
            with open(encoder_path, 'wb') as f:
                pickle.dump(model_package['label_encoder'], f)
        
        # Save feature names
        if model_package.get('feature_names'):
            features_path = self.models_dir / f"{base_name}_feature_names.txt"
            with open(features_path, 'w') as f:
                for feature_name in model_package['feature_names']:
                    f.write(f"{feature_name}\n")
    
    def validate_model_performance(self, model_path: str, 
                                 min_mi_sensitivity: float = 0.5,
                                 min_overall_accuracy: float = 0.7) -> Dict[str, Any]:
        """
        Validate that the model meets clinical performance requirements
        
        Args:
            model_path: Path to the model to validate
            min_mi_sensitivity: Minimum required MI sensitivity
            min_overall_accuracy: Minimum required overall accuracy
            
        Returns:
            Dict with validation results
        """
        try:
            # Load model
            with open(model_path, 'rb') as f:
                model_package = pickle.load(f)
            
            # Extract performance metrics from training metadata
            training_metadata = model_package.get('training_metadata', {})
            test_results = training_metadata.get('test_results', {})
            test_metrics = test_results.get('metrics', {})
            
            # Check overall metrics
            overall_accuracy = test_metrics.get('accuracy', 0.0)
            overall_f1 = test_metrics.get('f1_weighted', 0.0)
            
            # Look for MI-specific metrics
            mi_sensitivity = test_metrics.get('mi_sensitivity', 0.0)
            
            # Alternative: extract from classification report
            if mi_sensitivity == 0.0 and 'classification_report' in test_metrics:
                report = test_metrics['classification_report']
                if isinstance(report, dict) and 'MI' in report:
                    mi_sensitivity = report['MI'].get('recall', 0.0)
            
            # Validation checks
            validation_results = {
                'overall_accuracy': overall_accuracy,
                'overall_f1': overall_f1,
                'mi_sensitivity': mi_sensitivity,
                'meets_accuracy_requirement': overall_accuracy >= min_overall_accuracy,
                'meets_mi_sensitivity_requirement': mi_sensitivity >= min_mi_sensitivity,
                'validation_passed': False,
                'recommendations': []
            }
            
            # Overall validation
            if (validation_results['meets_accuracy_requirement'] and 
                validation_results['meets_mi_sensitivity_requirement']):
                validation_results['validation_passed'] = True
                validation_results['recommendations'].append("✅ Model meets all clinical requirements")
            else:
                if not validation_results['meets_accuracy_requirement']:
                    validation_results['recommendations'].append(
                        f"⚠️ Overall accuracy ({overall_accuracy:.3f}) below threshold ({min_overall_accuracy})"
                    )
                if not validation_results['meets_mi_sensitivity_requirement']:
                    validation_results['recommendations'].append(
                        f"🚨 MI sensitivity ({mi_sensitivity:.3f}) below threshold ({min_mi_sensitivity})"
                    )
                    validation_results['recommendations'].append(
                        "💡 Consider training with more MI samples from ECG Arrhythmia dataset"
                    )
            
            return validation_results
            
        except Exception as e:
            return {
                'validation_passed': False,
                'error': str(e),
                'recommendations': ['❌ Model validation failed - check model file integrity']
            }
    
    def deploy_model_to_production(self, model_path: str, 
                                 backup_existing: bool = True) -> Dict[str, Any]:
        """
        Deploy a validated model to production (make it the default)
        
        Args:
            model_path: Path to the model to deploy
            backup_existing: Whether to backup the existing production model
            
        Returns:
            Dict with deployment results
        """
        model_path = Path(model_path)
        production_model_path = self.models_dir / "production_model.pkl"
        
        try:
            # Backup existing production model if it exists
            if backup_existing and production_model_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.models_dir / f"production_model_backup_{timestamp}.pkl"
                shutil.copy2(production_model_path, backup_path)
            
            # Copy new model to production
            shutil.copy2(model_path, production_model_path)
            
            # Create deployment metadata
            deployment_info = {
                'deployed_at': datetime.now().isoformat(),
                'source_model': str(model_path),
                'production_model': str(production_model_path),
                'deployment_successful': True
            }
            
            # Save deployment info
            deployment_info_path = self.models_dir / "deployment_info.json"
            import json
            with open(deployment_info_path, 'w') as f:
                json.dump(deployment_info, f, indent=2)
            
            return deployment_info
            
        except Exception as e:
            return {
                'deployment_successful': False,
                'error': str(e)
            }
    
    def create_model_performance_report(self, model_path: str) -> Dict[str, Any]:
        """Create a comprehensive performance report for the model"""
        try:
            with open(model_path, 'rb') as f:
                model_package = pickle.load(f)
            
            training_metadata = model_package.get('training_metadata', {})
            model_info = model_package.get('model_info', {})
            
            # Extract key metrics
            test_results = training_metadata.get('test_results', {})
            test_metrics = test_results.get('metrics', {})
            
            report = {
                'model_summary': {
                    'model_type': model_info.get('model_type', 'Unknown'),
                    'training_date': model_info.get('training_date', 'Unknown'),
                    'dataset_source': model_info.get('dataset_source', 'Unknown'),
                    'mi_enhancement': model_info.get('mi_enhancement', False)
                },
                'performance_metrics': {
                    'overall_accuracy': test_metrics.get('accuracy', 0.0),
                    'overall_f1_score': test_metrics.get('f1_weighted', 0.0),
                    'overall_precision': test_metrics.get('precision_weighted', 0.0),
                    'overall_recall': test_metrics.get('recall_weighted', 0.0)
                },
                'clinical_metrics': {
                    'mi_sensitivity': test_metrics.get('mi_sensitivity', 0.0),
                    'norm_specificity': test_metrics.get('norm_specificity', 0.0)
                },
                'feature_info': {
                    'feature_count': len(model_package.get('feature_names', [])),
                    'has_feature_transformers': model_package.get('feature_transformers') is not None
                }
            }
            
            # Add clinical interpretation
            mi_sensitivity = report['clinical_metrics']['mi_sensitivity']
            if mi_sensitivity > 0.8:
                report['clinical_assessment'] = "🟢 Excellent MI detection capability"
            elif mi_sensitivity > 0.6:
                report['clinical_assessment'] = "🟡 Good MI detection capability"
            elif mi_sensitivity > 0.3:
                report['clinical_assessment'] = "🟠 Moderate MI detection capability"
            else:
                report['clinical_assessment'] = "🔴 Poor MI detection - needs improvement"
            
            return report
            
        except Exception as e:
            return {'error': str(e)}


def quick_train_and_deploy_mi_model(ptbxl_records: int = 2000,
                                  arrhythmia_records: int = 1000,
                                  target_mi_records: int = 500) -> Dict[str, Any]:
    """
    Quick training and deployment of MI-enhanced model
    
    Args:
        ptbxl_records: Number of PTB-XL records to use
        arrhythmia_records: Number of ECG Arrhythmia records to use
        target_mi_records: Target number of MI records
        
    Returns:
        Dict with training and deployment results
    """
    try:
        # Load combined dataset
        print("Loading combined dataset...")
        X, labels, ids, metadata, target_conditions, stats = run_combined_dataset_loading(
            ptbxl_max_records=ptbxl_records,
            arrhythmia_max_records=arrhythmia_records,
            target_mi_records=target_mi_records,
            sampling_rate=100
        )
        
        if len(X) == 0:
            return {'error': 'No data loaded for training'}
        
        # Run preprocessing
        print("Running preprocessing...")
        from models.preprocessing.preprocessing_pipeline import PreprocessingPipeline
        preprocessing = PreprocessingPipeline()
        prep_results = preprocessing.run(X, labels, max_records=len(X))
        
        # Run feature extraction
        print("Running feature extraction...")
        from models.feature_extraction.feature_extraction_pipeline import FeatureExtractionPipeline
        feature_pipeline = FeatureExtractionPipeline()
        feature_results = feature_pipeline.run(
            prep_results['X_preprocessed'],
            prep_results['y_encoded'],
            prep_results['label_encoder']
        )
        
        # Run training
        print("Running model training...")
        from models.training.phase4_model_training import Phase4Pipeline
        
        training_config = {
            'test_size': 0.2,
            'val_size': 0.2,
            'use_smote': True,
            'use_hyperparameter_tuning': True,
            'model_keys': ['random_forest', 'logistic_regression', 'gradient_boosting'],
            'create_visualizations': True,
            'save_models': False  # We'll handle saving
        }
        
        pipeline = Phase4Pipeline(training_config)
        training_results = pipeline.run(
            X=feature_results['X_features'],
            y=feature_results['y_encoded'],
            feature_names=feature_results['feature_names']
        )
        
        # Integrate model for deployment
        print("Integrating model for deployment...")
        integration_manager = ModelIntegrationManager()
        
        # Add necessary components to training results
        training_results['feature_transformers'] = feature_results.get('feature_transformers')
        training_results['label_encoder'] = prep_results['label_encoder']
        training_results['feature_names'] = feature_results['feature_names']
        
        # Save model
        model_path = integration_manager.save_phase4_model_for_deployment(
            training_results,
            model_name="quick_combined_mi_enhanced"
        )
        
        # Validate model
        validation_results = integration_manager.validate_model_performance(model_path)
        
        # Deploy if validation passes
        deployment_results = {}
        if validation_results['validation_passed']:
            deployment_results = integration_manager.deploy_model_to_production(model_path)
        
        return {
            'training_successful': True,
            'model_path': model_path,
            'validation_results': validation_results,
            'deployment_results': deployment_results,
            'dataset_stats': stats,
            'mi_improvement': f"MI records: {stats.get('mi_records', 0)} (vs previous: 0)"
        }
        
    except Exception as e:
        import traceback
        return {
            'training_successful': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def display_model_integration_interface():
    """Streamlit interface for model integration"""
    st.header("🔧 Model Integration & Deployment")
    
    st.info("💡 This interface helps integrate your trained MI-enhanced models into the clinical app")
    
    # Model training section
    with st.expander("🚀 Quick Train & Deploy", expanded=False):
        st.subheader("Train New MI-Enhanced Model")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            ptbxl_records = st.number_input("PTB-XL Records", 100, 10000, 2000)
        with col2:
            arrhythmia_records = st.number_input("ECG Arrhythmia Records", 50, 5000, 1000)
        with col3:
            target_mi_records = st.number_input("Target MI Records", 50, 2000, 500)
        
        if st.button("🔥 Train & Deploy Model"):
            with st.spinner("Training MI-enhanced model... This may take several minutes."):
                results = quick_train_and_deploy_mi_model(
                    ptbxl_records, arrhythmia_records, target_mi_records
                )
                
                if results.get('training_successful'):
                    st.success("✅ Model training completed!")
                    
                    # Show validation results
                    validation = results['validation_results']
                    if validation['validation_passed']:
                        st.success("🎯 Model validation PASSED - ready for clinical use!")
                    else:
                        st.warning("⚠️ Model validation issues detected")
                        for rec in validation['recommendations']:
                            st.write(rec)
                    
                    # Show deployment results
                    deployment = results.get('deployment_results', {})
                    if deployment.get('deployment_successful'):
                        st.success("🚀 Model deployed to production!")
                    
                    # Show improvement stats
                    st.info(results['mi_improvement'])
                    
                else:
                    st.error(f"❌ Training failed: {results.get('error', 'Unknown error')}")
    
    # Model management section
    st.subheader("📊 Model Management")
    
    # List available models
    models_dir = MODELS_DIR / "trained_models"
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pkl"))
        
        if model_files:
            selected_model = st.selectbox(
                "Select Model",
                model_files,
                format_func=lambda x: x.name
            )
            
            if selected_model:
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📋 Generate Performance Report"):
                        integration_manager = ModelIntegrationManager()
                        report = integration_manager.create_model_performance_report(str(selected_model))
                        
                        if 'error' not in report:
                            st.json(report)
                        else:
                            st.error(f"Error generating report: {report['error']}")
                
                with col2:
                    if st.button("🚀 Deploy to Production"):
                        integration_manager = ModelIntegrationManager()
                        
                        # Validate first
                        validation = integration_manager.validate_model_performance(str(selected_model))
                        
                        if validation['validation_passed']:
                            deployment = integration_manager.deploy_model_to_production(str(selected_model))
                            
                            if deployment['deployment_successful']:
                                st.success("✅ Model deployed to production!")
                            else:
                                st.error(f"❌ Deployment failed: {deployment.get('error')}")
                        else:
                            st.error("❌ Model failed validation - cannot deploy")
                            for rec in validation['recommendations']:
                                st.write(rec)
        else:
            st.info("📁 No trained models found. Train a model first.")
    else:
        st.info("📁 Models directory not found. Train a model first.")


if __name__ == "__main__":
    display_model_integration_interface()