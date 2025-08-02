"""
Complete training pipeline for ECG classification
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging

from config.model_config import ModelTrainingConfig, MODEL_CONFIGS
from config.settings import DATA_DIR, MODELS_DIR
from models.training.model_trainer import ModelTrainer
from models.training.model_evaluation import ModelEvaluator


class TrainingPipeline:
    """Complete training pipeline with evaluation and visualization"""
    
    def __init__(self, config: Optional[ModelTrainingConfig] = None):
        self.config = config or ModelTrainingConfig()
        self.trainer = ModelTrainer(self.config)
        self.evaluator = ModelEvaluator(save_dir=DATA_DIR / 'visualizations' / 'models')
        self.results = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def run(self, 
            X: np.ndarray,
            y: np.ndarray,
            feature_names: List[str],
            model_keys: Optional[List[str]] = None,
            visualize: bool = True) -> Dict[str, Any]:
        """
        Run complete training pipeline
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            model_keys: Models to train (None for all)
            visualize: Whether to create visualizations
            
        Returns:
            Dictionary with all training results
        """
        self.logger.info("STARTING MODEL TRAINING PIPELINE")
        self.logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Step 1: Load data
        self.logger.info("\nStep 1: Loading data")
        self.trainer.load_data(X, y, feature_names)
        
        # Step 2: Prepare data splits
        self.logger.info("\nStep 2: Preparing data splits")
        self.trainer.prepare_data_splits()
        
        # Step 3: Scale features
        self.logger.info("\nStep 3: Scaling features")
        self.trainer.scale_features()
        
        # Step 4: Setup models
        self.logger.info("\nStep 4: Setting up models")
        if model_keys is None:
            # Default models to train
            model_keys = ['random_forest', 'gradient_boosting', 'logistic_regression', 
                         'svm', 'xgboost'] if 'xgboost' in MODEL_CONFIGS else \
                        ['random_forest', 'gradient_boosting', 'logistic_regression', 'svm']
        
        self.trainer.setup_models(model_keys)
        
        # Step 5: Train models
        self.logger.info("\nStep 5: Training models")
        self.trainer.train_all_models()
        
        # Step 6: Evaluate models
        self.logger.info("\nStep 6: Evaluating models")
        comparison_df = self.trainer.evaluate_all_models()
        
        # Step 7: Cross-validation for best model
        self.logger.info("\nStep 7: Cross-validating best model")
        cv_results = self.trainer.cross_validate_model(self.trainer.best_model_name)
        
        # Step 8: Create ensemble
        self.logger.info("\nStep 8: Creating ensemble model")
        try:
            base_models = comparison_df.head(3)['Model'].tolist()
            self.trainer.create_ensemble(base_models, voting='soft')
            
            # Re-evaluate with ensemble
            comparison_df = self.trainer.evaluate_all_models()
        except Exception as e:
            self.logger.warning(f"Could not create ensemble: {e}")
        
        # Step 9: Test best model
        self.logger.info("\nStep 9: Testing best model")
        test_results = self.trainer.test_model()
        
        # Step 10: Create visualizations
        if visualize and self.config.create_visualizations:
            self.logger.info("\nStep 10: Creating visualizations")
            self._create_visualizations()
        
        # Step 11: Save models and results
        self.logger.info("\nStep 11: Saving models and results")
        self._save_results()
        
        # Compile results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.results = {
            'trainer': self.trainer,
            'comparison_df': comparison_df,
            'best_model_name': self.trainer.best_model_name,
            'best_model': self.trainer.best_model,
            'test_results': test_results,
            'cv_results': cv_results,
            'training_duration': duration,
            'config': self.config
        }
        
        self._print_summary()
        
        return self.results
    
    def _create_visualizations(self):
        """Create all visualizations"""
        # Get best model results
        best_results = self.trainer.results[self.trainer.best_model_name]
        test_results = self.trainer.test_results
        
        # Class distribution
        if self.config.plot_confusion_matrix:
            self.evaluator.plot_class_distribution(
                self.trainer.y, 
                self.config.class_names
            )
        
        # Model comparison
        self.evaluator.plot_model_comparison(self.trainer.comparison_df)
        
        # Confusion matrix
        if self.config.plot_confusion_matrix:
            self.evaluator.plot_confusion_matrix(
                test_results['predictions'],
                self.trainer.y_test,
                list(self.config.class_names.values()),
                self.trainer.best_model_name
            )
        
        # ROC curves
        if self.config.plot_roc_curves and test_results['probabilities'] is not None:
            self.evaluator.plot_roc_curves(
                self.trainer.y_test,
                test_results['probabilities'],
                list(self.config.class_names.values()),
                self.trainer.best_model_name
            )
            
            # Precision-Recall curves
            self.evaluator.plot_precision_recall_curves(
                self.trainer.y_test,
                test_results['probabilities'],
                list(self.config.class_names.values()),
                self.trainer.best_model_name
            )
        
        # Feature importance
        if self.config.plot_feature_importance:
            feature_importance = best_results.get('feature_importance')
            if feature_importance is not None:
                self.evaluator.plot_feature_importance(
                    feature_importance,
                    self.trainer.feature_names,
                    self.trainer.best_model_name
                )
        
        # Clinical metrics
        if self.config.clinical_metrics:
            self.evaluator.plot_clinical_metrics(
                test_results['metrics'],
                self.config.class_names
            )
        
        # Learning curves
        if self.config.plot_learning_curves:
            try:
                self.evaluator.plot_learning_curves(
                    self.trainer.best_model,
                    self.trainer.X_train_scaled,
                    self.trainer.y_train,
                    self.trainer.best_model_name
                )
            except Exception as e:
                self.logger.warning(f"Could not plot learning curves: {e}")
    
    def _save_results(self):
        """Save models and results"""
        save_dir = MODELS_DIR / 'trained_models'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best model
        if self.config.save_best_only:
            model_path = self.trainer.save_model(
                self.trainer.best_model,
                self.trainer.best_model_name,
                save_dir
            )
            self.logger.info(f"Best model saved: {model_path}")
        
        # Save all models if requested
        if self.config.save_all_models:
            for model_name, results in self.trainer.results.items():
                if model_name != self.trainer.best_model_name:
                    self.trainer.save_model(
                        results['model'],
                        model_name,
                        save_dir
                    )
        
        # Save trainer results
        self.trainer.save_results(save_dir)
        
        # Save pipeline results
        pipeline_results = {
            'best_model_name': self.trainer.best_model_name,
            'comparison_df': self.trainer.comparison_df,
            'test_results': self.trainer.test_results,
            'config': self.config,
            'training_date': datetime.now().isoformat()
        }
        
        results_path = save_dir / 'pipeline_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(pipeline_results, f)
        
        # Create evaluation report
        if hasattr(self.trainer, 'test_results'):
            report = self.evaluator.create_comprehensive_report(
                self.trainer.results[self.trainer.best_model_name],
                self.trainer.test_results,
                self.trainer.best_model_name
            )
            self.logger.info(f"\nEvaluation report saved")
    
    def _print_summary(self):
        """Print training summary"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("MODEL TRAINING COMPLETE!")
        self.logger.info("=" * 80)
        
        # Best model info
        test_metrics = self.trainer.test_results['metrics']
        self.logger.info(f"\nBest Model: {self.trainer.best_model_name}")
        self.logger.info(f"   Test Accuracy:  {test_metrics['accuracy']:.4f}")
        self.logger.info(f"   Test F1-Score:  {test_metrics['f1_weighted']:.4f}")
        self.logger.info(f"   Test Precision: {test_metrics['precision_weighted']:.4f}")
        self.logger.info(f"   Test Recall:    {test_metrics['recall_weighted']:.4f}")
        
        # Clinical metrics
        if self.config.clinical_metrics:
            self.logger.info(f"\nClinical Performance:")
            self.logger.info(f"   MI Sensitivity:   {test_metrics.get('MI_sensitivity', 0):.4f}")
            self.logger.info(f"   NORM Specificity: {test_metrics.get('NORM_specificity', 0):.4f}")
        
        # Training info
        self.logger.info(f"\nTotal training time: {self.results['training_duration']:.2f} seconds")
        self.logger.info(f"Models trained: {len(self.trainer.results)}")
        
        self.logger.info("\nReady for Phase 5: Deployment!")


# Convenience function for backwards compatibility
def run_phase4_model_training(X_features, y_encoded, feature_names, 
                            class_names=None, drive_path=None):
    """Backwards compatible function for Phase 4"""
    
    # Create config
    config = ModelTrainingConfig()
    if class_names:
        config.class_names = class_names
    
    # Run pipeline
    pipeline = TrainingPipeline(config)
    results = pipeline.run(
        X=X_features,
        y=y_encoded,
        feature_names=feature_names
    )
    
    return results