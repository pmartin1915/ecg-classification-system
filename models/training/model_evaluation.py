"""
Model evaluation and visualization utilities
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""
    
    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = Path(save_dir) if save_dir else Path('visualizations')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        self.colors = plt.cm.Set3(np.linspace(0, 1, 10))
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: List[str], model_name: str,
                            normalize: bool = True) -> None:
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = f'Normalized Confusion Matrix - {model_name}'
        else:
            fmt = 'd'
            title = f'Confusion Matrix - {model_name}'
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   square=True, cbar_kws={'shrink': 0.8})
        
        plt.title(title, fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add accuracy for each class
        accuracies = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(accuracies):
            plt.text(-0.5, i + 0.5, f'{acc:.2f}', ha='right', va='center')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png',
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, y_true: np.ndarray, y_proba: np.ndarray,
                       class_names: List[str], model_name: str) -> None:
        """Plot ROC curves for multi-class classification"""
        n_classes = len(class_names)
        
        # Binarize the output
        y_true_bin = np.zeros((len(y_true), n_classes))
        for i, label in enumerate(y_true):
            y_true_bin[i, label] = 1
        
        # Calculate ROC curve for each class
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        roc_auc_dict = {}
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            roc_auc_dict[class_names[i]] = roc_auc
            
            ax = axes[i]
            ax.plot(fpr, tpr, color=self.colors[i], lw=2,
                   label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {class_names[i]}')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
        
        # Overall plot
        ax = axes[n_classes]
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            ax.plot(fpr, tpr, color=self.colors[i], lw=2,
                   label=f'{class_names[i]} (AUC = {roc_auc_dict[class_names[i]]:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves - All Classes - {model_name}')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        # Hide unused subplot
        if n_classes < len(axes) - 1:
            for i in range(n_classes + 1, len(axes)):
                axes[i].axis('off')
        
        plt.suptitle(f'ROC Analysis - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / f'roc_curves_{model_name.lower().replace(" ", "_")}.png',
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(self, y_true: np.ndarray, y_proba: np.ndarray,
                                   class_names: List[str], model_name: str) -> None:
        """Plot precision-recall curves"""
        n_classes = len(class_names)
        
        # Binarize the output
        y_true_bin = np.zeros((len(y_true), n_classes))
        for i, label in enumerate(y_true):
            y_true_bin[i, label] = 1
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
            avg_precision = average_precision_score(y_true_bin[:, i], y_proba[:, i])
            
            ax.plot(recall, precision, color=self.colors[i], lw=2,
                   label=f'{class_names[i]} (AP = {avg_precision:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curves - {model_name}', fontsize=16)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'pr_curves_{model_name.lower().replace(" ", "_")}.png',
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_importance: np.ndarray,
                              feature_names: List[str], model_name: str,
                              top_n: int = 20) -> None:
        """Plot feature importance"""
        if feature_importance is None:
            return
        
        # Sort features by importance
        indices = np.argsort(feature_importance)[-top_n:]
        top_features = [feature_names[i] for i in indices]
        top_importance = feature_importance[indices]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_importance, color='steelblue', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances - {model_name}', fontsize=16)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(top_importance):
            ax.text(v + 0.001, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'feature_importance_{model_name.lower().replace(" ", "_")}.png',
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame) -> None:
        """Plot model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Metrics comparison
        ax = axes[0, 0]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(comparison_df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, comparison_df[metric], width,
                  label=metric, alpha=0.8)
        
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14)
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.05)
        
        # Training time
        ax = axes[0, 1]
        ax.bar(comparison_df['Model'], comparison_df['Training Time'], 
               color='coral', alpha=0.8)
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Training Time (seconds)', fontsize=12)
        ax.set_title('Model Training Times', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # F1-Score vs Training Time
        ax = axes[1, 0]
        scatter = ax.scatter(comparison_df['Training Time'], comparison_df['F1-Score'],
                           s=100, alpha=0.7, c=range(len(comparison_df)),
                           cmap='viridis')
        
        for idx, row in comparison_df.iterrows():
            ax.annotate(row['Model'], (row['Training Time'], row['F1-Score']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Training Time (seconds)', fontsize=12)
        ax.set_ylabel('F1-Score', fontsize=12)
        ax.set_title('F1-Score vs Training Time', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Additional metrics
        ax = axes[1, 1]
        if 'Cohen Kappa' in comparison_df.columns and 'Matthews Corr' in comparison_df.columns:
            x = np.arange(len(comparison_df))
            width = 0.35
            
            ax.bar(x - width/2, comparison_df['Cohen Kappa'], width,
                  label='Cohen Kappa', alpha=0.8)
            ax.bar(x + width/2, comparison_df['Matthews Corr'], width,
                  label='Matthews Correlation', alpha=0.8)
            
            ax.set_xlabel('Models', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('Additional Metrics Comparison', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.axis('off')
        
        plt.suptitle('Model Performance Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curves(self, model: Any, X: np.ndarray, y: np.ndarray,
                           model_name: str, cv: int = 5) -> None:
        """Plot learning curves"""
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, train_sizes=train_sizes,
            scoring='f1_weighted', n_jobs=-1, random_state=42
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                       train_scores_mean + train_scores_std, alpha=0.1, color='blue')
        ax.fill_between(train_sizes_abs, val_scores_mean - val_scores_std,
                       val_scores_mean + val_scores_std, alpha=0.1, color='orange')
        
        ax.plot(train_sizes_abs, train_scores_mean, 'o-', color='blue',
               label='Training score', lw=2)
        ax.plot(train_sizes_abs, val_scores_mean, 'o-', color='orange',
               label='Cross-validation score', lw=2)
        
        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title(f'Learning Curves - {model_name}', fontsize=16)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'learning_curves_{model_name.lower().replace(" ", "_")}.png',
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_distribution(self, y: np.ndarray, class_names: Dict[int, str]) -> None:
        """Plot class distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Count plot
        class_counts = pd.Series(y).value_counts().sort_index()
        class_labels = [class_names[i] for i in class_counts.index]
        
        bars = ax1.bar(class_labels, class_counts.values, alpha=0.8)
        
        # Color bars
        for bar, color in zip(bars, self.colors):
            bar.set_color(color)
        
        ax1.set_xlabel('Class', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Class Distribution', fontsize=14)
        
        # Add value labels
        for bar, count in zip(bars, class_counts.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}', ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(class_counts.values, labels=class_labels, autopct='%1.1f%%',
               colors=self.colors[:len(class_labels)], startangle=90)
        ax2.set_title('Class Distribution (%)', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_clinical_metrics(self, metrics: Dict[str, float], 
                            class_names: Dict[int, str]) -> None:
        """Plot clinical performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sensitivity by class
        ax = axes[0, 0]
        sensitivities = []
        specificities = []
        ppvs = []
        npvs = []
        labels = []
        
        for class_idx, class_name in class_names.items():
            if f'{class_name}_sensitivity' in metrics:
                sensitivities.append(metrics[f'{class_name}_sensitivity'])
                specificities.append(metrics[f'{class_name}_specificity'])
                ppvs.append(metrics.get(f'{class_name}_ppv', 0))
                npvs.append(metrics.get(f'{class_name}_npv', 0))
                labels.append(class_name)
        
        x = np.arange(len(labels))
        width = 0.2
        
        ax.bar(x - 1.5*width, sensitivities, width, label='Sensitivity', alpha=0.8)
        ax.bar(x - 0.5*width, specificities, width, label='Specificity', alpha=0.8)
        ax.bar(x + 0.5*width, ppvs, width, label='PPV', alpha=0.8)
        ax.bar(x + 1.5*width, npvs, width, label='NPV', alpha=0.8)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Clinical Metrics by Class', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.05)
        
        # MI Detection Focus
        ax = axes[0, 1]
        if 'MI_sensitivity' in metrics:
            mi_metrics = {
                'Sensitivity': metrics['MI_sensitivity'],
                'Specificity': metrics['MI_specificity'],
                'PPV': metrics.get('MI_ppv', 0),
                'NPV': metrics.get('MI_npv', 0)
            }
            
            bars = ax.bar(mi_metrics.keys(), mi_metrics.values(), 
                          color=['red', 'blue', 'green', 'orange'], alpha=0.8)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('MI Detection Performance', fontsize=14)
            ax.set_ylim(0, 1.05)
            
            # Add value labels
            for bar, value in zip(bars, mi_metrics.values()):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
            
            # Add clinical threshold line
            ax.axhline(y=0.9, color='red', linestyle='--', 
                      label='Clinical Threshold (0.90)')
            ax.legend()
        
        # Overall metrics
        ax = axes[1, 0]
        overall_metrics = {
            'Accuracy': metrics.get('accuracy', 0),
            'F1-Score': metrics.get('f1_weighted', 0),
            'Cohen Kappa': metrics.get('cohen_kappa', 0),
            'Matthews Corr': metrics.get('matthews_corrcoef', 0)
        }
        
        bars = ax.bar(overall_metrics.keys(), overall_metrics.values(),
                      color='steelblue', alpha=0.8)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Overall Performance Metrics', fontsize=14)
        ax.set_ylim(0, 1.05)
        
        # Add value labels
        for bar, value in zip(bars, overall_metrics.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
        
        # Clinical thresholds status
        ax = axes[1, 1]
        threshold_status = []
        threshold_labels = []
        threshold_colors = []
        
        # Check MI sensitivity
        mi_sens = metrics.get('MI_sensitivity', 0)
        threshold_status.append(mi_sens)
        threshold_labels.append(f'MI Sensitivity\n(≥0.90)')
        threshold_colors.append('green' if mi_sens >= 0.90 else 'red')
        
        # Check NORM specificity
        norm_spec = metrics.get('NORM_specificity', 0)
        threshold_status.append(norm_spec)
        threshold_labels.append(f'NORM Specificity\n(≥0.95)')
        threshold_colors.append('green' if norm_spec >= 0.95 else 'red')
        
        # Check overall accuracy
        accuracy = metrics.get('accuracy', 0)
        threshold_status.append(accuracy)
        threshold_labels.append(f'Overall Accuracy\n(≥0.85)')
        threshold_colors.append('green' if accuracy >= 0.85 else 'red')
        
        bars = ax.bar(threshold_labels, threshold_status, color=threshold_colors, alpha=0.8)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Clinical Threshold Status', fontsize=14)
        ax.set_ylim(0, 1.05)
        
        # Add threshold lines
        ax.axhline(y=0.90, xmin=0, xmax=0.33, color='black', linestyle='--', alpha=0.5)
        ax.axhline(y=0.95, xmin=0.33, xmax=0.66, color='black', linestyle='--', alpha=0.5)
        ax.axhline(y=0.85, xmin=0.66, xmax=1, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar, value in zip(bars, threshold_status):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle('Clinical Performance Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'clinical_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_report(self, results: Dict[str, Any],
                                  test_results: Dict[str, Any],
                                  model_name: str) -> str:
        """Create comprehensive evaluation report"""
        report = []
        report.append("=" * 80)
        report.append(f"MODEL EVALUATION REPORT - {model_name}")
        report.append("=" * 80)
        report.append("")
        
        # Validation results
        val_metrics = results['metrics']
        report.append("VALIDATION SET RESULTS:")
        report.append("-" * 40)
        report.append(f"Accuracy:  {val_metrics['accuracy']:.4f}")
        report.append(f"Precision: {val_metrics['precision_weighted']:.4f}")
        report.append(f"Recall:    {val_metrics['recall_weighted']:.4f}")
        report.append(f"F1-Score:  {val_metrics['f1_weighted']:.4f}")
        report.append("")
        
        # Test results
        test_metrics = test_results['metrics']
        report.append("TEST SET RESULTS:")
        report.append("-" * 40)
        report.append(f"Accuracy:  {test_metrics['accuracy']:.4f}")
        report.append(f"Precision: {test_metrics['precision_weighted']:.4f}")
        report.append(f"Recall:    {test_metrics['recall_weighted']:.4f}")
        report.append(f"F1-Score:  {test_metrics['f1_weighted']:.4f}")
        report.append("")
        
        # Clinical metrics
        report.append("CLINICAL PERFORMANCE:")
        report.append("-" * 40)
        report.append(f"MI Sensitivity:    {test_metrics.get('MI_sensitivity', 0):.4f}")
        report.append(f"MI Specificity:    {test_metrics.get('MI_specificity', 0):.4f}")
        report.append(f"NORM Specificity:  {test_metrics.get('NORM_specificity', 0):.4f}")
        report.append("")
        
        # Per-class performance
        report.append("PER-CLASS PERFORMANCE:")
        report.append("-" * 40)
        class_report = test_metrics['class_report']
        for class_str, class_metrics in class_report.items():
            if class_str not in ['accuracy', 'macro avg', 'weighted avg']:
                report.append(f"{class_str}:")
                report.append(f"  Precision: {class_metrics['precision']:.4f}")
                report.append(f"  Recall:    {class_metrics['recall']:.4f}")
                report.append(f"  F1-Score:  {class_metrics['f1-score']:.4f}")
                report.append(f"  Support:   {class_metrics['support']}")
        
        report.append("")
        report.append("=" * 80)
        
        # Save report
        report_text = "\n".join(report)
        report_path = self.save_dir / f'evaluation_report_{model_name.lower().replace(" ", "_")}.txt'
        
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        return report_text