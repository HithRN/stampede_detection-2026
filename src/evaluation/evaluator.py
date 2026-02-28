"""
Comprehensive evaluation module for stampede detection models
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_score, recall_score, f1_score
)
from tensorflow.keras.utils import to_categorical

from config.config import Config
from src.utils.helpers import print_section_header, print_subsection_header, sanitize_predictions
from src.utils.visualization import plot_confusion_matrix, plot_roc_curves


def evaluate_model_comprehensive(model, X_flow_val, X_scalar_val, y_val,
                                 original_frames_val=None, config=None):
    """
    Comprehensive model evaluation with metrics, confusion matrix, classification report, and ROC-AUC.
    
    Args:
        model: Trained model
        X_flow_val: Validation optical flow data
        X_scalar_val: Validation scalar features
        y_val: Validation labels (integer format)
        original_frames_val: Original frames (optional)
        config: Configuration dictionary (optional)
        
    Returns:
        Dictionary with evaluation results
    """
    # Ensure output directories exist
    Config.ensure_directories()
    
    print_section_header("COMPREHENSIVE MODEL EVALUATION")
    
    # Get class names
    if config is None:
        class_names = Config.CLASS_NAMES
    else:
        class_names = config.get('data', {}).get('class_names', Config.CLASS_NAMES)
    
    # Convert true labels to one-hot
    y_val_onehot = to_categorical(y_val, num_classes=len(class_names))
    
    # Get prediction probabilities
    print("Generating predictions...")
    y_pred_proba = model.predict([X_flow_val, X_scalar_val], verbose=1)
    
    # Sanitize predictions (handle NaN/Inf)
    if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
        print("WARNING: NaN/Inf detected in prediction probabilities. Applying safe fixes...")
        y_pred_proba = sanitize_predictions(y_pred_proba)
    
    # Convert to class predictions
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # === Classification Metrics ===
    print_subsection_header("CLASSIFICATION METRICS (Validation Set)")
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    print("\nConfusion Matrix (Validation):")
    print(cm)
    
    # Plot Confusion Matrix
    plot_confusion_matrix(
        cm,
        class_names=class_names,
        title='Confusion Matrix (Validation Set)',
        save_path=os.path.join(Config.FIGURES_DIR, 'confusion_matrix_validation.png')
    )
    
    # Classification Report
    print("\nClassification Report (Validation):")
    report = classification_report(y_val, y_pred, target_names=class_names)
    print(report)
    
    # Save report to file
    report_path = os.path.join(Config.FIGURES_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("Classification Report (Validation Set)\n")
        f.write("=" * 50 + "\n")
        f.write(report)
    print(f"Classification report saved to: {report_path}")
    
    # === Per-Class Metrics ===
    print_subsection_header("PER-CLASS METRICS")
    
    precision = precision_score(y_val, y_pred, average=None)
    recall = recall_score(y_val, y_pred, average=None)
    f1 = f1_score(y_val, y_pred, average=None)
    
    print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 55)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f}")
    
    # === ROC-AUC (One-vs-Rest) ===
    print_subsection_header("ROC-AUC METRICS (One-vs-Rest)")
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(class_names)
    
    # Safe ROC-AUC computation
    for i in range(n_classes):
        # Extract scores for class i
        scores = y_pred_proba[:, i]
        
        # Ensure scores are finite and in [0,1] range
        if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
            print(f"Warning: Invalid scores for class {i} ({class_names[i]}). Clipping...")
            scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
        scores = np.clip(scores, 0.0, 1.0)
        
        # Compute ROC curve and AUC
        try:
            fpr[i], tpr[i], _ = roc_curve(y_val_onehot[:, i], scores)
            roc_auc[i] = auc(fpr[i], tpr[i])
            print(f"Class {i} ({class_names[i]}): AUC = {roc_auc[i]:.4f}")
        except Exception as e:
            print(f"Failed to compute ROC for class {i} ({class_names[i]}): {e}")
            fpr[i] = np.array([0, 1])
            tpr[i] = np.array([0, 1])
            roc_auc[i] = 0.5  # fallback
    
    # ROC-AUC Plot
    plot_roc_curves(
        fpr, tpr, roc_auc,
        class_names=class_names,
        title='ROC Curves (Validation Set - One-vs-Rest)',
        save_path=os.path.join(Config.FIGURES_DIR, 'roc_auc_multiclass.png')
    )
    
    # === Overall Metrics ===
    print_subsection_header("OVERALL METRICS")
    
    accuracy = np.mean(y_val == y_pred)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    
    # Save predictions and probabilities
    np.save(os.path.join(Config.RESULTS_DIR, 'y_val_pred_proba.npy'), y_pred_proba)
    np.save(os.path.join(Config.RESULTS_DIR, 'y_val_pred.npy'), y_pred)
    np.save(os.path.join(Config.RESULTS_DIR, 'y_val_true.npy'), y_val)
    
    # === Evaluation Results Dictionary ===
    evaluation_results = {
        'confusion_matrix_val': cm,
        'classification_report_val': classification_report(
            y_val, y_pred, target_names=class_names, output_dict=True
        ),
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'class_names': class_names,
        'validation_metrics': {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist()
        }
    }
    
    print_section_header("EVALUATION COMPLETE")
    print(f"Overall validation accuracy: {accuracy:.4f}")
    print(f"Results saved to: {Config.RESULTS_DIR}")
    
    return evaluation_results


def evaluate_test_set(model, X_flow_test, X_scalar_test, y_test, class_names=None):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        X_flow_test: Test optical flow data
        X_scalar_test: Test scalar features
        y_test: Test labels
        class_names: List of class names (optional)
        
    Returns:
        Dictionary with test results
    """
    if class_names is None:
        class_names = Config.CLASS_NAMES
    
    print_section_header("TEST SET EVALUATION")
    
    # Get predictions
    y_test_pred_proba = model.predict([X_flow_test, X_scalar_test], verbose=1)
    
    # Sanitize predictions
    if np.any(np.isnan(y_test_pred_proba)) or np.any(np.isinf(y_test_pred_proba)):
        y_test_pred_proba = sanitize_predictions(y_test_pred_proba)
    
    y_test_pred = np.argmax(y_test_pred_proba, axis=1)
    
    # Confusion Matrix
    cm_test = confusion_matrix(y_test, y_test_pred)
    print("\nTest Set Confusion Matrix:")
    print(cm_test)
    
    plot_confusion_matrix(
        cm_test,
        class_names=class_names,
        title='Confusion Matrix (Test Set)',
        save_path=os.path.join(Config.FIGURES_DIR, 'confusion_matrix_test.png')
    )
    
    # Classification Report
    print("\nTest Set Classification Report:")
    report = classification_report(y_test, y_test_pred, target_names=class_names)
    print(report)
    
    # Save test report
    report_path = os.path.join(Config.FIGURES_DIR, 'classification_report_test.txt')
    with open(report_path, 'w') as f:
        f.write("Classification Report (Test Set)\n")
        f.write("=" * 50 + "\n")
        f.write(report)
    
    # Save test predictions
    np.save(os.path.join(Config.RESULTS_DIR, 'y_test_pred_proba.npy'), y_test_pred_proba)
    np.save(os.path.join(Config.RESULTS_DIR, 'y_test_pred.npy'), y_test_pred)
    np.save(os.path.join(Config.RESULTS_DIR, 'y_test_true.npy'), y_test)
    
    # Calculate metrics
    accuracy = np.mean(y_test == y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='macro')
    recall = recall_score(y_test, y_test_pred, average='macro')
    f1 = f1_score(y_test, y_test_pred, average='macro')
    
    test_results = {
        'confusion_matrix': cm_test,
        'classification_report': classification_report(
            y_test, y_test_pred, target_names=class_names, output_dict=True
        ),
        'metrics': {
            'accuracy': accuracy,
            'macro_precision': precision,
            'macro_recall': recall,
            'macro_f1': f1
        }
    }
    
    print(f"\nTest Set Accuracy: {accuracy:.4f}")
    print_section_header("TEST EVALUATION COMPLETE")
    
    return test_results


def generate_evaluation_summary_report(evaluation_results, test_results=None, save_path=None):
    """
    Generate a comprehensive summary report of evaluation results.
    
    Args:
        evaluation_results: Validation evaluation results
        test_results: Test evaluation results (optional)
        save_path: Path to save the report
    """
    if save_path is None:
        save_path = os.path.join(Config.RESULTS_DIR, 'evaluation_summary.txt')
    
    with open(save_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("STAMPEDE DETECTION - EVALUATION SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        # Validation Results
        f.write("VALIDATION SET RESULTS\n")
        f.write("-" * 70 + "\n")
        val_metrics = evaluation_results['validation_metrics']
        f.write(f"Accuracy: {val_metrics['accuracy']:.4f}\n")
        f.write(f"Macro Precision: {val_metrics['macro_precision']:.4f}\n")
        f.write(f"Macro Recall: {val_metrics['macro_recall']:.4f}\n")
        f.write(f"Macro F1-Score: {val_metrics['macro_f1']:.4f}\n\n")
        
        # Per-class metrics
        f.write("Per-Class Metrics:\n")
        class_names = evaluation_results['class_names']
        for i, name in enumerate(class_names):
            f.write(f"  {name}:\n")
            f.write(f"    Precision: {val_metrics['per_class_precision'][i]:.4f}\n")
            f.write(f"    Recall: {val_metrics['per_class_recall'][i]:.4f}\n")
            f.write(f"    F1-Score: {val_metrics['per_class_f1'][i]:.4f}\n")
            f.write(f"    ROC-AUC: {evaluation_results['roc_auc'][i]:.4f}\n")
        
        # Test Results (if available)
        if test_results:
            f.write("\n" + "=" * 70 + "\n")
            f.write("TEST SET RESULTS\n")
            f.write("-" * 70 + "\n")
            test_metrics = test_results['metrics']
            f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
            f.write(f"Macro Precision: {test_metrics['macro_precision']:.4f}\n")
            f.write(f"Macro Recall: {test_metrics['macro_recall']:.4f}\n")
            f.write(f"Macro F1-Score: {test_metrics['macro_f1']:.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"Evaluation summary report saved to: {save_path}")
