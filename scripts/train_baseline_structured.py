#!/usr/bin/env python3
"""
Baseline Model Training - Structured Data (Phase 1)
Loads partitioned structured numpy arrays, trains Logistic Regression and 
Random Forest models, evaluates them on the test set, and saves results.
"""

import os
import sys
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def save_model(model, name: str, out_dir: str):
    """Saves model to disk using joblib."""
    path = os.path.join(out_dir, f"{name}.joblib")
    joblib.dump(model, path)
    return path


def evaluate_model(y_true, y_prob, name: str, results_dir: str) -> dict:
    """Calculates evaluation metrics and serializes to JSON."""
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    
    metrics = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'brier_score': brier
    }
    
    path = os.path.join(results_dir, f"{name}_metrics.json")
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    return metrics


def plot_curves(models_results: dict, figures_dir: str):
    """Plots ROC and PR curves for multiple models."""
    plt.figure(figsize=(12, 5))
    
    # 1. ROC Curve
    plt.subplot(1, 2, 1)
    for name, res in models_results.items():
        fpr, tpr, _ = roc_curve(res['y'], res['prob'])
        plt.plot(fpr, tpr, label=f"{name} (AUC={res['metrics']['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test Set)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 2. PR Curve
    plt.subplot(1, 2, 2)
    for name, res in models_results.items():
        prec, rec, _ = precision_recall_curve(res['y'], res['prob'])
        plt.plot(rec, prec, label=f"{name} (AUC={res['metrics']['pr_auc']:.3f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Test Set)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(figures_dir, 'baseline_structured_curves.png')
    plt.savefig(output_path, dpi=config.FIGURE_DPI)


def main():
    # Ensure directories exist
    for d in [config.LOGS_DIR, config.MODELS_DIR, config.RESULTS_DIR, config.FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)
    
    # Configure logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(os.path.join(config.LOGS_DIR, f"train_baseline_{timestamp}.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger('train_baseline_structured')
    
    logger.info("="*80)
    logger.info("Phase 1: Baseline Models (Structured Data Only)")
    logger.info("="*80)
    
    # Load Numpy Arrays
    logger.info("Loading NumPy arrays from processed_data/ ...")
    try:
        X_train = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'X_train_structured.npy'))
        y_train = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'y_train.npy'))
        X_val   = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'X_val_structured.npy'))
        y_val   = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'y_val.npy'))
        X_test  = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'X_test_structured.npy'))
        y_test  = np.load(os.path.join(config.PROCESSED_DATA_DIR, 'y_test.npy'))
        logger.info(f"Loaded successfully. Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    except FileNotFoundError as e:
        logger.error("Could not find partitioned NumPy arrays! Ensure data_splitter.py was run.")
        logger.error(str(e))
        sys.exit(1)
        
    # Scaling Data (LogReg needs standardized features, particularly because we use L2 penalty implicitly)
    from sklearn.preprocessing import StandardScaler
    logger.info("Scaling features using StandardScaler (fit on train)...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    # Save the scaler for Phase 4 or inference
    save_model(scaler, 'structured_scaler', config.MODELS_DIR)

    # 1. Logistic Regression
    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS)
    lr.fit(X_train_s, y_train)
    save_model(lr, 'lr_baseline_structured', config.MODELS_DIR)
    with open(os.path.join(config.MODELS_DIR, 'logistic_regression_params.json'), 'w') as f:
        json.dump(config.LOGISTIC_REGRESSION_PARAMS, f, indent=2)

    # 2. Random Forest
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)
    rf.fit(X_train, y_train) # RF doesn't require standardization
    save_model(rf, 'rf_baseline_structured', config.MODELS_DIR)
    with open(os.path.join(config.MODELS_DIR, 'random_forest_params.json'), 'w') as f:
        json.dump(config.RANDOM_FOREST_PARAMS, f, indent=2)

    # Evaluation on Test set
    logger.info("Evaluating models on Test set...")
    lr_prob = lr.predict_proba(X_test_s)[:, 1]
    rf_prob = rf.predict_proba(X_test)[:, 1]

    lr_metrics = evaluate_model(y_test, lr_prob, 'lr_baseline_structured', config.RESULTS_DIR)
    rf_metrics = evaluate_model(y_test, rf_prob, 'rf_baseline_structured', config.RESULTS_DIR)

    logger.info(f"➜ Logistic Regression: ROC-AUC = {lr_metrics['roc_auc']:.4f}, PR-AUC = {lr_metrics['pr_auc']:.4f}")
    logger.info(f"➜ Random Forest      : ROC-AUC = {rf_metrics['roc_auc']:.4f}, PR-AUC = {rf_metrics['pr_auc']:.4f}")

    models_results = {
        'Logistic Regression': {'y': y_test, 'prob': lr_prob, 'metrics': lr_metrics},
        'Random Forest': {'y': y_test, 'prob': rf_prob, 'metrics': rf_metrics}
    }
    plot_curves(models_results, config.FIGURES_DIR)
    
    # Phase completion marker
    marker_path = os.path.join(config.LOGS_DIR, 'phase1_complete.txt')
    with open(marker_path, 'w') as f:
        f.write(f"Phase 1 completed at {datetime.now().isoformat()}\n")
    logger.info(f"Phase 1 completion marker written to {marker_path}")
    logger.info("Baseline structured modeling complete. Artifacts saved to models/ results/ and figures/")


if __name__ == '__main__':
    main()
