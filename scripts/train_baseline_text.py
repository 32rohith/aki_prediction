#!/usr/bin/env python3
"""
Task 12: Implement Phase 2: Text-Only Model
Trains baseline models using ONLY the BioClinicalBERT text embeddings.
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


def match_embeddings_to_split(split_df, all_embeds, all_embed_stay_ids):
    """
    Maps the global text_embeddings.npy to the specific patients in a data split.
    """
    id_to_idx = {sid: i for i, sid in enumerate(all_embed_stay_ids)}
    split_embeds = []
    
    for sid in split_df['stay_id']:
        idx = id_to_idx.get(sid)
        if idx is not None:
            split_embeds.append(all_embeds[idx])
        else:
            split_embeds.append(np.zeros(768, dtype=np.float32))
            
    return np.vstack(split_embeds)


def save_model(model, name: str, out_dir: str):
    path = os.path.join(out_dir, f"{name}.joblib")
    joblib.dump(model, path)
    return path


def evaluate_model(y_true, y_prob, name: str, results_dir: str) -> dict:
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
    plt.figure(figsize=(12, 5))
    
    # 1. ROC Curve
    plt.subplot(1, 2, 1)
    for name, res in models_results.items():
        fpr, tpr, _ = roc_curve(res['y'], res['prob'])
        plt.plot(fpr, tpr, label=f"{name} (AUC={res['metrics']['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Text Only (Test Set)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 2. PR Curve
    plt.subplot(1, 2, 2)
    for name, res in models_results.items():
        prec, rec, _ = precision_recall_curve(res['y'], res['prob'])
        plt.plot(rec, prec, label=f"{name} (AUC={res['metrics']['pr_auc']:.3f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve - Text Only (Test Set)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(figures_dir, 'baseline_text_curves.png')
    plt.savefig(output_path, dpi=config.FIGURE_DPI)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(os.path.join(config.LOGS_DIR, f"train_baseline_text_{timestamp}.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger('train_baseline_text')
    
    logger.info("="*80)
    logger.info("Task 12 / Phase 2: Training Text-Only Baseline Models")
    logger.info("="*80)
    
    # Verify Phase 1 completion
    phase1_marker = os.path.join(config.LOGS_DIR, 'phase1_complete.txt')
    if not os.path.exists(phase1_marker):
        logger.warning("Phase 1 completion marker not found. Ensure Phase 1 has been run.")
    
    # Load labels and patient dataframes
    data_dir = config.PROCESSED_DATA_DIR
    logger.info("Loading patient splits and global embeddings...")
    
    # Load structured dataset just to get the ordered stay_ids matching the numpy arrays
    df_stays = pd.read_csv(os.path.join(data_dir, 'structured_dataset.csv'), usecols=['subject_id', 'stay_id'])
    
    train_pts = set(pd.read_csv(os.path.join(data_dir, 'train_patients.csv'))['subject_id'])
    val_pts   = set(pd.read_csv(os.path.join(data_dir, 'val_patients.csv'))['subject_id'])
    test_pts  = set(pd.read_csv(os.path.join(data_dir, 'test_patients.csv'))['subject_id'])
    
    train_df = df_stays[df_stays['subject_id'].isin(train_pts)]
    val_df   = df_stays[df_stays['subject_id'].isin(val_pts)]
    test_df  = df_stays[df_stays['subject_id'].isin(test_pts)]
    
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test  = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Load global text embeddings
    all_embeds = np.load(os.path.join(data_dir, 'text_embeddings.npy'))
    all_ids    = np.load(os.path.join(data_dir, 'text_stay_ids.npy'))
    
    logger.info("Mapping embeddings to splits...")
    X_train_text = match_embeddings_to_split(train_df, all_embeds, all_ids)
    X_val_text   = match_embeddings_to_split(val_df, all_embeds, all_ids)
    X_test_text  = match_embeddings_to_split(test_df, all_embeds, all_ids)
    
    logger.info(f"Train Text shape: {X_train_text.shape}")
    logger.info(f"Test Text shape:  {X_test_text.shape}")

    # Scaling text feature embeddings is mostly standard for regression
    from sklearn.preprocessing import StandardScaler
    logger.info("Scaling text features (fit on train)...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_text)
    X_test_s = scaler.transform(X_test_text)
    
    save_model(scaler, 'text_scaler', config.MODELS_DIR)

    # 1. Logistic Regression (Text-only classifier)
    logger.info("Training Logistic Regression on Text...")
    lr = LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS)
    lr.fit(X_train_s, y_train)
    save_model(lr, 'text_classifier', config.MODELS_DIR)
    with open(os.path.join(config.MODELS_DIR, 'text_classifier_params.json'), 'w') as f:
        json.dump(config.LOGISTIC_REGRESSION_PARAMS, f, indent=2)

    # 2. Random Forest (Text-only)
    logger.info("Training Random Forest on Text...")
    rf = RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)
    rf.fit(X_train_text, y_train)
    save_model(rf, 'rf_baseline_text', config.MODELS_DIR)
    with open(os.path.join(config.MODELS_DIR, 'rf_text_params.json'), 'w') as f:
        json.dump(config.RANDOM_FOREST_PARAMS, f, indent=2)

    # Evaluation on Test set
    logger.info("Evaluating models on Test set...")
    lr_prob = lr.predict_proba(X_test_s)[:, 1]
    rf_prob = rf.predict_proba(X_test_text)[:, 1]

    lr_metrics = evaluate_model(y_test, lr_prob, 'text_classifier', config.RESULTS_DIR)
    rf_metrics = evaluate_model(y_test, rf_prob, 'rf_baseline_text', config.RESULTS_DIR)

    logger.info(f"➜ LogReg (Text): ROC-AUC = {lr_metrics['roc_auc']:.4f}, PR-AUC = {lr_metrics['pr_auc']:.4f}")
    logger.info(f"➜ RF (Text)    : ROC-AUC = {rf_metrics['roc_auc']:.4f}, PR-AUC = {rf_metrics['pr_auc']:.4f}")

    models_results = {
        'Logistic Regression (Text)': {'y': y_test, 'prob': lr_prob, 'metrics': lr_metrics},
        'Random Forest (Text)': {'y': y_test, 'prob': rf_prob, 'metrics': rf_metrics}
    }
    plot_curves(models_results, config.FIGURES_DIR)
    
    # Document temporal limitations
    logger.info("NOTE: Discharge summaries contain post-ICU information (temporal limitation).")
    logger.info("Future work should use temporally aligned clinical notes (e.g., nursing notes within 24h).")
    
    # Phase completion marker
    marker_path = os.path.join(config.LOGS_DIR, 'phase2_complete.txt')
    with open(marker_path, 'w') as f:
        f.write(f"Phase 2 completed at {datetime.now().isoformat()}\n")
    logger.info(f"Phase 2 completion marker written to {marker_path}")
    logger.info("Phase 2 text-only modeling complete. Artifacts saved.")


if __name__ == '__main__':
    main()
