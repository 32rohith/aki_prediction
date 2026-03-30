#!/usr/bin/env python3
"""
Task 15: Comprehensive Model Evaluation
Generates ROC curves, PR curves, calibration curves, bar chart comparisons,
and a unified CSV report across all trained models (Phase 1, 2, 3).
"""

import os
import sys
import glob
import json
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def match_embeddings_to_split(split_df, all_embeds, all_embed_stay_ids):
    id_to_idx = {sid: i for i, sid in enumerate(all_embed_stay_ids)}
    split_embeds = []
    for sid in split_df['stay_id']:
        idx = id_to_idx.get(sid)
        if idx is not None:
            split_embeds.append(all_embeds[idx])
        else:
            split_embeds.append(np.zeros(config.EMBEDDING_DIM, dtype=np.float32))
    return np.vstack(split_embeds)


def main():
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT,
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger('evaluate_all')

    logger.info("=" * 80)
    logger.info("Task 15: Comprehensive Model Evaluation Report")
    logger.info("=" * 80)

    data_dir = config.PROCESSED_DATA_DIR
    model_dir = config.MODELS_DIR
    results_dir = config.RESULTS_DIR
    figures_dir = config.FIGURES_DIR

    # Load test data
    logger.info("Loading test data...")
    X_test_struct = np.load(os.path.join(data_dir, 'X_test_structured.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    # Text embeddings for test
    df_stays = pd.read_csv(os.path.join(data_dir, 'structured_dataset.csv'), usecols=['subject_id', 'stay_id'])
    test_pts = set(pd.read_csv(os.path.join(data_dir, 'test_patients.csv'))['subject_id'])
    test_df = df_stays[df_stays['subject_id'].isin(test_pts)]

    all_embeds = np.load(os.path.join(data_dir, 'text_embeddings.npy'))
    all_ids = np.load(os.path.join(data_dir, 'text_stay_ids.npy'))
    X_test_text = match_embeddings_to_split(test_df, all_embeds, all_ids)

    # Load all models and scalers
    models_to_eval = {}

    # Phase 1: Structured-Only LR
    if os.path.exists(os.path.join(model_dir, 'lr_baseline_structured.joblib')):
        scaler_struct = joblib.load(os.path.join(model_dir, 'structured_scaler.joblib'))
        lr_struct = joblib.load(os.path.join(model_dir, 'lr_baseline_structured.joblib'))
        X_s = scaler_struct.transform(X_test_struct)
        models_to_eval['LR (Structured)'] = {'prob': lr_struct.predict_proba(X_s)[:, 1]}

    # Phase 1: Structured-Only RF
    if os.path.exists(os.path.join(model_dir, 'rf_baseline_structured.joblib')):
        rf_struct = joblib.load(os.path.join(model_dir, 'rf_baseline_structured.joblib'))
        models_to_eval['RF (Structured)'] = {'prob': rf_struct.predict_proba(X_test_struct)[:, 1]}

    # Phase 2: Text-Only LR
    if os.path.exists(os.path.join(model_dir, 'text_classifier.joblib')):
        text_scaler = joblib.load(os.path.join(model_dir, 'text_scaler.joblib'))
        text_lr = joblib.load(os.path.join(model_dir, 'text_classifier.joblib'))
        X_t_s = text_scaler.transform(X_test_text)
        models_to_eval['LR (Text)'] = {'prob': text_lr.predict_proba(X_t_s)[:, 1]}

    # Phase 2: Text-Only RF
    if os.path.exists(os.path.join(model_dir, 'rf_baseline_text.joblib')):
        rf_text = joblib.load(os.path.join(model_dir, 'rf_baseline_text.joblib'))
        models_to_eval['RF (Text)'] = {'prob': rf_text.predict_proba(X_test_text)[:, 1]}

    # Phase 3: Fusion MLP
    if os.path.exists(os.path.join(model_dir, 'fusion_model.joblib')):
        multi_scaler = joblib.load(os.path.join(model_dir, 'multimodal_scaler.joblib'))
        fusion_mlp = joblib.load(os.path.join(model_dir, 'fusion_model.joblib'))
        X_multi = np.hstack([X_test_struct, X_test_text])
        X_m_s = multi_scaler.transform(X_multi)
        models_to_eval['MLP (Fusion)'] = {'prob': fusion_mlp.predict_proba(X_m_s)[:, 1]}

    if not models_to_eval:
        logger.error("No trained models found!")
        return

    # --- 15.2: Compute metrics for all models ---
    records = []
    for name, data in models_to_eval.items():
        prob = data['prob']
        pred = (prob >= 0.5).astype(int)
        metrics = {
            'Model': name,
            'AUROC': roc_auc_score(y_test, prob),
            'AUPRC': average_precision_score(y_test, prob),
            'Accuracy': accuracy_score(y_test, pred),
            'Precision': precision_score(y_test, pred, zero_division=0),
            'Recall': recall_score(y_test, pred, zero_division=0),
            'F1': f1_score(y_test, pred, zero_division=0),
            'Brier': brier_score_loss(y_test, prob)
        }
        records.append(metrics)
        data['metrics'] = metrics

    df = pd.DataFrame(records)
    csv_path = os.path.join(results_dir, 'model_metrics.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"\nModel Metrics:\n{df.to_markdown(index=False)}")
    logger.info(f"Saved to {csv_path}")

    # --- 15.3: ROC Curves (single plot, all models) ---
    logger.info("Generating ROC curves...")
    plt.figure(figsize=(8, 6))
    for name, data in models_to_eval.items():
        fpr, tpr, _ = roc_curve(y_test, data['prob'])
        plt.plot(fpr, tpr, label=f"{name} (AUC={data['metrics']['AUROC']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - All Models (Test Set)')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'roc_curves.png'), dpi=config.FIGURE_DPI)
    plt.close()

    # --- 15.3: PR Curves (single plot, all models) ---
    logger.info("Generating Precision-Recall curves...")
    plt.figure(figsize=(8, 6))
    for name, data in models_to_eval.items():
        prec, rec, _ = precision_recall_curve(y_test, data['prob'])
        plt.plot(rec, prec, label=f"{name} (AUC={data['metrics']['AUPRC']:.3f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves - All Models (Test Set)')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pr_curves.png'), dpi=config.FIGURE_DPI)
    plt.close()

    # --- 15.4: Calibration Curves ---
    logger.info("Generating calibration curves...")
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    for name, data in models_to_eval.items():
        prob_true, prob_pred = calibration_curve(y_test, data['prob'], n_bins=config.CALIBRATION_BINS)
        plt.plot(prob_pred, prob_true, marker='o', label=name)
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curves (Reliability Diagrams)')
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'calibration_curves.png'), dpi=config.FIGURE_DPI)
    plt.close()

    # --- 15.5: Bar chart comparisons ---
    logger.info("Generating bar chart comparisons...")
    model_names = [r['Model'] for r in records]

    # AUROC comparison
    plt.figure(figsize=(10, 5))
    plt.bar(model_names, [r['AUROC'] for r in records], color='steelblue')
    plt.ylabel('AUROC')
    plt.title('AUROC Comparison Across Models')
    plt.xticks(rotation=15, ha='right')
    plt.ylim(0.4, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'auroc_comparison.png'), dpi=config.FIGURE_DPI)
    plt.close()

    # AUPRC comparison
    plt.figure(figsize=(10, 5))
    plt.bar(model_names, [r['AUPRC'] for r in records], color='darkorange')
    plt.ylabel('AUPRC')
    plt.title('AUPRC Comparison Across Models')
    plt.xticks(rotation=15, ha='right')
    plt.ylim(0.0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'auprc_comparison.png'), dpi=config.FIGURE_DPI)
    plt.close()

    # Precision/Recall/F1 grouped bar chart
    x = np.arange(len(model_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, [r['Precision'] for r in records], width, label='Precision', color='#2196F3')
    ax.bar(x, [r['Recall'] for r in records], width, label='Recall', color='#FF9800')
    ax.bar(x + width, [r['F1'] for r in records], width, label='F1 Score', color='#4CAF50')
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall, F1 Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'prf1_comparison.png'), dpi=config.FIGURE_DPI)
    plt.close()

    logger.info("All evaluation visualizations saved to figures/")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
