#!/usr/bin/env python3
"""
Task 16: Modality Masking Robustness Testing
Evaluates the trained Fusion MLP when entire modalities are masked at inference time.
Computes degradation percentages and generates robustness comparison visualizations.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             accuracy_score, precision_score, recall_score, f1_score)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_metrics(y_true, y_prob):
    pred = (y_prob >= 0.5).astype(int)
    return {
        'AUROC': roc_auc_score(y_true, y_prob),
        'AUPRC': average_precision_score(y_true, y_prob),
        'Accuracy': accuracy_score(y_true, pred),
        'Precision': precision_score(y_true, pred, zero_division=0),
        'Recall': recall_score(y_true, pred, zero_division=0),
        'F1': f1_score(y_true, pred, zero_division=0),
        'Brier': brier_score_loss(y_true, y_prob)
    }


def main():
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT,
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger('robustness_testing')

    logger.info("=" * 80)
    logger.info("Task 16 / Phase 4: Modality Masking Robustness Evaluation")
    logger.info("=" * 80)

    # Verify Phase 3 completion
    phase3_marker = os.path.join(config.LOGS_DIR, 'phase3_complete.txt')
    if not os.path.exists(phase3_marker):
        logger.warning("Phase 3 completion marker not found. Ensure Phase 3 has been run.")

    data_dir = config.PROCESSED_DATA_DIR
    model_dir = config.MODELS_DIR

    # Load test data
    logger.info("Loading test set data and trained models...")
    X_test_struct = np.load(os.path.join(data_dir, 'X_test_structured.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    # Reconstruct text embeddings for test split
    df_stays = pd.read_csv(os.path.join(data_dir, 'structured_dataset.csv'), usecols=['subject_id', 'stay_id'])
    test_pts = set(pd.read_csv(os.path.join(data_dir, 'test_patients.csv'))['subject_id'])
    test_df = df_stays[df_stays['subject_id'].isin(test_pts)]

    all_embeds = np.load(os.path.join(data_dir, 'text_embeddings.npy'))
    all_ids = np.load(os.path.join(data_dir, 'text_stay_ids.npy'))

    id_to_idx = {sid: i for i, sid in enumerate(all_ids)}
    split_embeds = []
    for sid in test_df['stay_id']:
        idx = id_to_idx.get(sid)
        if idx is not None:
            split_embeds.append(all_embeds[idx])
        else:
            split_embeds.append(np.zeros(config.EMBEDDING_DIM, dtype=np.float32))
    X_test_text = np.vstack(split_embeds)

    struct_dim = X_test_struct.shape[1]
    text_dim = X_test_text.shape[1]

    # Load Fusion MLP and scaler
    scaler = joblib.load(os.path.join(model_dir, 'multimodal_scaler.joblib'))
    fusion_model = joblib.load(os.path.join(model_dir, 'fusion_model.joblib'))

    results = []

    # --- 16.2: Full Fusion (Baseline) ---
    logger.info("\nScenario 1: Full Fusion (Baseline - No Masking)...")
    X_full = np.hstack([X_test_struct, X_test_text])
    X_full_s = scaler.transform(X_full)
    prob_full = fusion_model.predict_proba(X_full_s)[:, 1]
    metrics_full = get_metrics(y_test, prob_full)
    results.append({'Scenario': 'Full (Baseline)', **metrics_full})
    logger.info(f"  AUROC={metrics_full['AUROC']:.4f}, AUPRC={metrics_full['AUPRC']:.4f}")

    # --- 16.3: Mask Text ---
    logger.info("Scenario 2: Masked Text (zero vectors for text)...")
    X_masked_text = np.hstack([X_test_struct, np.zeros_like(X_test_text)])
    X_mt_s = scaler.transform(X_masked_text)
    prob_mt = fusion_model.predict_proba(X_mt_s)[:, 1]
    metrics_mt = get_metrics(y_test, prob_mt)
    results.append({'Scenario': 'Masked Text', **metrics_mt})
    logger.info(f"  AUROC={metrics_mt['AUROC']:.4f}, AUPRC={metrics_mt['AUPRC']:.4f}")

    # --- 16.4: Mask Structured ---
    logger.info("Scenario 3: Masked Structured (zero vectors for structured)...")
    X_masked_struct = np.hstack([np.zeros_like(X_test_struct), X_test_text])
    X_ms_s = scaler.transform(X_masked_struct)
    prob_ms = fusion_model.predict_proba(X_ms_s)[:, 1]
    metrics_ms = get_metrics(y_test, prob_ms)
    results.append({'Scenario': 'Masked Structured', **metrics_ms})
    logger.info(f"  AUROC={metrics_ms['AUROC']:.4f}, AUPRC={metrics_ms['AUPRC']:.4f}")

    # --- 16.5: Compute Degradation Percentages ---
    logger.info("\nComputing performance degradation percentages...")
    degradation = []
    for scenario_metrics in [metrics_mt, metrics_ms]:
        scenario_name = 'Masked Text' if scenario_metrics == metrics_mt else 'Masked Structured'
        deg = {}
        deg['Scenario'] = scenario_name
        for metric_name in ['AUROC', 'AUPRC', 'Accuracy', 'F1']:
            base_val = metrics_full[metric_name]
            mask_val = scenario_metrics[metric_name]
            pct_drop = ((base_val - mask_val) / base_val * 100) if base_val > 0 else 0.0
            deg[f'{metric_name}_pct_drop'] = round(pct_drop, 2)
        degradation.append(deg)

    # Save robustness metrics
    df_results = pd.DataFrame(results)
    robustness_csv_path = os.path.join(config.RESULTS_DIR, 'robustness_metrics.csv')
    df_results.to_csv(robustness_csv_path, index=False)
    logger.info(f"\nRobustness Metrics:\n{df_results.to_markdown(index=False)}")
    logger.info(f"Saved to {robustness_csv_path}")

    # Save degradation
    df_deg = pd.DataFrame(degradation)
    deg_csv_path = os.path.join(config.RESULTS_DIR, 'performance_degradation.csv')
    df_deg.to_csv(deg_csv_path, index=False)
    logger.info(f"\nDegradation Percentages:\n{df_deg.to_markdown(index=False)}")
    logger.info(f"Saved to {deg_csv_path}")

    # --- 16.6: Robustness Comparison Visualization ---
    logger.info("Generating robustness comparison visualization...")
    scenarios = [r['Scenario'] for r in results]
    aurocs = [r['AUROC'] for r in results]
    auprcs = [r['AUPRC'] for r in results]

    x = np.arange(len(scenarios))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, aurocs, width, label='AUROC', color='steelblue')
    bars2 = ax.bar(x + width / 2, auprcs, width, label='AUPRC', color='darkorange')
    ax.set_ylabel('Score')
    ax.set_title('Modality Masking Robustness: Performance Across Conditions')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_DIR, 'robustness_comparison.png'), dpi=config.FIGURE_DPI)
    plt.close()

    logger.info("NOTE: Masking simulates real-world missing modality scenarios at inference time (no retraining).")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
