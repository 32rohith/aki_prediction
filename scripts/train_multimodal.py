#!/usr/bin/env python3
"""
Task 13: Implement Phase 3: Multimodal Fusion Model
Trains an MLPClassifier on Early Fusion (concatenation) of Structured + Text Features.
"""

import os
import sys
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import joblib

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


def save_model(model, name, out_dir):
    path = os.path.join(out_dir, f"{name}.joblib")
    joblib.dump(model, path)
    return path


def evaluate_model(y_true, y_prob, name, results_dir):
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob),
        'brier_score': brier_score_loss(y_true, y_prob)
    }
    with open(os.path.join(results_dir, f"{name}_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    return metrics


def plot_curves(y_true, y_prob, name, figures_dir):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=f"Fusion MLP (AUC={roc_auc_score(y_true, y_prob):.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Multimodal Fusion (Test Set)')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(rec, prec, label=f"Fusion MLP (AUC={average_precision_score(y_true, y_prob):.3f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve - Multimodal Fusion (Test Set)')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'{name}_curves.png'), dpi=config.FIGURE_DPI)
    plt.close()


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(os.path.join(config.LOGS_DIR, f"train_multimodal_{timestamp}.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger('train_multimodal')

    logger.info("=" * 80)
    logger.info("Task 13 / Phase 3: Training Multimodal Fusion MLP")
    logger.info("=" * 80)

    # Verify Phase 1 and Phase 2 completion
    for phase, marker in [("Phase 1", "phase1_complete.txt"), ("Phase 2", "phase2_complete.txt")]:
        marker_path = os.path.join(config.LOGS_DIR, marker)
        if not os.path.exists(marker_path):
            logger.warning(f"{phase} completion marker not found at {marker_path}. Ensure it has been run.")

    data_dir = config.PROCESSED_DATA_DIR
    logger.info("Loading patient splits, structured data, and text embeddings...")

    # 1. Load splits to get ordered stay_ids
    df_stays = pd.read_csv(os.path.join(data_dir, 'structured_dataset.csv'), usecols=['subject_id', 'stay_id'])
    train_pts = set(pd.read_csv(os.path.join(data_dir, 'train_patients.csv'))['subject_id'])
    val_pts   = set(pd.read_csv(os.path.join(data_dir, 'val_patients.csv'))['subject_id'])
    test_pts  = set(pd.read_csv(os.path.join(data_dir, 'test_patients.csv'))['subject_id'])

    train_df = df_stays[df_stays['subject_id'].isin(train_pts)]
    val_df   = df_stays[df_stays['subject_id'].isin(val_pts)]
    test_df  = df_stays[df_stays['subject_id'].isin(test_pts)]

    # 2. Load Structured Data
    X_train_struct = np.load(os.path.join(data_dir, 'X_train_structured.npy'))
    X_val_struct   = np.load(os.path.join(data_dir, 'X_val_structured.npy'))
    X_test_struct  = np.load(os.path.join(data_dir, 'X_test_structured.npy'))

    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val   = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test  = np.load(os.path.join(data_dir, 'y_test.npy'))

    # 3. Load Text Embeddings
    all_embeds = np.load(os.path.join(data_dir, 'text_embeddings.npy'))
    all_ids    = np.load(os.path.join(data_dir, 'text_stay_ids.npy'))

    X_train_text = match_embeddings_to_split(train_df, all_embeds, all_ids)
    X_val_text   = match_embeddings_to_split(val_df, all_embeds, all_ids)
    X_test_text  = match_embeddings_to_split(test_df, all_embeds, all_ids)

    # 4. Early Fusion: Concatenation
    logger.info("Fusing Structured and Text features (early concatenation)...")
    X_train_multi = np.hstack([X_train_struct, X_train_text])
    X_val_multi   = np.hstack([X_val_struct, X_val_text])
    X_test_multi  = np.hstack([X_test_struct, X_test_text])

    logger.info(f"Multimodal Train shape: {X_train_multi.shape}")
    logger.info(f"Multimodal Val shape:   {X_val_multi.shape}")
    logger.info(f"Multimodal Test shape:  {X_test_multi.shape}")

    # 5. Scale using Phase 1 scaler (or fit new one on multimodal)
    logger.info("Scaling multimodal features (fit on train)...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_multi)
    X_val_s   = scaler.transform(X_val_multi)
    X_test_s  = scaler.transform(X_test_multi)

    save_model(scaler, 'multimodal_scaler', config.MODELS_DIR)

    # 6. Train MLP Classifier (per spec: hidden_layer_sizes=(256, 128, 64))
    logger.info("Training MLPClassifier (Multimodal Fusion)...")
    logger.info(f"MLP Params: {config.MLP_PARAMS}")
    mlp = MLPClassifier(**config.MLP_PARAMS)
    mlp.fit(X_train_s, y_train)
    save_model(mlp, 'fusion_model', config.MODELS_DIR)

    # Save hyperparameters including input dimensions
    fusion_params = config.MLP_PARAMS.copy()
    fusion_params['input_dim_structured'] = X_train_struct.shape[1]
    fusion_params['input_dim_text'] = X_train_text.shape[1]
    fusion_params['input_dim_total'] = X_train_multi.shape[1]
    # Convert tuple to list for JSON serialization
    fusion_params['hidden_layer_sizes'] = list(fusion_params['hidden_layer_sizes'])
    with open(os.path.join(config.MODELS_DIR, 'fusion_model_params.json'), 'w') as f:
        json.dump(fusion_params, f, indent=2)

    # 7. Evaluate on Test set
    logger.info("Evaluating Fusion MLP on Test set...")
    mlp_prob = mlp.predict_proba(X_test_s)[:, 1]
    mlp_metrics = evaluate_model(y_test, mlp_prob, 'fusion_model', config.RESULTS_DIR)

    logger.info(f"➜ Fusion MLP: ROC-AUC = {mlp_metrics['roc_auc']:.4f}, PR-AUC = {mlp_metrics['pr_auc']:.4f}, Brier = {mlp_metrics['brier_score']:.4f}")

    plot_curves(y_test, mlp_prob, 'fusion_model', config.FIGURES_DIR)

    # Document fusion of temporally valid structured + retrospective text features
    logger.info("NOTE: Phase 3 combines temporally valid structured data with retrospective discharge text embeddings.")

    # Phase completion marker
    marker_path = os.path.join(config.LOGS_DIR, 'phase3_complete.txt')
    with open(marker_path, 'w') as f:
        f.write(f"Phase 3 completed at {datetime.now().isoformat()}\n")
    logger.info(f"Phase 3 completion marker written to {marker_path}")
    logger.info("Phase 3 multimodal fusion modeling complete. Artifacts saved.")


if __name__ == '__main__':
    main()
