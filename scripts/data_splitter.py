#!/usr/bin/env python3
"""
Data Splitter Script
Partitions data into train/validation/test sets at patient level,
computes imputation medians, and creates missingness indicators.

Usage:
    python data_splitter.py [--processed-data-dir PROCESSED_DATA_DIR]
                            [--logs-dir LOGS_DIR]
"""

import argparse
import logging
import sys
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class Data_Splitter:
    """
    Partitions data into train/validation/test sets at patient level.
    Computes imputation and adds missingness indicators.
    """
    def __init__(self, data_dir: str, output_dir: str, seed: int = 42):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.seed = seed
        self.logger = logging.getLogger(__name__)
        self.structured_data_path = os.path.join(data_dir, 'structured_dataset.csv')

    def load_data(self) -> pd.DataFrame:
        self.logger.info("Loading structured dataset...")
        df = pd.read_csv(self.structured_data_path)
        self.logger.info(f"Loaded {len(df)} stays.")
        return df

    def split_patients(self, proportions: tuple = (0.70, 0.15, 0.15)) -> dict:
        self.logger.info("Splitting patients...")
        df = self.load_data()
        unique_patients = df['subject_id'].unique()
        
        np.random.seed(self.seed)
        shuffled = np.random.permutation(unique_patients)
        
        n_train = int(proportions[0] * len(shuffled))
        n_val = int(proportions[1] * len(shuffled))
        
        train_patients = shuffled[:n_train]
        val_patients = shuffled[n_train:n_train+n_val]
        test_patients = shuffled[n_train+n_val:]
        
        self.logger.info(f"Train patients: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")
        return {
            'train': train_patients,
            'val': val_patients,
            'test': test_patients
        }, df

    def assign_stays_to_splits(self, df: pd.DataFrame, patient_splits: dict) -> dict:
        self.logger.info("Assigning stays to splits...")
        stay_splits = {
            'train': df[df['subject_id'].isin(patient_splits['train'])].copy(),
            'val': df[df['subject_id'].isin(patient_splits['val'])].copy(),
            'test': df[df['subject_id'].isin(patient_splits['test'])].copy(),
        }
        for split_name, split_df in stay_splits.items():
            self.logger.info(f"  {split_name} stays: {len(split_df)}")
        return stay_splits

    def verify_split_integrity(self, patient_splits: dict) -> bool:
        s_train = set(patient_splits['train'])
        s_val = set(patient_splits['val'])
        s_test = set(patient_splits['test'])
        
        if len(s_train & s_val) > 0 or len(s_train & s_test) > 0 or len(s_val & s_test) > 0:
            self.logger.error("Patient overlap detected between splits!")
            return False
        self.logger.info("Split integrity verified: no patient overlap.")
        return True

    def verify_balance(self, stay_splits: dict) -> None:
        prevs = {}
        for k, v in stay_splits.items():
            prev = v['aki_label'].mean()
            prevs[k] = prev
            self.logger.info(f"  AKI prevalence {k}: {prev:.1%}")
            
        max_diff = max(prevs.values()) - min(prevs.values())
        if max_diff > config.AKI_PREVALENCE_WARNING_THRESHOLD:
            self.logger.warning(f"AKI prevalence differs by > {config.AKI_PREVALENCE_WARNING_THRESHOLD*100}% across splits (max diff: {max_diff*100:.1f}%)")

    def compute_imputation_values(self, train_df: pd.DataFrame) -> dict:
        self.logger.info("Computing imputation medians on training set...")
        features = [c for c in train_df.columns if c not in ['subject_id', 'stay_id', 'aki_label']]
        medians = {}
        for f in features:
            if train_df[f].isna().any():
                medians[f] = train_df[f].median()
                if pd.isna(medians[f]):
                    medians[f] = 0.0  # fallback if all are NaN
        return medians

    def apply_imputation(self, df: pd.DataFrame, imputation_dict: dict) -> pd.DataFrame:
        features_added = []
        for f in imputation_dict.keys():
            col_name = f + '_missing'
            if col_name not in df.columns:
                df[col_name] = df[f].isna().astype(int)
                features_added.append(col_name)
        
        for f, med in imputation_dict.items():
            df[f] = df[f].fillna(med)
            
        return df

    def save_splits(self, patient_splits: dict, stay_splits: dict) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. Save patient IDs
        for k, v in patient_splits.items():
            path = os.path.join(self.output_dir, f'{k}_patients.csv')
            pd.DataFrame({'subject_id': v}).to_csv(path, index=False)
            self.logger.info(f"  Saved {path}")
            
        # 2. Extract X and y, save as NPY
        exclude = ['subject_id', 'stay_id', 'aki_label']
        for k, v_df in stay_splits.items():
            y = v_df['aki_label'].values
            features = [c for c in v_df.columns if c not in exclude]
            X = v_df[features].values
            
            x_path = os.path.join(self.output_dir, f'X_{k}_structured.npy')
            y_path = os.path.join(self.output_dir, f'y_{k}.npy')
            np.save(x_path, X)
            np.save(y_path, y)
            self.logger.info(f"  Saved {x_path} ({X.shape}) and {y_path} ({y.shape})")
            
        # Save feature names
        names_path = os.path.join(self.output_dir, 'structured_feature_names.json')
        with open(names_path, 'w') as f:
            json.dump(features, f, indent=2)
        self.logger.info(f"  Saved {names_path} with {len(features)} features")

    def run(self) -> None:
        self.logger.info("=" * 80)
        self.logger.info("Starting Data Splitter")
        self.logger.info("=" * 80)
        
        patient_splits, df = self.split_patients((config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO))
        
        if not self.verify_split_integrity(patient_splits):
            raise ValueError("Split integrity verification failed.")
            
        stay_splits = self.assign_stays_to_splits(df, patient_splits)
        
        if sum(len(v) for v in stay_splits.values()) != len(df):
            raise ValueError("Total stays in splits does not match original dataset size.")
            
        self.verify_balance(stay_splits)
        
        imputation_dict = self.compute_imputation_values(stay_splits['train'])
        imp_path = os.path.join(self.output_dir, 'imputation_values.json')
        with open(imp_path, 'w') as f:
            json.dump(imputation_dict, f, indent=2)
        self.logger.info(f"Imputation values saved to {imp_path}")
            
        for k in stay_splits.keys():
            stay_splits[k] = self.apply_imputation(stay_splits[k], imputation_dict)
            
        self.save_splits(patient_splits, stay_splits)
        
        self.logger.info("=" * 80)
        self.logger.info("Data Splitter Complete")
        self.logger.info("=" * 80)

def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"data_splitting_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    return logger

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split structured data for AKI prediction")
    parser.add_argument('--processed-data-dir', type=str, default=config.PROCESSED_DATA_DIR,
                        help=f'Directory containing structured_dataset.csv (default: {config.PROCESSED_DATA_DIR})')
    parser.add_argument('--logs-dir', type=str, default=config.LOGS_DIR,
                        help=f'Directory for log files (default: {config.LOGS_DIR})')
    return parser.parse_args()

def main() -> int:
    args = parse_arguments()
    logger = setup_logging(args.logs_dir)
    
    try:
        splitter = Data_Splitter(
            data_dir=args.processed_data_dir,
            output_dir=args.processed_data_dir,
            seed=config.RANDOM_SEED
        )
        splitter.run()
        return 0
    except Exception as e:
        logger.error(f"Error during data splitting: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
