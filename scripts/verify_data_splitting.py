#!/usr/bin/env python3
"""
Task 9: Checkpoint - Verify Data Splitting
This script runs a formal verification of the data splitting outputs to ensure
correct shapes, non-overlapping patients, and proper imputation across train/val/test splits.
"""

import os
import sys
import numpy as np
import pandas as pd
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def verify_data_splits():
    print("="*80)
    print("Task 9 Checkpoint: Verifying Data Splitting Outputs")
    print("="*80)
    
    data_dir = config.PROCESSED_DATA_DIR
    
    # 1. Load patient DataFrames
    print("\n1. Verifying Patient Partitions...")
    train_df = pd.read_csv(os.path.join(data_dir, 'train_patients.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val_patients.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_patients.csv'))
    
    print(f"  Train patients: {len(train_df):,}")
    print(f"  Val patients:   {len(val_df):,}")
    print(f"  Test patients:  {len(test_df):,}")
    
    # Check for leakage
    train_subjects = set(train_df['subject_id'])
    val_subjects = set(val_df['subject_id'])
    test_subjects = set(test_df['subject_id'])
    
    leakage_train_val = train_subjects.intersection(val_subjects)
    leakage_train_test = train_subjects.intersection(test_subjects)
    leakage_val_test = val_subjects.intersection(test_subjects)
    
    print(f"  Patient Overlap Train/Val:  {len(leakage_train_val)}")
    print(f"  Patient Overlap Train/Test: {len(leakage_train_test)}")
    print(f"  Patient Overlap Val/Test:   {len(leakage_val_test)}")
    
    assert len(leakage_train_val) == 0, "Data Leakage detected between Train and Val!"
    assert len(leakage_train_test) == 0, "Data Leakage detected between Train and Test!"
    assert len(leakage_val_test) == 0, "Data Leakage detected between Val and Test!"
    
    # 2. Check Numpy Arrays
    print("\n2. Verifying Serialized NumPy Arrays...")
    X_train = np.load(os.path.join(data_dir, 'X_train_structured.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val_structured.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test_structured.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"  X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")
    print(f"  X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")
    
    assert X_train.shape[0] == y_train.shape[0], "Train shape mismatch!"
    assert X_val.shape[0] == y_val.shape[0], "Val shape mismatch!"
    assert X_test.shape[0] == y_test.shape[0], "Test shape mismatch!"
    
    # Ensure no NaN values exist in the arrays (Imputation should have handled them)
    print("\n3. Verifying Imputation (No NaNs)...")
    train_nans = np.isnan(X_train).sum()
    val_nans = np.isnan(X_val).sum()
    test_nans = np.isnan(X_test).sum()
    
    print(f"  NaNs in X_train: {train_nans}")
    print(f"  NaNs in X_val:   {val_nans}")
    print(f"  NaNs in X_test:  {test_nans}")
    
    assert train_nans == 0, "NaNs found in X_train! Imputation failed."
    assert val_nans == 0, "NaNs found in X_val! Imputation failed."
    assert test_nans == 0, "NaNs found in X_test! Imputation failed."
    
    # Check labels distribution
    print("\n4. Verifying Label Distribution (AKI Prevalence)...")
    print(f"  Train AKI Prevalence: {y_train.mean():.2%}")
    print(f"  Val AKI Prevalence:   {y_val.mean():.2%}")
    print(f"  Test AKI Prevalence:  {y_test.mean():.2%}")
    
    print("\n✅ Task 9 Checkpoint VERIFIED SUCCESSFULLY!")
    print("="*80)

if __name__ == '__main__':
    verify_data_splits()
