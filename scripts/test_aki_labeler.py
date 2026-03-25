#!/usr/bin/env python3
"""
Test script to verify AKI_Labeler creatinine itemid verification
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.aki_labeler import AKI_Labeler
import config

def test_creatinine_verification():
    """Test creatinine itemid verification."""
    print("Testing creatinine itemid verification...")
    print(f"Expected itemids: {[50912, 52546, 52024, 51081]}")
    
    labeler = AKI_Labeler(
        raw_data_dir=config.RAW_DATA_DIR,
        output_dir=config.PROCESSED_DATA_DIR
    )
    
    try:
        verified = labeler.verify_creatinine_itemids()
        print(f"\n✓ All creatinine itemids verified successfully!")
        print(f"  Verified itemids: {verified}")
        return True
    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        return False

if __name__ == "__main__":
    success = test_creatinine_verification()
    sys.exit(0 if success else 1)
