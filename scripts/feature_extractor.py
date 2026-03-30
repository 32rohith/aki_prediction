"""
Feature Extractor Module
Combines demographic features, baseline creatinine, and lab aggregations
into a single structured dataset for AKI prediction.
"""

import pandas as pd
import numpy as np
import logging
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from scripts.lab_aggregator import Lab_Aggregator


class Feature_Extractor:
    """
    Extracts and combines all structured features for AKI prediction:
      - Demographics: age (anchor_age), gender (binary), ICU type (one-hot)
      - Baseline creatinine (from labeled_stays.csv)
      - Lab aggregations from first 24 hours (via Lab_Aggregator)

    Requirements: 10.1-10.7, 11, 1.1, 1.3, 19, 20, 21
    """

    def __init__(self, raw_data_dir: str, processed_data_dir: str, logs_dir: str):
        """
        Initialize Feature_Extractor.

        Args:
            raw_data_dir: Path to directory containing raw MIMIC-IV CSV files
            processed_data_dir: Path to processed_data directory
            logs_dir: Path to logs directory
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.logs_dir = logs_dir

        self.labeled_stays_path = os.path.join(processed_data_dir, 'labeled_stays.csv')

        self.logger = logging.getLogger(__name__)

        # Data containers
        self._labeled_stays: pd.DataFrame | None = None
        self._patients: pd.DataFrame | None = None
        self._icustays: pd.DataFrame | None = None

        # Lab aggregator (initialised in run())
        self._lab_aggregator: Lab_Aggregator | None = None

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_labeled_stays(self) -> pd.DataFrame:
        if self._labeled_stays is None:
            self.logger.info("Loading labeled_stays.csv ...")
            self._labeled_stays = pd.read_csv(
                self.labeled_stays_path,
                parse_dates=['intime', 'outtime']
            )
            self.logger.info(f"  Loaded {len(self._labeled_stays):,} labeled stays")
        return self._labeled_stays

    def _load_patients(self) -> pd.DataFrame:
        if self._patients is None:
            path = os.path.join(self.raw_data_dir, 'patients.csv')
            self._patients = pd.read_csv(path)
            self.logger.info(f"  Loaded {len(self._patients):,} patients")
        return self._patients

    def _load_icustays(self) -> pd.DataFrame:
        if self._icustays is None:
            path = os.path.join(self.raw_data_dir, 'icustays.csv')
            self._icustays = pd.read_csv(path, parse_dates=['intime', 'outtime'])
            self.logger.info(f"  Loaded {len(self._icustays):,} ICU stays")
        return self._icustays

    # ------------------------------------------------------------------
    # Demographic feature extraction (Task 7.3)
    # ------------------------------------------------------------------

    def extract_demographics(self) -> pd.DataFrame:
        """
        Extract demographic features for all labeled stays.

        Features:
          - age: anchor_age from patients.csv (integer age at anchor_year)
          - gender_binary: 1 if 'M', 0 if 'F'
          - icu_type_onehot_*: one-hot encoding of first_careunit

        Returns:
            DataFrame with stay_id, age, gender_binary, icu_type_onehot_* columns
        """
        self.logger.info("Extracting demographic features ...")

        labeled_stays = self._load_labeled_stays()
        patients = self._load_patients()
        icustays = self._load_icustays()

        # Merge labeled stays with patients on subject_id
        demo = labeled_stays[['stay_id', 'subject_id']].merge(
            patients[['subject_id', 'gender', 'anchor_age']],
            on='subject_id',
            how='left'
        )

        # Age: use anchor_age directly (MIMIC-IV does not expose date_of_birth)
        demo['age'] = demo['anchor_age']

        # Gender binary: 1 = Male, 0 = Female
        demo['gender_binary'] = (demo['gender'].str.upper() == 'M').astype(int)

        # ICU type: first_careunit from icustays
        icu_type = icustays[['stay_id', 'first_careunit']].drop_duplicates('stay_id')
        demo = demo.merge(icu_type, on='stay_id', how='left')

        # One-hot encode ICU type
        icu_dummies = pd.get_dummies(
            demo['first_careunit'],
            prefix='icu_type',
            dtype=int
        )
        demo = pd.concat([demo, icu_dummies], axis=1)

        # Keep only relevant columns
        keep_cols = ['stay_id', 'age', 'gender_binary'] + list(icu_dummies.columns)
        demo = demo[keep_cols].copy()

        missing_age = demo['age'].isna().sum()
        if missing_age > 0:
            self.logger.warning(f"  {missing_age} stays have missing age")

        self.logger.info(
            f"  Demographics extracted. Shape: {demo.shape}  "
            f"ICU types: {list(icu_dummies.columns)}"
        )
        return demo

    # ------------------------------------------------------------------
    # Combined feature extraction (Task 7.4)
    # ------------------------------------------------------------------

    def extract_all_features(self) -> pd.DataFrame:
        """
        Combine demographics, baseline creatinine, and lab aggregations
        into a single DataFrame.

        Includes identifiers: subject_id, stay_id, aki_label.

        Returns:
            DataFrame with all structured features
        """
        self.logger.info("Extracting all structured features ...")

        labeled_stays = self._load_labeled_stays()

        # 1. Demographics
        demographics = self.extract_demographics()

        # 2. Baseline creatinine + identifiers from labeled_stays
        base_cols = labeled_stays[['subject_id', 'stay_id', 'aki_label', 'baseline_creatinine']].copy()

        # 3. Lab aggregations
        if self._lab_aggregator is None:
            self._lab_aggregator = Lab_Aggregator(
                raw_data_dir=self.raw_data_dir,
                labeled_stays_path=self.labeled_stays_path,
                logs_dir=self.logs_dir,
            )

        self._lab_aggregator.select_labs(min_coverage=config.LAB_COVERAGE_THRESHOLD)
        lab_features = self._lab_aggregator.aggregate_all_stays(
            labeled_stays[['stay_id', 'intime']]
        )

        # 4. Merge everything on stay_id
        combined = base_cols.merge(demographics, on='stay_id', how='left')
        combined = combined.merge(lab_features, on='stay_id', how='left')

        # Reorder: identifiers first
        id_cols = ['subject_id', 'stay_id', 'aki_label']
        other_cols = [c for c in combined.columns if c not in id_cols]
        combined = combined[id_cols + other_cols]

        self.logger.info(f"  Combined feature matrix shape: {combined.shape}")
        self.logger.info(f"  Total features (excl. identifiers): {len(other_cols)}")

        # Log feature extraction statistics
        self._log_feature_stats(combined)

        return combined

    def _log_feature_stats(self, df: pd.DataFrame) -> None:
        """Log feature extraction statistics to logs directory."""
        os.makedirs(self.logs_dir, exist_ok=True)
        log_path = os.path.join(self.logs_dir, 'feature_extraction.log')

        id_cols = {'subject_id', 'stay_id', 'aki_label'}
        feature_cols = [c for c in df.columns if c not in id_cols]

        missing_rates = df[feature_cols].isna().mean().sort_values(ascending=False)
        high_missing = missing_rates[missing_rates > 0.70]

        lines = [
            "Feature Extraction Statistics",
            "=" * 60,
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total stays: {len(df):,}",
            f"Total features: {len(feature_cols)}",
            f"AKI prevalence: {df['aki_label'].mean():.1%}",
            "",
            "Missing rate summary:",
            f"  Features with >70% missing: {len(high_missing)}",
            f"  Features with any missing: {(missing_rates > 0).sum()}",
            "",
            "Top 10 features by missing rate:",
        ]
        for feat, rate in missing_rates.head(10).items():
            lines.append(f"  {feat:<60} {rate:.1%}")

        if len(high_missing) > 0:
            lines.append("")
            lines.append("WARNING: Features with >70% missing rate:")
            for feat, rate in high_missing.items():
                lines.append(f"  {feat}: {rate:.1%}")

        with open(log_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')

        self.logger.info(f"  Feature extraction stats logged to: {log_path}")
        self.logger.info(f"  Total features: {len(feature_cols)}")

        if len(high_missing) > 0:
            for feat, rate in high_missing.items():
                self.logger.warning(f"  High missingness (>70%): {feat} = {rate:.1%}")

    def save_structured_dataset(self, df: pd.DataFrame) -> str:
        """
        Save the structured dataset to processed_data/structured_dataset.csv.

        Args:
            df: Combined feature DataFrame

        Returns:
            Path to saved file
        """
        os.makedirs(self.processed_data_dir, exist_ok=True)
        output_path = os.path.join(self.processed_data_dir, 'structured_dataset.csv')
        df.to_csv(output_path, index=False)
        self.logger.info(f"  Saved structured dataset to: {output_path}")
        self.logger.info(f"  Shape: {df.shape}")
        return output_path

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the full feature extraction pipeline:
          1. Extract demographics
          2. Aggregate lab features
          3. Combine all features
          4. Save to processed_data/structured_dataset.csv
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Feature Extraction Pipeline")
        self.logger.info("=" * 80)

        df = self.extract_all_features()
        self.save_structured_dataset(df)

        self.logger.info("=" * 80)
        self.logger.info("Feature Extraction Pipeline Complete")
        self.logger.info("=" * 80)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract structured features")
    parser.add_argument('--raw-data-dir', type=str, default=config.RAW_DATA_DIR)
    parser.add_argument('--processed-data-dir', type=str, default=config.PROCESSED_DATA_DIR)
    parser.add_argument('--logs-dir', type=str, default=config.LOGS_DIR)
    args = parser.parse_args()
    
    os.makedirs(args.logs_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    extractor = Feature_Extractor(args.raw_data_dir, args.processed_data_dir, args.logs_dir)
    extractor.run()

if __name__ == '__main__':
    main()
