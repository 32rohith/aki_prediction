"""
AKI Labeler Module
Implements KDIGO-based AKI labeling with strict temporal constraints
"""

import pandas as pd
import numpy as np
import logging
from datetime import timedelta
import sys
import os

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class AKI_Labeler:
    """
    Assigns binary AKI labels to ICU stays using KDIGO criteria.
    
    KDIGO Criteria:
    - 48-hour criterion: Creatinine increase ≥0.3 mg/dL within any 48-hour window after hour 24
    - 7-day criterion: Creatinine ≥1.5x baseline within 7 days after ICU admission OR until discharge
    
    Temporal Constraints:
    - Only ICU stays ≥24 hours are included
    - AKI detection uses only creatinine measurements after hour 24
    - Baseline creatinine is computed from same-admission measurements before ICU intime
    """
    
    def __init__(self, raw_data_dir: str, output_dir: str):
        """
        Initialize AKI_Labeler with data paths.
        
        Args:
            raw_data_dir: Path to directory containing raw MIMIC-IV CSV files
            output_dir: Path to directory for output labeled_stays.csv
        """
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        
        # All blood/serum creatinine itemids verified from d_labitems.csv
        self.creatinine_itemids = [50912, 52546, 52024, 51081]
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format=config.LOG_FORMAT,
            datefmt=config.LOG_DATE_FORMAT
        )
        
        # Data containers
        self.icustays = None
        self.labevents = None
        self.patients = None
        self.d_labitems = None
        
    def verify_creatinine_itemids(self):
        """
        Verify that all creatinine itemids exist in d_labitems.csv and correspond to blood/serum creatinine.
        Raises error if any itemid is invalid.
        """
        self.logger.info("Verifying creatinine itemids...")
        
        # Load d_labitems
        d_labitems_path = os.path.join(self.raw_data_dir, "d_labitems.csv")
        self.d_labitems = pd.read_csv(d_labitems_path)
        
        # Check each creatinine itemid
        verified_itemids = []
        for itemid in self.creatinine_itemids:
            item_row = self.d_labitems[self.d_labitems['itemid'] == itemid]
            
            if len(item_row) == 0:
                raise ValueError(f"Creatinine itemid {itemid} not found in d_labitems.csv")
            
            label = item_row['label'].values[0].lower()
            
            # Verify it's a creatinine test
            if 'creatinine' not in label:
                raise ValueError(f"Itemid {itemid} does not correspond to creatinine. Label: {label}")
            
            # Log the verified itemid
            self.logger.info(f"  ✓ Verified itemid {itemid}: {item_row['label'].values[0]}")
            verified_itemids.append(itemid)
        
        self.logger.info(f"All {len(verified_itemids)} creatinine itemids verified successfully")
        self.logger.info(f"Using itemids: {verified_itemids}")
        
        return verified_itemids

    def load_data(self):
        """Load required CSV files."""
        self.logger.info("Loading data files...")
        
        # Load ICU stays
        icustays_path = os.path.join(self.raw_data_dir, "icustays.csv")
        self.icustays = pd.read_csv(icustays_path, parse_dates=['intime', 'outtime'])
        self.logger.info(f"  Loaded {len(self.icustays)} ICU stays")
        
        # Load patients
        patients_path = os.path.join(self.raw_data_dir, "patients.csv")
        self.patients = pd.read_csv(patients_path)
        self.logger.info(f"  Loaded {len(self.patients)} patients")
        
        self.logger.info("Data loading complete")
    
    def filter_by_duration(self, stays_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter ICU stays to include only those with duration >= 24 hours.
        
        Args:
            stays_df: DataFrame with ICU stays containing intime and outtime columns
            
        Returns:
            Filtered DataFrame with duration_hours column added
        """
        self.logger.info("Filtering ICU stays by duration...")
        
        initial_count = len(stays_df)
        
        # Compute duration in hours
        stays_df = stays_df.copy()
        stays_df['duration_hours'] = (stays_df['outtime'] - stays_df['intime']).dt.total_seconds() / 3600
        
        # Filter stays >= 24 hours
        stays_filtered = stays_df[stays_df['duration_hours'] >= config.MIN_ICU_DURATION_HOURS].copy()
        
        excluded_count = initial_count - len(stays_filtered)
        retained_count = len(stays_filtered)
        
        self.logger.info(f"  Excluded {excluded_count} stays with duration < 24 hours")
        self.logger.info(f"  Retained {retained_count} stays with duration >= 24 hours")
        self.logger.info(f"  Retention rate: {100 * retained_count / initial_count:.1f}%")
        
        return stays_filtered

    def load_creatinine_data(self):
        """
        Load creatinine measurements from labevents.csv in chunks.
        Returns DataFrame with creatinine measurements only.
        """
        self.logger.info("Loading creatinine measurements from labevents.csv...")
        self.logger.info(f"  Using creatinine itemids: {self.creatinine_itemids}")
        self.logger.info("  Processing in chunks (this may take several minutes)...")
        
        labevents_path = os.path.join(self.raw_data_dir, "labevents.csv")
        
        # Process in chunks to handle large file
        chunk_size = config.CHUNK_SIZE
        creatinine_chunks = []
        total_rows = 0
        creatinine_count = 0
        
        for i, chunk in enumerate(pd.read_csv(labevents_path, chunksize=chunk_size)):
            total_rows += len(chunk)
            
            # Filter for creatinine measurements
            creat_chunk = chunk[chunk['itemid'].isin(self.creatinine_itemids)].copy()
            
            if len(creat_chunk) > 0:
                creatinine_chunks.append(creat_chunk)
                creatinine_count += len(creat_chunk)
            
            # Progress update every 1 million rows
            if (i + 1) % 100 == 0:
                self.logger.info(f"    Processed {total_rows:,} rows, found {creatinine_count:,} creatinine measurements")
        
        # Combine all chunks
        self.labevents = pd.concat(creatinine_chunks, ignore_index=True)
        
        # Parse charttime as datetime
        self.labevents['charttime'] = pd.to_datetime(self.labevents['charttime'])
        
        # Use valuenum for numeric creatinine values
        self.labevents = self.labevents[self.labevents['valuenum'].notna()].copy()
        
        self.logger.info(f"  Loaded {len(self.labevents):,} creatinine measurements from {total_rows:,} total lab events")
        self.logger.info(f"  Creatinine coverage: {100 * len(self.labevents) / total_rows:.3f}%")
        
        return self.labevents
    
    def compute_baseline_creatinine(self, stay_row: pd.Series, creatinine_df: pd.DataFrame) -> float:
        """
        Compute baseline creatinine for a single ICU stay.
        
        Baseline logic (same-admission only):
        1. Query creatinine with same hadm_id and charttime < intime
        2. If measurements exist before intime: baseline = minimum value
        3. If no measurements before intime: baseline = first creatinine during ICU stay
        
        Args:
            stay_row: Series with stay_id, hadm_id, intime, outtime
            creatinine_df: DataFrame with all creatinine measurements
            
        Returns:
            Baseline creatinine value (mg/dL), or None if cannot be determined
        """
        stay_id = stay_row['stay_id']
        hadm_id = stay_row['hadm_id']
        intime = stay_row['intime']
        outtime = stay_row['outtime']
        
        # Get all creatinine for this hospital admission
        admission_creat = creatinine_df[creatinine_df['hadm_id'] == hadm_id].copy()
        
        if len(admission_creat) == 0:
            self.logger.warning(f"  Stay {stay_id}: No creatinine measurements found for admission {hadm_id}")
            return None
        
        # Filter for measurements before ICU intime (same admission)
        before_icu = admission_creat[admission_creat['charttime'] < intime]
        
        if len(before_icu) > 0:
            # Baseline = minimum creatinine before ICU admission
            baseline = before_icu['valuenum'].min()
            return baseline
        else:
            # No measurements before ICU intime - use first creatinine during ICU stay
            during_icu = admission_creat[
                (admission_creat['charttime'] >= intime) & 
                (admission_creat['charttime'] <= outtime)
            ].sort_values('charttime')
            
            if len(during_icu) > 0:
                baseline = during_icu.iloc[0]['valuenum']
                return baseline
            else:
                self.logger.warning(f"  Stay {stay_id}: No creatinine measurements during ICU stay")
                return None

    def check_48h_criterion(self, stay_row: pd.Series, creatinine_df: pd.DataFrame) -> bool:
        """
        Check if 48-hour rolling window AKI criterion is satisfied.
        
        Criterion: Creatinine INCREASE >= 0.3 mg/dL within any 48-hour window after hour 24.
        
        CRITICAL: This checks for INCREASE only (cr_j - cr_i >= 0.3 where time_j > time_i),
        NOT absolute difference. Decreases should NOT trigger AKI.
        
        Args:
            stay_row: Series with stay_id, hadm_id, intime, outtime
            creatinine_df: DataFrame with all creatinine measurements
            
        Returns:
            True if criterion satisfied, False otherwise
        """
        stay_id = stay_row['stay_id']
        hadm_id = stay_row['hadm_id']
        intime = stay_row['intime']
        outtime = stay_row['outtime']
        
        # Get creatinine measurements for this admission
        admission_creat = creatinine_df[creatinine_df['hadm_id'] == hadm_id].copy()
        
        # Filter to measurements after hour 24 of ICU stay
        cutoff_time = intime + timedelta(hours=config.TEMPORAL_CUTOFF_HOURS)
        after_24h = admission_creat[
            (admission_creat['charttime'] >= cutoff_time) &
            (admission_creat['charttime'] <= outtime)
        ].sort_values('charttime')
        
        if len(after_24h) < 2:
            # Need at least 2 measurements to check for increase
            return False
        
        # Check all pairs within 48-hour window for increase >= 0.3
        creat_values = after_24h['valuenum'].values
        creat_times = after_24h['charttime'].values
        
        for i in range(len(creat_values)):
            for j in range(i + 1, len(creat_values)):
                time_diff_hours = (creat_times[j] - creat_times[i]) / np.timedelta64(1, 'h')
                
                # Check if within 48-hour window
                if time_diff_hours <= 48:
                    # Check for INCREASE (cr_j - cr_i >= 0.3)
                    increase = creat_values[j] - creat_values[i]
                    
                    if increase >= config.AKI_48H_INCREASE_THRESHOLD:
                        return True
                else:
                    # Times are sorted, so if time_diff > 48h, no need to check further j values for this i
                    break
        
        return False

    def check_7d_criterion(self, stay_row: pd.Series, baseline_creatinine: float, creatinine_df: pd.DataFrame) -> bool:
        """
        Check if 7-day criterion is satisfied.
        
        Criterion: Any creatinine >= 1.5 * baseline within 7 days after ICU admission OR until discharge.
        
        Args:
            stay_row: Series with stay_id, hadm_id, intime, outtime
            baseline_creatinine: Baseline creatinine value
            creatinine_df: DataFrame with all creatinine measurements
            
        Returns:
            True if criterion satisfied, False otherwise
        """
        stay_id = stay_row['stay_id']
        hadm_id = stay_row['hadm_id']
        intime = stay_row['intime']
        outtime = stay_row['outtime']
        
        # Get creatinine measurements for this admission
        admission_creat = creatinine_df[creatinine_df['hadm_id'] == hadm_id].copy()
        
        # Define evaluation window: after hour 24, up to min(outtime, intime + 7 days)
        cutoff_time = intime + timedelta(hours=config.TEMPORAL_CUTOFF_HOURS)
        window_end = min(outtime, intime + timedelta(days=7))
        
        # Filter to measurements in evaluation window
        eval_window = admission_creat[
            (admission_creat['charttime'] >= cutoff_time) &
            (admission_creat['charttime'] <= window_end)
        ]
        
        if len(eval_window) == 0:
            return False
        
        # Check if any measurement >= 1.5 * baseline
        threshold = config.AKI_7D_RATIO_THRESHOLD * baseline_creatinine
        max_creatinine = eval_window['valuenum'].max()
        
        return max_creatinine >= threshold

    def label_all_stays(self, stays_df: pd.DataFrame, creatinine_df: pd.DataFrame) -> pd.DataFrame:
        """
        Label all ICU stays with AKI using KDIGO criteria.
        
        Label assignment:
        - aki_label = 1 if (48h criterion OR 7d criterion)
        - aki_label = 0 if neither criterion satisfied
        
        Args:
            stays_df: DataFrame with filtered ICU stays (>= 24h duration)
            creatinine_df: DataFrame with all creatinine measurements
            
        Returns:
            DataFrame with columns: subject_id, hadm_id, stay_id, intime, outtime, 
                                   duration_hours, baseline_creatinine, aki_label
        """
        self.logger.info(f"Labeling {len(stays_df)} ICU stays with AKI criteria...")
        
        results = []
        no_baseline_count = 0
        aki_48h_count = 0
        aki_7d_count = 0
        aki_both_count = 0
        
        for counter, (idx, stay_row) in enumerate(stays_df.iterrows(), start=1):
            stay_id = stay_row['stay_id']
            
            # Progress update every 1000 stays
            if counter % 1000 == 0:
                self.logger.info(f"  Processed {counter}/{len(stays_df)} stays...")
            
            # Compute baseline creatinine
            baseline = self.compute_baseline_creatinine(stay_row, creatinine_df)
            
            if baseline is None:
                no_baseline_count += 1
                continue  # Skip stays without baseline
            
            # Check 48-hour criterion
            criterion_48h = self.check_48h_criterion(stay_row, creatinine_df)
            
            # Check 7-day criterion
            criterion_7d = self.check_7d_criterion(stay_row, baseline, creatinine_df)
            
            # Assign label
            aki_label = 1 if (criterion_48h or criterion_7d) else 0
            
            # Track criterion statistics
            if criterion_48h and criterion_7d:
                aki_both_count += 1
            elif criterion_48h:
                aki_48h_count += 1
            elif criterion_7d:
                aki_7d_count += 1
            
            # Store result
            results.append({
                'subject_id': stay_row['subject_id'],
                'hadm_id': stay_row['hadm_id'],
                'stay_id': stay_id,
                'intime': stay_row['intime'],
                'outtime': stay_row['outtime'],
                'duration_hours': stay_row['duration_hours'],
                'baseline_creatinine': baseline,
                'aki_label': aki_label
            })
        
        # Create results DataFrame
        labeled_df = pd.DataFrame(results)
        
        # Log statistics
        self.logger.info(f"\nLabeling complete:")
        self.logger.info(f"  Total stays processed: {len(stays_df)}")
        self.logger.info(f"  Stays without baseline creatinine: {no_baseline_count}")
        self.logger.info(f"  Stays with valid labels: {len(labeled_df)}")
        self.logger.info(f"\nAKI Statistics:")
        aki_count = (labeled_df['aki_label'] == 1).sum()
        no_aki_count = (labeled_df['aki_label'] == 0).sum()
        aki_prevalence = 100 * aki_count / len(labeled_df) if len(labeled_df) > 0 else 0
        self.logger.info(f"  AKI cases (label=1): {aki_count} ({aki_prevalence:.1f}%)")
        self.logger.info(f"  No AKI (label=0): {no_aki_count} ({100 - aki_prevalence:.1f}%)")
        self.logger.info(f"\nAKI Criterion Breakdown:")
        self.logger.info(f"  48h criterion only: {aki_48h_count}")
        self.logger.info(f"  7d criterion only: {aki_7d_count}")
        self.logger.info(f"  Both criteria: {aki_both_count}")
        self.logger.info(f"\nBaseline Creatinine Statistics:")
        self.logger.info(f"  Mean: {labeled_df['baseline_creatinine'].mean():.2f} mg/dL")
        self.logger.info(f"  Median: {labeled_df['baseline_creatinine'].median():.2f} mg/dL")
        self.logger.info(f"  Std: {labeled_df['baseline_creatinine'].std():.2f} mg/dL")
        self.logger.info(f"  Min: {labeled_df['baseline_creatinine'].min():.2f} mg/dL")
        self.logger.info(f"  Max: {labeled_df['baseline_creatinine'].max():.2f} mg/dL")
        
        return labeled_df
    
    def save_labeled_stays(self, labeled_df: pd.DataFrame):
        """
        Save labeled stays to CSV file.
        
        Args:
            labeled_df: DataFrame with labeled ICU stays
        """
        output_path = os.path.join(self.output_dir, "labeled_stays.csv")
        labeled_df.to_csv(output_path, index=False)
        self.logger.info(f"\nSaved labeled stays to: {output_path}")
        self.logger.info(f"Output shape: {labeled_df.shape}")

    def run(self):
        """
        Execute complete AKI labeling pipeline.
        
        Steps:
        1. Verify creatinine itemids
        2. Load data files
        3. Load creatinine measurements
        4. Filter ICU stays by duration
        5. Label all stays with AKI
        6. Save labeled stays
        """
        self.logger.info("="*80)
        self.logger.info("Starting AKI Labeling Pipeline")
        self.logger.info("="*80)
        
        # Step 1: Verify creatinine itemids
        self.verify_creatinine_itemids()
        
        # Step 2: Load data files
        self.load_data()
        
        # Step 3: Load creatinine measurements
        creatinine_df = self.load_creatinine_data()
        
        # Step 4: Filter ICU stays by duration
        filtered_stays = self.filter_by_duration(self.icustays)
        
        # Step 5: Label all stays
        labeled_stays = self.label_all_stays(filtered_stays, creatinine_df)
        
        # Step 6: Save results
        self.save_labeled_stays(labeled_stays)
        
        self.logger.info("="*80)
        self.logger.info("AKI Labeling Pipeline Complete")
        self.logger.info("="*80)
        
        # Document same-admission baseline limitation
        self.logger.info("\nMethodological Note:")
        self.logger.info("  Baseline creatinine is restricted to same-admission measurements")
        self.logger.info("  (same hadm_id) because MIMIC-IV does not reliably capture")
        self.logger.info("  outpatient history. This approach is reproducible and clinically")
        self.logger.info("  standard for ICU-based AKI research.")
