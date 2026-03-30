"""
Lab Aggregator Module
Selects clinically relevant laboratory tests and computes temporal aggregations
for the first 24 hours of ICU admission.
"""

import pandas as pd
import numpy as np
import logging
import os
import sys
from datetime import timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# AKI-relevant lab name substrings for case-insensitive matching
AKI_RELEVANT_SUBSTRINGS = [
    'urea nitrogen',  # BUN
    'bun',
    'sodium',
    'potassium',
    'chloride',
    'bicarbonate',
    'lactate',
    'white blood cell',  # WBC
    'wbc',
    'hemoglobin',
    'platelet',
    'glucose',
    'calcium',
    'magnesium',
    'phosphate',
]

# Creatinine itemid to exclude from aggregated features
CREATININE_ITEMID = 50912


class Lab_Aggregator:
    """
    Selects laboratory tests with sufficient coverage and computes temporal
    aggregations within the first 24 hours of ICU admission.

    Requirements: 11, 21, 1.1, 1.3, 10.5
    """

    def __init__(self, raw_data_dir: str, labeled_stays_path: str, logs_dir: str):
        """
        Initialize Lab_Aggregator.

        Args:
            raw_data_dir: Path to directory containing raw MIMIC-IV CSV files
            labeled_stays_path: Path to processed_data/labeled_stays.csv
            logs_dir: Path to logs directory
        """
        self.raw_data_dir = raw_data_dir
        self.labeled_stays_path = labeled_stays_path
        self.logs_dir = logs_dir

        self.logger = logging.getLogger(__name__)

        # Data containers (populated lazily)
        self._labeled_stays: pd.DataFrame | None = None
        self._d_labitems: pd.DataFrame | None = None
        self._labevents: pd.DataFrame | None = None
        self._selected_labs: list | None = None  # list of dicts

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

    def _load_d_labitems(self) -> pd.DataFrame:
        if self._d_labitems is None:
            path = os.path.join(self.raw_data_dir, 'd_labitems.csv')
            self._d_labitems = pd.read_csv(path)
            self.logger.info(f"  Loaded d_labitems with {len(self._d_labitems):,} items")
        return self._d_labitems

    def _load_labevents(self) -> pd.DataFrame:
        """Load labevents.csv in chunks, keeping only numeric values."""
        if self._labevents is not None:
            return self._labevents

        labeled_stays = self._load_labeled_stays()
        valid_hadm_ids = set(labeled_stays['hadm_id'].dropna().unique())
        
        path = os.path.join(self.raw_data_dir, 'labevents.csv')
        self.logger.info("Loading labevents.csv in large chunks ...")

        chunks = []
        total_rows = 0
        # Use 1,000,000 chunksize instead of config.CHUNK_SIZE for dramatic speedup
        for chunk in pd.read_csv(path, chunksize=1_000_000):
            total_rows += len(chunk)
            # Pre-filter by admission ID
            if 'hadm_id' in chunk.columns:
                chunk = chunk[chunk['hadm_id'].isin(valid_hadm_ids)]
            chunk = chunk[chunk['valuenum'].notna()].copy()
            
            if len(chunk) > 0:
                chunks.append(chunk)
                
            if total_rows % 5_000_000 == 0:
                self.logger.info(f"  Processed {total_rows:,} rows ...")

        self.logger.info("Concatenating chunks...")
        self._labevents = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        self._labevents['charttime'] = pd.to_datetime(self._labevents['charttime'])
        
        # Merge with labeled_stays to get stay_id and intime
        self.logger.info("Merging with labeled stays to assign stay_id...")
        stay_mapping = labeled_stays[['hadm_id', 'stay_id', 'intime']]
        self._labevents = self._labevents.merge(stay_mapping, on='hadm_id', how='inner')
        
        # Temporal filtering to reduce memory footprint
        self.logger.info("Applying strict temporal filtering (< 24h)...")
        cutoff = self._labevents['intime'] + pd.Timedelta(hours=config.TEMPORAL_CUTOFF_HOURS)
        self._labevents = self._labevents[self._labevents['charttime'] < cutoff]
        self._labevents.drop(columns=['intime'], inplace=True)
        
        self.logger.info(
            f"  Loaded {len(self._labevents):,} relevant lab events "
            f"(from {total_rows:,} total rows)"
        )
        return self._labevents

    # ------------------------------------------------------------------
    # Coverage computation
    # ------------------------------------------------------------------

    def compute_lab_coverage(self) -> pd.DataFrame:
        """
        Compute coverage percentage for each lab test type.

        Coverage = count(distinct stay_ids with that lab) / total_stays

        Returns:
            DataFrame with columns: itemid, label, coverage_pct
        """
        self.logger.info("Computing lab coverage ...")

        labeled_stays = self._load_labeled_stays()
        labevents = self._load_labevents()
        d_labitems = self._load_d_labitems()

        total_stays = len(labeled_stays)
        stay_ids = set(labeled_stays['stay_id'].unique())

        # Filter labevents to only stays in our cohort
        cohort_labs = labevents[labevents['stay_id'].isin(stay_ids)].copy()

        # Count distinct stays per itemid
        coverage_df = (
            cohort_labs.groupby('itemid')['stay_id']
            .nunique()
            .reset_index()
            .rename(columns={'stay_id': 'stay_count'})
        )
        coverage_df['coverage_pct'] = coverage_df['stay_count'] / total_stays

        # Merge with d_labitems to get labels
        coverage_df = coverage_df.merge(
            d_labitems[['itemid', 'label']],
            on='itemid',
            how='left'
        )

        coverage_df = coverage_df.sort_values('coverage_pct', ascending=False).reset_index(drop=True)
        self.logger.info(f"  Computed coverage for {len(coverage_df):,} lab types")
        return coverage_df[['itemid', 'label', 'coverage_pct']]

    # ------------------------------------------------------------------
    # Lab selection
    # ------------------------------------------------------------------

    def _is_aki_relevant(self, label: str) -> bool:
        """Return True if the lab label matches any AKI-relevant substring."""
        if not isinstance(label, str):
            return False
        label_lower = label.lower()
        return any(sub in label_lower for sub in AKI_RELEVANT_SUBSTRINGS)

    def select_labs(self, min_coverage: float = 0.30) -> list:
        """
        Select labs with coverage >= min_coverage, excluding creatinine.
        AKI-relevant labs are prioritised (listed first).

        Verifies all selected itemids against d_labitems.csv.

        Args:
            min_coverage: Minimum coverage fraction (default 0.30)

        Returns:
            List of dicts: [{itemid, label, coverage_pct}, ...]
        """
        self.logger.info(f"Selecting labs with coverage >= {min_coverage:.0%} ...")

        coverage_df = self.compute_lab_coverage()
        d_labitems = self._load_d_labitems()

        # Apply coverage threshold
        eligible = coverage_df[coverage_df['coverage_pct'] >= min_coverage].copy()
        self.logger.info(f"  Labs meeting coverage threshold: {len(eligible)}")

        # Exclude creatinine
        eligible = eligible[eligible['itemid'] != CREATININE_ITEMID].copy()
        self.logger.info(f"  After excluding creatinine (itemid {CREATININE_ITEMID}): {len(eligible)}")

        # Verify all itemids against d_labitems
        valid_itemids = set(d_labitems['itemid'].unique())
        invalid = eligible[~eligible['itemid'].isin(valid_itemids)]
        if len(invalid) > 0:
            self.logger.warning(
                f"  Removing {len(invalid)} itemids not found in d_labitems: "
                f"{invalid['itemid'].tolist()}"
            )
            eligible = eligible[eligible['itemid'].isin(valid_itemids)].copy()

        # Tag AKI-relevant labs
        eligible['aki_relevant'] = eligible['label'].apply(self._is_aki_relevant)

        # Sort: AKI-relevant first, then by coverage descending
        eligible = eligible.sort_values(
            ['aki_relevant', 'coverage_pct'],
            ascending=[False, False]
        ).reset_index(drop=True)

        selected = eligible[['itemid', 'label', 'coverage_pct']].to_dict('records')
        self._selected_labs = selected

        # Log selected labs
        self._log_selected_labs(selected)

        return selected

    def _log_selected_labs(self, selected: list) -> None:
        """Write selected labs to logs directory."""
        os.makedirs(self.logs_dir, exist_ok=True)
        log_path = os.path.join(self.logs_dir, 'selected_labs.log')

        lines = [
            "Selected Laboratory Tests for Feature Extraction",
            "=" * 60,
            f"Total selected: {len(selected)}",
            "",
            f"{'itemid':<10} {'coverage_pct':<15} {'label'}",
            "-" * 60,
        ]
        for lab in selected:
            lines.append(
                f"{lab['itemid']:<10} {lab['coverage_pct']:<15.3%} {lab['label']}"
            )

        with open(log_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')

        self.logger.info(f"  Selected labs logged to: {log_path}")
        self.logger.info(f"  Total selected labs: {len(selected)}")
        for lab in selected:
            self.logger.info(
                f"    itemid={lab['itemid']:>6}  coverage={lab['coverage_pct']:.1%}  {lab['label']}"
            )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_labs_for_stay(
        self,
        stay_id: int,
        intime: pd.Timestamp,
        lab_measurements_df: pd.DataFrame,
    ) -> dict:
        """
        Compute lab aggregations for a single ICU stay within the first 24 hours.

        Temporal cutoff: charttime < intime + 24h

        For each selected lab computes: mean, min, max, std, first, last.
        Creates a missingness indicator (1 = no measurements, 0 = present).

        Args:
            stay_id: ICU stay identifier
            intime: ICU admission time
            lab_measurements_df: DataFrame of lab events for this stay (pre-filtered)

        Returns:
            Dict of feature_name -> value
        """
        if self._selected_labs is None:
            raise RuntimeError("Call select_labs() before aggregate_labs_for_stay()")

        cutoff = intime + timedelta(hours=config.TEMPORAL_CUTOFF_HOURS)

        # Filter to 24-hour window
        window = lab_measurements_df[
            lab_measurements_df['charttime'] < cutoff
        ].copy()

        features: dict = {}

        for lab in self._selected_labs:
            itemid = lab['itemid']
            label_clean = lab['label'].lower().replace(' ', '_').replace(',', '').replace('/', '_')
            prefix = f"lab_{itemid}_{label_clean}"

            lab_vals = window[window['itemid'] == itemid].sort_values('charttime')

            if len(lab_vals) == 0:
                features[f"{prefix}_mean"] = np.nan
                features[f"{prefix}_min"] = np.nan
                features[f"{prefix}_max"] = np.nan
                features[f"{prefix}_std"] = np.nan
                features[f"{prefix}_first"] = np.nan
                features[f"{prefix}_last"] = np.nan
                features[f"{prefix}_missing"] = 1
            else:
                vals = lab_vals['valuenum'].values
                features[f"{prefix}_mean"] = float(np.mean(vals))
                features[f"{prefix}_min"] = float(np.min(vals))
                features[f"{prefix}_max"] = float(np.max(vals))
                features[f"{prefix}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
                features[f"{prefix}_first"] = float(vals[0])
                features[f"{prefix}_last"] = float(vals[-1])
                features[f"{prefix}_missing"] = 0

        return features

    def aggregate_all_stays(self, stays_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute lab aggregations for all ICU stays using vectorized operations.
        
        Args:
            stays_df: DataFrame with stay_id and intime columns
            
        Returns:
            DataFrame with stay_id + all lab feature columns
        """
        if self._selected_labs is None:
            raise RuntimeError("Call select_labs() before aggregate_all_stays()")
            
        self.logger.info(f"Aggregating labs for {len(stays_df):,} stays using Vectorized GroupBy ...")
        
        labevents = self._load_labevents()
        selected_itemids = [lab['itemid'] for lab in self._selected_labs]
        
        stay_ids = list(stays_df['stay_id'].unique())
        
        relevant_labs = labevents[labevents['itemid'].isin(selected_itemids)].copy()
        
        self.logger.info(f"  Relevant lab events (selected itemids, cohort stays): {len(relevant_labs):,}")
        
        if len(relevant_labs) == 0:
            self.logger.warning("No relevant lab events found!")
            return pd.DataFrame({'stay_id': stay_ids})
            
        # Ensure sorting by charttime for 'first' and 'last'
        relevant_labs = relevant_labs.sort_values(['stay_id', 'itemid', 'charttime'])
        
        # Compute aggregations vectorized
        agg_funcs = ['mean', 'min', 'max', 'std', 'first', 'last']
        grouped = relevant_labs.groupby(['stay_id', 'itemid'])['valuenum'].agg(agg_funcs)
        
        # Unstack to flat columns
        flat_df = grouped.unstack(level='itemid')
        
        # Reformat column names mapping
        lab_mapping = {lab['itemid']: lab['label'].lower().replace(' ', '_').replace(',', '').replace('/', '_') 
                       for lab in self._selected_labs}
        
        # Create mapping from (agg, itemid) to `lab_{itemid}_{label}_{agg}`
        new_cols = []
        for agg_func, itemid in flat_df.columns:
            prefix = f"lab_{itemid}_{lab_mapping.get(itemid, str(itemid))}"
            new_cols.append(f"{prefix}_{agg_func}")
            
        flat_df.columns = new_cols
        flat_df = flat_df.reset_index()
        
        # Merge back with all stay_ids to ensure everyone is present (even if no labs)
        result_df = pd.DataFrame({'stay_id': stay_ids})
        result_df = result_df.merge(flat_df, on='stay_id', how='left')
        
        # Fill missing indicators and fillna for std (0.0 if only 1 measurement)
        for lab in self._selected_labs:
            itemid = lab['itemid']
            prefix = f"lab_{itemid}_{lab_mapping[itemid]}"
            
            mean_col = f"{prefix}_mean"
            missing_col = f"{prefix}_missing"
            std_col = f"{prefix}_std"
            
            # If the aggregate columns were created
            if mean_col in result_df.columns:
                is_missing = result_df[mean_col].isna().astype(int)
                result_df[missing_col] = is_missing
                
                # Fill NaN std with 0.0 if not missing (means exactly 1 measurement)
                mask = (is_missing == 0) & (result_df[std_col].isna())
                result_df.loc[mask, std_col] = 0.0
                
                # We don't fill NaNs for mean/min/max/etc. here; Data splitter does that.
            else:
                # Lab was never measured for ANY stay
                for f in agg_funcs:
                    result_df[f"{prefix}_{f}"] = np.nan
                result_df[missing_col] = 1

        self.logger.info(f"  Lab aggregation complete. Shape: {result_df.shape}")
        return result_df
