"""
EDA Engine for AKI Prediction Pipeline
Performs exploratory data analysis, schema validation, and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
import os
from datetime import datetime
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class EDA_Engine:
    """
    Exploratory Data Analysis Engine
    Validates schemas, generates visualizations, and analyzes data quality
    """
    
    def __init__(self, raw_data_dir=None, output_dirs=None):
        """
        Initialize EDA Engine with data paths
        
        Args:
            raw_data_dir: Path to raw data directory
            output_dirs: Dictionary with 'figures', 'logs', 'results' paths
        """
        self.raw_data_dir = raw_data_dir or config.RAW_DATA_DIR
        self.output_dirs = output_dirs or {
            'figures': config.FIGURES_DIR,
            'logs': config.LOGS_DIR,
            'results': config.RESULTS_DIR
        }
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Data containers
        self.data = {}
        self.schema_info = {}
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logger = logging.getLogger('EDA_Engine')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        os.makedirs(self.output_dirs['logs'], exist_ok=True)
        
        # File handler
        log_file = os.path.join(self.output_dirs['logs'], 'eda_engine.log')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(config.LOG_FORMAT, config.LOG_DATE_FORMAT)
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def validate_schema(self):
        """
        Validate and document data schemas for all CSV files
        
        Returns:
            dict: Schema documentation for all files
        """
        self.logger.info("Starting schema validation...")
        
        csv_files = {
            'icustays': 'icustays.csv',
            'labevents': 'labevents.csv',
            'patients': 'patients.csv',
            'd_labitems': 'd_labitems.csv',
            'discharge': 'discharge.csv'
        }
        
        schema_doc = {}
        
        for name, filename in csv_files.items():
            filepath = os.path.join(self.raw_data_dir, filename)
            
            if not os.path.exists(filepath):
                self.logger.warning(f"File not found: {filepath}")
                continue
            
            self.logger.info(f"Validating schema for {filename}...")
            
            # Read file (sample for large files)
            try:
                df = pd.read_csv(filepath, nrows=10000)
                self.data[name] = df
            except Exception as e:
                self.logger.error(f"Error reading {filename}: {e}")
                continue
            
            # Extract schema information
            schema_info = {
                'filename': filename,
                'row_count': len(df),
                'column_count': len(df.columns),
                'columns': {},
                'primary_keys': [],
                'timestamp_columns': [],
                'foreign_keys': []
            }
            
            # Analyze each column
            for col in df.columns:
                col_info = {
                    'dtype': str(df[col].dtype),
                    'null_count': int(df[col].isnull().sum()),
                    'null_percentage': float(df[col].isnull().sum() / len(df) * 100),
                    'unique_count': int(df[col].nunique()),
                    'sample_values': df[col].dropna().head(3).tolist() if len(df[col].dropna()) > 0 else []
                }
                schema_info['columns'][col] = col_info
            
            # Identify primary keys (columns with unique values and no nulls)
            for col in df.columns:
                if df[col].nunique() == len(df) and df[col].isnull().sum() == 0:
                    schema_info['primary_keys'].append(col)
            
            # Identify timestamp columns
            timestamp_keywords = ['time', 'date', 'dob', 'dod']
            for col in df.columns:
                if any(keyword in col.lower() for keyword in timestamp_keywords):
                    schema_info['timestamp_columns'].append(col)
            
            # Identify foreign keys (common ID columns)
            fk_keywords = ['subject_id', 'hadm_id', 'stay_id', 'itemid']
            for col in df.columns:
                if col in fk_keywords and col not in schema_info['primary_keys']:
                    schema_info['foreign_keys'].append(col)
            
            schema_doc[name] = schema_info
            
            # Log summary
            self.logger.info(f"  Columns: {schema_info['column_count']}")
            self.logger.info(f"  Rows (sample): {schema_info['row_count']}")
            self.logger.info(f"  Primary keys: {schema_info['primary_keys']}")
            self.logger.info(f"  Timestamp columns: {schema_info['timestamp_columns']}")
            self.logger.info(f"  Foreign keys: {schema_info['foreign_keys']}")
        
        # Save schema documentation
        output_file = os.path.join(self.output_dirs['logs'], 'schema_documentation.json')
        with open(output_file, 'w') as f:
            json.dump(schema_doc, f, indent=2)
        
        self.logger.info(f"Schema documentation saved to {output_file}")
        self.schema_info = schema_doc
        
        return schema_doc
    
    def _load_full_data(self):
        """Load full datasets for analysis"""
        self.logger.info("Loading full datasets...")
        
        # Load ICU stays
        icustays_path = os.path.join(self.raw_data_dir, 'icustays.csv')
        self.data['icustays_full'] = pd.read_csv(icustays_path)
        self.logger.info(f"Loaded {len(self.data['icustays_full'])} ICU stays")
        
        # Load patients
        patients_path = os.path.join(self.raw_data_dir, 'patients.csv')
        self.data['patients_full'] = pd.read_csv(patients_path)
        self.logger.info(f"Loaded {len(self.data['patients_full'])} patients")
        
        # Parse timestamps
        self.data['icustays_full']['intime'] = pd.to_datetime(self.data['icustays_full']['intime'])
        self.data['icustays_full']['outtime'] = pd.to_datetime(self.data['icustays_full']['outtime'])
        
        # Compute duration in hours
        self.data['icustays_full']['duration_hours'] = (
            self.data['icustays_full']['outtime'] - self.data['icustays_full']['intime']
        ).dt.total_seconds() / 3600

    
    def analyze_cohort(self):
        """Generate cohort visualizations"""
        self.logger.info("Starting cohort analysis...")
        
        # Ensure data is loaded
        if 'icustays_full' not in self.data:
            self._load_full_data()
        
        # Create figures directory
        os.makedirs(self.output_dirs['figures'], exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = config.FIGURE_DPI
        
        # 1. ICU stay length distribution
        self.logger.info("Generating ICU stay length distribution...")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(self.data['icustays_full']['duration_hours'], bins=50, edgecolor='black')
        ax.set_xlabel('ICU Stay Duration (hours)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of ICU Stay Lengths', fontsize=14, fontweight='bold')
        ax.axvline(24, color='red', linestyle='--', label='24-hour threshold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dirs['figures'], 'icu_stay_length_distribution.png'), 
                    dpi=config.FIGURE_DPI)
        plt.close()
        
        # 2. ICU stays per patient
        self.logger.info("Generating ICU stays per patient distribution...")
        stays_per_patient = self.data['icustays_full'].groupby('subject_id').size()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(stays_per_patient, bins=range(1, min(20, stays_per_patient.max()+2)), 
                edgecolor='black', align='left')
        ax.set_xlabel('Number of ICU Stays per Patient', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of ICU Stays per Patient', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dirs['figures'], 'icu_stays_per_patient.png'), 
                    dpi=config.FIGURE_DPI)
        plt.close()
        
        # 3. Age distribution
        self.logger.info("Generating age distribution...")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(self.data['patients_full']['anchor_age'], bins=30, edgecolor='black')
        ax.set_xlabel('Age (years)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Patient Age Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dirs['figures'], 'age_distribution.png'), 
                    dpi=config.FIGURE_DPI)
        plt.close()
        
        # 4. Gender distribution
        self.logger.info("Generating gender distribution...")
        gender_counts = self.data['patients_full']['gender'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(gender_counts.index, gender_counts.values, edgecolor='black')
        ax.set_xlabel('Gender', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Patient Gender Distribution', fontsize=14, fontweight='bold')
        for i, v in enumerate(gender_counts.values):
            ax.text(i, v + 100, str(v), ha='center', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dirs['figures'], 'gender_distribution.png'), 
                    dpi=config.FIGURE_DPI)
        plt.close()
        
        self.logger.info("Cohort analysis complete")
        
        # Log statistics
        self.logger.info(f"Total ICU stays: {len(self.data['icustays_full'])}")
        self.logger.info(f"Unique patients: {self.data['icustays_full']['subject_id'].nunique()}")
        self.logger.info(f"Mean ICU stay duration: {self.data['icustays_full']['duration_hours'].mean():.2f} hours")
        self.logger.info(f"Median ICU stay duration: {self.data['icustays_full']['duration_hours'].median():.2f} hours")
    
    def analyze_labs(self):
        """Analyze laboratory data patterns"""
        self.logger.info("Starting laboratory data analysis...")
        
        # Load lab events (sample for memory efficiency)
        labevents_path = os.path.join(self.raw_data_dir, 'labevents.csv')
        self.logger.info("Loading lab events (this may take a moment)...")
        
        # Load in chunks to handle large file
        chunk_size = 100000
        lab_chunks = []
        for chunk in pd.read_csv(labevents_path, chunksize=chunk_size, nrows=500000):
            lab_chunks.append(chunk)
        
        labevents = pd.concat(lab_chunks, ignore_index=True)
        self.logger.info(f"Loaded {len(labevents)} lab events")
        
        # Load d_labitems for lab names
        d_labitems_path = os.path.join(self.raw_data_dir, 'd_labitems.csv')
        d_labitems = pd.read_csv(d_labitems_path)
        
        # Merge to get lab names
        labevents = labevents.merge(d_labitems[['itemid', 'label']], on='itemid', how='left')
        
        # Create figures directory
        os.makedirs(self.output_dirs['figures'], exist_ok=True)
        
        # 1. Creatinine distribution
        self.logger.info("Generating creatinine distribution...")
        creatinine_data = labevents[labevents['itemid'] == config.CREATININE_ITEMID]['valuenum'].dropna()
        
        if len(creatinine_data) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(creatinine_data, bins=50, edgecolor='black', range=(0, 10))
            ax.set_xlabel('Creatinine (mg/dL)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribution of Raw Creatinine Values', fontsize=14, fontweight='bold')
            ax.axvline(creatinine_data.median(), color='red', linestyle='--', 
                      label=f'Median: {creatinine_data.median():.2f}')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dirs['figures'], 'creatinine_distribution.png'), 
                       dpi=config.FIGURE_DPI)
            plt.close()
            
            self.logger.info(f"Creatinine - Mean: {creatinine_data.mean():.2f}, Median: {creatinine_data.median():.2f}")
        else:
            self.logger.warning("No creatinine data found")
        
        # 2. Lab measurement counts per admission
        self.logger.info("Generating lab measurement counts...")
        lab_counts = labevents.groupby('hadm_id').size()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(lab_counts, bins=50, edgecolor='black')
        ax.set_xlabel('Number of Lab Measurements', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Lab Measurement Counts per Admission', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dirs['figures'], 'lab_measurement_counts.png'), 
                   dpi=config.FIGURE_DPI)
        plt.close()
        
        # 3. Lab test coverage
        self.logger.info("Computing lab test coverage...")
        
        # Get unique admissions with each lab
        lab_coverage = labevents.groupby('itemid')['hadm_id'].nunique().sort_values(ascending=False)
        total_admissions = labevents['hadm_id'].nunique()
        lab_coverage_pct = (lab_coverage / total_admissions * 100).head(20)
        
        # Get lab names
        lab_names = []
        for itemid in lab_coverage_pct.index:
            lab_name = d_labitems[d_labitems['itemid'] == itemid]['label'].values
            if len(lab_name) > 0:
                lab_names.append(lab_name[0][:30])  # Truncate long names
            else:
                lab_names.append(f"ItemID {itemid}")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(lab_coverage_pct))
        ax.barh(y_pos, lab_coverage_pct.values, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(lab_names, fontsize=10)
        ax.set_xlabel('Coverage (%)', fontsize=12)
        ax.set_title('Top 20 Laboratory Tests by Coverage', fontsize=14, fontweight='bold')
        ax.axvline(30, color='red', linestyle='--', label='30% threshold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dirs['figures'], 'lab_coverage.png'), 
                   dpi=config.FIGURE_DPI)
        plt.close()
        
        self.logger.info("Laboratory data analysis complete")
    
    def analyze_missingness(self):
        """Compute and visualize missingness patterns"""
        self.logger.info("Starting missingness pattern analysis...")
        
        # Load lab events
        labevents_path = os.path.join(self.raw_data_dir, 'labevents.csv')
        self.logger.info("Loading lab events for missingness analysis...")
        
        # Load sample
        chunk_size = 100000
        lab_chunks = []
        for chunk in pd.read_csv(labevents_path, chunksize=chunk_size, nrows=500000):
            lab_chunks.append(chunk)
        
        labevents = pd.concat(lab_chunks, ignore_index=True)
        
        # Load d_labitems
        d_labitems_path = os.path.join(self.raw_data_dir, 'd_labitems.csv')
        d_labitems = pd.read_csv(d_labitems_path)
        
        # Compute missing rates for each lab test
        self.logger.info("Computing missing rates...")
        
        # Group by itemid and compute missing rate for valuenum
        lab_stats = labevents.groupby('itemid').agg({
            'valuenum': lambda x: (x.isnull().sum() / len(x) * 100),
            'hadm_id': 'count'
        }).reset_index()
        lab_stats.columns = ['itemid', 'missing_rate', 'total_measurements']
        
        # Merge with lab names
        lab_stats = lab_stats.merge(d_labitems[['itemid', 'label']], on='itemid', how='left')
        
        # Sort by missing rate
        lab_stats = lab_stats.sort_values('missing_rate', ascending=False)
        
        # Save to CSV
        os.makedirs(self.output_dirs['results'], exist_ok=True)
        output_file = os.path.join(self.output_dirs['results'], 'missingness_statistics.csv')
        lab_stats.to_csv(output_file, index=False)
        self.logger.info(f"Missingness statistics saved to {output_file}")
        
        # Log warnings for high missing rates
        high_missing = lab_stats[lab_stats['missing_rate'] > 70]
        if len(high_missing) > 0:
            self.logger.warning(f"Found {len(high_missing)} lab tests with >70% missing rate")
            for _, row in high_missing.head(5).iterrows():
                self.logger.warning(f"  {row['label']}: {row['missing_rate']:.1f}% missing")
        
        # Visualize top 20 by missing rate
        self.logger.info("Generating missingness visualization...")
        top_missing = lab_stats.head(20)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(top_missing))
        ax.barh(y_pos, top_missing['missing_rate'].values, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([str(label)[:30] for label in top_missing['label'].values], fontsize=10)
        ax.set_xlabel('Missing Rate (%)', fontsize=12)
        ax.set_title('Top 20 Laboratory Tests by Missing Rate', fontsize=14, fontweight='bold')
        ax.axvline(70, color='red', linestyle='--', label='70% threshold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dirs['figures'], 'missingness_rates.png'), 
                   dpi=config.FIGURE_DPI)
        plt.close()
        
        self.logger.info("Missingness pattern analysis complete")
        self.logger.info(f"Mean missing rate across all labs: {lab_stats['missing_rate'].mean():.2f}%")
    
    def run_full_eda(self):
        """Execute complete EDA pipeline"""
        self.logger.info("="*80)
        self.logger.info("Starting Full EDA Pipeline")
        self.logger.info("="*80)
        
        start_time = datetime.now()
        
        # Step 1: Validate schema
        self.validate_schema()
        
        # Step 2: Analyze cohort
        self.analyze_cohort()
        
        # Step 3: Analyze labs
        self.analyze_labs()
        
        # Step 4: Analyze missingness
        self.analyze_missingness()
        
        self.logger.info("="*80)
        self.logger.info("EDA Pipeline Complete")
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        self.logger.info(f"Total execution time: {duration:.2f} seconds")
        self.logger.info("="*80)


if __name__ == "__main__":
    # Run EDA Engine
    eda = EDA_Engine()
    eda.run_full_eda()
