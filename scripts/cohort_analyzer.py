#!/usr/bin/env python3
"""
Cohort Analyzer for AKI Prediction Pipeline
Generates statistical summaries and visualizations of patient cohorts with AKI labels.

This module implements comprehensive cohort analysis including:
- AKI prevalence statistics
- Demographic comparisons (AKI-positive vs AKI-negative)
- Creatinine distribution analysis
- Statistical summaries for continuous and categorical features
- Correlation analysis
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Cohort_Analyzer:
    """
    Analyzes patient cohorts with AKI labels and generates publication-quality visualizations.
    
    This class provides methods for:
    - Computing AKI prevalence statistics
    - Generating comparison plots for AKI-positive vs AKI-negative patients
    - Creating statistical summaries and correlation matrices
    - Saving all outputs to appropriate directories
    """
    
    def __init__(self, processed_data_dir: str, raw_data_dir: str, 
                 figures_dir: str, results_dir: str):
        """
        Initialize Cohort Analyzer with data and output directories.
        
        Args:
            processed_data_dir: Directory containing labeled_stays.csv
            raw_data_dir: Directory containing raw MIMIC-IV data
            figures_dir: Directory for saving visualizations
            results_dir: Directory for saving statistical summaries
        """
        self.processed_data_dir = processed_data_dir
        self.raw_data_dir = raw_data_dir
        self.figures_dir = figures_dir
        self.results_dir = results_dir
        
        # Create output directories if they don't exist
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Data containers
        self.labeled_stays = None
        self.patients = None
        self.creatinine_data = None
        
    def load_data(self) -> None:
        """Load labeled stays, patient demographics, and creatinine measurements."""
        self.logger.info("Loading data files...")
        
        # Load labeled stays
        labeled_path = os.path.join(self.processed_data_dir, "labeled_stays.csv")
        self.labeled_stays = pd.read_csv(labeled_path, parse_dates=['intime', 'outtime'])
        self.logger.info(f"  Loaded {len(self.labeled_stays)} labeled ICU stays")
        
        # Load patients for demographics
        patients_path = os.path.join(self.raw_data_dir, "patients.csv")
        self.patients = pd.read_csv(patients_path)
        self.logger.info(f"  Loaded {len(self.patients)} patients")
        
        # Merge to get demographics
        self.labeled_stays = self.labeled_stays.merge(
            self.patients[['subject_id', 'gender', 'anchor_age']],
            on='subject_id',
            how='left'
        )
        
        self.logger.info("Data loading complete")
        
    def compute_aki_prevalence(self) -> Dict[str, float]:
        """
        Compute AKI prevalence statistics.
        
        Returns:
            Dictionary with prevalence metrics
        """
        self.logger.info("Computing AKI prevalence...")
        
        total_stays = len(self.labeled_stays)
        aki_positive = (self.labeled_stays['aki_label'] == 1).sum()
        aki_negative = (self.labeled_stays['aki_label'] == 0).sum()
        
        prevalence_pct = 100 * aki_positive / total_stays
        
        stats = {
            'total_stays': total_stays,
            'aki_positive': aki_positive,
            'aki_negative': aki_negative,
            'prevalence_percent': prevalence_pct
        }
        
        self.logger.info(f"  Total ICU stays: {total_stays}")
        self.logger.info(f"  AKI-positive: {aki_positive} ({prevalence_pct:.2f}%)")
        self.logger.info(f"  AKI-negative: {aki_negative} ({100-prevalence_pct:.2f}%)")
        
        return stats
        
    def generate_aki_prevalence_plot(self) -> None:
        """Generate bar chart showing AKI prevalence."""
        self.logger.info("Generating AKI prevalence bar chart...")
        
        aki_counts = self.labeled_stays['aki_label'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(['No AKI', 'AKI'], aki_counts.values, 
                      color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}\n({100*height/len(self.labeled_stays):.1f}%)',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Number of ICU Stays', fontsize=12, fontweight='bold')
        ax.set_title('AKI Prevalence in ICU Cohort', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        output_path = os.path.join(self.figures_dir, 'aki_prevalence.png')
        plt.savefig(output_path, dpi=config.FIGURE_DPI, format=config.FIGURE_FORMAT, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"  Saved to: {output_path}")

    def generate_creatinine_increase_histogram(self) -> None:
        """Generate histogram of maximum creatinine increase for AKI-positive cases."""
        self.logger.info("Generating maximum creatinine increase histogram...")
        
        # Load creatinine measurements
        labevents_path = os.path.join(self.raw_data_dir, "labevents.csv")
        
        # Filter for AKI-positive stays
        aki_positive_stays = self.labeled_stays[self.labeled_stays['aki_label'] == 1]
        
        self.logger.info(f"  Loading creatinine data for {len(aki_positive_stays)} AKI-positive stays...")
        
        # Load creatinine measurements in chunks
        max_increases = []
        
        for chunk in pd.read_csv(labevents_path, chunksize=1000000, 
                                usecols=['subject_id', 'hadm_id', 'itemid', 'valuenum', 'charttime']):
            # Filter for creatinine itemids
            creat_chunk = chunk[chunk['itemid'].isin(config.CREATININE_ITEMIDS)].copy()
            
            if len(creat_chunk) == 0:
                continue
                
            # Merge with AKI-positive stays
            creat_chunk = creat_chunk.merge(
                aki_positive_stays[['stay_id', 'hadm_id', 'baseline_creatinine']],
                on='hadm_id',
                how='inner'
            )
            
            if len(creat_chunk) == 0:
                continue
            
            # Compute increase from baseline
            creat_chunk['increase'] = creat_chunk['valuenum'] - creat_chunk['baseline_creatinine']
            
            # Get max increase per stay
            stay_max = creat_chunk.groupby('stay_id')['increase'].max()
            max_increases.extend(stay_max.values)
        
        if len(max_increases) == 0:
            self.logger.warning("  No creatinine increase data found for AKI-positive cases")
            return
        
        max_increases = np.array(max_increases)
        
        # Generate histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(max_increases, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Maximum Creatinine Increase (mg/dL)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of AKI-Positive ICU Stays', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Maximum Creatinine Increase\n(AKI-Positive Cases)', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add statistics text
        stats_text = f'Mean: {np.mean(max_increases):.2f} mg/dL\nMedian: {np.median(max_increases):.2f} mg/dL'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_path = os.path.join(self.figures_dir, 'max_creatinine_increase.png')
        plt.savefig(output_path, dpi=config.FIGURE_DPI, format=config.FIGURE_FORMAT, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"  Saved to: {output_path}")
        self.logger.info(f"  Mean increase: {np.mean(max_increases):.2f} mg/dL")
        self.logger.info(f"  Median increase: {np.median(max_increases):.2f} mg/dL")
        
    def generate_baseline_creatinine_comparison(self) -> None:
        """Generate comparison plots for baseline creatinine: AKI-positive vs AKI-negative."""
        self.logger.info("Generating baseline creatinine comparison plots...")
        
        aki_positive = self.labeled_stays[self.labeled_stays['aki_label'] == 1]['baseline_creatinine']
        aki_negative = self.labeled_stays[self.labeled_stays['aki_label'] == 0]['baseline_creatinine']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram comparison
        axes[0].hist(aki_negative, bins=50, alpha=0.6, label='No AKI', color='#2ecc71', edgecolor='black')
        axes[0].hist(aki_positive, bins=50, alpha=0.6, label='AKI', color='#e74c3c', edgecolor='black')
        axes[0].set_xlabel('Baseline Creatinine (mg/dL)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Baseline Creatinine Distribution', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(axis='y', alpha=0.3, linestyle='--')
        
        # Box plot comparison
        data_to_plot = [aki_negative.dropna(), aki_positive.dropna()]
        bp = axes[1].boxplot(data_to_plot, labels=['No AKI', 'AKI'],
                            patch_artist=True, widths=0.6)
        
        # Color the boxes
        colors = ['#2ecc71', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        axes[1].set_ylabel('Baseline Creatinine (mg/dL)', fontsize=12, fontweight='bold')
        axes[1].set_title('Baseline Creatinine Comparison', fontsize=13, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add statistics
        stats_text = f'No AKI: {aki_negative.mean():.2f} ± {aki_negative.std():.2f}\n'
        stats_text += f'AKI: {aki_positive.mean():.2f} ± {aki_positive.std():.2f}'
        axes[1].text(0.5, 0.95, stats_text, transform=axes[1].transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_path = os.path.join(self.figures_dir, 'baseline_creatinine_comparison.png')
        plt.savefig(output_path, dpi=config.FIGURE_DPI, format=config.FIGURE_FORMAT, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"  Saved to: {output_path}")
        self.logger.info(f"  No AKI - Mean: {aki_negative.mean():.2f}, Std: {aki_negative.std():.2f}")
        self.logger.info(f"  AKI - Mean: {aki_positive.mean():.2f}, Std: {aki_positive.std():.2f}")
        
    def generate_age_comparison(self) -> None:
        """Generate comparison plots for age distribution: AKI-positive vs AKI-negative."""
        self.logger.info("Generating age distribution comparison plots...")
        
        aki_positive = self.labeled_stays[self.labeled_stays['aki_label'] == 1]['anchor_age']
        aki_negative = self.labeled_stays[self.labeled_stays['aki_label'] == 0]['anchor_age']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram comparison
        axes[0].hist(aki_negative, bins=30, alpha=0.6, label='No AKI', color='#2ecc71', edgecolor='black')
        axes[0].hist(aki_positive, bins=30, alpha=0.6, label='AKI', color='#e74c3c', edgecolor='black')
        axes[0].set_xlabel('Age (years)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Age Distribution', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(axis='y', alpha=0.3, linestyle='--')
        
        # Box plot comparison
        data_to_plot = [aki_negative.dropna(), aki_positive.dropna()]
        bp = axes[1].boxplot(data_to_plot, labels=['No AKI', 'AKI'],
                            patch_artist=True, widths=0.6)
        
        # Color the boxes
        colors = ['#2ecc71', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        axes[1].set_ylabel('Age (years)', fontsize=12, fontweight='bold')
        axes[1].set_title('Age Comparison', fontsize=13, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add statistics
        stats_text = f'No AKI: {aki_negative.mean():.1f} ± {aki_negative.std():.1f}\n'
        stats_text += f'AKI: {aki_positive.mean():.1f} ± {aki_positive.std():.1f}'
        axes[1].text(0.5, 0.95, stats_text, transform=axes[1].transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_path = os.path.join(self.figures_dir, 'age_comparison.png')
        plt.savefig(output_path, dpi=config.FIGURE_DPI, format=config.FIGURE_FORMAT, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"  Saved to: {output_path}")
        self.logger.info(f"  No AKI - Mean: {aki_negative.mean():.1f}, Std: {aki_negative.std():.1f}")
        self.logger.info(f"  AKI - Mean: {aki_positive.mean():.1f}, Std: {aki_positive.std():.1f}")

    def generate_statistical_summaries(self) -> None:
        """Generate summary tables for continuous and categorical features."""
        self.logger.info("Generating statistical summaries...")
        
        # Continuous features summary
        continuous_features = ['duration_hours', 'baseline_creatinine', 'anchor_age']
        
        summary_stats = []
        for feature in continuous_features:
            if feature in self.labeled_stays.columns:
                data = self.labeled_stays[feature].dropna()
                stats = {
                    'Feature': feature,
                    'Count': len(data),
                    'Mean': data.mean(),
                    'Median': data.median(),
                    'Std': data.std(),
                    'Min': data.min(),
                    'Max': data.max(),
                    'Q25': data.quantile(0.25),
                    'Q75': data.quantile(0.75)
                }
                summary_stats.append(stats)
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Save continuous features summary
        output_path = os.path.join(self.results_dir, 'continuous_features_summary.csv')
        summary_df.to_csv(output_path, index=False, float_format='%.3f')
        self.logger.info(f"  Saved continuous features summary to: {output_path}")
        
        # Categorical features summary
        categorical_features = ['gender', 'aki_label']
        
        freq_tables = {}
        for feature in categorical_features:
            if feature in self.labeled_stays.columns:
                freq = self.labeled_stays[feature].value_counts()
                freq_pct = 100 * self.labeled_stays[feature].value_counts(normalize=True)
                
                freq_df = pd.DataFrame({
                    'Value': freq.index,
                    'Count': freq.values,
                    'Percentage': freq_pct.values
                })
                freq_tables[feature] = freq_df
        
        # Save categorical features summary
        output_path = os.path.join(self.results_dir, 'categorical_features_summary.csv')
        with open(output_path, 'w') as f:
            for feature, freq_df in freq_tables.items():
                f.write(f"\n{feature}\n")
                freq_df.to_csv(f, index=False, float_format='%.2f')
                f.write("\n")
        
        self.logger.info(f"  Saved categorical features summary to: {output_path}")
        
    def generate_correlation_analysis(self) -> None:
        """Compute correlation matrix and generate heatmap for numeric features."""
        self.logger.info("Generating correlation analysis...")
        
        # Select numeric features
        numeric_features = ['duration_hours', 'baseline_creatinine', 'anchor_age', 'aki_label']
        
        # Filter to only existing columns
        available_features = [f for f in numeric_features if f in self.labeled_stays.columns]
        
        if len(available_features) < 2:
            self.logger.warning("  Insufficient numeric features for correlation analysis")
            return
        
        # Compute correlation matrix
        corr_data = self.labeled_stays[available_features].dropna()
        corr_matrix = corr_data.corr()
        
        # Save correlation matrix
        output_path = os.path.join(self.results_dir, 'correlation_matrix.csv')
        corr_matrix.to_csv(output_path, float_format='%.3f')
        self.logger.info(f"  Saved correlation matrix to: {output_path}")
        
        # Generate correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1, ax=ax)
        
        ax.set_title('Correlation Matrix of Numeric Features', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        output_path = os.path.join(self.figures_dir, 'correlation_heatmap.png')
        plt.savefig(output_path, dpi=config.FIGURE_DPI, format=config.FIGURE_FORMAT, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"  Saved correlation heatmap to: {output_path}")
        
    def run_all_analyses(self) -> None:
        """Execute all cohort analysis tasks."""
        self.logger.info("="*80)
        self.logger.info("Starting Cohort Analysis")
        self.logger.info("="*80)
        
        # Load data
        self.load_data()
        
        # AKI prevalence analysis
        self.logger.info("\n" + "="*80)
        self.logger.info("AKI Prevalence Analysis")
        self.logger.info("="*80)
        prevalence_stats = self.compute_aki_prevalence()
        self.generate_aki_prevalence_plot()
        
        # Creatinine analysis
        self.logger.info("\n" + "="*80)
        self.logger.info("Creatinine Analysis")
        self.logger.info("="*80)
        self.generate_creatinine_increase_histogram()
        self.generate_baseline_creatinine_comparison()
        
        # Age analysis
        self.logger.info("\n" + "="*80)
        self.logger.info("Age Distribution Analysis")
        self.logger.info("="*80)
        self.generate_age_comparison()
        
        # Statistical summaries
        self.logger.info("\n" + "="*80)
        self.logger.info("Statistical Summaries")
        self.logger.info("="*80)
        self.generate_statistical_summaries()
        
        # Correlation analysis
        self.logger.info("\n" + "="*80)
        self.logger.info("Correlation Analysis")
        self.logger.info("="*80)
        self.generate_correlation_analysis()
        
        self.logger.info("\n" + "="*80)
        self.logger.info("Cohort Analysis Complete")
        self.logger.info("="*80)


if __name__ == "__main__":
    # Setup logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT
    )
    
    # Initialize and run analyzer
    analyzer = Cohort_Analyzer(
        processed_data_dir=config.PROCESSED_DATA_DIR,
        raw_data_dir=config.RAW_DATA_DIR,
        figures_dir=config.FIGURES_DIR,
        results_dir=config.RESULTS_DIR
    )
    
    analyzer.run_all_analyses()
