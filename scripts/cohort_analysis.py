#!/usr/bin/env python3
"""
Cohort Analysis Script
Generates AKI prevalence statistics and comprehensive cohort visualizations.

Usage:
    python cohort_analysis.py [--processed-data-dir DIR] [--raw-data-dir DIR] [--figures-dir DIR] [--results-dir DIR]

Example:
    python cohort_analysis.py --processed-data-dir processed_data --raw-data-dir raw_data
"""

import argparse
import logging
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from scripts.cohort_analyzer import Cohort_Analyzer


def setup_logging(log_dir: str):
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"cohort_analysis_{timestamp}.log")
    
    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    
    return logger


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze patient cohorts with AKI labels and generate visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Outputs:
  - AKI prevalence bar chart
  - Maximum creatinine increase histogram (AKI-positive cases)
  - Baseline creatinine comparison plots (AKI-positive vs AKI-negative)
  - Age distribution comparison plots (AKI-positive vs AKI-negative)
  - Statistical summaries for continuous and categorical features
  - Correlation matrix and heatmap
        """
    )
    
    parser.add_argument(
        '--processed-data-dir',
        type=str,
        default=config.PROCESSED_DATA_DIR,
        help=f'Directory containing labeled_stays.csv (default: {config.PROCESSED_DATA_DIR})'
    )
    
    parser.add_argument(
        '--raw-data-dir',
        type=str,
        default=config.RAW_DATA_DIR,
        help=f'Directory containing raw MIMIC-IV CSV files (default: {config.RAW_DATA_DIR})'
    )
    
    parser.add_argument(
        '--figures-dir',
        type=str,
        default=config.FIGURES_DIR,
        help=f'Directory for output visualizations (default: {config.FIGURES_DIR})'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default=config.RESULTS_DIR,
        help=f'Directory for statistical summaries (default: {config.RESULTS_DIR})'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default=config.LOGS_DIR,
        help=f'Directory for log files (default: {config.LOGS_DIR})'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    
    # Log execution start
    start_time = datetime.now()
    logger.info("="*80)
    logger.info("COHORT ANALYSIS SCRIPT")
    logger.info("="*80)
    logger.info(f"Execution started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Processed data directory: {args.processed_data_dir}")
    logger.info(f"Raw data directory: {args.raw_data_dir}")
    logger.info(f"Figures directory: {args.figures_dir}")
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Log directory: {args.log_dir}")
    logger.info("")
    
    try:
        # Initialize and run cohort analyzer
        analyzer = Cohort_Analyzer(
            processed_data_dir=args.processed_data_dir,
            raw_data_dir=args.raw_data_dir,
            figures_dir=args.figures_dir,
            results_dir=args.results_dir
        )
        
        analyzer.run_all_analyses()
        
        # Log execution end
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info("")
        logger.info("="*80)
        logger.info(f"Execution completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total duration: {duration}")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during cohort analysis: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
