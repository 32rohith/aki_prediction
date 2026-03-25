#!/usr/bin/env python3
"""
AKI Labeling Script
Assigns binary AKI labels to ICU stays using KDIGO criteria with strict temporal constraints.

Usage:
    python labeling.py [--raw-data-dir RAW_DATA_DIR] [--output-dir OUTPUT_DIR]

Example:
    python labeling.py --raw-data-dir raw_data --output-dir processed_data
"""

import argparse
import logging
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from scripts.aki_labeler import AKI_Labeler


def setup_logging(log_dir: str):
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"labeling_{timestamp}.log")
    
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
        description="Label ICU stays with AKI using KDIGO criteria",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
KDIGO Criteria:
  - 48-hour criterion: Creatinine increase ≥0.3 mg/dL within any 48-hour window after hour 24
  - 7-day criterion: Creatinine ≥1.5x baseline within 7 days after ICU admission OR until discharge

Temporal Constraints:
  - Only ICU stays ≥24 hours are included
  - AKI detection uses only creatinine measurements after hour 24
  - Baseline creatinine is computed from same-admission measurements before ICU intime

Output:
  - labeled_stays.csv with columns: subject_id, hadm_id, stay_id, intime, outtime,
    duration_hours, baseline_creatinine, aki_label
        """
    )
    
    parser.add_argument(
        '--raw-data-dir',
        type=str,
        default=config.RAW_DATA_DIR,
        help=f'Directory containing raw MIMIC-IV CSV files (default: {config.RAW_DATA_DIR})'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=config.PROCESSED_DATA_DIR,
        help=f'Directory for output labeled_stays.csv (default: {config.PROCESSED_DATA_DIR})'
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
    logger.info("AKI LABELING SCRIPT")
    logger.info("="*80)
    logger.info(f"Execution started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Raw data directory: {args.raw_data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Log directory: {args.log_dir}")
    logger.info("")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Initialize and run AKI labeler
        labeler = AKI_Labeler(
            raw_data_dir=args.raw_data_dir,
            output_dir=args.output_dir
        )
        
        labeler.run()
        
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
        logger.error(f"Error during AKI labeling: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
