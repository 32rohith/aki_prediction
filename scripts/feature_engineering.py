#!/usr/bin/env python3
"""
Feature Engineering Script
Extracts structured features from the first 24 hours of ICU admission.

Usage:
    python feature_engineering.py [--raw-data-dir RAW_DATA_DIR]
                                   [--processed-data-dir PROCESSED_DATA_DIR]
                                   [--logs-dir LOGS_DIR]

Example:
    python feature_engineering.py --raw-data-dir raw_data \\
                                   --processed-data-dir processed_data \\
                                   --logs-dir logs

Output:
    processed_data/structured_dataset.csv  — feature matrix with identifiers,
    demographics, baseline creatinine, and lab aggregations for all ICU stays.
"""

import argparse
import logging
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from scripts.feature_extractor import Feature_Extractor


def setup_logging(log_dir: str) -> logging.Logger:
    """Configure logging to both file and console."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"feature_engineering_{timestamp}.log")

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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract structured features for AKI prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Features extracted:
  Demographics : age (anchor_age), gender (binary), ICU type (one-hot)
  Baseline     : baseline_creatinine from labeled_stays.csv
  Lab features : mean, min, max, std, first, last + missingness indicator
                 for each selected lab (coverage >= 30%, creatinine excluded)

Temporal constraint:
  Lab measurements are filtered to charttime < intime + 24h (no leakage).
        """,
    )

    parser.add_argument(
        '--raw-data-dir',
        type=str,
        default=config.RAW_DATA_DIR,
        help=f'Directory containing raw MIMIC-IV CSV files (default: {config.RAW_DATA_DIR})',
    )
    parser.add_argument(
        '--processed-data-dir',
        type=str,
        default=config.PROCESSED_DATA_DIR,
        help=f'Directory for processed data outputs (default: {config.PROCESSED_DATA_DIR})',
    )
    parser.add_argument(
        '--logs-dir',
        type=str,
        default=config.LOGS_DIR,
        help=f'Directory for log files (default: {config.LOGS_DIR})',
    )

    return parser.parse_args()


def main() -> int:
    """Main execution function."""
    args = parse_arguments()
    logger = setup_logging(args.logs_dir)

    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("FEATURE ENGINEERING SCRIPT")
    logger.info("=" * 80)
    logger.info(f"Execution started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Raw data directory:       {args.raw_data_dir}")
    logger.info(f"Processed data directory: {args.processed_data_dir}")
    logger.info(f"Logs directory:           {args.logs_dir}")
    logger.info("")

    os.makedirs(args.processed_data_dir, exist_ok=True)

    try:
        extractor = Feature_Extractor(
            raw_data_dir=args.raw_data_dir,
            processed_data_dir=args.processed_data_dir,
            logs_dir=args.logs_dir,
        )
        extractor.run()

        end_time = datetime.now()
        duration = end_time - start_time
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Execution completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total duration: {duration}")
        logger.info("=" * 80)
        return 0

    except Exception as e:
        logger.error(f"Error during feature engineering: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
