#!/usr/bin/env python3
"""
Master Pipeline Script
Executes the fully automated Multimodal End-to-End AKI Prediction pipeline sequentially.
Checks for phase completion markers before proceeding to dependent phases.
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

# Setup
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOGS_DIR, f'master_execution_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('master_pipeline')


def run_script(script_path: str):
    """Execute a pipeline script and handle errors."""
    logger.info("=" * 80)
    logger.info(f"STARTING: {os.path.basename(script_path)}")
    logger.info("=" * 80)

    start = datetime.now()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True, text=True, capture_output=True
        )
        duration = (datetime.now() - start).total_seconds()
        stdout_lines = result.stdout.strip().split('\n')
        for line in stdout_lines[-15:]:
            logger.info(f"   {line}")
        logger.info(f"SUCCESS: {os.path.basename(script_path)} ({duration:.1f}s)\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"FAILED: {os.path.basename(script_path)}")
        logger.error(f"Exit code: {e.returncode}")
        if e.stderr:
            for line in e.stderr.strip().split('\n')[-20:]:
                logger.error(f"   {line}")
        logger.error("Pipeline aborted due to critical failure.")
        sys.exit(1)


def check_marker(marker_name: str) -> bool:
    path = os.path.join(LOGS_DIR, marker_name)
    return os.path.exists(path)


def main():
    logger.info("=" * 80)
    logger.info("Multimodal Early AKI Prediction Pipeline (MIMIC-IV)")
    logger.info(f"Execution started at {datetime.now().isoformat()}")
    logger.info(f"Master log: {log_file}")
    logger.info("=" * 80 + "\n")

    # Phase 1: Structured Feature Generation
    phase1_scripts = [
        os.path.join(SCRIPTS_DIR, 'cohort_selector.py'),
        os.path.join(SCRIPTS_DIR, 'aki_labeler.py'),
        os.path.join(SCRIPTS_DIR, 'lab_aggregator.py'),
        os.path.join(SCRIPTS_DIR, 'data_splitter.py'),
    ]

    logger.info("--- PHASE 1: Structured Feature Generation ---")
    for script in phase1_scripts:
        if not os.path.exists(script):
            logger.error(f"Script missing: {script}")
            sys.exit(1)
        run_script(script)

    # Phase 1 Model Training
    logger.info("--- PHASE 1: Structured-Only Baseline Models ---")
    run_script(os.path.join(SCRIPTS_DIR, 'train_baseline_structured.py'))

    if not check_marker('phase1_complete.txt'):
        logger.error("Phase 1 completion marker not found after training!")
        sys.exit(1)

    # Phase 2: Text Feature Generation + Text-Only Models
    logger.info("--- PHASE 2: Text Feature Generation & Text-Only Models ---")
    run_script(os.path.join(SCRIPTS_DIR, 'text_processor.py'))
    run_script(os.path.join(SCRIPTS_DIR, 'train_baseline_text.py'))

    if not check_marker('phase2_complete.txt'):
        logger.error("Phase 2 completion marker not found after training!")
        sys.exit(1)

    # Phase 3: Multimodal Fusion
    logger.info("--- PHASE 3: Multimodal Fusion (MLP) ---")
    run_script(os.path.join(SCRIPTS_DIR, 'train_multimodal.py'))

    if not check_marker('phase3_complete.txt'):
        logger.error("Phase 3 completion marker not found after training!")
        sys.exit(1)

    # Phase 4: Evaluation & Robustness
    logger.info("--- PHASE 4: Comprehensive Evaluation & Robustness ---")
    run_script(os.path.join(SCRIPTS_DIR, 'generate_evaluation_report.py'))
    run_script(os.path.join(SCRIPTS_DIR, 'robustness_testing.py'))

    # Final Summary
    logger.info("=" * 80)
    logger.info("PIPELINE EXECUTED SUCCESSFULLY END-TO-END!")
    logger.info(f"Completed at {datetime.now().isoformat()}")
    logger.info(f"Master log preserved at: {log_file}")
    logger.info("Evaluation metrics: results/")
    logger.info("Visualizations: figures/")
    logger.info("Trained models: models/")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
