  # Implementation Plan: AKI Prediction Pipeline

  ## Overview

  This implementation plan provides a comprehensive, step-by-step guide for building a research-grade machine learning pipeline to predict Acute Kidney Injury (AKI) using MIMIC-IV data. The pipeline follows a phased approach with strict temporal constraints, implementing KDIGO-based AKI labeling and supporting three modeling paradigms: structured-only baseline, text-only analysis, and multimodal fusion.

  The implementation is organized into five major stages executed sequentially, with validation checkpoints to ensure correctness at each phase.

  ## Tasks

  - [x] 1. Project setup and directory structure
    - Create directory structure: raw_data, processed_data, figures, models, results, logs, scripts
    - Create requirements.txt with dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn, transformers, torch
    - Create main configuration file with random seed (42) and key parameters
    - _Requirements: 18, 20_

  - [x] 2. Implement EDA Engine for exploratory data analysis
    - [x] 2.1 Create EDA_Engine class with schema validation
      - Implement validate_schema() to read CSV files and document column names, data types, null counts
      - Identify and log primary keys (stay_id, subject_id, hadm_id) and foreign key relationships
      - Save schema documentation to logs/schema_documentation.json
      - _Requirements: 5, 27_
    
    - [x] 2.2 Implement cohort visualization methods
      - Generate histogram of ICU stay lengths (duration_hours)
      - Generate histogram of ICU stays per patient
      - Generate histogram of age distribution
      - Generate bar chart of gender distribution
      - Save all plots as 300 DPI PNG files to figures directory
      - _Requirements: 6_
    
    - [x] 2.3 Implement laboratory data analysis
      - Generate histogram of raw creatinine values
      - Generate histogram of baseline creatinine distribution
      - Plot creatinine trajectories for 5 AKI and 5 non-AKI example patients
      - Generate histogram of lab measurement counts per ICU stay
      - Compute and visualize percentage of ICU stays containing each lab test type
      - _Requirements: 7_
    
    - [x] 2.4 Implement missingness pattern analysis
      - Compute missing rate for each laboratory test type
      - Generate bar chart of missing rates sorted from highest to lowest
      - Save missingness statistics to results/missingness_statistics.csv
      - Log warning if any feature has missing rate > 70%
      - _Requirements: 8_
    
    - [x] 2.5 Create eda.py script with command-line interface
      - Implement argument parsing for data directory paths
      - Add logging with timestamps for execution start/end
      - Call all EDA_Engine methods in sequence
      - _Requirements: 19, 20_

  - [x] 3. Checkpoint - Verify EDA outputs
    - Ensure all visualizations are generated in figures directory
    - Verify schema documentation exists and is complete
    - Review missingness statistics for data quality issues
    - Ask the user if questions arise

  - [x] 4. Implement AKI Labeler with KDIGO criteria
    - [x] 4.1 Create AKI_Labeler class with creatinine verification
      - Implement verify_creatinine_itemid() to check itemid 50912 in d_labitems.csv
      - Raise error if itemid does not correspond to creatinine
      - Log verified itemid and label name
      - _Requirements: 2.7, 21_
    
    - [x] 4.2 Implement ICU stay duration filtering
      - Compute duration_hours as (outtime - intime).total_seconds() / 3600
      - Filter out ICU stays with duration < 24 hours
      - Log number of excluded and retained stays
      - _Requirements: 1.5, 23_
    
    - [x] 4.3 Implement baseline creatinine computation
      - For each ICU stay, query creatinine measurements with same hadm_id and charttime < intime
      - If measurements exist before intime: baseline = minimum value
      - If no measurements before intime: baseline = first creatinine during ICU stay
      - Log warning if baseline cannot be determined
      - Document that same-admission baseline is used due to MIMIC-IV data limitations
      - _Requirements: 2.1, 2.2, 24_
    
    - [x] 4.4 Implement 48-hour rolling window AKI detection
      - Filter creatinine measurements to charttime >= intime + 24h
      - For each pair of measurements within 48-hour window, check if increase >= 0.3 mg/dL
      - Ensure only increases (not decreases) trigger criterion
      - Return True if any pair satisfies criterion, False otherwise
      - _Requirements: 2.3, 25_
    
    - [x] 4.5 Implement 7-day criterion AKI detection
      - Filter creatinine to charttime >= intime + 24h AND charttime <= min(outtime, intime + 7d)
      - Check if any measurement >= 1.5 * baseline_creatinine
      - Return True if criterion satisfied, False otherwise
      - _Requirements: 2.4, 26_
    
    - [x] 4.6 Implement label assignment logic
      - Assign aki_label = 1 if 48h criterion OR 7d criterion is satisfied
      - Assign aki_label = 0 if neither criterion is satisfied
      - Save labeled_stays.csv with columns: subject_id, hadm_id, stay_id, intime, outtime, duration_hours, baseline_creatinine, aki_label
      - _Requirements: 2.5, 2.6_
    
    - [x] 4.7 Create labeling.py script with execution logging
      - Implement command-line argument parsing
      - Add comprehensive logging of labeling statistics
      - Log AKI prevalence and baseline creatinine statistics
      - _Requirements: 19, 20_

  - [x] 5. Implement Cohort Analyzer for AKI prevalence analysis
    - [x] 5.1 Create Cohort_Analyzer class for AKI statistics
      - Compute and report percentage of ICU stays with AKI label 1
      - Generate bar chart showing AKI prevalence
      - Generate histogram of maximum creatinine increase for AKI-positive cases
      - Generate comparison plots: baseline creatinine for AKI-positive vs AKI-negative
      - Generate comparison plots: age distribution for AKI-positive vs AKI-negative
      - Save all visualizations to figures directory
      - _Requirements: 9_
    
    - [x] 5.2 Implement statistical summary generation
      - Generate summary table with mean, median, std, min, max for continuous features
      - Generate frequency table for categorical features
      - Compute correlation matrix for numeric features
      - Generate correlation heatmap
      - Save all summaries to results directory
      - _Requirements: 34_

  - [x] 6. Checkpoint - Verify AKI labeling correctness
    - Review AKI prevalence (should be realistic for ICU population, typically 10-30%)
    - Verify baseline creatinine values are reasonable (typically 0.5-2.0 mg/dL)
    - Check that 48h criterion uses increases only, not absolute differences
    - Ensure all tests pass, ask the user if questions arise

  - [x] 7. Implement Feature Extractor for structured features
    - [x] 7.1 Create Lab_Aggregator for laboratory test selection
      - Compute coverage percentage for each lab test type
      - Select labs with coverage >= 30%
      - Prioritize AKI-relevant labs: BUN, Sodium, Potassium, Chloride, Bicarbonate, Lactate, WBC, Hemoglobin, Platelets, Glucose, Calcium, Magnesium, Phosphate
      - Verify all selected lab itemids against d_labitems.csv
      - Exclude creatinine from aggregated features (baseline already included separately)
      - Log final list of selected labs with itemids and coverage percentages
      - _Requirements: 11, 21_
    
    - [x] 7.2 Implement temporal filtering and lab aggregation
      - For each ICU stay, filter lab measurements to charttime < intime + 24h
      - For each selected lab, compute: mean, min, max, std, first, last
      - Create missingness indicator: 1 if no measurements, 0 otherwise
      - _Requirements: 1.1, 1.3, 10.5_
    
    - [x] 7.3 Implement demographic feature extraction
      - Extract age at ICU admission: (intime - date_of_birth).days / 365.25
      - Extract gender as binary: 1 if 'M', 0 if 'F'
      - Extract ICU type and apply one-hot encoding
      - Extract baseline_creatinine from labeled_stays.csv
      - _Requirements: 10.1, 10.2, 10.3, 10.4_
    
    - [x] 7.4 Create Feature_Extractor class to combine all features
      - Combine demographics, baseline creatinine, and lab aggregations
      - Include identifiers: subject_id, stay_id, aki_label
      - Save to processed_data/structured_dataset.csv
      - Log feature extraction statistics and final feature count
      - _Requirements: 10.6, 10.7_
    
    - [x] 7.5 Create feature_engineering.py script
      - Implement command-line argument parsing
      - Add execution logging with timestamps
      - Call Feature_Extractor methods in sequence
      - _Requirements: 19, 20_

  - [x] 8. Implement Data Splitter for patient-level splitting
    - [x] 8.1 Create Data_Splitter class with patient-level logic
      - Extract unique patient identifiers (subject_id)
      - Set random seed to 42
      - Shuffle patients and split into 70/15/15 proportions
      - Assign all ICU stays from each patient to same split
      - _Requirements: 3.1, 3.2, 3.3_
    
    - [x] 8.2 Implement split integrity verification
      - Verify no patient appears in multiple splits
      - Compute AKI prevalence in each split
      - Log warning if prevalence differs by > 10 percentage points
      - Verify sum of samples equals total ICU stays
      - _Requirements: 3.5, 3.6, 31_
    
    - [x] 8.3 Implement imputation value computation
      - Compute median for each feature using only training set
      - Apply median imputation to all splits
      - Create missingness indicators for all features with missing values
      - Save imputation values to logs for reproducibility
      - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
    
    - [x] 8.4 Save split data and metadata
      - Save train_patients.csv, val_patients.csv, test_patients.csv
      - Save X_train_structured.npy, y_train.npy, X_val_structured.npy, y_val.npy, X_test_structured.npy, y_test.npy
      - Log split statistics: patient counts, stay counts, AKI prevalence per split
      - _Requirements: 3.4_

  - [ ] 9. Checkpoint - Verify data splitting and feature engineering
    - Verify no patient overlap between splits
    - Check AKI prevalence balance across splits
    - Verify feature matrix shapes are correct
    - Ensure all tests pass, ask the user if questions arise

  - [ ] 10. Implement Text Processor for clinical text embeddings
    - [ ] 10.1 Create Text_Processor class with discharge summary linking
      - Link discharge summaries from discharge.csv to ICU stays using hadm_id
      - If multiple summaries per stay, concatenate chronologically
      - Log number of ICU stays with missing discharge summaries
      - Compute and report percentage of stays with available text
      - _Requirements: 12.1, 12.2, 30_
    
    - [ ] 10.2 Implement text cleaning and preprocessing
      - Remove non-ASCII characters
      - Normalize whitespace with regex
      - Remove special characters
      - _Requirements: 12.3_
    
    - [ ] 10.3 Implement BioClinicalBERT embedding generation
      - Load BioClinicalBERT model: 'emilyalsentzer/Bio_ClinicalBERT'
      - Tokenize text with max_length=512 and truncation
      - Extract [CLS] token embeddings (768-dimensional)
      - Process in batches of 32 for memory efficiency
      - For missing text, create zero vector of dimension 768
      - _Requirements: 12.4, 12.5, 30.1, 32.3_
    
    - [ ] 10.4 Save embeddings and split by patient assignments
      - Save text_embeddings.npy with shape (n_stays, 768)
      - Save text_stay_ids.csv mapping row indices to stay_id
      - Split embeddings according to patient splits from Data_Splitter
      - Save X_train_text.npy, X_val_text.npy, X_test_text.npy
      - _Requirements: 12.5_
    
    - [ ] 10.5 Document temporal limitations in logs
      - Document that discharge summaries contain post-ICU information
      - Note this is a known temporal limitation for representation analysis
      - Document that future work will use temporally aligned clinical notes
      - _Requirements: 12.6, 12.7_

  - [ ] 11. Implement Phase 1: Structured-Only Baseline Models
    - [ ] 11.1 Create Structured_Model_Trainer class
      - Load training and validation structured data
      - Implement feature scaling with StandardScaler fitted on training set
      - Apply same scaling to validation set
      - Save scaler to models/scaler.pkl
      - _Requirements: 28_
    
    - [ ] 11.2 Train Logistic Regression model
      - Initialize LogisticRegression with random_state=42, max_iter=1000, solver='lbfgs', class_weight='balanced'
      - Train on scaled training data
      - Save model to models/logistic_regression.pkl
      - Save hyperparameters to models/logistic_regression_params.json
      - _Requirements: 13.1, 13.3, 13.5, 13.6_
    
    - [ ] 11.3 Train Random Forest model
      - Initialize RandomForestClassifier with n_estimators=100, random_state=42, max_depth=10, class_weight='balanced'
      - Train on scaled training data
      - Save model to models/random_forest.pkl
      - Save hyperparameters to models/random_forest_params.json
      - _Requirements: 13.2, 13.3, 13.5, 13.6_
    
    - [ ] 11.4 Create train_structured.py script
      - Implement command-line argument parsing
      - Add logging with training duration and validation metrics
      - Document that Phase 1 uses only temporally valid structured data
      - Create phase completion marker: logs/phase1_complete.txt
      - _Requirements: 13.7, 19, 20, 22.1_

  - [ ] 12. Implement Phase 2: Text-Only Model
    - [ ] 12.1 Create Text_Model_Trainer class
      - Load training and validation text embeddings
      - Verify Phase 1 completion before proceeding
      - _Requirements: 22.6_
    
    - [ ] 12.2 Train text-only classifier
      - Initialize LogisticRegression with random_state=42, max_iter=1000, solver='lbfgs'
      - Train on text embeddings only (no structured features)
      - Save model to models/text_classifier.pkl
      - Save hyperparameters to models/text_classifier_params.json
      - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_
    
    - [ ] 12.3 Create train_text.py script
      - Implement command-line argument parsing
      - Add logging with training duration and validation metrics
      - Document temporal limitations of discharge summaries
      - Create phase completion marker: logs/phase2_complete.txt
      - _Requirements: 14.6, 19, 20, 22.2_

  - [ ] 13. Implement Phase 3: Multimodal Fusion Model
    - [ ] 13.1 Create Fusion_Model_Trainer class
      - Load training and validation structured features and text embeddings
      - Verify Phase 1 and Phase 2 completion before proceeding
      - Load scaler from Phase 1
      - _Requirements: 22.6_
    
    - [ ] 13.2 Implement feature concatenation
      - Scale structured features using saved scaler
      - Concatenate scaled structured features with text embeddings
      - Verify concatenated feature dimensions
      - _Requirements: 15.2_
    
    - [ ] 13.3 Train multimodal MLP
      - Initialize MLPClassifier with hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam', random_state=42, max_iter=500, early_stopping=True, validation_fraction=0.1
      - Train on concatenated features
      - Save model to models/fusion_model.pkl
      - Save hyperparameters including input dimensions to models/fusion_model_params.json
      - _Requirements: 15.1, 15.3, 15.4, 15.5_
    
    - [ ] 13.4 Create train_multimodal.py script
      - Implement command-line argument parsing
      - Add logging with training duration and validation metrics
      - Document combination of temporally valid structured and retrospective text features
      - Create phase completion marker: logs/phase3_complete.txt
      - _Requirements: 15.6, 19, 20, 22.3_

  - [ ] 14. Checkpoint - Verify all models trained successfully
    - Verify all model files exist in models directory
    - Check that hyperparameter JSON files are complete
    - Review validation metrics for reasonable performance
    - Ensure all tests pass, ask the user if questions arise

  - [ ] 15. Implement comprehensive model evaluation
    - [ ] 15.1 Create Metrics_Calculator class
      - Load all trained models from models directory
      - Load test data (structured and text)
      - _Requirements: 16_
    
    - [ ] 15.2 Implement metric computation for all models
      - Compute AUROC, AUPRC, accuracy, precision, recall, F1 score
      - Compute Brier score for calibration assessment
      - Save all metrics to results/model_metrics.csv
      - _Requirements: 16.1, 16.2, 16.3, 16.4_
    
    - [ ] 15.3 Generate ROC and Precision-Recall curves
      - Generate ROC curves for all models on single plot
      - Generate Precision-Recall curves for all models on single plot
      - Save to figures/roc_curves.png and figures/pr_curves.png
      - _Requirements: 16.5_
    
    - [ ] 15.4 Generate calibration curves
      - Compute 10-bin calibration curves for all models
      - Plot reliability diagrams with perfect calibration reference line
      - Save to figures/calibration_curves.png
      - _Requirements: 16 (calibration metrics)_
    
    - [ ] 15.5 Generate model comparison visualizations
      - Create bar chart comparing AUROC across all models
      - Create bar chart comparing AUPRC across all models
      - Create grouped bar chart for precision, recall, F1 score
      - Save all comparison plots to figures directory
      - _Requirements: 35_

  - [ ] 16. Implement Phase 4: Modality Masking Robustness Testing
    - [ ] 16.1 Create Modality_Masker class
      - Load trained fusion model and scaler
      - Load test data (structured and text)
      - Verify Phase 3 completion before proceeding
      - _Requirements: 22.4_
    
    - [ ] 16.2 Evaluate full fusion model
      - Concatenate scaled structured features with text embeddings
      - Compute all metrics on test set with full modalities
      - Save as baseline for degradation comparison
      - _Requirements: 17.1_
    
    - [ ] 16.3 Evaluate with text modality masked
      - Replace text embeddings with zero vectors (same dimension)
      - Evaluate fusion model at inference time (no retraining)
      - Compute all metrics
      - _Requirements: 17.2, 17.4_
    
    - [ ] 16.4 Evaluate with structured modality masked
      - Replace structured features with zero vectors (same dimension)
      - Evaluate fusion model at inference time (no retraining)
      - Compute all metrics
      - _Requirements: 17.3, 17.5_
    
    - [ ] 16.5 Compute performance degradation
      - Calculate percentage drop for each masking condition relative to full model
      - Save robustness metrics to results/robustness_metrics.csv
      - Save degradation percentages to results/performance_degradation.csv
      - _Requirements: 17.6, 17.7_
    
    - [ ] 16.6 Generate robustness comparison visualizations
      - Create bar chart comparing performance across masking conditions
      - Visualize degradation patterns
      - Save to figures/robustness_comparison.png
      - Document that masking simulates real-world missing modality scenarios
      - _Requirements: 17.8_

  - [ ] 17. Implement memory-efficient processing and error recovery
    - [ ] 17.1 Add chunked reading for large files
      - Implement chunked CSV reading for files > 1 GB
      - Release memory for intermediate data structures
      - Log peak memory usage for major processing steps
      - Log warning if available memory < 10% of total
      - _Requirements: 32_
    
    - [ ] 17.2 Implement checkpointing for text embeddings
      - Save intermediate embedding results every 1000 samples
      - Check for existing checkpoints on script restart
      - Resume from last checkpoint if available
      - Remove temporary checkpoint files when complete
      - Log checkpoint creation and restoration events
      - _Requirements: 33_

  - [ ] 18. Create master execution script and documentation
    - [ ] 18.1 Create run_pipeline.py master script
      - Implement sequential execution of all phases
      - Check for phase completion markers before proceeding
      - Add comprehensive logging to logs/master_execution.log
      - Record execution sequence and timestamps
      - _Requirements: 20.5_
    
    - [ ] 18.2 Generate requirements.txt with pinned versions
      - List all Python dependencies with version numbers
      - Include: pandas, numpy, scikit-learn, matplotlib, seaborn, transformers, torch
      - _Requirements: 20.4_
    
    - [ ] 18.3 Create README.md with usage instructions
      - Document directory structure
      - Provide step-by-step execution instructions
      - Document expected outputs for each phase
      - Include troubleshooting section

  - [ ] 19. Final checkpoint - End-to-end pipeline validation
    - Run complete pipeline from raw data to final evaluation
    - Verify all outputs exist in expected locations
    - Review final metrics for scientific validity
    - Check that all visualizations are publication-quality (300 DPI)
    - Ensure reproducibility with fixed random seed
    - Ask the user if questions arise or if ready to proceed with publication

  ## Notes

  - All tasks reference specific requirements for traceability
  - The pipeline enforces strict temporal constraints (24-hour cutoff) throughout
  - Patient-level splitting prevents data contamination
  - Phase-based execution ensures proper modality separation
  - Checkpoints enable validation at critical stages
  - All random operations use seed 42 for reproducibility
  - Class imbalance is handled with class_weight='balanced' in classifiers
  - Calibration metrics (Brier score, calibration curves) assess prediction reliability
  - Test-time modality masking evaluates robustness without retraining
  - Memory-efficient processing supports large-scale datasets
