# Requirements Document: AKI Prediction Pipeline

## Introduction

This document specifies requirements for a research-grade machine learning pipeline to predict Acute Kidney Injury (AKI) after the first 24 hours of ICU admission using MIMIC-IV data. The system implements KDIGO-based AKI labeling methodology with strict temporal constraints to prevent data leakage, supporting three modeling approaches: structured-only, text-only, and multimodal fusion. The pipeline is designed for scientific publication with emphasis on reproducibility, proper validation, and research rigor.

## Glossary

- **AKI_Labeler**: Component that assigns binary AKI labels to ICU stays based on KDIGO criteria
- **Baseline_Creatinine**: Lowest creatinine value within the same hospital admission (same hadm_id) measured before ICU intime, or first ICU creatinine if none exists before intime
- **BioClinicalBERT**: Pre-trained transformer model for clinical text embedding
- **Cohort_Analyzer**: Component that generates statistical summaries and visualizations of patient cohorts
- **Data_Splitter**: Component that partitions data into train/validation/test sets at patient level
- **EDA_Engine**: Exploratory Data Analysis component that generates publication-quality visualizations
- **Feature_Extractor**: Component that derives structured features from raw temporal data
- **Fusion_Model**: Multimodal neural network that combines structured and text features
- **ICU_Stay**: Single continuous period of intensive care unit admission
- **KDIGO_Criteria**: Kidney Disease: Improving Global Outcomes clinical practice guidelines for AKI diagnosis
- **Lab_Aggregator**: Component that computes statistical summaries of laboratory values
- **Missingness_Indicator**: Binary feature indicating whether original value was missing
- **Modality_Masker**: Component that replaces feature vectors with zeros to test robustness
- **Structured_Model**: Machine learning model using only tabular clinical features
- **Temporal_Cutoff**: 24-hour boundary after ICU admission; only data before this point may be used as features
- **Text_Model**: Machine learning model using only clinical note embeddings
- **Training_Set**: Subset of data used for model parameter learning and imputation statistics


## Requirements

### Requirement 1: Temporal Data Constraint

**User Story:** As a researcher, I want to ensure no data leakage occurs, so that the model predictions are clinically valid and temporally realistic.

#### Acceptance Criteria

1. THE Feature_Extractor SHALL use only data with timestamps before 24 hours after ICU admission intime
2. THE AKI_Labeler SHALL ignore all creatinine measurements within the first 24 hours of ICU stay for AKI detection
3. WHEN computing aggregated features, THE Lab_Aggregator SHALL filter measurements to include only those with charttime less than intime plus 24 hours
4. THE Feature_Extractor SHALL exclude baseline creatinine from aggregated creatinine features to prevent label leakage
5. FOR ALL ICU stays with duration less than 24 hours, THE AKI_Labeler SHALL exclude them from the dataset

### Requirement 2: KDIGO-Based AKI Labeling

**User Story:** As a researcher, I want to label AKI cases using validated clinical criteria, so that the prediction target aligns with medical standards.

#### Acceptance Criteria

1. WHEN determining baseline creatinine, THE AKI_Labeler SHALL use the lowest creatinine value within the same hospital admission (same hadm_id) measured before ICU intime
2. IF no creatinine measurements exist before ICU intime within the same hospital admission, THEN THE AKI_Labeler SHALL use the first creatinine measurement during the ICU stay as baseline
3. WHEN evaluating the 48-hour criterion, THE AKI_Labeler SHALL check if any creatinine measurement after hour 24 shows an increase of 0.3 mg/dL or greater within any 48-hour rolling window
4. WHEN evaluating the 7-day criterion, THE AKI_Labeler SHALL check if any creatinine measurement after hour 24 reaches 1.5 times the baseline value within 7 days after ICU admission OR until ICU discharge, whichever comes first
5. IF either the 48-hour criterion or the 7-day criterion is satisfied, THEN THE AKI_Labeler SHALL assign label 1 to the ICU stay
6. IF neither criterion is satisfied, THEN THE AKI_Labeler SHALL assign label 0 to the ICU stay
7. THE AKI_Labeler SHALL verify that creatinine corresponds to itemid 50912 by cross-referencing d_labitems.csv
8. THE AKI_Labeler SHALL document that baseline creatinine is restricted to same-admission measurements because MIMIC-IV does not reliably capture outpatient history

### Requirement 3: Patient-Level Data Splitting

**User Story:** As a researcher, I want to split data at the patient level, so that the same patient does not appear in multiple splits and evaluation is unbiased.

#### Acceptance Criteria

1. THE Data_Splitter SHALL partition unique patients into training, validation, and test sets with proportions 70%, 15%, and 15% respectively
2. THE Data_Splitter SHALL use random seed 42 for all random operations
3. WHEN a patient has multiple ICU stays, THE Data_Splitter SHALL assign all ICU stays from that patient to the same data split
4. THE Data_Splitter SHALL log the number of patients and ICU stays in each split
5. FOR ALL patients in the validation set, THE Data_Splitter SHALL ensure no patient identifier appears in the training set
6. FOR ALL patients in the test set, THE Data_Splitter SHALL ensure no patient identifier appears in the training or validation sets

### Requirement 4: Missing Data Handling

**User Story:** As a researcher, I want to handle missing data systematically, so that models can process incomplete records without introducing bias.

#### Acceptance Criteria

1. WHEN computing imputation values, THE Feature_Extractor SHALL calculate median values using only the Training_Set
2. WHEN a feature value is missing, THE Feature_Extractor SHALL replace it with the corresponding median value from the Training_Set
3. FOR ALL features with missing values, THE Feature_Extractor SHALL create a corresponding Missingness_Indicator feature
4. THE Missingness_Indicator SHALL have value 1 when the original feature was missing and value 0 when the original feature was present
5. THE Feature_Extractor SHALL apply the same imputation medians computed from the Training_Set to validation and test sets

### Requirement 5: Dataset Inspection and Schema Validation

**User Story:** As a researcher, I want to understand the structure of raw data files, so that I can identify relationships and design appropriate processing logic.

#### Acceptance Criteria

1. FOR ALL CSV files in the dataset, THE EDA_Engine SHALL print column names, data types, null value counts, and total row counts
2. THE EDA_Engine SHALL identify and document primary key columns for each table
3. THE EDA_Engine SHALL identify and document timestamp columns for each table
4. THE EDA_Engine SHALL identify and document foreign key relationships between tables
5. THE EDA_Engine SHALL save the schema documentation to a structured format in the logs directory


### Requirement 6: Cohort Visualization

**User Story:** As a researcher, I want publication-quality visualizations of the patient cohort, so that I can understand population characteristics and communicate findings.

#### Acceptance Criteria

1. THE Cohort_Analyzer SHALL generate a histogram showing the distribution of ICU stay lengths in hours
2. THE Cohort_Analyzer SHALL generate a histogram showing the number of ICU stays per patient
3. THE Cohort_Analyzer SHALL generate a histogram showing the age distribution of patients
4. THE Cohort_Analyzer SHALL generate a bar chart showing the gender distribution of patients
5. FOR ALL visualizations, THE Cohort_Analyzer SHALL include axis labels, titles, and legends where appropriate
6. FOR ALL visualizations, THE Cohort_Analyzer SHALL save output as high-resolution PNG files with minimum 300 DPI to the figures directory
7. THE Cohort_Analyzer SHALL use a consistent color scheme across all visualizations

### Requirement 7: Laboratory Data Analysis

**User Story:** As a researcher, I want to analyze laboratory measurement patterns, so that I can identify data quality issues and understand feature availability.

#### Acceptance Criteria

1. THE EDA_Engine SHALL generate a histogram showing the distribution of raw creatinine values across all measurements
2. THE EDA_Engine SHALL generate a histogram showing the distribution of baseline creatinine values across ICU stays
3. THE EDA_Engine SHALL generate line plots showing creatinine trajectories over time for at least 5 example patients with AKI and 5 without AKI
4. THE EDA_Engine SHALL generate a histogram showing the distribution of laboratory measurement counts per ICU stay
5. THE EDA_Engine SHALL compute and visualize the percentage of ICU stays containing each laboratory test type
6. FOR ALL laboratory visualizations, THE EDA_Engine SHALL save output as high-resolution PNG files to the figures directory

### Requirement 8: Missingness Pattern Analysis

**User Story:** As a researcher, I want to understand missing data patterns, so that I can assess data quality and justify imputation strategies.

#### Acceptance Criteria

1. THE EDA_Engine SHALL compute the missing rate for each laboratory test type across all ICU stays
2. THE EDA_Engine SHALL generate a bar chart showing missing rates sorted from highest to lowest
3. IF computationally feasible within 10 minutes, THEN THE EDA_Engine SHALL generate a heatmap showing missingness patterns across features and samples
4. THE EDA_Engine SHALL save missingness statistics to a CSV file in the results directory
5. THE EDA_Engine SHALL log a warning if any feature has a missing rate exceeding 70%

### Requirement 9: AKI Prevalence Analysis

**User Story:** As a researcher, I want to analyze AKI distribution and characteristics, so that I can assess class balance and understand the prediction target.

#### Acceptance Criteria

1. WHEN AKI labeling is complete, THE Cohort_Analyzer SHALL compute and report the percentage of ICU stays with AKI label 1
2. THE Cohort_Analyzer SHALL generate a bar chart showing AKI prevalence across the dataset
3. THE Cohort_Analyzer SHALL generate a histogram showing the distribution of maximum creatinine increase values for AKI-positive cases
4. THE Cohort_Analyzer SHALL generate comparison plots showing baseline creatinine distributions for AKI-positive versus AKI-negative cases
5. THE Cohort_Analyzer SHALL generate comparison plots showing age distributions for AKI-positive versus AKI-negative cases
6. THE Cohort_Analyzer SHALL save all AKI analysis visualizations to the figures directory

### Requirement 10: Structured Feature Engineering

**User Story:** As a researcher, I want to extract clinically relevant structured features from the first 24 hours, so that models can learn from tabular data.

#### Acceptance Criteria

1. THE Feature_Extractor SHALL extract patient age at ICU admission as a numeric feature
2. THE Feature_Extractor SHALL extract patient gender as a binary feature
3. THE Feature_Extractor SHALL extract ICU type and encode it using one-hot encoding
4. THE Feature_Extractor SHALL extract baseline creatinine as a numeric feature
5. FOR ALL selected laboratory tests, THE Lab_Aggregator SHALL compute mean, minimum, maximum, standard deviation, first value, and last value within the first 24 hours
6. THE Feature_Extractor SHALL save the processed structured dataset to processed_data/structured_dataset.csv
7. THE Feature_Extractor SHALL include patient identifier, ICU stay identifier, and AKI label in the output dataset


### Requirement 11: Laboratory Test Selection

**User Story:** As a researcher, I want to select clinically relevant and frequently available laboratory tests, so that features are both informative and have sufficient coverage.

#### Acceptance Criteria

1. THE Feature_Extractor SHALL include only laboratory tests that are present in at least 30% of ICU stays
2. THE Feature_Extractor SHALL prioritize the following AKI-relevant laboratory tests if they meet the 30% frequency threshold: BUN, Sodium, Potassium, Chloride, Bicarbonate, Lactate, WBC, Hemoglobin, Platelets, Glucose, Calcium, Magnesium, Phosphate
3. THE Feature_Extractor SHALL verify all laboratory test itemids by cross-referencing d_labitems.csv before feature extraction
4. THE Feature_Extractor SHALL document the final list of selected laboratory tests with their itemids, label names, and exact coverage percentages in the logs directory
5. THE Feature_Extractor SHALL exclude creatinine from aggregated laboratory features since baseline creatinine is already included as a separate feature to prevent label leakage

### Requirement 12: Clinical Text Processing (Phase 2 Only)

**User Story:** As a researcher, I want to process discharge summaries into embeddings for Phase 2 text modeling, so that I can analyze the predictive value of clinical text with acknowledged temporal limitations.

#### Acceptance Criteria

1. THE Text_Model SHALL link discharge summaries from discharge.csv to ICU stays using hospital admission identifiers
2. WHEN multiple discharge summaries exist for a single ICU stay, THE Text_Model SHALL concatenate them in chronological order
3. THE Text_Model SHALL remove special characters, excessive whitespace, and non-ASCII characters from text
4. THE Text_Model SHALL use BioClinicalBERT to generate embeddings for each discharge summary
5. THE Text_Model SHALL save precomputed embeddings to processed_data/text_embeddings.npy with corresponding ICU stay identifiers
6. THE Text_Model SHALL document explicitly that discharge summaries are used retrospectively for representation analysis and contain post-ICU information
7. THE Text_Model SHALL document that clinical text usage represents a known temporal limitation and that future work will incorporate temporally aligned clinical notes
8. THE Text_Model SHALL NOT be used in Phase 1 structured baseline experiments

### Requirement 13: Structured-Only Baseline Models (Phase 1)

**User Story:** As a researcher, I want to train baseline models using only structured features, so that I can establish performance benchmarks without any text data.

#### Acceptance Criteria

1. THE Structured_Model SHALL train a Logistic Regression classifier using only structured features from the Training_Set
2. THE Structured_Model SHALL train a Random Forest classifier using only structured features from the Training_Set
3. THE Structured_Model SHALL use random seed 42 for all models with stochastic components
4. THE Structured_Model SHALL NOT use any text embeddings or discharge summary data in Phase 1 experiments
5. THE Structured_Model SHALL save trained model artifacts to the models directory
6. THE Structured_Model SHALL log hyperparameters for each trained model
7. THE Structured_Model SHALL document that Phase 1 models use only temporally valid structured data from the first 24 hours

### Requirement 14: Text-Only Model (Phase 2)

**User Story:** As a researcher, I want to train a model using only clinical text embeddings, so that I can assess the predictive value of textual information with acknowledged temporal limitations.

#### Acceptance Criteria

1. THE Text_Model SHALL train a classifier using only BioClinicalBERT embeddings from the Training_Set
2. THE Text_Model SHALL use random seed 42 for all stochastic operations
3. THE Text_Model SHALL NOT use any structured features in Phase 2 text-only experiments
4. THE Text_Model SHALL save the trained model artifact to the models directory
5. THE Text_Model SHALL log model architecture and hyperparameters
6. THE Text_Model SHALL document that this model uses retrospective discharge summaries with known temporal limitations

### Requirement 15: Multimodal Fusion Model (Phase 3)

**User Story:** As a researcher, I want to train a model that combines structured and text features, so that I can leverage complementary information from both modalities.

#### Acceptance Criteria

1. THE Fusion_Model SHALL implement a multi-layer perceptron that accepts concatenated structured features and text embeddings as input
2. THE Fusion_Model SHALL train using paired structured and text data from the Training_Set
3. THE Fusion_Model SHALL use random seed 42 for weight initialization and training
4. THE Fusion_Model SHALL save the trained model artifact to the models directory
5. THE Fusion_Model SHALL log model architecture, layer dimensions, and hyperparameters
6. THE Fusion_Model SHALL document that it combines temporally valid structured features with retrospective text embeddings

### Requirement 16: Model Evaluation Metrics

**User Story:** As a researcher, I want comprehensive evaluation metrics, so that I can assess model performance from multiple perspectives.

#### Acceptance Criteria

1. FOR ALL trained models, THE evaluation component SHALL compute AUROC on the test set
2. FOR ALL trained models, THE evaluation component SHALL compute AUPRC on the test set
3. FOR ALL trained models, THE evaluation component SHALL compute accuracy, precision, recall, and F1 score on the test set
4. THE evaluation component SHALL save all metrics to a structured CSV file in the results directory
5. THE evaluation component SHALL generate ROC curves and Precision-Recall curves for all models and save them to the figures directory


### Requirement 17: Missing Modality Robustness Testing (Phase 4)

**User Story:** As a researcher, I want to test model robustness when modalities are missing at test time, so that I can understand degradation patterns and clinical applicability.

#### Acceptance Criteria

1. THE Modality_Masker SHALL perform test-time modality masking only, not train separate reduced models
2. WHEN testing the Fusion_Model with missing text modality, THE Modality_Masker SHALL replace text embeddings with zero vectors of the same dimension at inference time
3. WHEN testing the Fusion_Model with missing structured modality, THE Modality_Masker SHALL replace structured features with zero vectors of the same dimension at inference time
4. THE Modality_Masker SHALL evaluate the full trained Fusion_Model on the test set with text modality masked and report all standard metrics
5. THE Modality_Masker SHALL evaluate the full trained Fusion_Model on the test set with structured modality masked and report all standard metrics
6. THE Modality_Masker SHALL compute and report the percentage performance drop for each masking condition relative to full modality performance
7. THE Modality_Masker SHALL save robustness test results to a CSV file in the results directory
8. THE Modality_Masker SHALL document that masking simulates real-world scenarios where one modality may be unavailable at inference time

### Requirement 18: Project Directory Structure

**User Story:** As a researcher, I want a well-organized project structure, so that artifacts are easy to locate and the project is maintainable.

#### Acceptance Criteria

1. THE system SHALL create the following directories if they do not exist: raw_data, processed_data, figures, models, results, logs, scripts
2. THE system SHALL store original CSV files in the raw_data directory
3. THE system SHALL store processed datasets and embeddings in the processed_data directory
4. THE system SHALL store all visualizations in the figures directory
5. THE system SHALL store trained model artifacts in the models directory
6. THE system SHALL store evaluation metrics and analysis results in the results directory
7. THE system SHALL store execution logs and documentation in the logs directory
8. THE system SHALL store executable Python scripts in the scripts directory

### Requirement 19: Modular Script Architecture

**User Story:** As a researcher, I want modular scripts for each pipeline stage, so that I can execute, debug, and modify components independently.

#### Acceptance Criteria

1. THE system SHALL implement an eda.py script that performs all exploratory data analysis and visualization tasks
2. THE system SHALL implement a labeling.py script that performs KDIGO-based AKI labeling
3. THE system SHALL implement a feature_engineering.py script that extracts and processes structured features
4. THE system SHALL implement a train_structured.py script that trains structured-only models
5. THE system SHALL implement a train_text.py script that processes text and trains text-only models
6. THE system SHALL implement a train_multimodal.py script that trains the fusion model
7. FOR ALL scripts, THE system SHALL include command-line argument parsing for key parameters
8. FOR ALL scripts, THE system SHALL log execution start time, end time, and key outputs

### Requirement 20: Reproducibility and Logging

**User Story:** As a researcher, I want comprehensive logging and reproducibility controls, so that experiments can be replicated and audited.

#### Acceptance Criteria

1. THE system SHALL use random seed 42 for all random number generators including NumPy, Python random, and framework-specific generators
2. FOR ALL data transformations, THE system SHALL log the transformation type, input shape, output shape, and timestamp
3. FOR ALL model training runs, THE system SHALL log hyperparameters, training duration, and final metrics
4. THE system SHALL save a requirements.txt file listing all Python package dependencies with version numbers
5. THE system SHALL create a master log file that records the execution sequence of all pipeline stages
6. WHEN an error occurs, THE system SHALL log the full stack trace and context information to the logs directory

### Requirement 21: Creatinine Itemid Verification

**User Story:** As a researcher, I want to verify laboratory test identifiers, so that I can ensure correct data extraction.

#### Acceptance Criteria

1. THE AKI_Labeler SHALL read d_labitems.csv to verify that itemid 50912 corresponds to creatinine
2. IF itemid 50912 does not correspond to creatinine, THEN THE AKI_Labeler SHALL raise an error and halt execution
3. THE AKI_Labeler SHALL log the verified label name and itemid to the logs directory
4. THE Feature_Extractor SHALL verify all selected laboratory test itemids against d_labitems.csv before feature extraction
5. THE Feature_Extractor SHALL log a mapping of itemid to laboratory test name for all selected tests

### Requirement 22: Phase-Based Execution Control

**User Story:** As a researcher, I want to execute the pipeline in distinct phases with strict modality separation, so that I can validate each stage and prevent mixing of baseline and text experiments.

#### Acceptance Criteria

1. THE system SHALL support execution of Phase 1 (structured-only baseline) using only temporally valid structured features with no text data
2. THE system SHALL support execution of Phase 2 (text-only model) using only discharge summary embeddings with documented temporal limitations
3. THE system SHALL support execution of Phase 3 (multimodal fusion) combining structured and text features only after Phases 1 and 2 are complete
4. THE system SHALL support execution of Phase 4 (test-time modality masking) only after Phase 3 completion
5. THE system SHALL create phase completion marker files in the logs directory to track pipeline progress
6. WHEN a phase is executed, THE system SHALL verify that all prerequisite phases have been completed
7. THE system SHALL enforce strict separation between Phase 1 structured baseline and Phase 2 text modeling to prevent modality mixing


### Requirement 23: ICU Stay Duration Filtering

**User Story:** As a researcher, I want to exclude very short ICU stays, so that the prediction task is clinically meaningful and temporally feasible.

#### Acceptance Criteria

1. THE AKI_Labeler SHALL compute ICU stay duration as the difference between outtime and intime from icustays.csv
2. WHEN an ICU stay has duration less than 24 hours, THE AKI_Labeler SHALL exclude it from the final dataset
3. THE AKI_Labeler SHALL log the number of ICU stays excluded due to insufficient duration
4. THE AKI_Labeler SHALL log the number of ICU stays retained after duration filtering

### Requirement 24: Baseline Creatinine Computation Logic

**User Story:** As a researcher, I want a clear and correct baseline creatinine calculation restricted to same-admission measurements, so that KDIGO criteria are properly applied with reproducible methodology.

#### Acceptance Criteria

1. WHEN computing baseline creatinine for an ICU stay, THE AKI_Labeler SHALL query only creatinine measurements with the same hospital admission identifier (hadm_id) as the ICU stay
2. THE AKI_Labeler SHALL NOT use creatinine measurements from prior hospital admissions or outpatient records
3. THE AKI_Labeler SHALL filter creatinine measurements to include only those with charttime before the ICU intime within the same admission
4. IF at least one creatinine measurement exists before ICU intime within the same admission, THEN THE AKI_Labeler SHALL set baseline creatinine to the minimum value among those measurements
5. IF no creatinine measurements exist before ICU intime within the same admission, THEN THE AKI_Labeler SHALL set baseline creatinine to the first creatinine measurement during the ICU stay
6. THE AKI_Labeler SHALL log a warning if baseline creatinine cannot be determined for any ICU stay
7. THE AKI_Labeler SHALL document that same-admission baseline is used because MIMIC-IV does not reliably capture outpatient history and this approach is reproducible and clinically standard

### Requirement 25: Rolling Window AKI Detection

**User Story:** As a researcher, I want to detect AKI using a 48-hour rolling window, so that acute changes in kidney function are captured.

#### Acceptance Criteria

1. WHEN evaluating the 48-hour criterion, THE AKI_Labeler SHALL consider only creatinine measurements with charttime at least 24 hours after ICU intime
2. FOR ALL pairs of creatinine measurements where the time difference is at most 48 hours, THE AKI_Labeler SHALL compute the absolute difference in creatinine values
3. IF any creatinine difference within a 48-hour window is greater than or equal to 0.3 mg/dL, THEN THE AKI_Labeler SHALL mark the 48-hour criterion as satisfied
4. THE AKI_Labeler SHALL use vectorized operations or efficient algorithms to avoid quadratic time complexity when possible

### Requirement 26: Seven-Day AKI Detection

**User Story:** As a researcher, I want to detect AKI using the 7-day baseline comparison with proper temporal bounds, so that subacute kidney injury is captured without unintended future leakage.

#### Acceptance Criteria

1. WHEN evaluating the 7-day criterion, THE AKI_Labeler SHALL consider only creatinine measurements with charttime at least 24 hours after ICU intime
2. THE AKI_Labeler SHALL compute the evaluation window end as the minimum of ICU outtime and intime plus 7 days to prevent leakage beyond ICU discharge
3. FOR ALL creatinine measurements within the evaluation window, THE AKI_Labeler SHALL compute the ratio of the measurement to baseline creatinine
4. IF any creatinine ratio is greater than or equal to 1.5, THEN THE AKI_Labeler SHALL mark the 7-day criterion as satisfied
5. THE AKI_Labeler SHALL document that the evaluation window is bounded by ICU discharge to prevent using post-ICU data for stays shorter than 7 days

### Requirement 27: Data Type Validation

**User Story:** As a researcher, I want to validate data types during loading, so that type-related errors are caught early.

#### Acceptance Criteria

1. WHEN loading CSV files, THE system SHALL verify that timestamp columns can be parsed as datetime objects
2. WHEN loading CSV files, THE system SHALL verify that numeric columns contain valid numeric values or missing indicators
3. IF a data type validation fails, THEN THE system SHALL log the specific column, row, and invalid value
4. THE system SHALL convert timestamp columns to datetime objects with timezone awareness if timezone information is present
5. THE system SHALL log data type conversion statistics including the number of successful conversions and failures

### Requirement 28: Feature Scaling and Normalization

**User Story:** As a researcher, I want to scale numeric features appropriately, so that models converge efficiently and features contribute proportionally.

#### Acceptance Criteria

1. WHEN training models that require feature scaling, THE system SHALL compute scaling parameters using only the Training_Set
2. THE system SHALL apply standardization (zero mean, unit variance) to continuous numeric features
3. THE system SHALL apply the same scaling transformation computed from the Training_Set to validation and test sets
4. THE system SHALL save scaling parameters to the models directory for reproducibility
5. THE system SHALL log the mean and standard deviation for each scaled feature


### Requirement 29: Model Hyperparameter Documentation

**User Story:** As a researcher, I want all hyperparameters documented, so that experiments are transparent and reproducible.

#### Acceptance Criteria

1. FOR ALL trained models, THE system SHALL save a JSON file containing all hyperparameters to the models directory
2. THE hyperparameter file SHALL include model type, random seed, learning rate (if applicable), regularization parameters, and architecture specifications
3. THE system SHALL include hyperparameter values in the master log file
4. WHEN using default hyperparameters, THE system SHALL explicitly document that defaults were used

### Requirement 30: Clinical Note Availability Handling

**User Story:** As a researcher, I want to handle cases where discharge summaries are missing, so that the pipeline can process incomplete data gracefully.

#### Acceptance Criteria

1. WHEN an ICU stay has no corresponding discharge summary, THE Text_Model SHALL create a zero vector embedding of the appropriate dimension
2. THE Text_Model SHALL log the number of ICU stays with missing discharge summaries
3. THE Text_Model SHALL compute and report the percentage of ICU stays with available text data
4. WHEN training text-only or multimodal models, THE system SHALL include ICU stays with missing text data using zero vector representations

### Requirement 31: Train-Validation-Test Consistency Verification

**User Story:** As a researcher, I want to verify data split integrity, so that evaluation is valid and no data contamination occurs.

#### Acceptance Criteria

1. AFTER data splitting, THE Data_Splitter SHALL verify that no patient identifier appears in multiple splits
2. THE Data_Splitter SHALL compute and log the AKI prevalence in each split
3. IF AKI prevalence differs by more than 10 percentage points between splits, THEN THE Data_Splitter SHALL log a warning about potential imbalance
4. THE Data_Splitter SHALL save patient identifiers for each split to separate files in the processed_data directory
5. THE Data_Splitter SHALL verify that the sum of samples across all splits equals the total number of ICU stays

### Requirement 32: Memory-Efficient Processing

**User Story:** As a researcher, I want memory-efficient data processing, so that the pipeline can handle large datasets without crashing.

#### Acceptance Criteria

1. WHEN processing laboratory events, THE system SHALL use chunked reading if the file size exceeds 1 GB
2. THE system SHALL release memory for intermediate data structures after they are no longer needed
3. WHEN generating embeddings, THE system SHALL process text in batches rather than loading all text into memory simultaneously
4. THE system SHALL log peak memory usage for each major processing step
5. IF available memory falls below 10% of total system memory, THEN THE system SHALL log a warning

### Requirement 33: Error Recovery and Checkpointing

**User Story:** As a researcher, I want to resume processing after failures, so that long-running computations do not need to restart from scratch.

#### Acceptance Criteria

1. WHEN processing embeddings, THE Text_Model SHALL save intermediate results every 1000 samples
2. WHEN a script is restarted, THE system SHALL check for existing checkpoint files and resume from the last checkpoint
3. THE system SHALL save checkpoint metadata including timestamp and number of processed samples
4. WHEN all processing is complete, THE system SHALL remove temporary checkpoint files
5. THE system SHALL log checkpoint creation and restoration events

### Requirement 34: Statistical Summary Generation

**User Story:** As a researcher, I want automated statistical summaries, so that I can quickly understand dataset characteristics.

#### Acceptance Criteria

1. THE Cohort_Analyzer SHALL generate a summary table showing mean, median, standard deviation, minimum, and maximum for all continuous features
2. THE Cohort_Analyzer SHALL generate a frequency table for all categorical features
3. THE Cohort_Analyzer SHALL compute and report the correlation matrix for numeric features
4. THE Cohort_Analyzer SHALL save all statistical summaries to CSV files in the results directory
5. THE Cohort_Analyzer SHALL generate a correlation heatmap and save it to the figures directory

### Requirement 35: Model Comparison Visualization

**User Story:** As a researcher, I want comparative visualizations of model performance, so that I can easily identify the best performing approach.

#### Acceptance Criteria

1. THE evaluation component SHALL generate a bar chart comparing AUROC across all trained models
2. THE evaluation component SHALL generate a bar chart comparing AUPRC across all trained models
3. THE evaluation component SHALL generate a grouped bar chart showing precision, recall, and F1 score for all models
4. THE evaluation component SHALL overlay ROC curves for all models on a single plot for direct comparison
5. THE evaluation component SHALL save all comparison visualizations to the figures directory

