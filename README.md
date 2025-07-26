# Scripts for Uncovering the Impact of Battery Design Parameters on Health and Lifetime Using Short Charging Segments
This repository contains scripts for extracting features from partial charges and simulation files using partial voltage intervals. These features are then
used to model battery state of health (SOH) and remaining useful life (RUL). It is the official implementation of the paper:  "Uncovering the Impact of Battery Design Parameters on Health and Lifetime Using Short Charging Segments". The entire analysis pipeline is implemented in Python and includes: `featureextraction` for data preprocessing, `trainmodels` and `main`for modelling SOH and RUL using several machine learning methods: Random Forest(RF), Support Vector Regression (SVR), XGBoost, Multi-layer Perceptron (MLP), and Linear Regression (LR).

# Data
The original analysis was performed using ten datasets designed to emulate electric vehicle (EV) operation under World Harmonized Light Vehicles Test Cycle (WLTC) driving profiles. Both the raw and processed data can be found on Zenodo: https://doi.org/10.5281/zenodo.15626215. Only one multistep fast-charging dataset is publicly available.

New data can be added to the analysis, as long as it follows the same format as the processed dataset. Specifically, it must adhere to the required folder structure, file naming conventions, and file formats.
<img width="316" height="164" alt="image" src="https://github.com/user-attachments/assets/2de6e6b6-445f-4063-8e2f-004f8196c301" />
<img width="315" height="175" alt="image" src="https://github.com/user-attachments/assets/110880b6-892b-4ce4-9c30-288ef27ac345" />
<img width="343" height="186" alt="image" src="https://github.com/user-attachments/assets/afff6b4e-432e-43f1-a39d-89f3b1d2626a" />

# Scripts
The repository contains six scripts:

-   `featureextraction`: A script for extracting 29 physically interpretable features from processed structural data. It relies on the `features_extraction` functionand requires the processed files EFC.mat and EOCV.mat.
-   `functions`: A script containing utility functions for data processing and model preparation. It includes `split_battery_data_with_train_test_dynamic` for splitting the dataset into training and test sets, `process_features_with_physics` for dynamically determining whether to embed physical features, `predict_and_plot` for predicting physical features using partial charges and generating plots, and `process_train_val_test_data_optimized` for embedding aging-related columns during dataset preparation.
-   `learnphysics`:  A script for deriving the mapping relationship between simulation profiles and partial charges based on extracted features. This mapping is established using three categories of features: voltage, capacity, and capacity deviation. The process is implemented through the function `process_and_analyze_battery_data`.
-   `main`: A script used to extract features, map simulation features, organize all battery datasets, train models, and evaluate their performance with or without physical features derived from digital twin simulation files.
-   `trainmodels`: A script used to train models using the training dataset. It includes `train_and_evaluate_rf_mmd` for training a Random Forest (RF) model with MMD loss, `train_and_evaluate_rf` for standard RF training, and `evaluate_voltage_range` for training RF models using randomly selected partial charge segments.
-   `visualization`: A script used to visualize the performance of trained models with or without physical features derived from digital twin simulation files, and to determine feature importance using SHAP analysis.

# License
This project is licensed under the MIT License.

# Acknowledgements
This work is supported by Nordic Energy Research (Vehicle battery storage for green transport and grid stability in the Nordics), the Swedish Electromobility Center and the Swedish Energy Agency project EV drivers in the driver’s seat.

