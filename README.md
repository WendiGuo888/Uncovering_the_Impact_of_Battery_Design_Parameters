# Scripts for Uncovering the Impact of Battery Design Parameters on Health and Lifetime Using Short Charging Segments
This repository contains the scripts needed for extracting features from partial charges and simulation files by using partial voltage intervals, and use these features for modelling battery state of health and remaining useful lifetime. fficial implementation of the paper:  "Uncovering the Impact of Battery Design Parameters on Health and Lifetime Using Short Charging Segments" This entire analysis pipeline was written and performed in python, using `featureextraction` for data preprocessing, `trainmodels` and `main`for modelling SOH and RUL by random forest(RF), support vector regression (SVR), XGBoost, multi-layer perception (MLP), and linear regression (LR).

# Data
The original analysis was performed using ten datasets, the designed to emulate the operation of EVs using the WLTC driving profiles. The raw data and processed data can be found at Zenodo at https://doi.org/10.5281/zenodo.15626215.

New data can also be added to the analysis, if it follows the same format as the processed dataset. That is, needs to have the following folder, file, and naming structure.

<img width="316" height="164" alt="image" src="https://github.com/user-attachments/assets/2de6e6b6-445f-4063-8e2f-004f8196c301" />
<img width="315" height="175" alt="image" src="https://github.com/user-attachments/assets/110880b6-892b-4ce4-9c30-288ef27ac345" />
<img width="343" height="186" alt="image" src="https://github.com/user-attachments/assets/afff6b4e-432e-43f1-a39d-89f3b1d2626a" />

# Scripts
The repository contains six scripts:

-   `featureextraction`: A script for extracting 29 physically interpretable features from processed structure data; the script depends on the function `features_extraction`. Furthermore, the extraction depends on the processed **EFC.mat**, **EOCV.mat**.
-   `functions`: A script for splitting train and test datasets in function `split_battery_data_with_train_test_dynamic`, dynamically decides whether to embed physical features in function `process_features_with_physics`, predict physical features using partial charges in function `predict_and_plot`, setting embed aging columns in function `process_train_val_test_data_optimized`.
-   `learnphysics`: A script for get the mapping relationship between the simulation profiles and partial charges based on extracted features. It get mapping relationship depends on three categories features: voltage, capacity, and capacity deviation. Using the function `process_and_analyze_battery_data`.
-   `main`: A script used to extract features, map simulation features, organize all battery datasets, and train models, evaluate the performance with or without physical features from digital twin simulation files.
-   `trainmodels`: A script used to train models using the training dataset. It includes training RF model with MMD loss in function `train_and_evaluate_rf_mmd`, training RF model in function `train_and_evaluate_rf`, and train RF models using random partial charges in function `evaluate_voltage_range`.
-   `visualization`: A script used to visualize the trained model performance with our without physical features from digital twin simulation files, and determine feature importance using SHAP analysis.

# License
This project is licensed under the MIT License.

# Acknowledgements
This work is supported by Nordic Energy Research (Vehicle battery storage for green transport and grid stability in the Nordics), the Swedish Electromobility Center and the Swedish Energy Agency project EV drivers in the driver’s seat.

