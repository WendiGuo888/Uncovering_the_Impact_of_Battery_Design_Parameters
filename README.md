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

-   `




