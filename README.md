# Scripts for Uncovering the Impact of Battery Design Parameters on Health and Lifetime Using Short Charging Segments
This repository contains the scripts needed for extracting features from partial charges and simulation files by using partial voltage intervals, and use these features for modelling battery state of health and remaining useful lifetime. fficial implementation of the paper:  "Uncovering the Impact of Battery Design Parameters on Health and Lifetime Using Short Charging Segments" This entire analysis pipeline was written and performed in python, using `featureextraction` for data preprocessing, `trainmodels` and `main`for modelling SOH and RUL by random forest(RF), support vector regression (SVR), XGBoost, multi-layer perception (MLP), and linear regression (LR).

# Data
The original analysis was performed using ten datasets, the designed to emulate the operation of EVs using the WLTC driving profiles. The raw data and processed data can be found at Zenodo at https://doi.org/10.5281/zenodo.15626215.

New data can also be added to the analysis, if it follows the same format as the processed dataset. That is, needs to have the following folder, file, and naming structure.

<img width="218" height="507" alt="image" src="https://github.com/user-attachments/assets/30c45921-a88e-41a2-a386-c8fdffbbdbe5" />
<img width="206" height="80" alt="image" src="https://github.com/user-attachments/assets/8c5b7150-3181-45a0-b7ee-85b6f4bebecf" />
<img width="222" height="96" alt="image" src="https://github.com/user-attachments/assets/24852b53-21a6-4700-a1dc-f83ec76db18d" />




