"""
Main file for model training and performance evaluation.
"""

# %% Packages
import pickle
import time
import pandas as pd
import numpy as np
import warnings
import shap

from featureextraction import features_extraction
from learndesign import process_and_analyze_battery_data

from functions import split_battery_data_with_train_test_dynamic
from functions import process_train_val_test_data_optimized
from functions import process_features_with_physics
from functions import remove_low_outliers
from functions import remove_high_outliers_iqr
from functions import compute_mean_std_after_outlier_removal_by_setting

from trainmodels import train_and_evaluate_rf_mmd
from trainmodels import train_and_evaluate_rf
from trainmodels import evaluate_voltage_range

from visualization import plot_grouped_violin
from visualization import plot_with_error_bars_and_trend_with_split
from visualization import plot_heatmap
from visualization import plot_shap_beeswarm
from visualization import prepare_radar_data
from visualization import plot_radar_custom
from visualization import plot_voltage_mae_heatmap

from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Ignore warning
warnings.filterwarnings("ignore")  

# %% Extract features
cell01 = features_extraction('1C25degC/mat/data.mat', '1C25degC/mat/EOCV.mat', '1C25degC/mat/EFC.mat', '1C25degC/mat/RPT_EFC.mat', '1C25degC/mat/SOH.mat', V_min=3.65, V_max=4.1, I_min=100, I_max=500) 
cell02 = features_extraction('1.3C25degC/mat/data.mat', '1.3C25degC/mat/EOCV.mat', '1.3C25degC/mat/EFC.mat', '1.3C25degC/mat/RPT_EFC.mat', '1.3C25degC/mat/SOH.mat', V_min=3.65, V_max=4.1, I_min=100, I_max=500) 
cell03 = features_extraction('2C25degCs2/mat/data.mat', '2C25degCs2/mat/EOCV.mat', '2C25degCs2/mat/EFC.mat', '2C25degCs2/mat/RPT_EFC.mat', '2C25degCs2/mat/SOH.mat', V_min=3.65, V_max=4.1, I_min=100, I_max=500) 
cell04 = features_extraction('MC25degCS1/mat/data.mat', 'MC25degCS1/mat/EOCV.mat', 'MC25degCS1/mat/EFC.mat', 'MC25degCS1/mat/RPT_EFC.mat', 'MC25degCS1/mat/SOH.mat', V_min=3.65, V_max=4.1, I_min=100, I_max=500) 
cell05 = features_extraction('2C35degC/mat/data.mat', '2C35degC/mat/EOCV.mat', '2C35degC/mat/EFC.mat', '2C35degC/mat/RPT_EFC.mat', '2C35degC/mat/SOH.mat', V_min=3.65, V_max=4.1, I_min=100, I_max=500) 
cell06 = features_extraction('1.3C15degCs1/mat/data.mat', '1.3C15degCs1/mat/EOCV.mat', '1.3C15degCs1/mat/EFC.mat', '1.3C15degCs1/mat/RPT_EFC.mat', '1.3C15degCs1/mat/SOH.mat', V_min=3.65, V_max=4.1, I_min=100, I_max=500) 
cell07 = features_extraction('1.3C15degCs2/mat/data.mat', '1.3C15degCs2/mat/EOCV.mat', '1.3C15degCs2/mat/EFC.mat', '1.3C15degCs2/mat/RPT_EFC.mat', '1.3C15degCs2/mat/SOH.mat', V_min=3.65, V_max=4.1, I_min=100, I_max=500) 
cell08 = features_extraction('2C0degCs1/mat/data.mat', '2C0degCs1/mat/EOCV.mat', '2C0degCs1/mat/EFC.mat', '2C0degCs1/mat/RPT_EFC.mat', '2C0degCs1/mat/SOH.mat', V_min=3.65, V_max=4.1, I_min=100, I_max=500) 
cell09 = features_extraction('2C0degCs2/mat/data.mat', '2C0degCs2/mat/EOCV.mat', '2C0degCs2/mat/EFC.mat', '2C0degCs2/mat/RPT_EFC.mat', '2C0degCs2/mat/SOH.mat', V_min=3.65, V_max=4.1, I_min=100, I_max=500)
cell10 = features_extraction('MC25degCs2/mat/data.mat', 'MC25degCs2/mat/EOCV.mat', 'MC25degCs2/mat/EFC.mat', 'MC25degCs2/mat/RPT_EFC.mat', 'MC25degCs2/mat/SOH.mat', V_min=3.65, V_max=4.1, I_min=100, I_max=500)

#%% mapping simulation features
result_epss = process_and_analyze_battery_data('digital_twin_data/CCCV generationv1.txt', 'epsspos', 2.95, 4.2)
result_rpneg = process_and_analyze_battery_data('digital_twin_data/CCCV generationv2.txt', 'rpneg', 2.95, 4.2)
result_Lneg = process_and_analyze_battery_data('digital_twin_data/CCCV generationv3.txt', 'Lneg', 2.95, 4.2)
result_cspos = process_and_analyze_battery_data('digital_twin_data/CCCV generationv4.txt', 'cspos', 2.95, 4.2)
result_Lpos = process_and_analyze_battery_data('digital_twin_data/CCCV generationv5.txt', 'Lpos', 2.95, 4.2)
result_Dneg = process_and_analyze_battery_data('digital_twin_data/CCCV generationv6.txt', 'Dneg', 2.95, 4.2)

#%% aging condition mapping
# aging conditions
current_mapping = {

    'cell01': [1.0, 1.0, 1.0, 1.0],
    'cell02': [1.3, 1.3, 1.3, 1.3],
    'cell03': [2.0, 2.0, 2.0, 2.0],
    'cell04': [2.0, 2.0, 1.5, 1.0],
    'cell05': [2.0, 2.0, 2.0, 2.0],
    'cell06': [1.3, 1.3, 1.3, 1.3],
    'cell07': [1.3, 1.3, 1.3, 1.3],
    'cell08': [2.0, 2.0, 2.0, 2.0],
    'cell09': [2.0, 2.0, 2.0, 2.0],
    'cell10': [2.0, 1.5, 1.5, 1.0],
}

temperature_mapping = {

    'cell01': 25, 
    'cell02': 25, 
    'cell03': 25, 
    'cell04': 25, 
    'cell05': 35,
    'cell06': 15, 
    'cell07': 15, 
    'cell08': 0, 
    'cell09': 0, 
    'cell10': 25, 
}   

#%% organize all battery datasets

feature_dataframes = [cell01, cell02, cell03, cell04, cell05, cell06, cell07, cell08, cell09, cell10] 

#%% read parameters 

with open("robust_params_with_phys.pkl", "rb") as f:
    robust_params_with_phys = pickle.load(f)
    
with open("robust_params.pkl", "rb") as f:
    robust_params = pickle.load(f)
    
with open("all_best_params.pkl", "rb") as f:
    loaded_params = pickle.load(f)
    
best_mmd_params = loaded_params["best_mmd_params"]
best_rf_params_multitask = loaded_params["best_rf_params_multitask"]
best_rf_params_multitask_no_aging = loaded_params["best_rf_params_multitask_no_aging"]

#%% test different training ratio with physics or without physics (use Monte Carlo splitting methods)
train_ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

num_seeds = 100
rng = np.random.default_rng(seed=42)  
random_seeds = rng.integers(0, 10000, num_seeds)  

# use all physical features
initial_physical_features = ["rpneg_predicted", "Lneg_predicted", 'Dneg_predicted', "cspos_predicted", "epsspos_predicted", 'Lpos_predicted'] 

#embed physical parmaeters to train and test datasets, must be run before process train and test data without validation
result_dictionaries = [result_epss, result_rpneg, result_Lneg, result_cspos, result_Lpos, result_Dneg]
variable_names = ['epsspos', 'rpneg', 'Lneg', 'cspos', 'Lpos', 'Dneg']

physical_feature_conditions = [
    {"label": "With Physical Features", "features": initial_physical_features + ["EFC"]},
    {"label": "Without Physical Features", "features": ["EFC"]} 
]

feature_names = ['EFC', 'seq_CC', 'seq_CV', 'EOCV', 'IC_peak', 'IC_area', 'V_peak', 'Q_DV',  
                'slope_DQ', 'slope_I', 'min_DQ',
                'media_I', 'kurt_I', 'skew_I', 'std_I','shanEntro_I', 
                'media_V', 'skew_V', 'std_V','shanEntro_V', 'slope_V'
                'kurt_Q', 'skew_Q', 'std_Q', 
                'media_DQ','kurt_DQ', 'skew_DQ', 'std_DQ','shanEntro_DQ',
                'epsspos_predicted', 'rpneg_predicted', 'Lneg_predicted', 'cspos_predicted', 'Lpos_predicted', 'Dneg_predicted'] 

target_col_indices = slice(-13, -11)  # Select the target columns: 12th to 11th from the end

results_comparison_mc_mmd = []

# iterate over training ratios and physical feature conditions
for train_ratio in train_ratios:
    for condition in physical_feature_conditions:
        label = condition["label"]
        current_physical_features = condition["features"]
        print(f"\nTrain Ratio: {train_ratio}, Condition: {label}, Features: {current_physical_features}")

        for seed in random_seeds:
            # Randomly split the dataset into training and testing sets
            train_dataframes, test_dataframes = split_battery_data_with_train_test_dynamic(
                feature_dataframes=feature_dataframes,
                current_mapping=current_mapping,
                temperature_mapping=temperature_mapping,
                train_ratio=train_ratio,  
                random_state=seed
            )

            # process training battery data: feature selection, normalization, and labeling
            processed_train_features = process_features_with_physics(
                feature_dataframes=train_dataframes, 
                result_dictionaries=result_dictionaries, 
                variable_names=variable_names,  
                include_physical_features=True
            )

            # process testing battery data: apply the same preprocessing as the training data
            processed_test_features = process_features_with_physics(
                feature_dataframes=test_dataframes, 
                result_dictionaries=result_dictionaries, 
                variable_names=variable_names, 
                include_physical_features=True
            )

            # Call feature selection and data processing functions
            train_selected_data, test_selected_data, _, _= process_train_val_test_data_optimized(
                train_dataframes=processed_train_features,
                test_dataframes=processed_test_features,
                target_col_indices=target_col_indices,
                feature_names=feature_names,
                cor_limit=16,
                important_physical_features=current_physical_features,
                random_state=42,
                aging_columns=None
            )

            X_train = pd.concat([data['X'] for data in train_selected_data['train'].values()]).values
            y_train = pd.concat([data['y'] for data in train_selected_data['train'].values()]).values
            X_test = pd.concat([data['X'] for data in test_selected_data['test'].values()]).values
            y_test = pd.concat([data['y'] for data in test_selected_data['test'].values()]).values

            # train RF model
            _, rf_results_mmd, _, _, _, _ = train_and_evaluate_rf_mmd(
                train_selected_data=train_selected_data,
                test_selected_data=test_selected_data,
                n_estimators=best_rf_params_multitask_no_aging["n_estimators"],
                max_depth=int(best_rf_params_multitask_no_aging["max_depth"]),
                min_samples_split=best_rf_params_multitask_no_aging["min_samples_split"],
                min_samples_leaf=best_rf_params_multitask_no_aging["min_samples_leaf"],
                max_features=best_rf_params_multitask_no_aging["max_features"],
                bootstrap=best_rf_params_multitask_no_aging["bootstrap"],
                hyperparameter_search=False,
                weights=(1, 0.001),
                lambda_mmd_soh=best_mmd_params["lambda_mmd_soh"],
                lambda_mmd_rul=best_mmd_params["lambda_mmd_rul"]
                
            )

            # save results
            results_comparison_mc_mmd.append({
                "train_ratio": train_ratio,
                "random_seed": seed,
                "physical_features": label,
                "SOH_R2": rf_results_mmd["Test R²"]["SOH"],
                "RUL_R2": rf_results_mmd["Test R²"]["RUL"],
                "SOH_MAE": rf_results_mmd["Test MAE"]["SOH"],
                "RUL_MAE": rf_results_mmd["Test MAE"]["RUL"]
            })
            
            print(f"Train Ratio: {train_ratio}, Condition: {label}, R² & MAE collected.")

results_comparison_mmd_df = pd.DataFrame(results_comparison_mc_mmd)

results_comparison_mmd_df_cleaned = results_comparison_mmd_df.groupby("train_ratio", group_keys=False).apply(remove_low_outliers)

train_ratio_mapping = {0.3: "30%", 0.4: "40%", 0.5: "50%", 0.6: "60%", 0.7: "70%", 0.8: "80%"}

results_comparison_melted = results_comparison_mmd_df_cleaned.melt(
    id_vars=["train_ratio", "random_seed", "physical_features"], 
    value_vars=["SOH_R2", "RUL_R2", "SOH_MAE", "RUL_MAE"], 
    var_name="metric", 
    value_name="value"
)

metric_mapping = {
    "SOH_R2": "SOH R²",
    "RUL_R2": "RUL R²",
    "SOH_MAE": "SOH MAE",
    "RUL_MAE": "RUL MAE"
}

results_comparison_melted["metric"] = results_comparison_melted["metric"].map(metric_mapping)

results_comparison_melted["train_ratio"] = results_comparison_melted["train_ratio"].map(train_ratio_mapping)

# plot violin
for metric in results_comparison_melted["metric"].unique():
    metric_data = results_comparison_melted[results_comparison_melted["metric"] == metric]
    
    if metric_data.empty:
        print(f"⚠️ Skip {metric}, data is empty.")
        continue

    ylabel = metric
    plot_grouped_violin(data=metric_data, ylabel=ylabel)
    
#%% cycle-wise with or without physics 
# The early prediction capability with or without physical features
random_seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]

physical_feature_conditions = [
    {"label": "With Physical Features", "features": initial_physical_features + ["EFC"]},
    {"label": "Without Physical Features", "features": ["EFC"]} 
]

aging_columns = ['current_1', 'current_2', 'current_3', 'current_4','temperature']

test_row_indices = [20, 40, 60, 80, 100, 120, 140, 160, 180]

detailed_results_cycle = {
    "cycle": [], 
    "physical_features": [], 
    "metric": [],
    "run_id": [], 
    "error_value": []
}

for condition in physical_feature_conditions:
    label = condition["label"]
    current_physical_features = condition["features"]
    print(f"\nCondition: {label}, Features: {current_physical_features}")

    for run_id, seed in enumerate(random_seeds):
        train_dataframes, test_dataframes = split_battery_data_with_train_test_dynamic(
            feature_dataframes=feature_dataframes,
            current_mapping=current_mapping,
            temperature_mapping=temperature_mapping,
            train_ratio=0.8,
            random_state=seed
        )
        
        processed_train_features = process_features_with_physics(
            feature_dataframes=train_dataframes, 
            result_dictionaries=result_dictionaries, 
            variable_names=variable_names, 
            include_physical_features=True
        )

        processed_test_features = process_features_with_physics(
            feature_dataframes=test_dataframes, 
            result_dictionaries=result_dictionaries, 
            variable_names=variable_names, 
            include_physical_features=True
        )

        train_selected_data, test_selected_data, _, _ = process_train_val_test_data_optimized(
            train_dataframes=train_dataframes,
            test_dataframes=test_dataframes,
            target_col_indices=target_col_indices,
            feature_names=feature_names,
            cor_limit=16,
            important_physical_features=current_physical_features,
            random_state=42,
            aging_columns=None
        )
        
        X_train_full = pd.concat([data['X'] for data in train_selected_data['train'].values()]).values
        y_train_full = pd.concat([data['y'] for data in train_selected_data['train'].values()]).values
        X_test_full = pd.concat([data['X'] for data in test_selected_data['test'].values()]).values
        y_test_full = pd.concat([data['y'] for data in test_selected_data['test'].values()]).values
   
        for cycle in test_row_indices:
            if cycle >= len(X_train_full):  
                continue
        
            X_test = X_test_full[:cycle]
            y_test = y_test_full[:cycle]
            
            # train RF model
            _, _, _, y_test_pred, _, _ = train_and_evaluate_rf(
                train_selected_data={"train": {"feature1": {"X": pd.DataFrame(X_train_full), "y": pd.DataFrame(y_train_full)}}},
                test_selected_data={"test": {"feature1": {"X": pd.DataFrame(X_test), "y": pd.DataFrame(y_test)}}},
                n_estimators=best_rf_params_multitask["n_estimators"],
                max_depth=int(best_rf_params_multitask["max_depth"]),
                min_samples_split=best_rf_params_multitask["min_samples_split"],
                min_samples_leaf=best_rf_params_multitask["min_samples_leaf"],
                max_features=best_rf_params_multitask["max_features"],
                bootstrap=best_rf_params_multitask["bootstrap"],
                hyperparameter_search=False,
                weights=(1, 0.001),
            )

            soh_mae = mean_absolute_error(y_test[:, 0], y_test_pred[:, 0])
            rul_mae = mean_absolute_error(y_test[:, 1], y_test_pred[:, 1])
            
            soh_mape = mean_absolute_percentage_error(y_test[:, 0], y_test_pred[:, 0])
            rul_mape = mean_absolute_percentage_error(y_test[:, 1], y_test_pred[:, 1])
 
            for metric, value in zip(["SOH MAE", "RUL MAE", "SOH MAPE", "RUL MAPE"], [soh_mae, rul_mae, soh_mape, rul_mape]):
                detailed_results_cycle["cycle"].append(cycle)
                detailed_results_cycle["physical_features"].append(label)
                detailed_results_cycle["metric"].append(metric)
                detailed_results_cycle["run_id"].append(run_id)
                detailed_results_cycle["error_value"].append(value)
                
detailed_cycle_df = pd.DataFrame(detailed_results_cycle)

filtered_cycle_df = detailed_cycle_df.groupby(["cycle", "physical_features", "metric"], group_keys=False).apply(remove_high_outliers_iqr)

summary_cycle_df = filtered_cycle_df.groupby(
    ["cycle", "physical_features", "metric"]
).agg(
    error_mean=("error_value", "mean"),
    error_std=("error_value", "std")
).reset_index()
      
# plot results
plot_with_error_bars_and_trend_with_split(
    summary_cycle_df, "SOH MAE", "SOH MAE", "SOH MAE",
    threshold=0.03, stable_window=2, max_fluctuation=0.1
)

plot_with_error_bars_and_trend_with_split(
    summary_cycle_df, "RUL MAE", "RUL MAE", "RUL MAE",
    threshold=3.0, stable_window=2, max_fluctuation=5
)

#%% repeat 10 seed num for shap analysis of SOH and RUL
shap_soh_all = []
shap_rul_all = []
feature_all = []

best_rf_params_cleaned = {
    k: (int(v) if k == "max_depth" and isinstance(v, float) else v) 
    for k, v in best_rf_params_multitask.items() 
    if k in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "bootstrap"]
}

for seed in random_seeds:

    train_dataframes, test_dataframes = split_battery_data_with_train_test_dynamic(
        feature_dataframes=feature_dataframes,
        current_mapping=current_mapping,
        temperature_mapping=temperature_mapping,
        train_ratio=0.8,
        random_state=seed
    )

    processed_train = process_features_with_physics(
        train_dataframes, result_dictionaries, variable_names, include_physical_features=True
    )
    processed_test = process_features_with_physics(
        test_dataframes, result_dictionaries, variable_names, include_physical_features=True
    )

    train_selected_data, test_selected_data, global_selected_features, _ = process_train_val_test_data_optimized(
        train_dataframes=processed_train,
        test_dataframes=processed_test,
        target_col_indices=target_col_indices,
        feature_names=feature_names,
        cor_limit=16,
        important_physical_features=initial_physical_features + ["EFC"],
        random_state=42,
        aging_columns=None
    )

    X_train_df = pd.concat([d['X'] for d in train_selected_data['train'].values()])
    y_train_df = pd.concat([d['y'] for d in train_selected_data['train'].values()])
    X_test_df  = pd.concat([d['X'] for d in test_selected_data['test'].values()])
    y_test_df  = pd.concat([d['y'] for d in test_selected_data['test'].values()])

    X_train = X_train_df.values
    y_train = y_train_df.values
    X_test = X_test_df.values
    y_test = y_test_df.values
    feature_names_used = X_train_df.columns.tolist()

    X_test_soh = X_test[:60]
    X_test_rul = X_test[:80]
    y_train_weighted = y_train * (1.0, 0.001)

    # train RF model
    rf = MultiOutputRegressor(RandomForestRegressor(random_state=42, **best_rf_params_cleaned))
    rf.fit(X_train, y_train_weighted)
    rf_soh = rf.estimators_[0]
    rf_rul = rf.estimators_[1]

    # ---- SHAP ----
    explainer_soh = shap.TreeExplainer(rf_soh)
    explainer_rul = shap.TreeExplainer(rf_rul)

    X_soh_all = np.concatenate([X_train, X_test_soh], axis=0)
    X_rul_all = np.concatenate([X_train, X_test_rul], axis=0)

    shap_soh = explainer_soh.shap_values(X_soh_all)
    shap_rul = explainer_rul.shap_values(X_rul_all)

    # ---- Record feature names and absolute mean SHAP values ----
    shap_soh_all.append(np.mean(np.abs(shap_soh), axis=0))
    shap_rul_all.append(np.mean(np.abs(shap_rul), axis=0))
    feature_all.append(feature_names_used)
    
# plot heatmap for SHAP value
feature_name_mapping = {
    "Dneg_predicted": "Diffusion coef of NE",
    "Lneg_predicted": "Thickness of NE",
    "Lpos_predicted": "Thickness of PE",
    "cspos_predicted": "Lithium Concentration of PE",
    "epsspos_predicted": "Porosity of PE",
    "rpneg_predicted": "Particle Radius of NE",
    "EFC": "Equivalent Full Cycles",
    "ECOV": "EOCV",
    "media_I": "Median of Current",
    "skew_I": "Skewness of Current",
    "kurt_I": "Kurtosis of Current",
    "std_I": "Standard Deviation of Current",
    "seq_CV": "Charging Time",
    "shanEntro_I": "ShanEn of Current",
    "slope_I": "Slope of Current",
    "media_DQ": "Median of DQ",
    "min_DQ": "Minimum of DQ",
    "std_Q": "Standard Deviation of Q",
    "std_DQ": "Standard Deviation of DQ",
    "IC_area": "IC area",
    "skew_V": "Skewness of Voltage",    
}

df_soh_list, df_rul_list = [], []
for i in range(len(random_seeds)):
    df_soh_list.append(pd.DataFrame({'Feature': feature_all[i], f'Seed_{i}': shap_soh_all[i]}))
    df_rul_list.append(pd.DataFrame({'Feature': feature_all[i], f'Seed_{i}': shap_rul_all[i]}))

df_soh_merged = df_soh_list[0]
df_rul_merged = df_rul_list[0]

for i in range(1, len(random_seeds)):
    df_soh_merged = df_soh_merged.merge(df_soh_list[i], on='Feature', how='outer')
    df_rul_merged = df_rul_merged.merge(df_rul_list[i], on='Feature', how='outer')

df_soh_merged = df_soh_merged.set_index('Feature')
df_rul_merged = df_rul_merged.set_index('Feature')

soh_df_renamed = df_soh_merged.rename(index=feature_name_mapping)
rul_df_renamed = df_rul_merged.rename(index=feature_name_mapping)

plot_heatmap(soh_df_renamed, title="SOH SHAP Mean Absolute Value")
plot_heatmap(rul_df_renamed, title="RUL SHAP Mean Absolute Value")

#%% replicate shap results 
train_dataframes, test_dataframes = split_battery_data_with_train_test_dynamic(
    feature_dataframes=feature_dataframes,
    current_mapping=current_mapping,
    temperature_mapping=temperature_mapping,
    train_ratio=0.8,
    random_state=909
)

processed_train_features = process_features_with_physics(
    feature_dataframes=train_dataframes, 
    result_dictionaries=result_dictionaries, 
    variable_names=variable_names, 
    include_physical_features=True
)

processed_test_features = process_features_with_physics(
    feature_dataframes=test_dataframes, 
    result_dictionaries=result_dictionaries, 
    variable_names=variable_names, 
    include_physical_features=True
)

train_selected_data, test_selected_data, global_selected_features, _ = process_train_val_test_data_optimized(
    train_dataframes=processed_train_features,
    test_dataframes=processed_test_features,
    target_col_indices=target_col_indices,
    feature_names=feature_names,
    cor_limit=16,
    important_physical_features=initial_physical_features,  
    random_state=42,
    aging_columns=None
)

X_train = pd.concat([data['X'] for data in train_selected_data['train'].values()]).values
y_train = pd.concat([data['y'] for data in train_selected_data['train'].values()]).values
X_test = pd.concat([data['X'] for data in test_selected_data['test'].values()]).values
y_test = pd.concat([data['y'] for data in test_selected_data['test'].values()]).values

feature_names = X_train_df.columns.tolist()

y_train_weighted = y_train * (1.0, 0.001)

rf_params = robust_params_with_phys["Random Forest"]
rf_multi = MultiOutputRegressor(RandomForestRegressor(random_state=42, **rf_params))
rf_multi.fit(X_train, y_train_weighted)

rf_soh = rf_multi.estimators_[0]
rf_rul = rf_multi.estimators_[1]

X_all = np.concatenate([X_train, X_test], axis=0)
X_df_all = pd.DataFrame(X_all, columns=feature_names)

explainer_soh = shap.TreeExplainer(rf_soh)
explainer_rul = shap.TreeExplainer(rf_rul)

shap_values_soh_all = explainer_soh.shap_values(X_all)
shap_values_rul_all = explainer_rul.shap_values(X_all)

feature_name_mapping_rf = {
    "Dneg_predicted": "Diffusion coef of NE",
    "Lneg_predicted": "Thickness of NE",
    "Lpos_predicted": "Thickness of PE",
    "cspos_predicted": "Lithium Concentration of PE",
    "epsspos_predicted": "Porosity of PE",
    "rpneg_predicted": "Particle Radius of NE",
    "EFC": "Equivalent Full Cycles",
    "ECOV": "EOCV",
    "media_I": "Median of Current",
    "skew_I": "Skewness of Current",
    "kurt_I": "Kurtosis of Current",
    "std_I": "Standard Deviation of Current",
    "seq_CV": "Charging Time",
    "slope_I": "Slope of Current",
    "min_DQ": "Minimum of DQ",
    "std_DQ": "Standard Deviation of DQ"
}

X_df_full = X_df_all.rename(columns=feature_name_mapping_rf)
        
plot_shap_beeswarm(
    shap_values=shap_values_rul_all,
    X_df=X_df_full,
    title="RUL SHAP Impact",
    figsize=(7, 4),     
    max_display=16,        
    label_fontsize=11,
    value_fontsize=10,
    title_fontsize=12,
)

plot_shap_beeswarm(
    shap_values=shap_values_soh_all,
    X_df=X_df_full,
    title="SOH SHAP Impact",
    figsize=(7, 4),   
    max_display=16,      
    label_fontsize=11,
    value_fontsize=10,
    title_fontsize=12,
)

#%% three cases to compare the physical features function 
model_order = ["Linear Regression", "SVR", "XGBoost", "MLP", "Random Forest"]
model_mapping = {
    "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "SVR": SVR(),
    "XGBoost": XGBRegressor(random_state=42, n_jobs=-1),
    "MLP": MLPRegressor(max_iter=2000, random_state=42),
    "Linear Regression": LinearRegression()
}

evaluation_configs = [
    {
        "param_source": "robust_params",  
        "feature_setting": "Without Physical Features",  
        "label": "NoPhysParam_NoPhysFeat"
    },
    {
        "param_source": "robust_params",
        "feature_setting": "With Physical Features",
        "label": "NoPhysParam_WithPhysFeat"
    },
    {
        "param_source": "robust_params_with_phys", 
        "feature_setting": "With Physical Features",
        "label": "WithPhysParam_WithPhysFeat"
    }
]

results_detailed_with_phys = {"model": [], "setting": [], "metric": [], "seed": [], "value": []}
results_summary_with_phys = {"model": [], "setting": [], "metric": [], "mean": [], "std": []}

random_seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
for config in evaluation_configs:
    param_dict = globals()[config["param_source"]]  
    physical_features = (
        initial_physical_features + ["EFC"]
        if config["feature_setting"] == "With Physical Features"
        else ["EFC"]
    )
    setting_label = config["label"]

    print(f"\n📌 Evaluating setting: {setting_label}")

    for model_name in model_order:
        print(f"\n🔍 Model: {model_name}")
        test_mae_soh_list, test_mae_rul_list, train_time_list = [], [], []

        for seed in random_seeds:

            train_dataframes, test_dataframes = split_battery_data_with_train_test_dynamic(
                feature_dataframes=feature_dataframes,
                current_mapping=current_mapping,
                temperature_mapping=temperature_mapping,
                train_ratio=0.8,
                random_state=seed
            )

            processed_train_features = process_features_with_physics(
                feature_dataframes=train_dataframes,
                result_dictionaries=result_dictionaries,
                variable_names=variable_names,
                include_physical_features=True
            )

            processed_test_features = process_features_with_physics(
                feature_dataframes=test_dataframes,
                result_dictionaries=result_dictionaries,
                variable_names=variable_names,
                include_physical_features=True
            )

            train_selected_data, test_selected_data, _, _ = process_train_val_test_data_optimized(
                train_dataframes=processed_train_features,
                test_dataframes=processed_test_features,
                target_col_indices=target_col_indices,
                feature_names=feature_names,
                cor_limit=16,
                important_physical_features=physical_features,
                random_state=42,
                aging_columns=None
            )

            X_train = pd.concat([v["X"] for v in train_selected_data["train"].values()]).values
            y_train = pd.concat([v["y"] for v in train_selected_data["train"].values()]).values
            X_test = pd.concat([v["X"] for v in test_selected_data["test"].values()]).values
            y_test = pd.concat([v["y"] for v in test_selected_data["test"].values()]).values

            best_params = param_dict.get(model_name, {})

            if isinstance(best_params, dict) and "SOH" in best_params and "RUL" in best_params:
                base_model = model_mapping[model_name]
                soh_model = base_model.__class__(**best_params["SOH"])
                rul_model = base_model.__class__(**best_params["RUL"])

                start_time = time.time()
                soh_model.fit(X_train, y_train[:, 0])
                rul_model.fit(X_train, y_train[:, 1])
                train_time = time.time() - start_time

                y_pred = np.column_stack([
                    soh_model.predict(X_test),
                    rul_model.predict(X_test)
                ])
            else:
                base_model = model_mapping[model_name]
                base_model.set_params(**best_params)

                start_time = time.time()
                base_model.fit(X_train, y_train * (1, 0.001))
                train_time = time.time() - start_time
                y_pred = base_model.predict(X_test) / (1, 0.001)

            test_mae = {
                "SOH": mean_absolute_error(y_test[:, 0], y_pred[:, 0]),
                "RUL": mean_absolute_error(y_test[:, 1], y_pred[:, 1])
            }

            for metric, value in zip(["Test SOH MAE", "Test RUL MAE", "Train Time"],
                                     [test_mae["SOH"], test_mae["RUL"], train_time]):
                results_detailed_with_phys["model"].append(model_name)
                results_detailed_with_phys["setting"].append(setting_label)
                results_detailed_with_phys["metric"].append(metric)
                results_detailed_with_phys["seed"].append(seed)
                results_detailed_with_phys["value"].append(value)

            test_mae_soh_list.append(test_mae["SOH"])
            test_mae_rul_list.append(test_mae["RUL"])
            train_time_list.append(train_time)

        for metric, values in zip(["Test SOH MAE", "Test RUL MAE", "Train Time"],
                                  [test_mae_soh_list, test_mae_rul_list, train_time_list]):
            results_summary_with_phys["model"].append(model_name)
            results_summary_with_phys["setting"].append(setting_label)
            results_summary_with_phys["metric"].append(metric)
            results_summary_with_phys["mean"].append(np.mean(values))
            results_summary_with_phys["std"].append(np.std(values))

results_detailed_df_with_phys = pd.DataFrame(results_detailed_with_phys)
results_summary_df_with_phys = pd.DataFrame(results_summary_with_phys)

# plot radar figures
results_summary_trimmed_df = compute_mean_std_after_outlier_removal_by_setting(results_detailed_df_with_phys, group_col="setting", drop_extreme=True)

soh_df_trimmed = results_summary_trimmed_df[results_summary_trimmed_df["metric"] == "Test SOH MAE"]
rul_df_trimmed = results_summary_trimmed_df[results_summary_trimmed_df["metric"] == "Test RUL MAE"]
time_df_trimmed = results_summary_trimmed_df[results_summary_trimmed_df["metric"] == "Train Time"]

soh_radar_data = prepare_radar_data(soh_df_trimmed, metric_name="Test SOH MAE")
rul_radar_data = prepare_radar_data(rul_df_trimmed, metric_name="Test RUL MAE")

legend_labels = ["NoPhysParam_NoP*", "NoPhysParam_WithP*", "WithPhysParam_WithP*"]

# RUL MAE radar plot
plot_radar_custom(
    data_df=rul_radar_data,
    legend_labels=legend_labels,
    title_text="RUL MAE",
)

# SOH MAE radar plot
plot_radar_custom(
    data_df=soh_radar_data,
    legend_labels=legend_labels,
    title_text="SOH MAE",
)

#%% random voltage range
# initialize list to store results
V_MAXIMUM = np.arange(3.85, 4.1 + 0.05, 0.05)
V_MINIMUM = np.arange(3.5, 3.8 + 0.05, 0.05)
# ----------------------------
# create a mapping between file names and feature variables
# ----------------------------
file_feature_mapping = {
    '1C25degC': 'cell01',
    '1.3C25degC': 'cell02',
    '2C25degCs2': 'cell03',
    'MC25degCS1': 'cell04',
    '2C35degC': 'cell05',
    '1.3C15degCs1': 'cell06',
    '1.3C15degCs2': 'cell07',
    '2C0degCs1':'cell08',
    '2C0degCs2':'cell09',
    'MC25degCs2':'cell10',

}

# without physical parameters
error_dict_no_phys = evaluate_voltage_range(
    rf_params=robust_params["Random Forest"],
    V_MAXIMUM=V_MAXIMUM,
    V_MINIMUM=V_MINIMUM,
    file_feature_mapping=file_feature_mapping,
    current_mapping=current_mapping,
    temperature_mapping=temperature_mapping,
    feature_names=feature_names,
    result_dictionaries=result_dictionaries,
    variable_names=variable_names,
    include_physical_features=False
)

# with physical parameters
error_dict = evaluate_voltage_range(
    rf_params=robust_params_with_phys["Random Forest"],
    V_MAXIMUM=V_MAXIMUM,
    V_MINIMUM=V_MINIMUM,
    file_feature_mapping=file_feature_mapping,
    current_mapping=current_mapping,
    temperature_mapping=temperature_mapping,
    feature_names=feature_names,
    result_dictionaries=result_dictionaries,
    variable_names=variable_names,
    include_physical_features=True
)

# plot MAE error
plot_voltage_mae_heatmap(
    error_dict=error_dict_no_phys,
    metric_key="RUL_MAE",
    title="RUL MAE Without P*"
)

plot_voltage_mae_heatmap(
    error_dict=error_dict_no_phys,
    metric_key="SOH_MAE",
    title="SOH MAE Without P*"
)

plot_voltage_mae_heatmap(
    error_dict=error_dict,
    metric_key="RUL_MAE",
    title="RUL MAE With P*"
)

plot_voltage_mae_heatmap(
    error_dict=error_dict,
    metric_key="SOH_MAE",
    title="SOH MAE With P*"
)
