"""
These are functions to deal with features
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#%% random split data as train and test

def split_battery_data_with_train_test_dynamic(
    feature_dataframes, current_mapping, temperature_mapping, train_ratio=0.7, random_state=None
):
    """
    Randomly split battery dataset while incorporating aging conditions (current, temperature)

    parameters:
    - feature_dataframes: List of DataFrames containing battery feature data
    - current_mapping: Dictionary of current conditions for each battery
    - temperature_mapping: Dictionary of temperature conditions for each battery
    - train_ratio
    - random_state: Random seed to ensure reproducible splits

    return:
    - train_dataframes
    - test_dataframes
    """
    # random seed
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(feature_dataframes))

    # Generate indices for training and test set split
    train_split_idx = int(len(feature_dataframes) * train_ratio)
    train_indices = shuffled_indices[:train_split_idx]
    test_indices = shuffled_indices[train_split_idx:]

    # Initialize training and test sets based on split indices
    train_dataframes = []
    test_dataframes = []

    # Integrate aging conditions (current, temperature) into each battery’s data
    for i, df in enumerate(feature_dataframes, start=1):
        feature_key = f"feature{i}"
        current_condition = current_mapping.get(feature_key, [0, 0, 0, 0])  
        temperature_condition = temperature_mapping.get(feature_key, 25) 
  
        current_df = pd.DataFrame([current_condition] * len(df), 
                                  columns=[f'current_{j+1}' for j in range(len(current_condition))])
        temperature_df = pd.DataFrame({'temperature': [temperature_condition] * len(df)})
        conditioned_df = pd.concat([df.reset_index(drop=True), current_df, temperature_df], axis=1)

        # Split training and test sets based on indices
        if i - 1 in train_indices:
            train_dataframes.append(conditioned_df)
        elif i - 1 in test_indices:
            test_dataframes.append(conditioned_df)
    return train_dataframes, test_dataframes

#%% Dynamically decide whether to embed physical features
def process_features_with_physics(
    feature_dataframes, 
    result_dictionaries, 
    variable_names,  
    include_physical_features=True
):
    """
    Process experimental data and physical features
    
    parameters:
    - feature_dataframes: list of DataFrames containing experimental features for each battery
    - result_dictionaries: list of dictionaries containing results from the physical feature prediction model
    - variable_names: list of physical feature variable names

    - include_physical_features: whether to embed physical features (Boolean value)
    
    return:
    - processed_features: Each experimental feature DataFrame (with or without embedded physical features)
    """
    processed_features = []

    for idx, feature_df in enumerate(feature_dataframes):
        all_predictions = []

        # Predict each physical feature using the trained model
        for i, (result, var_name) in enumerate(zip(result_dictionaries, variable_names)):
            predictions = predict_and_plot(
                result, 
                feature_df, 
                var_name, 
                color_idx=i,  
                save_path=None
            )
            if predictions is not None:
                all_predictions.append(predictions)

        # Dynamically decide whether to embed physical features
        if include_physical_features:
            if all(len(pred) == len(feature_df) for pred in all_predictions):

                for pred, name in zip(all_predictions, variable_names):
                    feature_df[name + '_predicted'] = pred
                print(f"Physical features added to dataset {idx+1}.")
            else:
                print(f"Error: Prediction lengths do not match for dataset {idx+1}. Skipping physical feature embedding.")
        else:
            print(f"Physical features skipped for dataset {idx+1}.")

        # save
        processed_features.append(feature_df)
    return processed_features

#%% predict physical features

def predict_and_plot(result_dict, feature_df, variable_name, color_idx, save_path=None):
    # obtain feature_df 
    features_in_model = result_dict['features_df'].columns.drop([variable_name, 'log_'+variable_name])
    
    # Select features in feature_df that overlap with features_in_model
    matching_features = [feature for feature in features_in_model if feature in feature_df.columns]
    
    # if there are columns in feature that match the model features
    if matching_features:
    
        X_feature = feature_df[matching_features]
    
        scaler = StandardScaler()
        X_feature_scaled = scaler.fit_transform(X_feature)  
    
        # predict with rf_model  
        log_predictions = result_dict['rf_model'].predict(X_feature_scaled)
        
        # Convert logarithmic predictions back to original scale
        predictions = np.exp(log_predictions)
        
        return predictions
    else:
        print("No matching features found between the trained model and feature1 DataFrame.")   
        
        
#%% optimize aging columns setting of process data definition

feature_names = ['EFC', 'seq_CC', 'seq_CV', 'EOCV', 'IC_peak', 'IC_area', 'V_peak', 'Q_DV',  
                'slope_DQ', 'slope_I', 'min_DQ',
                'media_I', 'kurt_I', 'skew_I', 'std_I','shanEntro_I', 
                'media_V', 'skew_V', 'std_V','shanEntro_V', 'slope_V'
                'kurt_Q', 'skew_Q', 'std_Q', 
                'media_DQ','kurt_DQ', 'skew_DQ', 'std_DQ','shanEntro_DQ',
                'epsspos_predicted', 'rpneg_predicted', 'Lneg_predicted', 'cspos_predicted', 'Lpos_predicted', 'Dneg_predicted'] 

target_col_indices = slice(-13, -11)  

aging_columns = ['current_1', 'current_2', 'current_3', 'current_4','temperature']

# Select the 12th and 11th columns from the end as target variables
current_physical_features = ["EFC"]

def process_train_val_test_data_optimized(
    train_dataframes,
    test_dataframes,
    target_col_indices,
    feature_names,
    cor_limit=16,
    important_physical_features=None,
    random_state=42,
    aging_columns=None
):
    """
    Optimize logic: if only EFC is specified, perform Spearman correlation-based selection, excluding physical features
    """

    all_physical_features = ["rpneg_predicted", "Lneg_predicted", "cspos_predicted", 
                             "epsspos_predicted", "Lpos_predicted", "Dneg_predicted", "EFC"]

    all_train_features = []
    all_train_targets = []
    for df in train_dataframes:
        y = df.iloc[:, target_col_indices]
        X = df.drop(df.columns[target_col_indices], axis=1)
        # Check whether aging_columns is a non-empty list
        if aging_columns is not None and len(aging_columns) > 0:
            X = X.drop(columns=aging_columns, errors="ignore")
        all_train_features.append(X)
        all_train_targets.append(y)

    # perform global feature selection based on feature importance or correlation
    all_train_features_combined = pd.concat(all_train_features, axis=0)
    all_train_targets_combined = pd.concat(all_train_targets, axis=0)
    global_q_loss = all_train_targets_combined.iloc[:, 0]
    global_rul = all_train_targets_combined.iloc[:, 1]

    # Spearman correlation
    cor_soh = all_train_features_combined.corrwith(global_q_loss, method="spearman")
    cor_rul = all_train_features_combined.corrwith(global_rul, method="spearman")
    combined_cor = cor_soh.abs() + cor_rul.abs()

    # Initialize selected features
    global_selected_features = []

    # the logic for physical feature selection
    if not important_physical_features or important_physical_features == []:
        # If no physical features are specified, keep only EFC
        if "EFC" in combined_cor.index:
            combined_cor.loc["EFC"] = combined_cor.max() + 1 
            global_selected_features.append("EFC")
        combined_cor = combined_cor.drop([feature for feature in all_physical_features if feature != "EFC"], errors="ignore")
    else:
        # When physical features are specified, keep the selected physical features and EFC, and exclude the rest
        for feature in all_physical_features:
            if feature in important_physical_features and feature in combined_cor.index:
                combined_cor.loc[feature] = combined_cor.max() + 1  
                global_selected_features.append(feature)
    
        if "EFC" not in global_selected_features and "EFC" in combined_cor.index:
            combined_cor.loc["EFC"] = combined_cor.max() + 1  
            global_selected_features.append("EFC")
            
        # Exclude unspecified physical features
        combined_cor = combined_cor.drop([feature for feature in all_physical_features if feature not in global_selected_features], errors="ignore")

        # Select additional features based on correlation with the target (e.g., EFC)
        additional_features = combined_cor.drop(global_selected_features, errors="ignore").nlargest(
            cor_limit - len(global_selected_features)
        ).index.tolist()
        global_selected_features += additional_features

    print(f"Globally selected features (optimized): {global_selected_features}")

    global_scaler = MinMaxScaler()

    # Helper function for processing dataset
    def process_dataframe_list(dataframes, dataset_name):
        processed_data = {}
        for i, df in enumerate(dataframes, start=1):
            y = df.iloc[:, target_col_indices]
            X = df.drop(df.columns[target_col_indices], axis=1)
            if aging_columns is not None and len(aging_columns) > 0:
                aging_data = X[aging_columns].copy()
                X = X.drop(columns=aging_columns, errors="ignore")
            else:
                aging_data = pd.DataFrame()

            X_filtered = X[global_selected_features]

            if dataset_name == "train" and i == 1:
                global_scaler.fit(X_filtered)
            X_norm = global_scaler.transform(X_filtered)

            if not aging_data.empty:
                X_final = pd.concat(
                    [pd.DataFrame(X_norm, columns=global_selected_features),
                     aging_data.reset_index(drop=True)],
                    axis=1
                )
            else:
                X_final = pd.DataFrame(X_norm, columns=global_selected_features)

            processed_data[f"feature{i}"] = {"X": X_final, "y": y.reset_index(drop=True)}
        return processed_data

    train_selected_data = {"train": process_dataframe_list(train_dataframes, "train")}
    test_selected_data = {"test": process_dataframe_list(test_dataframes, "test")}
    return train_selected_data, test_selected_data, global_selected_features, global_scaler

def remove_low_outliers(group):
    Q1_soh = group["SOH_R2"].quantile(0.25)
    Q3_soh = group["SOH_R2"].quantile(0.75)
    IQR_soh = Q3_soh - Q1_soh
    lower_bound_soh = Q1_soh - 0 * IQR_soh

    Q1_rul = group["RUL_R2"].quantile(0.25)
    Q3_rul = group["RUL_R2"].quantile(0.75)
    IQR_rul = Q3_rul - Q1_rul
    lower_bound_rul = Q1_rul - 0 * IQR_rul  
    return group[(group["SOH_R2"] >= lower_bound_soh) & (group["RUL_R2"] >= lower_bound_rul)]

def remove_high_outliers_iqr(group):
    Q1 = group["error_value"].quantile(0.25)
    Q3 = group["error_value"].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 0 * IQR
    return group[group["error_value"] <= upper_bound]

def compute_mean_std_after_outlier_removal_by_setting(detailed_df, group_col="setting", drop_extreme=True):
    """
    For each model + setting + metric group, drop max/min and compute mean/std
    
    """
    summary_data = {
        "model": [], group_col: [], "metric": [],
        "mean": [], "std": []
    }

    grouped = detailed_df.groupby(["model", group_col, "metric"])

    for (model, setting_val, metric), group in grouped:
        values = group["value"].copy().values

        if drop_extreme and len(values) > 2:
            values = sorted(values)
            values = values[1:-1]

        summary_data["model"].append(model)
        summary_data[group_col].append(setting_val)
        summary_data["metric"].append(metric)
        summary_data["mean"].append(np.mean(values))
        summary_data["std"].append(np.std(values))

    summary_df = pd.DataFrame(summary_data)
    return summary_df
