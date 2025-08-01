"""
This file learns the mapping relationship from simulation profiles.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
#%%
def shannon_entropy(xdata,binnumber):
    # Calculate frequency and intervals using the numpy.histogram function
    counts, bin_edges = np.histogram(xdata, bins=binnumber, density=True)
    # Replace cases with a frequency of 0 with a small non-zero value
    counts[counts == 0] = 1e-10
    # Calculate the probability
    probabilities = counts / np.sum(counts)
    # Calculate shannon entropy
    shannon_entropy = -np.sum(probabilities * np.log2(probabilities))
    return shannon_entropy

def movmean(data, window_size):
    smoothed_data = []
    if window_size % 2 == 0:
        for i in range(len(data)):
            start = max(0, i - 1- (window_size-2) // 2 )
            end = min(len(data), i + (window_size - 2) // 2)
            window = data[start:end+1]
            smoothed_data.append(np.mean(window))
    else:
        for i in range(len(data)):
            start = max(0, i - (window_size-1) // 2 )
            end = min(len(data), i + (window_size - 1) // 2)
            window = data[start:end+1]
            smoothed_data.append(np.mean(window))              
    return np.array(smoothed_data)

def calculate_slope(data):
    x = np.arange(len(data))
    return np.polyfit(x, data, 1)[0]


def extract_voltage_features(voltage_series):
    # Extract various statistical features from a pandas series of V data
    features = {

        'media_V': voltage_series.median(),
        'std_V': voltage_series.std(),
        'skew_V': voltage_series.skew(),
        'shanEntro_V': shannon_entropy(voltage_series,len(voltage_series)),
        'slope_V':calculate_slope(voltage_series)
        
    }
    return features

def extract_capacity_features(capacity_series):
    # Extract various statistical features from a pandas series of Q data
    features = {

        'std_Q': capacity_series.std(),
        'skew_Q': capacity_series.skew(),
        'kurt_Q': capacity_series.kurt(),
        
    }
    return features

def extract_DQ_features(DQ_series):
    # Extract various statistical features from a pandas series of D1 data
    features = {

        'media_DQ': DQ_series.median(),
        'std_DQ': DQ_series.std(),
        'skew_DQ': DQ_series.skew(),
        'kurt_DQ': DQ_series.kurt(),
        'shanEntro_DQ': shannon_entropy(DQ_series,len(DQ_series)),
        'slope_DQ':calculate_slope(DQ_series)
    }
    return features
#%%
def process_and_analyze_battery_data(file_path, variable_name, voltage_start_threshold, voltage_end_threshold):
    # read data
    df = pd.read_csv(file_path, delim_whitespace=True, skiprows=9, index_col=False, names=[variable_name, 'Time', 'SOC', 'Voltage'])

    # data group
    grouped = df.groupby(variable_name)
    charging_segments = {}
    Q0 = 2

    for epsspos, group in grouped:
        group['Q'] = group['SOC'] * Q0
        start_idx = group[group['Voltage'] >= voltage_start_threshold].index.min()
        end_idx = group[group['Voltage'] >= voltage_end_threshold].index.min()
        if pd.isna(start_idx) or pd.isna(end_idx):
            continue
        charging_df = group.loc[start_idx:end_idx]
        
        # Calculate the length of the charging process
        seq_CC = end_idx - start_idx + 1
        charging_df['seq_CC'] = seq_CC * 10  # Add charging length to the DataFrame
        
        # Optionally, calculate the total sequence length reaching 4.2V if needed
        seq_CV = group[group['Voltage'] >= voltage_end_threshold].shape[0]
        charging_df['seq_CC'] = seq_CV * 10
        
        charging_segments[epsspos] = charging_df

    # features extraction
    all_features = []
    for epsspos, segment in charging_segments.items():
        V_selected = segment['Voltage']
        voltage_features = extract_voltage_features(V_selected)
        voltage_features[variable_name] = epsspos

        Q_selected = segment['Q']
        Q_features = extract_capacity_features(Q_selected)
        
        segment = segment.copy()
        segment.loc[:, 'DQ'] = segment['Q'] - segment['Q'].iloc[0]
        DQ_features = extract_DQ_features(segment['DQ'])
        
        # Extract the first values of seqCC and seqCV
        seq_cc = segment['seq_CC'].iloc[0]  # Take the first value

        combined_features = {
            **voltage_features, 
            **Q_features, 
            **DQ_features,  
            'seq_CC': seq_cc

            }
        all_features.append(combined_features)

    features_df = pd.DataFrame(all_features)
    features_df = features_df[[variable_name] + [col for col in features_df.columns if col != variable_name]]

    # model train and evaluation
    features_df['log_'+variable_name] = np.log(features_df[variable_name])
    y = features_df['log_'+variable_name]
    X = features_df.drop([variable_name, 'log_'+variable_name], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_features': [1.0, 'sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error', verbose=2)
    grid_search.fit(X_train_scaled, y_train)
    best_rf = grid_search.best_estimator_
    print("Best parameters:", grid_search.best_params_)

    y_pred = best_rf.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)  

    rf = RandomForestRegressor(n_estimators=150, max_features=1.0, max_depth=None, min_samples_split=2, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)

    # feature importance analysis
    feature_importances = best_rf.feature_importances_
    important_features = feature_importances > np.mean(feature_importances)
    important_feature_names = X.columns[important_features]

    # only use important features
    X_train_important = X_train_scaled[:, important_features]
    X_test_important = X_test_scaled[:, important_features]

    rf_important = RandomForestRegressor(
    n_estimators=grid_search.best_params_['n_estimators'], 
    max_features=grid_search.best_params_['max_features'],
    max_depth=grid_search.best_params_['max_depth'],
    min_samples_split=grid_search.best_params_['min_samples_split'],
    random_state=42
    )
    
    rf_important.fit(X_train_important, y_train)
    y_pred_important = rf_important.predict(X_test_important)
    mse_important = mean_squared_error(y_test, y_pred_important)    

    colormap = cm.get_cmap('coolwarm')

    # plot feature importances
    plt.figure(figsize=(5, 3))
    feature_names = X.columns
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # normalize the importance range
    normalize = plt.Normalize(vmin=np.min(importances), vmax=np.max(importances))
    colors = colormap(normalize(importances[indices]))  
    
    plt.bar(range(X.shape[1]), importances[indices], color=colors, align='center')
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Features', fontsize = 16)
    plt.ylabel('Importance', fontsize = 16)   
    sm = cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=np.min(importances), vmax=np.max(importances)))
    sm.set_array([])   
    plt.show()

    # return important features and model
    return {
        'features_df': features_df,
        'mse': mse,# use all features to train RF
        'mse_importance': mse_important,# use important features to train RF
        'important_feature_names': important_feature_names.tolist(),
        'rf_model': best_rf,
        'rf_important_model': rf_important
    }

#%%
if  __name__ == '__main__':
    result_epss = process_and_analyze_battery_data('CCCV generation #1.txt', 'epsspos', 2.95, 4.2)

