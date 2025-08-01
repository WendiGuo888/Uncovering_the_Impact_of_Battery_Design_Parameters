"""
These are training models. 
"""

import os
import time
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor
from scipy.spatial.distance import cdist

from featureextraction import features_extraction
from functions import split_battery_data_with_train_test_dynamic
from functions import process_features_with_physics
from functions import process_train_val_test_data_optimized
from functions import prepare_battery_dataset
#%% define multitask RF with MMD loss
# MMD Loss
def mmd_rbf_per_sample(X, Y, gamma=1.0):
    """MMD distance (sample-wise)"""
    XX = np.exp(-gamma * cdist(X, X, 'sqeuclidean'))
    YY = np.exp(-gamma * cdist(Y, Y, 'sqeuclidean'))
    XY = np.exp(-gamma * cdist(X, Y, 'sqeuclidean'))
    mmd_per_sample = XX.mean(axis=1) - 2 * XY.mean(axis=1).mean() + YY.mean()
    return mmd_per_sample

# train RF model
def train_random_forest_multitask_optimized_mmd(
    X_train, y_train, X_test=None, y_test=None,
    n_estimators=100, max_depth=None,
    min_samples_split=2, min_samples_leaf=1,
    max_features="sqrt", bootstrap=True,
    hyperparameter_search=False, weights=(1, 0.001),
    lambda_mmd_soh=0.1, lambda_mmd_rul=0.1  # MMD weight
):
    """
    Train a multi-task Random Forest model with support for MMD constraint, and compute the MMD impact based on the test set
    """

    # target weight
    soh_weight, rul_weight = weights
    y_train_weighted = y_train.copy()
    y_train_weighted[:, 0] *= soh_weight 
    y_train_weighted[:, 1] *= rul_weight 
    
    # MMD impact
    if X_test is not None and y_test is not None:
        mmd_values = mmd_rbf_per_sample(X_train, X_test)  # Compute MMD between training features and X_test
        mmd_values = mmd_values.reshape(-1, 1) 
        
        y_train_weighted[:, 0] += lambda_mmd_soh * mmd_values[:, 0]  
        y_train_weighted[:, 1] += lambda_mmd_rul * mmd_values[:, 0] 

        print(f"📊 MMD Loss (based on Test Set): {mmd_values.mean():.6f}")

    # train RF model
    if hyperparameter_search:
        print("🔍 Starting hyperparameter tuning...")

        param_grid = {
            'n_estimators': [50, 100, 200, 300, 400, 500, 600],
            'max_depth': [5, 10, 15, 20, 30, 50, None],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 5, 10, 15],
            'max_features': ['sqrt', 'log2', 0.5, 0.8],
            'bootstrap': [True, False]
        }

        rf = RandomForestRegressor(random_state=42)

        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_grid,
            n_iter=50, 
            scoring='neg_mean_squared_error',
            cv=5,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )

        random_search.fit(X_train, y_train_weighted)
        rf = random_search.best_estimator_
        print(f"✅ Best hyperparameters: {random_search.best_params_}")

    else:
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=42
        )
        rf.fit(X_train, y_train_weighted)
    return rf

# train and evaluate
def train_and_evaluate_rf_mmd(
    train_selected_data, test_selected_data,
    n_estimators=100, max_depth=None,
    min_samples_split = 20, min_samples_leaf = 2,
    max_features = 'log2', bootstrap = True,
    hyperparameter_search=False, weights=(1.0, 0.001),
    lambda_mmd_soh=0.1, lambda_mmd_rul=0.1 
):
    """
    Train Random Forest with MMD regularization
    """

    # training set
    X_train = pd.concat([data['X'] for data in train_selected_data['train'].values()])
    y_train = pd.concat([data['y'] for data in train_selected_data['train'].values()])

    # testing set
    X_test = pd.concat([data['X'] for data in test_selected_data['test'].values()])
    y_test = pd.concat([data['y'] for data in test_selected_data['test'].values()])

    # convert DataFrame to NumPy array
    X_train, y_train = X_train.values, y_train.values
    X_test, y_test = X_test.values, y_test.values

    # train RF model
    rf_model = train_random_forest_multitask_optimized_mmd(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        n_estimators=n_estimators, max_depth=max_depth,
        hyperparameter_search=False,
        weights=weights, 
        lambda_mmd_soh=lambda_mmd_soh,
        lambda_mmd_rul=lambda_mmd_rul,
    )

    # training set prediction
    y_train_pred_weighted = rf_model.predict(X_train)
    y_train_pred = y_train_pred_weighted.copy()
    y_train_pred[:, 0] /= weights[0]
    y_train_pred[:, 1] /= weights[1]

    # testing set prediction
    y_test_pred_weighted = rf_model.predict(X_test)
    y_test_pred = y_test_pred_weighted.copy()
    y_test_pred[:, 0] /= weights[0]
    y_test_pred[:, 1] /= weights[1]

    # test error
    test_mse = {
        "SOH": mean_squared_error(y_test[:, 0], y_test_pred[:, 0]),
        "RUL": mean_squared_error(y_test[:, 1], y_test_pred[:, 1])
    }
    test_rmse = {key: np.sqrt(value) for key, value in test_mse.items()}
    test_mae = {
        "SOH": mean_absolute_error(y_test[:, 0], y_test_pred[:, 0]),
        "RUL": mean_absolute_error(y_test[:, 1], y_test_pred[:, 1])
    }
    test_r2 = {
        "SOH": r2_score(y_test[:, 0], y_test_pred[:, 0]),
        "RUL": r2_score(y_test[:, 1], y_test_pred[:, 1])
    }
    
    results = {
        "Test MSE": test_mse,
        "Test RMSE": test_rmse,
        "Test MAE": test_mae,
        "Test R²": test_r2
    }
    return rf_model, results, y_train_pred, y_test_pred, y_train, y_test
#%% define multitask RF 
def train_random_forest_multitask_optimized(
    X_train, y_train, 
    n_estimators=100, max_depth=None, 
    min_samples_split=2, min_samples_leaf=1,
    max_features="sqrt", bootstrap=True,
    hyperparameter_search=False, weights=(1.0, 0.001)
):
    """
    Optimized Random Forest training function with support for weighting and hyperparameter tuning.
    """
    # target weight
    soh_weight, rul_weight = weights
    y_train_weighted = y_train.copy()
    y_train_weighted[:, 0] *= soh_weight 
    y_train_weighted[:, 1] *= rul_weight  

    # shuffle training set
    X_train, y_train_weighted = shuffle(X_train, y_train_weighted, random_state=42)

    # model training
    if hyperparameter_search:
        print("Starting hyperparameter search...")
        param_grid = {
            'n_estimators': [50, 100, 150, 200, 300, 400, 500, 600],
            'max_depth': [5, 10, 15, 20, 30, 50, None],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 5, 10, 15],
            'max_features': ['sqrt', 'log2', 0.5, 0.8]
        }
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=5,  
            verbose=2,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train_weighted)
        rf = grid_search.best_estimator_
        print(f"Best Parameters: {grid_search.best_params_}")
    else:
        rf = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        rf.fit(X_train, y_train_weighted)
    return rf

def train_and_evaluate_rf(
    train_selected_data, test_selected_data,
    n_estimators=100, max_depth=None,
    min_samples_split=2, min_samples_leaf=1,
    max_features='sqrt', bootstrap=True,
    hyperparameter_search=False, weights=(1.0, 0.001)
):
    """
    Train a multi-task Random Forest model on the training data and evaluate its performance on the test data, with support for target weighting.
    
    parameters:
    - train_selected_data: Training data dictionary
    - test_selected_data: Testing data dictionary
    - n_estimators: Number of decision trees in the Random Forest 
    - max_depth: maximum depth of the decision trees
    - hyperparameter_search: whether to perform hyperparameter search
    - weights: (SOH weight, RUL weight)

    return:
    - rf_model: Trained Random Forest model
    - results: Test error dictionary
    - y_train_pred: Predicted values for the training set
    - y_test_pred: Test set predictions
    - y_train, y_test: ground truth
    """
    # training set
    X_train = pd.concat([data['X'] for data in train_selected_data['train'].values()])
    y_train = pd.concat([data['y'] for data in train_selected_data['train'].values()])

    # testing set
    X_test = pd.concat([data['X'] for data in test_selected_data['test'].values()])
    y_test = pd.concat([data['y'] for data in test_selected_data['test'].values()])

    # convert DataFrame to NumPy array
    X_train, y_train = X_train.values, y_train.values
    X_test, y_test = X_test.values, y_test.values

    # train RF model
    rf_model = train_random_forest_multitask_optimized(
        X_train=X_train,
        y_train=y_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
        hyperparameter_search=hyperparameter_search,
        weights=weights
    )

    # store training set prediction results
    y_train_pred_weighted = rf_model.predict(X_train)
    y_train_pred = y_train_pred_weighted.copy()
    y_train_pred[:, 0] /= weights[0]
    y_train_pred[:, 1] /= weights[1]

    # store testing set prediction results
    y_test_pred_weighted = rf_model.predict(X_test)
    y_test_pred = y_test_pred_weighted.copy()
    y_test_pred[:, 0] /= weights[0]
    y_test_pred[:, 1] /= weights[1]

    # test set error
    test_mse = {
        "SOH": mean_squared_error(y_test[:, 0], y_test_pred[:, 0]),
        "RUL": mean_squared_error(y_test[:, 1], y_test_pred[:, 1])
    }
    test_rmse = {key: np.sqrt(value) for key, value in test_mse.items()}
    test_mae = {
        "SOH": mean_absolute_error(y_test[:, 0], y_test_pred[:, 0]),
        "RUL": mean_absolute_error(y_test[:, 1], y_test_pred[:, 1])
    }
    
    test_r2 = {
    "SOH": r2_score(y_test[:, 0], y_test_pred[:, 0]),
    "RUL": r2_score(y_test[:, 1], y_test_pred[:, 1])
}

    # save results
    results = {
        "Test MSE": test_mse,
        "Test RMSE": test_rmse,
        "Test MAE": test_mae,
        "Test R²": test_r2
    }
    return rf_model, results, y_train_pred, y_test_pred, y_train, y_test
#%% evaluate random charging segments performance
def evaluate_voltage_range(    
    rf_params,
    V_MAXIMUM, V_MINIMUM,
    file_feature_mapping,
    current_mapping,
    temperature_mapping,
    result_dictionaries,
    variable_names,
    include_physical_features=True):
    """
    iterate over different voltage ranges to extract features, train the model, and record errors
    
    parameters：
    - rf_params: define Random Forest hyperparameter dictionary
    - include_physical_features: Check whether to include physical features
    
    return：
    - error_dict: {(v_max, v_min): {...metric...}, ...}
    """
    error_dict = {}

    for i, v_max in enumerate(V_MAXIMUM):
        for j, v_min in enumerate(V_MINIMUM):
            if v_min >= v_max:
                continue

            print(f"\nProcessing Voltage Range: {v_min}V - {v_max}V, Physical = {include_physical_features}")

            feature_dataframes = []
            for file_name, feature_name in file_feature_mapping.items():
                data_path = f"{file_name}/data.mat"
                if not os.path.exists(data_path):
                    continue
                features = features_extraction(
                    f"{file_name}/data.mat",
                    f"{file_name}/EOCV.mat",
                    f"{file_name}/EFC.mat",
                    f"{file_name}/RPT_EFC.mat",
                    f"{file_name}/SOH.mat",
                    V_min=v_min, V_max=v_max, I_min=100, I_max=500
                )
                if features is not None and not features.empty:
                    feature_dataframes.append(features)

            if not feature_dataframes:
                continue

            # split data into training and testing sets
            train_dataframes, test_dataframes = split_battery_data_with_train_test_dynamic(
                feature_dataframes=feature_dataframes,
                current_mapping=current_mapping,
                temperature_mapping=temperature_mapping,
                train_ratio=0.8,
                random_state=42
            )
            if not train_dataframes or not test_dataframes:
                continue

            # Include physical features
            processed_train = process_features_with_physics(
                feature_dataframes=train_dataframes,
                result_dictionaries=result_dictionaries,
                variable_names=variable_names,
                include_physical_features=include_physical_features
            )
            processed_test = process_features_with_physics(
                feature_dataframes=test_dataframes,
                result_dictionaries=result_dictionaries,
                variable_names=variable_names,
                include_physical_features=include_physical_features
            )

            # perform feature selection based on correlation
            phys_feature_list = {
                True: ["rpneg_predicted", "Dneg_predicted", "Lneg_predicted", "cspos_predicted", "Lpos_predicted", "epsspos_predicted"],
                False: ["EFC"]
            }
            train_selected, test_selected, _, _ = process_train_val_test_data_optimized(
                train_dataframes=processed_train,
                test_dataframes=processed_test,
                target_col_indices=slice(-13, -11),
                cor_limit=16,
                important_physical_features=phys_feature_list[include_physical_features],
                random_state=42,
                aging_columns=None
            )

            # prepare training data
            X_train = pd.concat([d["X"] for d in train_selected["train"].values()]).values
            y_train = pd.concat([d["y"] for d in train_selected["train"].values()]).values
            X_test = pd.concat([d["X"] for d in test_selected["test"].values()]).values
            y_test = pd.concat([d["y"] for d in test_selected["test"].values()]).values
            
            if X_train.size == 0 or y_train.size == 0 or X_test.size == 0 or y_test.size == 0:
                continue

            # train the model & make predictions
            y_train_weighted = y_train * (1, 0.001)

            rf_multi = MultiOutputRegressor(RandomForestRegressor(random_state=42, **rf_params))
            rf_multi.fit(X_train, y_train_weighted)
            y_test_pred = rf_multi.predict(X_test) / (1, 0.001)

            # evaluate
            soh_mae = mean_absolute_error(y_test[:, 0], y_test_pred[:, 0])
            rul_mae = mean_absolute_error(y_test[:, 1], y_test_pred[:, 1])
            soh_r2 = r2_score(y_test[:, 0], y_test_pred[:, 0])
            rul_r2 = r2_score(y_test[:, 1], y_test_pred[:, 1])

            print(f"✅ {v_min:.2f}V - {v_max:.2f}V: SOH_R2={soh_r2:.3f}, RUL_R2={rul_r2:.3f}")

            error_dict[(v_max, v_min)] = {
                "SOH_MAE": soh_mae,
                "SOH_R2": soh_r2,
                "RUL_MAE": rul_mae,
                "RUL_R2": rul_r2
            }
    return error_dict
#%% search best param for AI 
param_grids = {
    "Random Forest": {
        "n_estimators": [50, 100, 200, 300, 400, 500, 600],
        "max_depth": [5, 10, 15, 20, 30, 50, None],
        "min_samples_split": [2, 5, 10, 15, 20],
        "min_samples_leaf": [1, 2, 5, 10, 15],
        "max_features": ['sqrt', 'log2', 0.5, 0.8],
        'bootstrap': [True, False]
    },
    
    "SVR": {
        "C": np.logspace(-2, 2, 5), #
        "epsilon": np.linspace(0.01, 0.3, 5),
        "kernel": ["linear", "rbf", "poly", "sigmoid"], 
        "gamma": np.logspace(-4, 2, 5),  
        "degree": [2, 3, 4]  
    },

    "XGBoost": {
        "n_estimators": [50, 100, 200, 300, 500],
        "max_depth": [3, 6, 10, 15],
        "learning_rate": np.logspace(-3, 0, 5),
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha": np.logspace(-2, 1, 4),  
        "reg_lambda": np.logspace(-2, 1, 4), 
        "min_child_weight": [1, 3, 5, 10], 
        "gamma": [0, 0.1, 0.2, 0.5, 1]
    },

    "MLP": {
        "hidden_layer_sizes": [(4,), (8,), (16,), (8, 4), (16, 8), (4, 2)],
        "activation": ["relu", "tanh"],
        "solver": ["adam", "lbfgs"],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "learning_rate": ["constant", "adaptive"],
    },
    
    "Linear Regression": {
        "fit_intercept": [True, False]
}}

def train_model_with_gridsearch(model_name, X_train, y_train, search_type="random", n_iter=50):

    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    model_mapping = {
        "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "SVR": SVR(),
        "XGBoost": XGBRegressor(random_state=42, n_jobs=-1),
        # "GPR": GaussianProcessRegressor(),
        "MLP": MLPRegressor(max_iter=1000, random_state=42),
        "Linear Regression": LinearRegression()
    }
    
    base_model = model_mapping[model_name]
    param_grid = param_grids.get(model_name, {})
    
    if model_name in ["Random Forest", "MLP", "Linear Regression"]:
        print(f"\n🔍 Tuning {model_name} (Multi-task optimization)")  
        search = GridSearchCV(
            base_model,
            param_grid,
            scoring='neg_mean_squared_error',
            cv=5, n_jobs=-1, verbose=1
        ) if search_type == "grid" else RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring='neg_mean_squared_error',
            cv=5, n_jobs=-1, verbose=1, random_state=42
        )
    
        search.fit(X_train, y_train)
        best_estimator = search.best_estimator_
        best_params = search.best_params_
        
    else:

        best_estimators = []
        best_params_list = []

        for i in range(y_train.shape[1]):  
            print(f"\n🔍 Tuning for Target {i+1}/{y_train.shape[1]} in {model_name}")
    
            search = GridSearchCV(
                base_model, param_grid, scoring='neg_mean_squared_error',
                cv=5, n_jobs=-1, verbose=1
            ) if search_type == "grid" else RandomizedSearchCV(
                base_model, param_distributions=param_grid, n_iter=n_iter,
                scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=1, random_state=42
            )
    
            search.fit(X_train, y_train[:, i])  
            best_estimators.append(search.best_estimator_)  
            best_params_list.append(search.best_params_)  
            
        best_estimator = MultiOutputRegressor(best_estimators)  

        if len(best_params_list) == 2:
            best_params = {
                "SOH": best_params_list[0],
                "RUL": best_params_list[1]
            }
        else:
            print(f"⚠ Warning: Unexpected number of best_params for {model_name}: {best_params_list}")
            best_params = {
                "SOH": {},
                "RUL": {}
            }         
    return best_estimator, best_params 

def train_and_evaluate_model(model_name, X_train, y_train, X_test, y_test, best_params, weights):
    y_train_scaled = y_train * weights
    start_time = time.time()

    model_mapping = {
        "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "SVR": SVR(),
        "XGBoost": XGBRegressor(random_state=42, n_jobs=-1),
        # "GPR": GaussianProcessRegressor(),
        "MLP": MLPRegressor(max_iter=1000, random_state=42),
        "Linear Regression": LinearRegression()
    }

    base_model = model_mapping[model_name]
    
    if best_params is None:
        print(f"⚠ Warning: {model_name} received None best_params, using default parameters.")
        best_params = {} 

    if not isinstance(best_params, dict):
        print(f"⚠ Warning: {model_name} best_params is not a dictionary. Received: {best_params}")
        best_params = {}
    
    if model_name in ["Random Forest", "MLP", "Linear Regression"]:
        base_model.set_params(**best_params)
        model = base_model
        model.fit(X_train, y_train_scaled)

    else:
      
        if isinstance(best_params, dict) and "SOH" in best_params and "RUL" in best_params:
            soh_model = base_model.__class__(**best_params["SOH"])  
            rul_model = base_model.__class__(**best_params["RUL"])  
        else:
            print(f"⚠ Warning: Unexpected format for {model_name} params, using default model.")
            soh_model = base_model.__class__()
            rul_model = base_model.__class__()

        soh_model.fit(X_train, y_train_scaled[:, 0]) 
        rul_model.fit(X_train, y_train_scaled[:, 1])  

    train_time = time.time() - start_time  

    if model_name in ["Random Forest", "MLP", "Linear Regression"]:
        y_pred = model.predict(X_test) / weights 
    else:
        y_pred_soh = soh_model.predict(X_test) / weights[0] 
        y_pred_rul = rul_model.predict(X_test) / weights[1] 
        y_pred = np.column_stack((y_pred_soh, y_pred_rul))  

    test_mae = {
        "SOH": mean_absolute_error(y_test[:, 0], y_pred[:, 0]),
        "RUL": mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    }
    return train_time, test_mae

def robust_model_param_search(
    model_name,
    feature_dataframes,
    current_mapping,
    temperature_mapping,
    param_grid,
    target_col_indices,
    result_dictionaries,      
    variable_names,                
    n_trials=30,
    n_iter=20,
    cv=5,
    train_ratio=0.8,
    random_state=42,
    aging_columns=None,
    cor_limit=16,
    weights=(1.0, 0.001),
    physical_features=["EFC"],
):

    model_scores = []
    all_params = []

    rng = np.random.RandomState(42)  
    seeds = rng.randint(0, 10000, size=n_trials)

    for i, seed in enumerate(seeds):
        print(f"\n🚀 Trial {i+1}/{n_trials} for {model_name}")
        
        data = prepare_battery_dataset(
            feature_dataframes=feature_dataframes,
            current_mapping=current_mapping,
            temperature_mapping=temperature_mapping,
            result_dictionaries=result_dictionaries,
            variable_names=variable_names,
            target_col_indices=target_col_indices,
            important_physical_features=physical_features,
            train_ratio=train_ratio,
            seed=seed,
            include_physical_features=True,
            aging_columns=aging_columns,
            return_dataframe=False
        )
        
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]

        _, best_params = train_model_with_gridsearch(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            search_type="random",
            n_iter=n_iter
        )

        train_time, test_mae = train_and_evaluate_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            best_params=best_params,
            weights=weights
        )

        weighted_score = weights[0] * test_mae["SOH"] + weights[1] * test_mae["RUL"]
        model_scores.append(weighted_score)
        all_params.append(best_params)

    best_index = int(np.argmin(model_scores))
    best_score = model_scores[best_index]
    best_param = all_params[best_index]

    print(f"\n✅ Best overall params for {model_name} (Score={best_score:.4f}):\n{best_param}")
    return best_param, best_score