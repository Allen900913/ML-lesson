import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
from sklearn.datasets import fetch_california_housing
import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


housing_dataset = fetch_california_housing()


housing = pd.DataFrame(housing_dataset.data, columns=housing_dataset.feature_names)
housing['MEDV'] = housing_dataset.target
housing.head()

X  = housing.drop(['MEDV'],axis=1).values
y = housing['MEDV'].values



def objective(trial, X=X, y=y):
    """
    A function to train a model using different hyperparamerters combinations provided by Optuna. 
    Log loss of validation data predictions is returned to estimate hyperparameters effectiveness.
    """
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4)

    param = {
        'verbosity': 0,
        'objective': 'reg:squarederror',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000, step=100),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
    }
    
    # 訓練模型，設置評估集和 early stopping
    reg = xgb.XGBRegressor(**param)
    
    reg.fit(X_train, y_train, 
            eval_set=[(X_valid, y_valid)], 
            early_stopping_rounds=10, 
            eval_metric="rmse",  # 設置評估指標
            verbose=False)

    # 預測並計算 MSE
    preds = reg.predict(X_valid)
    mse = mean_squared_error(y_valid, preds)
    
    return mse



study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials = 10)

# Showing optimization results
print('Number of finished trials:', len(study.trials))
print('Best trial parameters:', study.best_trial.params)
print('Best score:', study.best_value)