from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def select_from_importances(importances, threshold=0):
    selected_ids = [i for i, imp in enumerate(importances) if imp >= threshold]
    return selected_ids


lgbm_reg = 'lgbm_reg'
lgbm_class = 'lgbm_class'

estimators_dict = {
    lgbm_reg: LGBMRegressor
}

scalers_dict = {
    'standard': StandardScaler,
    'minmax': MinMaxScaler
}


def spawn_scaler(scaler):
    return scalers_dict[scaler]()


def spawn_estimator(estimator, params):
    return estimators_dict[estimator](**params)


default_params = {
    lgbm_reg: {'colsample_bytree': 1, 'n_jobs': -1, 'n_estimators': 1000,
               'learning_rate': 0.05, 'subsample': 1, 'num_leaves': 31, 'reg_alpha': 0.0,
               'reg_lambda': 0.0, 'max_bin': 15, 'min_split_gain': 0.0001}
}
