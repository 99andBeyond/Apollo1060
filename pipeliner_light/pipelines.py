import json
import os

import joblib
import pandas as pd

from .algos import scalers_dict

pd.options.mode.chained_assignment = None
import numpy as np


class ClassicPipe:
    def __init__(self, features, estimator='lgbm', classification=False, scaler='standard',
                 loader=None, y_endpoint=None, feature_selection=None, load_from=None):

        assert scaler in scalers_dict.keys(), 'Your scaler is not supported. Supported scalers: standard, minmax '
        self.scaler = scaler
        self.feature_selection = feature_selection
        self.features = features
        self.load_from = load_from
        self.classification = classification
        self.y_endpoint = y_endpoint
        self.features_importances = []
        self.n_estimators = []

        if loader:
            self.fitted_scalers = loader['fitted_scalers']
            self.fitted_selectors = loader['fitted_selectors']
            self.fitted_estimators = loader['fitted_estimators']
            self.fitted_estimators_perf = loader['fitted_estimators_perf']
            self.optimal_params = loader['optimal_params']
            self.estimator = estimator
        else:
            self.fitted_scalers = []
            self.fitted_selectors = []
            self.fitted_estimators = []
            self.fitted_estimators_perf = {"train": [], "val": []}
            self.optimal_params = {}
            self.estimator = estimator
            if self.classification:
                self.estimator += '_class'
            else:
                self.estimator += '_reg'

    def predict_vector(self, feature_vector):
        predictions_list = []
        feature_vector = feature_vector.reshape(1, -1)
        for scaler, selector, estimator in zip(self.fitted_scalers, self.fitted_selectors, self.fitted_estimators):
            feature_vector_scaled = scaler.transform(feature_vector)
            if self.feature_selection is not None:
                feature_vector_scaled = feature_vector_scaled[:, selector]
            predictions_list.append(estimator.predict(feature_vector_scaled))
        mean_prediction = np.array(predictions_list).mean(axis=0)[0]

        return mean_prediction

    @classmethod
    def load(cls, model_folder):
        loader = {}
        fitted_estimators_perf = {"train": [], "val": []}
        fitted_estimators = []
        fitted_selectors = []
        fitted_scalers = []

        perf_df = pd.read_csv(os.path.join(model_folder, 'performance.csv'))
        metrics = list(perf_df.columns.values)[1:]

        for i, row in perf_df.iterrows():
            try:
                fitted_estimators_perf[row['set']].append({metric: row[metric] for metric in metrics})
            except KeyError:
                pass

        cv_predicted_df = pd.read_csv(os.path.join(model_folder, 'cv_predicted_training_set.csv'))

        with open(os.path.join(model_folder, 'args.json')) as args_in:
            args_dict = json.load(args_in)

        for i in range(0, 10000):
            try:
                scaler = joblib.load(os.path.join(model_folder, 'model_{!s}'.format(i), 'scaler.sav'))
                estimator = joblib.load(os.path.join(model_folder, 'model_{!s}'.format(i), 'estimator.sav'))
                if args_dict['feature_selection'] is not None:
                    selector = np.load(os.path.join(model_folder, 'model_{!s}'.format(i), 'selector.npy'))
                    fitted_selectors.append(selector)
                fitted_scalers.append(scaler)
                fitted_estimators.append(estimator)

            except FileNotFoundError as error:
                print('Have read {!s} models'.format(i))
                break

        loader['fitted_scalers'] = fitted_scalers
        if args_dict['feature_selection'] is not None:
            loader['fitted_selectors'] = fitted_selectors
        loader['fitted_estimators'] = fitted_estimators
        loader['fitted_estimators_perf'] = fitted_estimators_perf
        loader['cv_predicted_df'] = cv_predicted_df

        with open(os.path.join(model_folder, 'params.json')) as params_in:
            params_dict = json.load(params_in)

        loader['optimal_params'] = params_dict

        args_dict['loader'] = loader

        return cls(**args_dict)
