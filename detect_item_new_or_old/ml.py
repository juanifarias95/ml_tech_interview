import pandas as pd
import sklearn

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
import xgboost as xgb
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from pathlib import Path

import joblib
import warnings
import logging
import os

warnings.filterwarnings('ignore')
logger = logging.getLogger()


class MLModel(object):
    def __init__(self, params, report_folder=None):
        self.params = params
        self.report_folder = report_folder

    def train_model(self, X: pd.DataFrame, y: pd.DataFrame) -> xgb.Booster:
        xgb_model = XGBClassifier(**self.params.train_params)

        xgb_param = xgb_model.get_xgb_params()
        xgb_train = xgb.DMatrix(X.values, label=y.values)
        cv_result = xgb.cv(xgb_param, xgb_train, stratified=True, num_boost_round=xgb_model.get_params()['n_estimators'],
                           nfold=self.params.cv_folds, early_stopping_rounds=self.params.early_stopping, verbose_eval=True)
        xgb_model.set_params(n_estimators=cv_result.shape[0])
        
        # fit model
        xgb_model.fit(X, y)

        # Save model
        path_model = Path(os.path.join(self.report_folder, "model_new_used_item.xgb"))
        joblib.dump(xgb_model, path_model)

        return xgb_model

    def load_model(self, path: str):
        model = joblib.load(path)

        return model

    def test_model(self, model: xgb.Booster, X: pd.DataFrame, y: pd.DataFrame) -> (str, float):
        preds = model.predict(X)
        report = classification_report(y, preds)
        cm = confusion_matrix(y, preds)
        tn, fp, fn, tp = cm.ravel()
        fpr, tpr, thresholds = metrics.roc_curve(y, preds)
        auc = metrics.auc(fpr, tpr)

        return report, tn, fp, fn, tp, auc

