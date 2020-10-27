import pandas as pd
import numpy as np
import sklearn

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
import xgboost as xgb
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from pathlib import Path

import itertools
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
        self.models = []
        self.auc_results = np.array([])


    def train_model(self, df: pd.DataFrame, fold: int) -> xgb.Booster:
        num_cols = ["shipping_free", "price", "accepts_mercadopago", "automatic_relist",
                    "initial_quantity",	"sold_quantity", "available_quantity", "quantity"]

        cat_cols = [c for c in df.columns if c not in num_cols and c not in ("kfold", "target")]

        df = self.feature_engineer(df, cat_cols)

        features = [f for f in df.columns if f not in ("kfold", "target")]

        for col in features:
            if col not in num_cols:
                lbl = LabelEncoder()
                lbl.fit(df[col])
                df.loc[:, col] = lbl.transform(df[col])
        
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        X_train = df_train[features].values
        X_valid = df_valid[features].values

        model = xgb.XGBClassifier(**self.params.train_params)

        model.fit(X_train, df_train.target.values)

        valid_preds = model.predict_proba(X_valid)[:, 1]

        auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

        logger.info(f"Fold = {fold}, AUC = {auc}")

        self.models.append(model)
        self.auc_results = np.concatenate((self.auc_results, [auc]))

    def save_model(self, xgb_model):
        path_model = Path(os.path.join(self.report_folder, "model_new_used_item.xgb"))
        joblib.dump(xgb_model, path_model)

    def load_model(self, path: str):
        model = joblib.load(path)

        return model

    def test_model(self, model: xgb.Booster, df: pd.DataFrame) -> (str, float):
        num_cols = ["shipping_free", "price", "accepts_mercadopago", "automatic_relist",
                    "initial_quantity",	"sold_quantity", "available_quantity", "quantity"]

        cat_cols = [c for c in df.columns if c not in num_cols and c not in ("target")]

        df = self.feature_engineer(df, cat_cols)

        features = [f for f in df.columns if f not in ("target")]

        for col in features:
            if col not in num_cols:
                lbl = LabelEncoder()
                lbl.fit(df[col])
                df.loc[:, col] = lbl.transform(df[col])

        X = df.drop(["target"], axis=1).values
        y = df.target.values

        preds = model.predict_proba(X)[:, 1]

        auc = metrics.roc_auc_score(y, preds)

        logger.info(f"AUC Test = {auc}")

    def feature_engineer(self, df:pd.DataFrame, cat_cols: list) -> pd.DataFrame:
        combi = list(itertools.combinations(cat_cols, 2))
        for c1, c2 in combi:
            df.loc[:, c1 + "_" + c2] = df[c1].astype(str) + "_" + df[c2].astype(str)

        return df

