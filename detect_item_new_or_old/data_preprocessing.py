import pandas as pd
import numpy as np

import logging
import warnings

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
logger = logging.getLogger()


class DataPreprocessing(object):
    def __init__(self, params):
        self.params = params

    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.params.cols_to_drop:
            df.drop([col], axis=1, inplace=True)

        df.drop(["international_delivery_mode"], axis=1, inplace=True)
        df.drop(["title"], axis=1, inplace=True)
        df.drop(["seller_address_city"], axis=1, inplace=True)
        df.drop(["site_id"], axis=1, inplace=True)
        df.drop(["base_price"], axis=1, inplace=True)
        df.drop(["seller_address_country"], axis=1, inplace=True)

        return df

    def change_target_type(self, df: pd.DataFrame) -> pd.DataFrame:
        df['target'] = df['target'].replace(['new'], 1)
        df['target'] = df['target'].replace(['used'], 0)
        
        return df

    def drop_anormal_values_train(self, X_train_df: pd.DataFrame) -> pd.DataFrame:
        idxs = X_train_df[(X_train_df.seller_address_state == "80.0") | (X_train_df.seller_address_state == "100.0") | (X_train_df.seller_address_state == "5500.0")].index 
        X_train_df.drop(idxs, inplace=True)
        
        idxs = X_train_df[(X_train_df.shipping_free == "80.0") | (X_train_df.shipping_free == "100.0") | (X_train_df.shipping_free == "5500.0")].index 
        X_train_df.drop(idxs, inplace=True)
        
        X_train_df['accepts_mercadopago'] = X_train_df['accepts_mercadopago'].replace(['True'], 1)
        X_train_df['accepts_mercadopago'] = X_train_df['accepts_mercadopago'].replace(['False'], 0)
        
        X_train_df['shipping_free'] = X_train_df['shipping_free'].replace(['True'], 1)
        X_train_df['shipping_free'] = X_train_df['shipping_free'].replace(['False'], 0)
        
        X_train_df['automatic_relist'] = X_train_df['automatic_relist'].replace([True], 1)
        X_train_df['automatic_relist'] = X_train_df['automatic_relist'].replace([False], 0)
        
        idxs = X_train_df[X_train_df.currency_id == "active"].index
        X_train_df.drop(idxs, inplace=True)
        
        idxs = X_train_df[(X_train_df.status == "1") | (X_train_df.status == "not_yet_active")].index 
        X_train_df.drop(idxs, inplace=True)
        
        return X_train_df

    def drop_anormal_values_test(self, X_test_df: pd.DataFrame) -> pd.DataFrame:
        idxs = X_test_df[(X_test_df.seller_address_state == "70.0")].index 
        X_test_df.drop(idxs, inplace=True) 

        idxs = X_test_df[(X_test_df.shipping_free == "70.0")].index 
        X_test_df.drop(idxs, inplace=True)

        X_test_df['accepts_mercadopago'] = X_test_df['accepts_mercadopago'].replace(['True'], 1)
        X_test_df['accepts_mercadopago'] = X_test_df['accepts_mercadopago'].replace(['False'], 0)

        X_test_df['automatic_relist'] = X_test_df['automatic_relist'].replace([True], 1)
        X_test_df['automatic_relist'] = X_test_df['automatic_relist'].replace([False], 0)

        X_test_df['shipping_free'] = X_test_df['shipping_free'].replace(['True'], 1)
        X_test_df['shipping_free'] = X_test_df['shipping_free'].replace(['False'], 0)

        idxs = X_test_df[(X_test_df.currency_id == "active")].index 
        X_test_df.drop(idxs, inplace=True)

        idxs = X_test_df[(X_test_df.status == "1")].index 
        X_test_df.drop(idxs, inplace=True)
        
        return X_test_df

    def drop_target_null_values(self, df: pd.DataFrame) -> pd.DataFrame:
        idxs = df[df["target"].isna()].index
        df.drop(idxs, inplace=True)
        
        return df

    def fill_null_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df["seller_address_state"].fillna(df["seller_address_state"].mode()[0], inplace=True)
        
        return df

    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates()
        
        return df

    def manage_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df["price"] = df["price"].astype("float64")
        df["accepts_mercadopago"] = df["accepts_mercadopago"].astype("float64")
        df["shipping_free"] = df["shipping_free"].astype("float64")
        df["initial_quantity"] = df["initial_quantity"].astype("float64")
        df["target"] = df["target"].astype("int")

        return df

    def delete_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["price"] < self.params.price_outlier_high]
        df = df[df["price"] > self.params.price_outlier_low]
            
        return df

    def get_quantity_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        df["quantity"] = df["initial_quantity"] + df["sold_quantity"] + df["available_quantity"]
        
        return df

    def get_price_bin_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        df["price_bin"] = pd.qcut(df["price"], 5, labels=["very_cheap", "cheap", "medium", "expensive", "very_expensive"])
        
        return df

    def get_labeled_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = pd.get_dummies(df)

        return df


    def get_scaled_data(self, df: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        for col in self.params.cols_to_standarize:
            df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
        
        return df

    def drop_diff_cols(self, X_train_df: pd.DataFrame, X_test_df: pd.DataFrame) -> pd.DataFrame:
        remaining_cols = list(set(X_train_df.columns) - set(X_test_df))
        X_train_df.drop(remaining_cols, axis=1, inplace=True)

        return X_train_df

    def preprocess_data(self, df: pd.DataFrame, data_type: str = "train") -> pd.DataFrame:
        df = self.drop_columns(df)
        df = self.change_target_type(df)
        if data_type == "train":
            df = self.drop_anormal_values_train(df)
        else:
            df = self.drop_anormal_values_test(df)
        df = self.drop_target_null_values(df)
        df = self.fill_null_values(df)
        df = self.drop_duplicates(df)
        df = self.manage_data_types(df)
        df = self.delete_outliers(df)
        df = self.get_quantity_feature(df)
        df = self.get_price_bin_feature(df)
        df = self.get_labeled_data(df)
        df = self.get_scaled_data(df)

        return df
