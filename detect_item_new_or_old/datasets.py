import json
import pandas as pd

from collections import defaultdict

import os
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger()


class DatasetCreation(object):
    def __init__(self, params):
        self.params = params


    def build_dataset(self):
        data = [json.loads(x) for x in open(self.params.data_json_path)]
        target = lambda x: x.get("condition")
        N = self.params.N
        X_train = data[:N]
        X_test = data[N:]
        y_train = [target(x) for x in X_train]
        y_test = [target(x) for x in X_test]
        for x in X_train:
            del x["condition"]
        for y in X_test:
            del y["condition"]

        return X_train, y_train, X_test, y_test

    def delete_not_used_keys(self, dataset: list) -> list:
        for item in dataset:
            for key in self.params.keys_to_del:
                del item[key]

        return dataset

    def get_data(self, dataset: list) -> dict:
        data = defaultdict(list)
        for item in dataset:
            for key in item.keys():
                if key == "seller_address":
                    data[key + "_country"].append(item[key]["country"]["name"])
                    data[key + "_city"].append(item[key]["city"]["name"])
                    data[key + "_state"].append(item[key]["state"]["name"])
                elif key == "shipping":
                    data[key + "_free"].append(item[key]["free_shipping"])
                else:
                    data[key].append(item[key])

        return data

    def review_dataset_size(self, data: dict, what_data: str) -> None:
        size = self.params.train_size if what_data == "train" else self.params.test_size
        for key in data.keys():
            assert len(data[key]) == size

    def create_dataframe(self, data: dict, y: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(columns=list(data.keys()))
        for col in df.columns:
            df[col] = data[col]
        df["target"] = y

        return df

    def save_dataframe(self, df: pd.DataFrame, data_type: str) -> None:
        df.to_csv(os.path.join(self.params.base_data_path, data_type + ".csv"), index=False)

    def create_dataset(self):
        logging.info("\n Building dataset from JSON file...\n")
        X_train, y_train, X_test, y_test = self.build_dataset()
        X_train = self.delete_not_used_keys(X_train)
        X_test = self.delete_not_used_keys(X_test)
        data_train = self.get_data(X_train)
        data_test = self.get_data(X_test)
        self.review_dataset_size(data_train, what_data="train")
        self.review_dataset_size(data_test, what_data="test")
        train_df = self.create_dataframe(data_train, y_train)
        test_df = self.create_dataframe(data_test, y_test)
        self.save_dataframe(train_df, data_type="train")
        self.save_dataframe(test_df, data_type="test")
