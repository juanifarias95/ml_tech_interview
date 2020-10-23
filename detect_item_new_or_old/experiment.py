"""
Exercise description
--------------------

In the context of Mercadolibre's Marketplace an algorithm is needed to
predict if an item listed in the markeplace is new or used.

Your task to design a machine learning model to predict if an item is new or
used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k.jsonlines` and a
function to read that dataset in `build_dataset`.

For the evaluation you will have to choose an appropiate metric and also
elaborate an argument on why that metric was chosen.

The deliverables are:
    - This file including all the code needed to define and evaluate a model.
    - A text file with a short explanation on the criteria applied to choose
      the metric and the performance achieved on that metric.
"""

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from numpy.random import seed
from datasets import DatasetCreation
from data_preprocessing import DataPreprocessing
from params import DatasetParams, DataPreprocessingParams, ExpParams, MLParams
from ml import MLModel

import argparse
import logging
import logging.config
import numpy as np
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore")
logger = logging.getLogger()


class NewUsedItemExperiment(object):
    def __init__(self, params):
        self.params = params

    def parse_arguments(self):
        """
        Instantiates a new ArgumentParser

        :returns: Python's ArgumentParser
        """
        argparser = argparse.ArgumentParser()
        argparser.add_argument('--train', action='store_true', default=False,
                                help="If set, this argument will force the experiment to train over train dataset")
        argparser.add_argument('--test', action='store_true', default=False,
                                help="If set, this argument will force the experiment to test over test dataset")
        argparser.add_argument('--test-models', default=None, help="Path where XBoost model and discretizers are")    
        
        self.args = argparser.parse_args()

    def prepare_environment(self):
        """
        Creates folders, set up some attributes to use accross the
        experiment, shuch as output directories, logger, etc.
        """
        # Get the experiment folder from child class
        path = os.getcwd()
        if self.args.train:
            self.out_dir = Path(os.path.join(path, self.params.path_train_results, str(datetime.now().strftime("%Y%m%d_%H%M%S"))))
        elif self.args.test:
            self.out_dir = Path(os.path.join(path, self.params.path_test_results, str(datetime.now().strftime("%Y%m%d_%H%M%S"))))

        self.out_dir.mkdir(parents=False, exist_ok=False)
        # Prepare logger
        log_path = os.path.join(self.out_dir, self.params.output_results_file)
        logging.basicConfig(filename=log_path, level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.set_seed()

    def set_seed(self, num=42):
        """
        You must have a real, unavoidable and urgent reason
        to use a different number
        """
        seed(num)

    def kickstart(self):
        """
        Entry method to run the experiment.
        It is in charge of setting up the proper
        variables, creating all the related directories and running
        the experiment.
        """
        self.parse_arguments()
        self.prepare_environment()
        self.logger.info("\nStarting experiment\n")
        self.logger.info("\nSaving results to {}".format(self.out_dir))
        self.run()
        self.logger.info("\nExperiment finished!")

    def prepare_data_for_experiment(self):
        data_preprocessing_obj = DataPreprocessing(DataPreprocessingParams)
 
        if not os.path.exists(os.path.join(ExpParams.data_path, "train.csv")):
            data_creation_obj = DatasetCreation(DatasetParams)
            data_creation_obj.create_dataset()

        df_train = pd.read_csv(os.path.join(ExpParams.data_path, "train.csv"))
        df_test = pd.read_csv(os.path.join(ExpParams.data_path, "test.csv"))

        df_train = data_preprocessing_obj.preprocess_data(df_train, data_type="train")
        df_test = data_preprocessing_obj.preprocess_data(df_test, data_type="test")

        df_train = data_preprocessing_obj.drop_diff_cols(df_train, df_test)

        return df_train, df_test

    def run(self):
        """
        Main experiment function
        """
        df_train, df_test = self.prepare_data_for_experiment()

        if self.args.test:
            if not self.args.test_models:
                sys.exis("You must specify which model you want to test. Remember it is a folder located on reports/train/ and" 
                         "the folder is like 20201023_183350")
            self.test_experiment(df_test)
        else:
            self.train_experiment(df_train)

    def train_experiment(self, df_train):     
        ml_model_obj = MLModel(MLParams, self.out_dir)

        X_train = df_train.drop(["target"], axis=1)
        y_train = df_train["target"]

        self.logger.info("Training model...\n")
        xgb_model = ml_model_obj.train_model(X_train, y_train)

        self.logger.info("\nTRAIN RESULTS\n")
        self.logger.info("=============\n")

        report, tn, fp, fn, tp = ml_model_obj.test_model(xgb_model, X_train, y_train)
        self.logger.info(f"Classification report: \n{report}\n")
        self.logger.info(f"True Negatives = {tn}\n")
        self.logger.info(f"False Positives = {fp}\n")
        self.logger.info(f"False Negatives = {fn}\n")
        self.logger.info(f"True Positives = {tp}\n")
        self.logger.info(f"AUC = {auc}")
        

    def test_experiment(self, df_test):
        ml_model_obj = MLModel(MLParams)
        path = os.getcwd()
        model_folder = os.path.join(path, self.params.path_train_results, self.args.test_models, "model_new_used_item.xgb")
        xgb_model = ml_model_obj.load_model(model_folder)

        X_test = df_test.drop(["target"], axis=1)
        y_test = df_test["target"]

        self.logger.info("TEST RESULTS")
        self.logger.info("============\n")
        report, tn, fp, fn, tp, auc = ml_model_obj.test_model(xgb_model, X_test, y_test)
        self.logger.info(f"Classification report: \n{report}\n")
        self.logger.info(f"True Negatives = {tn}\n")
        self.logger.info(f"False Positives = {fp}\n")
        self.logger.info(f"False Negatives = {fn}\n")
        self.logger.info(f"True Positives = {tp}\n")
        self.logger.info(f"AUC = {auc}")


if __name__ == "__main__":
    new_used_item_obj = NewUsedItemExperiment(ExpParams)
    new_used_item_obj.kickstart()

