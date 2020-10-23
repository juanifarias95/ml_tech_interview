from dataclasses import dataclass
from typing import List, Dict

@dataclass(repr=True)
class DatasetParams:
    """
    Parameters used for dataset creation
    """
    base_data_path: str = "../data"
    data_json_path: str = base_data_path + "/MLA_100k.jsonlines"
    keys_to_del = ['sub_status', 'deal_ids', 'seller_id',
                         'variations', 'location', 'attributes', 'tags', 'parent_item_id',
                         'coverage_areas', 'category_id', 'descriptions', 'last_updated', 'pictures',
                         'id', "non_mercado_pago_payment_methods",
                         'thumbnail', 'date_created', 'secure_thumbnail', 'stop_time',
                         'subtitle', 'start_time', 'permalink', 'geolocation']
    N: int = -10000
    test_size: int = 10000
    train_size: int = 90000


@dataclass(repr=True)
class DataPreprocessingParams:
    """
    Params used for data preprocessing (drop features, outliers, null values, feature engineer)
    """
    cols_to_drop = ["warranty", "seller_contact", "listing_source", "official_store_id",
                          "differential_pricing", "original_price", "video_id", "catalog_product_id"]

    price_outlier_high: int = 4000000
    price_outlier_low: int = 0
    cols_to_standarize = ["price", "quantity", "initial_quantity", "sold_quantity", "available_quantity"]


@dataclass
class MLParams:
    """
    Params for ML model
    """
    early_stopping: int = 100
    cv_folds: int = 5
    train_params = {"learning_rate":0.1, "n_estimators":140, "max_depth":5,
                          "min_child_weight":3, "gamma":0.2, "subsample":0.6, "colsample_bytree":1.0,
                          "objective":'binary:logistic', "nthread":4, "scale_pos_weight":1, "seed":27}


@dataclass
class ExpParams:
    """
    Parameters for this experiment
    """
    path_train_results: str = "reports/train"
    path_test_results: str = "reports/test"
    output_results_file: str = "results.log"
    data_path: str = "../data"
