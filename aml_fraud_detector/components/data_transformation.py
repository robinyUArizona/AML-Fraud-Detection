import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder, CountEncoder

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        """
        This function is responsible for data transformation
        """
        try:          
            # Preprocessing for numerical features:
            num_transformer = make_pipeline(
                SimpleImputer(strategy='median'),  # Impute missing values with median
                RobustScaler()  # Scale numerical features
            )
            
            # Preprocessing for categorical features:
            # Frequency Encoding for high cardinality features
            freq_encoder = CountEncoder(normalize=True) 
            # One-Hot Encoding for low cardinality features
            one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

            # Apply different encodings to different categorical features
            cat_transformer = make_column_transformer(
                (freq_encoder, ['account', 'account_1']),  # Frequency Encoding for account and account_1
                (one_hot_encoder, ['payment_format', 'day']),  # One-Hot Encoding for others
                remainder="drop"  # Drop columns not explicitly transformed
            )

            preprocessor = make_column_transformer(
                (num_transformer, numerical_columns),  # Apply numerical transformer to numerical features
                (cat_transformer, categorical_columns),  # Apply categorical transformer to categorical features
                remainder="drop"  # Drop columns not explicitly transformed
            )

            logging.info(f"Preprocessed both numerical and categorical columns")

            return preprocessor
            
        except Exception as e:
            raise CustomerException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        logging.info(f"\nEntered the 'data transformation' method or component")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"Reading train and test data completed")

            train_df.columns = train_df.columns.str.lower().str.replace(' ', '_').str.replace('.', '_')
            test_df.columns = test_df.columns.str.lower().str.replace(' ', '_').str.replace('.', '_')
            logging.info("Train and Test dataframe columns name renamed")

            logging.info(f"Train Dataframe Head : \n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head : \n{test_df.head().to_string()}")


            # Convert the "from_bank" and "to_bank" column data type from int (numeric) data type to object (category) data type
            train_df["from_bank"] = train_df["from_bank"].astype("object")
            train_df["to_bank"] = train_df["to_bank"].astype("object")
            test_df["from_bank"] = test_df["from_bank"].astype("object")
            test_df["to_bank"] = test_df["to_bank"].astype("object")

            # Convert the "timestamp" column to datetime format
            train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
            test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

            # Extract date, day, and time from the Timestamp
            train_df["date"] = train_df["timestamp"].dt.date
            train_df["day"] = train_df["timestamp"].dt.day_name()
            train_df["time"] = train_df["timestamp"].dt.time
            test_df["date"] = test_df["timestamp"].dt.date
            test_df["day"] = test_df["timestamp"].dt.day_name()
            test_df["time"] = test_df["timestamp"].dt.time

            # Get Independent features (drop unwanted columns) and Dependent feature
            target_column_name = "is_laundering"        
            drop_columns = [target_column_name, "timestamp", "date", "time", "amount_paid", "receiving_currency", "payment_currency", "from_bank", "to_bank"]
            input_features_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Get column names of input features which need to be transformed
            numerical_features = input_features_train_df.select_dtypes(include=np.number).columns.tolist()
            logging.info(f"Columns name of numerical features: {numerical_features}")
            categorical_features = input_features_train_df.select_dtypes(include=object).columns.tolist()
            logging.info(f"Columns name of categorical features: {categorical_features}")

            logging.info(f"Obtaining proprocessing object")
            preprocessing_obj = self.get_data_transformer_object(numerical_features, categorical_features)

            logging.info(f"Applying preprocessing object on training and testing datasets.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_train_arr = input_feature_train_arr.toarray()  # Convert to dense array
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)
            input_feature_test_arr = input_feature_test_arr.toarray()    # Convert to dense array
            
            # concatenating all input features and target feature along column wise
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info(f"Saved data preprocessing object")

            return(
                train_arr,
                test_arr
            )

        except Exception as e:
            raise CustomerException(e, sys)

