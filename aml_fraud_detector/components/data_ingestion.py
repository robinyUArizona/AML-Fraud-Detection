import os
import sys
from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts", "train.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")
    raw_data_path:str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info(f"Entered the 'data ingestion' method or component")
        try:
            # Construct the path dynamically
            csv_file = os.path.join(
                "/Users/robins/Desktop/Robins World/Data Science - Machine Learning Prep/01 - MLOps/AML-Fraud-Detection/notebook/data","HI-Small_Trans.csv"
            )
            logging.info(f"CSV File Path:, {csv_file}")

            # Check if the file exists
            if os.path.exists(csv_file):
                logging.info(f"File found: {csv_file}")
            else:
                logging.info(f"Error: File not found at {csv_file}")

            df = pd.read_csv(csv_file)
            # Take 50,000 sampples of the data
            df_sample = df.sample(n=50000, random_state=6)
            logging.info(f"Read the dataset as DataFrame")
            # create directory for raw data
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)), exist_ok=True)
            # save raw data
            df_sample.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info(f"Train Test split initiated")
            train_set, test_set = train_test_split(df_sample, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Ingestion of the data completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomerException(e, sys)




