import os
import sys
from src.exception import CustomerException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation =  DataTransformation()
    # data_transformation.initiate_data_transformation(train_data, test_data)
    train_arr, test_arr = data_transformation.initiate_data_transformation(train_data, test_data)

    # model_trainer = ModelTrainer()
    # model_trainer.initiate_model_trainer(train_arr, test_arr)

    model_eval_obj = ModelEvaluation()
    model_eval_obj.initiate_model_evaluation(train_arr, test_arr)