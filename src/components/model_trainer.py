import os
import sys
from dataclasses import dataclass

from xgboost import XGBClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from src.exception import CustomerException
from src.logger import logging
from src.utils import save_object, upsampling_train_data, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info(f"Get Independent features and Dependent feature from Train and Test datasets")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info(f"Imbalance dataset - upsampling the train data")
            X_train_smp, y_train_smp = upsampling_train_data(X_train, y_train)

            # Initialize the classifiers
            models = {
                "Random Forest": RandomForestClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier()
            }

            model_report:dict = evaluate_models(
                X_train=X_train_smp, 
                y_train=y_train_smp, 
                X_test=X_test,
                y_test=y_test,
                models=models)
            
            # Models and its corresponding Recall score from dict
            models_recall_score = {model: recall_result[0]["Recall"] for model, recall_result in model_report[1].items()}
            logging.info(f"The models and their corresponding Recall score: \n{models_recall_score}")

            # Finding the best model and its score
            best_model_name, best_score = max(models_recall_score.items(), key=lambda item: item[1])
            logging.info(f"Best Model: {best_model_name}, Recall score: {best_score}")
            print(f"Best Model: {best_model_name}, Score: {best_score}")
            
            best_model = models[best_model_name]

            save_object(
                 file_path = self.model_trainer_config.trained_model_file_path,
                 obj = best_model
            )
        
            # Prediction on Test data
            predicted = best_model.predict(X_test)
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision_score = precision_score(y_test, predicted, average='weighted')

            logging.info(f"Model Training completed")
            return precision_score

        except Exception as e:
            logging.info("Exception occured at Model Training")
            raise CustomerException(e, sys)
