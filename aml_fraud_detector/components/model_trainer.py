import os
import sys
from dataclasses import dataclass

from xgboost import XGBClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import save_object, upsampling_train_data, evaluate_models


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
                # "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier()
            }

            params = {
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    # 'criterion': ['gini', 'entropy'],
                    # 'max_depth': [None, 10, 20, 30],
                    # 'min_samples_split': [2, 5, 10],
                    # 'min_samples_leaf': [1, 2, 4],
                    # 'max_features': ['sqrt', 'log2', None]
                },
                "AdaBoost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0],
                    # 'algorithm': ['SAMME', 'SAMME.R']
                },
                # "Gradient Boosting": {
                #     # 'n_estimators': [50, 100, 200],
                #     # 'learning_rate': [0.01, 0.1, 0.05, 0.001],
                #     # 'subsample': [0.6, 0.7, 0.8, 0.9],
                #     # 'max_depth': [3, 5, 7, 9],
                #     # 'min_samples_split': [2, 5, 10],
                #     # 'min_samples_leaf': [1, 2, 4]
                # },
                "XGBoost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.05, 0.001],
                    # 'max_depth': [3, 5, 7, 9],
                    # 'min_child_weight': [1, 3, 5],
                    # 'gamma': [0, 0.1, 0.2],
                    # 'subsample': [0.6, 0.7, 0.8, 0.9],
                    # 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
                }
            }

            model_report:dict = evaluate_models(
                X_train=X_train_smp, 
                y_train=y_train_smp, 
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params)
            
            # Models and its corresponding Recall score from dict
            models_recall_score = {model: recall_result[0]["Recall"] for model, recall_result in model_report[1].items()}
            logging.info(f"The models and their corresponding Recall score: \n{models_recall_score}")

            # Finding the best model and its score
            best_model_name, best_score = max(models_recall_score.items(), key=lambda item: item[1])
            logging.info(f"Best Model: {best_model_name} with Recall score: {best_score}")
            print(f"Best Model: {best_model_name} with Recall score: {best_score}")
            
            best_model = models[best_model_name]

            save_object(
                 file_path = self.model_trainer_config.trained_model_file_path,
                 obj = best_model
            )
        
            # Prediction on Test data
            predicted = best_model.predict(X_test)
            from sklearn.metrics import precision_score, recall_score, f1_score
            recall_Score = recall_score(predicted, y_test, average='weighted')

            logging.info(f"Model Training completed")
            logging.info(f"Final Recall score for the {best_model}: {recall_Score}")
            print(f"Final Recall score for the best model i.e. {best_model}: {recall_Score}")
            return precision_score

        except Exception as e:
            logging.info("Exception occured at Model Training")
            raise CustomerException(e, sys)
