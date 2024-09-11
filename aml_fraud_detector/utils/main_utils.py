import os
import sys
import dill
import numpy as np
import pandas as pd

from aml_fraud_detector.logger import logging
from aml_fraud_detector.exception import CustomerException

from collections import Counter
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        logging.info(f'Exception Occured in save_object function utils')
        raise CustomerException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.info(f'Exception Occured in load_object function utils')
        raise CustomerException(e, sys)


def upsampling_train_data(X, y):
    try: 
        sm = SMOTE(sampling_strategy='auto', random_state=42)
        logging.info(f"Before SMOTE: {Counter(y)}")
        X_sm, y_sm = sm.fit_resample(X, y)   
        logging.info(f"After SMOTE: {Counter(y_sm)}")
        logging.info(f"Upsampling the minority class data completed") 
        return X_sm, y_sm
    except Exception as e:
        logging.info(f"Exception occured during upsampling the minority class")
        raise CustomerException(e, sys)


def model_metrics(y_test, y_pred):
    try:     
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred) 
        return precision, recall, f1, cm
    except Exception as e:
        logging.info(f"Exception occured during metrics calculation")


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        train_report = {}
        test_report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            # Grid Search
            logging.info(f"Grid Search started for {model}")
            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)
            logging.info(f"Grid Search completed for {model}")

            # Setting model with best hyperparameters
            logging.info(f"Best parameters: {gs.best_params_} for {model}")
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            # Predict on Train data
            y_train_pred = model.predict(X_train)
            # Predict Test data
            y_test_pred = model.predict(X_test)

            # Get evaluation metrics for train and test data
            precision_train, recall_train, f1_train, cm_train = model_metrics(y_train, y_train_pred)
            train_model_score = []
            train_model_score.append({
                "Precision" : precision_train,
                "Recall" : recall_train,
                "F1 score": f1_train,
                "Confusion Matrix": cm_train
            })
            train_report[list(models.keys())[i]] = train_model_score

            precision_test, recall_test, f1_test, cm_test = model_metrics(y_test, y_test_pred)
            test_model_score = []
            test_model_score.append({
                "Precision" : precision_test,
                "Recall" : recall_test,
                "F1 score": f1_test,
                "Confusion Matrix": cm_test
            })
            test_report[list(models.keys())[i]] = test_model_score

        logging.info(f"\n Metrics calculation on Train Data: \n{train_report}")
        logging.info(f"\n Metrics calculation on Test Data: \n{test_report}")
        return train_report, test_report

    except Exception as e:
        logging.info(f"Exception occured during model training")
        raise CustomerException(e, sys)
    
