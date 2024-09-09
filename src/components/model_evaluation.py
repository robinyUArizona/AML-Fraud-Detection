import os
import sys
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from urllib.parse import urlparse

from src.exception import CustomerException
from src.logger import logging
from src.utils import model_metrics, load_object

class ModelEvaluation:
    def __init__(self):
        logging.info("Model evaluation started")

    def eval_metrics(self, y_test, y_pred):
        precision, recall, f1, cm = model_metrics(y_test, y_pred)
        return precision, recall, f1, cm
    
    def initiate_model_evaluation(self, train_array, test_array):
        try:
            X_test, y_test = (test_array[:, :-1], test_array[:,-1])

            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(file_path=model_path)

            # mlflow.set_registry_uri("")
            logging.info("model has register")


            with mlflow.start_run():
                predictions = model.predict(X_test)
                signature = infer_signature(X_test, predictions)
                (precision, recall, f1, cm) = self.eval_metrics(y_test, predictions)

                print(f"Precision: {precision}")
                print(f"Recall:{recall}")
                print(f"F1 score: {f1}")
                print(f"Confusion Matrix: {cm}")

                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)
                # mlflow.log_metric("cm", cm)

                # For remote server only (Dagshub)
                # remote_server_uri = "https://dagshub.com/robinyUArizona/MLflow-Basic.mlflow"
                # remote_server_uri = "http://ec2-44-220-146-6.compute-1.amazonaws.com:5000/"
                # mlflow.set_tracking_uri(remote_server_uri)
                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                print(tracking_url_type_store)

                 # Model registry does not work with file store
                if tracking_url_type_store != "file":
                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForestBestModel")
                else:
                    mlflow.sklearn.log_model(model, "model", signature=signature)
        except Exception as e:
            raise CustomerException
