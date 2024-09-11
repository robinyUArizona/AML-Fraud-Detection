from aml_fraud_detector.logger import logging
from aml_fraud_detector.exception import CustomerException
import sys

logging.info("Welcome to our custom log")


try:
    a = 1/0
except Exception as e:
    logging.info("==== Divide by Zero ====")
    raise CustomerException(e, sys)