import sys
import pandas as pd

from src.exception import CustomerException
from src.logger import logging
from src.utils import load_object


class PredictionPipeline:
    def __init__(self):
        pass 


class CustomData:
    def __init__(
            self,
            timestamp: str,
            from_bank: int,
            account: str,
            to_bank: int,
            account_1: int,
            amount_received: float,
            receiving_currency: str,
            amount_paid: float,
            payment_currency: str,
            payment_format: str,
            is_laundering: int,
            ):
    
        self.timestamp = timestamp
        self.from_bank = from_bank
