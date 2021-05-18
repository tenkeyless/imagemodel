from typing import Optional

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.distribute.tpu_strategy import TPUStrategy


class Predictor:
    def __init__(
            self,
            model: Model,
            predict_dataset: tf.data.Dataset,
            predict_dataset_description: str,
            predict_batch_size: int,
            strategy_optional: Optional[TPUStrategy] = None):
        self.model: Model = model
        self.predict_dataset: tf.data.Dataset = predict_dataset
        self.predict_dataset_description: str = predict_dataset_description
        self.predict_batch_size: int = predict_batch_size
        self.strategy_optional: Optional[TPUStrategy] = strategy_optional
    
    def predict(self):
        pass
