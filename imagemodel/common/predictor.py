from typing import Generic, Optional, TypeVar

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.distribute.tpu_strategy import TPUStrategy

from imagemodel.common.datasets.pipeline import Pipeline

PI = TypeVar('PI', bound=Pipeline)


class Predictor(Generic[PI]):
    def __init__(
            self,
            model: Model,
            predict_pipeline: PI,
            predict_batch_size: int,
            strategy_optional: Optional[TPUStrategy] = None):
        self.model: Model = model
        self.predict_pipeline: PI = predict_pipeline
        self.predict_batch_size: int = predict_batch_size
        self.strategy_optional: Optional[TPUStrategy] = strategy_optional
        
        self.predict_dataset: tf.data.Dataset = self.predict_pipeline.get_input_zipped_dataset()
        self.predict_dataset_num: int = len(self.predict_dataset)
        self.predict_dataset = self.predict_dataset.batch(self.predict_batch_size, drop_remainder=True)
    
    def predict(self):
        pass
