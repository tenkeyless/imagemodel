from typing import Optional

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.distribute.tpu_strategy import TPUStrategy

from imagemodel.common.datasets.pipeline import Pipeline


class Predictor:
    def __init__(
            self,
            model: Model,
            predict_pipeline: Pipeline,
            predict_batch_size: int,
            strategy_optional: Optional[TPUStrategy] = None):
        self.model: Model = model
        self.predict_pipeline: Pipeline = predict_pipeline
        self.predict_batch_size: int = predict_batch_size
        self.strategy_optional: Optional[TPUStrategy] = strategy_optional
        
        self.predict_dataset: tf.data.Dataset = tf.data.Dataset.zip(
                self.predict_pipeline.preprocessor.get_input_dataset())
        self.predict_dataset_num: int = len(self.predict_dataset)
        self.predict_dataset = self.predict_dataset.batch(self.predict_batch_size, drop_remainder=True)
        self.predict_dataset = self.predict_dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)
    
    def predict(self):
        return self.model.predict(self.predict_dataset, batch_size=self.predict_batch_size, verbose=1)
