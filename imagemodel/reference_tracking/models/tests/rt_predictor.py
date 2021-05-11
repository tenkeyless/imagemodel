from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.distribute.tpu_strategy import TPUStrategy

import _path  # noqa
from imagemodel.common.predictor import Predictor
from imagemodel.reference_tracking.datasets.pipeline import RTPipeline


class RTPredictor(Predictor[RTPipeline]):
    def __init__(
            self,
            model: Model,
            predict_pipeline: RTPipeline,
            predict_batch_size: int,
            filled_empty_with: Tuple[int, int, int],
            strategy_optional: Optional[TPUStrategy] = None):
        super().__init__(model, predict_pipeline, predict_batch_size, strategy_optional)
        self.predict_dataset: tf.data.Dataset = self.predict_pipeline.get_input_zipped_dataset_filled(filled_empty_with)
    
    def predict(self):
        for predict_data in self.predict_dataset:
            predicted = self.model.predict(predict_data[0], batch_size=self.predict_batch_size, verbose=1)
