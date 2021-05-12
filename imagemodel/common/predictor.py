from typing import Generic, Optional, TypeVar

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
        
        self.setup_predict_dataset()
    
    def setup_predict_dataset(self):
        pass
    
    def predict(self):
        pass
