from typing import Callable, Optional

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.distribute.tpu_strategy import TPUStrategy

from imagemodel.common.predictor import Predictor
from imagemodel.reference_tracking.datasets.pipeline import RTPipeline


class RTPredictor(Predictor[RTPipeline]):
    def __init__(
            self,
            model: Model,
            predict_pipeline: RTPipeline,
            predict_batch_size: int,
            post_processing: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], None],
            strategy_optional: Optional[TPUStrategy] = None):
        super().__init__(model, predict_pipeline, predict_batch_size, strategy_optional)
        self.post_processing: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], None] = post_processing
    
    def setup_predict_dataset(self):
        filename_dataset = self.predict_pipeline.feeder.filename_optional
        
        self.predict_dataset = self.predict_pipeline.get_input_zipped_dataset()
        self.predict_dataset_num: int = len(self.predict_dataset)
        
        self.predict_dataset = tf.data.Dataset.zip((self.predict_dataset, filename_dataset))
        self.predict_dataset = self.predict_dataset.batch(self.predict_batch_size, drop_remainder=True)
    
    def predict(self):
        for predict_data in self.predict_dataset:
            # predict_data = (zipped_dataset, color_info_dataset, filename_dataset, color_map)
            predicted = self.model.predict((predict_data[0][0], predict_data[0][1], predict_data[0][2]), verbose=1)
            
            predicted_current_bin_label = predicted[2]
            
            # prev_bin_label = predict_data[0][2]
            bin_color_map = predict_data[0][3][1]
            current_filenames = predict_data[1]
            
            self.post_processing(predicted_current_bin_label, bin_color_map, current_filenames)
