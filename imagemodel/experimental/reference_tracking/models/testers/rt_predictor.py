from typing import Callable, Optional

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.distribute.tpu_strategy import TPUStrategy

from imagemodel.common.predictor import Predictor


class RTPredictor(Predictor):
    def __init__(
            self,
            model: Model,
            predict_dataset: tf.data.Dataset,
            predict_dataset_description: str,
            predict_filename_dataset: Optional[tf.data.Dataset],
            predict_batch_size: int,
            post_processing: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], None],
            strategy_optional: Optional[TPUStrategy] = None):
        super().__init__(model, predict_dataset, predict_dataset_description, predict_batch_size, strategy_optional)
        self.post_processing: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], None] = post_processing
        
        self.predict_dataset_num: int = len(self.predict_dataset)
        self.predict_dataset = tf.data.Dataset.zip((self.predict_dataset, predict_filename_dataset))
        self.predict_dataset = self.predict_dataset.batch(self.predict_batch_size, drop_remainder=True)
    
    def predict(self):
        for predict_data in self.predict_dataset:
            # predict_data = ((main_image, ref_image, (ref_label_with_bin, bin_color_map)), filenames)
            # predicted = (main_bw_label, ref_bw_label, predicted_main_label_with_bin)
            predicted = self.model.predict((predict_data[0][0], predict_data[0][1], predict_data[0][2][0]), verbose=1)
            predicted_current_bin_label = predicted[2]
            
            bin_color_map = predict_data[0][2][1]
            current_filenames = predict_data[1]
            print(current_filenames)
            
            self.post_processing(predicted_current_bin_label, bin_color_map, current_filenames)
