from typing import Callable, List

import tensorflow as tf
import tf_clahe

from imagemodel.reference_tracking.datasets.rt_preprocessor_helper import (
    BaseRTPreprocessorInputHelper,
    RTPreprocessorInputHelper
)


class ClaheRTPreprocessorInputHelper(BaseRTPreprocessorInputHelper, RTPreprocessorInputHelper):
    def main_image_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        return [lambda img: tf_clahe.clahe(img, tile_grid_size=[8, 8], clip_limit=2.0),
                # TODO: tf_clahe should specify the size for reshaping.
                lambda img: tf.reshape(img, (256, 256, 1)),
                lambda img: tf.cast(img, tf.float32) / 255.0]
    
    def ref_image_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        return [lambda img: tf_clahe.clahe(img, tile_grid_size=[8, 8], clip_limit=2.0),
                # TODO: tf_clahe should specify the size for reshaping.
                lambda img: tf.reshape(img, (256, 256, 1)),
                lambda img: tf.cast(img, tf.float32) / 255.0]
