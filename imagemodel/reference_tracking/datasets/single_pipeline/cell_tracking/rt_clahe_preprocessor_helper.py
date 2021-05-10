from typing import Callable, List

import tensorflow as tf
import tf_clahe

from imagemodel.reference_tracking.datasets.single_pipeline.rt_preprocessor_helper import (
    BaseRTPreprocessorInputHelper, RTPreprocessorInputHelper
)


class ClaheRTPreprocessorInputHelper(BaseRTPreprocessorInputHelper, RTPreprocessorInputHelper):
    def main_image_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def _clahe(img: tf.Tensor) -> tf.Tensor:
            return tf_clahe.clahe(img, tile_grid_size=[8, 8], clip_limit=2.0)
        
        @tf.autograph.experimental.do_not_convert
        def _reshape(img: tf.Tensor) -> tf.Tensor:
            return tf.reshape(img, (256, 256, 1))
        
        @tf.autograph.experimental.do_not_convert
        def _cast_norm(img: tf.Tensor) -> tf.Tensor:
            return tf.cast(img, tf.float32) / 255.0
        
        return [_clahe, _reshape, _cast_norm]
    
    def ref_image_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def _clahe(img: tf.Tensor) -> tf.Tensor:
            return tf_clahe.clahe(img, tile_grid_size=[8, 8], clip_limit=2.0)
        
        @tf.autograph.experimental.do_not_convert
        def _reshape(img: tf.Tensor) -> tf.Tensor:
            return tf.reshape(img, (256, 256, 1))
        
        @tf.autograph.experimental.do_not_convert
        def _cast_norm(img: tf.Tensor) -> tf.Tensor:
            return tf.cast(img, tf.float32) / 255.0
        
        return [_clahe, _reshape, _cast_norm]
