from typing import Callable, List

import tensorflow as tf
import tf_clahe

from imagemodel.binary_segmentations.datasets.bs_preprocessor_helper import (
    BSPreprocessorInputHelper,
    BSPreprocessorOutputHelper, BaseBSPreprocessorInOutHelper
)


class RTPreprocessorInOutHelper(
        BaseBSPreprocessorInOutHelper,
        BSPreprocessorInputHelper,
        BSPreprocessorOutputHelper):
    def __init__(self, input_datasets: List[tf.data.Dataset], output_datasets: List[tf.data.Dataset]):
        super().__init__(input_datasets=input_datasets, output_datasets=output_datasets)
    
    def image_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        return [lambda img: tf_clahe.clahe(img, tile_grid_size=[8, 8], clip_limit=2.0),
                # TODO: tf_clahe should specify the size for reshaping.
                lambda img: tf.reshape(img, (256, 256, 1)),
                lambda img: tf.cast(img, tf.float32) / 255.0]
