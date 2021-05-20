from typing import Callable, List, Tuple

import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from imagemodel.common.datasets.regularizer_helper import RegularizerInputHelper, RegularizerOutputHelper


def apply_funcs_to(dataset: tf.data.Dataset, functions: List[Callable[[tf.Tensor], tf.Tensor]]) -> tf.data.Dataset:
    _dataset = dataset
    for f in functions:
        _dataset = _dataset.map(f, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return _dataset


class RTRegularizerInputHelper(RegularizerInputHelper):
    def get_main_image_dataset(self) -> tf.data.Dataset:
        pass
    
    def main_image_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def get_ref_image_dataset(self) -> tf.data.Dataset:
        pass
    
    def ref_image_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def get_ref_color_label_dataset(self) -> tf.data.Dataset:
        pass
    
    def ref_color_label_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def get_inputs(self) -> List[tf.data.Dataset]:
        main_image_dataset = apply_funcs_to(self.get_main_image_dataset(), self.main_image_regularizer_func())
        ref_image_dataset = apply_funcs_to(self.get_ref_image_dataset(), self.ref_image_regularizer_func())
        ref_color_label_dataset = apply_funcs_to(
                self.get_ref_color_label_dataset(),
                self.ref_color_label_regularizer_func())
        return [main_image_dataset, ref_image_dataset, ref_color_label_dataset]


class RTRegularizerOutputHelper(RegularizerOutputHelper):
    def get_main_bw_mask_dataset(self) -> tf.data.Dataset:
        pass
    
    def main_bw_mask_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def get_ref_bw_mask_dataset(self) -> tf.data.Dataset:
        pass
    
    def ref_bw_mask_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def get_main_color_label_dataset(self) -> tf.data.Dataset:
        pass
    
    def main_color_label_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def get_outputs(self) -> List[tf.data.Dataset]:
        main_bw_mask_dataset = apply_funcs_to(self.get_main_bw_mask_dataset(), self.main_bw_mask_regularizer_func())
        ref_bw_mask_dataset = apply_funcs_to(self.get_ref_bw_mask_dataset(), self.ref_bw_mask_regularizer_func())
        main_color_label_dataset = apply_funcs_to(
                self.get_main_color_label_dataset(),
                self.main_color_label_regularizer_func())
        return [main_bw_mask_dataset, ref_bw_mask_dataset, main_color_label_dataset]


class BaseRTRegularizerInputHelper(RTRegularizerInputHelper):
    def __init__(self, datasets: List[tf.data.Dataset], height_width_tuple: Tuple[int, int]):
        self._datasets: List[tf.data.Dataset] = datasets
        self._height_width_tuple: Tuple[int, int] = height_width_tuple
    
    def get_main_image_dataset(self) -> tf.data.Dataset:
        return self._datasets[0]
    
    def main_image_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def _resize(img: tf.Tensor) -> tf.Tensor:
            return tf.image.resize(img, self._height_width_tuple)
        
        return [_resize]
    
    def get_ref_image_dataset(self) -> tf.data.Dataset:
        return self._datasets[1]
    
    def ref_image_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def _resize(img: tf.Tensor) -> tf.Tensor:
            return tf.image.resize(img, self._height_width_tuple)
        
        return [_resize]
    
    def get_ref_color_label_dataset(self) -> tf.data.Dataset:
        return self._datasets[2]
    
    def ref_color_label_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def _resize_nn(img: tf.Tensor) -> tf.Tensor:
            return tf.image.resize(img, self._height_width_tuple, method=ResizeMethod.NEAREST_NEIGHBOR)
        
        return [_resize_nn]


class BaseRTRegularizerOutputHelper(RTRegularizerOutputHelper):
    def __init__(self, datasets: List[tf.data.Dataset], height_width_tuple: Tuple[int, int]):
        self._datasets: List[tf.data.Dataset] = datasets
        self._height_width_tuple: Tuple[int, int] = height_width_tuple
    
    def get_main_bw_mask_dataset(self) -> tf.data.Dataset:
        return self._datasets[0]
    
    def main_bw_mask_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def _resize(img: tf.Tensor) -> tf.Tensor:
            return tf.image.resize(img, self._height_width_tuple)
        
        return [_resize]
    
    def get_ref_bw_mask_dataset(self) -> tf.data.Dataset:
        return self._datasets[1]
    
    def ref_bw_mask_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def _resize(img: tf.Tensor) -> tf.Tensor:
            return tf.image.resize(img, self._height_width_tuple)
        
        return [_resize]
    
    def get_main_color_label_dataset(self) -> tf.data.Dataset:
        return self._datasets[2]
    
    def main_color_label_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def _resize_nn(img: tf.Tensor) -> tf.Tensor:
            return tf.image.resize(img, self._height_width_tuple, method=ResizeMethod.NEAREST_NEIGHBOR)
        
        return [_resize_nn]
