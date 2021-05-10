from typing import Callable, List, Tuple

import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from imagemodel.common.datasets.regularizer_helper import RegularizerInputHelper, RegularizerOutputHelper


class RTRegularizerInputHelper(RegularizerInputHelper):
    def get_dataset(self) -> tf.data.Dataset:
        pass
    
    def main_image_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def ref_image_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def ref_color_label_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def __input_regularizers(self, main_images: tf.Tensor, ref_images: tf.Tensor, ref_labels: tf.Tensor) -> \
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        result_main_images = main_images
        for f in self.main_image_regularizer_func():
            result_main_images = f(result_main_images)
        
        result_ref_images = ref_images
        for f in self.ref_image_regularizer_func():
            result_ref_images = f(result_ref_images)
        
        result_ref_labels = ref_labels
        for f in self.ref_color_label_regularizer_func():
            result_ref_labels = f(result_ref_labels)
        
        return result_main_images, result_ref_images, result_ref_labels
    
    def get_inputs(self) -> List[tf.data.Dataset]:
        dataset = self.get_dataset()
        dataset = dataset.map(self.__input_regularizers)
        return [dataset]


class RTRegularizerOutputHelper(RegularizerOutputHelper):
    def get_dataset(self) -> tf.data.Dataset:
        pass
    
    def main_bw_mask_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def ref_bw_mask_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def ref_color_label_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def main_color_label_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def __output_augments(
            self,
            main_bw_images: tf.Tensor,
            ref_bw_images: tf.Tensor,
            ref_label_images: tf.Tensor,
            main_label_images: tf.Tensor) -> \
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        result_main_bw_images = main_bw_images
        for f in self.main_bw_mask_regularizer_func():
            result_main_bw_images = f(result_main_bw_images)
        
        result_ref_bw_images = ref_bw_images
        for f in self.ref_bw_mask_regularizer_func():
            result_ref_bw_images = f(result_ref_bw_images)
        
        result_ref_label_images = ref_label_images
        for f in self.ref_color_label_regularizer_func():
            result_ref_label_images = f(result_ref_label_images)
        
        result_main_label_images = main_label_images
        for f in self.main_color_label_regularizer_func():
            result_main_label_images = f(result_main_label_images)
        
        return result_main_bw_images, result_ref_bw_images, result_ref_label_images, result_main_label_images
    
    def get_outputs(self) -> List[tf.data.Dataset]:
        dataset = self.get_dataset()
        dataset = dataset.map(self.__output_augments)
        return [dataset]


class BaseRTRegularizerInputHelper(RTRegularizerInputHelper):
    def __init__(self, datasets: List[tf.data.Dataset], height_width_tuple: Tuple[int, int]):
        self._datasets: List[tf.data.Dataset] = datasets
        self._height_width_tuple: Tuple[int, int] = height_width_tuple
    
    def get_dataset(self) -> tf.data.Dataset:
        return self._datasets[0]
    
    def main_image_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def _resize(img: tf.Tensor) -> tf.Tensor:
            return tf.image.resize(img, self._height_width_tuple)
        
        return [_resize]
    
    def ref_image_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def _resize(img: tf.Tensor) -> tf.Tensor:
            return tf.image.resize(img, self._height_width_tuple)
        
        return [_resize]
    
    def ref_color_label_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def _resize_nn(img: tf.Tensor) -> tf.Tensor:
            return tf.image.resize(img, self._height_width_tuple, method=ResizeMethod.NEAREST_NEIGHBOR)
        
        return [_resize_nn]


class BaseRTRegularizerOutputHelper(RTRegularizerOutputHelper):
    def __init__(self, datasets: List[tf.data.Dataset], height_width_tuple: Tuple[int, int]):
        self._datasets: List[tf.data.Dataset] = datasets
        self._height_width_tuple: Tuple[int, int] = height_width_tuple
    
    def get_dataset(self) -> tf.data.Dataset:
        return self._datasets[0]
    
    def main_bw_mask_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def _resize(img: tf.Tensor) -> tf.Tensor:
            return tf.image.resize(img, self._height_width_tuple)
        
        return [_resize]
    
    def ref_bw_mask_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def _resize(img: tf.Tensor) -> tf.Tensor:
            return tf.image.resize(img, self._height_width_tuple)
        
        return [_resize]
    
    def ref_color_label_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def _resize_nn(img: tf.Tensor) -> tf.Tensor:
            return tf.image.resize(img, self._height_width_tuple, method=ResizeMethod.NEAREST_NEIGHBOR)
        
        return [_resize_nn]
    
    def main_color_label_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def _resize_nn(img: tf.Tensor) -> tf.Tensor:
            return tf.image.resize(img, self._height_width_tuple, method=ResizeMethod.NEAREST_NEIGHBOR)
        
        return [_resize_nn]
