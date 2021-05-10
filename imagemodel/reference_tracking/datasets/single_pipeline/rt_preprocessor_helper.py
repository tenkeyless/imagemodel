from typing import Callable, List, Tuple

import tensorflow as tf
from image_keras.tf.utils.images import (
    tf_change_order,
    tf_generate_random_color_map,
    tf_image_detach_with_id_color_list,
    tf_shrink3D
)

from imagemodel.common.datasets.preprocessor_helper import PreprocessorInputHelper, PreprocessorOutputHelper


class RTPreprocessorInputHelper(PreprocessorInputHelper):
    def get_dataset(self) -> tf.data.Dataset:
        pass
    
    def main_image_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def ref_image_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def ref_color_bin_label_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def __input_preprocess(self, main_images: tf.Tensor, ref_images: tf.Tensor, ref_labels: tf.Tensor) -> \
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        result_main_images = main_images
        for f in self.main_image_preprocess_func():
            result_main_images = f(result_main_images)
        
        result_ref_images = ref_images
        for f in self.ref_image_preprocess_func():
            result_ref_images = f(result_ref_images)
        
        result_ref_labels = ref_labels
        for f in self.ref_color_bin_label_preprocess_func():
            result_ref_labels = f(result_ref_labels)
        
        return result_main_images, result_ref_images, result_ref_labels
    
    def get_inputs(self) -> List[tf.data.Dataset]:
        dataset = self.get_dataset()
        dataset = dataset.map(self.__input_preprocess)
        return [dataset]


class RTPreprocessorOutputHelper(PreprocessorOutputHelper):
    def get_dataset(self) -> tf.data.Dataset:
        pass
    
    def main_bw_mask_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def ref_bw_mask_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def main_color_label_preprocess_func(self) -> List[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
        pass
    
    def __output_augments(
            self,
            main_bw_images: tf.Tensor,
            ref_bw_images: tf.Tensor,
            ref_label_images: tf.Tensor,
            main_label_images: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        result_main_bw_images = main_bw_images
        for f in self.main_bw_mask_preprocess_func():
            result_main_bw_images = f(result_main_bw_images)
        
        result_ref_bw_images = ref_bw_images
        for f in self.ref_bw_mask_preprocess_func():
            result_ref_bw_images = f(result_ref_bw_images)
        
        result_new_main_label_images = None
        result_ref_label_images = ref_label_images
        result_main_label_images = main_label_images
        for f in self.main_color_label_preprocess_func():
            result_new_main_label_images = f(result_ref_label_images, result_main_label_images)
        
        return result_main_bw_images, result_ref_bw_images, result_new_main_label_images
    
    def get_outputs(self) -> List[tf.data.Dataset]:
        dataset = self.get_dataset()
        dataset = dataset.map(self.__output_augments)
        return [dataset]


@tf.autograph.experimental.do_not_convert
def tf_color_to_random_map(ref_label_img, bin_size, exclude_first=1):
    return tf_generate_random_color_map(
            ref_label_img,
            bin_size=bin_size,
            shuffle_exclude_first=exclude_first,
            seed=42)


def tf_image_detach_with_id_color_probability_list(
        color_img,
        id_color_list,
        bin_num: int,
        resize_by_power_of_two: int = 0):
    result = tf_image_detach_with_id_color_list(color_img, id_color_list, bin_num, 1.0)
    ratio = 2 ** resize_by_power_of_two
    result2 = tf_shrink3D(result, tf.shape(result)[-3] // ratio, tf.shape(result)[-2] // ratio, bin_num)
    result2 = tf.divide(result2, ratio ** 2)
    return result2


@tf.autograph.experimental.do_not_convert
def tf_input_ref_label_preprocessing_function(label, color_info, bin_size):
    result = tf_image_detach_with_id_color_probability_list(label, color_info, bin_size, 0)
    result = tf.reshape(result, (256 // (2 ** 0), 256 // (2 ** 0), bin_size))
    result = tf_change_order(result, color_info[0])
    result = tf.squeeze(result)
    return result


@tf.autograph.experimental.do_not_convert
def tf_output_label_processing(label, color_info, bin_size):
    result = tf_image_detach_with_id_color_probability_list(label, color_info, bin_size, 0)
    result = tf.reshape(result, (256 // (2 ** 0), 256 // (2 ** 0), bin_size))
    result = tf_change_order(result, color_info[0])
    return result


class BaseRTPreprocessorInputHelper(RTPreprocessorInputHelper):
    def __init__(self, datasets: List[tf.data.Dataset], bin_size: int):
        self._datasets: List[tf.data.Dataset] = datasets
        self.bin_size: int = bin_size
    
    def get_dataset(self) -> tf.data.Dataset:
        return self._datasets[0]
    
    def main_image_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def _cast_norm(img: tf.Tensor) -> tf.Tensor:
            return tf.cast(img, tf.float32) / 255.0
        
        return [_cast_norm]
    
    def ref_image_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def _cast_norm(img: tf.Tensor) -> tf.Tensor:
            return tf.cast(img, tf.float32) / 255.0
        
        return [_cast_norm]
    
    def ref_color_bin_label_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def _generate_ref_color_bin(image: tf.Tensor) -> tf.Tensor:
            ref_img_color_list = tf_color_to_random_map(image, self.bin_size, 1)
            ref_color_bin_separated = tf_input_ref_label_preprocessing_function(
                    image,
                    ref_img_color_list,
                    self.bin_size)
            return ref_color_bin_separated
        
        return [_generate_ref_color_bin]


class BaseRTPreprocessorOutputHelper(RTPreprocessorOutputHelper):
    def __init__(self, datasets: List[tf.data.Dataset], bin_size: int):
        self._datasets: List[tf.data.Dataset] = datasets
        self.bin_size: int = bin_size
    
    def get_dataset(self) -> tf.data.Dataset:
        return self._datasets[0]
    
    def main_bw_mask_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def _greater_cast(img: tf.Tensor) -> tf.Tensor:
            return tf.cast(tf.greater(tf.cast(img, tf.float32), 0.5), tf.float32)
        
        return [_greater_cast]
    
    def get_ref_bw_mask_dataset(self) -> tf.data.Dataset:
        return self._datasets[1]
    
    def ref_bw_mask_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def _greater_cast(img: tf.Tensor) -> tf.Tensor:
            return tf.cast(tf.greater(tf.cast(img, tf.float32), 0.5), tf.float32)
        
        return [_greater_cast]
    
    def main_color_label_preprocess_func(self) -> List[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def _generate_main_color_bin(ref_image: tf.Tensor, main_image: tf.Tensor) -> tf.Tensor:
            ref_img_color_list = tf_color_to_random_map(ref_image, self.bin_size, 1)
            main_color_bin_separated = tf_output_label_processing(
                    main_image,
                    ref_img_color_list,
                    self.bin_size)
            return main_color_bin_separated
        
        return [_generate_main_color_bin]
