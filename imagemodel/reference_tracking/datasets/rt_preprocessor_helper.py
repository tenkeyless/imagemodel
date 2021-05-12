from typing import Callable, List

import tensorflow as tf
from image_keras.tf.utils.images import (
    tf_change_order,
    tf_generate_random_color_map,
    tf_image_detach_with_id_color_list,
    tf_shrink3D
)

from imagemodel.common.datasets.preprocessor_helper import PreprocessorInputHelper, PreprocessorOutputHelper


def apply_funcs_to(dataset: tf.data.Dataset, functions: List[Callable[[tf.Tensor], tf.Tensor]]) -> tf.data.Dataset:
    _dataset = dataset
    for f in functions:
        _dataset = _dataset.map(f, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return _dataset


class RTPreprocessorInputHelper(PreprocessorInputHelper):
    def get_main_image_dataset(self) -> tf.data.Dataset:
        pass
    
    def main_image_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def get_ref_image_dataset(self) -> tf.data.Dataset:
        pass
    
    def ref_image_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def get_ref_color_label_dataset(self) -> tf.data.Dataset:
        pass
    
    # def get_ref_img_color_list_dataset(self) -> tf.data.Dataset:
    #     pass
    
    # def ref_random_color_info_map(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    #     pass
    
    def ref_color_bin_label_preprocess_func(self) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
        pass
    
    def get_inputs(self) -> List[tf.data.Dataset]:
        main_image_dataset = apply_funcs_to(self.get_main_image_dataset(), self.main_image_preprocess_func())
        ref_image_dataset = apply_funcs_to(self.get_ref_image_dataset(), self.ref_image_preprocess_func())
        ref_color_bin_label_dataset = self.ref_color_bin_label_preprocess_func()(self.get_ref_color_label_dataset())
        return [main_image_dataset, ref_image_dataset, ref_color_bin_label_dataset]


class RTPreprocessorOutputHelper(PreprocessorOutputHelper):
    def get_main_bw_mask_dataset(self) -> tf.data.Dataset:
        pass
    
    def main_bw_mask_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def get_ref_bw_mask_dataset(self) -> tf.data.Dataset:
        pass
    
    def ref_bw_mask_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def get_ref_color_label_dataset(self) -> tf.data.Dataset:
        pass
    
    def get_main_color_label_dataset(self) -> tf.data.Dataset:
        pass
    
    def main_color_label_preprocess_func(self) -> Callable[[tf.data.Dataset, tf.data.Dataset], tf.data.Dataset]:
        pass
    
    def get_outputs(self) -> List[tf.data.Dataset]:
        main_bw_mask_dataset = apply_funcs_to(self.get_main_bw_mask_dataset(), self.main_bw_mask_preprocess_func())
        ref_bw_mask_dataset = apply_funcs_to(self.get_ref_bw_mask_dataset(), self.ref_bw_mask_preprocess_func())
        main_color_bin_label_dataset = self.main_color_label_preprocess_func()(
                self.get_ref_color_label_dataset(),
                self.get_main_color_label_dataset())
        return [main_bw_mask_dataset, ref_bw_mask_dataset, main_color_bin_label_dataset]


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
    
    def get_main_image_dataset(self) -> tf.data.Dataset:
        return self._datasets[0]
    
    def main_image_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def _cast_norm(img: tf.Tensor) -> tf.Tensor:
            return tf.cast(img, tf.float32) / 255.0
        
        return [_cast_norm]
    
    def get_ref_image_dataset(self) -> tf.data.Dataset:
        return self._datasets[1]
    
    def ref_image_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def _cast_norm(img: tf.Tensor) -> tf.Tensor:
            return tf.cast(img, tf.float32) / 255.0
        
        return [_cast_norm]
    
    def get_ref_color_label_dataset(self) -> tf.data.Dataset:
        return self._datasets[2]
    
    def ref_color_bin_label_preprocess_func(self) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
        def _generate_ref_color_bin(dataset: tf.data.Dataset) -> tf.data.Dataset:
            ref_img_color_list_dataset = dataset.map(
                    lambda img: (img, tf_color_to_random_map(img, self.bin_size, 1)),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            ref_color_bin_separated_dataset = ref_img_color_list_dataset.map(
                    lambda img, color_info: tf_input_ref_label_preprocessing_function(img, color_info, self.bin_size),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            return ref_color_bin_separated_dataset
        
        return _generate_ref_color_bin


class BaseRTPreprocessorOutputHelper(RTPreprocessorOutputHelper):
    def __init__(self, datasets: List[tf.data.Dataset], bin_size: int):
        self._datasets: List[tf.data.Dataset] = datasets
        self.bin_size: int = bin_size
    
    def get_main_bw_mask_dataset(self) -> tf.data.Dataset:
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
    
    def get_ref_color_label_dataset(self) -> tf.data.Dataset:
        return self._datasets[2]
    
    def get_main_color_label_dataset(self) -> tf.data.Dataset:
        return self._datasets[3]
    
    def main_color_label_preprocess_func(self) -> Callable[[tf.data.Dataset, tf.data.Dataset], tf.data.Dataset]:
        def _generate_main_color_bin(ref_dataset: tf.data.Dataset, main_dataset: tf.data.Dataset) -> tf.data.Dataset:
            ref_img_color_list_dataset = ref_dataset.map(
                    lambda img: tf_color_to_random_map(img, self.bin_size, 1),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            zipped_dataset = tf.data.Dataset.zip((main_dataset, ref_img_color_list_dataset))
            main_color_bin_separated_dataset = zipped_dataset.map(
                    lambda img, color_info: tf_output_label_processing(img, color_info, self.bin_size),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
            return main_color_bin_separated_dataset
        
        return _generate_main_color_bin
