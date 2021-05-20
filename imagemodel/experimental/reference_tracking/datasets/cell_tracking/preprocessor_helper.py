from typing import Callable, List, Tuple

import tensorflow as tf
import tf_clahe

from imagemodel.experimental.reference_tracking.datasets.rt_preprocessor_helper import (
    BaseRTPreprocessorInputHelper,
    BaseRTPreprocessorOutputHelper,
    RTPreprocessorInputHelper,
    RTPreprocessorOutputHelper,
    apply_funcs_to,
    tf_color_to_random_map,
    tf_input_ref_label_preprocessing_function
)


class RTCellTrackingPreprocessorInputHelper(BaseRTPreprocessorInputHelper, RTPreprocessorInputHelper):
    def get_inputs(self) -> List[tf.data.Dataset]:
        main_image_dataset = apply_funcs_to(self.get_main_image_dataset(), self.main_image_preprocess_func())
        ref_image_dataset = apply_funcs_to(self.get_ref_image_dataset(), self.ref_image_preprocess_func())
        return [main_image_dataset,
                ref_image_dataset,
                self.get_ref_color_label_dataset()]


class RTCellTrackingPreprocessorOutputHelper(BaseRTPreprocessorOutputHelper, RTPreprocessorOutputHelper):
    def get_outputs(self) -> List[tf.data.Dataset]:
        main_bw_mask_dataset = apply_funcs_to(self.get_main_bw_mask_dataset(), self.main_bw_mask_preprocess_func())
        ref_bw_mask_dataset = apply_funcs_to(self.get_ref_bw_mask_dataset(), self.ref_bw_mask_preprocess_func())
        return [main_bw_mask_dataset, ref_bw_mask_dataset, self.get_main_color_label_dataset()]


class ClaheRTPreprocessorInputHelper(RTCellTrackingPreprocessorInputHelper, RTPreprocessorInputHelper):
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


class ClaheRTPreprocessorPredictInputHelper(RTCellTrackingPreprocessorInputHelper):
    def __init__(self, datasets: List[tf.data.Dataset], bin_size: int, fill_with: Tuple[int, int, int]):
        super().__init__(datasets=datasets, bin_size=bin_size)
        self.fill_with = fill_with
    
    def get_inputs(self) -> List[tf.data.Dataset]:
        main_image_dataset = apply_funcs_to(self.get_main_image_dataset(), self.main_image_preprocess_func())
        ref_image_dataset = apply_funcs_to(self.get_ref_image_dataset(), self.ref_image_preprocess_func())
        
        def generate_filled_color_map(color_map):
            @tf.autograph.experimental.do_not_convert
            def _color_fill(_color, _color_index, fill_with: Tuple[int, int, int]):
                fill_empty_with = tf.repeat([fill_with], repeats=tf.shape(_color_index)[-1], axis=0)
                fill_empty_with = tf.cast(fill_empty_with, tf.float32)
                filled_bin = tf.concat([_color, fill_empty_with], axis=0)
                filled_bin = filled_bin[:tf.shape(_color_index)[-1], :]
                result = tf.gather(filled_bin, tf.cast(_color_index, tf.int32), axis=0)
                return result
            
            return _color_fill(color_map[1], color_map[0], self.fill_with)
        
        def label_to_separate_bin_map(img, color_map):
            return tf_input_ref_label_preprocessing_function(img, color_map, self.bin_size)
        
        ref_color_bin_label__color_map_dataset = self.get_ref_color_label_dataset().map(
                lambda img: (img, tf_color_to_random_map(img, self.bin_size, 1)),
                num_parallel_calls=tf.data.experimental.AUTOTUNE).map(
                lambda img, color_map: (
                    label_to_separate_bin_map(img, color_map), generate_filled_color_map(color_map)),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        return [main_image_dataset, ref_image_dataset, ref_color_bin_label__color_map_dataset]
