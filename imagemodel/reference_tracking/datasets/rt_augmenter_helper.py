from typing import Callable, List, Tuple

import tensorflow as tf

from imagemodel.common.datasets.augmenter_helper import AugmenterInputHelper, AugmenterOutputHelper


def apply_funcs_to(
        dataset: tf.data.Dataset,
        function_parallels: List[Tuple[Callable[[tf.Tensor], tf.Tensor], bool]]) -> tf.data.Dataset:
    _dataset = dataset
    for f_p in function_parallels:
        _dataset = _dataset.map(f_p, num_parallel_calls=tf.data.experimental.AUTOTUNE) if f_p[1] else _dataset.map(f_p)
    return _dataset


class RTAugmenterInputHelper(AugmenterInputHelper):
    def get_main_image_dataset(self) -> tf.data.Dataset:
        pass
    
    def main_image_augment_func(self) -> List[Tuple[Callable[[tf.Tensor], tf.Tensor], bool]]:
        pass
    
    def get_ref_image_dataset(self) -> tf.data.Dataset:
        pass
    
    def ref_image_augment_func(self) -> List[Tuple[Callable[[tf.Tensor], tf.Tensor], bool]]:
        pass
    
    def get_ref_color_label_dataset(self) -> tf.data.Dataset:
        pass
    
    def ref_color_label_augment_func(self) -> List[Tuple[Callable[[tf.Tensor], tf.Tensor], bool]]:
        pass
    
    def get_inputs(self) -> List[tf.data.Dataset]:
        main_image_dataset = apply_funcs_to(self.get_main_image_dataset(), self.main_image_augment_func())
        ref_image_dataset = apply_funcs_to(self.get_ref_image_dataset(), self.ref_image_augment_func())
        ref_color_label_dataset = apply_funcs_to(
                self.get_ref_color_label_dataset(),
                self.ref_color_label_augment_func())
        return [main_image_dataset, ref_image_dataset, ref_color_label_dataset]


class RTAugmenterOutputHelper(AugmenterOutputHelper):
    def get_main_bw_mask_dataset(self) -> tf.data.Dataset:
        pass
    
    def main_bw_mask_augment_func(self) -> List[Tuple[Callable[[tf.Tensor], tf.Tensor], bool]]:
        pass
    
    def get_ref_bw_mask_dataset(self) -> tf.data.Dataset:
        pass
    
    def ref_bw_mask_augment_func(self) -> List[Tuple[Callable[[tf.Tensor], tf.Tensor], bool]]:
        pass
    
    def get_main_color_label_dataset(self) -> tf.data.Dataset:
        pass
    
    def main_color_label_augment_func(self) -> List[Tuple[Callable[[tf.Tensor], tf.Tensor], bool]]:
        pass
    
    def get_ref_color_label_dataset(self) -> tf.data.Dataset:
        pass
    
    def ref_color_label_augment_func(self) -> List[Tuple[Callable[[tf.Tensor], tf.Tensor], bool]]:
        pass
    
    def get_outputs(self) -> List[tf.data.Dataset]:
        main_bw_mask_dataset = apply_funcs_to(self.get_main_bw_mask_dataset(), self.main_bw_mask_augment_func())
        ref_bw_mask_dataset = apply_funcs_to(self.get_ref_bw_mask_dataset(), self.ref_bw_mask_augment_func())
        ref_color_label_dataset = apply_funcs_to(
                self.get_ref_color_label_dataset(),
                self.ref_color_label_augment_func())
        main_color_label_dataset = apply_funcs_to(
                self.get_main_color_label_dataset(),
                self.main_color_label_augment_func())
        return [main_bw_mask_dataset, ref_bw_mask_dataset, ref_color_label_dataset, main_color_label_dataset]


class BaseRTAugmenterInputHelper(RTAugmenterInputHelper):
    def __init__(self, datasets: List[tf.data.Dataset]):
        self._datasets: List[tf.data.Dataset] = datasets
    
    def get_main_image_dataset(self) -> tf.data.Dataset:
        return self._datasets[0]
    
    def get_ref_image_dataset(self) -> tf.data.Dataset:
        return self._datasets[1]
    
    def get_ref_color_label_dataset(self) -> tf.data.Dataset:
        return self._datasets[2]
    
    def main_image_augment_func(self) -> List[Tuple[Callable[[tf.Tensor], tf.Tensor], bool]]:
        return []
    
    def ref_image_augment_func(self) -> List[Tuple[Callable[[tf.Tensor], tf.Tensor], bool]]:
        return []
    
    def ref_color_label_augment_func(self) -> List[Tuple[Callable[[tf.Tensor], tf.Tensor], bool]]:
        return []


class BaseRTAugmenterOutputHelper(RTAugmenterOutputHelper):
    def __init__(self, datasets: List[tf.data.Dataset]):
        self._datasets: List[tf.data.Dataset] = datasets
    
    def get_main_bw_mask_dataset(self) -> tf.data.Dataset:
        return self._datasets[0]
    
    def get_ref_bw_mask_dataset(self) -> tf.data.Dataset:
        return self._datasets[1]
    
    def get_ref_color_label_dataset(self) -> tf.data.Dataset:
        return self._datasets[2]
    
    def get_main_color_label_dataset(self) -> tf.data.Dataset:
        return self._datasets[3]
    
    def main_bw_mask_augment_func(self) -> List[Tuple[Callable[[tf.Tensor], tf.Tensor], bool]]:
        return []
    
    def ref_bw_mask_augment_func(self) -> List[Tuple[Callable[[tf.Tensor], tf.Tensor], bool]]:
        return []
    
    def ref_color_label_augment_func(self) -> List[Tuple[Callable[[tf.Tensor], tf.Tensor], bool]]:
        return []
    
    def main_color_label_augment_func(self) -> List[Tuple[Callable[[tf.Tensor], tf.Tensor], bool]]:
        return []
