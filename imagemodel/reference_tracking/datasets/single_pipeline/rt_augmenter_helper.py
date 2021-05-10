from typing import Callable, List, Tuple

import tensorflow as tf

from imagemodel.common.datasets.augmenter_helper import AugmenterInputHelper, AugmenterOutputHelper


class RTAugmenterInputHelper(AugmenterInputHelper):
    def get_dataset(self) -> tf.data.Dataset:
        pass
    
    def main_image_augment_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def ref_image_augment_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def ref_color_label_augment_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def __input_augments(self, main_images: tf.Tensor, ref_images: tf.Tensor, ref_labels: tf.Tensor) -> \
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        result_main_images = main_images
        for f in self.main_image_augment_func():
            result_main_images = f(result_main_images)
        
        result_ref_images = ref_images
        for f in self.ref_image_augment_func():
            result_ref_images = f(result_ref_images)
        
        result_ref_labels = ref_labels
        for f in self.ref_color_label_augment_func():
            result_ref_labels = f(result_ref_labels)
        
        return result_main_images, result_ref_images, result_ref_labels
    
    def get_inputs(self) -> List[tf.data.Dataset]:
        dataset = self.get_dataset()
        dataset = dataset.map(self.__input_augments)
        return [dataset]


class RTAugmenterOutputHelper(AugmenterOutputHelper):
    def get_dataset(self) -> tf.data.Dataset:
        pass
    
    def main_bw_mask_augment_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def ref_bw_mask_augment_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def main_color_label_augment_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def ref_color_label_augment_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass
    
    def __output_augments(
            self,
            main_bw_images: tf.Tensor,
            ref_bw_images: tf.Tensor,
            ref_label_images: tf.Tensor,
            main_label_images: tf.Tensor) -> \
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        result_main_bw_images = main_bw_images
        for f in self.main_bw_mask_augment_func():
            result_main_bw_images = f(result_main_bw_images)
        
        result_ref_bw_images = ref_bw_images
        for f in self.ref_bw_mask_augment_func():
            result_ref_bw_images = f(result_ref_bw_images)
        
        result_ref_label_images = ref_label_images
        for f in self.main_color_label_augment_func():
            result_ref_label_images = f(result_ref_label_images)
        
        result_main_label_images = main_label_images
        for f in self.ref_color_label_augment_func():
            result_main_label_images = f(result_main_label_images)
        
        return result_main_bw_images, result_ref_bw_images, result_ref_label_images, result_main_label_images
    
    def get_outputs(self) -> List[tf.data.Dataset]:
        dataset = self.get_dataset()
        dataset = dataset.map(self.__output_augments)
        return [dataset]


class BaseRTAugmenterInputHelper(RTAugmenterInputHelper):
    def __init__(self, datasets: List[tf.data.Dataset]):
        self._datasets: List[tf.data.Dataset] = datasets
    
    def get_dataset(self) -> tf.data.Dataset:
        return self._datasets[0]
    
    def main_image_augment_func(self) -> List[Tuple[Callable[[tf.Tensor], tf.Tensor], bool]]:
        return []
    
    def ref_image_augment_func(self) -> List[Tuple[Callable[[tf.Tensor], tf.Tensor], bool]]:
        return []
    
    def ref_color_label_augment_func(self) -> List[Tuple[Callable[[tf.Tensor], tf.Tensor], bool]]:
        return []


class BaseRTAugmenterOutputHelper(RTAugmenterOutputHelper):
    def __init__(self, datasets: List[tf.data.Dataset]):
        self._datasets: List[tf.data.Dataset] = datasets
    
    def get_dataset(self) -> tf.data.Dataset:
        return self._datasets[0]
    
    def main_bw_mask_augment_func(self) -> List[Tuple[Callable[[tf.Tensor], tf.Tensor], bool]]:
        return []
    
    def ref_bw_mask_augment_func(self) -> List[Tuple[Callable[[tf.Tensor], tf.Tensor], bool]]:
        return []
    
    def ref_color_label_augment_func(self) -> List[Tuple[Callable[[tf.Tensor], tf.Tensor], bool]]:
        return []
    
    def main_color_label_augment_func(self) -> List[Tuple[Callable[[tf.Tensor], tf.Tensor], bool]]:
        return []
