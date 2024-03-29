from typing import Callable, List

import tensorflow as tf

from imagemodel.binary_segmentations.datasets.bs_augmenter_helper import BSAugmenterInputHelper, BSAugmenterOutputHelper


def __img_augment_func(img: tf.Tensor, seed: int = 42) -> tf.Tensor:
    augmented_img = tf.image.random_flip_left_right(img, seed)
    augmented_img = tf.image.random_flip_up_down(augmented_img, seed)
    return augmented_img


class BSOxfordIIITPetAugmenterInOutHelper(BSAugmenterInputHelper, BSAugmenterOutputHelper):
    def __init__(self, input_datasets: List[tf.data.Dataset], output_datasets: List[tf.data.Dataset]):
        self._input_datasets: List[tf.data.Dataset] = input_datasets
        self._output_datasets: List[tf.data.Dataset] = output_datasets

    def get_image_dataset(self) -> tf.data.Dataset:
        return self._input_datasets[0]

    def get_mask_dataset(self) -> tf.data.Dataset:
        return self._output_datasets[0]

    def image_augment_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        return [lambda img: __img_augment_func(img, 42)]

    def mask_augment_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        return [lambda img: __img_augment_func(img, 42)]


class BSOxfordIIITPetAugmenterInputHelper(BSAugmenterInputHelper):
    def __init__(self, datasets: List[tf.data.Dataset]):
        self._datasets: List[tf.data.Dataset] = datasets

    def get_image_dataset(self) -> tf.data.Dataset:
        return self._datasets[0]

    def image_augment_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        return [lambda img: tf.image.random_flip_left_right(img, 42),
                lambda img: tf.image.random_flip_up_down(img, 42)]


class BSOxfordIIITPetAugmenterOutputHelper(BSAugmenterOutputHelper):
    def __init__(self, datasets: List[tf.data.Dataset]):
        self._datasets: List[tf.data.Dataset] = datasets

    def get_mask_dataset(self) -> tf.data.Dataset:
        return self._datasets[0]

    def mask_augment_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        return [lambda img: tf.image.random_flip_left_right(img, 42),
                lambda img: tf.image.random_flip_up_down(img, 42)]
