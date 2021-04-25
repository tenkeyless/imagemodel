from typing import List, Callable

import tensorflow as tf

from imagemodel.binary_segmentations.datasets.bs_augmenter import BSAugmenter
from imagemodel.binary_segmentations.datasets.bs_augmenter_helper import BSAugmenterInputHelper, BSAugmenterOutputHelper
from imagemodel.common.datasets.manipulator.manipulator import SupervisedManipulator


class FlipBSAugmenter(BSAugmenter):
    def __init__(self, manipulator: SupervisedManipulator):
        self._inout_helper = FlipBSAugmenterInOutHelper(
            input_datasets=manipulator.get_input_dataset(),
            output_datasets=manipulator.get_output_dataset(),
        )

    @property
    def input_helper(self) -> BSAugmenterInputHelper:
        return self._inout_helper

    @property
    def output_helper(self) -> BSAugmenterOutputHelper:
        return self._inout_helper


class FlipBSAugmenterInOutHelper(BSAugmenterInputHelper, BSAugmenterOutputHelper):
    def __init__(
            self,
            input_datasets: List[tf.data.Dataset],
            output_datasets: List[tf.data.Dataset],
    ):
        self._input_datasets: List[tf.data.Dataset] = input_datasets
        self._output_datasets: List[tf.data.Dataset] = output_datasets

    def get_image_dataset(self) -> tf.data.Dataset:
        return self._input_datasets[0]

    def image_augment_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        return [lambda img: tf.image.random_flip_left_right(img, 42)]

    def get_mask_dataset(self) -> tf.data.Dataset:
        return self._output_datasets[0]

    def mask_augment_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        return [lambda img: tf.image.random_flip_left_right(img, 42)]
