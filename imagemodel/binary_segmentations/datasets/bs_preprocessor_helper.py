from typing import Callable, List

import tensorflow as tf
from imagemodel.common.datasets.manipulator.helper import (
    ManipulatorInputHelper,
    ManipulatorOutputHelper,
)


class BSPreprocessorInputHelper(ManipulatorInputHelper):
    """
    <Interface>

    Methods to get dataset for binary segmentation.
    In this binary segmentation, we need raw image and corresponding mask.

    Methods
    -------
    get_image() -> tf.data.Dataset
        Image dataset. It returns `tf.data.Dataset`.
        `height`, `width` and 0~255 range of tf.uint8.
    """

    def get_image_dataset(self) -> tf.data.Dataset:
        """
        Image dataset.
        `height`, `width` and 0~255 range of tf.uint8.

        Returns
        -------
        tf.data.Dataset
            Dataset
        """
        pass

    def image_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass

    def get_inputs(self) -> List[tf.data.Dataset]:
        image_dataset = self.get_image_dataset()
        for f in self.image_preprocess_func():
            image_dataset = image_dataset.map(f)
        return [image_dataset]


class BSPreprocessorOutputHelper(ManipulatorOutputHelper):
    """
    <Interface>

    Methods to get dataset for binary segmentation.
    In this binary segmentation, we need raw image and corresponding mask.

    Methods
    -------
    get_mask() -> tf.data.Dataset
        Mask dataset. It returns `tf.data.Dataset`.
        `height`, `width` and 0 or 255 value of tf.uint8.
    """

    def get_mask_dataset(self) -> tf.data.Dataset:
        """
        Mask dataset.
        `height`, `width` and 0 or 255 value of tf.uint8.

        Returns
        -------
        tf.data.Dataset
            Dataset
        """
        pass

    def mask_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass

    def get_outputs(self) -> List[tf.data.Dataset]:
        mask_dataset = self.get_mask_dataset()
        for f in self.mask_preprocess_func():
            mask_dataset = mask_dataset.map(f)
        return [mask_dataset]


class BaseBSPreprocessorInputHelper(BSPreprocessorInputHelper):
    def __init__(self, datasets: List[tf.data.Dataset]):
        self._datasets: List[tf.data.Dataset] = datasets

    def get_image_dataset(self) -> tf.data.Dataset:
        return self._datasets[0]

    def image_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        f1 = lambda img: tf.cast(img, tf.float32) / 255.0
        return [f1]


class BaseBSPreprocessorOutputHelper(BSPreprocessorOutputHelper):
    def __init__(self, datasets: List[tf.data.Dataset]):
        self._datasets: List[tf.data.Dataset] = datasets

    def get_mask_dataset(self) -> tf.data.Dataset:
        return self._datasets[0]

    def mask_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        f1 = lambda img: tf.cast(img, tf.float32)
        f2 = lambda img: tf.cast(tf.greater(img, 0.5), tf.float32)
        return [f1, f2]


class BaseBSPreprocessorInOutHelper(
    BSPreprocessorInputHelper, BSPreprocessorOutputHelper
):
    def __init__(
        self,
        input_datasets: List[tf.data.Dataset],
        output_datasets: List[tf.data.Dataset],
    ):
        self._input_datasets: List[tf.data.Dataset] = input_datasets
        self._output_datasets: List[tf.data.Dataset] = output_datasets

    def get_image_dataset(self) -> tf.data.Dataset:
        return self._input_datasets[0]

    def image_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        f1 = lambda img: tf.cast(img, tf.float32) / 255.0
        return [f1]

    def get_mask_dataset(self) -> tf.data.Dataset:
        return self._output_datasets[0]

    def mask_preprocess_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        f1 = lambda img: tf.cast(img, tf.float32)
        f2 = lambda img: tf.cast(tf.greater(img, 0.5), tf.float32)
        return [f1, f2]
