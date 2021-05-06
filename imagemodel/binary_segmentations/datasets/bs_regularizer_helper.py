from typing import Callable, List, Tuple

import tensorflow as tf

from imagemodel.common.datasets.regularizer_helper import RegularizerInputHelper, RegularizerOutputHelper


class BSRegularizerInputHelper(RegularizerInputHelper):
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

    def image_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass

    def get_inputs(self) -> List[tf.data.Dataset]:
        image_dataset = self.get_image_dataset()
        for f in self.image_regularizer_func():
            image_dataset = image_dataset.map(f, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return [image_dataset]


class BSRegularizerOutputHelper(RegularizerOutputHelper):
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

    def mask_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        pass

    def get_outputs(self) -> List[tf.data.Dataset]:
        mask_dataset = self.get_mask_dataset()
        for f in self.mask_regularizer_func():
            mask_dataset = mask_dataset.map(f, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return [mask_dataset]


class BaseBSRegularizerInputHelper(BSRegularizerInputHelper):
    def __init__(self, datasets: List[tf.data.Dataset], height_width_tuple: Tuple[int, int]):
        self._datasets: List[tf.data.Dataset] = datasets
        self._height_width_tuple: Tuple[int, int] = height_width_tuple

    def get_image_dataset(self) -> tf.data.Dataset:
        return self._datasets[0]

    def image_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        return [lambda img: tf.image.resize(img, self._height_width_tuple)]


class BaseBSRegularizerOutputHelper(BSRegularizerOutputHelper):
    def __init__(self, datasets: List[tf.data.Dataset], height_width_tuple: Tuple[int, int]):
        self._datasets: List[tf.data.Dataset] = datasets
        self._height_width_tuple: Tuple[int, int] = height_width_tuple

    def get_mask_dataset(self) -> tf.data.Dataset:
        return self._datasets[0]

    def mask_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        return [lambda img: tf.image.resize(img, self._height_width_tuple)]


class BaseBSRegularizerInOutHelper(BSRegularizerInputHelper, BSRegularizerOutputHelper):
    def __init__(
            self,
            input_datasets: List[tf.data.Dataset],
            output_datasets: List[tf.data.Dataset],
            height_width_tuple: Tuple[int, int]):
        self._input_datasets: List[tf.data.Dataset] = input_datasets
        self._output_datasets: List[tf.data.Dataset] = output_datasets
        self._height_width_tuple: Tuple[int, int] = height_width_tuple

    def get_image_dataset(self) -> tf.data.Dataset:
        return self._input_datasets[0]

    def image_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        return [lambda img: tf.image.resize(img, self._height_width_tuple)]

    def get_mask_dataset(self) -> tf.data.Dataset:
        return self._output_datasets[0]

    def mask_regularizer_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        return [lambda img: tf.image.resize(img, self._height_width_tuple)]
