from abc import ABCMeta

import tensorflow as tf


class BSFeederHelper(metaclass=ABCMeta):
    """
    <Interface>

    Methods to get dataset for binary segmentation.
    In this binary segmentation, we need raw image and corresponding mask.

    Methods
    -------
    get_image() -> tf.data.Dataset
        Image dataset. It returns `tf.data.Dataset`.
        `height`, `width` and 0~255 range of tf.uint8.
    get_mask() -> tf.data.Dataset
        Mask dataset. It returns `tf.data.Dataset`.
        `height`, `width` and 0 or 255 value of tf.uint8.
    """

    def get_image(self) -> tf.data.Dataset:
        """
        Image dataset.
        `height`, `width` and 0~255 range of tf.uint8.

        Returns
        -------
        tf.data.Dataset
            Dataset
        """
        pass

    def get_mask(self) -> tf.data.Dataset:
        """
        Mask dataset.
        `height`, `width` and 0 or 255 value of tf.uint8.

        Returns
        -------
        tf.data.Dataset
            Dataset
        """
        pass
