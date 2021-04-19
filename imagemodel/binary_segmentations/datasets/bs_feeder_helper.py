from typing import List

import tensorflow as tf

from imagemodel.common.datasets.feeder_helper import FeederInputHelper, FeederOutputHelper


class BSFeederInputHelper(FeederInputHelper):
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

    def get_inputs(self) -> List[tf.data.Dataset]:
        # 1. cache, batch, repeat handle
        # train_dataset.cache().shuffle(self.buffer_size).batch(batch_size).repeat()
        # 2. prefetch handle
        # .prefetch(
        #     buffer_size=tf.data.experimental.AUTOTUNE
        # )
        return [self.get_image()]


class BSFeederOutputHelper(FeederOutputHelper):
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

    def get_outputs(self) -> List[tf.data.Dataset]:
        return [self.get_mask()]
