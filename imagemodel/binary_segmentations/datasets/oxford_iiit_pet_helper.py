import tensorflow as tf
from imagemodel.binary_segmentations.datasets.binary_segmentations_helper import (
    BSFeederHelper,
)
from imagemodel.binary_segmentations.datasets.oxford_iiit_pet_data_feeder import (
    BaseOxfordPetDataFeeder,
)


@tf.autograph.experimental.do_not_convert
def oxford_pet_mask_to_binary(label):
    """
    Change 1=Background, 2=Border, 0=Object to 1=Object,Border 0=Background.

    Parameters
    ----------
    label : `tf.Tensor`
        1=Background, 2=Border, 0=Object image

    Returns
    -------
    `tf.Tensor`
        1=Object,Border 0=Background image
    """
    casted_label = tf.cast(label, tf.int8)
    casted_label = casted_label - 1
    result = tf.math.abs(casted_label)
    return tf.cast(result, tf.uint8)


@tf.autograph.experimental.do_not_convert
def oxford_pet_mask_to_binary2(label):
    """
    Change 1=Background, 2=Border, 0=Object to 1=Object 0=Background,Border.

    Parameters
    ----------
    label : `tf.Tensor`
        1=Background, 2=Border, 0=Object image

    Returns
    -------
    `tf.Tensor`
        1=Object 0=Background,Border image
    """
    casted_label = tf.cast(label, tf.int8)
    casted_label = tf.clip_by_value(casted_label, 0, 1)
    result = 1 - casted_label
    return tf.cast(result, tf.uint8)


class BaseBSOxfordIIITPetFeederHelper(BSFeederHelper):
    def __init__(self, data_feeder: BaseOxfordPetDataFeeder):
        self.data_feeder: BaseOxfordPetDataFeeder = data_feeder


class BSOxfordIIITPetFeederHelper(BaseBSOxfordIIITPetFeederHelper):
    def get_image(self) -> tf.data.Dataset:
        return self.data_feeder.get_img_dataset()

    def get_mask(self) -> tf.data.Dataset:
        return self.data_feeder.get_mask_dataset().map(oxford_pet_mask_to_binary)


class BSOxfordIIITPetFeederHelper2(BaseBSOxfordIIITPetFeederHelper):
    def get_image(self) -> tf.data.Dataset:
        return self.data_feeder.get_img_dataset()

    def get_mask(self) -> tf.data.Dataset:
        return self.data_feeder.get_mask_dataset().map(oxford_pet_mask_to_binary2)
