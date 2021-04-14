import tensorflow as tf
from imagemodel.binary_segmentations.datasets.bs_feeder_helper import (
    BSFeederInputHelper,
    BSFeederOutputHelper,
)
from imagemodel.common.datasets.descriptor.oxford_iiit_pet_data_descriptor import (
    OxfordIIITPetDataDescriptor,
)


@tf.autograph.experimental.do_not_convert
def oxford_pet_mask_to_binary(label: tf.Tensor) -> tf.Tensor:
    """
    Change 2=Background, 3=Border, 1=Object to 1=Object,Border 0=Background.

    Parameters
    ----------
    label : `tf.Tensor`
        1=Object, 2=Background, 3=Border image

    Returns
    -------
    `tf.Tensor`
        1=Object,Border 0=Background image
    """
    casted_label = tf.cast(label, tf.int16)
    casted_label = casted_label - 2
    result = tf.math.abs(casted_label)
    return tf.cast(result, tf.uint8)


@tf.autograph.experimental.do_not_convert
def oxford_pet_mask_to_binary2(label: tf.Tensor) -> tf.Tensor:
    """
    Change 2=Background, 3=Border, 1=Object to 1=Object 0=Background,Border.

    Parameters
    ----------
    label : `tf.Tensor`
        1=Object, 2=Background, 3=Border image

    Returns
    -------
    `tf.Tensor`
        1=Object 0=Background,Border image
    """
    casted_label = tf.cast(label, tf.int16)
    casted_label = -casted_label + 2
    casted_label = tf.clip_by_value(casted_label, 0, 1)
    result = 1 - casted_label
    return tf.cast(result, tf.uint8)


class BSOxfordIIITPetFeederInputHelper(BSFeederInputHelper):
    def __init__(self, data_descriptor: OxfordIIITPetDataDescriptor):
        self._data_descriptor = data_descriptor

    def get_image(self) -> tf.data.Dataset:
        return self._data_descriptor.get_img_dataset()


class BSOxfordIIITPetFeederOutputHelper(BSFeederOutputHelper):
    def __init__(self, data_descriptor: OxfordIIITPetDataDescriptor):
        self._data_descriptor = data_descriptor

    def get_mask(self) -> tf.data.Dataset:
        return self._data_descriptor.get_mask_dataset().map(oxford_pet_mask_to_binary)


class BSOxfordIIITPetFeederOutputHelper2(BSFeederOutputHelper):
    def __init__(self, data_descriptor: OxfordIIITPetDataDescriptor):
        self._data_descriptor = data_descriptor

    def get_mask(self) -> tf.data.Dataset:
        return self._data_descriptor.get_mask_dataset().map(oxford_pet_mask_to_binary2)
