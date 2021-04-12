import tensorflow as tf
from imagemodel.binary_segmentations.datasets.bs_helper import (
    BSPurposeInputHelper,
    BSPurposeOutputHelper,
)
from imagemodel.binary_segmentations.datasets.oxford_iiit_pet.oxford_iiit_pet_data_descriptor import (
    OxfordIIITPetDataDescriptor,
)


@tf.autograph.experimental.do_not_convert
def oxford_pet_mask_to_binary(label: tf.Tensor) -> tf.Tensor:
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
def oxford_pet_mask_to_binary2(label: tf.Tensor) -> tf.Tensor:
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


class BSOxfordIIITPetPurposeInputHelper(BSPurposeInputHelper):
    def __init__(self, data_descriptor: OxfordIIITPetDataDescriptor):
        self._data_descriptor = data_descriptor

    def get_image(self) -> tf.data.Dataset:
        return self._data_descriptor.get_img_dataset()


class BSOxfordIIITPetPurposeOutputHelper(BSPurposeOutputHelper):
    def __init__(self, data_descriptor: OxfordIIITPetDataDescriptor):
        self._data_descriptor = data_descriptor

    def get_mask(self) -> tf.data.Dataset:
        return self._data_descriptor.get_mask_dataset().map(oxford_pet_mask_to_binary)


class BSOxfordIIITPetPurposeOutputHelper2(BSPurposeOutputHelper):
    def __init__(self, data_descriptor: OxfordIIITPetDataDescriptor):
        self._data_descriptor = data_descriptor

    def get_mask(self) -> tf.data.Dataset:
        return self._data_descriptor.get_mask_dataset().map(oxford_pet_mask_to_binary2)
