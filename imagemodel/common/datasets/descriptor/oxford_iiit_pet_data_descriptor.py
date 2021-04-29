from typing import Optional

import tensorflow as tf
import tensorflow_datasets as tfds

from imagemodel.common.datasets.descriptor.data_descriptor import BaseTFDataDescriptor
from imagemodel.common.utils.tfds import append_tfds_str_range


class OxfordIIITPetDataDescriptor(BaseTFDataDescriptor):
    def __init__(self, original_dataset: tf.data.Dataset):
        """
        Description for tfds "oxford_iiit_pet:3.*.*".
        https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet

        Methods
        -------
        get_img_dataset() -> tf.data.Dataset
            `tf.Tensor` of shape=(height, width, 3), dtype=uint8.
        get_mask_dataset() -> tf.data.Dataset
            `tf.Tensor` of shape=(height, width, 1), dtype=uint8.
        get_filename_dataset() -> tf.data.Dataset
            `tf.Tensor` of shape=(), dtype=string.
        get_label_dataset() -> tf.data.Dataset
            `tf.Tensor` of shape=(), dtype=int64.
        get_species_dataset() -> tf.data.Dataset
            `tf.Tensor` of shape=(), dtype=int64.

        Examples
        --------
        >>> data_descriptor = OxfordIIITPetDataDescriptor.init_with_train_dataset(
        ...     begin_optional=None, end_optional=80
        ... )
        >>> for d in data_descriptor.get_label_dataset().take(1)
        ...     print(d)
        ...
        tf.Tensor(1, shape=(), dtype=int64)
        """
        super(OxfordIIITPetDataDescriptor, self).__init__(
            original_dataset=original_dataset
        )

    @staticmethod
    def __from_tfds_oxford_iiit_pet_3(split_option: str):
        return tfds.load("oxford_iiit_pet:3.*.*", split=split_option, with_info=False)

    @classmethod
    def init_with_split_option(cls, split_option: str = "train"):
        return cls(original_dataset=cls.__from_tfds_oxford_iiit_pet_3(split_option))

    @classmethod
    def init_with_train_dataset(
            cls, begin_optional: Optional[int] = None, end_optional: Optional[int] = None
    ):
        option_string = append_tfds_str_range("train", begin_optional, end_optional)
        return cls.init_with_split_option(option_string)

    @classmethod
    def init_with_test_dataset(
            cls, begin_optional: Optional[int] = None, end_optional: Optional[int] = None
    ):
        option_string = append_tfds_str_range("test", begin_optional, end_optional)
        return cls.init_with_split_option(option_string)

    def get_img_dataset(self) -> tf.data.Dataset:
        """
        Color image dataset.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(height, width, 3), dtype=uint8.
        """
        return self.original_dataset.map(lambda el: el["image"])

    def get_mask_dataset(self) -> tf.data.Dataset:
        """
        Categorical mask dataset.
        0=Object, 1=Background, 2=Border.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(height, width, 1), dtype=uint8.
        """
        return self.original_dataset.map(lambda el: el["segmentation_mask"])

    def get_filename_dataset(self) -> tf.data.Dataset:
        """
        Filename dataset.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(), dtype=string.
        """
        return self.original_dataset.map(lambda el: el["file_name"])

    def get_label_dataset(self) -> tf.data.Dataset:
        """
        Label dataset.
        33=Sphynx, 12=english_cocker_spaniel, ...

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(), dtype=int64.
        """
        return self.original_dataset.map(lambda el: el["label"])

    def get_species_dataset(self) -> tf.data.Dataset:
        """
        Species dataset.
        0=Cat, 1=Dog

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(), dtype=int64.
        """
        return self.original_dataset.map(lambda el: el["species"])
