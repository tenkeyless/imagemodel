import tensorflow as tf
import tensorflow_datasets as tfds
from imagemodel.common.datasets.interfaces.data_feeder import BaseTFDSDataFeeder


class BaseOxfordPetDataFeeder(BaseTFDSDataFeeder):
    def __init__(
        self, original_dataset: tf.data.Dataset, original_info: tfds.core.DatasetInfo
    ):
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
        >>> di = tfds.load("oxford_iiit_pet:3.*.*", split="train[80%:]", with_info=True)
        >>> data_feeder = BaseOxfordPetDataFeeder(original_dataset=di[0], original_info=di[1])
        >>> for d in data_feeder.get_label_dataset().take(1)
        ...     print(d)
        ...
        tf.Tensor(1, shape=(), dtype=int64)
        """
        super(BaseOxfordPetDataFeeder, self).__init__(
            original_dataset=original_dataset, original_info=original_info
        )

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
