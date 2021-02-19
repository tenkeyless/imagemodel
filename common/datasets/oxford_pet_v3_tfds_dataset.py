from typing import Optional

import tensorflow as tf
import tensorflow_datasets as tfds

from datasets.interfaces.dataset import TfdsDatasetInterface


class OxfordPetV3TfdsDataset(TfdsDatasetInterface):
    def __init__(self, batch_size: int = 64):
        self._dataset, self._info = tfds.load("oxford_iiit_pet:3.*.*", with_info=True)
        self.train_length = self._info.splits["train"].num_examples
        self.train_steps_per_epochs = self.train_length // batch_size
        self.buffer_size = 1000

    def normalize(self, input_image, input_mask):
        input_image = tf.cast(input_image, tf.float32) / 255.0
        input_mask -= 1
        return input_image, input_mask

    @tf.function
    def load_image_train(self, datapoint):
        input_image = tf.image.resize(datapoint["image"], (128, 128))
        input_mask = tf.image.resize(datapoint["segmentation_mask"], (128, 128))

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

        input_image, input_mask = self.normalize(input_image, input_mask)

        return input_image, input_mask

    def load_image_test(self, datapoint):
        input_image = tf.image.resize(datapoint["image"], (128, 128))
        input_mask = tf.image.resize(datapoint["segmentation_mask"], (128, 128))

        input_image, input_mask = self.normalize(input_image, input_mask)

        return input_image, input_mask

    def get_training_dataset(self, batch_size_optional: Optional[int] = None):
        batch_size = batch_size_optional or self.batch_size

        train_dataset = self._dataset["train"].map(
            self.load_image_train,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        train_dataset = (
            train_dataset.cache().shuffle(self.buffer_size).batch(batch_size).repeat()
        )
        train_dataset = train_dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )
        return train_dataset

    def get_training_dataset_length(self) -> int:
        return self._info.splits["train"].num_examples

    def get_validation_dataset(self, batch_size_optional: Optional[int] = None):
        return None

    def get_validation_dataset_length(self) -> int:
        return 0

    def get_test_dataset(self, batch_size_optional: Optional[int] = None):
        batch_size = batch_size_optional or self.batch_size

        test_dataset = self._dataset["test"].map(self.load_image_test)
        test_dataset = test_dataset.batch(batch_size)
        return test_dataset

    def get_test_dataset_length(self) -> int:
        return self._info.splits["test"].num_examples

    def get_test_dataset_filenames(self):
        test_dataset_filenames = self._dataset["test"].map(lambda el: el["file_name"])
        return test_dataset_filenames
