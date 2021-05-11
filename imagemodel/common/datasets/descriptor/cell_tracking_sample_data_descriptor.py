import os
from typing import Optional

import tensorflow as tf

from imagemodel.common.datasets.descriptor.cell_tracking_data_descriptor import CellTrackingDataDescriptor
from imagemodel.common.datasets.descriptor.data_descriptor import BaseTFDataDescriptor
from imagemodel.common.utils.tf_images import decode_png


def get_filename_from_fullpath(name):
    return tf.strings.split(name, sep="/")[-1]


def combine_folder_file(a, b):
    return a + "/" + b


# Don't use this.
# Example for read from disk or google cloud storage.
class CellTrackingSampleTestDataDescriptor(CellTrackingDataDescriptor, BaseTFDataDescriptor):
    def __init__(self, original_dataset: Optional[tf.data.Dataset], base_folder: str):
        super().__init__(original_dataset=original_dataset, base_folder=base_folder)
        self.sample_folder: str = os.path.join(self.base_folder, "framed_sample")
    
    def get_sample_filename_dataset(self) -> tf.data.Dataset:
        """
        Sample filename dataset.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(), dtype=string.
        """
        dataset = CellTrackingSampleTestDataDescriptor.base_files(self.sample_folder)
        return dataset
    
    def get_main_img_dataset(self) -> tf.data.Dataset:
        """
        Grayscale image dataset.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(height, width, 1), dtype=uint8.
        """
        dataset: tf.data.Dataset = CellTrackingSampleTestDataDescriptor.base_files(
                self.sample_folder,
                shuffle_seed=None)
        dataset = dataset.map(lambda fname: combine_folder_file(self.main_image_folder, fname))
        dataset = dataset.map(decode_png, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset
    
    def get_main_mask_dataset(self) -> tf.data.Dataset:
        """
        Color main mask dataset.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(height, width, 3), dtype=uint8.
        """
        dataset: tf.data.Dataset = CellTrackingSampleTestDataDescriptor.base_files(
                self.sample_folder,
                shuffle_seed=None)
        dataset = dataset.map(lambda fname: combine_folder_file(self.main_label_folder, fname))
        dataset = dataset.map(lambda el: decode_png(el, 3), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset
    
    def get_main_bw_mask_dataset(self) -> tf.data.Dataset:
        """
        Black white main mask dataset.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(height, width, 1), dtype=uint8.
        """
        dataset: tf.data.Dataset = CellTrackingSampleTestDataDescriptor.base_files(
                self.sample_folder,
                shuffle_seed=None)
        dataset = dataset.map(lambda fname: combine_folder_file(self.main_bw_label_folder, fname))
        dataset = dataset.map(decode_png, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset
    
    def get_ref_img_dataset(self) -> tf.data.Dataset:
        """
        Grayscale image dataset.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(height, width, 1), dtype=uint8.
        """
        dataset: tf.data.Dataset = CellTrackingSampleTestDataDescriptor.base_files(
                self.sample_folder,
                shuffle_seed=None)
        dataset = dataset.map(lambda fname: combine_folder_file(self.ref_image_folder, fname))
        dataset = dataset.map(decode_png, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset
    
    def get_ref_mask_dataset(self) -> tf.data.Dataset:
        """
        Color ref mask dataset.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(height, width, 3), dtype=uint8.
        """
        dataset: tf.data.Dataset = CellTrackingSampleTestDataDescriptor.base_files(
                self.sample_folder,
                shuffle_seed=None)
        dataset = dataset.map(lambda fname: combine_folder_file(self.ref_label_folder, fname))
        dataset = dataset.map(lambda el: decode_png(el, 3), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset
    
    def get_ref_bw_mask_dataset(self) -> tf.data.Dataset:
        """
        Black white ref mask dataset.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(height, width, 1), dtype=uint8.
        """
        dataset: tf.data.Dataset = CellTrackingSampleTestDataDescriptor.base_files(
                self.sample_folder,
                shuffle_seed=None)
        dataset = dataset.map(lambda fname: combine_folder_file(self.ref_bw_label_folder, fname))
        dataset = dataset.map(decode_png, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset
