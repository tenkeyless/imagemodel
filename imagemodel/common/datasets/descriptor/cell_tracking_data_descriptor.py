import os
from typing import Optional

import tensorflow as tf
from image_keras.tf.utils.images import decode_png

from imagemodel.common.datasets.descriptor.data_descriptor import BaseTFDataDescriptor


def get_filename_from_fullpath(name):
    return tf.strings.split(name, sep="/")[-1]


def combine_folder_file(a, b):
    return a + "/" + b


# Don't use this.
# Example for read from disk or google cloud storage.
class CellTrackingDataDescriptor(BaseTFDataDescriptor):
    def __init__(self, original_dataset: Optional[tf.data.Dataset], base_folder: str):
        super().__init__(original_dataset=original_dataset)
        self.base_folder: str = base_folder
    
    @staticmethod
    def __base_files(folder_name: str, shuffle_seed: Optional[int] = 42) -> tf.data.Dataset:
        shuffle = True if shuffle_seed else False
        return tf.data.Dataset.list_files(folder_name + "/*", shuffle=shuffle, seed=shuffle_seed).map(
                get_filename_from_fullpath)
    
    @staticmethod
    def __base_files_with_folder(folder_name: str, shuffle_seed: Optional[int] = 42) -> tf.data.Dataset:
        file_dataset: tf.data.Dataset = CellTrackingDataDescriptor.__base_files(folder_name, shuffle_seed)
        return file_dataset.map(lambda fname: combine_folder_file(folder_name, fname))
    
    def get_filename_dataset(self) -> tf.data.Dataset:
        """
        Filename dataset.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(), dtype=string.
        """
        main_image_folder = os.path.join(self.base_folder, "framed_image", "zero")
        dataset = CellTrackingDataDescriptor.__base_files(main_image_folder)
        return dataset
    
    def get_main_img_dataset(self) -> tf.data.Dataset:
        """
        Grayscale image dataset.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(height, width, 1), dtype=uint8.
        """
        main_image_folder = os.path.join(self.base_folder, "framed_image", "zero")
        dataset = CellTrackingDataDescriptor.__base_files_with_folder(main_image_folder)
        dataset = dataset.map(decode_png)
        return dataset
    
    def get_main_mask_dataset(self) -> tf.data.Dataset:
        """
        Color main mask dataset.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(height, width, 3), dtype=uint8.
        """
        main_label_folder: str = os.path.join(self.base_folder, "framed_label", "zero")
        dataset = CellTrackingDataDescriptor.__base_files_with_folder(os.path.join(main_label_folder))
        dataset = dataset.map(lambda el: decode_png(el, 3))
        return dataset
    
    def get_main_bw_mask_dataset(self) -> tf.data.Dataset:
        """
        Black white main mask dataset.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(height, width, 1), dtype=uint8.
        """
        main_bw_label_folder: str = os.path.join(self.base_folder, "framed_bw_label", "zero")
        dataset = CellTrackingDataDescriptor.__base_files_with_folder(main_bw_label_folder)
        dataset = dataset.map(decode_png)
        return dataset
    
    def get_ref_img_dataset(self) -> tf.data.Dataset:
        """
        Grayscale image dataset.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(height, width, 1), dtype=uint8.
        """
        ref_image_folder: str = os.path.join(self.base_folder, "framed_image", "p1")
        dataset = CellTrackingDataDescriptor.__base_files_with_folder(ref_image_folder)
        dataset = dataset.map(decode_png)
        return dataset
    
    def get_ref_mask_dataset(self) -> tf.data.Dataset:
        """
        Color ref mask dataset.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(height, width, 3), dtype=uint8.
        """
        ref_label_folder: str = os.path.join(self.base_folder, "framed_label", "p1")
        dataset = CellTrackingDataDescriptor.__base_files_with_folder(ref_label_folder)
        dataset = dataset.map(lambda el: decode_png(el, 3))
        return dataset
    
    def get_ref_bw_mask_dataset(self) -> tf.data.Dataset:
        """
        Black white ref mask dataset.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(height, width, 1), dtype=uint8.
        """
        ref_bw_label_folder: str = os.path.join(self.base_folder, "framed_bw_label", "p1")
        dataset = CellTrackingDataDescriptor.__base_files_with_folder(ref_bw_label_folder)
        dataset = dataset.map(decode_png)
        return dataset
