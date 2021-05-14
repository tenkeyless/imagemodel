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
    def __init__(
            self,
            original_dataset: Optional[tf.data.Dataset],
            base_folder: str,
            shuffle: bool = True,
            cache: bool = False):
        super().__init__(original_dataset=original_dataset)
        self.base_folder: str = base_folder
        
        self.main_image_folder: str = os.path.join(self.base_folder, "framed_image", "zero")
        self.main_label_folder: str = os.path.join(self.base_folder, "framed_label", "zero")
        self.main_bw_label_folder: str = os.path.join(self.base_folder, "framed_bw_label", "zero")
        
        self.ref_image_folder: str = os.path.join(self.base_folder, "framed_image", "p1")
        self.ref_label_folder: str = os.path.join(self.base_folder, "framed_label", "p1")
        self.ref_bw_label_folder: str = os.path.join(self.base_folder, "framed_bw_label", "p1")
        
        self.shuffle = shuffle
        self.random_seed = 42
        self.cache = cache
        
        self.filename_base_folder: str = self.main_image_folder
    
    def get_filename_dataset(self) -> tf.data.Dataset:
        """
        Filename dataset.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(), dtype=string.
        """
        if self.shuffle:
            filename_dataset = tf.data.Dataset.list_files(
                    self.filename_base_folder + "/*",
                    shuffle=True,
                    seed=self.random_seed)
        else:
            filename_dataset = tf.data.Dataset.list_files(self.filename_base_folder + "/*", shuffle=False)
        filename_dataset = filename_dataset.map(get_filename_from_fullpath)
        return filename_dataset
    
    def get_main_img_dataset(self) -> tf.data.Dataset:
        """
        Grayscale image dataset.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(height, width, 1), dtype=uint8.
        """
        dataset = self.get_filename_dataset()
        dataset = dataset.map(lambda fname: combine_folder_file(self.main_image_folder, fname))
        dataset = dataset.map(decode_png, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.cache:
            dataset = dataset.cache()
        return dataset
    
    def get_main_mask_dataset(self) -> tf.data.Dataset:
        """
        Color main mask dataset.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(height, width, 3), dtype=uint8.
        """
        dataset = self.get_filename_dataset()
        dataset = dataset.map(lambda fname: combine_folder_file(self.main_label_folder, fname))
        dataset = dataset.map(lambda el: decode_png(el, 3), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.cache:
            dataset = dataset.cache()
        return dataset
    
    def get_main_bw_mask_dataset(self) -> tf.data.Dataset:
        """
        Black white main mask dataset.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(height, width, 1), dtype=uint8.
        """
        dataset = self.get_filename_dataset()
        dataset = dataset.map(lambda fname: combine_folder_file(self.main_bw_label_folder, fname))
        dataset = dataset.map(decode_png, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.cache:
            dataset = dataset.cache()
        return dataset
    
    def get_ref_img_dataset(self) -> tf.data.Dataset:
        """
        Grayscale image dataset.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(height, width, 1), dtype=uint8.
        """
        dataset = self.get_filename_dataset()
        dataset = dataset.map(lambda fname: combine_folder_file(self.ref_image_folder, fname))
        dataset = dataset.map(decode_png, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.cache:
            dataset = dataset.cache()
        return dataset
    
    def get_ref_mask_dataset(self) -> tf.data.Dataset:
        """
        Color ref mask dataset.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(height, width, 3), dtype=uint8.
        """
        dataset = self.get_filename_dataset()
        dataset = dataset.map(lambda fname: combine_folder_file(self.ref_label_folder, fname))
        dataset = dataset.map(lambda el: decode_png(el, 3), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.cache:
            dataset = dataset.cache()
        return dataset
    
    def get_ref_bw_mask_dataset(self) -> tf.data.Dataset:
        """
        Black white ref mask dataset.

        Returns
        -------
        tf.data.Dataset
            `tf.Tensor` of shape=(height, width, 1), dtype=uint8.
        """
        dataset = self.get_filename_dataset()
        dataset = dataset.map(lambda fname: combine_folder_file(self.ref_bw_label_folder, fname))
        dataset = dataset.map(decode_png, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.cache:
            dataset = dataset.cache()
        return dataset
