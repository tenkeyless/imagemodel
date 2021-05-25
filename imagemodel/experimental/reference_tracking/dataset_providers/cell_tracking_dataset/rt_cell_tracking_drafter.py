from typing import Optional, Tuple

import tensorflow as tf

from imagemodel.common.utils.tf_images import decode_png
from imagemodel.experimental.reference_tracking.dataset_providers.rt_drafter import RTDrafterP, RTDrafterT


class RTCellTrackingDrafterT(RTDrafterT):
    """
    Examples
    --------
    >>> import os
    >>>
    >>> base_folder = "/data/tracking_training"
    >>> main_image_folder: str = os.path.join(base_folder, "framed_image", "zero")
    >>> main_label_folder: str = os.path.join(base_folder, "framed_label", "zero")
    >>> main_bw_label_folder: str = os.path.join(base_folder, "framed_bw_label", "zero")
    >>> ref_image_folder: str = os.path.join(base_folder, "framed_image", "p1")
    >>> ref_label_folder: str = os.path.join(base_folder, "framed_label", "p1")
    >>> ref_bw_label_folder: str = os.path.join(base_folder, "framed_bw_label", "p1")
    >>> folders = (
    ...     main_image_folder,
    ...     ref_image_folder,
    ...     main_label_folder,
    ...     ref_label_folder,
    ...     main_bw_label_folder,
    ...     ref_bw_label_folder)
    ...
    >>> from imagemodel.experimental.reference_tracking.dataset_providers.cell_tracking_dataset.\
    ...     rt_cell_tracking_drafter import RTCellTrackingDrafterT
    >>> dt = RTCellTrackingDrafterT(folders, shuffle_for_trainer=True, shuffle=False, random_seed=42)
    >>> for d in dt.out_dataset.take(1):
    ...     print(d)
    ...
    """
    
    def __init__(
            self,
            folders: Tuple[str, str, str, str, str, str],
            shuffle_for_trainer: bool,
            shuffle: bool,
            random_seed: Optional[int]):
        self.filename_base_folder: str = folders[0]
        self.folders: Tuple[str, str, str, str, str, str] = folders
        self.shuffle_for_trainer: bool = shuffle_for_trainer
        self.shuffle: bool = shuffle
        self.random_seed: Optional[int] = random_seed
    
    def get_filename_dataset(self) -> tf.data.Dataset:
        def get_filename_from_fullpath(name):
            return tf.strings.split(name, sep="/")[-1]
        
        file_folder_dataset = tf.data.Dataset.list_files(
                self.filename_base_folder + "/*",
                shuffle=self.shuffle,
                seed=self.random_seed)
        filename_dataset = file_folder_dataset.map(
                get_filename_from_fullpath,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return filename_dataset
    
    def __to_file_folder_dataset(self, filename: tf.Tensor) -> \
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        def __combine_folder_file(a, b):
            return a + "/" + b
        
        return (
            __combine_folder_file(self.folders[0], filename),
            __combine_folder_file(self.folders[1], filename),
            __combine_folder_file(self.folders[2], filename),
            __combine_folder_file(self.folders[3], filename),
            __combine_folder_file(self.folders[4], filename),
            __combine_folder_file(self.folders[5], filename))
    
    def __load_image(
            self,
            main_image_file_folder: str,
            ref_image_file_folder: str,
            main_label_folder: str,
            ref_label_folder: str,
            main_bw_label_folder: str,
            ref_bw_label_folder: str):
        return (
            decode_png(main_image_file_folder),
            decode_png(ref_image_file_folder),
            decode_png(main_label_folder, 3),
            decode_png(ref_label_folder, 3),
            decode_png(main_bw_label_folder),
            decode_png(ref_bw_label_folder))
    
    @property
    def out_dataset(self) -> tf.data.Dataset:
        filename_dataset = self.get_filename_dataset()
        # TODO: Resolves an issue where memory usage increases without limit when using shuffle.
        if self.shuffle_for_trainer:
            # filename_dataset = filename_dataset.shuffle(len(filename_dataset))
            filename_dataset = filename_dataset.shuffle(512)
        file_folder_dataset = filename_dataset.map(
                self.__to_file_folder_dataset,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        image_dataset = file_folder_dataset.map(self.__load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return image_dataset


class RTCellTrackingDrafterP(RTDrafterP):
    """
    Examples
    --------
    >>> import os
    >>>
    >>> base_folder = "/data/tracking_training"
    >>> main_image_folder: str = os.path.join(base_folder, "framed_image", "zero")
    >>> ref_image_folder: str = os.path.join(base_folder, "framed_image", "p1")
    >>> ref_label_folder: str = os.path.join(base_folder, "framed_label", "p1")
    >>> filename_folder: str = "/data/tracking_test2/framed_sample"
    >>> folders = (
    ...      main_image_folder,
    ...      ref_image_folder,
    ...      ref_label_folder)
    ...
    >>> from imagemodel.experimental.reference_tracking.dataset_providers.cell_tracking_dataset.\
    ...     rt_cell_tracking_drafter import RTCellTrackingDrafterP
    >>> dt = RTCellTrackingDrafterP(filename_folder, folders, False, 42)
    >>> for d in dt.out_dataset.take(1):
    ...      print(d)
    ...
    """
    
    def __init__(
            self,
            filename_folder: Optional[str],
            folders: Tuple[str, str, str],
            shuffle: bool,
            random_seed: Optional[int]):
        self.filename_base_folder = filename_folder or folders[0]
        self.folders = folders
        self.shuffle = shuffle
        self.random_seed = random_seed
    
    def get_base_file_folder_dataset(self) -> tf.data.Dataset:
        return tf.data.Dataset.list_files(
                self.filename_base_folder + "/*",
                shuffle=self.shuffle,
                seed=self.random_seed)
    
    def __to_filename_dataset(self, file_folder: str) -> tf.data.Dataset:
        def get_filename_from_fullpath(name):
            return tf.strings.split(name, sep="/")[-1]
        
        return get_filename_from_fullpath(file_folder)
    
    def __to_file_folder_dataset(self, filename: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        def __combine_folder_file(a, b):
            return a + "/" + b
        
        return (
            filename,
            __combine_folder_file(self.folders[0], filename),
            __combine_folder_file(self.folders[1], filename),
            __combine_folder_file(self.folders[2], filename))
    
    def __load_image(
            self,
            filename: tf.Tensor,
            main_image_file_folder: str,
            ref_image_file_folder: str,
            ref_label_folder: str):
        return (
            filename,
            decode_png(main_image_file_folder),
            decode_png(ref_image_file_folder),
            decode_png(ref_label_folder, 3))
    
    @property
    def out_dataset(self) -> tf.data.Dataset:
        base_file_folder_dataset = self.get_base_file_folder_dataset()
        filename_dataset = base_file_folder_dataset.map(
                self.__to_filename_dataset,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        file_folder_dataset = filename_dataset.map(
                self.__to_file_folder_dataset,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        image_dataset = file_folder_dataset.map(self.__load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return image_dataset
