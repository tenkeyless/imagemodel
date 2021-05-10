from typing import List, Tuple

import tensorflow as tf
from image_keras.tf.utils.images import decode_png

from imagemodel.common.datasets.descriptor.cell_tracking_data_descriptor import (
    CellTrackingDataDescriptor,
    combine_folder_file
)
from imagemodel.reference_tracking.datasets.rt_feeder_helper import RTFeederInputHelper, RTFeederOutputHelper


class CellTrackingFeederInputHelper(RTFeederInputHelper):
    def __init__(self, data_descriptor: CellTrackingDataDescriptor):
        self._data_descriptor = data_descriptor
        self.len_filenames = len(self._data_descriptor.get_filename_dataset())
        self.dataset = self._data_descriptor.get_filename_dataset()
    
    def get_inputs(self) -> List[tf.data.Dataset]:
        def __input_filename_to_main_ref_set(filename: str) -> Tuple[str, str, str]:
            main_path_filename: str = combine_folder_file(self._data_descriptor.main_image_folder, filename)
            ref_path_filename: str = combine_folder_file(self._data_descriptor.ref_image_folder, filename)
            ref_label_path_filename: str = combine_folder_file(self._data_descriptor.ref_label_folder, filename)
            return main_path_filename, ref_path_filename, ref_label_path_filename
        
        def __input_decode_imgs(main_path_filename: str, ref_path_filename: str, ref_label_path_filename: str) -> \
                Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            main_images: tf.Tensor = decode_png(main_path_filename)
            ref_images: tf.Tensor = decode_png(ref_path_filename)
            ref_labels: tf.Tensor = decode_png(ref_label_path_filename, 3)
            return main_images, ref_images, ref_labels
        
        dataset = self.dataset
        dataset = dataset.shuffle(buffer_size=self.len_filenames, seed=42)
        dataset = dataset.map(__input_filename_to_main_ref_set)
        # dataset = dataset.map(__input_decode_imgs)
        dataset = dataset.map(__input_decode_imgs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return [dataset]


class CellTrackingFeederOutputHelper(RTFeederOutputHelper):
    def __init__(self, data_descriptor: CellTrackingDataDescriptor):
        self._data_descriptor = data_descriptor
        self.len_filenames = len(self._data_descriptor.get_filename_dataset())
        self.dataset = self._data_descriptor.get_filename_dataset()
    
    def get_outputs(self) -> List[tf.data.Dataset]:
        def __output_filename_to_main_ref_set(filename: str) -> Tuple[str, str, str, str]:
            main_bw_path_filename: str = combine_folder_file(self._data_descriptor.main_bw_label_folder, filename)
            ref_bw_path_filename: str = combine_folder_file(self._data_descriptor.ref_bw_label_folder, filename)
            ref_label_path_filename: str = combine_folder_file(self._data_descriptor.ref_label_folder, filename)
            main_label_path_filename: str = combine_folder_file(self._data_descriptor.main_label_folder, filename)
            return main_bw_path_filename, ref_bw_path_filename, ref_label_path_filename, main_label_path_filename
        
        def __output_decode_imgs(
                main_bw_path_filename: str,
                ref_bw_path_filename: str,
                ref_label_path_filename: str,
                main_label_path_filename: str) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            main_bw_images: tf.Tensor = decode_png(main_bw_path_filename)
            ref_bw_images: tf.Tensor = decode_png(ref_bw_path_filename)
            ref_labels: tf.Tensor = decode_png(ref_label_path_filename, 3)
            main_labels: tf.Tensor = decode_png(main_label_path_filename, 3)
            return main_bw_images, ref_bw_images, ref_labels, main_labels
        
        dataset = self.dataset
        dataset = dataset.shuffle(buffer_size=self.len_filenames, seed=42)
        dataset = dataset.map(__output_filename_to_main_ref_set)
        # dataset = dataset.map(__output_decode_imgs)
        dataset = dataset.map(__output_decode_imgs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return [dataset]
