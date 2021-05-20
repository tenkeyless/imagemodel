import os
from typing import List, Tuple

import cv2
import tensorflow as tf

from imagemodel.common.datasets.manipulator.manipulator import SupervisedManipulator
from imagemodel.common.reporter import Reporter
from imagemodel.experimental.reference_tracking.datasets.cell_tracking.preprocessor_helper import (
    ClaheRTPreprocessorPredictInputHelper, RTCellTrackingPreprocessorInputHelper, RTCellTrackingPreprocessorOutputHelper
)
from imagemodel.experimental.reference_tracking.datasets.rt_preprocessor import BaseRTPreprocessor, RTPreprocessor
from imagemodel.experimental.reference_tracking.datasets.rt_preprocessor_helper import (
    tf_color_to_random_map, tf_input_ref_label_preprocessing_function, tf_output_label_processing
)


class RTCellTrackingPreprocessor(BaseRTPreprocessor, RTPreprocessor):
    def __init__(self, manipulator: SupervisedManipulator, bin_size: int, cache_inout: bool = False):
        super().__init__(manipulator, bin_size)
        self._input_helper = RTCellTrackingPreprocessorInputHelper(
                datasets=manipulator.get_input_dataset(),
                bin_size=bin_size)
        self._output_helper = RTCellTrackingPreprocessorOutputHelper(
                datasets=manipulator.get_output_dataset(),
                bin_size=bin_size)
        self.cache_inout = cache_inout
    
    def get_zipped_dataset(self) -> tf.data.Dataset:
        """
        It returns combined input and output dataset.

        Returns
        -------
        tf.data.Dataset
            It returns tuple of `(Tuple of inputs, Tuple of outputs)`.
        """
        bin_size = self._input_helper.bin_size
        
        input_dataset = tf.data.Dataset.zip(tuple(self._input_helper.get_inputs()))
        output_dataset = tf.data.Dataset.zip(tuple(self._output_helper.get_outputs()))
        
        inout_dataset = tf.data.Dataset.zip((input_dataset, output_dataset))
        if self.cache_inout:
            inout_dataset = inout_dataset.cache()
        
        def generate_color_map(_input_dataset, _output_dataset):
            color_map = tf_color_to_random_map(_input_dataset[2], bin_size, 1)
            return (_input_dataset[0], _input_dataset[1], _input_dataset[2], color_map), (
                _output_dataset[0], _output_dataset[1], _output_dataset[2])
        
        def generate_bin_label(_input_dataset, _output_dataset):
            return (
                (
                    _input_dataset[0],
                    _input_dataset[1],
                    tf_input_ref_label_preprocessing_function(_input_dataset[2], _input_dataset[3], bin_size)
                ),
                (
                    _output_dataset[0],
                    _output_dataset[1],
                    tf_output_label_processing(_output_dataset[2], _input_dataset[3], bin_size)
                ))
        
        return inout_dataset.map(
                generate_color_map, num_parallel_calls=tf.data.experimental.AUTOTUNE).map(
                generate_bin_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    def plot_zipped_dataset(self, sample_num: int, target_base_folder: str):
        """
        Plot zipped dataset.
        
        Examples
        --------
        >>> from imagemodel.experimental.reference_tracking.configs.datasets import Datasets
        >>> pipeline = Datasets("rt_cell_tracking_training_1").get_pipeline(resize_to=(256, 256))
        >>> pipeline.preprocessor.plot_zipped_dataset(
        ...     sample_num=4, target_base_folder="/reference_tracking_results/test")
        
        Parameters
        ----------
        sample_num
        target_base_folder

        Returns
        -------

        """
        dataset = self.get_zipped_dataset()
        
        files: List[str] = []
        _base_folder: str = target_base_folder
        
        if target_base_folder.startswith("gs://"):
            _base_folder = "/tmp"
        
        for i, d in enumerate(dataset.take(sample_num)):
            inputs = d[0]
            
            main_img_file_name = "{}_RTCellTrackingPreprocessor_input_main_img.png".format(i)
            main_img_fullpath = os.path.join(_base_folder, main_img_file_name)
            cv2.imwrite(main_img_fullpath, inputs[0].numpy() * 255)
            files.append(main_img_fullpath)
            
            ref_img_file_name = "{}_RTCellTrackingPreprocessor_input_ref_img.png".format(i)
            ref_img_fullpath = os.path.join(_base_folder, ref_img_file_name)
            cv2.imwrite(ref_img_fullpath, inputs[1].numpy() * 255)
            files.append(ref_img_fullpath)
            
            for b in range(inputs[2].shape[-1]):
                ref_bin_file_name = "{}_RTCellTrackingPreprocessor_bin_{:02d}_input.png".format(i, b)
                ref_bin_fullpath = os.path.join(_base_folder, ref_bin_file_name)
                cv2.imwrite(ref_bin_fullpath, inputs[2][..., b:b + 1].numpy() * 255)
                files.append(ref_bin_fullpath)
            
            outputs = d[1]
            
            main_bw_label_file_name = "{}_RTCellTrackingPreprocessor_output_main_bw_label.png".format(i)
            main_bw_label_fullpath = os.path.join(_base_folder, main_bw_label_file_name)
            cv2.imwrite(main_bw_label_fullpath, outputs[0].numpy() * 255)
            files.append(main_bw_label_fullpath)
            
            ref_bw_label_file_name = "{}_RTCellTrackingPreprocessor_output_ref_bw_label.png".format(i)
            ref_bw_label_fullpath = os.path.join(_base_folder, ref_bw_label_file_name)
            cv2.imwrite(ref_bw_label_fullpath, outputs[1].numpy() * 255)
            files.append(ref_bw_label_fullpath)
            
            for b in range(outputs[2].shape[-2]):
                main_bin_file_name = "{}_RTCellTrackingPreprocessor_bin_{:02d}_output.png".format(i, b)
                main_bin_fullpath = os.path.join(_base_folder, main_bin_file_name)
                cv2.imwrite(main_bin_fullpath, outputs[2][..., b:b + 1, 0].numpy() * 255)
                files.append(main_bin_fullpath)
        
        if target_base_folder.startswith("gs://"):
            for file in files:
                Reporter.upload_file_to_google_storage(target_base_folder, file)


class RTCellTrackingPredictPreprocessor(RTCellTrackingPreprocessor):
    def __init__(self, manipulator: SupervisedManipulator, bin_size: int, fill_with: Tuple[int, int, int]):
        super().__init__(manipulator, bin_size)
        self._input_helper = ClaheRTPreprocessorPredictInputHelper(
                datasets=manipulator.get_input_dataset(),
                bin_size=bin_size,
                fill_with=fill_with)
