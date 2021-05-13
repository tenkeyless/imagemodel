from typing import Tuple

import tensorflow as tf

from imagemodel.common.datasets.manipulator.manipulator import SupervisedManipulator
from imagemodel.reference_tracking.datasets.cell_tracking.preprocessor_helper import (
    ClaheRTPreprocessorPredictInputHelper, RTCellTrackingPreprocessorInputHelper, RTCellTrackingPreprocessorOutputHelper
)
from imagemodel.reference_tracking.datasets.rt_preprocessor import BaseRTPreprocessor, RTPreprocessor
from imagemodel.reference_tracking.datasets.rt_preprocessor_helper import (
    tf_color_to_random_map, tf_input_ref_label_preprocessing_function, tf_output_label_processing
)


class RTCellTrackingPreprocessor(BaseRTPreprocessor, RTPreprocessor):
    def __init__(self, manipulator: SupervisedManipulator, bin_size: int):
        super().__init__(manipulator, bin_size)
        self._input_helper = RTCellTrackingPreprocessorInputHelper(
                datasets=manipulator.get_input_dataset(),
                bin_size=bin_size)
        self._output_helper = RTCellTrackingPreprocessorOutputHelper(
                datasets=manipulator.get_output_dataset(),
                bin_size=bin_size)
    
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
        
        return inout_dataset.map(generate_color_map).map(generate_bin_label)


class RTCellTrackingPredictPreprocessor(RTCellTrackingPreprocessor):
    def __init__(self, manipulator: SupervisedManipulator, bin_size: int, fill_with: Tuple[int, int, int]):
        super().__init__(manipulator, bin_size)
        self._input_helper = ClaheRTPreprocessorPredictInputHelper(
                datasets=manipulator.get_input_dataset(),
                bin_size=bin_size,
                fill_with=fill_with)
