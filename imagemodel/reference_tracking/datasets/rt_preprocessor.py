from abc import ABCMeta

from imagemodel.common.datasets.manipulator.manipulator import SupervisedManipulator
from imagemodel.common.datasets.preprocessor import Preprocessor
from imagemodel.reference_tracking.datasets.rt_preprocessor_helper import (
    BaseRTPreprocessorInputHelper, BaseRTPreprocessorOutputHelper, RTPreprocessorInputHelper,
    RTPreprocessorOutputHelper
)


class RTPreprocessor(Preprocessor[RTPreprocessorInputHelper, RTPreprocessorOutputHelper], metaclass=ABCMeta):
    pass


class BaseRTPreprocessor(RTPreprocessor):
    def __init__(self, manipulator: SupervisedManipulator, bin_size: int):
        self._input_helper = BaseRTPreprocessorInputHelper(datasets=manipulator.get_input_dataset(), bin_size=bin_size)
        self._output_helper = BaseRTPreprocessorOutputHelper(
                datasets=manipulator.get_output_dataset(),
                bin_size=bin_size)
    
    @property
    def input_helper(self) -> RTPreprocessorInputHelper:
        return self._input_helper
    
    @property
    def output_helper(self) -> RTPreprocessorOutputHelper:
        return self._output_helper
