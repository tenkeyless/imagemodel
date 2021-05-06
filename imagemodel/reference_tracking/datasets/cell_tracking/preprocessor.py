from imagemodel.common.datasets.manipulator.manipulator import SupervisedManipulator
from imagemodel.reference_tracking.datasets.cell_tracking.preprocessor_helper import ClaheRTPreprocessorInputHelper
from imagemodel.reference_tracking.datasets.rt_preprocessor import BaseRTPreprocessor, RTPreprocessor


class RTCellTrackingPreprocessor(BaseRTPreprocessor, RTPreprocessor):
    def __init__(self, manipulator: SupervisedManipulator, bin_size: int):
        super().__init__(manipulator, bin_size)
        self._input_helper = ClaheRTPreprocessorInputHelper(datasets=manipulator.get_input_dataset(), bin_size=bin_size)
