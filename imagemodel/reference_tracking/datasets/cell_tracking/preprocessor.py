from imagemodel.binary_segmentations.datasets.cell_tracking.preprocessor_helper import ClaheBSPreprocessorInOutHelper
from imagemodel.common.datasets.manipulator.manipulator import SupervisedManipulator


class RTCellTrackingPreprocessor(BaseRTPreprocessor, RTPreprocessor):
    def __init__(self, manipulator: SupervisedManipulator):
        super().__init__(manipulator)
        self._inout_helper = ClaheBSPreprocessorInOutHelper(
                input_datasets=manipulator.get_input_dataset(),
                output_datasets=manipulator.get_output_dataset())
