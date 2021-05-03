from abc import abstractmethod

from imagemodel.binary_segmentations.datasets.bs_feeder import BSFeeder
from imagemodel.binary_segmentations.datasets.bs_feeder_helper import (BSFeederInputHelper, BSFeederOutputHelper)
from imagemodel.binary_segmentations.datasets.cell_tracking.feeder_helper import (
    CellTrackingFeederInputHelper,
    CellTrackingFeederOutputHelper
)
from imagemodel.common.datasets.descriptor.cell_tracking_data_descriptor import CellTrackingDataDescriptor


class BSCellTrackingFeeder(BSFeeder):
    @abstractmethod
    def __init__(self, cell_tracking_data_descriptor: CellTrackingDataDescriptor):
        self._cell_tracking_data_descriptor = cell_tracking_data_descriptor
    
    @property
    def input_helper(self) -> BSFeederInputHelper:
        return CellTrackingFeederInputHelper(data_descriptor=self._cell_tracking_data_descriptor)
    
    @property
    def output_helper(self) -> BSFeederOutputHelper:
        return CellTrackingFeederOutputHelper(data_descriptor=self._cell_tracking_data_descriptor)


class BSCellTrackingTrainingFeeder(BSCellTrackingFeeder):
    @property
    def feeder_data_description(self):
        return "Cell Tracking Training Dataset"
    
    def __init__(self):
        cell_tracking_data_descriptor = CellTrackingDataDescriptor(
                original_dataset=None,
                base_folder="gs://cell_dataset/dataset/tracking_training")
        super().__init__(cell_tracking_data_descriptor=cell_tracking_data_descriptor)


class BSCellTrackingValidationFeeder(BSCellTrackingFeeder):
    @property
    def feeder_data_description(self):
        return "Cell Tracking Validation Dataset"
    
    def __init__(self):
        cell_tracking_data_descriptor = CellTrackingDataDescriptor(
                original_dataset=None,
                base_folder="gs://cell_dataset/dataset/tracking_validation")
        super().__init__(cell_tracking_data_descriptor=cell_tracking_data_descriptor)


class BSCellTrackingTestFeeder(BSCellTrackingFeeder):
    @property
    def feeder_data_description(self):
        return "Cell Tracking Test Dataset"
    
    def __init__(self):
        cell_tracking_data_descriptor = CellTrackingDataDescriptor(
                original_dataset=None,
                base_folder="gs://cell_dataset/dataset/tracking_test")
        super().__init__(cell_tracking_data_descriptor=cell_tracking_data_descriptor)
