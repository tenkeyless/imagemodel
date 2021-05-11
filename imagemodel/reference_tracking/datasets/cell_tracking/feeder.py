from abc import abstractmethod

from imagemodel.common.datasets.descriptor.cell_tracking_data_descriptor import CellTrackingDataDescriptor
from imagemodel.common.datasets.descriptor.cell_tracking_sample_data_descriptor import \
    CellTrackingSampleTestDataDescriptor
from imagemodel.reference_tracking.datasets.cell_tracking.feeder_helper import (
    CellTrackingFeederInputHelper,
    CellTrackingFeederOutputHelper
)
from imagemodel.reference_tracking.datasets.rt_feeder import RTFeeder
from imagemodel.reference_tracking.datasets.rt_feeder_helper import RTFeederInputHelper, RTFeederOutputHelper


class RTCellTrackingFeeder(RTFeeder):
    @abstractmethod
    def __init__(self, cell_tracking_data_descriptor: CellTrackingDataDescriptor):
        self._cell_tracking_data_descriptor = cell_tracking_data_descriptor
    
    @property
    def input_helper(self) -> RTFeederInputHelper:
        return CellTrackingFeederInputHelper(data_descriptor=self._cell_tracking_data_descriptor)
    
    @property
    def output_helper(self) -> RTFeederOutputHelper:
        return CellTrackingFeederOutputHelper(data_descriptor=self._cell_tracking_data_descriptor)


class RTCellTrackingTrainingFeeder(RTCellTrackingFeeder):
    @property
    def feeder_data_description(self):
        return "Cell Tracking Training Google Storage Dataset"
    
    def __init__(self):
        cell_tracking_data_descriptor = CellTrackingDataDescriptor(
                original_dataset=None,
                base_folder="data/tracking_training")
        super().__init__(cell_tracking_data_descriptor=cell_tracking_data_descriptor)


class RTCellTrackingValidationFeeder(RTCellTrackingFeeder):
    @property
    def feeder_data_description(self):
        return "Cell Tracking Validation Google Storage Dataset"
    
    def __init__(self):
        cell_tracking_data_descriptor = CellTrackingDataDescriptor(
                original_dataset=None,
                base_folder="data/tracking_validation")
        super().__init__(cell_tracking_data_descriptor=cell_tracking_data_descriptor)


class RTGSCellTrackingTrainingFeeder(RTCellTrackingFeeder):
    @property
    def feeder_data_description(self):
        return "Cell Tracking Training Google Storage Dataset"
    
    def __init__(self):
        cell_tracking_data_descriptor = CellTrackingDataDescriptor(
                original_dataset=None,
                base_folder="gs://cell_dataset/dataset/tracking_training")
        super().__init__(cell_tracking_data_descriptor=cell_tracking_data_descriptor)


class RTGSCellTrackingValidationFeeder(RTCellTrackingFeeder):
    @property
    def feeder_data_description(self):
        return "Cell Tracking Validation Google Storage Dataset"
    
    def __init__(self):
        cell_tracking_data_descriptor = CellTrackingDataDescriptor(
                original_dataset=None,
                base_folder="gs://cell_dataset/dataset/tracking_validation")
        super().__init__(cell_tracking_data_descriptor=cell_tracking_data_descriptor)


class RTGSCellTrackingTestFeeder(RTCellTrackingFeeder):
    @property
    def feeder_data_description(self):
        return "Cell Tracking Test Google Storage Dataset"
    
    def __init__(self):
        cell_tracking_data_descriptor = CellTrackingDataDescriptor(
                original_dataset=None,
                base_folder="gs://cell_dataset/dataset/tracking_test")
        super().__init__(cell_tracking_data_descriptor=cell_tracking_data_descriptor)


class RTGSCellTrackingSampleTestFeeder(RTCellTrackingFeeder):
    @property
    def feeder_data_description(self):
        return "Cell Tracking Sample Test Google Storage Dataset"
    
    def __init__(self):
        cell_tracking_data_descriptor = CellTrackingSampleTestDataDescriptor(
                original_dataset=None,
                base_folder="gs://cell_dataset/dataset/tracking_test")
        super().__init__(cell_tracking_data_descriptor=cell_tracking_data_descriptor)
