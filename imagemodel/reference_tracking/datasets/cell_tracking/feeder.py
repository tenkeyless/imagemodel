from abc import abstractmethod
from typing import Optional

import tensorflow as tf

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
    def filename_optional(self) -> Optional[tf.data.Dataset]:
        return self._cell_tracking_data_descriptor.get_filename_dataset()
    
    @property
    def input_helper(self) -> RTFeederInputHelper:
        return CellTrackingFeederInputHelper(data_descriptor=self._cell_tracking_data_descriptor)
    
    @property
    def output_helper(self) -> RTFeederOutputHelper:
        return CellTrackingFeederOutputHelper(data_descriptor=self._cell_tracking_data_descriptor)


class RTCellTrackingTrainingFeeder(RTCellTrackingFeeder):
    @property
    def feeder_data_description(self):
        return "Cell Tracking Training Dataset"
    
    def __init__(self):
        cell_tracking_data_descriptor = CellTrackingDataDescriptor(
                original_dataset=None,
                base_folder="data/tracking_training",
                cache=False)
        super().__init__(cell_tracking_data_descriptor=cell_tracking_data_descriptor)


class RTCellTrackingValidationFeeder(RTCellTrackingFeeder):
    @property
    def feeder_data_description(self):
        return "Cell Tracking Validation Dataset"
    
    def __init__(self):
        cell_tracking_data_descriptor = CellTrackingDataDescriptor(
                original_dataset=None,
                base_folder="data/tracking_validation",
                shuffle=False,
                cache=False)
        super().__init__(cell_tracking_data_descriptor=cell_tracking_data_descriptor)


class RTCellTrackingTestFeeder(RTCellTrackingFeeder):
    @property
    def feeder_data_description(self):
        return "Cell Tracking Test Dataset"
    
    def __init__(self):
        cell_tracking_data_descriptor = CellTrackingDataDescriptor(
                original_dataset=None,
                base_folder="data/tracking_test",
                shuffle=False)
        super().__init__(cell_tracking_data_descriptor=cell_tracking_data_descriptor)


class RTCellTrackingSampleTestFeeder(RTCellTrackingFeeder):
    @property
    def feeder_data_description(self):
        return "Cell Tracking Sample Test"
    
    def __init__(self):
        cell_tracking_data_descriptor = CellTrackingSampleTestDataDescriptor(
                original_dataset=None,
                base_folder="data/tracking_test",
                shuffle=False)
        super().__init__(cell_tracking_data_descriptor=cell_tracking_data_descriptor)


class RTCellTrackingSample2TestFeeder(RTCellTrackingFeeder):
    @property
    def feeder_data_description(self):
        return "Cell Tracking Sample 2 Test"
    
    def __init__(self):
        cell_tracking_data_descriptor = CellTrackingSampleTestDataDescriptor(
                original_dataset=None,
                base_folder="data/tracking_test2",
                shuffle=False)
        super().__init__(cell_tracking_data_descriptor=cell_tracking_data_descriptor)


class RTGSCellTrackingTrainingFeeder(RTCellTrackingFeeder):
    @property
    def feeder_data_description(self):
        return "Cell Tracking Training Google Storage Dataset"
    
    def __init__(self):
        cell_tracking_data_descriptor = CellTrackingDataDescriptor(
                original_dataset=None,
                base_folder="gs://cell_dataset/dataset/tracking_training",
                cache=True)
        super().__init__(cell_tracking_data_descriptor=cell_tracking_data_descriptor)


class RTGSCellTrackingValidationFeeder(RTCellTrackingFeeder):
    @property
    def feeder_data_description(self):
        return "Cell Tracking Validation Google Storage Dataset"
    
    def __init__(self):
        cell_tracking_data_descriptor = CellTrackingDataDescriptor(
                original_dataset=None,
                base_folder="gs://cell_dataset/dataset/tracking_validation",
                shuffle=False,
                cache=True)
        super().__init__(cell_tracking_data_descriptor=cell_tracking_data_descriptor)


class RTGSCellTrackingTestFeeder(RTCellTrackingFeeder):
    @property
    def feeder_data_description(self):
        return "Cell Tracking Test Google Storage Dataset"
    
    def __init__(self):
        cell_tracking_data_descriptor = CellTrackingDataDescriptor(
                original_dataset=None,
                base_folder="gs://cell_dataset/dataset/tracking_test",
                shuffle=False)
        super().__init__(cell_tracking_data_descriptor=cell_tracking_data_descriptor)


class RTGSCellTrackingSampleTestFeeder(RTCellTrackingFeeder):
    @property
    def feeder_data_description(self):
        return "Cell Tracking Sample Test Google Storage Dataset"
    
    def __init__(self):
        cell_tracking_data_descriptor = CellTrackingSampleTestDataDescriptor(
                original_dataset=None,
                base_folder="gs://cell_dataset/dataset/tracking_test",
                shuffle=False)
        super().__init__(cell_tracking_data_descriptor=cell_tracking_data_descriptor)


class RTGSCellTrackingSample2TestFeeder(RTCellTrackingFeeder):
    @property
    def feeder_data_description(self):
        return "Cell Tracking Sample 2 Test Google Storage Dataset"
    
    def __init__(self):
        cell_tracking_data_descriptor = CellTrackingSampleTestDataDescriptor(
                original_dataset=None,
                base_folder="gs://cell_dataset/dataset/tracking_test2",
                shuffle=False)
        super().__init__(cell_tracking_data_descriptor=cell_tracking_data_descriptor)
