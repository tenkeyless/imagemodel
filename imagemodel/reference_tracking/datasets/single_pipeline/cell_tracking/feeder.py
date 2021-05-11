from abc import abstractmethod
from typing import Optional

import tensorflow as tf

from imagemodel.common.datasets.descriptor.cell_tracking_data_descriptor import CellTrackingDataDescriptor
from imagemodel.reference_tracking.datasets.rt_feeder import RTFeeder
from imagemodel.reference_tracking.datasets.rt_feeder_helper import RTFeederInputHelper, RTFeederOutputHelper
from imagemodel.reference_tracking.datasets.single_pipeline.cell_tracking.feeder_helper import (
    CellTrackingFeederInputHelper,
    CellTrackingFeederOutputHelper
)


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
    """
    Examples
    --------
    >>> from imagemodel.reference_tracking.datasets.rt_feeder import RTFeeder
    >>> from imagemodel.reference_tracking.datasets.\
             single_pipeline.cell_tracking.feeder import RTCellTrackingTrainingFeeder
    >>> training_feeder = RTCellTrackingTrainingFeeder()
    
    >>> import cv2
    >>> in_datas = training_feeder.get_input_dataset()
    >>> for _index, in_data in enumerate(in_datas[0].take(10)):
    ...     cv2.imwrite("{}_input_1.png".format(_index), in_data[0].numpy())
    ... cv2.imwrite("{}_input_2.png".format(_index), in_data[1].numpy())
    ... cv2.imwrite("{}_input_3.png".format(_index), in_data[2].numpy())
    ...
    >>> out_datas = training_feeder.get_output_dataset()
    >>> for _index, out_data in enumerate(out_datas[0].take(10)):
    ...     cv2.imwrite("{}_output_1.png".format(_index), out_data[0].numpy())
    ...     cv2.imwrite("{}_output_2.png".format(_index), out_data[1].numpy())
    ...     cv2.imwrite("{}_output_3.png".format(_index), out_data[2].numpy())
    ...     cv2.imwrite("{}_output_4.png".format(_index), out_data[3].numpy())
    ...
    """
    
    @property
    def feeder_data_description(self):
        return "Cell Tracking Training Dataset"
    
    def __init__(self):
        cell_tracking_data_descriptor = CellTrackingDataDescriptor(
                original_dataset=None,
                base_folder="data/tracking_training")
        super().__init__(cell_tracking_data_descriptor=cell_tracking_data_descriptor)


class RTCellTrackingValidationFeeder(RTCellTrackingFeeder):
    @property
    def feeder_data_description(self):
        return "Cell Tracking Validation Dataset"
    
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
