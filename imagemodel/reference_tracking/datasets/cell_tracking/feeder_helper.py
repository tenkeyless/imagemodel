import tensorflow as tf

from imagemodel.common.datasets.descriptor.cell_tracking_data_descriptor import CellTrackingDataDescriptor
from imagemodel.reference_tracking.datasets.rt_feeder_helper import RTFeederInputHelper, RTFeederOutputHelper


class CellTrackingFeederInputHelper(RTFeederInputHelper):
    def __init__(self, data_descriptor: CellTrackingDataDescriptor):
        self._data_descriptor = data_descriptor
    
    def get_main_image(self) -> tf.data.Dataset:
        return self._data_descriptor.get_main_img_dataset()
    
    def get_ref_image(self) -> tf.data.Dataset:
        return self._data_descriptor.get_ref_img_dataset()
    
    def get_ref_color_label(self) -> tf.data.Dataset:
        return self._data_descriptor.get_ref_mask_dataset()


class CellTrackingFeederOutputHelper(RTFeederOutputHelper):
    def __init__(self, data_descriptor: CellTrackingDataDescriptor):
        self._data_descriptor = data_descriptor
    
    def get_main_bw_mask(self) -> tf.data.Dataset:
        return self._data_descriptor.get_main_bw_mask_dataset()
    
    def get_ref_bw_mask(self) -> tf.data.Dataset:
        return self._data_descriptor.get_ref_bw_mask_dataset()
    
    def get_main_color_label(self) -> tf.data.Dataset:
        return self._data_descriptor.get_main_mask_dataset()
