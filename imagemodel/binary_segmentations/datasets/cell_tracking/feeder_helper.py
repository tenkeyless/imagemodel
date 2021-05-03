import tensorflow as tf

from imagemodel.binary_segmentations.datasets.bs_feeder_helper import BSFeederInputHelper, BSFeederOutputHelper
from imagemodel.common.datasets.descriptor.cell_tracking_data_descriptor import CellTrackingDataDescriptor


class CellTrackingFeederInputHelper(BSFeederInputHelper):
    def __init__(self, data_descriptor: CellTrackingDataDescriptor):
        self._data_descriptor = data_descriptor
    
    def get_image(self) -> tf.data.Dataset:
        return self._data_descriptor.get_main_img_dataset()


class CellTrackingFeederOutputHelper(BSFeederOutputHelper):
    def __init__(self, data_descriptor: CellTrackingDataDescriptor):
        self._data_descriptor = data_descriptor
    
    def get_mask(self) -> tf.data.Dataset:
        return self._data_descriptor.get_main_bw_mask_dataset()
