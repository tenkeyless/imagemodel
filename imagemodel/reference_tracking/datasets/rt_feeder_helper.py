from typing import List

import tensorflow as tf

from imagemodel.common.datasets.feeder_helper import FeederInputHelper, FeederOutputHelper


class RTFeederInputHelper(FeederInputHelper):
    def get_main_image(self) -> tf.data.Dataset:
        pass
    
    def get_ref_image(self) -> tf.data.Dataset:
        pass
    
    def get_ref_color_label(self) -> tf.data.Dataset:
        pass
    
    def get_inputs(self) -> List[tf.data.Dataset]:
        return [self.get_main_image(), self.get_ref_image(), self.get_ref_color_label()]


class RTFeederOutputHelper(FeederOutputHelper):
    def get_main_bw_mask(self) -> tf.data.Dataset:
        pass
    
    def get_ref_bw_mask(self) -> tf.data.Dataset:
        pass
    
    def get_main_color_label(self) -> tf.data.Dataset:
        pass
    
    def get_outputs(self) -> List[tf.data.Dataset]:
        return [self.get_main_bw_mask(), self.get_ref_bw_mask(), self.get_main_color_label()]
