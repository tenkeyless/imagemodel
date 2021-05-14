import os
from typing import Optional

import tensorflow as tf

from imagemodel.common.datasets.descriptor.cell_tracking_data_descriptor import CellTrackingDataDescriptor
from imagemodel.common.datasets.descriptor.data_descriptor import BaseTFDataDescriptor


# Don't use this.
# Example for read from disk or google cloud storage.
class CellTrackingSampleTestDataDescriptor(CellTrackingDataDescriptor, BaseTFDataDescriptor):
    def __init__(self, original_dataset: Optional[tf.data.Dataset], base_folder: str, shuffle: bool = True):
        super().__init__(original_dataset=original_dataset, base_folder=base_folder, shuffle=shuffle)
        self.sample_folder: str = os.path.join(self.base_folder, "framed_sample")
        
        self.filename_base_folder: str = self.sample_folder
