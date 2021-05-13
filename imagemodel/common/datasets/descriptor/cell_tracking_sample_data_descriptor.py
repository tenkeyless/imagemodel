import os
from typing import Optional

import tensorflow as tf

from imagemodel.common.datasets.descriptor.cell_tracking_data_descriptor import CellTrackingDataDescriptor
from imagemodel.common.datasets.descriptor.data_descriptor import BaseTFDataDescriptor


def get_filename_from_fullpath(name):
    return tf.strings.split(name, sep="/")[-1]


# Don't use this.
# Example for read from disk or google cloud storage.
class CellTrackingSampleTestDataDescriptor(CellTrackingDataDescriptor, BaseTFDataDescriptor):
    def __init__(self, original_dataset: Optional[tf.data.Dataset], base_folder: str, shuffle: bool = True):
        super().__init__(original_dataset=original_dataset, base_folder=base_folder, shuffle=shuffle)
        self.sample_folder: str = os.path.join(self.base_folder, "framed_sample")
        
        self.base_file_dataset = tf.data.Dataset.list_files(self.sample_folder + "/*", shuffle=False).map(
                get_filename_from_fullpath)
        self.base_file_dataset_len = len(self.base_file_dataset)
