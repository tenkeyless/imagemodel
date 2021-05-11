from abc import ABCMeta
from typing import Optional

import tensorflow as tf

from imagemodel.common.datasets.feeder import Feeder
from imagemodel.reference_tracking.datasets.rt_feeder_helper import RTFeederInputHelper, RTFeederOutputHelper


class RTFeeder(Feeder[RTFeederInputHelper, RTFeederOutputHelper], metaclass=ABCMeta):
    @property
    def filename_optional(self) -> Optional[tf.data.Dataset]:
        return None
    
    pass
