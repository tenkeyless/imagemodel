from abc import ABCMeta

from imagemodel.common.datasets.feeder import Feeder
from imagemodel.reference_tracking.datasets.rt_feeder_helper import RTFeederInputHelper, RTFeederOutputHelper


class RTFeeder(Feeder[RTFeederInputHelper, RTFeederOutputHelper], metaclass=ABCMeta):
    pass
