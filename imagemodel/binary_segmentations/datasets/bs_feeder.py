from abc import ABCMeta

from imagemodel.binary_segmentations.datasets.bs_feeder_helper import BSFeederInputHelper, BSFeederOutputHelper
from imagemodel.common.datasets.feeder import Feeder


class BSFeeder(Feeder[BSFeederInputHelper, BSFeederOutputHelper], metaclass=ABCMeta):
    pass
