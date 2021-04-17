from abc import ABCMeta

from imagemodel.binary_segmentations.datasets.bs_feeder_helper import (
    BSFeederInputHelper,
    BSFeederOutputHelper,
)
from imagemodel.common.datasets.manipulator.manipulator import SupervisedManipulator


class BSFeeder(SupervisedManipulator[BSFeederInputHelper, BSFeederOutputHelper],
               metaclass=ABCMeta):
    pass
