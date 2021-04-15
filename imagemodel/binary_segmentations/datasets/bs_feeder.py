from abc import ABCMeta

from imagemodel.binary_segmentations.datasets.bs_feeder_helper import (
    BSFeederInputHelper,
    BSFeederOutputHelper,
)
from imagemodel.common.datasets.manipulator.manipulator import SupervisedManipulator


class BSFeeder(SupervisedManipulator, metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "input_helper")
            and callable(subclass.input_helper)
            and hasattr(subclass, "output_helper")
            and callable(subclass.output_helper)
            or NotImplemented
        )

    @property
    def input_helper(self) -> BSFeederInputHelper:
        pass

    @property
    def output_helper(self) -> BSFeederOutputHelper:
        pass
