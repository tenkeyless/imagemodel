from abc import ABCMeta, abstractmethod
from typing import TypeVar

from .feeder_helper import FeederInputHelper, FeederOutputHelper
from .manipulator.manipulator import SupervisedManipulator

HI = TypeVar('HI', bound=FeederInputHelper)
HO = TypeVar('HO', bound=FeederOutputHelper)


class Feeder(SupervisedManipulator[HI, HO], metaclass=ABCMeta):
    @property
    @abstractmethod
    def feeder_data_description(self):
        pass

    pass
