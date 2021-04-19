from abc import ABCMeta
from typing import TypeVar

from .feeder_helper import FeederInputHelper, FeederOutputHelper
from .manipulator.manipulator import SupervisedManipulator

HI = TypeVar('HI', bound=FeederInputHelper)
HO = TypeVar('HO', bound=FeederOutputHelper)


class Feeder(SupervisedManipulator[HI, HO],
             metaclass=ABCMeta):
    pass
