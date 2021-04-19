from abc import ABCMeta
from typing import TypeVar

from .manipulator.manipulator import SupervisedManipulator
from .regularizer_helper import RegularizerOutputHelper, RegularizerInputHelper

HI = TypeVar('HI', bound=RegularizerInputHelper)
HO = TypeVar('HO', bound=RegularizerOutputHelper)


class Regularizer(SupervisedManipulator[HI, HO],
                  metaclass=ABCMeta):
    pass
