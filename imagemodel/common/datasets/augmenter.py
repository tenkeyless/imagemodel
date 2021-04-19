from abc import ABCMeta
from typing import TypeVar

from .augmenter_helper import AugmenterInputHelper, AugmenterOutputHelper
from .manipulator.manipulator import SupervisedManipulator

HI = TypeVar('HI', bound=AugmenterInputHelper)
HO = TypeVar('HO', bound=AugmenterOutputHelper)


class Augmenter(SupervisedManipulator[HI, HO],
                metaclass=ABCMeta):
    pass
