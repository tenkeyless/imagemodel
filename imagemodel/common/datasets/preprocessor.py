from abc import ABCMeta
from typing import TypeVar

from .manipulator.manipulator import SupervisedManipulator
from .preprocessor_helper import PreprocessorOutputHelper, PreprocessorInputHelper

HI = TypeVar('HI', bound=PreprocessorInputHelper)
HO = TypeVar('HO', bound=PreprocessorOutputHelper)


class Preprocessor(SupervisedManipulator[HI, HO],
                   metaclass=ABCMeta):
    pass
