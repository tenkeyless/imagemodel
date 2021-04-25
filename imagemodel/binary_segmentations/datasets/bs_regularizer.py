from abc import ABCMeta
from typing import Tuple

from imagemodel.binary_segmentations.datasets.bs_regularizer_helper import (
    BaseBSRegularizerInOutHelper,
    BSRegularizerInputHelper,
    BSRegularizerOutputHelper,
)
from imagemodel.common.datasets.manipulator.manipulator import SupervisedManipulator
from imagemodel.common.datasets.regularizer import Regularizer


class BSRegularizer(Regularizer[BSRegularizerInputHelper, BSRegularizerOutputHelper], metaclass=ABCMeta):
    pass


class BaseBSRegularizer(BSRegularizer):
    def __init__(self, manipulator: SupervisedManipulator, height_width_tuple: Tuple[int, int] = (256, 256)):
        self._inout_helper = BaseBSRegularizerInOutHelper(
                input_datasets=manipulator.get_input_dataset(),
                output_datasets=manipulator.get_output_dataset(),
                height_width_tuple=height_width_tuple)

    @property
    def input_helper(self) -> BSRegularizerInputHelper:
        return self._inout_helper

    @property
    def output_helper(self) -> BSRegularizerOutputHelper:
        return self._inout_helper
