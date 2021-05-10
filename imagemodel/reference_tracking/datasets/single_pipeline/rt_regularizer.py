from abc import ABCMeta
from typing import Tuple

from imagemodel.common.datasets.manipulator.manipulator import SupervisedManipulator
from imagemodel.common.datasets.regularizer import Regularizer
from imagemodel.reference_tracking.datasets.single_pipeline.rt_regularizer_helper import (
    BaseRTRegularizerInputHelper,
    BaseRTRegularizerOutputHelper,
    RTRegularizerInputHelper,
    RTRegularizerOutputHelper
)


class RTRegularizer(Regularizer[RTRegularizerInputHelper, RTRegularizerOutputHelper], metaclass=ABCMeta):
    pass


class BaseRTRegularizer(RTRegularizer):
    def __init__(self, manipulator: SupervisedManipulator, height_width_tuple: Tuple[int, int] = (256, 256)):
        self._input_helper = BaseRTRegularizerInputHelper(
                manipulator.get_input_dataset(),
                height_width_tuple=height_width_tuple)
        self._output_helper = BaseRTRegularizerOutputHelper(
                manipulator.get_output_dataset(),
                height_width_tuple=height_width_tuple)
    
    @property
    def input_helper(self) -> RTRegularizerInputHelper:
        return self._input_helper
    
    @property
    def output_helper(self) -> RTRegularizerOutputHelper:
        return self._output_helper
