from abc import ABCMeta

from imagemodel.common.datasets.augmenter import Augmenter
from imagemodel.common.datasets.manipulator.manipulator import SupervisedManipulator
from imagemodel.reference_tracking.datasets.single_pipeline.rt_augmenter_helper import (
    BaseRTAugmenterInputHelper,
    BaseRTAugmenterOutputHelper,
    RTAugmenterInputHelper,
    RTAugmenterOutputHelper
)


class RTAugmenter(Augmenter[RTAugmenterInputHelper, RTAugmenterOutputHelper], metaclass=ABCMeta):
    pass


class BaseRTAugmenter(RTAugmenter):
    def __init__(self, manipulator: SupervisedManipulator):
        self._input_helper = BaseRTAugmenterInputHelper(manipulator.get_input_dataset())
        self._output_helper = BaseRTAugmenterOutputHelper(manipulator.get_output_dataset())
    
    @property
    def input_helper(self) -> RTAugmenterInputHelper:
        return self._input_helper
    
    @property
    def output_helper(self) -> RTAugmenterOutputHelper:
        return self._output_helper
