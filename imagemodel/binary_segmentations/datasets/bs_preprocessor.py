from abc import ABCMeta

from imagemodel.binary_segmentations.datasets.bs_preprocessor_helper import (
    BaseBSPreprocessorInOutHelper,
    BSPreprocessorInputHelper,
    BSPreprocessorOutputHelper,
)
from imagemodel.common.datasets.manipulator.manipulator import SupervisedManipulator


class BSPreprocessor(SupervisedManipulator[BSPreprocessorInputHelper, BSPreprocessorOutputHelper],
                     metaclass=ABCMeta):
    pass


class BaseBSPreprocessor(BSPreprocessor):
    def __init__(self, manipulator: SupervisedManipulator):
        self._inout_helper = BaseBSPreprocessorInOutHelper(
            input_datasets=manipulator.get_input_dataset(),
            output_datasets=manipulator.get_output_dataset(),
        )

    @property
    def input_helper(self) -> BSPreprocessorInputHelper:
        return self._inout_helper

    @property
    def output_helper(self) -> BSPreprocessorOutputHelper:
        return self._inout_helper
