from abc import abstractmethod

from imagemodel.binary_segmentations.datasets.bs_augmenter import BSAugmenter
from imagemodel.binary_segmentations.datasets.bs_augmenter_helper import (
    BSAugmenterInputHelper,
    BSAugmenterOutputHelper,
)
from imagemodel.binary_segmentations.datasets.oxford_iiit_pet.augmenter_helper import (
    BSOxfordIIITPetAugmenterInputHelper,
    BSOxfordIIITPetAugmenterOutputHelper,
)
from imagemodel.common.datasets.manipulator.manipulator import SupervisedManipulator


class BSOxfordIIITPetAugmenter(BSAugmenter):
    @abstractmethod
    def __init__(self, manipulator: SupervisedManipulator):
        self._manipulator: SupervisedManipulator = manipulator

    @property
    def input_helper(self) -> BSAugmenterInputHelper:
        return BSOxfordIIITPetAugmenterInputHelper(
            datasets=self._manipulator.get_input_dataset()
        )

    @property
    def output_helper(self) -> BSAugmenterOutputHelper:
        return BSOxfordIIITPetAugmenterOutputHelper(
            datasets=self._manipulator.get_output_dataset()
        )
