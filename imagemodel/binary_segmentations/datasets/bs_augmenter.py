from abc import ABCMeta

from imagemodel.binary_segmentations.datasets.bs_augmenter_helper import (
    BaseBSAugmenterInOutHelper,
    BSAugmenterInputHelper,
    BSAugmenterOutputHelper,
)
from imagemodel.common.datasets.augmenter import Augmenter
from imagemodel.common.datasets.manipulator.manipulator import SupervisedManipulator


class BSAugmenter(Augmenter[BSAugmenterInputHelper, BSAugmenterOutputHelper],
                  metaclass=ABCMeta):
    pass


class BaseBSAugmenter(BSAugmenter):
    """
    [summary]

    Examples
    --------
    >>> from imagemodel.binary_segmentations.datasets.oxford_iiit_pet import feeder
    >>> training_feeder = feeder.BSOxfordIIITPetTrainingFeeder()
    >>> from imagemodel.binary_segmentations.datasets import bs_augmenter
    >>> augmenter = bs_augmenter.BaseBSAugmenter(manipulator=training_feeder)
    >>> import cv2
    >>> for index, inout in enumerate(augmenter.get_zipped_dataset().take(10)):
    ...     input_dataset = inout[0]
    ...     input_img = input_dataset[0]
    ...     cv2.imwrite("augmented_{:02d}_img.png".format(index), input_img.numpy())
    ...     output_dataset = inout[1]
    ...     output_mask = output_dataset[0]
    ...     cv2.imwrite("augmented_{:02d}_mask.png".format(index), output_mask.numpy()*255)
    """

    def __init__(self, manipulator: SupervisedManipulator):
        self._inout_helper = BaseBSAugmenterInOutHelper(
            input_datasets=manipulator.get_input_dataset(),
            output_datasets=manipulator.get_output_dataset(),
        )

    @property
    def input_helper(self) -> BSAugmenterInputHelper:
        return self._inout_helper

    @property
    def output_helper(self) -> BSAugmenterOutputHelper:
        return self._inout_helper
