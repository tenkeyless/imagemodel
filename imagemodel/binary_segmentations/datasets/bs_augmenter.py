from abc import ABCMeta, abstractmethod

from imagemodel.binary_segmentations.datasets.bs_augmenter_helper import (
    BaseBSAugmenterInOutHelper,
    BSAugmenterInputHelper,
    BSAugmenterOutputHelper,
)
from imagemodel.common.datasets.manipulator.manipulator import SupervisedManipulator


class BSAugmenter(SupervisedManipulator, metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "input_helper")
            and callable(subclass.input_helper)
            and hasattr(subclass, "output_helper")
            and callable(subclass.output_helper)
            or NotImplemented
        )

    @property
    def input_helper(self) -> BSAugmenterInputHelper:
        pass

    @property
    def output_helper(self) -> BSAugmenterOutputHelper:
        pass


class BaseBSAugmenter(BSAugmenter):
    """
    [summary]

    Examples
    --------
    >>> from imagemodel.binary_segmentations.datasets.oxford_iiit_pet.feeder import BSOxfordIIITPetTrainingFeeder
    >>> training_feeder = BSOxfordIIITPetTrainingFeeder()
    >>> from imagemodel.binary_segmentations.datasets.bs_augmenter import BaseBSAugmenter
    >>> augmenter = BaseBSAugmenter(manipulator=training_feeder)
    >>> import cv2
    >>> for index, inout in enumerate(augmenter.get_zipped_dataset().take(10)):
    ...     input_dataset = inout[0]
    ...     input_img = input_dataset[0]
    ...     cv2.imwrite("augmented_img_{:02d}.png".format(index), input_img.numpy())
    ...     output_dataset = inout[1]
    ...     output_mask = output_dataset[0]
    ...     cv2.imwrite("augmented_mask_{:02d}.png".format(index), output_mask.numpy()*255)
    """

    def __init__(self, manipulator: SupervisedManipulator):
        self._manipulator: SupervisedManipulator = manipulator
        self._inout_helper = BaseBSAugmenterInOutHelper(
            input_datasets=self._manipulator.get_input_dataset(),
            output_datasets=self._manipulator.get_output_dataset(),
        )

    @property
    def input_helper(self) -> BSAugmenterInputHelper:
        return self._inout_helper

    @property
    def output_helper(self) -> BSAugmenterOutputHelper:
        return self._inout_helper
