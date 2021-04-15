from abc import ABCMeta
from typing import Tuple

from imagemodel.binary_segmentations.datasets.bs_regularizer_helper import (
    BaseBSRegularizerInOutHelper,
    BSRegularizerInputHelper,
    BSRegularizerOutputHelper,
)
from imagemodel.common.datasets.manipulator.manipulator import SupervisedManipulator


class BSRegularizer(SupervisedManipulator, metaclass=ABCMeta):
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
    def input_helper(self) -> BSRegularizerInputHelper:
        pass

    @property
    def output_helper(self) -> BSRegularizerOutputHelper:
        pass


class BaseBSRegularizer(BSRegularizer):
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

    def __init__(
        self,
        manipulator: SupervisedManipulator,
        height_width_tuple: Tuple[int, int] = (256, 256),
    ):
        self._inout_helper = BaseBSRegularizerInOutHelper(
            input_datasets=manipulator.get_input_dataset(),
            output_datasets=manipulator.get_output_dataset(),
            height_width_tuple=height_width_tuple,
        )

    @property
    def input_helper(self) -> BSRegularizerInputHelper:
        return self._inout_helper

    @property
    def output_helper(self) -> BSRegularizerOutputHelper:
        return self._inout_helper
