from typing import Callable

import tensorflow as tf
from imagemodel.binary_segmentations.datasets.bs_augmenter import (
    BaseBSAugmenter,
    BSAugmenter,
)
from imagemodel.binary_segmentations.datasets.bs_feeder import BSFeeder
from imagemodel.binary_segmentations.datasets.bs_preprocessor import (
    BaseBSPreprocessor,
    BSPreprocessor,
)
from imagemodel.binary_segmentations.datasets.bs_regularizer import (
    BaseBSRegularizer,
    BSRegularizer,
)


class BSPipeline:
    def __init__(
        self,
        feeder: BSFeeder,
        augmenter_func: Callable[[BSFeeder], BSAugmenter] = BaseBSAugmenter,
        regularizer_func: Callable[[BSAugmenter], BSRegularizer] = BaseBSRegularizer,
        preprocessor_func: Callable[
            [BSRegularizer], BSPreprocessor
        ] = BaseBSPreprocessor,
    ):
        """
        [summary]

        Parameters
        ----------
        feeder : BSFeeder
            [description]
        augmenter_func : Callable[[BSFeeder], BSAugmenter], optional, default=BaseBSAugmenter
            [description]
        regularizer_func : Callable[[BSAugmenter], BSRegularizer], optional, default=BaseBSRegularizer
            [description]
        preprocessor_func : Callable[[BSRegularizer], BSPreprocessor], optional, default=BaseBSPreprocessor
            [description]

        Examples
        --------
        >>> from imagemodel.binary_segmentations.datasets.oxford_iiit_pet.feeder import BSOxfordIIITPetTrainingFeeder
        >>> training_feeder = BSOxfordIIITPetTrainingFeeder()
        >>> from imagemodel.binary_segmentations.datasets.pipeline import BSPipeline
        >>> bs_pipeline = BSPipeline(training_feeder)
        >>> for d in bs_pipeline.get_zipped_dataset().take(1):
        ...     print(d[0][0].shape)
        ...     print(d[1][0].shape)
        """
        self.feeder: BSFeeder = feeder
        self.augmenter: BSAugmenter = augmenter_func(self.feeder)
        self.regularizer: BSRegularizer = regularizer_func(self.augmenter)
        self.preprocessor: BSPreprocessor = preprocessor_func(self.regularizer)

    def get_zipped_dataset(self) -> tf.data.Dataset:
        return self.preprocessor.get_zipped_dataset()
