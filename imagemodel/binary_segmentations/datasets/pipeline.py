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
            regularizer_func: Callable[[BSAugmenter], BSRegularizer] = (
                    lambda el_bs_augmenter: BaseBSRegularizer(el_bs_augmenter, (256, 256))),
            preprocessor_func: Callable[
                [BSRegularizer], BSPreprocessor
            ] = BaseBSPreprocessor,
    ):
        """
        Pipeline for Binary Segmentation.

        Parameters
        ----------
        feeder: BSFeeder
        augmenter_func: Callable[[BSFeeder], BSAugmenter], default=BaseBSAugmenter
        regularizer_func: Callable[[BSAugmenter], BSRegularizer], default=BaseBSRegularizer
        preprocessor_func: Callable[[BSRegularizer], BSPreprocessor], default=BaseBSPreprocessor
        
        Examples
        --------
        >>> from imagemodel.binary_segmentations.datasets.oxford_iiit_pet import feeder
        >>> training_feeder = feeder.BSOxfordIIITPetTrainingFeeder()
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
