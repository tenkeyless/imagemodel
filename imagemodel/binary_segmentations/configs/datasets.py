from enum import Enum
from typing import Callable, Optional, Tuple

from imagemodel.binary_segmentations.datasets.augmenters.flip_augmenter import FlipBSAugmenter
from imagemodel.binary_segmentations.datasets.bs_augmenter import BSAugmenter
from imagemodel.binary_segmentations.datasets.bs_regularizer import BSRegularizer, BaseBSRegularizer
from imagemodel.binary_segmentations.datasets.cell_tracking.feeder import (
    BSGSCellTrackingTrainingFeeder,
    BSGSCellTrackingValidationFeeder
)
from imagemodel.binary_segmentations.datasets.cell_tracking.preprocessor import BSCellTrackingPreprocessor
from imagemodel.binary_segmentations.datasets.oxford_iiit_pet import feeder
from imagemodel.binary_segmentations.datasets.pipeline import BSPipeline
from imagemodel.common.datasets.pipeline import Pipeline


class Datasets(Enum):
    """
    Examples
    --------
    # >>> from imagemodel.binary_segmentations.datasets.cell_tracking.preprocessor import BSCellTrackingPreprocessor
    # >>> from imagemodel.binary_segmentations.datasets.cell_tracking.feeder import (
    # ...     BSCellTrackingTrainingFeeder,
    # ...     BSCellTrackingValidationFeeder
    # ... )
    # >>> from imagemodel.binary_segmentations.datasets.pipeline import BSPipeline
    >>> training_feeder = BSGSCellTrackingTrainingFeeder()
    >>> bs_training_pipeline = BSPipeline(training_feeder, preprocessor_func=BSCellTrackingPreprocessor)
    >>>
    >>> training_feeder.get_zipped_dataset()
    >>> bs_training_pipeline.get_zipped_dataset()
    >>>
    >>> import cv2
    >>> for _id, d in enumerate(bs_training_pipeline.get_zipped_dataset().take(10)):
    ...    cv2.imwrite("d_{}_d1.png".format(_id), d[0][0].numpy()*255)
    ...    cv2.imwrite("d_{}_d2.png".format(_id), d[1][0].numpy()*255)
    ...
    """
    bs_oxford_iiit_pet_v3_training_1 = "bs_oxford_iiit_pet_v3_training_1"
    bs_oxford_iiit_pet_v3_training_2 = "bs_oxford_iiit_pet_v3_training_2"
    bs_oxford_iiit_pet_v3_validation_1 = "bs_oxford_iiit_pet_v3_validation_1"
    bs_oxford_iiit_pet_v3_test_1 = "bs_oxford_iiit_pet_v3_test_1"
    bs_gs_cell_tracking_training_1 = "bs_gs_cell_tracking_training_1"
    bs_gs_cell_tracking_validation_1 = "bs_gs_cell_tracking_validation_1"
    none = "none"
    
    def get_pipeline(self, resize_to: Tuple[int, int]) -> Optional[Pipeline]:
        regularizer_func: Callable[[BSAugmenter], BSRegularizer] = lambda el_bs_augmenter: BaseBSRegularizer(
                el_bs_augmenter,
                resize_to)
        if self == Datasets.bs_oxford_iiit_pet_v3_training_1:
            training_feeder = feeder.BSOxfordIIITPetTrainingFeeder()
            bs_training_pipeline = BSPipeline(
                    training_feeder,
                    regularizer_func=regularizer_func,
                    augmenter_func=FlipBSAugmenter)
            return bs_training_pipeline
        if self == Datasets.bs_oxford_iiit_pet_v3_training_2:
            training_feeder = feeder.BSOxfordIIITPetTrainingFeeder()
            bs_training_pipeline = BSPipeline(training_feeder, regularizer_func=regularizer_func)
            return bs_training_pipeline
        elif self == Datasets.bs_oxford_iiit_pet_v3_validation_1:
            validation_feeder = feeder.BSOxfordIIITPetValidationFeeder()
            bs_validation_pipeline = BSPipeline(validation_feeder, regularizer_func=regularizer_func)
            return bs_validation_pipeline
        elif self == Datasets.bs_oxford_iiit_pet_v3_test_1:
            test_feeder = feeder.BSOxfordIIITPetTestFeeder()
            bs_test_feeder = BSPipeline(test_feeder, regularizer_func=regularizer_func)
            return bs_test_feeder
        elif self == Datasets.bs_gs_cell_tracking_training_1:
            training_feeder = BSGSCellTrackingTrainingFeeder()
            bs_training_pipeline = BSPipeline(
                    training_feeder,
                    regularizer_func=regularizer_func,
                    preprocessor_func=BSCellTrackingPreprocessor)
            return bs_training_pipeline
        elif self == Datasets.bs_gs_cell_tracking_validation_1:
            validation_feeder = BSGSCellTrackingValidationFeeder()
            bs_validation_pipeline = BSPipeline(
                    validation_feeder,
                    regularizer_func=regularizer_func,
                    preprocessor_func=BSCellTrackingPreprocessor)
            return bs_validation_pipeline
        else:
            return None
    
    @staticmethod
    def get_default() -> str:
        return Datasets.none.value
