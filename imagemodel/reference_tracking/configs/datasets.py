from enum import Enum
from typing import Callable, Optional, Tuple

from imagemodel.common.datasets.pipeline import Pipeline
from imagemodel.reference_tracking.datasets.cell_tracking.feeder import (
    RTCellTrackingTrainingFeeder,
    RTCellTrackingValidationFeeder,
    RTGSCellTrackingSampleTestFeeder,
    RTGSCellTrackingTrainingFeeder,
    RTGSCellTrackingValidationFeeder
)
from imagemodel.reference_tracking.datasets.pipeline import RTPipeline
from imagemodel.reference_tracking.datasets.rt_augmenter import RTAugmenter
from imagemodel.reference_tracking.datasets.rt_regularizer import BaseRTRegularizer, RTRegularizer
from imagemodel.reference_tracking.datasets.single_pipeline.cell_tracking.feeder import (
    RTCellTrackingTrainingFeeder as RTSingleCellTrackingTrainingFeeder,
    RTCellTrackingValidationFeeder as RTSingleCellTrackingValidationFeeder,
    RTGSCellTrackingTrainingFeeder as RTSingleGSCellTrackingTrainingFeeder,
    RTGSCellTrackingValidationFeeder as RTSingleGSCellTrackingValidationFeeder
)
from imagemodel.reference_tracking.datasets.single_pipeline.pipeline import RTSinglePipeline
from imagemodel.reference_tracking.datasets.single_pipeline.rt_augmenter import RTAugmenter as SingleRTAugmenter
from imagemodel.reference_tracking.datasets.single_pipeline.rt_regularizer import (
    BaseRTRegularizer as BaseSingleRTRegularizer,
    RTRegularizer as SingleRTRegularizer
)


class Datasets(Enum):
    rt_cell_tracking_training_1 = "rt_cell_tracking_training_1"
    rt_cell_tracking_validation_1 = "rt_cell_tracking_validation_1"
    rt_cell_tracking_training_2 = "rt_cell_tracking_training_2"
    rt_cell_tracking_validation_2 = "rt_cell_tracking_validation_2"
    rt_gs_cell_tracking_training_1 = "rt_gs_cell_tracking_training_1"
    rt_gs_cell_tracking_validation_1 = "rt_gs_cell_tracking_validation_1"
    rt_gs_cell_tracking_training_2 = "rt_gs_cell_tracking_training_2"
    rt_gs_cell_tracking_validation_2 = "rt_gs_cell_tracking_validation_2"
    rt_gs_cell_sample_test_1 = "rt_gs_cell_sample_test_1"
    none = "none"
    
    def get_pipeline(self, resize_to: Tuple[int, int]) -> Optional[Pipeline]:
        regularizer_func: Callable[[RTAugmenter], RTRegularizer] = lambda el_bs_augmenter: BaseRTRegularizer(
                el_bs_augmenter,
                resize_to)
        if self == Datasets.rt_cell_tracking_training_1:
            training_feeder = RTCellTrackingTrainingFeeder()
            rt_training_pipeline = RTPipeline(training_feeder, regularizer_func=regularizer_func)
            return rt_training_pipeline
        elif self == Datasets.rt_cell_tracking_validation_1:
            validation_feeder = RTCellTrackingValidationFeeder()
            rt_validation_pipeline = RTPipeline(validation_feeder, regularizer_func=regularizer_func)
            return rt_validation_pipeline
        elif self == Datasets.rt_cell_tracking_training_2:
            regularizer_func: Callable[[SingleRTAugmenter], SingleRTRegularizer] = lambda \
                    el_bs_augmenter: BaseSingleRTRegularizer(el_bs_augmenter, resize_to)
            training_feeder = RTSingleCellTrackingTrainingFeeder()
            rt_training_pipeline = RTSinglePipeline(training_feeder, regularizer_func=regularizer_func)
            return rt_training_pipeline
        elif self == Datasets.rt_cell_tracking_validation_2:
            regularizer_func: Callable[[SingleRTAugmenter], SingleRTRegularizer] = lambda \
                    el_bs_augmenter: BaseSingleRTRegularizer(el_bs_augmenter, resize_to)
            validation_feeder = RTSingleCellTrackingValidationFeeder()
            rt_validation_pipeline = RTSinglePipeline(validation_feeder, regularizer_func=regularizer_func)
            return rt_validation_pipeline
        elif self == Datasets.rt_gs_cell_tracking_training_1:
            training_feeder = RTGSCellTrackingTrainingFeeder()
            rt_training_pipeline = RTPipeline(training_feeder, regularizer_func=regularizer_func)
            return rt_training_pipeline
        elif self == Datasets.rt_gs_cell_tracking_validation_1:
            validation_feeder = RTGSCellTrackingValidationFeeder()
            rt_validation_pipeline = RTPipeline(validation_feeder, regularizer_func=regularizer_func)
            return rt_validation_pipeline
        elif self == Datasets.rt_gs_cell_tracking_training_2:
            regularizer_func: Callable[[SingleRTAugmenter], SingleRTRegularizer] = lambda \
                    el_bs_augmenter: BaseSingleRTRegularizer(el_bs_augmenter, resize_to)
            training_feeder = RTSingleGSCellTrackingTrainingFeeder()
            rt_training_pipeline = RTSinglePipeline(training_feeder, regularizer_func=regularizer_func)
            return rt_training_pipeline
        elif self == Datasets.rt_gs_cell_tracking_validation_2:
            regularizer_func: Callable[[SingleRTAugmenter], SingleRTRegularizer] = lambda \
                    el_bs_augmenter: BaseSingleRTRegularizer(el_bs_augmenter, resize_to)
            validation_feeder = RTSingleGSCellTrackingValidationFeeder()
            rt_validation_pipeline = RTSinglePipeline(validation_feeder, regularizer_func=regularizer_func)
            return rt_validation_pipeline
        elif self == Datasets.rt_gs_cell_sample_test_1:
            test_sample_feeder = RTGSCellTrackingSampleTestFeeder()
            rt_test_sample_pipeline = RTPipeline(test_sample_feeder, regularizer_func=regularizer_func)
            return rt_test_sample_pipeline
        else:
            return None
    
    @staticmethod
    def get_default() -> str:
        return Datasets.none.value
