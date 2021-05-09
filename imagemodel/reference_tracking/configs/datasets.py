from enum import Enum
from typing import Callable, Optional, Tuple

from imagemodel.common.datasets.pipeline import Pipeline
from imagemodel.reference_tracking.datasets.cell_tracking.feeder import (
    RTCellTrackingTrainingFeeder, RTCellTrackingValidationFeeder, RTGSCellTrackingTrainingFeeder,
    RTGSCellTrackingValidationFeeder
)
from imagemodel.reference_tracking.datasets.pipeline import RTPipeline
from imagemodel.reference_tracking.datasets.rt_augmenter import RTAugmenter
from imagemodel.reference_tracking.datasets.rt_regularizer import BaseRTRegularizer, RTRegularizer


class Datasets(Enum):
    rt_cell_tracking_training_1 = "rt_cell_tracking_training_1"
    rt_cell_tracking_validation_1 = "rt_cell_tracking_validation_1"
    rt_gs_cell_tracking_training_1 = "rt_gs_cell_tracking_training_1"
    rt_gs_cell_tracking_validation_1 = "rt_gs_cell_tracking_validation_1"
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
        elif self == Datasets.rt_gs_cell_tracking_training_1:
            training_feeder = RTGSCellTrackingTrainingFeeder()
            rt_training_pipeline = RTPipeline(training_feeder, regularizer_func=regularizer_func)
            return rt_training_pipeline
        elif self == Datasets.rt_gs_cell_tracking_validation_1:
            validation_feeder = RTGSCellTrackingValidationFeeder()
            rt_validation_pipeline = RTPipeline(validation_feeder, regularizer_func=regularizer_func)
            return rt_validation_pipeline
        else:
            return None
    
    @staticmethod
    def get_default() -> str:
        return Datasets.none.value
