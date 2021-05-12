from enum import Enum
from typing import Callable, Optional, Tuple

from imagemodel.common.datasets.feeder import Feeder
from imagemodel.common.datasets.pipeline import Pipeline
from imagemodel.reference_tracking.datasets.cell_tracking.feeder import (
    RTCellTrackingSample2TestFeeder, RTCellTrackingSampleTestFeeder,
    RTCellTrackingTrainingFeeder,
    RTCellTrackingValidationFeeder,
    RTGSCellTrackingSample2TestFeeder,
    RTGSCellTrackingSampleTestFeeder,
    RTGSCellTrackingTrainingFeeder,
    RTGSCellTrackingValidationFeeder
)
from imagemodel.reference_tracking.datasets.cell_tracking.preprocessor import RTCellTrackingPredictPreprocessor
from imagemodel.reference_tracking.datasets.pipeline import RTPipeline
from imagemodel.reference_tracking.datasets.rt_augmenter import RTAugmenter
from imagemodel.reference_tracking.datasets.rt_preprocessor import RTPreprocessor
from imagemodel.reference_tracking.datasets.rt_regularizer import BaseRTRegularizer, RTRegularizer


class Datasets(Enum):
    rt_cell_tracking_training_1 = "rt_cell_tracking_training_1"
    rt_cell_tracking_validation_1 = "rt_cell_tracking_validation_1"
    rt_gs_cell_tracking_training_1 = "rt_gs_cell_tracking_training_1"
    rt_gs_cell_tracking_validation_1 = "rt_gs_cell_tracking_validation_1"
    rt_cell_sample_test_1 = "rt_cell_sample_test_1"  # for Predict
    rt_cell_sample_2_test_1 = "rt_cell_sample_2_test_1"  # for Predict
    rt_gs_cell_sample_test_1 = "rt_gs_cell_sample_test_1"  # for Predict
    rt_gs_cell_sample_2_test_1 = "rt_gs_cell_sample_2_test_1"  # for Predict
    none = "none"
    
    def get_feeder(self) -> Optional[Feeder]:
        if self == Datasets.rt_cell_tracking_training_1:
            return RTCellTrackingTrainingFeeder()
        elif self == Datasets.rt_cell_tracking_validation_1:
            return RTCellTrackingValidationFeeder()
        elif self == Datasets.rt_gs_cell_tracking_training_1:
            return RTGSCellTrackingTrainingFeeder()
        elif self == Datasets.rt_gs_cell_tracking_validation_1:
            return RTGSCellTrackingValidationFeeder()
        elif self == Datasets.rt_cell_sample_test_1:
            return RTCellTrackingSampleTestFeeder()
        elif self == Datasets.rt_cell_sample_2_test_1:
            return RTCellTrackingSample2TestFeeder()
        elif self == Datasets.rt_gs_cell_sample_test_1:
            return RTGSCellTrackingSampleTestFeeder()
        elif self == Datasets.rt_gs_cell_sample_2_test_1:
            return RTGSCellTrackingSample2TestFeeder()
        else:
            return None
    
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
        elif self == Datasets.rt_cell_sample_test_1:
            preprocessor_func: Callable[[RTRegularizer], RTPreprocessor] = lambda \
                    el_rt_augmenter: RTCellTrackingPredictPreprocessor(el_rt_augmenter, 30, fill_with=(255, 255, 255))
            test_sample_feeder = RTCellTrackingSampleTestFeeder()
            rt_test_sample_pipeline = RTPipeline(
                    test_sample_feeder,
                    regularizer_func=regularizer_func,
                    preprocessor_func=preprocessor_func)
            return rt_test_sample_pipeline
        elif self == Datasets.rt_cell_sample_2_test_1:
            preprocessor_func: Callable[[RTRegularizer], RTPreprocessor] = lambda \
                    el_rt_augmenter: RTCellTrackingPredictPreprocessor(el_rt_augmenter, 30, fill_with=(255, 255, 255))
            test_sample_feeder = RTCellTrackingSample2TestFeeder()
            rt_test_sample_pipeline = RTPipeline(
                    test_sample_feeder,
                    regularizer_func=regularizer_func,
                    preprocessor_func=preprocessor_func)
            return rt_test_sample_pipeline
        elif self == Datasets.rt_gs_cell_sample_test_1:
            preprocessor_func: Callable[[RTRegularizer], RTPreprocessor] = lambda \
                    el_rt_augmenter: RTCellTrackingPredictPreprocessor(el_rt_augmenter, 30, fill_with=(255, 255, 255))
            test_sample_feeder = RTGSCellTrackingSampleTestFeeder()
            rt_test_sample_pipeline = RTPipeline(
                    test_sample_feeder,
                    regularizer_func=regularizer_func,
                    preprocessor_func=preprocessor_func)
            return rt_test_sample_pipeline
        elif self == Datasets.rt_gs_cell_sample_2_test_1:
            preprocessor_func: Callable[[RTRegularizer], RTPreprocessor] = lambda \
                    el_rt_augmenter: RTCellTrackingPredictPreprocessor(el_rt_augmenter, 30, fill_with=(255, 255, 255))
            test_sample_feeder = RTGSCellTrackingSample2TestFeeder()
            rt_test_sample_pipeline = RTPipeline(
                    test_sample_feeder,
                    regularizer_func=regularizer_func,
                    preprocessor_func=preprocessor_func)
            return rt_test_sample_pipeline
        else:
            return None
    
    @staticmethod
    def get_default() -> str:
        return Datasets.none.value
