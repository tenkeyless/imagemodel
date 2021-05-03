from enum import Enum
from typing import Optional

from imagemodel.binary_segmentations.datasets.oxford_iiit_pet import feeder
from imagemodel.binary_segmentations.datasets.pipeline import BSPipeline
from imagemodel.binary_segmentations.models.trainers.augmenter import FlipBSAugmenter
from imagemodel.common.datasets.pipeline import Pipeline


class Datasets(Enum):
    bs_oxford_iiit_pet_v3_training_1 = "bs_oxford_iiit_pet_v3_training_1"
    bs_oxford_iiit_pet_v3_training_2 = "bs_oxford_iiit_pet_v3_training_2"
    bs_oxford_iiit_pet_v3_validation_1 = "bs_oxford_iiit_pet_v3_validation_1"
    bs_oxford_iiit_pet_v3_test_1 = "bs_oxford_iiit_pet_v3_test_1"
    none = "none"
    
    def get_pipeline(self) -> Optional[Pipeline]:
        if self == Datasets.bs_oxford_iiit_pet_v3_training_1:
            training_feeder = feeder.BSOxfordIIITPetTrainingFeeder()
            bs_training_pipeline = BSPipeline(training_feeder, augmenter_func=FlipBSAugmenter)
            return bs_training_pipeline
        if self == Datasets.bs_oxford_iiit_pet_v3_training_2:
            training_feeder = feeder.BSOxfordIIITPetTrainingFeeder()
            bs_training_pipeline = BSPipeline(training_feeder)
            return bs_training_pipeline
        elif self == Datasets.bs_oxford_iiit_pet_v3_validation_1:
            validation_feeder = feeder.BSOxfordIIITPetValidationFeeder()
            bs_validation_pipeline = BSPipeline(validation_feeder)
            return bs_validation_pipeline
        elif self == Datasets.bs_oxford_iiit_pet_v3_test_1:
            test_feeder = feeder.BSOxfordIIITPetTestFeeder()
            bs_test_feeder = BSPipeline(test_feeder)
            return bs_test_feeder
        else:
            return None
    
    @staticmethod
    def get_default() -> str:
        return Datasets.none.value
