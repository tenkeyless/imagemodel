# from enum import Enum
# from typing import Optional
#
# import tensorflow as tf
#
# from imagemodel.binary_segmentations.datasets.bs_feeder import BSFeeder
# from imagemodel.binary_segmentations.datasets.oxford_iiit_pet.feeder import (
#     BSOxfordIIITPetTrainingFeeder, BSOxfordIIITPetValidationFeeder, BSOxfordIIITPetTestFeeder
# )
# from imagemodel.binary_segmentations.datasets.pipeline import BSPipeline
#
#
# class Datasets(Enum):
#     oxford_iiit_pet_v3_training = "oxford_iiit_pet_v3_training"
#     oxford_iiit_pet_v3_validation = "oxford_iiit_pet_v3_validation"
#     oxford_iiit_pet_v3_test = "oxford_iiit_pet_v3_test"
#     none = "none"
#
#     def get_dataset(self) -> Optional[tf.data.Dataset]:
#         if self == Datasets.oxford_iiit_pet_v3_training:
#             training_feeder: BSFeeder = BSOxfordIIITPetTrainingFeeder()
#             bs_pipeline: BSPipeline = BSPipeline(training_feeder)
#             return bs_pipeline.get_zipped_dataset()
#         elif self == Datasets.oxford_iiit_pet_v3_validation:
#             validation_feeder: BSFeeder = BSOxfordIIITPetValidationFeeder()
#             bs_pipeline: BSPipeline = BSPipeline(validation_feeder)
#             return bs_pipeline.get_zipped_dataset()
#         elif self == Datasets.oxford_iiit_pet_v3_test:
#             test_feeder: BSFeeder = BSOxfordIIITPetTestFeeder()
#             bs_pipeline: BSPipeline = BSPipeline(test_feeder)
#             return bs_pipeline.get_zipped_dataset()
#         else:
#             return None
#
#     @staticmethod
#     def get_default() -> str:
#         return Datasets.none.value
