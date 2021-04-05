from typing import List, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds
from imagemodel.binary_segmentations.datasets.oxford_iiit_pet_data_feeder import (
    BaseOxfordPetDataFeeder,
)
from imagemodel.binary_segmentations.datasets.oxford_iiit_pet_helper import (
    BaseBSOxfordIIITPetFeederHelper,
    BSOxfordIIITPetFeederHelper,
)
from imagemodel.common.datasets.interfaces.feeder import TFSupervisionFeeder


class BSOxfordIIITPetFeeder(TFSupervisionFeeder):
    def __init__(self, data_feeder: BaseOxfordPetDataFeeder):
        self.data_feeder = data_feeder
        self.helper: BaseBSOxfordIIITPetFeederHelper = BSOxfordIIITPetFeederHelper(
            data_feeder
        )

    def get_inputs(self) -> List[tf.data.Dataset]:
        return [self.helper.get_image()]

    def get_outputs(self) -> List[tf.data.Dataset]:
        return [self.helper.get_mask()]


class BSOxfordIIITPetTrainingFeeder(BSOxfordIIITPetFeeder):
    def __init__(self):
        dataset_info_tuple: Tuple[tf.data.Dataset, tfds.core.DatasetInfo] = tfds.load(
            "oxford_iiit_pet:3.*.*", split="train[:80%]", with_info=True
        )
        data_feeder = BaseOxfordPetDataFeeder(
            original_dataset=dataset_info_tuple[0], original_info=dataset_info_tuple[1]
        )
        super().__init__(data_feeder)


class BSOxfordIIITPetValidationFeeder(BSOxfordIIITPetFeeder):
    def __init__(self):
        dataset_info_tuple: Tuple[tf.data.Dataset, tfds.core.DatasetInfo] = tfds.load(
            "oxford_iiit_pet:3.*.*", split="train[80%:]", with_info=True
        )
        data_feeder = BaseOxfordPetDataFeeder(
            original_dataset=dataset_info_tuple[0], original_info=dataset_info_tuple[1]
        )
        super().__init__(data_feeder)


class BSOxfordIIITPetTestFeeder(BSOxfordIIITPetFeeder):
    def __init__(self):
        dataset_info_tuple: Tuple[tf.data.Dataset, tfds.core.DatasetInfo] = tfds.load(
            "oxford_iiit_pet:3.*.*", split="test", with_info=True
        )
        data_feeder = BaseOxfordPetDataFeeder(
            original_dataset=dataset_info_tuple[0], original_info=dataset_info_tuple[1]
        )
        super().__init__(data_feeder)
