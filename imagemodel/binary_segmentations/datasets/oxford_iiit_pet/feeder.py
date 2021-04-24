from abc import abstractmethod

from imagemodel.binary_segmentations.datasets.bs_feeder import BSFeeder
from imagemodel.binary_segmentations.datasets.bs_feeder_helper import (
    BSFeederInputHelper,
    BSFeederOutputHelper,
)
from imagemodel.binary_segmentations.datasets.oxford_iiit_pet.feeder_helper import (
    BSOxfordIIITPetFeederInputHelper,
    BSOxfordIIITPetFeederOutputHelper
)
from imagemodel.common.datasets.descriptor.oxford_iiit_pet_data_descriptor import \
    OxfordIIITPetDataDescriptor


class BSOxfordIIITPetFeeder(BSFeeder):
    @abstractmethod
    def __init__(
            self,
            oxford_iiit_pet_3_data_descriptor: OxfordIIITPetDataDescriptor):
        self._oxford_iiit_pet_3_data_descriptor = oxford_iiit_pet_3_data_descriptor

    @property
    def input_helper(self) -> BSFeederInputHelper:
        return BSOxfordIIITPetFeederInputHelper(
            data_descriptor=self._oxford_iiit_pet_3_data_descriptor
        )

    @property
    def output_helper(self) -> BSFeederOutputHelper:
        return BSOxfordIIITPetFeederOutputHelper(
            data_descriptor=self._oxford_iiit_pet_3_data_descriptor
        )


class BSOxfordIIITPetTrainingFeeder(BSOxfordIIITPetFeeder):
    @property
    def feeder_data_description(self):
        return "oxford_iiit_pet:3.*.* training dataset. train[:80%]"

    def __init__(self):
        super(BSOxfordIIITPetTrainingFeeder, self).__init__(
            oxford_iiit_pet_3_data_descriptor=OxfordIIITPetDataDescriptor.init_with_train_dataset(
                begin_optional=None, end_optional=80
            )
        )


class BSOxfordIIITPetValidationFeeder(BSOxfordIIITPetFeeder):
    @property
    def feeder_data_description(self):
        return "oxford_iiit_pet:3.*.* validation dataset. train[80%:]"

    def __init__(self):
        super(BSOxfordIIITPetValidationFeeder, self).__init__(
            oxford_iiit_pet_3_data_descriptor=OxfordIIITPetDataDescriptor.init_with_train_dataset(
                begin_optional=80, end_optional=None
            )
        )


class BSOxfordIIITPetTestFeeder(BSOxfordIIITPetFeeder):
    @property
    def feeder_data_description(self):
        return "oxford_iiit_pet:3.*.* test dataset. test"

    def __init__(self):
        super(BSOxfordIIITPetTestFeeder, self).__init__(
            oxford_iiit_pet_3_data_descriptor=OxfordIIITPetDataDescriptor.init_with_test_dataset()
        )
