from enum import Enum
from typing import Optional

from imagemodel.common.datasets.oxford_pet_v3_tfds_dataset import OxfordPetV3TfdsDataset
from imagemodel.common.datasets.interfaces.dataset import TfdsDatasetInterface


class Datasets(Enum):
    oxford_iiit_pet_v3 = "oxford_iiit_pet_v3"
    none = "none"

    def get_dataset(self) -> Optional[TfdsDatasetInterface]:
        if self == Datasets.oxford_iiit_pet_v3:
            return OxfordPetV3TfdsDataset()
        else:
            return None

    @staticmethod
    def get_default() -> str:
        return Datasets.none.value
