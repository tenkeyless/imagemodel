from enum import Enum

from segmentations.models.model_interface import ModelInterface
from segmentations.models.unet import UNetModel


class Models(Enum):
    unet = "unet"
    unet_level = "unet_level"
    deeplab_v3 = "deeplab_v3"

    def get_model(self) -> ModelInterface:
        if self == Models.unet:
            return UNetModel()
        else:
            return UNetModel()

    @staticmethod
    def get_default() -> str:
        return Models.unet.value
