from enum import Enum

from imagemodel.binary_segmentations.models.model_interface import ModelInterface
from imagemodel.binary_segmentations.models.unet import UNetModel
from imagemodel.binary_segmentations.models.unet_based_mobilenetv2 import (
    UNetBasedMobilenetv2Model,
)
from imagemodel.binary_segmentations.models.unet_level import UNetLevelModelManager


class Models(Enum):
    unet = "unet"
    unet_level = "unet_level"
    unet_based_mobilenetv2 = "unet_based_mobilenetv2"
    deeplab_v3 = "deeplab_v3"

    def get_model(self) -> ModelInterface:
        if self == Models.unet:
            return UNetModel()
        elif self == Models.unet_level:
            return UNetLevelModelManager()
        elif self == Models.unet_based_mobilenetv2:
            return UNetBasedMobilenetv2Model()
        else:
            return UNetModel()

    @staticmethod
    def get_default() -> str:
        return Models.unet.value
