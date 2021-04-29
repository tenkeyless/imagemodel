from enum import Enum

from imagemodel.category_segmentations.models.deeplab_v3 import DeeplabV3Model
from imagemodel.category_segmentations.models.model_interface import ModelInterface
from imagemodel.category_segmentations.models.unet import UNetModel
from imagemodel.category_segmentations.models.unet_based_mobilenetv2 import (
    UNetBasedMobilenetv2Model,
)
from imagemodel.category_segmentations.models.unet_level import UNetLevelModel


class Models(Enum):
    unet = "unet"
    unet_level = "unet_level"
    unet_based_mobilenetv2 = "unet_based_mobilenetv2"
    deeplab_v3 = "deeplab_v3"

    def get_model(self) -> ModelInterface:
        if self == Models.unet:
            return UNetModel()
        elif self == Models.unet_level:
            return UNetLevelModel()
        elif self == Models.unet_based_mobilenetv2:
            return UNetBasedMobilenetv2Model()
        elif self == Models.deeplab_v3:
            return DeeplabV3Model()
        else:
            return UNetModel()

    @staticmethod
    def get_default() -> str:
        return Models.unet.value
