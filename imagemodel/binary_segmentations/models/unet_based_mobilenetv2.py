# Based on https://www.tensorflow.org/tutorials/images/segmentation

# Setup
# -----
# $ pip install tensorflow-metadata
# $ pip install -q git+https://github.com/tensorflow/examples.git
# # for `from tensorflow_examples.models.pix2pix import pix2pix`

from typing import Optional, Tuple, Dict, List

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow_examples.models.pix2pix import pix2pix
from typing_extensions import TypedDict

from imagemodel.binary_segmentations.models.common_arguments import ModelArguments
from imagemodel.binary_segmentations.models.common_model_manager import (
    CommonModelManager,
    CommonModelManagerDictGeneratable
)
from imagemodel.common.utils.function import get_default_args
from imagemodel.common.utils.gpu_check import check_first_gpu
from imagemodel.common.utils.optional import optional_map

check_first_gpu()


class UNetBasedMobileNetV2ArgumentsDict(TypedDict):
    input_shape: Optional[Tuple[int, int, int]]


class UNetBasedMobileNetV2Arguments(ModelArguments[UNetBasedMobileNetV2ArgumentsDict]):
    def __init__(self, dic: UNetBasedMobileNetV2ArgumentsDict):
        self.dic = dic

    @classmethod
    def init_from_str_dict(cls, string_dict: Dict[str, str]):
        return cls(cls.convert_str_dict(string_dict))

    # noinspection DuplicatedCode
    @classmethod
    def convert_str_dict(cls, string_dict: Dict[str, str]) -> UNetBasedMobileNetV2ArgumentsDict:
        __keys: List[str] = list(UNetBasedMobileNetV2ArgumentsDict.__annotations__)

        # input shape
        input_shape_optional_str: Optional[str] = string_dict.get(__keys[1])
        input_shape_optional: Optional[Tuple[int, int, int]] = optional_map(input_shape_optional_str, eval)
        input_shape_tuples_optional: Optional[Tuple[int, ...]] = tuple(map(int, input_shape_optional))
        if input_shape_tuples_optional is not None:
            if type(input_shape_tuples_optional) is not tuple:
                raise ValueError("'input_shape' should be tuple of 3 ints. `Tuple[int, int, int]`.")
            if len(input_shape_tuples_optional) != 3:
                raise ValueError("'input_shape' should be tuple of 3 ints. `Tuple[int, int, int]`.")

        return UNetBasedMobileNetV2ArgumentsDict(input_shape=input_shape_tuples_optional)

    @property
    def input_shape(self) -> Optional[Tuple[int, int, int]]:
        return self.dic.get('input_shape')


class UNetBasedMobileNetV2ModelManager(
        CommonModelManager,
        CommonModelManagerDictGeneratable[UNetBasedMobileNetV2Arguments]):

    def __init__(self, input_shape: Optional[Tuple[int, int, int]] = None):
        __model_default_args = get_default_args(self.unet_based_mobilenetv2)
        __model_default_values = UNetBasedMobileNetV2Arguments(__model_default_args)

        self.input_shape: Tuple[int, int, int] = input_shape or __model_default_values.input_shape

    @classmethod
    def init_with_dict(cls, option_dict: Optional[UNetBasedMobileNetV2ArgumentsDict] = None):
        if option_dict is not None:
            unet_based_mobilenetv2_arguments = UNetBasedMobileNetV2Arguments(option_dict)
            return cls(input_shape=unet_based_mobilenetv2_arguments.input_shape)
        else:
            return cls()

    @classmethod
    def init_with_str_dict(cls, option_str_dict: Optional[Dict[str, str]] = None):
        if option_str_dict is not None:
            unet_based_mobilenetv2_arguments_dict = UNetBasedMobileNetV2Arguments.convert_str_dict(option_str_dict)
            return cls.init_with_dict(unet_based_mobilenetv2_arguments_dict)
        else:
            return cls()

    def setup_model(self) -> Model:
        return self.unet_based_mobilenetv2(input_shape=self.input_shape)

    # noinspection DuplicatedCode
    @staticmethod
    def unet_based_mobilenetv2(input_shape: Tuple[int, int, int] = (128, 128, 3)):
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False)

        # Use the activations of these layers
        layer_names = [
            "block_1_expand_relu",  # 64x64
            "block_3_expand_relu",  # 32x32
            "block_6_expand_relu",  # 16x16
            "block_13_expand_relu",  # 8x8
            "block_16_project",  # 4x4
        ]
        layers = [base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

        down_stack.trainable = False

        up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),  # 32x32 -> 64x64
        ]

        inputs = tf.keras.layers.Input(shape=input_shape)
        x = inputs

        # Downsampling through the model
        skips = down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding="same")  # 64x64 -> 128x128

        x = last(x)

        return Model(inputs=[inputs], outputs=[x])
