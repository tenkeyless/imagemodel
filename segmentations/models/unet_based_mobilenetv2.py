# Based on https://www.tensorflow.org/tutorials/images/segmentation

# Setup
# -----
# $ pip install tensorflow-metadata
# $ pip install -q git+https://github.com/tensorflow/examples.git

from typing import Dict, Optional

import tensorflow as tf
from segmentations.models.model_interface import ModelInterface
from tensorflow.keras.models import Model
from tensorflow_examples.models.pix2pix import pix2pix
from typing_extensions import TypedDict
from utils.function import get_default_args
from utils.optional import optional_map

base_model = tf.keras.applications.MobileNetV2(
    input_shape=[128, 128, 3], include_top=False
)

# 이 층들의 활성화를 이용합시다
layer_names = [
    "block_1_expand_relu",  # 64x64
    "block_3_expand_relu",  # 32x32
    "block_6_expand_relu",  # 16x16
    "block_13_expand_relu",  # 8x8
    "block_16_project",  # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# 특징추출 모델을 만듭시다
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),  # 32x32 -> 64x64
]


def unet_based_mobilenetv2(output_channels):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = inputs

    # 모델을 통해 다운샘플링합시다
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # 건너뛰기 연결을 업샘플링하고 설정하세요
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # 이 모델의 마지막 층입니다
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2, padding="same"
    )  # 64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


class UNetBasedMobilenetv2ArgumentsDict(TypedDict):
    output_channels: Optional[int]


class UNetBasedMobilenetv2Model(ModelInterface[UNetBasedMobilenetv2ArgumentsDict]):
    __default_args = get_default_args(unet_based_mobilenetv2)

    def func(self):
        return unet_based_mobilenetv2

    def get_model(self, option_dict: UNetBasedMobilenetv2ArgumentsDict) -> Model:
        return unet_based_mobilenetv2(
            output_channels=option_dict.get("output_channels")
            or self.__default_args["output_channels"]
        )

    def convert_str_model_option_dict(
        self, option_dict: Dict[str, str]
    ) -> UNetBasedMobilenetv2ArgumentsDict:
        # output channels
        output_channels_optional_str: Optional[str] = option_dict.get("output_channels")
        output_channel_optional: Optional[int] = optional_map(
            output_channels_optional_str, eval
        )

        return UNetBasedMobilenetv2ArgumentsDict(
            output_channels=output_channel_optional
        )
