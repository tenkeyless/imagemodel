import os
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from category_segmentations.models.model_interface import ModelInterface
from common.utils.function import get_default_args
from common.utils.functional import compose_left
from common.utils.optional import optional_map
from tensorflow.keras.layers import (
    Conv2D,
    Dropout,
    Input,
    Layer,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
)
from tensorflow.keras.models import Model
from typing_extensions import TypedDict


def unet_base_conv_2d(
    filter_num: int,
    kernel_size: int = 3,
    activation="relu",
    padding="same",
    kernel_initializer="he_normal",
    name_optional: Optional[str] = None,
) -> Layer:
    return Conv2D(
        filters=filter_num,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
        kernel_initializer=kernel_initializer,
        name=name_optional,
    )


def unet_base_sub_sampling(pool_size=(2, 2)) -> Layer:
    return MaxPooling2D(pool_size=pool_size)


def unet_base_up_sampling(
    filter_num: int,
    up_size: Tuple[int, int] = (2, 2),
    kernel_size: int = 3,
    activation="relu",
    padding="same",
    kernel_initializer="he_normal",
) -> Layer:
    up_sample_func = UpSampling2D(size=up_size)
    conv_func = Conv2D(
        filters=filter_num,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
        kernel_initializer=kernel_initializer,
    )
    return compose_left(up_sample_func, conv_func)


def unet(
    input_shape: Tuple[int, int, int] = (256, 256, 1),
    output_channels: int = 2,
    input_name: str = "unet_input",
    output_name: str = "unet_output",
    base_filters: int = 16,
) -> Model:
    # Input
    input: Layer = Input(shape=input_shape, name=input_name)

    # Encoder
    conv1: Layer = unet_base_conv_2d(base_filters * 4)(input)
    conv1 = unet_base_conv_2d(base_filters * 4)(conv1)
    pool1: Layer = unet_base_sub_sampling()(conv1)

    conv2: Layer = unet_base_conv_2d(base_filters * 8)(pool1)
    conv2 = unet_base_conv_2d(base_filters * 8)(conv2)
    pool2: Layer = unet_base_sub_sampling()(conv2)

    conv3: Layer = unet_base_conv_2d(base_filters * 16)(pool2)
    conv3 = unet_base_conv_2d(base_filters * 16)(conv3)
    pool3: Layer = unet_base_sub_sampling()(conv3)

    conv4: Layer = unet_base_conv_2d(base_filters * 32)(pool3)
    conv4 = unet_base_conv_2d(base_filters * 32)(conv4)
    pool4: Layer = unet_base_sub_sampling()(conv4)

    # Intermediate
    conv5: Layer = unet_base_conv_2d(base_filters * 64)(pool4)
    conv5 = unet_base_conv_2d(base_filters * 64)(conv5)
    drop1: Layer = Dropout(0.5)(conv5)

    # Decoder
    up1: Layer = unet_base_up_sampling(base_filters * 32)(drop1)
    merge1: Layer = concatenate([conv4, up1])
    conv6: Layer = unet_base_conv_2d(base_filters * 32)(merge1)
    conv6 = unet_base_conv_2d(base_filters * 32)(conv6)

    up2: Layer = unet_base_up_sampling(base_filters * 16)(conv6)
    merge2: Layer = concatenate([conv3, up2])
    conv7: Layer = unet_base_conv_2d(base_filters * 16)(merge2)
    conv7 = unet_base_conv_2d(base_filters * 16)(conv7)

    up3: Layer = unet_base_up_sampling(base_filters * 8)(conv7)
    merge3: Layer = concatenate([conv2, up3])
    conv8: Layer = unet_base_conv_2d(base_filters * 8)(merge3)
    conv8 = unet_base_conv_2d(base_filters * 8)(conv8)

    up4: Layer = unet_base_up_sampling(base_filters * 4)(conv8)
    merge4: Layer = concatenate([conv1, up4])
    conv9: Layer = unet_base_conv_2d(base_filters * 4)(merge4)
    conv9 = unet_base_conv_2d(base_filters * 4)(conv9)

    # Output
    output: Layer = unet_base_conv_2d(
        output_channels, kernel_size=1, activation="sigmoid", name_optional=output_name
    )(conv9)

    return Model(inputs=[input], outputs=[output])


class UNetArgumentsDict(TypedDict):
    input_shape: Optional[Tuple[int, int, int]]
    output_channels: Optional[int]
    input_name: Optional[str]
    output_name: Optional[str]
    base_filters: Optional[int]


# UNetArgumentsDict = TypedDict("UNetArgumentsDict", get_annotations(unet))


class UNetModel(ModelInterface[UNetArgumentsDict]):
    __default_args = get_default_args(unet)

    def func(self):
        return unet

    def get_model(self, option_dict: UNetArgumentsDict) -> Model:
        return unet(
            input_shape=option_dict.get("input_shape")
            or self.__default_args["input_shape"],
            output_channels=option_dict.get("output_channels")
            or self.__default_args["output_channels"],
            input_name=option_dict.get("input_name")
            or self.__default_args["input_name"],
            output_name=option_dict.get("output_name")
            or self.__default_args["output_name"],
            base_filters=option_dict.get("base_filters")
            or self.__default_args["base_filters"],
        )

    def convert_str_model_option_dict(
        self, option_dict: Dict[str, str]
    ) -> UNetArgumentsDict:
        # input shape
        input_shape_optional_str: Optional[str] = option_dict.get("input_shape")
        input_shape_optional: Optional[Tuple[int, int, int]] = optional_map(
            input_shape_optional_str, eval
        )
        if input_shape_optional is not None:
            if type(input_shape_optional) is not tuple:
                raise ValueError(
                    "'input_shape' should be tuple of 3 ints. `Tuple[int, int, int]`."
                )
            if len(input_shape_optional) != 3:
                raise ValueError(
                    "'input_shape' should be tuple of 3 ints. `Tuple[int, int, int]`."
                )

        # output channels
        output_channels_optional_str: Optional[str] = option_dict.get("output_channels")
        output_channels_optional: Optional[int] = optional_map(
            output_channels_optional_str, eval
        )

        # input name
        input_name_optional_str: Optional[str] = option_dict.get("input_name")

        # output name
        output_name_optional_str: Optional[str] = option_dict.get("output_name")

        # base filters
        base_filters_optional_str: Optional[str] = option_dict.get("base_filters")
        base_filters_optional: Optional[int] = optional_map(
            base_filters_optional_str, eval
        )

        return UNetArgumentsDict(
            input_shape=input_shape_optional,
            output_channels=output_channels_optional,
            input_name=input_name_optional_str,
            output_name=output_name_optional_str,
            base_filters=base_filters_optional,
        )

    def post_processing(self, predicted_result):
        def create_mask(pred_mask):
            pred_mask = tf.argmax(pred_mask, axis=-1)
            pred_mask = pred_mask[..., tf.newaxis]
            return pred_mask

        return create_mask(predicted_result)

    def save_post_processed_result(self, filename: str, result):
        foldername_only: str = os.path.dirname(filename)
        filename_only: str = os.path.basename(filename)
        filename_without_extension: str = filename_only[: filename_only.rfind(".")]
        new_filename: str = os.path.join(
            foldername_only, "{}.npy".format(filename_without_extension)
        )
        np.save(new_filename, result)
