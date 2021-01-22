from typing import Optional, Tuple

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
from utils.functional import compose_left


def unet_base_conv_2d(
    filter_num: int,
    kernel_size: int = 3,
    activation="relu",
    padding="same",
    kernel_initializer="he_normal",
    name_optional: Optional[str] = None,
):
    return Conv2D(
        filters=filter_num,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
        kernel_initializer=kernel_initializer,
        name=name_optional,
    )


def unet_base_sub_sampling(pool_size=(2, 2)):
    return MaxPooling2D(pool_size=pool_size)


def unet_base_up_sampling(
    filter_num: int,
    up_size: Tuple[int, int] = (2, 2),
    kernel_size: int = 3,
    activation="relu",
    padding="same",
    kernel_initializer="he_normal",
):
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
    input_name: str = "unet_input",
    output_name: str = "unet_output",
    base_filters: int = 16,
):
    # Input
    input = Input(shape=input_shape, name=input_name)

    # Encoder
    conv1: Layer = unet_base_conv_2d(base_filters * 4)(input)
    conv1: Layer = unet_base_conv_2d(base_filters * 4)(conv1)
    pool1: Layer = unet_base_sub_sampling()(conv1)

    conv2: Layer = unet_base_conv_2d(base_filters * 8)(pool1)
    conv2: Layer = unet_base_conv_2d(base_filters * 8)(conv2)
    pool2: Layer = unet_base_sub_sampling()(conv2)

    conv3: Layer = unet_base_conv_2d(base_filters * 16)(pool2)
    conv3: Layer = unet_base_conv_2d(base_filters * 16)(conv3)
    pool3: Layer = unet_base_sub_sampling()(conv3)

    conv4: Layer = unet_base_conv_2d(base_filters * 32)(pool3)
    conv4: Layer = unet_base_conv_2d(base_filters * 32)(conv4)
    pool4: Layer = unet_base_sub_sampling()(conv4)

    # Intermediate
    conv5: Layer = unet_base_conv_2d(base_filters * 64)(pool4)
    conv5: Layer = unet_base_conv_2d(base_filters * 64)(conv5)
    drop1 = Dropout(0.5)(conv5)

    # Decoder
    up1: Layer = unet_base_up_sampling(base_filters * 32)(drop1)
    merge1 = concatenate([conv4, up1])
    conv6: Layer = unet_base_conv_2d(base_filters * 32)(merge1)
    conv6: Layer = unet_base_conv_2d(base_filters * 32)(conv6)

    up2: Layer = unet_base_up_sampling(base_filters * 16)(conv6)
    merge2 = concatenate([conv3, up2])
    conv7: Layer = unet_base_conv_2d(base_filters * 16)(merge2)
    conv7: Layer = unet_base_conv_2d(base_filters * 16)(conv7)

    up3: Layer = unet_base_up_sampling(base_filters * 8)(conv7)
    merge3 = concatenate([conv2, up3])
    conv8: Layer = unet_base_conv_2d(base_filters * 8)(merge3)
    conv8: Layer = unet_base_conv_2d(base_filters * 8)(conv8)

    up4: Layer = unet_base_up_sampling(base_filters * 4)(conv8)
    merge4 = concatenate([conv1, up4])
    conv9: Layer = unet_base_conv_2d(base_filters * 4)(merge4)
    conv9: Layer = unet_base_conv_2d(base_filters * 4)(conv9)

    # Output
    conv10: Layer = unet_base_conv_2d(2)(conv9)
    conv11: Layer = unet_base_conv_2d(
        1, kernel_size=1, activation="sigmoid", name_optional=output_name
    )(conv10)

    return Model(inputs=[input], outputs=[conv11])
