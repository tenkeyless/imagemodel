import warnings
from typing import Dict, List, Optional, Tuple

from segmentations.models.model_interface import ModelInterface
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
from utils.function import get_default_args
from utils.functional import compose_left
from utils.optional import optional_map


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


def unet_level(
    level: int = 4,
    input_shape: Tuple[int, int, int] = (256, 256, 1),
    input_name: str = "unet_input",
    output_name: str = "unet_output",
    base_filters: int = 16,
) -> Model:
    # Parameter Check
    if level <= 0:
        raise ValueError("`level` should be greater than 0.")
    if level >= 6:
        warnings.warn("`level` recommended to be smaller than 6.", RuntimeWarning)

    # Prepare
    filter_nums: List[int] = []
    for l in range(level):
        filter_nums.append(base_filters * 4 * (2 ** l))

    # Input
    input: Layer = Input(shape=input_shape, name=input_name)

    # Skip connections
    skip_connections: List[Layer] = []

    # Encoder
    encoder: Layer = input
    for filter_num in filter_nums[:-1]:
        encoder = unet_base_conv_2d(filter_num)(encoder)
        encoder = unet_base_conv_2d(filter_num)(encoder)
        skip_connections.append(encoder)
        encoder = unet_base_sub_sampling()(encoder)

    # Intermediate
    intermedate: Layer = unet_base_conv_2d(filter_nums[-1])(encoder)
    intermedate = unet_base_conv_2d(filter_nums[-1])(intermedate)
    intermedate = Dropout(0.5)(intermedate)

    # Decoder
    decoder: Layer = intermedate
    for filter_num_index, filter_num in enumerate(filter_nums[:-1][::-1]):
        decoder = unet_base_up_sampling(filter_num)(decoder)
        skip_layer: Layer = skip_connections[::-1][filter_num_index]
        decoder = concatenate([skip_layer, decoder])
        decoder = unet_base_conv_2d(filter_num)(decoder)
        decoder = unet_base_conv_2d(filter_num)(decoder)

    # Output
    output: Layer = unet_base_conv_2d(2)(decoder)
    output = unet_base_conv_2d(
        1, kernel_size=1, activation="sigmoid", name_optional=output_name
    )(output)

    return Model(inputs=[input], outputs=[output])


class UNetLevelArgumentsDict(TypedDict):
    level: Optional[int]
    input_shape: Optional[Tuple[int, int, int]]
    input_name: Optional[str]
    output_name: Optional[str]
    base_filters: Optional[int]


class UNetLevelModel(ModelInterface[UNetLevelArgumentsDict]):
    __default_args = get_default_args(unet_level)

    def func(self):
        return unet_level

    def get_model(self, option_dict: UNetLevelArgumentsDict) -> Model:
        return unet_level(
            level=option_dict.get("level") or self.__default_args["level"],
            input_shape=option_dict.get("input_shape")
            or self.__default_args["input_shape"],
            input_name=option_dict.get("input_name")
            or self.__default_args["input_name"],
            output_name=option_dict.get("output_name")
            or self.__default_args["output_name"],
            base_filters=option_dict.get("base_filters")
            or self.__default_args["base_filters"],
        )

    def convert_str_model_option_dict(
        self, option_dict: Dict[str, str]
    ) -> UNetLevelArgumentsDict:
        # level
        level_optional_str: Optional[str] = option_dict.get("level")
        level_optional: Optional[int] = optional_map(level_optional_str, eval)

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

        # input name
        input_name_optional_str: Optional[str] = option_dict.get("input_name")

        # output name
        output_name_optional_str: Optional[str] = option_dict.get("output_name")

        # base filters
        base_filters_optional_str: Optional[str] = option_dict.get("base_filters")
        base_filters_optional: Optional[int] = optional_map(
            base_filters_optional_str, eval
        )

        return UNetLevelArgumentsDict(
            level=level_optional,
            input_shape=input_shape_optional,
            input_name=input_name_optional_str,
            output_name=output_name_optional_str,
            base_filters=base_filters_optional,
        )
