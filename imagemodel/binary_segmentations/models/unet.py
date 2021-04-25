from typing import Dict, Optional, Tuple, List

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

from imagemodel.binary_segmentations.models.common_arguments import ModelArguments
from imagemodel.binary_segmentations.models.common_model_manager import (
    CommonModelManagerDictGeneratable,
    CommonModelManager
)
from imagemodel.common.utils.function import get_default_args
from imagemodel.common.utils.functional import compose_left
from imagemodel.common.utils.gpu_check import check_first_gpu
from imagemodel.common.utils.optional import optional_map

check_first_gpu()


class UNetArgumentsDict(TypedDict):
    input_shape: Optional[Tuple[int, int, int]]
    input_name: Optional[str]
    output_name: Optional[str]
    base_filters: Optional[int]


class UNetArguments(ModelArguments[UNetArgumentsDict]):
    def __init__(self, dic: UNetArgumentsDict):
        self.dic = dic

    @classmethod
    def init_from_str_dict(cls, string_dict: Dict[str, str]):
        return cls(cls.convert_str_dict(string_dict))

    # noinspection DuplicatedCode
    @classmethod
    def convert_str_dict(cls, string_dict: Dict[str, str]) -> UNetArgumentsDict:
        __keys: List[str] = list(UNetArgumentsDict.__annotations__)

        # input shape
        input_shape_optional_str: Optional[str] = string_dict.get(__keys[1])
        input_shape_optional: Optional[Tuple[int, int, int]] = optional_map(input_shape_optional_str, eval)
        input_shape_tuples_optional: Optional[Tuple[int, ...]] = tuple(map(int, input_shape_optional))
        if input_shape_tuples_optional is not None:
            if type(input_shape_tuples_optional) is not tuple:
                raise ValueError("'input_shape' should be tuple of 3 ints. `Tuple[int, int, int]`.")
            if len(input_shape_tuples_optional) != 3:
                raise ValueError("'input_shape' should be tuple of 3 ints. `Tuple[int, int, int]`.")

        # input name
        input_name_optional_str: Optional[str] = string_dict.get(__keys[2])

        # output name
        output_name_optional_str: Optional[str] = string_dict.get(__keys[3])

        # base filters
        base_filters_optional_str: Optional[str] = string_dict.get(__keys[4])
        base_filters_optional: Optional[int] = optional_map(base_filters_optional_str, eval)

        return UNetArgumentsDict(
            input_shape=input_shape_tuples_optional,
            input_name=input_name_optional_str,
            output_name=output_name_optional_str,
            base_filters=base_filters_optional)

    @property
    def input_shape(self) -> Optional[Tuple[int, int, int]]:
        return self.dic.get('input_shape')

    @property
    def input_name(self) -> Optional[str]:
        return self.dic.get('input_name')

    @property
    def output_name(self) -> Optional[str]:
        return self.dic.get('output_name')

    @property
    def base_filters(self) -> Optional[int]:
        return self.dic.get('base_filters')


class UNetModelManager(CommonModelManager, CommonModelManagerDictGeneratable[UNetArgumentsDict]):
    def __init__(
            self,
            input_shape: Optional[Tuple[int, int, int]] = None,
            input_name: Optional[str] = None,
            output_name: Optional[str] = None,
            base_filters: Optional[int] = None):
        __model_default_args = get_default_args(self.unet)
        __model_default_values = UNetArguments(__model_default_args)

        self.input_shape: Tuple[int, int, int] = input_shape or __model_default_values.input_shape
        self.input_name: str = input_name or __model_default_values.input_name
        self.input_name = self.layer_name_correction(self.input_name)
        self.output_name: str = output_name or __model_default_values.output_name
        self.output_name = self.layer_name_correction(self.output_name)
        self.base_filters: int = base_filters or __model_default_values.base_filters

    @classmethod
    def init_with_dict(cls, option_dict: Optional[UNetArgumentsDict] = None):
        if option_dict is not None:
            unet_arguments = UNetArguments(option_dict)
            return cls(
                input_shape=unet_arguments.input_shape,
                input_name=unet_arguments.input_name,
                output_name=unet_arguments.output_name,
                base_filters=unet_arguments.base_filters)
        else:
            return cls()

    @classmethod
    def init_with_str_dict(cls, option_str_dict: Optional[Dict[str, str]] = None):
        if option_str_dict is not None:
            unet_arguments_dict = UNetArguments.convert_str_dict(option_str_dict)
            return cls.init_with_dict(unet_arguments_dict)
        else:
            return cls()

    def setup_model(self) -> Model:
        return self.unet(
            input_shape=self.input_shape,
            input_name=self.input_name,
            output_name=self.output_name,
            base_filters=self.base_filters)

    # noinspection DuplicatedCode
    @staticmethod
    def unet(
            input_shape: Tuple[int, int, int] = (256, 256, 1),
            input_name: str = "unet_input",
            output_name: str = "unet_output",
            base_filters: int = 16,
    ) -> Model:
        def __unet_base_conv_2d(
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

        def __unet_base_sub_sampling(pool_size=(2, 2)) -> Layer:
            return MaxPooling2D(pool_size=pool_size)

        def __unet_base_up_sampling(
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

        # Input
        input_layer: Layer = Input(shape=input_shape, name=input_name)

        # Encoder
        conv1: Layer = __unet_base_conv_2d(base_filters * 4)(input_layer)
        conv1 = __unet_base_conv_2d(base_filters * 4)(conv1)
        pool1: Layer = __unet_base_sub_sampling()(conv1)

        conv2: Layer = __unet_base_conv_2d(base_filters * 8)(pool1)
        conv2 = __unet_base_conv_2d(base_filters * 8)(conv2)
        pool2: Layer = __unet_base_sub_sampling()(conv2)

        conv3: Layer = __unet_base_conv_2d(base_filters * 16)(pool2)
        conv3 = __unet_base_conv_2d(base_filters * 16)(conv3)
        pool3: Layer = __unet_base_sub_sampling()(conv3)

        conv4: Layer = __unet_base_conv_2d(base_filters * 32)(pool3)
        conv4 = __unet_base_conv_2d(base_filters * 32)(conv4)
        pool4: Layer = __unet_base_sub_sampling()(conv4)

        # Intermediate
        conv5: Layer = __unet_base_conv_2d(base_filters * 64)(pool4)
        conv5 = __unet_base_conv_2d(base_filters * 64)(conv5)
        drop1: Layer = Dropout(0.5)(conv5)

        # Decoder
        up1: Layer = __unet_base_up_sampling(base_filters * 32)(drop1)
        merge1: Layer = concatenate([conv4, up1])
        conv6: Layer = __unet_base_conv_2d(base_filters * 32)(merge1)
        conv6 = __unet_base_conv_2d(base_filters * 32)(conv6)

        up2: Layer = __unet_base_up_sampling(base_filters * 16)(conv6)
        merge2: Layer = concatenate([conv3, up2])
        conv7: Layer = __unet_base_conv_2d(base_filters * 16)(merge2)
        conv7 = __unet_base_conv_2d(base_filters * 16)(conv7)

        up3: Layer = __unet_base_up_sampling(base_filters * 8)(conv7)
        merge3: Layer = concatenate([conv2, up3])
        conv8: Layer = __unet_base_conv_2d(base_filters * 8)(merge3)
        conv8 = __unet_base_conv_2d(base_filters * 8)(conv8)

        up4: Layer = __unet_base_up_sampling(base_filters * 4)(conv8)
        merge4: Layer = concatenate([conv1, up4])
        conv9: Layer = __unet_base_conv_2d(base_filters * 4)(merge4)
        conv9 = __unet_base_conv_2d(base_filters * 4)(conv9)

        # Output
        conv10: Layer = __unet_base_conv_2d(2)(conv9)
        output: Layer = __unet_base_conv_2d(
            1, kernel_size=1, activation="sigmoid", name_optional=output_name
        )(conv10)

        return Model(inputs=[input_layer], outputs=[output])


class UNetArgumentsDict(TypedDict):
    input_shape: Optional[Tuple[int, int, int]]
    input_name: Optional[str]
    output_name: Optional[str]
    base_filters: Optional[int]
