import warnings
from typing import List, Optional, Tuple, Dict

from tensorflow.keras.layers import Conv2D, Dropout, Input, Layer, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from typing_extensions import TypedDict

from imagemodel.binary_segmentations.models.common_arguments import ModelArguments
from imagemodel.binary_segmentations.models.common_model_manager import (
    CommonModelManager,
    CommonModelManagerDictGeneratable
)
from imagemodel.common.utils.function import get_default_args
from imagemodel.common.utils.functional import compose_left
from imagemodel.common.utils.gpu_check import check_first_gpu
from imagemodel.common.utils.optional import optional_map

check_first_gpu()


class UNetLevelArgumentsDict(TypedDict):
    level: Optional[int]
    input_shape: Optional[Tuple[int, int, int]]
    input_name: Optional[str]
    output_name: Optional[str]
    base_filters: Optional[int]


class UNetLevelArguments(ModelArguments[UNetLevelArgumentsDict]):
    def __init__(self, dic: UNetLevelArgumentsDict):
        self.dic = dic

    @classmethod
    def init_from_str_dict(cls, string_dict: Dict[str, str]):
        return cls(cls.convert_str_dict(string_dict))

    # noinspection DuplicatedCode
    @classmethod
    def convert_str_dict(cls, string_dict: Dict[str, str]) -> UNetLevelArgumentsDict:
        __keys: List[str] = list(UNetLevelArgumentsDict.__annotations__)

        # level
        level_optional_str: Optional[str] = string_dict.get(__keys[0])
        level_optional: Optional[int] = optional_map(level_optional_str, eval)

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

        return UNetLevelArgumentsDict(
            level=level_optional,
            input_shape=input_shape_tuples_optional,
            input_name=input_name_optional_str,
            output_name=output_name_optional_str,
            base_filters=base_filters_optional)

    @property
    def level(self) -> Optional[int]:
        return self.dic.get('level')

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


class UNetLevelModelManager(CommonModelManager, CommonModelManagerDictGeneratable[UNetLevelArgumentsDict]):
    def __init__(
            self,
            level: Optional[int] = None,
            input_shape: Optional[Tuple[int, int, int]] = None,
            input_name: Optional[str] = None,
            output_name: Optional[str] = None,
            base_filters: Optional[int] = None):
        __model_default_args = get_default_args(self.unet_level)
        __model_default_values = UNetLevelArguments(__model_default_args)

        self.level: int = level or __model_default_values.level
        self.input_shape: Tuple[int, int, int] = input_shape or __model_default_values.input_shape
        self.input_name: str = input_name or __model_default_values.input_name
        self.input_name = self.layer_name_correction(self.input_name)
        self.output_name: str = output_name or __model_default_values.output_name
        self.output_name = self.layer_name_correction(self.output_name)
        self.base_filters: int = base_filters or __model_default_values.base_filters

    @classmethod
    def init_with_dict(cls, option_dict: Optional[UNetLevelArgumentsDict] = None):
        if option_dict is not None:
            unet_level_arguments = UNetLevelArguments(option_dict)
            return cls(
                level=unet_level_arguments.level,
                input_shape=unet_level_arguments.input_shape,
                input_name=unet_level_arguments.input_name,
                output_name=unet_level_arguments.output_name,
                base_filters=unet_level_arguments.base_filters)
        else:
            return cls()

    @classmethod
    def init_with_str_dict(cls, option_str_dict: Optional[Dict[str, str]] = None):
        if option_str_dict is not None:
            unet_level_arguments_dict = UNetLevelArguments.convert_str_dict(option_str_dict)
            return cls.init_with_dict(unet_level_arguments_dict)
        else:
            return cls()

    def setup_model(self) -> Model:
        return self.unet_level(
            level=self.level,
            input_shape=self.input_shape,
            input_name=self.input_name,
            output_name=self.output_name,
            base_filters=self.base_filters)

    # noinspection DuplicatedCode
    @staticmethod
    def unet_level(
            level: int = 4,
            input_shape: Tuple[int, int, int] = (256, 256, 1),
            input_name: str = "unet_input",
            output_name: str = "unet_output",
            base_filters: int = 16,
    ) -> Model:
        def __unet_level_base_conv_2d(
                _filter_num: int,
                kernel_size: int = 3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
                name_optional: Optional[str] = None,
        ) -> Layer:
            return Conv2D(
                filters=_filter_num,
                kernel_size=kernel_size,
                activation=activation,
                padding=padding,
                kernel_initializer=kernel_initializer,
                name=name_optional,
            )

        def __unet_base_sub_sampling(pool_size=(2, 2)) -> Layer:
            return MaxPooling2D(pool_size=pool_size)

        def __unet_base_up_sampling(
                _filter_num: int,
                up_size: Tuple[int, int] = (2, 2),
                kernel_size: int = 3,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
        ) -> Layer:
            up_sample_func = UpSampling2D(size=up_size)
            conv_func = Conv2D(
                filters=_filter_num,
                kernel_size=kernel_size,
                activation=activation,
                padding=padding,
                kernel_initializer=kernel_initializer,
            )
            return compose_left(up_sample_func, conv_func)

        # Parameter Check
        if level <= 0:
            raise ValueError("`level` should be greater than 0.")
        if level >= 6:
            warnings.warn("`level` recommended to be smaller than 6.", RuntimeWarning)

        # Prepare
        filter_nums: List[int] = []
        for level_iter in range(level):
            filter_nums.append(base_filters * 4 * (2 ** level_iter))

        # Input
        input_layer: Layer = Input(shape=input_shape, name=input_name)

        # Skip connections
        skip_connections: List[Layer] = []

        # Encoder
        encoder: Layer = input_layer
        for filter_num in filter_nums[:-1]:
            encoder = __unet_level_base_conv_2d(filter_num)(encoder)
            encoder = __unet_level_base_conv_2d(filter_num)(encoder)
            skip_connections.append(encoder)
            encoder = __unet_base_sub_sampling()(encoder)

        # Intermediate
        intermedate: Layer = __unet_level_base_conv_2d(filter_nums[-1])(encoder)
        intermedate = __unet_level_base_conv_2d(filter_nums[-1])(intermedate)
        intermedate = Dropout(0.5)(intermedate)

        # Decoder
        decoder: Layer = intermedate
        for filter_num_index, filter_num in enumerate(filter_nums[:-1][::-1]):
            decoder = __unet_base_up_sampling(filter_num)(decoder)
            skip_layer: Layer = skip_connections[::-1][filter_num_index]
            decoder = concatenate([skip_layer, decoder])
            decoder = __unet_level_base_conv_2d(filter_num)(decoder)
            decoder = __unet_level_base_conv_2d(filter_num)(decoder)

        # Output
        output: Layer = __unet_level_base_conv_2d(2)(decoder)
        output = __unet_level_base_conv_2d(1, kernel_size=1, activation="sigmoid", name_optional=output_name)(output)

        return Model(inputs=[input_layer], outputs=[output])
