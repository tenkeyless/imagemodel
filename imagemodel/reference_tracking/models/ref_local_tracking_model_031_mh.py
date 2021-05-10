import warnings
from typing import Dict, List, Optional, Tuple

from tensorflow.keras.layers import Conv2D, Dropout, Input, Layer, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from typing_extensions import TypedDict

from imagemodel.common.models.common_arguments import ModelArguments
from imagemodel.common.models.common_model_manager import (
    CommonModelManager,
    CommonModelManagerDictGeneratable
)
from imagemodel.common.utils.function import get_default_args
from imagemodel.common.utils.functional import compose_left
from imagemodel.common.utils.gpu_check import check_first_gpu
from imagemodel.reference_tracking.layers.aggregation_layer import AggregationLayer
from imagemodel.reference_tracking.layers.aggregation_layer2 import AggregationLayer2
from imagemodel.reference_tracking.layers.ref_local_layer7_mh import RefLocal7MH
from imagemodel.reference_tracking.layers.ref_local_layer8_mh import RefLocal8MH
from imagemodel.reference_tracking.layers.shrink_layer import ShrinkLayer

check_first_gpu()


class RefLocalTrackingModel031MHArgumentsDict(TypedDict):
    unet_l4_model_main: Optional[Model]
    unet_l4_model_ref: Optional[Model]
    bin_num: Optional[int]
    input_main_image_shape: Optional[Tuple[int, int, int]]
    input_ref_image_shape: Optional[Tuple[int, int, int]]
    input_ref_bin_label_shape: Optional[Tuple[int, int, int]]
    head_num: Optional[int]


class RefLocalTrackingModel031MHArguments(ModelArguments[RefLocalTrackingModel031MHArgumentsDict]):
    def __init__(self, dic: RefLocalTrackingModel031MHArgumentsDict):
        self.dic = dic
    
    @classmethod
    def init_from_str_dict(cls, string_dict: Dict[str, str]):
        return cls(cls.convert_str_dict(string_dict))
    
    # noinspection DuplicatedCode
    @classmethod
    def convert_str_dict(cls, string_dict: Dict[str, str]) -> RefLocalTrackingModel031MHArgumentsDict:
        raise NotImplementedError
        # __keys: List[str] = list(RefLocalTrackingModel031ArgumentsDict.__annotations__)
        #
        # # # main u-net
        # # level_optional_str: Optional[str] = string_dict.get(__keys[0])
        # # level_optional: Optional[int] = optional_map(level_optional_str, eval)
        # #
        # # # level
        # # level_optional_str: Optional[str] = string_dict.get(__keys[0])
        # # level_optional: Optional[int] = optional_map(level_optional_str, eval)
        # #
        # # # base filters
        # # base_filters_optional_str: Optional[str] = string_dict.get(__keys[1])
        # # base_filters_optional: Optional[int] = optional_map(base_filters_optional_str, eval)
        #
        # # bin num
        # bin_num_optional_str: Optional[str] = string_dict.get(__keys[2])
        # bin_num_optional: Optional[int] = optional_map(bin_num_optional_str, eval)
        #
        # def __check_image_shape(image_shape_optional_str: Optional[str], name: str) -> Optional[Tuple[int, ...]]:
        #     image_shape_optional: Optional[Tuple[int, int, int]] = optional_map(image_shape_optional_str, eval)
        #     image_shape_tuples_optional: Optional[Tuple[int, ...]] = tuple(map(int, image_shape_optional))
        #     if image_shape_tuples_optional is not None:
        #         if type(image_shape_tuples_optional) is not tuple:
        #             raise ValueError("'{}' should be tuple of 3 ints. `Tuple[int, int, int]`.".format(name))
        #         if len(image_shape_tuples_optional) != 3:
        #             raise ValueError("'{}' should be tuple of 3 ints. `Tuple[int, int, int]`.".format(name))
        #     return image_shape_tuples_optional
        #
        # # input main image shape
        # input_main_image_shape_optional_str: Optional[str] = string_dict.get(__keys[3])
        # input_main_image_shape_tuples_optional = __check_image_shape(
        #         input_main_image_shape_optional_str,
        #         'input_main_image_shape')
        #
        # # input ref image shape
        # input_ref_image_shape_optional_str: Optional[str] = string_dict.get(__keys[4])
        # input_ref_image_shape_tuples_optional = __check_image_shape(
        #         input_ref_image_shape_optional_str,
        #         'input_ref_image_shape')
        #
        # # input ref bin label shape
        # input_ref_bin_label_shape_optional_str: Optional[str] = string_dict.get(__keys[5])
        # input_ref_bin_label_shape_tuples_optional = __check_image_shape(
        #         input_ref_bin_label_shape_optional_str,
        #         'input_ref_bin_label_shape')
        #
        # return RefLocalTrackingModel031ArgumentsDict(
        #         unet_l4_model_main=level_optional,
        #         unet_l4_model_ref=base_filters_optional,
        #         bin_num=bin_num_optional,
        #         input_main_image_shape=input_main_image_shape_tuples_optional,
        #         input_ref_image_shape=input_ref_image_shape_tuples_optional,
        #         input_ref_bin_label_shape=input_ref_bin_label_shape_tuples_optional)
    
    @property
    def unet_l4_model_main(self) -> Optional[Model]:
        return self.dic.get('unet_l4_model_main')
    
    @property
    def unet_l4_model_ref(self) -> Optional[Model]:
        return self.dic.get('unet_l4_model_ref')
    
    @property
    def bin_num(self) -> Optional[int]:
        return self.dic.get('bin_num')
    
    @property
    def input_main_image_shape(self) -> Optional[Tuple[int, int, int]]:
        return self.dic.get('input_main_image_shape')
    
    @property
    def input_ref_image_shape(self) -> Optional[Tuple[int, int, int]]:
        return self.dic.get('input_ref_image_shape')
    
    @property
    def input_ref_bin_label_shape(self) -> Optional[Tuple[int, int, int]]:
        return self.dic.get('input_ref_bin_label_shape')
    
    @property
    def head_num(self) -> Optional[int]:
        return self.dic.get('head_num')


# noinspection DuplicatedCode
def unet_level(
        level: int = 4,
        input_shape: Tuple[int, int, int] = (256, 256, 1),
        input_name: str = "unet_input",
        output_name: str = "unet_output",
        base_filters: int = 16) -> Model:
    def __unet_level_base_conv_2d(
            _filter_num: int,
            kernel_size: int = 3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
            name_optional: Optional[str] = None) -> Layer:
        return Conv2D(
                filters=_filter_num,
                kernel_size=kernel_size,
                activation=activation,
                padding=padding,
                kernel_initializer=kernel_initializer,
                name=name_optional)
    
    def __unet_base_sub_sampling(pool_size=(2, 2)) -> Layer:
        return MaxPooling2D(pool_size=pool_size)
    
    def __unet_base_up_sampling(
            _filter_num: int,
            up_size: Tuple[int, int] = (2, 2),
            kernel_size: int = 3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal") -> Layer:
        up_sample_func = UpSampling2D(size=up_size)
        conv_func = Conv2D(
                filters=_filter_num,
                kernel_size=kernel_size,
                activation=activation,
                padding=padding,
                kernel_initializer=kernel_initializer)
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


class RefLocalTrackingModel031MHManager(
        CommonModelManager,
        CommonModelManagerDictGeneratable[RefLocalTrackingModel031MHArgumentsDict]):
    def __init__(
            self,
            unet_l4_model_main: Optional[Model] = None,
            unet_l4_model_ref: Optional[Model] = None,
            bin_num: Optional[int] = None,
            input_main_image_shape: Optional[Tuple[int, int, int]] = None,
            input_ref_image_shape: Optional[Tuple[int, int, int]] = None,
            input_ref_bin_label_shape: Optional[Tuple[int, int, int]] = None,
            head_num: Optional[int] = None):
        __model_default_args = get_default_args(self.ref_local_tracking_model_031_mh)
        __model_default_values = RefLocalTrackingModel031MHArguments(__model_default_args)
        
        self.unet_l4_model_main: Model = unet_l4_model_main or __model_default_values.unet_l4_model_main
        self.unet_l4_model_ref: Model = unet_l4_model_ref or __model_default_values.unet_l4_model_main
        self.bin_num: int = bin_num or __model_default_values.bin_num
        self.input_main_image_shape: Tuple[int, int, int] = \
            input_main_image_shape or __model_default_values.input_main_image_shape
        self.input_ref_image_shape: Tuple[int, int, int] = \
            input_ref_image_shape or __model_default_values.input_ref_image_shape
        self.input_ref_bin_label_shape: Tuple[int, int, int] = \
            input_ref_bin_label_shape or __model_default_values.input_ref_bin_label_shape
        self.head_num: int = head_num or __model_default_values.head_num
    
    @classmethod
    def init_with_dict(cls, option_dict: Optional[RefLocalTrackingModel031MHArgumentsDict] = None):
        if option_dict is not None:
            ref_local_tracking_model_031_arguments = RefLocalTrackingModel031MHArguments(option_dict)
            return cls(
                    unet_l4_model_main=ref_local_tracking_model_031_arguments.unet_l4_model_main,
                    unet_l4_model_ref=ref_local_tracking_model_031_arguments.unet_l4_model_ref,
                    input_main_image_shape=ref_local_tracking_model_031_arguments.input_main_image_shape,
                    input_ref_image_shape=ref_local_tracking_model_031_arguments.input_ref_image_shape,
                    input_ref_bin_label_shape=ref_local_tracking_model_031_arguments.input_ref_bin_label_shape)
        else:
            return cls()
    
    @classmethod
    def init_with_str_dict(cls, option_str_dict: Optional[Dict[str, str]] = None):
        if option_str_dict is not None:
            unet_level_arguments_dict = RefLocalTrackingModel031MHArguments.convert_str_dict(option_str_dict)
            return cls.init_with_dict(unet_level_arguments_dict)
        else:
            return cls()
    
    def setup_model(self) -> Model:
        return self.ref_local_tracking_model_031_mh(
                unet_l4_model_main=self.unet_l4_model_main,
                unet_l4_model_ref=self.unet_l4_model_ref,
                bin_num=self.bin_num,
                input_main_image_shape=self.input_main_image_shape,
                input_ref_image_shape=self.input_ref_image_shape,
                input_ref_bin_label_shape=self.input_ref_bin_label_shape,
                head_num=self.head_num)
    
    # noinspection DuplicatedCode
    @staticmethod
    def ref_local_tracking_model_031_mh(
            unet_l4_model_main: Model = unet_level(level=4, input_shape=(256, 256, 1)),
            unet_l4_model_ref: Model = unet_level(level=4, input_shape=(256, 256, 1)),
            bin_num: int = 30,
            input_main_image_shape: Tuple[int, int, int] = (256, 256, 1),
            input_ref_image_shape: Tuple[int, int, int] = (256, 256, 1),
            input_ref_bin_label_shape: Tuple[int, int, int] = (256, 256, 30),
            head_num: int = 4) -> Model:
        def __aggregation_up(feature_map, filters: int):
            up_layer: Layer = UpSampling2D()(feature_map)
            up_conv_layer: Layer = Conv2D(
                    filters=filters,
                    kernel_size=3,
                    padding="same",
                    kernel_initializer="he_normal",
                    activation="relu")(up_layer)
            return up_conv_layer
        
        def __get_unet_layer(unet_model: Model):
            skip_names = [unet_model.layers[11].name, unet_model.layers[8].name, unet_model.layers[5].name,
                          unet_model.layers[2].name]
            model = Model(
                    inputs=unet_model.input,
                    outputs=[unet_model.get_layer(skip_names[0]).output, unet_model.get_layer(skip_names[1]).output,
                             unet_model.get_layer(skip_names[2]).output, unet_model.get_layer(skip_names[3]).output,
                             unet_model.output])
            return model
        
        main_unet_model = __get_unet_layer(unet_l4_model_main)
        ref_unet_model = __get_unet_layer(unet_l4_model_ref)
        
        # Inputs
        main_image_input: Layer = Input(shape=input_main_image_shape)
        ref_image_input: Layer = Input(shape=input_ref_image_shape)
        ref_label_input: Layer = Input(shape=input_ref_bin_label_shape)
        
        ref_label_1_input = ShrinkLayer(bin_num=bin_num, resize_by_power_of_two=3)(ref_label_input)
        ref_label_2_input = ShrinkLayer(bin_num=bin_num, resize_by_power_of_two=2)(ref_label_input)
        ref_label_3_input = ShrinkLayer(bin_num=bin_num, resize_by_power_of_two=1)(ref_label_input)
        ref_label_4_input = ShrinkLayer(bin_num=bin_num, resize_by_power_of_two=0)(ref_label_input)
        
        main_unet = main_unet_model(main_image_input)
        ref_unet = ref_unet_model(ref_image_input)
        
        # First
        diff_local1: Layer = RefLocal8MH(mode="dot", k_size=5, intermediate_dim=256, head_num=head_num)(
                [main_unet[0], ref_unet[0]])
        diff_agg1 = AggregationLayer(bin_size=bin_num, k_size=5, aggregate_mode="weighted_sum")(
                [diff_local1, ref_label_1_input])
        up1: Layer = __aggregation_up(diff_agg1, filters=bin_num)
        
        # Second
        diff_local2: Layer = RefLocal7MH(mode="dot", k_size=5, intermediate_dim=128, head_num=head_num)(
                [main_unet[1], ref_unet[1]])
        diff_agg2 = AggregationLayer2(bin_size=bin_num, k_size=5, aggregate_mode="weighted_sum")(
                [diff_local2, ref_label_2_input, up1])
        up2: Layer = __aggregation_up(diff_agg2, filters=bin_num)
        
        # Third
        diff_local3: Layer = RefLocal7MH(mode="dot", k_size=5, intermediate_dim=64, head_num=head_num)(
                [main_unet[2], ref_unet[2]])
        diff_agg3 = AggregationLayer2(bin_size=bin_num, k_size=5, aggregate_mode="weighted_sum")(
                [diff_local3, ref_label_3_input, up2])
        up3: Layer = __aggregation_up(diff_agg3, filters=bin_num)
        
        # Fourth
        diff_local4: Layer = RefLocal7MH(mode="dot", k_size=5, intermediate_dim=32, head_num=head_num)(
                [main_unet[3], ref_unet[3]])
        diff_agg4 = AggregationLayer2(bin_size=bin_num, k_size=5, aggregate_mode="weighted_sum")(
                [diff_local4, ref_label_4_input, up3])
        
        conv1 = Conv2D(filters=60, kernel_size=1, activation="relu", padding="same", kernel_initializer="he_normal")(
                diff_agg4)
        conv1 = Conv2D(filters=60, kernel_size=1, activation="relu", padding="same", kernel_initializer="he_normal")(
                conv1)
        
        # Outputs
        conv2 = Conv2D(
                filters=bin_num,
                kernel_size=1,
                activation="softmax",
                padding="same",
                kernel_initializer="he_normal")(conv1)
        
        # Input - [Main Image, Ref Image, Ref Color Bin Label]
        # Output - [Main BW Mask, Ref BW Mask, Main Color Bin Label]
        return Model(
                inputs=[main_image_input, ref_image_input, ref_label_input],
                outputs=[main_unet[4], ref_unet[4], conv2])
