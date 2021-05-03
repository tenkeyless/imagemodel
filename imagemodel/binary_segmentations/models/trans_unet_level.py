import warnings
from typing import Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    Conv2D,
    Dense,
    Dropout,
    Embedding,
    Input,
    Layer,
    LayerNormalization,
    MaxPooling2D,
    MultiHeadAttention,
    Reshape,
    UpSampling2D,
    concatenate
)
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


class Patches(Layer):
    def __init__(self, patch_size: int):
        super(Patches, self).__init__()
        self.patch_size: int = patch_size
    
    def call(self, images, **kwargs):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID")
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
    def get_config(self):
        return {"patch_size": self.patch_size}


class PatchEncoder(Layer):
    def __init__(self, num_patches: int, projection_dim: int):
        super(PatchEncoder, self).__init__()
        self.num_patches: int = num_patches
        # self.projection = Dense(units=projection_dim)
        self.projection = Reshape((-1, projection_dim))  # x = x.flatten(2)
        # transpose? x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        # 이게 있기는 하지만 안해도 될 것 같다.
        # block에서 attention 계산하는데, 다시 x.permute(0, 2, 1, 3)로 계산하기 때문이라 생각한다.
        self.position_embedding = Embedding(input_dim=num_patches, output_dim=projection_dim)
        # self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
    
    def call(self, patch, **kwargs):
        # positions = tf.range(start=0, limit=self.num_patches, delta=1)
        # TODO: Is 'zero positions' implemented correctly?
        positions = tf.zeros(self.num_patches, dtype=tf.int32)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
    def get_config(self):
        return {"num_patches": self.num_patches, "projection_dim": self.projection_dim}


class Mlp(Layer):
    def __init__(self, units: int, dropout_rate: float, **kwargs):
        super().__init__(**kwargs)
        self.dense_layer = Dense(units, activation=tf.nn.gelu)
        self.dropout_layer = Dropout(dropout_rate)
    
    def call(self, inputs, **kwargs):
        x = self.dense_layer(inputs)
        x = self.dropout_layer(x)
        return x
    
    def get_config(self):
        return {"units": self.units, "dropout_rate": self.dropout_rate}


class TransformerBlock(Layer):
    def __init__(
            self,
            mh_num_heads: int,
            mh_projection_dim: int,
            mlp_transformer_units: List[int],
            mh_dropout: float = 0.1,
            mlp_dropout: float = 0.1,
            **kwargs):
        super().__init__(**kwargs)
        self.norm_layer_1 = LayerNormalization(epsilon=1e-6)
        self.norm_layer_2 = LayerNormalization(epsilon=1e-6)
        self.multi_head_attention_layer = MultiHeadAttention(
                num_heads=mh_num_heads,
                key_dim=mh_projection_dim,
                dropout=mh_dropout)
        self.mlp_layers: List[Layer] = []
        for mlp_transformer_unit in mlp_transformer_units:
            self.mlp_layers.append(Mlp(mlp_transformer_unit, mlp_dropout))
    
    def call(self, inputs, **kwargs):
        # 레이어 정규화(normalization) 1.
        x1 = self.norm_layer_1(inputs)
        # 멀티 헤드 어텐션 레이어 생성.
        attention_output = self.multi_head_attention_layer(x1, x1)
        # 스킵 연결 1.
        x2 = Add()([attention_output, inputs])
        # 레이어 정규화 2.
        x3 = self.norm_layer_2(x2)
        # MLP.
        for mlp_layer in self.mlp_layers:
            x3 = mlp_layer(x3)
        # 스킵 연결 2.
        return Add()([x3, x2])
    
    def get_config(self):
        return {
            "mh_num_heads": self.mh_num_heads,
            "mh_projection_dim": self.mh_projection_dim,
            "mlp_transformer_units": self.mlp_transformer_units,
            "mh_dropout": self.mh_dropout,
            "mlp_dropout": self.mlp_dropout}


class ViT(Layer):
    def __init__(self, patch_size: int, projection_dim: int, num_transformer_layers: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        self.patch_size: int = patch_size
        self.projection_dim: int = projection_dim
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        
        self.patches_layer: Layer = Patches(self.patch_size)
        self.encoded_patches_layer: Optional[Layer] = None
        self.norm_layer: Layer = LayerNormalization(epsilon=1e-6)
        
        # transformer_units = [self.projection_dim * 2, self.projection_dim]
        transformer_units = [3072, self.projection_dim]  # ViT-B/16 configuration
        
        self.transformer_block_layers: List[Layer] = []
        for _ in range(num_transformer_layers):
            self.transformer_block_layers.append(
                    TransformerBlock(
                            mh_num_heads=num_heads,
                            mh_projection_dim=projection_dim,
                            mlp_transformer_units=transformer_units))
    
    def build(self, input_shape):
        num_patches = input_shape[1] ** 2
        self.encoded_patches_layer = PatchEncoder(num_patches, self.projection_dim)
    
    def call(self, inputs, **kwargs):
        # 패치 생성
        patches = self.patches_layer(inputs)
        # 패치 인코딩
        encoded_patches = self.encoded_patches_layer(patches)
        # 트랜스포머 블록
        for _transformer_block_layer in self.transformer_block_layers:
            encoded_patches = _transformer_block_layer(encoded_patches)
        
        representation = self.norm_layer(encoded_patches)
        return representation
    
    def get_config(self):
        return {
            "patch_size": self.patch_size,
            "projection_dim": self.projection_dim,
            "num_transformer_layers": self.num_transformer_layers,
            "num_heads": self.num_heads}


class TransUNetLevelArgumentsDict(TypedDict):
    level: Optional[int]
    input_shape: Optional[Tuple[int, int, int]]
    input_name: Optional[str]
    output_name: Optional[str]
    base_filters: Optional[int]


class TransUNetLevelArguments(ModelArguments[TransUNetLevelArgumentsDict]):
    def __init__(self, dic: TransUNetLevelArgumentsDict):
        self.dic = dic
    
    @classmethod
    def init_from_str_dict(cls, string_dict: Dict[str, str]):
        return cls(cls.convert_str_dict(string_dict))
    
    # noinspection DuplicatedCode
    @classmethod
    def convert_str_dict(cls, string_dict: Dict[str, str]) -> TransUNetLevelArgumentsDict:
        __keys: List[str] = list(TransUNetLevelArgumentsDict.__annotations__)
        
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
        
        return TransUNetLevelArgumentsDict(
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


class TransUNetLevelModelManager(CommonModelManager, CommonModelManagerDictGeneratable[TransUNetLevelArgumentsDict]):
    def __init__(
            self,
            level: Optional[int] = None,
            input_shape: Optional[Tuple[int, int, int]] = None,
            input_name: Optional[str] = None,
            output_name: Optional[str] = None,
            base_filters: Optional[int] = None):
        __model_default_args = get_default_args(self.trans_unet_level)
        __model_default_values = TransUNetLevelArguments(__model_default_args)
        
        self.level: int = level or __model_default_values.level
        self.input_shape: Tuple[int, int, int] = input_shape or __model_default_values.input_shape
        self.input_name: str = input_name or __model_default_values.input_name
        self.input_name = self.layer_name_correction(self.input_name)
        self.output_name: str = output_name or __model_default_values.output_name
        self.output_name = self.layer_name_correction(self.output_name)
        self.base_filters: int = base_filters or __model_default_values.base_filters
    
    @classmethod
    def init_with_dict(cls, option_dict: Optional[TransUNetLevelArgumentsDict] = None):
        if option_dict is not None:
            trans_unet_level_arguments = TransUNetLevelArguments(option_dict)
            return cls(
                    level=trans_unet_level_arguments.level,
                    input_shape=trans_unet_level_arguments.input_shape,
                    input_name=trans_unet_level_arguments.input_name,
                    output_name=trans_unet_level_arguments.output_name,
                    base_filters=trans_unet_level_arguments.base_filters)
        else:
            return cls()
    
    @classmethod
    def init_with_str_dict(cls, option_str_dict: Optional[Dict[str, str]] = None):
        if option_str_dict is not None:
            trans_unet_level_arguments_dict = TransUNetLevelArguments.convert_str_dict(option_str_dict)
            return cls.init_with_dict(trans_unet_level_arguments_dict)
        else:
            return cls()
    
    def setup_model(self) -> Model:
        return self.trans_unet_level(
                level=self.level,
                input_shape=self.input_shape,
                input_name=self.input_name,
                output_name=self.output_name,
                base_filters=self.base_filters)
    
    # noinspection DuplicatedCode
    @staticmethod
    def trans_unet_level(
            level: int = 4,
            input_shape: Tuple[int, int, int] = (256, 256, 1),
            input_name: str = "trans_unet_input",
            output_name: str = "trans_unet_output",
            base_filters: int = 16) -> Model:
        def __trans_unet_level_base_conv_2d(
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
        
        def __trans_unet_base_sub_sampling(pool_size=(2, 2)) -> Layer:
            return MaxPooling2D(pool_size=pool_size)
        
        def __trans_unet_base_up_sampling(
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
            encoder = __trans_unet_level_base_conv_2d(filter_num)(encoder)
            encoder = __trans_unet_level_base_conv_2d(filter_num)(encoder)
            skip_connections.append(encoder)
            encoder = __trans_unet_base_sub_sampling()(encoder)
        
        # Intermediate
        patch_size = input_shape[0] // (2 ** (level - 1))
        projection_dim = 384
        vit_head = 6
        vit_transform_layers = 6
        
        intermedate: Layer = __trans_unet_level_base_conv_2d(filter_nums[-1])(encoder)
        intermedate = __trans_unet_level_base_conv_2d(filter_nums[-1])(intermedate)
        intermedate = Conv2D(
                filters=projection_dim,
                kernel_size=1,
                strides=1,
                activation="relu",
                padding="same",
                kernel_initializer="he_normal")(intermedate)  # (B, 32, 32, projection_dim)
        intermedate = ViT(
                patch_size=patch_size,
                projection_dim=projection_dim,
                num_transformer_layers=vit_transform_layers,
                num_heads=vit_head)(intermedate)
        intermedate = Reshape((patch_size, patch_size, projection_dim))(intermedate)
        intermedate = Dropout(0.5)(intermedate)
        intermedate = Conv2D(
                filters=512,
                kernel_size=3,
                activation="relu",
                padding='same',
                kernel_initializer="he_normal")(intermedate)
        
        # Decoder
        decoder: Layer = intermedate
        for filter_num_index, filter_num in enumerate(filter_nums[:-1][::-1]):
            decoder = __trans_unet_base_up_sampling(filter_num)(decoder)
            skip_layer: Layer = skip_connections[::-1][filter_num_index]
            decoder = concatenate([skip_layer, decoder])
            decoder = __trans_unet_level_base_conv_2d(filter_num)(decoder)
            decoder = __trans_unet_level_base_conv_2d(filter_num)(decoder)
        
        # Output
        output: Layer = __trans_unet_level_base_conv_2d(2)(decoder)
        output = __trans_unet_level_base_conv_2d(1, kernel_size=1, activation="sigmoid", name_optional=output_name)(
                output)
        
        return Model(inputs=[input_layer], outputs=[output])
