# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
"""
Keras 용 EfficientNet 모델.

참조:
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (ICML 2019)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.util.tf_export import keras_export


BASE_WEIGHTS_PATH = "https://storage.googleapis.com/keras-applications/"

WEIGHTS_HASHES = {
    "b0": ("902e53a9f72be733fc0bcb005b3ebbac", "50bc09e76180e00e4465e1a485ddc09d"),
    "b1": ("1d254153d4ab51201f1646940f018540", "74c4e6b3e1f6a1eea24c589628592432"),
    "b2": ("b15cce36ff4dcbd00b6dd88e7857a6ad", "111f8e2ac8aa800a7a99e3239f7bfb39"),
    "b3": ("ffd1fdc53d0ce67064dc6a9c7960ede0", "af6d107764bb5b1abb91932881670226"),
    "b4": ("18c95ad55216b8f92d7e70b3a046e2fc", "ebc24e6d6c33eaebbd558eafbeedf1ba"),
    "b5": ("ace28f2a6363774853a83a0b21b9421a", "38879255a25d3c92d5e44e04ae6cec6f"),
    "b6": ("165f6e37dce68623721b423839de8be5", "9ecce42647a20130c1f39a5d4cb75743"),
    "b7": ("8c03f828fec3ef71311cd463b6759d99", "cbcfe4450ddf6f3ad90b1b398090fe4a"),
}

DEFAULT_BLOCKS_ARGS = [
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 32,
        "filters_out": 16,
        "expand_ratio": 1,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 2,
        "filters_in": 16,
        "filters_out": 24,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 2,
        "filters_in": 24,
        "filters_out": 40,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 3,
        "filters_in": 40,
        "filters_out": 80,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 3,
        "filters_in": 80,
        "filters_out": 112,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 4,
        "filters_in": 112,
        "filters_out": 192,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 192,
        "filters_out": 320,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
]

CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal"},
}

DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {"scale": 1.0 / 3.0, "mode": "fan_out", "distribution": "uniform"},
}

layers = VersionAwareLayers()

BASE_DOCSTRING = """
{name} 아키텍처를 인스턴스화합니다.

선택적으로 ImageNet에서 사전 트레이닝된 가중치를 로드합니다.
모델에서 사용하는 데이터 형식 규칙은 Keras 구성 `~/.keras/keras.json`에 지정된 규칙입니다.
구성하지 않은 경우, 기본값은 `"channels_last"`입니다.

References
----------
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (ICML 2019)


Arguments
---------
include_top: Whether to include the fully-connected
    layer at the top of the network. Defaults to True.
weights: One of `None` (random initialization),
        'imagenet' (pre-training on ImageNet),
        or the path to the weights file to be loaded. Defaults to 'imagenet'.
input_tensor: Optional Keras tensor
    (i.e. output of `layers.Input()`)
    to use as image input for the model.
input_shape: Optional shape tuple, only to be specified
    if `include_top` is False.
    It should have exactly 3 inputs channels.
pooling: Optional pooling mode for feature extraction
    when `include_top` is `False`. Defaults to None.
    - `None` means that the output of the model will be
        the 4D tensor output of the
        last convolutional layer.
    - `avg` means that global average pooling
        will be applied to the output of the
        last convolutional layer, and thus
        the output of the model will be a 2D tensor.
    - `max` means that global max pooling will
        be applied.
classes: Optional number of classes to classify images
    into, only to be specified if `include_top` is True, and
    if no `weights` argument is specified. Defaults to 1000 (number of
    ImageNet classes).
classifier_activation: A `str` or callable. The activation function to use
    on the "top" layer. Ignored unless `include_top=True`. Set
    `classifier_activation=None` to return the logits of the "top" layer.
    Defaults to 'softmax'.

Returns
-------
    `keras.Model` 인스턴스.
"""


def EfficientNet(
    width_coefficient,
    depth_coefficient,
    default_size,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    activation="swish",
    blocks_args="default",
    model_name="efficientnet",
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    """
    주어진 스케일링 계수를 사용하여, EfficientNet 아키텍처를 인스턴스화합니다.

    선택적으로 ImageNet에서 사전 트레이닝된 가중치를 로드합니다.
    모델에서 사용하는 데이터 형식 규칙은 Keras 구성 `~/.keras/keras.json`에 지정된 규칙입니다.

    Reference
    ---------
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (ICML 2019)

    Parameters
    ----------
    width_coefficient : float
        네트워크 width에 대한 스케일링 계수.
    depth_coefficient : float
        네트워크 depth에 대한 스케일링 계수.
    default_size : int
        기본 입력 이미지 크기.
    dropout_rate : float, optional, default=0.2
        최종 분류기 레이어 이전의 드롭아웃 비율.
    drop_connect_rate : float, optional, default=0.2
        스킵 연결에서 드롭아웃 비율.
    depth_divisor : int, optional, default=8
        네트워크 width의 유닛.
    activation : str, optional, default="swish"
        활성화 함수.
    blocks_args : str, optional, default="default"
        블록 모듈을 구성하기 위한 dicts, 매개변수 list.
    model_name : str, optional, default="efficientnet"
        모델 이름
    include_top : bool, optional, default=True
        네트워크 top에 완전 연결 레이어를 포함할지 여부.
    weights : str, optional, default="imagenet"
        `None`(무작위 초기화), 'imagenet' (ImageNet에 대해 사전 트레이닝) 중 하나 또는 로드할 가중치 파일의 경로입니다.
    input_tensor : [type], optional, default=None
        모델의 이미지 입력으로 사용할 선택적 Keras 텐서(즉, `layers.Input()`의 출력).
    input_shape : [type], optional, default=None
        선택적 shape 튜플, `include_top`이 False인 경우에만 지정됩니다.
        정확히 3개 입력 채널이 있어야 합니다.
    pooling : [type], optional, default=None
        `include_top`이 `False` 인 경우, 특성 추출을 위한 선택적 풀링 모드
        - `None` 모델의 출력이 마지막 컨볼루션 레이어의 4D 텐서 출력이 됨을 의미합니다.
        - `avg` 글로벌 평균 풀링이 마지막 컨볼루션 레이어의 출력에 적용됨을 의미합니다. 따라서, 모델의 출력은 2D 텐서가 됩니다.
        - `max` 글로벌 최대 풀링이 적용됨을 의미합니다.
    classes : int, optional, default=1000
        이미지를 분류할 클래스 수 (선택 사항). `include_top`이  True이고, `weights` 인수가 지정되지 않은 경우에만 지정됩니다.
    classifier_activation : str, optional, default="softmax"
        `str` 또는 callable. "top" 레이어에서 사용할 활성화 함수입니다.
        `include_top=True`가 아니면 무시됩니다.
        "top" 레이어의 로짓을 반환하려면, `classifier_activation=None`을 설정하십시오.

    Returns
    -------
    `keras.Model`
        `keras.Model` 인스턴스.

    Raises
    ------
    ValueError
        `weights`에 대한 인수가 잘못되었거나, 입력 shape이 잘못된 경우.
    ValueError
        사전 트레이닝된 top 레이어를 사용할 때, `classifier_activation`이 `softmax` 또는 `None`이 아닌 경우
    """
    if blocks_args == "default":
        blocks_args = DEFAULT_BLOCKS_ARGS

    if not (weights in {"imagenet", None} or file_io.file_exists_v2(weights)):
        raise ValueError(
            "`weights` 인수는 `None` (무작위 초기화), "
            "`imagenet` (ImageNet에 대해 사전 트레이닝된) 또는, "
            "로드할 가중치 파일의 경로여야 합니다."
        )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            '`include_top`이 true이고, `"imagenet"`으로 `weights`를 사용하는 경우,'
            "`classes`는 1000이어야 합니다."
        )

    # 적절한 입력 shape을 결정합니다.
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    def round_filters(filters, divisor=depth_divisor):
        """깊이 승수(depth multiplier)를 기반으로 한 필터 수를 반올림하는 메서드입니다."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # 반올림의 내림으로 인해 10% 이상 내려가지 않도록 하십시오.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """깊이 승수(depth multiplier)를 기반으로 한 반올림 반복 횟수입니다."""
        return int(math.ceil(depth_coefficient * repeats))

    # Build stem
    x = img_input
    x = layers.Rescaling(1.0 / 255.0)(x)
    x = layers.Normalization(axis=bn_axis)(x)

    x = layers.ZeroPadding2D(
        padding=imagenet_utils.correct_pad(x, 3), name="stem_conv_pad"
    )(x)
    x = layers.Conv2D(
        round_filters(32),
        3,
        strides=2,
        padding="valid",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name="stem_conv",
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name="stem_bn")(x)
    x = layers.Activation(activation, name="stem_activation")(x)

    # Build blocks
    blocks_args = copy.deepcopy(blocks_args)

    b = 0
    blocks = float(sum(round_repeats(args["repeats"]) for args in blocks_args))
    for (i, args) in enumerate(blocks_args):
        assert args["repeats"] > 0
        # Update block input and output filters based on depth multiplier.
        args["filters_in"] = round_filters(args["filters_in"])
        args["filters_out"] = round_filters(args["filters_out"])

        for j in range(round_repeats(args.pop("repeats"))):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args["strides"] = 1
                args["filters_in"] = args["filters_out"]
            x = block(
                x,
                activation,
                drop_connect_rate * b / blocks,
                name="block{}{}_".format(i + 1, chr(j + 97)),
                **args
            )
            b += 1

    # Build top
    x = layers.Conv2D(
        round_filters(1280),
        1,
        padding="same",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name="top_conv",
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name="top_bn")(x)
    x = layers.Activation(activation, name="top_activation")(x)
    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name="top_dropout")(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(
            classes,
            activation=classifier_activation,
            kernel_initializer=DENSE_KERNEL_INITIALIZER,
            name="predictions",
        )(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D(name="max_pool")(x)

    # 모델이 `input_tensor`의 잠재적 선행자(potential predecessors)를 고려하는지 확인합니다.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # 모델 생성
    model = training.Model(inputs, x, name=model_name)

    # 가중치 불러오기
    if weights == "imagenet":
        if include_top:
            file_suffix = ".h5"
            file_hash = WEIGHTS_HASHES[model_name[-2:]][0]
        else:
            file_suffix = "_notop.h5"
            file_hash = WEIGHTS_HASHES[model_name[-2:]][1]
        file_name = model_name + file_suffix
        weights_path = data_utils.get_file(
            file_name,
            BASE_WEIGHTS_PATH + file_name,
            cache_subdir="models",
            file_hash=file_hash,
        )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    return model


def block(
    inputs,
    activation="swish",
    drop_rate=0.0,
    name="",
    filters_in=32,
    filters_out=16,
    kernel_size=3,
    strides=1,
    expand_ratio=1,
    se_ratio=0.0,
    id_skip=True,
):
    """
    Inverted ResNet 블록.

    Parameters
    ----------
    inputs : [type]
        입력 텐서.
    activation : str, optional, default="swish"
        활성화 함수.
    drop_rate : float, optional, default=0.0
        0과 1 사이의 float, 드롭할 입력 유닛의 비율입니다.
    name : str, optional, default=""
        블록 라벨.
    filters_in : int, optional, default=32
        입력 필터의 수.
    filters_out : int, optional, default=16
        출력 필터의 수.
    kernel_size : int, optional, default=3
        컨볼루션 윈도우의 차원.
    strides : int, optional, default=1
        컨볼루션의 스트라이드.
    expand_ratio : int, optional, default=1
        입력 필터에 대한 스케일링 계수.
    se_ratio : float, optional, default=0.0
        0과 1 사이의 float, 입력 필터를 squeeze 하는 비율.
    id_skip : bool, optional, default=True
        [description], by

    Returns
    -------
    [type]
        블록에 대한 출력 텐서.
    """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(
            filters,
            1,
            padding="same",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "expand_conv",
        )(inputs)
        x = layers.BatchNormalization(axis=bn_axis, name=name + "expand_bn")(x)
        x = layers.Activation(activation, name=name + "expand_activation")(x)
    else:
        x = inputs

    # Depthwise 컨볼루션
    if strides == 2:
        x = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, kernel_size), name=name + "dwconv_pad"
        )(x)
        conv_pad = "valid"
    else:
        conv_pad = "same"
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        padding=conv_pad,
        use_bias=False,
        depthwise_initializer=CONV_KERNEL_INITIALIZER,
        name=name + "dwconv",
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + "bn")(x)
    x = layers.Activation(activation, name=name + "activation")(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D(name=name + "se_squeeze")(x)
        se = layers.Reshape((1, 1, filters), name=name + "se_reshape")(se)
        se = layers.Conv2D(
            filters_se,
            1,
            padding="same",
            activation=activation,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "se_reduce",
        )(se)
        se = layers.Conv2D(
            filters,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "se_expand",
        )(se)
        x = layers.multiply([x, se], name=name + "se_excite")

    # Output phase
    x = layers.Conv2D(
        filters_out,
        1,
        padding="same",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + "project_conv",
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + "project_bn")(x)
    if id_skip and strides == 1 and filters_in == filters_out:
        if drop_rate > 0:
            x = layers.Dropout(
                drop_rate, noise_shape=(None, 1, 1, 1), name=name + "drop"
            )(x)
        x = layers.add([x, inputs], name=name + "add")
    return x


@keras_export(
    "keras.applications.efficientnet.EfficientNetB0",
    "keras.applications.EfficientNetB0",
)
def EfficientNetB0(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs
):
    return EfficientNet(
        1.0,
        1.0,
        224,
        0.2,
        model_name="efficientnetb0",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs
    )


@keras_export(
    "keras.applications.efficientnet.EfficientNetB1",
    "keras.applications.EfficientNetB1",
)
def EfficientNetB1(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs
):
    return EfficientNet(
        1.0,
        1.1,
        240,
        0.2,
        model_name="efficientnetb1",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs
    )


@keras_export(
    "keras.applications.efficientnet.EfficientNetB2",
    "keras.applications.EfficientNetB2",
)
def EfficientNetB2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs
):
    return EfficientNet(
        1.1,
        1.2,
        260,
        0.3,
        model_name="efficientnetb2",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs
    )


@keras_export(
    "keras.applications.efficientnet.EfficientNetB3",
    "keras.applications.EfficientNetB3",
)
def EfficientNetB3(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs
):
    return EfficientNet(
        1.2,
        1.4,
        300,
        0.3,
        model_name="efficientnetb3",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs
    )


@keras_export(
    "keras.applications.efficientnet.EfficientNetB4",
    "keras.applications.EfficientNetB4",
)
def EfficientNetB4(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs
):
    return EfficientNet(
        1.4,
        1.8,
        380,
        0.4,
        model_name="efficientnetb4",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs
    )


@keras_export(
    "keras.applications.efficientnet.EfficientNetB5",
    "keras.applications.EfficientNetB5",
)
def EfficientNetB5(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs
):
    return EfficientNet(
        1.6,
        2.2,
        456,
        0.4,
        model_name="efficientnetb5",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs
    )


@keras_export(
    "keras.applications.efficientnet.EfficientNetB6",
    "keras.applications.EfficientNetB6",
)
def EfficientNetB6(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs
):
    return EfficientNet(
        1.8,
        2.6,
        528,
        0.5,
        model_name="efficientnetb6",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs
    )


@keras_export(
    "keras.applications.efficientnet.EfficientNetB7",
    "keras.applications.EfficientNetB7",
)
def EfficientNetB7(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs
):
    return EfficientNet(
        2.0,
        3.1,
        600,
        0.5,
        model_name="efficientnetb7",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs
    )


EfficientNetB0.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB0")
EfficientNetB1.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB1")
EfficientNetB2.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB2")
EfficientNetB3.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB3")
EfficientNetB4.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB4")
EfficientNetB5.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB5")
EfficientNetB6.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB6")
EfficientNetB7.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB7")


@keras_export("keras.applications.efficientnet.preprocess_input")
def preprocess_input(x, data_format=None):  # pylint: disable=unused-argument
    return x


@keras_export("keras.applications.efficientnet.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__