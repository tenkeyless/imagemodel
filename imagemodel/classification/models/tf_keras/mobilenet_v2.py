# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""
Keras 용 MobileNet v2 모델.

MobileNetV2는 일반 아키텍처이며, 여러 사용 사례에 사용할 수 있습니다.
사용 사례에 따라, 다른 입력 레이어 크기와 다른 너비 factor를 사용할 수 있습니다.
이를 통해 서로 다른 너비 모델이 multiply-add 횟수를 줄여, 모바일 장치의 추론 비용을 줄일 수 있습니다.
MobileNetV2는, bottlenecking 특성이 있는 inverted residual 블록을 사용한다는 점을 제외하면, 원래 MobileNet과 매우 유사합니다.
원래 MobileNet보다 매개변수 수가 현저히 적습니다.
MobileNets는 32x32보다 큰 모든 입력 크기를 지원하며, 더 큰 이미지 크기는 더 나은 성능을 제공합니다.
매개변수의 수와 multiply-add의 수는, 각 레이어의 필터 수를 증가/감소시키는, `alpha` 매개변수를 사용하여 수정할 수 있습니다.
이미지 크기와 `alpha` 매개변수를 변경하면, ImageNet 가중치가 제공되는, 논문에 나온 22개 모델을 모두 만들 수 있습니다.
이 논문은 1.0(100% MobileNet이라고도 함), 0.35, 0.5, 0.75, 1.0, 1.3 및 1.4의 `alpha`값을 사용하여, MobileNet의 성능을 보여줍니다.
이러한 `alpha` 값 각각에 대해, 5가지 다른 입력 이미지 크기(224, 192, 160, 128, 96)에 대한 가중치가 제공됩니다.

다음 표는 다양한 입력 크기에 대해 MobileNet의 성능을 설명합니다.
------------------------------------------------------------------------
MACs은 Multiply Adds를 나타냅니다.
|분류 체크포인트|MACs (M)|매개변수 (M)|Top 1 정확도|Top 5 정확도|
--------------------------|------------|---------------|---------|----|---------
| [mobilenet_v2_1.4_224]  | 582 | 6.06 |          75.0 | 92.5 |
| [mobilenet_v2_1.3_224]  | 509 | 5.34 |          74.4 | 92.1 |
| [mobilenet_v2_1.0_224]  | 300 | 3.47 |          71.8 | 91.0 |
| [mobilenet_v2_1.0_192]  | 221 | 3.47 |          70.7 | 90.1 |
| [mobilenet_v2_1.0_160]  | 154 | 3.47 |          68.8 | 89.0 |
| [mobilenet_v2_1.0_128]  | 99  | 3.47 |          65.3 | 86.9 |
| [mobilenet_v2_1.0_96]   | 56  | 3.47 |          60.3 | 83.2 |
| [mobilenet_v2_0.75_224] | 209 | 2.61 |          69.8 | 89.6 |
| [mobilenet_v2_0.75_192] | 153 | 2.61 |          68.7 | 88.9 |
| [mobilenet_v2_0.75_160] | 107 | 2.61 |          66.4 | 87.3 |
| [mobilenet_v2_0.75_128] | 69  | 2.61 |          63.2 | 85.3 |
| [mobilenet_v2_0.75_96]  | 39  | 2.61 |          58.8 | 81.6 |
| [mobilenet_v2_0.5_224]  | 97  | 1.95 |          65.4 | 86.4 |
| [mobilenet_v2_0.5_192]  | 71  | 1.95 |          63.9 | 85.4 |
| [mobilenet_v2_0.5_160]  | 50  | 1.95 |          61.0 | 83.2 |
| [mobilenet_v2_0.5_128]  | 32  | 1.95 |          57.7 | 80.8 |
| [mobilenet_v2_0.5_96]   | 18  | 1.95 |          51.2 | 75.8 |
| [mobilenet_v2_0.35_224] | 59  | 1.66 |          60.3 | 82.9 |
| [mobilenet_v2_0.35_192] | 43  | 1.66 |          58.2 | 81.2 |
| [mobilenet_v2_0.35_160] | 30  | 1.66 |          55.7 | 79.1 |
| [mobilenet_v2_0.35_128] | 20  | 1.66 |          50.8 | 75.0 |
| [mobilenet_v2_0.35_96]  | 11  | 1.66 |          45.5 | 70.4 |

참조:
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) (CVPR 2018)
"""
from __future__ import absolute_import, division, print_function

from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.utils import data_utils, layer_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export

BASE_WEIGHT_PATH = (
    "https://storage.googleapis.com/tensorflow/" "keras-applications/mobilenet_v2/"
)
layers = None


@keras_export(
    "keras.applications.mobilenet_v2.MobileNetV2", "keras.applications.MobileNetV2"
)
def MobileNetV2(
    input_shape=None,
    alpha=1.0,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs
):
    """
    MobileNetV2 아키텍처를 인스턴스화합니다.

    선택적으로 ImageNet에서 사전 트레이닝된 가중치를 로드합니다.

    참고 : 각 Keras 애플리케이션에는 특정 종류의 입력 전처리가 필요합니다.
    MobileNetV2의 경우, 입력을 모델에 전달하기 전에 입력에 대해,
    `tf.keras.applications.mobilenet_v2.preprocess_input`을 호출해야 합니다.

    References
    ----------
    - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) (CVPR 2018)

    Parameters
    ----------
    input_shape : [type], optional, default=None
        입력 이미지 해상도가 (224, 224, 3)이 아닌 모델을 사용하려는 경우, 지정할 Optional shape 튜플입니다.
        정확히 3개의 입력 채널 (224, 224, 3)이 있어야 합니다.
        `input_tensor`로부터 `input_shape`를 추론하려는 경우, 이 옵션을 생략할 수도 있습니다.
        `input_tensor`와 `input_shape`를 모두 포함하는 경우,
        일치한다면 `input_shape`가 사용되며, 모양이 일치하지 않으면 오류가 발생합니다.
        예 : `(160, 160, 3)`은 하나의 유효한 값입니다.
    alpha : float, optional, default=1.0
        0과 1 사이의 Float. 네트워크의 너비를 제어합니다.
        이것은 MobileNetV2 논문에서 너비 승수(width multiplier)라고 알려져 있지만,
        Keras의 `applications.MobileNetV1` 모델과 일관성을 유지하기 위해 이 이름이 유지됩니다.
        - `alpha` < 1.0이면, 각 레이어의 필터 수를 비례적으로 줄입니다.
        - `alpha` > 1.0이면, 각 레이어의 필터 수를 비례적으로 늘립니다.
        - `alpha` = 1이면, 논문으로부터의 기본 필터 수가 각 레이어에 사용됩니다.
    include_top : bool, optional, default=True
        네트워크 상단에 있는 완전 연결 레이어를 포함할지 여부
    weights : str, optional, default="imagenet"
        `None`(무작위 초기화), `'imagenet'`(ImageNet에 대해 사전 트레이닝된) 또는 로드할 가중치 파일의 경로 중 하나입니다.
    input_tensor : [type], optional, default=None
        모델의 이미지 입력으로 사용할 Optional Keras 텐서 (즉,`layers.Input()`의 출력)
    pooling : str, optional, default=None
        `include_top`이 `False`인 경우, 특성 추출을 위한 선택적 풀링 모드입니다.
        - `None`은 모델의 출력이 마지막 컨볼루션 블록의 4D 텐서 출력이 될 것임을 의미합니다.
        - `avg`는 글로벌 평균 풀링이 마지막 컨볼루션 블록의 출력에 적용되므로, 모델의 출력이 2D텐서가 될 것임을 의미합니다.
        - 'max'는 글로벌 최대 풀링이 적용됨을 의미합니다.
    classes : int, optional, default=1000
        이미지를 분류할 클래스 수. `include_top`이 `True`이고, `weights` 인수가 지정되지 않은 경우에만, 지정합니다.
    classifier_activation : str, optional, default="softmax"
        `str` 또는 callable. "top" 레이어에서 사용할 활성화 함수입니다. `include_top=True`가 아니면 무시됩니다.
        "top" 레이어의 로짓을 반환하려면, `classifier_activation=None`을 설정하십시오.
    **kwargs:
        이전 버전과의 호환성 만을 위해.

    Returns
    -------
    `keras.Model`
        `keras.Model` 인스턴스.

    Raises
    ------
    ValueError
        `weights`에 대한 인수가 잘못되었거나, 입력 shape이 잘못되었거나, 알파가 잘못된 경우, `weights='imagenet'`인 행
    ValueError
        사전 트레이닝된 top 레이어를 사용할 때, `classifier_activation`이 `softmax` 또는 `None`이 아닌 경우
    """
    global layers
    if "layers" in kwargs:
        layers = kwargs.pop("layers")
    else:
        layers = VersionAwareLayers()
    if kwargs:
        raise ValueError("알 수 없는 인수(들): %s" % (kwargs,))
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

    # 적절한 입력 shape과 기본 크기를 결정합니다.
    # input_shape 및 input_tensor가 모두 사용되는 경우, 일치해야 합니다.
    if input_shape is not None and input_tensor is not None:
        try:
            is_input_t_tensor = backend.is_keras_tensor(input_tensor)
        except ValueError:
            try:
                is_input_t_tensor = backend.is_keras_tensor(
                    layer_utils.get_source_inputs(input_tensor)
                )
            except ValueError:
                raise ValueError(
                    "input_tensor: ", input_tensor, "가 input_tensor 타입이 아닙니다."
                )
        if is_input_t_tensor:
            if backend.image_data_format == "channels_first":
                if backend.int_shape(input_tensor)[1] != input_shape[1]:
                    raise ValueError(
                        "input_shape: ",
                        input_shape,
                        "및 input_tensor: ",
                        input_tensor,
                        "동일한 shape 요구사항을 충족하지 않습니다.",
                    )
            else:
                if backend.int_shape(input_tensor)[2] != input_shape[1]:
                    raise ValueError(
                        "input_shape: ",
                        input_shape,
                        "및 input_tensor: ",
                        input_tensor,
                        "동일한 shape 요구사항을 충족하지 않습니다.",
                    )
        else:
            raise ValueError("input_tensor 지정: ", input_tensor, "이 keras tensor가 아닙니다.")

    default_size = None
    # input_shape가 None이면, input_tensor로부터 shape을 추론합니다.
    if input_shape is None and input_tensor is not None:

        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError(
                "input_tensor: ",
                input_tensor,
                "은 타입: ",
                type(input_tensor),
                "이는 유요한 타입이 아닙니다.",
            )

        if input_shape is None and not backend.is_keras_tensor(input_tensor):
            default_size = 224
        elif input_shape is None and backend.is_keras_tensor(input_tensor):
            if backend.image_data_format() == "channels_first":
                rows = backend.int_shape(input_tensor)[2]
                cols = backend.int_shape(input_tensor)[3]
            else:
                rows = backend.int_shape(input_tensor)[1]
                cols = backend.int_shape(input_tensor)[2]

            if rows == cols and rows in [96, 128, 160, 192, 224]:
                default_size = rows
            else:
                default_size = 224

    # input_shape가 None이고 input_tensor가 없는 경우
    elif input_shape is None:
        default_size = 224

    # input_shape가 None이 아니면, 기본 크기로 가정합니다.
    else:
        if backend.image_data_format() == "channels_first":
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [96, 128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )

    if backend.image_data_format() == "channels_last":
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == "imagenet":
        if alpha not in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]:
            raise ValueError(
                "imagenet 가중치가 로드되는 경우, "
                "알파는 `0.35`, `0.50`, `0.75`, `1.0`, `1.3`, `1.4`중 하나만 될 수 있습니다."
            )

        if rows != cols or rows not in [96, 128, 160, 192, 224]:
            rows = 224
            logging.warning(
                "`input_shape`가 정의되지 않았거나, 정사각형이 아니거나, "
                "`rows`가 [96, 128, 160, 192, 224]에 없습니다."
                "입력 shape (224, 224)에 대한 가중치가 기본값으로 로드됩니다."
            )

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = layers.Conv2D(
        first_block_filters,
        kernel_size=3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        name="Conv1",
    )(img_input)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name="bn_Conv1"
    )(x)
    x = layers.ReLU(6.0, name="Conv1_relu")(x)

    x = _inverted_res_block(
        x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0
    )

    x = _inverted_res_block(
        x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1
    )
    x = _inverted_res_block(
        x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2
    )

    x = _inverted_res_block(
        x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3
    )
    x = _inverted_res_block(
        x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4
    )
    x = _inverted_res_block(
        x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5
    )

    x = _inverted_res_block(
        x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6
    )
    x = _inverted_res_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7
    )
    x = _inverted_res_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8
    )
    x = _inverted_res_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9
    )

    x = _inverted_res_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10
    )
    x = _inverted_res_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11
    )
    x = _inverted_res_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12
    )

    x = _inverted_res_block(
        x, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13
    )
    x = _inverted_res_block(
        x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14
    )
    x = _inverted_res_block(
        x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15
    )

    x = _inverted_res_block(
        x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16
    )

    # 논문에 명시된 바와 같이, 마지막 conv에 적용된 alpha 없음:
    # 너비 승수(width multiplier)가 1보다 크면, 출력 채널 수를 늘립니다.
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = layers.Conv2D(last_block_filters, kernel_size=1, use_bias=False, name="Conv_1")(
        x
    )
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name="Conv_1_bn"
    )(x)
    x = layers.ReLU(6.0, name="out_relu")(x)

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(classes, activation=classifier_activation, name="predictions")(
            x
        )

    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D()(x)

    # 모델이 `input_tensor`의 잠재적 선행자(potential predecessors)를 고려하는지 확인합니다.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # 모델 생성
    model = training.Model(inputs, x, name="mobilenetv2_%0.2f_%s" % (alpha, rows))

    # 가중치 불러오기
    if weights == "imagenet":
        if include_top:
            model_name = (
                "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_"
                + str(alpha)
                + "_"
                + str(rows)
                + ".h5"
            )
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = data_utils.get_file(
                model_name, weight_path, cache_subdir="models"
            )
        else:
            model_name = (
                "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_"
                + str(alpha)
                + "_"
                + str(rows)
                + "_no_top"
                + ".h5"
            )
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = data_utils.get_file(
                model_name, weight_path, cache_subdir="models"
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    """
    Inverted ResNet 블록.

    Parameters
    ----------
    inputs : [type]
        [description]
    expansion : [type]
        [description]
    stride : [type]
        [description]
    alpha : [type]
        [description]
    filters : [type]
        [description]
    block_id : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1

    in_channels = backend.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = "block_{}_".format(block_id)

    if block_id:
        # Expand
        x = layers.Conv2D(
            expansion * in_channels,
            kernel_size=1,
            padding="same",
            use_bias=False,
            activation=None,
            name=prefix + "expand",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + "expand_BN"
        )(x)
        x = layers.ReLU(6.0, name=prefix + "expand_relu")(x)
    else:
        prefix = "expanded_conv_"

    # Depthwise
    if stride == 2:
        x = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, 3), name=prefix + "pad"
        )(x)
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        activation=None,
        use_bias=False,
        padding="same" if stride == 1 else "valid",
        name=prefix + "depthwise",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + "depthwise_BN"
    )(x)

    x = layers.ReLU(6.0, name=prefix + "depthwise_relu")(x)

    # Project
    x = layers.Conv2D(
        pointwise_filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        activation=None,
        name=prefix + "project",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + "project_BN"
    )(x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + "add")([inputs, x])
    return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 내림으로 인해 10% 이상 내려가지 않도록 하십시오.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@keras_export("keras.applications.mobilenet_v2.preprocess_input")
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode="tf")


@keras_export("keras.applications.mobilenet_v2.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode="",
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_TF,
    error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC,
)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__
