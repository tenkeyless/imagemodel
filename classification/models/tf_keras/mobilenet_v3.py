# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=missing-function-docstring
"""
Keras 용 MobileNet v3 모델.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.keras import backend
from tensorflow.python.keras import models
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export


# TODO(scottzhu): Change this to the GCS path.
BASE_WEIGHT_PATH = (
    "https://storage.googleapis.com/tensorflow/" "keras-applications/mobilenet_v3/"
)
WEIGHTS_HASHES = {
    "large_224_0.75_float": (
        "765b44a33ad4005b3ac83185abf1d0eb",
        "e7b4d1071996dd51a2c2ca2424570e20",
    ),
    "large_224_1.0_float": (
        "59e551e166be033d707958cf9e29a6a7",
        "037116398e07f018c0005ffcb0406831",
    ),
    "large_minimalistic_224_1.0_float": (
        "675e7b876c45c57e9e63e6d90a36599c",
        "a2c33aed672524d1d0b4431808177695",
    ),
    "small_224_0.75_float": (
        "cb65d4e5be93758266aa0a7f2c6708b7",
        "4d2fe46f1c1f38057392514b0df1d673",
    ),
    "small_224_1.0_float": (
        "8768d4c2e7dee89b9d02b2d03d65d862",
        "be7100780f875c06bcab93d76641aa26",
    ),
    "small_minimalistic_224_1.0_float": (
        "99cd97fb2fcdad2bf028eb838de69e37",
        "20d4e357df3f7a6361f3a288857b1051",
    ),
}

layers = VersionAwareLayers()


BASE_DOCSTRING = """
{name} 아키텍처를 인스턴스화합니다.

References
----------
- [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf) (ICCV 2019)

다음 표는 MobileNet의 성능을 설명합니다.
------------------------------------------------------------------------
MACs은 Multiply Adds를 나타냅니다.

|Classification Checkpoint|MACs(M)|Parameters(M)|Top1 Accuracy|Pixel1 CPU(ms)|
|---|---|---|---|---|
| mobilenet_v3_large_1.0_224              | 217 | 5.4 |   75.6   |   51.2  |
| mobilenet_v3_large_0.75_224             | 155 | 4.0 |   73.3   |   39.8  |
| mobilenet_v3_large_minimalistic_1.0_224 | 209 | 3.9 |   72.3   |   44.1  |
| mobilenet_v3_small_1.0_224              | 66  | 2.9 |   68.1   |   15.8  |
| mobilenet_v3_small_0.75_224             | 44  | 2.4 |   65.4   |   12.8  |
| mobilenet_v3_small_minimalistic_1.0_224 | 65  | 2.0 |   61.9   |   12.2  |

6개 모델 모두에 대한 가중치는 [여기](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet/README.md)에 있는 TensorFlow 체크포인트로부터 가져와서 변환됩니다.

선택적으로 ImageNet에서 사전 트레이닝된 가중치를 로드합니다.

참고 : 각 Keras 애플리케이션에는 특정 종류의 입력 전처리가 필요합니다.
MobileNetV3 경우, 입력을 모델에 전달하기 전에 입력에 대해,
`tf.keras.applications.mobilenet_v3.preprocess_input`을 호출해야 합니다.

Parameters
----------
input_shape : [type], optional, default=None
    입력 이미지 해상도가 (224, 224, 3)이 아닌 모델을 사용하려는 경우, 지정할 선택적인 shape 튜플입니다.
    (224, 224, 3)같이, 정확히 3개 입력 채널이 있어야 합니다.
    `input_tensor`에서 `input_shape`를 추론하려는 경우, 이 옵션을 생략할 수도 있습니다.
    `input_tensor`와 `input_shape`를 모두 포함하도록 선택하면 일치하는 경우 input_shape가 사용되며, 
    shape이 일치하지 않으면 오류가 발생합니다.
    예) `(160, 160, 3)` 유효한 값입니다.
alpha : float, optional, default=1.0
    네트워크의 너비를 제어합니다.
    이것은 MobileNetV3 논문에서 깊이 승수(depth multiplier)라고 알려져 있지만,
    Keras의 `applications.MobileNetV1` 모델과 일관성을 유지하기 위해 이 이름이 유지됩니다.
    - `alpha` < 1.0이면, 각 레이어의 필터 수를 비례적으로 줄입니다.
    - `alpha` > 1.0이면, 각 레이어의 필터 수를 비례적으로 늘립니다.
    - `alpha` = 1이면, 논문으로부터의 기본 필터 수가 각 레이어에 사용됩니다.
minimalistic : bool, optional, default=False
    크고 작은 모델 외에도, 이 모듈에는 소위 minimalistic 모델도 포함되어 있습니다. 
    이러한 모델은 MobilenetV3와 동일한 레이어 별(per-layer) 차원 특성을 갖지만, 
    고급 블록(squeeze-and-excite 유닛, hard-swish 및 5x5 컨볼루션)을 활용하지 않습니다.
    이러한 모델은 CPU에서 덜 효율적이지만, GPU/DSP에서는 훨씬 더 성능이 좋습니다.
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
dropout_rate : float, optional, default=0.2
    마지막 레이어에 대해 드롭할 입력 유닛의 비율입니다.
classifier_activation : str or callable, optional, default="softmax"
    "top" 레이어에서 사용할 활성화 함수입니다. `include_top=True`가 아니면 무시됩니다.
    "top" 레이어의 로짓을 반환하려면, `classifier_activation=None`을 설정하십시오.

Returns
-------
`keras.Model`
    `keras.Model` 인스턴스.

Raises
------
ValueError
    `weights`에 대한 인수가 잘못되었거나, weights='imagenet'일 때,
    입력 shape이 잘못된 경우 또는, alpha, rows가 잘못된 경우
ValueError
    사전 트레이닝된 top 레이어를 사용할 때, `classifier_activation`이 `softmax` 또는 `None`이 아닌 경우
"""


def MobileNetV3(
    stack_fn,
    last_point_ch,
    input_shape=None,
    alpha=1.0,
    model_type="large",
    minimalistic=False,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    classes=1000,
    pooling=None,
    dropout_rate=0.2,
    classifier_activation="softmax",
):
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
            raise ValueError(
                "input_tensor specified: ", input_tensor, "이 keras 텐서가 아닙니다."
            )

    # input_shape가 None이면 input_tensor에서 shape을 추론합니다.
    if input_shape is None and input_tensor is not None:

        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError(
                "input_tensor: ",
                input_tensor,
                "is type: ",
                type(input_tensor),
                "이는 유효한 타입이 아닙니다.",
            )

        if backend.is_keras_tensor(input_tensor):
            if backend.image_data_format() == "channels_first":
                rows = backend.int_shape(input_tensor)[2]
                cols = backend.int_shape(input_tensor)[3]
                input_shape = (3, cols, rows)
            else:
                rows = backend.int_shape(input_tensor)[1]
                cols = backend.int_shape(input_tensor)[2]
                input_shape = (cols, rows, 3)
    # input_shape가 None이고, input_tensor가 None인, 표준 shape을 사용하는 경우
    if input_shape is None and input_tensor is None:
        input_shape = (None, None, 3)

    if backend.image_data_format() == "channels_last":
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]
    if rows and cols and (rows < 32 or cols < 32):
        raise ValueError(
            "Input 크기는 적어도 32x32 이어야 합니다.; 하지만, 다음의 shape을 받았습니다. `input_shape="
            + str(input_shape)
            + "`"
        )
    if weights == "imagenet":
        if (
            not minimalistic
            and alpha not in [0.75, 1.0]
            or minimalistic
            and alpha != 1.0
        ):
            raise ValueError(
                "imagenet 가중치가 로드되는 경우, "
                "alpha는 non minimalistic에 대해 `0.75`, `1.0` 중 하나이어야 하고, "
                "minimalistic에 대해 `1.0` 이어야 합니다."
            )

        if rows != cols or rows != 224:
            logging.warning(
                "`input_shape`가 정의되지 않았거나, 정사각형이 아니거나, `rows`가 224가 아닙니다. "
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

    if minimalistic:
        kernel = 3
        activation = relu
        se_ratio = None
    else:
        kernel = 5
        activation = hard_swish
        se_ratio = 0.25

    x = img_input
    x = layers.Rescaling(1.0 / 255.0)(x)
    x = layers.Conv2D(
        16, kernel_size=3, strides=(2, 2), padding="same", use_bias=False, name="Conv"
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name="Conv/BatchNorm"
    )(x)
    x = activation(x)

    x = stack_fn(x, kernel, activation, se_ratio)

    last_conv_ch = _depth(backend.int_shape(x)[channel_axis] * 6)

    # 너비 승수가 1보다 크면, 출력 채널 수를 늘립니다.
    if alpha > 1.0:
        last_point_ch = _depth(last_point_ch * alpha)
    x = layers.Conv2D(
        last_conv_ch, kernel_size=1, padding="same", use_bias=False, name="Conv_1"
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name="Conv_1/BatchNorm"
    )(x)
    x = activation(x)
    x = layers.Conv2D(
        last_point_ch, kernel_size=1, padding="same", use_bias=True, name="Conv_2"
    )(x)
    x = activation(x)

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        if channel_axis == 1:
            x = layers.Reshape((last_point_ch, 1, 1))(x)
        else:
            x = layers.Reshape((1, 1, last_point_ch))(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv2D(classes, kernel_size=1, padding="same", name="Logits")(x)
        x = layers.Flatten()(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Activation(activation=classifier_activation, name="Predictions")(x)
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
    model = models.Model(inputs, x, name="MobilenetV3" + model_type)

    # 가중치 불러오기
    if weights == "imagenet":
        model_name = "{}{}_224_{}_float".format(
            model_type, "_minimalistic" if minimalistic else "", str(alpha)
        )
        if include_top:
            file_name = "weights_mobilenet_v3_" + model_name + ".h5"
            file_hash = WEIGHTS_HASHES[model_name][0]
        else:
            file_name = "weights_mobilenet_v3_" + model_name + "_no_top.h5"
            file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = data_utils.get_file(
            file_name,
            BASE_WEIGHT_PATH + file_name,
            cache_subdir="models",
            file_hash=file_hash,
        )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


@keras_export("keras.applications.MobileNetV3Small")
def MobileNetV3Small(
    input_shape=None,
    alpha=1.0,
    minimalistic=False,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    classes=1000,
    pooling=None,
    dropout_rate=0.2,
    classifier_activation="softmax",
):
    def stack_fn(x, kernel, activation, se_ratio):
        def depth(d):
            return _depth(d * alpha)

        x = _inverted_res_block(x, 1, depth(16), 3, 2, se_ratio, relu, 0)
        x = _inverted_res_block(x, 72.0 / 16, depth(24), 3, 2, None, relu, 1)
        x = _inverted_res_block(x, 88.0 / 24, depth(24), 3, 1, None, relu, 2)
        x = _inverted_res_block(x, 4, depth(40), kernel, 2, se_ratio, activation, 3)
        x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 4)
        x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 5)
        x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 6)
        x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 7)
        x = _inverted_res_block(x, 6, depth(96), kernel, 2, se_ratio, activation, 8)
        x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 9)
        x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 10)
        return x

    return MobileNetV3(
        stack_fn,
        1024,
        input_shape,
        alpha,
        "small",
        minimalistic,
        include_top,
        weights,
        input_tensor,
        classes,
        pooling,
        dropout_rate,
        classifier_activation,
    )


@keras_export("keras.applications.MobileNetV3Large")
def MobileNetV3Large(
    input_shape=None,
    alpha=1.0,
    minimalistic=False,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    classes=1000,
    pooling=None,
    dropout_rate=0.2,
    classifier_activation="softmax",
):
    def stack_fn(x, kernel, activation, se_ratio):
        def depth(d):
            return _depth(d * alpha)

        x = _inverted_res_block(x, 1, depth(16), 3, 1, None, relu, 0)
        x = _inverted_res_block(x, 4, depth(24), 3, 2, None, relu, 1)
        x = _inverted_res_block(x, 3, depth(24), 3, 1, None, relu, 2)
        x = _inverted_res_block(x, 3, depth(40), kernel, 2, se_ratio, relu, 3)
        x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 4)
        x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 5)
        x = _inverted_res_block(x, 6, depth(80), 3, 2, None, activation, 6)
        x = _inverted_res_block(x, 2.5, depth(80), 3, 1, None, activation, 7)
        x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 8)
        x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 9)
        x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 10)
        x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 11)
        x = _inverted_res_block(x, 6, depth(160), kernel, 2, se_ratio, activation, 12)
        x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 13)
        x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 14)
        return x

    return MobileNetV3(
        stack_fn,
        1280,
        input_shape,
        alpha,
        "large",
        minimalistic,
        include_top,
        weights,
        input_tensor,
        classes,
        pooling,
        dropout_rate,
        classifier_activation,
    )


MobileNetV3Small.__doc__ = BASE_DOCSTRING.format(name="MobileNetV3Small")
MobileNetV3Large.__doc__ = BASE_DOCSTRING.format(name="MobileNetV3Large")


def relu(x):
    return layers.ReLU()(x)


def hard_sigmoid(x):
    return layers.ReLU(6.0)(x + 3.0) * (1.0 / 6.0)


def hard_swish(x):
    return layers.Multiply()([hard_sigmoid(x), x])


# 이 함수는 원본 tf 저장소에서 가져왔습니다.
# 이는 모든 레이어가 8로 나눌 수 있는 채널 숫자를 갖도록 합니다.
# 여기에서 볼 수 있습니다.
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py


def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _se_block(inputs, filters, se_ratio, prefix):
    x = layers.GlobalAveragePooling2D(name=prefix + "squeeze_excite/AvgPool")(inputs)
    if backend.image_data_format() == "channels_first":
        x = layers.Reshape((filters, 1, 1))(x)
    else:
        x = layers.Reshape((1, 1, filters))(x)
    x = layers.Conv2D(
        _depth(filters * se_ratio),
        kernel_size=1,
        padding="same",
        name=prefix + "squeeze_excite/Conv",
    )(x)
    x = layers.ReLU(name=prefix + "squeeze_excite/Relu")(x)
    x = layers.Conv2D(
        filters, kernel_size=1, padding="same", name=prefix + "squeeze_excite/Conv_1"
    )(x)
    x = hard_sigmoid(x)
    x = layers.Multiply(name=prefix + "squeeze_excite/Mul")([inputs, x])
    return x


def _inverted_res_block(
    x, expansion, filters, kernel_size, stride, se_ratio, activation, block_id
):
    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
    shortcut = x
    prefix = "expanded_conv/"
    infilters = backend.int_shape(x)[channel_axis]
    if block_id:
        # Expand
        prefix = "expanded_conv_{}/".format(block_id)
        x = layers.Conv2D(
            _depth(infilters * expansion),
            kernel_size=1,
            padding="same",
            use_bias=False,
            name=prefix + "expand",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + "expand/BatchNorm",
        )(x)
        x = activation(x)

    if stride == 2:
        x = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, kernel_size),
            name=prefix + "depthwise/pad",
        )(x)
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding="same" if stride == 1 else "valid",
        use_bias=False,
        name=prefix + "depthwise",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "depthwise/BatchNorm",
    )(x)
    x = activation(x)

    if se_ratio:
        x = _se_block(x, _depth(infilters * expansion), se_ratio, prefix)

    x = layers.Conv2D(
        filters, kernel_size=1, padding="same", use_bias=False, name=prefix + "project"
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "project/BatchNorm",
    )(x)

    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + "Add")([shortcut, x])
    return x


@keras_export("keras.applications.mobilenet_v3.preprocess_input")
def preprocess_input(x, data_format=None):  # pylint: disable=unused-argument
    return x


@keras_export("keras.applications.mobilenet_v3.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode="",
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_TF,
    error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC,
)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__