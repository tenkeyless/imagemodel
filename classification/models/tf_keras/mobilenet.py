# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
Keras 용 MobileNet v1 모델.

MobileNet은 일반적인 아키텍처이며 여러 사용 사례에 사용할 수 있습니다.
사용 사례에 따라, 다른 입력 레이어 크기와 다른 너비 factor를 사용할 수 있습니다.
이를 통해 서로 다른 너비 모델이 multiply-adds 횟수를 줄여, 모바일 장치에 대해 추론 비용을 줄일 수 있습니다.

MobileNets는 32 x 32보다 큰 모든 입력 크기를 지원하며, 더 큰 이미지 크기는 더 나은 성능을 제공합니다.
매개 변수 수와 multiply-adds 수는, 각 레이어의 필터 수를 증가/감소시키는, `alpha` 매개변수를 사용하여 수정할 수 있습니다.
이미지 크기와 `alpha` 매개변수를 변경하여, ImageNet 가중치를 제공되는, 논문의 16개 모델을 모두 제작할 수 있습니다.

이 논문은 1.0(100 % MobileNet이라고도 함), 0.75, 0.5 및 0.25의 `alpha`값을 사용하여, MobileNet의 성능을 보여줍니다.
이러한 `alpha` 값 각각에 대해, 4가지 입력 이미지 크기(224, 192, 160, 128)에 대한 가중치가 제공됩니다.

다음 표는 크기 224 x 224에서 100 % MobileNet의 크기와 정확도를 설명합니다.
----------------------------------------------------------------------------
Width Multiplier (alpha) | ImageNet Acc |  Multiply-Adds (M) |  Params (M)
----------------------------------------------------------------------------
|   1.0 MobileNet-224    |    70.6 %     |        529        |     4.2     |
|   0.75 MobileNet-224   |    68.4 %     |        325        |     2.6     |
|   0.50 MobileNet-224   |    63.7 %     |        149        |     1.3     |
|   0.25 MobileNet-224   |    50.6 %     |        41         |     0.5     |
----------------------------------------------------------------------------

다음 표는 다양한 입력 크기에 대해 100 % MobileNet의 성능을 설명합니다.
------------------------------------------------------------------------
|     Resolution      | ImageNet Acc | Multiply-Adds (M) | Params (M)  |
------------------------------------------------------------------------
|  1.0 MobileNet-224  |    70.6 %    |        529        |     4.2     |
|  1.0 MobileNet-192  |    69.1 %    |        529        |     4.2     |
|  1.0 MobileNet-160  |    67.2 %    |        529        |     4.2     |
|  1.0 MobileNet-128  |    64.4 %    |        529        |     4.2     |
------------------------------------------------------------------------

참조:
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export

BASE_WEIGHT_PATH = (
    "https://storage.googleapis.com/tensorflow/" "keras-applications/mobilenet/"
)
layers = None


@keras_export("keras.applications.mobilenet.MobileNet", "keras.applications.MobileNet")
def MobileNet(
    input_shape=None,
    alpha=1.0,
    depth_multiplier=1,
    dropout=1e-3,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs
):
    """
    MobileNet 아키텍처를 인스턴스화합니다.

    선택적으로 ImageNet에서 사전 트레이닝된 가중치를 로드합니다.
    모델에서 사용하는 데이터 형식 규칙은 `tf.keras.backend.image_data_format()`에 지정된 규칙입니다.

    참고 : 각 Keras 애플리케이션에는 특정 종류의 입력 전처리가 필요합니다.
    MobileNet의 경우, 입력을 모델에 전달하기 전에 입력에 대해,
    `tf.keras.applications.mobilenet.preprocess_input`을 호출해야 합니다.

    References
    ----------
    - [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

    Parameters
    ----------
    input_shape : [type], optional, default=None
        선택적 shape 튜플, `include_top`이 `False`인 경우에만 지정됩니다.
        (그렇지 않으면 입력 shape은 `(224, 224, 3)` (`'channels_last'` 데이터 형식을 사용하는 경우) 또는
        `(3, 224, 224)` (`'channels_first'` 데이터 형식을 사용하는 경우)이어야 합니다.)
        정확히 3개 입력 채널이 있어야 합니다. 그리고 너비와 높이는 32보다 커야합니다.
        예) `(200, 200, 3)` 유효한 값입니다.
        `input_tensor`가 제공되면, `input_shape`는 무시됩니다.
    alpha : float, optional, default=1.0
        네트워크의 너비를 제어합니다.
        이것은 MobileNet 논문에서 너비 승수(width multiplier)라고 알려져 있습니다.
        - `alpha` < 1.0이면, 각 레이어의 필터 수를 비례적으로 줄입니다.
        - `alpha` > 1.0이면, 각 레이어의 필터 수를 비례적으로 늘립니다.
        - `alpha` = 1이면, 논문으로부터의 기본 필터 수가 각 레이어에 사용됩니다.
    depth_multiplier : int, optional, default=1.0
        depthwise 컨볼루션을 위한 깊이 승수(Depth multiplier).
        이를 MobileNet 논문에서는 해상도 승수(resolution multiplier)라고 합니다.
    dropout : float, optional, default=0.001
        Dropout 비율
    include_top : bool, optional, default=True
        네트워크 상단에 있는 완전 연결 레이어를 포함할지 여부
    weights : str, optional, default="imagenet"
        `None`(무작위 초기화), `'imagenet'`(ImageNet에 대해 사전 트레이닝된) 또는 로드할 가중치 파일의 경로 중 하나입니다.
    input_tensor : [type], optional, default=None
        모델의 이미지 입력으로 사용할 Optional Keras 텐서 (즉,`layers.Input()`의 출력)
        `input_tensor`는 여러 다른 네트워크간에 입력을 공유하는 데 유용합니다.
    pooling : str, optional, default=None
        `include_top`이 `False`인 경우, 특성 추출을 위한 선택적 풀링 모드입니다.
        - `None`은 모델의 출력이 마지막 컨볼루션 블록의 4D 텐서 출력이 될 것임을 의미합니다.
        - `avg`는 글로벌 평균 풀링이 마지막 컨볼루션 블록의 출력에 적용되므로, 모델의 출력이 2D텐서가 될 것임을 의미합니다.
        - 'max'는 글로벌 최대 풀링이 적용됨을 의미합니다.
    classes : int, optional, default=1000
        이미지를 분류할 클래스 수. `include_top`이 `True`이고, `weights` 인수가 지정되지 않은 경우에만, 지정합니다.
    classifier_activation : str or callable, optional, default="softmax"
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
        `weights`에 대한 인수가 잘못되었거나, 입력 shape이 잘못된 경우
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
    if input_shape is None:
        default_size = 224
    else:
        if backend.image_data_format() == "channels_first":
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [128, 160, 192, 224]:
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
        if depth_multiplier != 1:
            raise ValueError(
                "imagenet 가중치를 로드하는 경우, 깊이 승수(depth multiplier)는 1이어야 합니다."
            )

        if alpha not in [0.25, 0.50, 0.75, 1.0]:
            raise ValueError(
                "imagenet 가중치가 로드되는 경우, "
                "alpha는 `0.25`, `0.50`, `0.75` 또는 `1.0`중 하나만 될 수 있습니다."
            )

        if rows != cols or rows not in [128, 160, 192, 224]:
            rows = 224
            logging.warning(
                "`input_shape`가 정의되지 않았거나, 정사각형이 아니거나, `rows`가 [128, 160, 192, 224] 중 하나가 아닙니다. "
                "입력 shape (224, 224)에 대한 가중치가 기본값으로 로드됩니다."
            )

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

    x = _depthwise_conv_block(
        x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2
    )
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

    x = _depthwise_conv_block(
        x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4
    )
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6
    )
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

    x = _depthwise_conv_block(
        x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12
    )
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

    if include_top:
        if backend.image_data_format() == "channels_first":
            shape = (int(1024 * alpha), 1, 1)
        else:
            shape = (1, 1, int(1024 * alpha))

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Reshape(shape, name="reshape_1")(x)
        x = layers.Dropout(dropout, name="dropout")(x)
        x = layers.Conv2D(classes, (1, 1), padding="same", name="conv_preds")(x)
        x = layers.Reshape((classes,), name="reshape_2")(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Activation(activation=classifier_activation, name="predictions")(x)
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
    model = training.Model(inputs, x, name="mobilenet_%0.2f_%s" % (alpha, rows))

    # 가중치 불러오기
    if weights == "imagenet":
        if alpha == 1.0:
            alpha_text = "1_0"
        elif alpha == 0.75:
            alpha_text = "7_5"
        elif alpha == 0.50:
            alpha_text = "5_0"
        else:
            alpha_text = "2_5"

        if include_top:
            model_name = "mobilenet_%s_%d_tf.h5" % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = data_utils.get_file(
                model_name, weight_path, cache_subdir="models"
            )
        else:
            model_name = "mobilenet_%s_%d_tf_no_top.h5" % (alpha_text, rows)
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = data_utils.get_file(
                model_name, weight_path, cache_subdir="models"
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    """
    초기 컨볼루션 레이어를 추가합니다. (배치 정규화 및 relu6 사용)

    Parameters
    ----------
    inputs : [type]
        `(rows, cols, 3)` (`channels_last` 데이터 형식 사용) 또는
        `(3, rows, cols)` (`channels_first` 데이터 형식 사용) shape의 입력 텐서
        정확히 3개의 입력 채널이 있어야 하며, 너비와 높이는 32보다 작아서는 안됩니다. 예) `(224, 224, 3)`는 하나의 유효한 값입니다.
    filters : int
        출력 공간의 차원(즉, 컨볼루션의 출력 필터 수)
    alpha : float
        네트워크의 너비를 컨트롤합니다.
        - `alpha` < 1.0이면, 각 레이어의 필터 수를 비례적으로 줄입니다.
        - `alpha` > 1.0이면, 각 레이어의 필터 수를 비례적으로 늘립니다.
        - `alpha` = 1이면, 논문으로부터의 기본 필터 수가 각 레이어에 사용됩니다.
    kernel : tuple, optional, default=(3, 3)
        2D 컨볼루션 윈도우의 너비와 높이를 지정하는, 정수 또는 2개 정수의 튜플/리스트입니다.
        모든 공간 차원에 대해 동일한 값을 지정하는 단일 정수일 수 있습니다.
    strides : tuple, optional, default=(1, 1)
        너비와 높이를 따라 컨볼루션의 스트라이드를 지정하는, 정수 또는 2개 정수의 튜플/리스트입니다.
        모든 공간 차원에 대해 동일한 값을 지정하는 단일 정수일 수 있습니다.
        stride value != 1을 지정하는 것은 '`dilation_rate` value != 1을 지정하는 것과 호환되지 않습니다.

        # Input shape
        4D tensor with shape: data_format='channels_first'인 경우, `(samples, channels, rows, cols)`
        또는 4D tensor with shape: data_format='channels_last'인 경우, `(samples, rows, cols, channels)`.

        # Output shape
        4D tensor with shape: data_format='channels_first'인 경우, `(samples, filters, new_rows, new_cols)`
        또는 4D tensor with shape: data_format='channels_last'인 경우, `(samples, new_rows, new_cols, filters)`.
        `rows` 및 `cols` 값은 stride로 인해 변경되었을 수 있습니다.

    Returns
    -------
    [type]
        블록의 출력 텐서
    """
    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
    filters = int(filters * alpha)
    x = layers.Conv2D(
        filters, kernel, padding="same", use_bias=False, strides=strides, name="conv1"
    )(inputs)
    x = layers.BatchNormalization(axis=channel_axis, name="conv1_bn")(x)
    return layers.ReLU(6.0, name="conv1_relu")(x)


def _depthwise_conv_block(
    inputs,
    pointwise_conv_filters,
    alpha,
    depth_multiplier=1,
    strides=(1, 1),
    block_id=1,
):
    """
    Depthwise 컨볼루션 블록을 추가합니다.

    Depthwise 컨볼루션 블록은 depthwise conv, 배치 정규화, relu6, pointwise 컨볼루션, 배치 정규화 및 relu6 활성화로 구성됩니다.

    Parameters
    ----------
    inputs : [type]
        `(rows, cols, channels)` (`channels_last` 데이터 형식 사용) 또는
        `(channels, rows, cols)` (`channels_first` 데이터 형식 사용) shape의 입력 텐서
    pointwise_conv_filters : int
        출력 공간의 차원. (즉, pointwise 컨볼루션의 출력 필터 수)
    alpha : [type]
        네트워크의 너비를 컨트롤합니다.
        - `alpha` < 1.0이면, 각 레이어의 필터 수를 비례적으로 줄입니다.
        - `alpha` > 1.0이면, 각 레이어의 필터 수를 비례적으로 늘립니다.
        - `alpha` = 1이면, 논문으로부터의 기본 필터 수가 각 레이어에 사용됩니다.
    depth_multiplier : int, optional, default=1
        각 입력 채널에 대한 depthwise 컨볼루션 출력 채널 수입니다.
        depthwise 컨볼루션 출력 채널의 총 수는 `filters_in * depth_multiplier`와 같습니다.
    strides : tuple, optional, default=(1, 1)
        너비와 높이를 따라 컨볼루션의 스트라이드를 지정하는, 정수 또는 2개 정수의 튜플/리스트입니다.
        모든 공간 차원에 대해 동일한 값을 지정하는 단일 정수일 수 있습니다.
        stride value != 1을 지정하는 것은 '`dilation_rate` value != 1을 지정하는 것과 호환되지 않습니다.
    block_id : int, optional, default=1
        블록 숫자를 지정하는 고유 ID.

        # Input shape
        4D tensor with shape: data_format='channels_first'인 경우, `(batch, channels, rows, cols)`
        또는 4D tensor with shape: data_format='channels_last'인 경우, `(batch, rows, cols, channels)`.

        # Output shape
        4D tensor with shape: data_format='channels_first'인 경우, `(batch, filters, new_rows, new_cols)`
        또는 4D tensor with shape: data_format='channels_last'인 경우, `(batch, new_rows, new_cols, filters)`.
        `rows` 및 `cols` 값은 stride로 인해 변경되었을 수 있습니다.

    Returns
    -------
    [type]
        블록의 출력 텐서
    """
    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(((0, 1), (0, 1)), name="conv_pad_%d" % block_id)(
            inputs
        )
    x = layers.DepthwiseConv2D(
        (3, 3),
        padding="same" if strides == (1, 1) else "valid",
        depth_multiplier=depth_multiplier,
        strides=strides,
        use_bias=False,
        name="conv_dw_%d" % block_id,
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="conv_dw_%d_bn" % block_id)(x)
    x = layers.ReLU(6.0, name="conv_dw_%d_relu" % block_id)(x)

    x = layers.Conv2D(
        pointwise_conv_filters,
        (1, 1),
        padding="same",
        use_bias=False,
        strides=(1, 1),
        name="conv_pw_%d" % block_id,
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="conv_pw_%d_bn" % block_id)(x)
    return layers.ReLU(6.0, name="conv_pw_%d_relu" % block_id)(x)


@keras_export("keras.applications.mobilenet.preprocess_input")
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode="tf")


@keras_export("keras.applications.mobilenet.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode="",
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_TF,
    error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC,
)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__