# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
Keras 용 Xception V1 모델.

ImageNet에 대해, 이 모델은 0.790의 top-1 검증 정확도와 0.945의 top-5 검증 정확도를 얻습니다.

참조:
- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) (CVPR 2017)

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
from tensorflow.python.util.tf_export import keras_export


TF_WEIGHTS_PATH = (
    "https://storage.googleapis.com/tensorflow/keras-applications/"
    "xception/xception_weights_tf_dim_ordering_tf_kernels.h5"
)
TF_WEIGHTS_PATH_NO_TOP = (
    "https://storage.googleapis.com/tensorflow/keras-applications/"
    "xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
)

layers = VersionAwareLayers()


@keras_export("keras.applications.xception.Xception", "keras.applications.Xception")
def Xception(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    """
    Xception 아키텍쳐를 인스턴스화합니다.

    기본값으로, ImageNet에 대해 사전 트레이닝된 가중치를 로드합니다.
    모델에서 사용하는 데이터 형식 규칙은 Keras 구성 `~/.keras/keras.json`에 지정된 규칙입니다.
    이 모델의 기본 입력 이미지 크기는 299x299입니다.

    참고 : 각 Keras 애플리케이션에는 특정 종류의 입력 전처리가 필요합니다.
    Xception 경우, 입력을 모델에 전달하기 전에 입력에 대해,
    `tf.keras.applications.xception.preprocess_input`을 호출해야 합니다.

    References
    ----------
    - [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) (CVPR 2017)

    Parameters
    ----------
    include_top : bool, optional, default=True
        네트워크 상단에 있는 완전 연결 레이어를 포함할지 여부.
    weights : str, optional, default="imagenet"
        `None`(무작위 초기화), 'imagenet' (ImageNet에 대해 사전 트레이닝) 중 하나 또는 로드할 가중치 파일의 경로입니다.
    input_tensor : [type], optional, default=None
        모델의 이미지 입력으로 사용할 선택적 Keras 텐서(즉, `layers.Input()`의 출력).
    input_shape : [type], optional, default=None
        선택적 shape 튜플, `include_top`이 `False`인 경우에만 지정됩니다.
        (그렇지 않으면 입력 shape은 `(299, 299, 3)`)
        정확히 3개 입력 채널이 있어야 합니다. 그리고 너비와 높이는 71보다 커야합니다.
        예) `(150, 150, 3)` 유효한 값입니다.
    pooling : [type], optional, default=None
        `include_top`이 `False` 인 경우, 특성 추출을 위한 선택적 풀링 모드
        - `None` 모델의 출력이 마지막 컨볼루션 블록의 4D 텐서 출력이 됨을 의미합니다.
        - `avg` 글로벌 평균 풀링이 마지막 컨볼루션 블록의 출력에 적용됨을 의미합니다. 따라서, 모델의 출력은 2D 텐서가 됩니다.
        - `max` 글로벌 최대 풀링이 적용됨을 의미합니다.
    classes : int, optional, default=1000
        이미지를 분류할 클래스 수 (선택 사항). `include_top`이 `True`이고, `weights` 인수가 지정되지 않은 경우에만 지정됩니다.
    classifier_activation : str, optional, default="softmax"
        "top" 레이어에서 사용할 활성화 함수입니다.
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
        default_size=299,
        min_size=71,
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

    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1

    x = layers.Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name="block1_conv1")(
        img_input
    )
    x = layers.BatchNormalization(axis=channel_axis, name="block1_conv1_bn")(x)
    x = layers.Activation("relu", name="block1_conv1_act")(x)
    x = layers.Conv2D(64, (3, 3), use_bias=False, name="block1_conv2")(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block1_conv2_bn")(x)
    x = layers.Activation("relu", name="block1_conv2_act")(x)

    residual = layers.Conv2D(
        128, (1, 1), strides=(2, 2), padding="same", use_bias=False
    )(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.SeparableConv2D(
        128, (3, 3), padding="same", use_bias=False, name="block2_sepconv1"
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block2_sepconv1_bn")(x)
    x = layers.Activation("relu", name="block2_sepconv2_act")(x)
    x = layers.SeparableConv2D(
        128, (3, 3), padding="same", use_bias=False, name="block2_sepconv2"
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block2_sepconv2_bn")(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="block2_pool")(
        x
    )
    x = layers.add([x, residual])

    residual = layers.Conv2D(
        256, (1, 1), strides=(2, 2), padding="same", use_bias=False
    )(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation("relu", name="block3_sepconv1_act")(x)
    x = layers.SeparableConv2D(
        256, (3, 3), padding="same", use_bias=False, name="block3_sepconv1"
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block3_sepconv1_bn")(x)
    x = layers.Activation("relu", name="block3_sepconv2_act")(x)
    x = layers.SeparableConv2D(
        256, (3, 3), padding="same", use_bias=False, name="block3_sepconv2"
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block3_sepconv2_bn")(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="block3_pool")(
        x
    )
    x = layers.add([x, residual])

    residual = layers.Conv2D(
        728, (1, 1), strides=(2, 2), padding="same", use_bias=False
    )(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation("relu", name="block4_sepconv1_act")(x)
    x = layers.SeparableConv2D(
        728, (3, 3), padding="same", use_bias=False, name="block4_sepconv1"
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block4_sepconv1_bn")(x)
    x = layers.Activation("relu", name="block4_sepconv2_act")(x)
    x = layers.SeparableConv2D(
        728, (3, 3), padding="same", use_bias=False, name="block4_sepconv2"
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block4_sepconv2_bn")(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="block4_pool")(
        x
    )
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = "block" + str(i + 5)

        x = layers.Activation("relu", name=prefix + "_sepconv1_act")(x)
        x = layers.SeparableConv2D(
            728, (3, 3), padding="same", use_bias=False, name=prefix + "_sepconv1"
        )(x)
        x = layers.BatchNormalization(axis=channel_axis, name=prefix + "_sepconv1_bn")(
            x
        )
        x = layers.Activation("relu", name=prefix + "_sepconv2_act")(x)
        x = layers.SeparableConv2D(
            728, (3, 3), padding="same", use_bias=False, name=prefix + "_sepconv2"
        )(x)
        x = layers.BatchNormalization(axis=channel_axis, name=prefix + "_sepconv2_bn")(
            x
        )
        x = layers.Activation("relu", name=prefix + "_sepconv3_act")(x)
        x = layers.SeparableConv2D(
            728, (3, 3), padding="same", use_bias=False, name=prefix + "_sepconv3"
        )(x)
        x = layers.BatchNormalization(axis=channel_axis, name=prefix + "_sepconv3_bn")(
            x
        )

        x = layers.add([x, residual])

    residual = layers.Conv2D(
        1024, (1, 1), strides=(2, 2), padding="same", use_bias=False
    )(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation("relu", name="block13_sepconv1_act")(x)
    x = layers.SeparableConv2D(
        728, (3, 3), padding="same", use_bias=False, name="block13_sepconv1"
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block13_sepconv1_bn")(x)
    x = layers.Activation("relu", name="block13_sepconv2_act")(x)
    x = layers.SeparableConv2D(
        1024, (3, 3), padding="same", use_bias=False, name="block13_sepconv2"
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block13_sepconv2_bn")(x)

    x = layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding="same", name="block13_pool"
    )(x)
    x = layers.add([x, residual])

    x = layers.SeparableConv2D(
        1536, (3, 3), padding="same", use_bias=False, name="block14_sepconv1"
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block14_sepconv1_bn")(x)
    x = layers.Activation("relu", name="block14_sepconv1_act")(x)

    x = layers.SeparableConv2D(
        2048, (3, 3), padding="same", use_bias=False, name="block14_sepconv2"
    )(x)
    x = layers.BatchNormalization(axis=channel_axis, name="block14_sepconv2_bn")(x)
    x = layers.Activation("relu", name="block14_sepconv2_act")(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
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
    model = training.Model(inputs, x, name="xception")

    # 가중치 불러오기
    if weights == "imagenet":
        if include_top:
            weights_path = data_utils.get_file(
                "xception_weights_tf_dim_ordering_tf_kernels.h5",
                TF_WEIGHTS_PATH,
                cache_subdir="models",
                file_hash="0a58e3b7378bc2990ea3b43d5981f1f6",
            )
        else:
            weights_path = data_utils.get_file(
                "xception_weights_tf_dim_ordering_tf_kernels_notop.h5",
                TF_WEIGHTS_PATH_NO_TOP,
                cache_subdir="models",
                file_hash="b0042744bf5b25fce3cb969f33bebb97",
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


@keras_export("keras.applications.xception.preprocess_input")
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode="tf")


@keras_export("keras.applications.xception.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode="",
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_TF,
    error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC,
)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__