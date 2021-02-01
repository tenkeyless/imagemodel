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
Keras 용 DenseNet 모델.

참조:
- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (CVPR 2017)
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


BASE_WEIGTHS_PATH = (
    "https://storage.googleapis.com/tensorflow/" "keras-applications/densenet/"
)
DENSENET121_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH + "densenet121_weights_tf_dim_ordering_tf_kernels.h5"
)
DENSENET121_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH + "densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5"
)
DENSENET169_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH + "densenet169_weights_tf_dim_ordering_tf_kernels.h5"
)
DENSENET169_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH + "densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5"
)
DENSENET201_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH + "densenet201_weights_tf_dim_ordering_tf_kernels.h5"
)
DENSENET201_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH + "densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5"
)

layers = VersionAwareLayers()


def dense_block(x, blocks, name):
    """
    dense 블록

    Parameters
    ----------
    x : [type]
        입력 텐서
    blocks : int
        빌딩 블록의 수.
    name : str
        블록 라벨

    Returns
    -------
    [type]
        블록에 대한 출력 텐서
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + "_block" + str(i + 1))
    return x


def transition_block(x, reduction: float, name: str):
    """
    transition 블록

    Parameters
    ----------
    x : [type]
        입력 텐서
    reduction : float
        transition 레이어에서 compression 비율.
    name : str
        블록 라벨

    Returns
    -------
    [type]
        블록에 대한 출력 텐서
    """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_bn")(x)
    x = layers.Activation("relu", name=name + "_relu")(x)
    x = layers.Conv2D(
        int(backend.int_shape(x)[bn_axis] * reduction),
        1,
        use_bias=False,
        name=name + "_conv",
    )(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + "_pool")(x)
    return x


def conv_block(x, growth_rate: float, name: str):
    """
    dense 블록에 대한 빌딩 블록

    Parameters
    ----------
    x : [type]
        입력 텐서
    growth_rate : float
        dense 레이어에서 growth 비율
    name : str
        블록 라벨

    Returns
    -------
    [type]
        블록에 대한 출력 텐서
    """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_0_bn")(
        x
    )
    x1 = layers.Activation("relu", name=name + "_0_relu")(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1, use_bias=False, name=name + "_1_conv")(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn")(
        x1
    )
    x1 = layers.Activation("relu", name=name + "_1_relu")(x1)
    x1 = layers.Conv2D(
        growth_rate, 3, padding="same", use_bias=False, name=name + "_2_conv"
    )(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + "_concat")([x, x1])
    return x


def DenseNet(
    blocks,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    """
    DenseNet 아키텍처를 인스턴스화합니다.

    선택적으로 ImageNet에서 사전 트레이닝된 가중치를 로드합니다.
    모델에서 사용하는 데이터 형식 규칙은 Keras 구성 `~/.keras/keras.json`에 지정된 규칙입니다.

    참고 : 각 Keras 애플리케이션에는 특정 종류의 입력 전처리가 필요합니다.
    DenseNet 경우, 입력을 모델에 전달하기 전에 입력에 대해,
    `tf.keras.applications.densenet.preprocess_input`을 호출해야 합니다.

    Reference
    ---------
    - [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (CVPR 2017)

    Parameters
    ----------
    blocks : [type]
        4개 Dense 레이어를 위한 빌딩 블록의 수.
    include_top : bool, optional, default=True
        네트워크 top에 완전 연결 레이어를 포함할지 여부.
    weights : str, optional, default="imagenet"
        `None`(무작위 초기화), 'imagenet' (ImageNet에 대해 사전 트레이닝) 중 하나 또는 로드할 가중치 파일의 경로입니다.
    input_tensor : [type], optional, default=None
        모델의 이미지 입력으로 사용할 선택적 Keras 텐서(즉, `layers.Input()`의 출력).
    input_shape : [type], optional, default=None
        선택적 shape 튜플, `include_top`이 `False`인 경우에만 지정됩니다.
        (그렇지 않으면 입력 shape은 `(224, 224, 3)` (`'channels_last'` 데이터 형식을 사용하는 경우) 또는
        `(3, 224, 224)` (`'channels_first'` 데이터 형식을 사용하는 경우)이어야 합니다.)
        정확히 3개 입력 채널이 있어야 합니다. 그리고 너비와 높이는 32보다 커야합니다.
        예) `(200, 200, 3)` 유효한 값입니다.
    pooling : [type], optional, default=None
        `include_top`이 `False` 인 경우, 특성 추출을 위한 선택적 풀링 모드
        - `None` 모델의 출력이 마지막 컨볼루션 레이어의 4D 텐서 출력이 됨을 의미합니다.
        - `avg` 글로벌 평균 풀링이 마지막 컨볼루션 레이어의 출력에 적용됨을 의미합니다. 따라서, 모델의 출력은 2D 텐서가 됩니다.
        - `max` 글로벌 최대 풀링이 적용됨을 의미합니다.
    classes : int, optional, default=1000
        이미지를 분류할 클래스 수 (선택 사항). `include_top`이  `True`이고, `weights` 인수가 지정되지 않은 경우에만 지정됩니다.
    classifier_activation : str or callable, optional, default="softmax"
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
        default_size=224,
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

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name="conv1/conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="conv1/bn")(x)
    x = layers.Activation("relu", name="conv1/relu")(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool1")(x)

    x = dense_block(x, blocks[0], name="conv2")
    x = transition_block(x, 0.5, name="pool2")
    x = dense_block(x, blocks[1], name="conv3")
    x = transition_block(x, 0.5, name="pool3")
    x = dense_block(x, blocks[2], name="conv4")
    x = transition_block(x, 0.5, name="pool4")
    x = dense_block(x, blocks[3], name="conv5")

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="bn")(x)
    x = layers.Activation("relu", name="relu")(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)

        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(classes, activation=classifier_activation, name="predictions")(
            x
        )
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
    if blocks == [6, 12, 24, 16]:
        model = training.Model(inputs, x, name="densenet121")
    elif blocks == [6, 12, 32, 32]:
        model = training.Model(inputs, x, name="densenet169")
    elif blocks == [6, 12, 48, 32]:
        model = training.Model(inputs, x, name="densenet201")
    else:
        model = training.Model(inputs, x, name="densenet")

    # 가중치 불러오기
    if weights == "imagenet":
        if include_top:
            if blocks == [6, 12, 24, 16]:
                weights_path = data_utils.get_file(
                    "densenet121_weights_tf_dim_ordering_tf_kernels.h5",
                    DENSENET121_WEIGHT_PATH,
                    cache_subdir="models",
                    file_hash="9d60b8095a5708f2dcce2bca79d332c7",
                )
            elif blocks == [6, 12, 32, 32]:
                weights_path = data_utils.get_file(
                    "densenet169_weights_tf_dim_ordering_tf_kernels.h5",
                    DENSENET169_WEIGHT_PATH,
                    cache_subdir="models",
                    file_hash="d699b8f76981ab1b30698df4c175e90b",
                )
            elif blocks == [6, 12, 48, 32]:
                weights_path = data_utils.get_file(
                    "densenet201_weights_tf_dim_ordering_tf_kernels.h5",
                    DENSENET201_WEIGHT_PATH,
                    cache_subdir="models",
                    file_hash="1ceb130c1ea1b78c3bf6114dbdfd8807",
                )
        else:
            if blocks == [6, 12, 24, 16]:
                weights_path = data_utils.get_file(
                    "densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5",
                    DENSENET121_WEIGHT_PATH_NO_TOP,
                    cache_subdir="models",
                    file_hash="30ee3e1110167f948a6b9946edeeb738",
                )
            elif blocks == [6, 12, 32, 32]:
                weights_path = data_utils.get_file(
                    "densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5",
                    DENSENET169_WEIGHT_PATH_NO_TOP,
                    cache_subdir="models",
                    file_hash="b8c4d4c20dd625c148057b9ff1c1176b",
                )
            elif blocks == [6, 12, 48, 32]:
                weights_path = data_utils.get_file(
                    "densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5",
                    DENSENET201_WEIGHT_PATH_NO_TOP,
                    cache_subdir="models",
                    file_hash="c13680b51ded0fb44dff2d8f86ac8bb1",
                )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


@keras_export(
    "keras.applications.densenet.DenseNet121", "keras.applications.DenseNet121"
)
def DenseNet121(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
):
    """Densenet121 아키텍쳐를 인스턴스화 합니다."""
    return DenseNet(
        [6, 12, 24, 16],
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
    )


@keras_export(
    "keras.applications.densenet.DenseNet169", "keras.applications.DenseNet169"
)
def DenseNet169(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
):
    """Densenet169 아키텍쳐를 인스턴스화 합니다."""
    return DenseNet(
        [6, 12, 32, 32],
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
    )


@keras_export(
    "keras.applications.densenet.DenseNet201", "keras.applications.DenseNet201"
)
def DenseNet201(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
):
    """Densenet201 아키텍쳐를 인스턴스화 합니다."""
    return DenseNet(
        [6, 12, 48, 32],
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
    )


@keras_export("keras.applications.densenet.preprocess_input")
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode="torch")


@keras_export("keras.applications.densenet.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode="",
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_TORCH,
    error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC,
)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__

DOC = """

Reference
---------
- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (CVPR 2017)

선택적으로 ImageNet에서 사전 트레이닝된 가중치를 로드합니다.
모델에서 사용하는 데이터 형식 규칙은 Keras 구성 `~/.keras/keras.json`에 지정된 규칙입니다.

참고 : 각 Keras 애플리케이션에는 특정 종류의 입력 전처리가 필요합니다.
DenseNet 경우, 입력을 모델에 전달하기 전에 입력에 대해,
`tf.keras.applications.densenet.preprocess_input`을 호출해야 합니다.

Parameters
----------
include_top : bool, optional, default=True
    네트워크 top에 완전 연결 레이어를 포함할지 여부.
weights : str, optional, default="imagenet"
    `None`(무작위 초기화), 'imagenet' (ImageNet에 대해 사전 트레이닝) 중 하나 또는 로드할 가중치 파일의 경로입니다.
input_tensor : [type], optional, default=None
    모델의 이미지 입력으로 사용할 선택적 Keras 텐서(즉, `layers.Input()`의 출력).
input_shape : [type], optional, default=None
    선택적 shape 튜플, `include_top`이 False인 경우에만 지정됩니다.
    (그렇지 않으면 입력 shape은 `(224, 224, 3)` (`'channels_last'` 데이터 형식을 사용하는 경우) 또는
    `(3, 224, 224)` (`'channels_first'` 데이터 형식을 사용하는 경우)이어야 합니다.)
    정확히 3개 입력 채널이 있어야 합니다. 그리고 너비와 높이는 32보다 커야합니다.
    예) `(200, 200, 3)` 유효한 값입니다.
pooling : [type], optional, default=None
    `include_top`이 `False` 인 경우, 특성 추출을 위한 선택적 풀링 모드
    - `None` 모델의 출력이 마지막 컨볼루션 레이어의 4D 텐서 출력이 됨을 의미합니다.
    - `avg` 글로벌 평균 풀링이 마지막 컨볼루션 레이어의 출력에 적용됨을 의미합니다. 따라서, 모델의 출력은 2D 텐서가 됩니다.
    - `max` 글로벌 최대 풀링이 적용됨을 의미합니다.
classes : int, optional, default=1000
    이미지를 분류할 클래스 수 (선택 사항). `include_top`이  True이고, `weights` 인수가 지정되지 않은 경우에만 지정됩니다.

Returns
-------
`keras.Model`
    `keras.Model` 인스턴스.
"""

setattr(DenseNet121, "__doc__", DenseNet121.__doc__ + DOC)
setattr(DenseNet169, "__doc__", DenseNet169.__doc__ + DOC)
setattr(DenseNet201, "__doc__", DenseNet201.__doc__ + DOC)
