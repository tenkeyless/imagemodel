# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
Keras 용 Inception-ResNet V2 모델.

참조:
- [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)(AAAI 2017)
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


BASE_WEIGHT_URL = (
    "https://storage.googleapis.com/tensorflow/"
    "keras-applications/inception_resnet_v2/"
)
layers = None


@keras_export(
    "keras.applications.inception_resnet_v2.InceptionResNetV2",
    "keras.applications.InceptionResNetV2",
)
def InceptionResNetV2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    **kwargs
):
    """
    Inception-ResNet v2 아키텍처를 인스턴스화합니다.

    선택적으로 ImageNet에서 사전 트레이닝된 가중치를 로드합니다.
    모델에서 사용하는 데이터 형식 규칙은 Keras 구성 `~/.keras/keras.json`에 지정된 규칙입니다.

    참고 : 각 Keras 애플리케이션에는 특정 종류의 입력 전처리가 필요합니다.
    InceptionResNetV2 경우, 입력을 모델에 전달하기 전에 입력에 대해,
    `tf.keras.applications.inception_resnet_v2.preprocess_input`을 호출해야 합니다.

    Reference
    ---------
    - [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261) (AAAI 2017)

    Parameters
    ----------
    include_top : bool, optional, default=True
        네트워크 top에 완전 연결 레이어를 포함할지 여부.
    weights : str, optional, default="imagenet"
        `None`(무작위 초기화), 'imagenet' (ImageNet에 대해 사전 트레이닝) 중 하나 또는 로드할 가중치 파일의 경로입니다.
    input_tensor : [type], optional, default=None
        모델의 이미지 입력으로 사용할 선택적 Keras 텐서(즉, `layers.Input()`의 출력).
    input_shape : [type], optional, default=None
        선택적 shape 튜플, `include_top`이 `False`인 경우에만 지정됩니다.
        (그렇지 않으면 입력 shape은 `(299, 299, 3)` (`'channels_last'` 데이터 형식을 사용하는 경우) 또는
        `(3, 299, 299)` (`'channels_first'` 데이터 형식을 사용하는 경우)이어야 합니다.)
        정확히 3개 입력 채널이 있어야 합니다. 그리고 너비와 높이는 75보다 커야합니다.
        예) `(150, 150, 3)` 유효한 값입니다.
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
    **kwargs:
        이전 버전과의 호환성 만을 위해.

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
    global layers
    if "layers" in kwargs:
        layers = kwargs.pop("layers")
    else:
        layers = VersionAwareLayers()
    if kwargs:
        raise ValueError("Unknown argument(s): %s" % (kwargs,))
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
        min_size=75,
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

    # Stem block: 35 x 35 x 192
    x = conv2d_bn(img_input, 32, 3, strides=2, padding="valid")
    x = conv2d_bn(x, 32, 3, padding="valid")
    x = conv2d_bn(x, 64, 3)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = conv2d_bn(x, 80, 1, padding="valid")
    x = conv2d_bn(x, 192, 3, padding="valid")
    x = layers.MaxPooling2D(3, strides=2)(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn(x, 96, 1)
    branch_1 = conv2d_bn(x, 48, 1)
    branch_1 = conv2d_bn(branch_1, 64, 5)
    branch_2 = conv2d_bn(x, 64, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_pool = layers.AveragePooling2D(3, strides=1, padding="same")(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if backend.image_data_format() == "channels_first" else 3
    x = layers.Concatenate(axis=channel_axis, name="mixed_5b")(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(
            x, scale=0.17, block_type="block35", block_idx=block_idx
        )

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding="valid")
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 256, 3)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding="valid")
    branch_pool = layers.MaxPooling2D(3, strides=2, padding="valid")(x)
    branches = [branch_0, branch_1, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name="mixed_6a")(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(
            x, scale=0.1, block_type="block17", block_idx=block_idx
        )

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 256, 1)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding="valid")
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding="valid")
    branch_2 = conv2d_bn(x, 256, 1)
    branch_2 = conv2d_bn(branch_2, 288, 3)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding="valid")
    branch_pool = layers.MaxPooling2D(3, strides=2, padding="valid")(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name="mixed_7a")(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(
            x, scale=0.2, block_type="block8", block_idx=block_idx
        )
    x = inception_resnet_block(
        x, scale=1.0, activation=None, block_type="block8", block_idx=10
    )

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn(x, 1536, 1, name="conv_7b")

    if include_top:
        # Classification block
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
    model = training.Model(inputs, x, name="inception_resnet_v2")

    # 가중치 불러오기
    if weights == "imagenet":
        if include_top:
            fname = "inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5"
            weights_path = data_utils.get_file(
                fname,
                BASE_WEIGHT_URL + fname,
                cache_subdir="models",
                file_hash="e693bd0210a403b3192acc6073ad2e96",
            )
        else:
            fname = "inception_resnet_v2_weights_" "tf_dim_ordering_tf_kernels_notop.h5"
            weights_path = data_utils.get_file(
                fname,
                BASE_WEIGHT_URL + fname,
                cache_subdir="models",
                file_hash="d19885ff4a710c122648d3b5c3b684e4",
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def conv2d_bn(
    x,
    filters,
    kernel_size,
    strides=1,
    padding="same",
    activation="relu",
    use_bias=False,
    name=None,
):
    """
    conv + BN을 적용하는 유틸리티 함수입니다.

    Parameters
    ----------
    x : [type]
        입력 텐서
    filters : int
        `Conv2D` 필터 수
    kernel_size : int
        `Conv2D` 커널 크기
    strides : int, optional, default=1
        `Conv2D` 스트라이드
    padding : str, optional, default="same"
        `Conv2D` 패딩 모드
    activation : str, optional, default="relu"
        `Conv2D` 활성화 함수
    use_bias : bool, optional, default=False
        `Conv2D` bias 사용 여부
    name : [type], optional, default=None
        연산의 이름;
        활성화의 경우, `name + '_ac'`가 되고, 배치 표준 레이어의 경우, `name + '_bn'`이 됩니다.

    Returns
    -------
    [type]
        `Conv2D` 및 `BatchNormalization`을 적용한 후 텐서를 출력합니다.
    """
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        name=name,
    )(x)
    if not use_bias:
        bn_axis = 1 if backend.image_data_format() == "channels_first" else 3
        bn_name = None if name is None else name + "_bn"
        x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + "_ac"
        x = layers.Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation="relu"):
    """
    Inception-ResNet 블록을 추가합니다.

    이 함수는 논문에 언급된 3가지 타입의 Inception-ResNet 블록을 빌드하며,
    `block_type` 인수(공식 TF-slim 구현에서 사용되는 블록 이름)에 의해 제어됩니다.
    - Inception-ResNet-A: `block_type='block35'`
    - Inception-ResNet-B: `block_type='block17'`
    - Inception-ResNet-C: `block_type='block8'`

    Parameters
    ----------
    x : [type]
        입력 텐서
    scale : [type]
        바로가기 브랜치에 합산(add)하기 전에, residual(즉, inception 모듈을 통해 `x`를 전달한 출력)의 배율을 조정하는 scale 인수.
        `r`을 residual 브랜치로부터의 출력이라고 하면, 이 블록의 출력은 `x + scale * r`이 됩니다.
    block_type : str
        `'block35'`, `'block17'` 또는 `'block8'`, residual 브랜치에서 네트워크 구조를 결정합니다.
    block_idx : [type]
        레이어 이름을 생성하는 데 사용되는 `int`입니다. Inception-ResNet 블록은 이 네트워크에서 여러 번 반복됩니다.
        우리는 각각의 반복을 식별하기 위해 `block_idx`를 사용합니다.
        예를 들어, 첫 번째 Inception-ResNet-A 블록에는 `block_type='block35', block_idx=0`이 될 것이고,
        레이어 이름에는 공통 접두사 'block35_0'`이 있습니다.
    activation : str, optional, default="relu"
        블록의 끝에서 사용할 활성화 함수입니다. ([활성화](../activations.md) 참조)
        `activation=None`이면, 활성화가 적용되지 않습니다. (즉, "linear" 활성화 : `a(x) = x`)

    Returns
    -------
    [type]
        블록에 대한 출력 텐서.

    Raises
    ------
    ValueError
        `block_type`이 `'block35'`, `'block17'` 또는 `'block8'` 중 하나가 아닌 경우.
    """
    if block_type == "block35":
        branch_0 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(branch_1, 32, 3)
        branch_2 = conv2d_bn(x, 32, 1)
        branch_2 = conv2d_bn(branch_2, 48, 3)
        branch_2 = conv2d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == "block17":
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 128, 1)
        branch_1 = conv2d_bn(branch_1, 160, [1, 7])
        branch_1 = conv2d_bn(branch_1, 192, [7, 1])
        branches = [branch_0, branch_1]
    elif block_type == "block8":
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn(branch_1, 256, [3, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError(
            "알 수 없는 Inception-ResNet 블록 타입입니다. "
            '"block35", "block17" 또는 "block8" 중 하나가 되어야 하지만, '
            "다음을 받았습니다. : " + str(block_type)
        )

    block_name = block_type + "_" + str(block_idx)
    channel_axis = 1 if backend.image_data_format() == "channels_first" else 3
    mixed = layers.Concatenate(axis=channel_axis, name=block_name + "_mixed")(branches)
    up = conv2d_bn(
        mixed,
        backend.int_shape(x)[channel_axis],
        1,
        activation=None,
        use_bias=True,
        name=block_name + "_conv",
    )

    x = layers.Lambda(
        lambda inputs, scale: inputs[0] + inputs[1] * scale,
        output_shape=backend.int_shape(x)[1:],
        arguments={"scale": scale},
        name=block_name,
    )([x, up])
    if activation is not None:
        x = layers.Activation(activation, name=block_name + "_ac")(x)
    return x


@keras_export("keras.applications.inception_resnet_v2.preprocess_input")
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode="tf")


@keras_export("keras.applications.inception_resnet_v2.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode="",
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_TF,
    error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC,
)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__