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
Keras 용 NASNet-A 모델.

NASNet은, 관심 데이터 세트에 대해 직접 모델 아키텍처를 학습하여 자동으로 설계된 모델 제품군인, 
신경 아키텍처 검색 네트워크(Neural Architecture Search Network)를 말합니다.

여기에서 우리는 NASNet-A를 고려하여, CIFAR-10 데이터 세트에서 최고 성능 모델을 찾은 다음, ImageNet 2012 데이터 세트로 확장하여, 
CIFAR-10 및 ImageNet 2012에 대해 최첨단 성능을 얻었습니다.
ImageNet 2012에 적합한, NASNet-A 모델과 각각의 가중치만, 제공됩니다.

아래 표는 ImageNet 2012에 대한 성능을 설명합니다.
--------------------------------------------------------------------------------
|      Architecture       | Top-1 Acc | Top-5 Acc |  Multiply-Adds |  Params (M)|
--------------------------------------------------------------------------------
|   NASNet-A (4 @ 1056)  |   74.0 %  |   91.6 %  |       564 M    |     5.3    |
|   NASNet-A (6 @ 4032)  |   82.7 %  |   96.2 %  |      23.8 B    |    88.9    |
--------------------------------------------------------------------------------

참조:
- [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012) (CVPR 2018)
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


BASE_WEIGHTS_PATH = (
    "https://storage.googleapis.com/tensorflow/" "keras-applications/nasnet/"
)
NASNET_MOBILE_WEIGHT_PATH = BASE_WEIGHTS_PATH + "NASNet-mobile.h5"
NASNET_MOBILE_WEIGHT_PATH_NO_TOP = BASE_WEIGHTS_PATH + "NASNet-mobile-no-top.h5"
NASNET_LARGE_WEIGHT_PATH = BASE_WEIGHTS_PATH + "NASNet-large.h5"
NASNET_LARGE_WEIGHT_PATH_NO_TOP = BASE_WEIGHTS_PATH + "NASNet-large-no-top.h5"

layers = VersionAwareLayers()


def NASNet(
    input_shape=None,
    penultimate_filters=4032,
    num_blocks=6,
    stem_block_filters=96,
    skip_reduction=True,
    filter_multiplier=2,
    include_top=True,
    weights=None,
    input_tensor=None,
    pooling=None,
    classes=1000,
    default_size=None,
    classifier_activation="softmax",
):
    """
    NASNet 모델을 인스턴스화합니다.

    선택적으로 ImageNet에서 사전 트레이닝된 가중치를 로드합니다.
    모델에서 사용하는 데이터 형식 규칙은 Keras 구성 `~/.keras/keras.json`에 지정된 규칙입니다.

    References
    ----------
    - [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012) (CVPR 2018)

    Parameters
    ----------
    input_shape : [type], optional, default=None
        선택적 shape 튜플.
        입력 shape은 NASNetLarge에 대해 기본값이 `(331, 331, 3)`이고, NASNetMobile에 대해 `(224, 224, 3)`입니다.
        정확히 3개 입력 채널이 있어야 합니다. 그리고 너비와 높이는 32보다 커야합니다.
        예) `(224, 224, 3)` 유효한 값입니다.
    penultimate_filters : int, optional, default=4032
        두 번째(penultimate) 레이어의 필터 수입니다.
        NASNet 모델은 `NASNet (N @ P)` 표기법을 사용합니다. 여기서 :
        - N은 블록의 수 입니다.
        - P은 두 번째(penultimate) 필터의 수 입니다.
    num_blocks : int, optional, default=6
        NASNet 모델의 반복되는 블록 수입니다.
        NASNet 모델은 `NASNet (N @ P)` 표기법을 사용합니다. 여기서 :
        - N은 블록의 수 입니다.
        - P은 두 번째(penultimate) 필터의 수 입니다.
    stem_block_filters : int, optional, default=96
        초기 스템(stem) 블록의 필터 수
    skip_reduction : bool, optional, default=True
        네트워크의 끝 부분에서 감소(reduction) 단계를 스킵할지 여부입니다.
    filter_multiplier : int, optional, default=2
        네트워크의 너비를 제어합니다.
        - `filter_multiplier` < 1.0이면, 각 레이어의 필터 수를 비례적으로 줄입니다.
        - `filter_multiplier` > 1.0이면, 각 레이어의 필터 수를 비례적으로 늘립니다.
        - `filter_multiplier` = 1이면, 논문으로부터의 기본 필터 수가 각 레이어에 사용됩니다.
    include_top : bool, optional, default=True
        네트워크 상단에 있는 완전 연결 레이어를 포함할지 여부
    weights : [type], optional, default=None
        `None`(무작위 초기화), `'imagenet'`(ImageNet에 대해 사전 트레이닝된) 중 하나입니다.
    input_tensor : [type], optional, default=None
        모델의 이미지 입력으로 사용할 Optional Keras 텐서 (즉,`layers.Input()`의 출력)
    pooling : [type], optional, default=None
        `include_top`이 `False`인 경우, 특성 추출을 위한 선택적 풀링 모드입니다.
        - `None`은 모델의 출력이 마지막 컨볼루션 블록의 4D 텐서 출력이 될 것임을 의미합니다.
        - `avg`는 글로벌 평균 풀링이 마지막 컨볼루션 블록의 출력에 적용되므로, 모델의 출력이 2D텐서가 될 것임을 의미합니다.
        - 'max'는 글로벌 최대 풀링이 적용됨을 의미합니다.
    classes : int, optional, default=1000
        이미지를 분류할 클래스 수. `include_top`이 `True`이고, `weights` 인수가 지정되지 않은 경우에만, 지정합니다.
    default_size : [type], optional, default=None
        Specifies the default image size of the model
    classifier_activation : str or callable, optional, default "softmax"
        "top" 레이어에서 사용할 활성화 함수입니다. `include_top=True`가 아니면 무시됩니다.
        "top" 레이어의 로짓을 반환하려면, `classifier_activation=None`을 설정하십시오.

    Returns
    -------
    `keras.Model`
        `keras.Model` 인스턴스.

    Raises
    ------
    ValueError
        `weights`에 대한 인수가 잘못되었거나, 입력 shape이 잘못된 경우, 또는 `penultimate_filters` 값이 잘못된 경우
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

    if isinstance(input_shape, tuple) and None in input_shape and weights == "imagenet":
        raise ValueError(
            "NASNet의 입력 shape을 지정하고, `ImageNet` 가중치를 로드할 때, input_shape 인수는 정적(static)이어야 합니다. (None 항목 없음)"
            "다음을 받았습니다.: `input_shape=" + str(input_shape) + "`."
        )

    if default_size is None:
        default_size = 331

    # 적절한 입력 shape과 기본 크기를 결정합니다.
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=True,
        weights=weights,
    )

    if backend.image_data_format() != "channels_last":
        logging.warning(
            'NASNet 모델 제품군은 입력 데이터 형식 "channels_last"(너비, 높이, 채널)만 사용할 수 있습니다. '
            '그러나 당신의 설정은 기본 데이터 형식 "channels_first"(채널, 너비, 높이)를 지정하고 있습니다. '
            '`~/.keras/keras.json`에 있는 당신의 Keras 구성에서 `image_data_format = "channels_last"`로 설정해야 합니다.'
            '지금 반환되는 모델은 입력이 "channels_last" 데이터 형식을 따를 것으로 예상됩니다.'
        )
        backend.set_image_data_format("channels_last")
        old_data_format = "channels_first"
    else:
        old_data_format = None

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if penultimate_filters % (24 * (filter_multiplier ** 2)) != 0:
        raise ValueError(
            "NASNet-A 모델의 경우, `penultimate_filters` 는 24 * (`filter_multiplier` ** 2)의 곱이어야 합니다. 현재 값 : %d"
            % penultimate_filters
        )

    channel_dim = 1 if backend.image_data_format() == "channels_first" else -1
    filters = penultimate_filters // 24

    x = layers.Conv2D(
        stem_block_filters,
        (3, 3),
        strides=(2, 2),
        padding="valid",
        use_bias=False,
        name="stem_conv1",
        kernel_initializer="he_normal",
    )(img_input)

    x = layers.BatchNormalization(
        axis=channel_dim, momentum=0.9997, epsilon=1e-3, name="stem_bn1"
    )(x)

    p = None
    x, p = _reduction_a_cell(
        x, p, filters // (filter_multiplier ** 2), block_id="stem_1"
    )
    x, p = _reduction_a_cell(x, p, filters // filter_multiplier, block_id="stem_2")

    for i in range(num_blocks):
        x, p = _normal_a_cell(x, p, filters, block_id="%d" % (i))

    x, p0 = _reduction_a_cell(
        x, p, filters * filter_multiplier, block_id="reduce_%d" % (num_blocks)
    )

    p = p0 if not skip_reduction else p

    for i in range(num_blocks):
        x, p = _normal_a_cell(
            x, p, filters * filter_multiplier, block_id="%d" % (num_blocks + i + 1)
        )

    x, p0 = _reduction_a_cell(
        x, p, filters * filter_multiplier ** 2, block_id="reduce_%d" % (2 * num_blocks)
    )

    p = p0 if not skip_reduction else p

    for i in range(num_blocks):
        x, p = _normal_a_cell(
            x,
            p,
            filters * filter_multiplier ** 2,
            block_id="%d" % (2 * num_blocks + i + 1),
        )

    x = layers.Activation("relu")(x)

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

    model = training.Model(inputs, x, name="NASNet")

    # 가중치 불러오기
    if weights == "imagenet":
        if default_size == 224:  # mobile version
            if include_top:
                weights_path = data_utils.get_file(
                    "nasnet_mobile.h5",
                    NASNET_MOBILE_WEIGHT_PATH,
                    cache_subdir="models",
                    file_hash="020fb642bf7360b370c678b08e0adf61",
                )
            else:
                weights_path = data_utils.get_file(
                    "nasnet_mobile_no_top.h5",
                    NASNET_MOBILE_WEIGHT_PATH_NO_TOP,
                    cache_subdir="models",
                    file_hash="1ed92395b5b598bdda52abe5c0dbfd63",
                )
            model.load_weights(weights_path)
        elif default_size == 331:  # large version
            if include_top:
                weights_path = data_utils.get_file(
                    "nasnet_large.h5",
                    NASNET_LARGE_WEIGHT_PATH,
                    cache_subdir="models",
                    file_hash="11577c9a518f0070763c2b964a382f17",
                )
            else:
                weights_path = data_utils.get_file(
                    "nasnet_large_no_top.h5",
                    NASNET_LARGE_WEIGHT_PATH_NO_TOP,
                    cache_subdir="models",
                    file_hash="d81d89dc07e6e56530c4e77faddd61b5",
                )
            model.load_weights(weights_path)
        else:
            raise ValueError("ImageNet 가중치는 NASNetLarge 또는 NASNetMobile로만 로드할 수 있습니다.")
    elif weights is not None:
        model.load_weights(weights)

    if old_data_format:
        backend.set_image_data_format(old_data_format)

    return model


@keras_export(
    "keras.applications.nasnet.NASNetMobile", "keras.applications.NASNetMobile"
)
def NASNetMobile(
    input_shape=None,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
):
    """
    ImageNet 모드에서, NASNet 모델을 인스턴스화합니다.

    선택적으로 ImageNet에서 사전 트레이닝된 가중치를 로드합니다.
    모델에서 사용하는 데이터 형식 규칙은 Keras 구성 `~/.keras/keras.json`에 지정된 규칙입니다.

    참고 : 각 Keras 애플리케이션에는 특정 종류의 입력 전처리가 필요합니다.
    NASNet 경우, 입력을 모델에 전달하기 전에 입력에 대해,
    `tf.keras.applications.nasnet.preprocess_input`을 호출해야 합니다.

    References
    ----------
    - [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012) (CVPR 2018)

    Parameters
    ----------
    input_shape : [type], optional, default=None
        선택적 shape 튜플.
        `include_top`이 `False`인 경우에만 지정됩니다.
        (그렇지 않으면, NASNetMobile에 대해 입력 shape이 `(224, 224, 3)` 이어야 합니다.)
        정확히 3개 입력 채널이 있어야 합니다. 그리고 너비와 높이는 32보다 커야합니다.
        예) `(224, 224, 3)` 유효한 값입니다.
    include_top : bool, optional, default=True
        네트워크 상단에 있는 완전 연결 레이어를 포함할지 여부
    weights : str, optional, default="imagenet"
        `None`(무작위 초기화), 'imagenet' (ImageNet에 대해 사전 트레이닝) 중 하나.
        'imagenet' 가중치를 불러오는 경우, `input_shape`이 (224, 224, 3) 이어야 합니다.
    input_tensor : [type], optional, default=None
        모델의 이미지 입력으로 사용할 Optional Keras 텐서 (즉,`layers.Input()`의 출력)
    pooling : [type], optional, default=None
        `include_top`이 `False`인 경우, 특성 추출을 위한 선택적 풀링 모드입니다.
        - `None`은 모델의 출력이 마지막 컨볼루션 블록의 4D 텐서 출력이 될 것임을 의미합니다.
        - `avg`는 글로벌 평균 풀링이 마지막 컨볼루션 블록의 출력에 적용되므로, 모델의 출력이 2D텐서가 될 것임을 의미합니다.
        - 'max'는 글로벌 최대 풀링이 적용됨을 의미합니다.
    classes : int, optional, default=1000
        이미지를 분류할 클래스 수. `include_top`이 `True`이고, `weights` 인수가 지정되지 않은 경우에만, 지정합니다.

    Returns
    -------
    `keras.Model`
        `keras.Model` 인스턴스.

    Raises
    ------
    ValueError
        `weights`에 대한 인수가 잘못되었거나, 입력 shape이 잘못된 경우.
    RuntimeError
        separable 컨볼루션을 지원하지 않는 백엔드로 이 모델을 실행하려는 경우.
    """
    return NASNet(
        input_shape,
        penultimate_filters=1056,
        num_blocks=4,
        stem_block_filters=32,
        skip_reduction=False,
        filter_multiplier=2,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        default_size=224,
    )


@keras_export("keras.applications.nasnet.NASNetLarge", "keras.applications.NASNetLarge")
def NASNetLarge(
    input_shape=None,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
):
    """
    ImageNet 모드에서, NASNet 모델을 인스턴스화합니다.

    선택적으로 ImageNet에서 사전 트레이닝된 가중치를 로드합니다.
    모델에서 사용하는 데이터 형식 규칙은 Keras 구성 `~/.keras/keras.json`에 지정된 규칙입니다.

    참고 : 각 Keras 애플리케이션에는 특정 종류의 입력 전처리가 필요합니다.
    NASNet 경우, 입력을 모델에 전달하기 전에 입력에 대해,
    `tf.keras.applications.nasnet.preprocess_input`을 호출해야 합니다.

    References
    ----------
    - [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012) (CVPR 2018)


    Parameters
    ----------
    input_shape : [type], optional, default=None
        선택적 shape 튜플.
        `include_top`이 `False`인 경우에만 지정됩니다.
        (그렇지 않으면, NASNetLarge에 대해 입력 shape이 `(331, 331, 3)` 이어야 합니다.)
        정확히 3개 입력 채널이 있어야 합니다. 그리고 너비와 높이는 32보다 커야합니다.
        예) `(224, 224, 3)` 유효한 값입니다.
    include_top : bool, optional, default=True
        네트워크 상단에 있는 완전 연결 레이어를 포함할지 여부
    weights : str, optional, default="imagenet"
        `None`(무작위 초기화), 'imagenet' (ImageNet에 대해 사전 트레이닝) 중 하나.
        'imagenet' 가중치를 불러오는 경우, `input_shape`이 (331, 331, 3) 이어야 합니다.
    input_tensor : [type], optional, default=None
        모델의 이미지 입력으로 사용할 Optional Keras 텐서 (즉,`layers.Input()`의 출력)
    pooling : [type], optional, default=None
        `include_top`이 `False`인 경우, 특성 추출을 위한 선택적 풀링 모드입니다.
        - `None`은 모델의 출력이 마지막 컨볼루션 블록의 4D 텐서 출력이 될 것임을 의미합니다.
        - `avg`는 글로벌 평균 풀링이 마지막 컨볼루션 블록의 출력에 적용되므로, 모델의 출력이 2D텐서가 될 것임을 의미합니다.
        - 'max'는 글로벌 최대 풀링이 적용됨을 의미합니다.
    classes : int, optional, default=1000
        이미지를 분류할 클래스 수. `include_top`이 `True`이고, `weights` 인수가 지정되지 않은 경우에만, 지정합니다.

    Returns
    -------
    `keras.Model`
        `keras.Model` 인스턴스.

    Raises
    ------
    ValueError
        `weights`에 대한 인수가 잘못되었거나, 입력 shape이 잘못된 경우.
    RuntimeError
        separable 컨볼루션을 지원하지 않는 백엔드로 이 모델을 실행하려는 경우.
    """
    return NASNet(
        input_shape,
        penultimate_filters=4032,
        num_blocks=6,
        stem_block_filters=96,
        skip_reduction=True,
        filter_multiplier=2,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        default_size=331,
    )


def _separable_conv_block(
    ip, filters, kernel_size=(3, 3), strides=(1, 1), block_id=None
):
    """
    [relu-separable conv-batchnorm]의 두 블록을 추가합니다.

    Parameters
    ----------
    ip : [type]
        입력 텐서
    filters : [type]
        레이어 당 출력 필터 수
    kernel_size : tuple, optional, default=(3, 3)
        separable 컨볼루션의 커널 크기
    strides : tuple, optional, default=(1, 1)
        다운 샘플링을 위한 스트라이드 컨볼루션
    block_id : [type], optional, default=None
        시작하는 block_id

    Returns
    -------
    [type]
        Keras 텐서
    """
    channel_dim = 1 if backend.image_data_format() == "channels_first" else -1

    with backend.name_scope("separable_conv_block_%s" % block_id):
        x = layers.Activation("relu")(ip)
        if strides == (2, 2):
            x = layers.ZeroPadding2D(
                padding=imagenet_utils.correct_pad(x, kernel_size),
                name="separable_conv_1_pad_%s" % block_id,
            )(x)
            conv_pad = "valid"
        else:
            conv_pad = "same"
        x = layers.SeparableConv2D(
            filters,
            kernel_size,
            strides=strides,
            name="separable_conv_1_%s" % block_id,
            padding=conv_pad,
            use_bias=False,
            kernel_initializer="he_normal",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_dim,
            momentum=0.9997,
            epsilon=1e-3,
            name="separable_conv_1_bn_%s" % (block_id),
        )(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(
            filters,
            kernel_size,
            name="separable_conv_2_%s" % block_id,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_dim,
            momentum=0.9997,
            epsilon=1e-3,
            name="separable_conv_2_bn_%s" % (block_id),
        )(x)
    return x


def _adjust_block(p, ip, filters, block_id=None):
    """
    입력 `previous path`를 `input`의 shape과 일치하도록 조정합니다.

    필터의 출력 개수를 변경해야 하는 상황에서 사용됩니다.

    Parameters
    ----------
    p : [type]
        수정되어야 하는 입력 텐서
    ip : [type]
        shape이 일치해야 하는 입력 텐서
    filters : [type]
        일치시킬 출력 필터 수
    block_id : [type], optional, default=None
        시작하는 block_id

    Returns
    -------
    [type]
        조정된 Keras 텐서
    """
    channel_dim = 1 if backend.image_data_format() == "channels_first" else -1
    img_dim = 2 if backend.image_data_format() == "channels_first" else -2

    ip_shape = backend.int_shape(ip)

    if p is not None:
        p_shape = backend.int_shape(p)

    with backend.name_scope("adjust_block"):
        if p is None:
            p = ip

        elif p_shape[img_dim] != ip_shape[img_dim]:
            with backend.name_scope("adjust_reduction_block_%s" % block_id):
                p = layers.Activation("relu", name="adjust_relu_1_%s" % block_id)(p)
                p1 = layers.AveragePooling2D(
                    (1, 1),
                    strides=(2, 2),
                    padding="valid",
                    name="adjust_avg_pool_1_%s" % block_id,
                )(p)
                p1 = layers.Conv2D(
                    filters // 2,
                    (1, 1),
                    padding="same",
                    use_bias=False,
                    name="adjust_conv_1_%s" % block_id,
                    kernel_initializer="he_normal",
                )(p1)

                p2 = layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(p)
                p2 = layers.Cropping2D(cropping=((1, 0), (1, 0)))(p2)
                p2 = layers.AveragePooling2D(
                    (1, 1),
                    strides=(2, 2),
                    padding="valid",
                    name="adjust_avg_pool_2_%s" % block_id,
                )(p2)
                p2 = layers.Conv2D(
                    filters // 2,
                    (1, 1),
                    padding="same",
                    use_bias=False,
                    name="adjust_conv_2_%s" % block_id,
                    kernel_initializer="he_normal",
                )(p2)

                p = layers.concatenate([p1, p2], axis=channel_dim)
                p = layers.BatchNormalization(
                    axis=channel_dim,
                    momentum=0.9997,
                    epsilon=1e-3,
                    name="adjust_bn_%s" % block_id,
                )(p)

        elif p_shape[channel_dim] != filters:
            with backend.name_scope("adjust_projection_block_%s" % block_id):
                p = layers.Activation("relu")(p)
                p = layers.Conv2D(
                    filters,
                    (1, 1),
                    strides=(1, 1),
                    padding="same",
                    name="adjust_conv_projection_%s" % block_id,
                    use_bias=False,
                    kernel_initializer="he_normal",
                )(p)
                p = layers.BatchNormalization(
                    axis=channel_dim,
                    momentum=0.9997,
                    epsilon=1e-3,
                    name="adjust_bn_%s" % block_id,
                )(p)
    return p


def _normal_a_cell(ip, p, filters, block_id=None):
    """
    NASNet-A 용 Normal 셀을 추가합니다. (논문의 Fig. 4)

    Parameters
    ----------
    ip : [type]
        입력 텐서 `x`
    p : [type]
        입력 텐서 `p`
    filters : [type]
        출력 필터 수
    block_id : [type], optional, default=None
        시작하는 block_id

    Returns
    -------
    [type]
        Keras 텐서
    """
    channel_dim = 1 if backend.image_data_format() == "channels_first" else -1

    with backend.name_scope("normal_A_block_%s" % block_id):
        p = _adjust_block(p, ip, filters, block_id)

        h = layers.Activation("relu")(ip)
        h = layers.Conv2D(
            filters,
            (1, 1),
            strides=(1, 1),
            padding="same",
            name="normal_conv_1_%s" % block_id,
            use_bias=False,
            kernel_initializer="he_normal",
        )(h)
        h = layers.BatchNormalization(
            axis=channel_dim,
            momentum=0.9997,
            epsilon=1e-3,
            name="normal_bn_1_%s" % block_id,
        )(h)

        with backend.name_scope("block_1"):
            x1_1 = _separable_conv_block(
                h, filters, kernel_size=(5, 5), block_id="normal_left1_%s" % block_id
            )
            x1_2 = _separable_conv_block(
                p, filters, block_id="normal_right1_%s" % block_id
            )
            x1 = layers.add([x1_1, x1_2], name="normal_add_1_%s" % block_id)

        with backend.name_scope("block_2"):
            x2_1 = _separable_conv_block(
                p, filters, (5, 5), block_id="normal_left2_%s" % block_id
            )
            x2_2 = _separable_conv_block(
                p, filters, (3, 3), block_id="normal_right2_%s" % block_id
            )
            x2 = layers.add([x2_1, x2_2], name="normal_add_2_%s" % block_id)

        with backend.name_scope("block_3"):
            x3 = layers.AveragePooling2D(
                (3, 3),
                strides=(1, 1),
                padding="same",
                name="normal_left3_%s" % (block_id),
            )(h)
            x3 = layers.add([x3, p], name="normal_add_3_%s" % block_id)

        with backend.name_scope("block_4"):
            x4_1 = layers.AveragePooling2D(
                (3, 3),
                strides=(1, 1),
                padding="same",
                name="normal_left4_%s" % (block_id),
            )(p)
            x4_2 = layers.AveragePooling2D(
                (3, 3),
                strides=(1, 1),
                padding="same",
                name="normal_right4_%s" % (block_id),
            )(p)
            x4 = layers.add([x4_1, x4_2], name="normal_add_4_%s" % block_id)

        with backend.name_scope("block_5"):
            x5 = _separable_conv_block(
                h, filters, block_id="normal_left5_%s" % block_id
            )
            x5 = layers.add([x5, h], name="normal_add_5_%s" % block_id)

        x = layers.concatenate(
            [p, x1, x2, x3, x4, x5],
            axis=channel_dim,
            name="normal_concat_%s" % block_id,
        )
    return x, ip


def _reduction_a_cell(ip, p, filters, block_id=None):
    """
    NASNet-A 용 Reduction 셀을 추가합니다. (논문의 Fig. 4)

    Parameters
    ----------
    ip : [type]
        입력 텐서 `x`
    p : [type]
        입력 텐서 `p`
    filters : [type]
        출력 필터 수
    block_id : [type], optional, default=None
        시작하는 block_id

    Returns
    -------
    [type]
        Keras 텐서
    """
    channel_dim = 1 if backend.image_data_format() == "channels_first" else -1

    with backend.name_scope("reduction_A_block_%s" % block_id):
        p = _adjust_block(p, ip, filters, block_id)

        h = layers.Activation("relu")(ip)
        h = layers.Conv2D(
            filters,
            (1, 1),
            strides=(1, 1),
            padding="same",
            name="reduction_conv_1_%s" % block_id,
            use_bias=False,
            kernel_initializer="he_normal",
        )(h)
        h = layers.BatchNormalization(
            axis=channel_dim,
            momentum=0.9997,
            epsilon=1e-3,
            name="reduction_bn_1_%s" % block_id,
        )(h)
        h3 = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(h, 3),
            name="reduction_pad_1_%s" % block_id,
        )(h)

        with backend.name_scope("block_1"):
            x1_1 = _separable_conv_block(
                h,
                filters,
                (5, 5),
                strides=(2, 2),
                block_id="reduction_left1_%s" % block_id,
            )
            x1_2 = _separable_conv_block(
                p,
                filters,
                (7, 7),
                strides=(2, 2),
                block_id="reduction_right1_%s" % block_id,
            )
            x1 = layers.add([x1_1, x1_2], name="reduction_add_1_%s" % block_id)

        with backend.name_scope("block_2"):
            x2_1 = layers.MaxPooling2D(
                (3, 3),
                strides=(2, 2),
                padding="valid",
                name="reduction_left2_%s" % block_id,
            )(h3)
            x2_2 = _separable_conv_block(
                p,
                filters,
                (7, 7),
                strides=(2, 2),
                block_id="reduction_right2_%s" % block_id,
            )
            x2 = layers.add([x2_1, x2_2], name="reduction_add_2_%s" % block_id)

        with backend.name_scope("block_3"):
            x3_1 = layers.AveragePooling2D(
                (3, 3),
                strides=(2, 2),
                padding="valid",
                name="reduction_left3_%s" % block_id,
            )(h3)
            x3_2 = _separable_conv_block(
                p,
                filters,
                (5, 5),
                strides=(2, 2),
                block_id="reduction_right3_%s" % block_id,
            )
            x3 = layers.add([x3_1, x3_2], name="reduction_add3_%s" % block_id)

        with backend.name_scope("block_4"):
            x4 = layers.AveragePooling2D(
                (3, 3),
                strides=(1, 1),
                padding="same",
                name="reduction_left4_%s" % block_id,
            )(x1)
            x4 = layers.add([x2, x4])

        with backend.name_scope("block_5"):
            x5_1 = _separable_conv_block(
                x1, filters, (3, 3), block_id="reduction_left4_%s" % block_id
            )
            x5_2 = layers.MaxPooling2D(
                (3, 3),
                strides=(2, 2),
                padding="valid",
                name="reduction_right5_%s" % block_id,
            )(h3)
            x5 = layers.add([x5_1, x5_2], name="reduction_add4_%s" % block_id)

        x = layers.concatenate(
            [x2, x3, x4, x5], axis=channel_dim, name="reduction_concat_%s" % block_id
        )
        return x, ip


@keras_export("keras.applications.nasnet.preprocess_input")
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode="tf")


@keras_export("keras.applications.nasnet.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode="",
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_TF,
    error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC,
)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__