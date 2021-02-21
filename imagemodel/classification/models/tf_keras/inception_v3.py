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
Keras 용 Inception V3 모델.

참조:
- [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567) (CVPR 2016)
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


WEIGHTS_PATH = (
    "https://storage.googleapis.com/tensorflow/keras-applications/"
    "inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels.h5"
)
WEIGHTS_PATH_NO_TOP = (
    "https://storage.googleapis.com/tensorflow/keras-applications/"
    "inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
)

layers = VersionAwareLayers()


@keras_export(
    "keras.applications.inception_v3.InceptionV3", "keras.applications.InceptionV3"
)
def InceptionV3(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    """
    Inception v3 아키텍처를 인스턴스화합니다.

    선택적으로 ImageNet에서 사전 트레이닝된 가중치를 로드합니다.
    모델에서 사용하는 데이터 형식 규칙은 `tf.keras.backend.image_data_format()`에 지정된 규칙입니다.

    참고 : 각 Keras 애플리케이션에는 특정 종류의 입력 전처리가 필요합니다.
    InceptionV3 경우, 입력을 모델에 전달하기 전에 입력에 대해,
    `tf.keras.applications.inception_v3.preprocess_input`을 호출해야 합니다.

    Reference
    ---------
    - [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567) (CVPR 2016)

    Parameters
    ----------
    include_top : bool, optional, default=True
        네트워크 마지막 레이어로서, top의 완전 연결 레이어를 포함할지 여부.
    weights : str, optional, default="imagenet"
        `None`(무작위 초기화), 'imagenet' (ImageNet에 대해 사전 트레이닝) 중 하나 또는 로드할 가중치 파일의 경로입니다.
    input_tensor : [type], optional, default=None
        모델의 이미지 입력으로 사용할 선택적 Keras 텐서(즉, `layers.Input()`의 출력).
        `input_tensor`는 여러 다른 네트워크간에 입력을 공유하는 데 유용합니다.
    input_shape : [type], optional, default=None
        선택적 shape 튜플, `include_top`이 `False`인 경우에만 지정됩니다.
        (그렇지 않으면 입력 shape은 `(299, 299, 3)` (`'channels_last'` 데이터 형식을 사용하는 경우) 또는
        `(3, 299, 299)` (`'channels_first'` 데이터 형식을 사용하는 경우)이어야 합니다.)
        정확히 3개 입력 채널이 있어야 합니다. 그리고 너비와 높이는 75보다 커야합니다.
        예) `(150, 150, 3)` 유효한 값입니다.
        `input_tensor`가 제공되면, `input_shape`는 무시됩니다.
    pooling : [type], optional, default=None
        `include_top`이 `False` 인 경우, 특성 추출을 위한 선택적 풀링 모드
        - `None` 모델의 출력이 마지막 컨볼루션 블록의 4D 텐서 출력이 됨을 의미합니다.
        - `avg` 글로벌 평균 풀링이 마지막 컨볼루션 블록의 출력에 적용됨을 의미합니다. 따라서, 모델의 출력은 2D 텐서가 됩니다.
        - `max` 글로벌 최대 풀링이 적용됨을 의미합니다.
    classes : int, optional, default=1000
        이미지를 분류할 클래스 수 (선택 사항). `include_top`이 `True`이고, `weights` 인수가 지정되지 않은 경우에만 지정됩니다.
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

    if backend.image_data_format() == "channels_first":
        channel_axis = 1
    else:
        channel_axis = 3

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding="valid")
    x = conv2d_bn(x, 32, 3, 3, padding="valid")
    x = conv2d_bn(x, 64, 3, 3)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding="valid")
    x = conv2d_bn(x, 192, 3, 3, padding="valid")
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name="mixed0",
    )

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name="mixed1",
    )

    # mixed 2: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name="mixed2",
    )

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding="valid")

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding="valid")

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name="mixed3"
    )

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name="mixed4",
    )

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name="mixed" + str(5 + i),
        )

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name="mixed7",
    )

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding="valid")

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding="valid")

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name="mixed8"
    )

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name="mixed9_" + str(i)
        )

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis
        )

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name="mixed" + str(9 + i),
        )
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
    model = training.Model(inputs, x, name="inception_v3")

    # 가중치 불러오기
    if weights == "imagenet":
        if include_top:
            weights_path = data_utils.get_file(
                "inception_v3_weights_tf_dim_ordering_tf_kernels.h5",
                WEIGHTS_PATH,
                cache_subdir="models",
                file_hash="9a0d58056eeedaa3f26cb7ebd46da564",
            )
        else:
            weights_path = data_utils.get_file(
                "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
                WEIGHTS_PATH_NO_TOP,
                cache_subdir="models",
                file_hash="bcbd6486424b2319ff4ef7d526e38f63",
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def conv2d_bn(x, filters, num_row, num_col, padding="same", strides=(1, 1), name=None):
    """
    conv + BN을 적용하는 유틸리티 함수입니다.

    Parameters
    ----------
    x : [type]
        입력 텐서
    filters : int
        `Conv2D` 필터 수
    num_row : [type]
        컨볼루션 커널 높이
    num_col : [type]
        컨볼루션 커널 너비
    padding : str, optional, default="same"
        `Conv2D` 패딩 모드
    strides : tuple, optional, default=(1, 1)
        `Conv2D` 스트라이드
    name : [type], optional, default=None
        연산의 이름;
        컨볼루션의 경우, `name + '_conv'`가 되고, 배치 표준 레이어의 경우, `name + '_bn'`이 됩니다.

    Returns
    -------
    [type]
        `Conv2D` 및 `BatchNormalization`을 적용한 후 텐서를 출력합니다.
    """
    if name is not None:
        bn_name = name + "_bn"
        conv_name = name + "_conv"
    else:
        bn_name = None
        conv_name = None
    if backend.image_data_format() == "channels_first":
        bn_axis = 1
    else:
        bn_axis = 3
    x = layers.Conv2D(
        filters,
        (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name,
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = layers.Activation("relu", name=name)(x)
    return x


@keras_export("keras.applications.inception_v3.preprocess_input")
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode="tf")


@keras_export("keras.applications.inception_v3.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode="",
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_TF,
    error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC,
)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__