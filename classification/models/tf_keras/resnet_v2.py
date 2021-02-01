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
"""
Keras 용 ResNet v2 모델.

참조:
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (CVPR 2016)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.applications import resnet
from tensorflow.python.util.tf_export import keras_export


@keras_export(
    "keras.applications.resnet_v2.ResNet50V2", "keras.applications.ResNet50V2"
)
def ResNet50V2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    """
    ResNet50V2 아키텍쳐를 인스턴스화 합니다.
    """

    def stack_fn(x):
        x = resnet.stack2(x, 64, 3, name="conv2")
        x = resnet.stack2(x, 128, 4, name="conv3")
        x = resnet.stack2(x, 256, 6, name="conv4")
        return resnet.stack2(x, 512, 3, stride1=1, name="conv5")

    return resnet.ResNet(
        stack_fn,
        True,
        True,
        "resnet50v2",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        classifier_activation=classifier_activation,
    )


@keras_export(
    "keras.applications.resnet_v2.ResNet101V2", "keras.applications.ResNet101V2"
)
def ResNet101V2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    """
    ResNet101V2 아키텍쳐를 인스턴스화 합니다.
    """

    def stack_fn(x):
        x = resnet.stack2(x, 64, 3, name="conv2")
        x = resnet.stack2(x, 128, 4, name="conv3")
        x = resnet.stack2(x, 256, 23, name="conv4")
        return resnet.stack2(x, 512, 3, stride1=1, name="conv5")

    return resnet.ResNet(
        stack_fn,
        True,
        True,
        "resnet101v2",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        classifier_activation=classifier_activation,
    )


@keras_export(
    "keras.applications.resnet_v2.ResNet152V2", "keras.applications.ResNet152V2"
)
def ResNet152V2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    """
    ResNet152V2 아키텍쳐를 인스턴스화 합니다.
    """

    def stack_fn(x):
        x = resnet.stack2(x, 64, 3, name="conv2")
        x = resnet.stack2(x, 128, 8, name="conv3")
        x = resnet.stack2(x, 256, 36, name="conv4")
        return resnet.stack2(x, 512, 3, stride1=1, name="conv5")

    return resnet.ResNet(
        stack_fn,
        True,
        True,
        "resnet152v2",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        classifier_activation=classifier_activation,
    )


@keras_export("keras.applications.resnet_v2.preprocess_input")
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode="tf")


@keras_export("keras.applications.resnet_v2.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode="",
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_TF,
    error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC,
)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__

DOC = """

선택적으로 ImageNet에서 사전 트레이닝된 가중치를 로드합니다.
모델에서 사용하는 데이터 형식 규칙은 Keras 구성 `~/.keras/keras.json`에 지정된 규칙입니다.

참고 : 각 Keras 애플리케이션에는 특정 종류의 입력 전처리가 필요합니다.
ResNetV2의 경우, 입력을 모델에 전달하기 전에 입력에 대해,
`tf.keras.applications.resnet_v2.preprocess_input`을 호출해야 합니다.

References
----------
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (CVPR 2016)

Parameters
----------
include_top : bool
    네트워크 상단에 있는 완전 연결 레이어를 포함할지 여부
weights : [type]
    `None`(무작위 초기화), `'imagenet'`(ImageNet에 대해 사전 트레이닝된) 중 하나입니다.
input_tensor : [type]
    모델의 이미지 입력으로 사용할 Optional Keras 텐서 (즉,`layers.Input()`의 출력)
input_shape : [type]
    선택적 shape 튜플, `include_top`이 `False`인 경우에만 지정됩니다.
    (그렇지 않으면 입력 shape은 `(224, 224, 3)` (`'channels_last'` 데이터 형식을 사용하는 경우) 또는
    `(3, 224, 224)` (`'channels_first'` 데이터 형식을 사용하는 경우)이어야 합니다.)
    정확히 3개 입력 채널이 있어야 합니다. 그리고 너비와 높이는 32보다 커야합니다.
    예) `(200, 200, 3)` 유효한 값입니다.
pooling : [type]
    `include_top`이 `False`인 경우, 특성 추출을 위한 선택적 풀링 모드입니다.
    - `None`은 모델의 출력이 마지막 컨볼루션 블록의 4D 텐서 출력이 될 것임을 의미합니다.
    - `avg`는 글로벌 평균 풀링이 마지막 컨볼루션 블록의 출력에 적용되므로, 모델의 출력이 2D텐서가 될 것임을 의미합니다.
    - 'max'는 글로벌 최대 풀링이 적용됨을 의미합니다.
classes : int
    이미지를 분류할 클래스 수. `include_top`이 `True`이고, `weights` 인수가 지정되지 않은 경우에만, 지정합니다.
classifier_activation : str or callable
    "top" 레이어에서 사용할 활성화 함수입니다. `include_top=True`가 아니면 무시됩니다.
    "top" 레이어의 로짓을 반환하려면, `classifier_activation=None`을 설정하십시오.

Returns
-------
`keras.Model`
    `keras.Model` 인스턴스.
"""

setattr(ResNet50V2, "__doc__", ResNet50V2.__doc__ + DOC)
setattr(ResNet101V2, "__doc__", ResNet101V2.__doc__ + DOC)
setattr(ResNet152V2, "__doc__", ResNet152V2.__doc__ + DOC)
