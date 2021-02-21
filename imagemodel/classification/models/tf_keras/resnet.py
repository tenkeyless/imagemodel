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
Keras 용 ResNet 모델.

참조:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (CVPR 2015)
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


BASE_WEIGHTS_PATH = (
    "https://storage.googleapis.com/tensorflow/keras-applications/resnet/"
)
WEIGHTS_HASHES = {
    "resnet50": (
        "2cb95161c43110f7111970584f804107",
        "4d473c1dd8becc155b73f8504c6f6626",
    ),
    "resnet101": (
        "f1aeb4b969a6efcfb50fad2f0c20cfc5",
        "88cf7a10940856eca736dc7b7e228a21",
    ),
    "resnet152": (
        "100835be76be38e30d865e96f2aaae62",
        "ee4c566cf9a93f14d82f913c2dc6dd0c",
    ),
    "resnet50v2": (
        "3ef43a0b657b3be2300d5770ece849e0",
        "fac2f116257151a9d068a22e544a4917",
    ),
    "resnet101v2": (
        "6343647c601c52e1368623803854d971",
        "c0ed64b8031c3730f411d2eb4eea35b5",
    ),
    "resnet152v2": (
        "a49b44d1979771252814e80f8ec446f9",
        "ed17cf2e0169df9d443503ef94b23b33",
    ),
    "resnext50": (
        "67a5b30d522ed92f75a1f16eef299d1a",
        "62527c363bdd9ec598bed41947b379fc",
    ),
    "resnext101": (
        "34fb605428fcc7aa4d62f44404c11509",
        "0f678c91647380debd923963594981b3",
    ),
}

layers = None


def ResNet(
    stack_fn,
    preact,
    use_bias,
    model_name="resnet",
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
    ResNet, ResNetV2, 및 ResNeXt 아키텍쳐의 인스턴스화.

    선택적으로 ImageNet에서 사전 트레이닝된 가중치를 로드합니다.
    모델에서 사용하는 데이터 형식 규칙은 Keras 구성 `~/.keras/keras.json`에 지정된 규칙입니다.

    References
    ----------
    - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (CVPR 2015)

    Parameters
    ----------
    stack_fn : [type]
        stacked residual 블록에 대한 출력 텐서를 반환하는 함수.
    preact : [type]
        사전 활성화(pre-activation) 사용 여부. (ResNetV2의 경우 True, ResNet 및 ResNeXt의 경우 False)
    use_bias : [type]
        컨볼루션 레이어에 바이어스를 사용할지 여부. (ResNet 및 ResNetV2의 경우 True, ResNeXt의 경우 False)
    model_name : str, optional, default="resnet"
        모델 이름
    include_top : bool, optional, default=True
        네트워크 상단에 있는 완전 연결 레이어를 포함할지 여부
    weights : str, optional, default="imagenet"
        `None`(무작위 초기화), 'imagenet' (ImageNet에 대해 사전 트레이닝) 중 하나 또는 로드할 가중치 파일의 경로입니다.
    input_tensor : [type], optional, default=None
        모델의 이미지 입력으로 사용할 선택적 Keras 텐서(즉, `layers.Input()`의 출력).
    input_shape : [type], optional, default=None
        선택적 shape 튜플, `include_top`이 `False`인 경우에만 지정됩니다.
        (그렇지 않으면 입력 shape은 `(224, 224, 3)` (`'channels_last'` 데이터 형식을 사용하는 경우) 또는
        `(3, 224, 224)` (`'channels_first'` 데이터 형식을 사용하는 경우)이어야 합니다.)
        정확히 3개 입력 채널이 있어야 합니다.
    pooling : [type], optional, default=None
        `include_top`이 `False`인 경우, 특성 추출을 위한 선택적 풀링 모드입니다.
        - `None`은 모델의 출력이 마지막 컨볼루션 블록의 4D 텐서 출력이 될 것임을 의미합니다.
        - `avg`는 글로벌 평균 풀링이 마지막 컨볼루션 블록의 출력에 적용되므로, 모델의 출력이 2D텐서가 될 것임을 의미합니다.
        - 'max'는 글로벌 최대 풀링이 적용됨을 의미합니다.
    classes : int, optional, default=1000
        이미지를 분류할 클래스 수. `include_top`이 `True`이고, `weights` 인수가 지정되지 않은 경우에만, 지정합니다.
    classifier_activation : str or callable, optional, default="softmax"
        "top" 레이어에서 사용할 활성화 함수입니다. `include_top=True`가 아니면 무시됩니다.
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

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name="conv1_conv")(x)

    if not preact:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="conv1_bn")(
            x
        )
        x = layers.Activation("relu", name="conv1_relu")(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool1_pool")(x)

    x = stack_fn(x)

    if preact:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="post_bn")(x)
        x = layers.Activation("relu", name="post_relu")(x)

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
    model = training.Model(inputs, x, name=model_name)

    # 가중치 불러오기
    if (weights == "imagenet") and (model_name in WEIGHTS_HASHES):
        if include_top:
            file_name = model_name + "_weights_tf_dim_ordering_tf_kernels.h5"
            file_hash = WEIGHTS_HASHES[model_name][0]
        else:
            file_name = model_name + "_weights_tf_dim_ordering_tf_kernels_notop.h5"
            file_hash = WEIGHTS_HASHES[model_name][1]
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


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """
    A residual block.

    Parameters
    ----------
    x : [type]
        입력 텐서
    filters : int
        bottleneck 레이어의 필터.
    kernel_size : int, optional, default=3
        bottleneck 레이어의 커널 크기.
    stride : int, optional, default=1
        첫 번째 레이어의 stride.
    conv_shortcut : bool, optional, default=True
        True면 컨볼루션 shortcut을 사용하고, 그렇지 않으면 identity shortcut을 사용합니다.
    name : [type], optional, default=None
        블록 라벨

    Returns
    -------
    [type]
        residual 블록에 대한 출력 텐서.
    """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride, name=name + "_0_conv")(
            x
        )
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + "_0_bn"
        )(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + "_1_conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn")(
        x
    )
    x = layers.Activation("relu", name=name + "_1_relu")(x)

    x = layers.Conv2D(filters, kernel_size, padding="SAME", name=name + "_2_conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_2_bn")(
        x
    )
    x = layers.Activation("relu", name=name + "_2_relu")(x)

    x = layers.Conv2D(4 * filters, 1, name=name + "_3_conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_3_bn")(
        x
    )

    x = layers.Add(name=name + "_add")([shortcut, x])
    x = layers.Activation("relu", name=name + "_out")(x)
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    """
    stacked residual 블록 집합입니다.

    Parameters
    ----------
    x : [type]
        입력 텐서
    filters : int
        블록에서 bottleneck 레이어의 필터 수.
    blocks : int
        블록에서 stacked 블록의 블록 수
    stride1 : int, optional, default=2
        첫 번째 블록의 첫 번째 레이어의 stride.
    name : [type], optional, default=None
        스택 라벨

    Returns
    -------
    [type]
        stacked 블록에 대한 출력 텐서.
    """
    x = block1(x, filters, stride=stride1, name=name + "_block1")
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + "_block" + str(i))
    return x


def block2(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    """
    residual 블록.

    Parameters
    ----------
    x : [type]
        입력 텐서
    filters : int
        bottleneck 레이어의 필터 수.
    kernel_size : int, optional, default=3
        bottleneck 레이어의 커널 크기.
    stride : int, optional, default=1
        첫 번째 레이어의 stride.
    conv_shortcut : bool, optional, default=False
        True면 컨볼루션 shortcut을 사용하고, 그렇지 않으면 identity shortcut을 사용합니다.
    name : [type], optional, default=None
        블록 라벨

    Returns
    -------
    [type]
        residual 블록에 대한 출력 텐서.
    """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    preact = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + "_preact_bn"
    )(x)
    preact = layers.Activation("relu", name=name + "_preact_relu")(preact)

    if conv_shortcut:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride, name=name + "_0_conv")(
            preact
        )
    else:
        shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = layers.Conv2D(filters, 1, strides=1, use_bias=False, name=name + "_1_conv")(
        preact
    )
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn")(
        x
    )
    x = layers.Activation("relu", name=name + "_1_relu")(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + "_2_pad")(x)
    x = layers.Conv2D(
        filters, kernel_size, strides=stride, use_bias=False, name=name + "_2_conv"
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_2_bn")(
        x
    )
    x = layers.Activation("relu", name=name + "_2_relu")(x)

    x = layers.Conv2D(4 * filters, 1, name=name + "_3_conv")(x)
    x = layers.Add(name=name + "_out")([shortcut, x])
    return x


def stack2(x, filters, blocks, stride1=2, name=None):
    """
    stacked residual 블록의 세트

    Parameters
    ----------
    x : [type]
        입력 텐서
    filters : int
        블록에서 bottleneck 레이어의 필터 수.
    blocks : int
        블록에서 stacked 블록의 블록 수
    stride1 : int, optional, default=2
        첫 번째 블록의 첫 번째 레이어의 stride.
    name : [type], optional, default=None
        스택 라벨

    Returns
    -------
    [type]
        stacked 블록에 대한 출력 텐서.
    """
    x = block2(x, filters, conv_shortcut=True, name=name + "_block1")
    for i in range(2, blocks):
        x = block2(x, filters, name=name + "_block" + str(i))
    x = block2(x, filters, stride=stride1, name=name + "_block" + str(blocks))
    return x


def block3(
    x, filters, kernel_size=3, stride=1, groups=32, conv_shortcut=True, name=None
):
    """
    residual 블록.

    Parameters
    ----------
    x : [type]
        입력 텐서
    filters : int
        bottleneck 레이어의 필터 수.
    kernel_size : int, optional, default=3
        bottleneck 레이어의 커널 크기.
    stride : int, optional, default=1
        첫 번째 레이어의 stride.
    groups : int, optional, default=32
        grouped 컨볼루션에 대한 그룹 크기.
    conv_shortcut : bool, optional, default=True
        True면 컨볼루션 shortcut을 사용하고, 그렇지 않으면 identity shortcut을 사용합니다.
    name : [type], optional, default=None
        블록 라벨

    Returns
    -------
    [type]
        residual 블록에 대한 출력 텐서.
    """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(
            (64 // groups) * filters,
            1,
            strides=stride,
            use_bias=False,
            name=name + "_0_conv",
        )(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + "_0_bn"
        )(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, use_bias=False, name=name + "_1_conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn")(
        x
    )
    x = layers.Activation("relu", name=name + "_1_relu")(x)

    c = filters // groups
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + "_2_pad")(x)
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        depth_multiplier=c,
        use_bias=False,
        name=name + "_2_conv",
    )(x)
    x_shape = backend.int_shape(x)[1:-1]
    x = layers.Reshape(x_shape + (groups, c, c))(x)
    x = layers.Lambda(
        lambda x: sum(x[:, :, :, :, i] for i in range(c)), name=name + "_2_reduce"
    )(x)
    x = layers.Reshape(x_shape + (filters,))(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_2_bn")(
        x
    )
    x = layers.Activation("relu", name=name + "_2_relu")(x)

    x = layers.Conv2D(
        (64 // groups) * filters, 1, use_bias=False, name=name + "_3_conv"
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_3_bn")(
        x
    )

    x = layers.Add(name=name + "_add")([shortcut, x])
    x = layers.Activation("relu", name=name + "_out")(x)
    return x


def stack3(x, filters, blocks, stride1=2, groups=32, name=None):
    """
    stacked residual 블록의 세트

    Parameters
    ----------
    x : [type]
        입력 텐서
    filters : int
        블록에서 bottleneck 레이어의 필터 수.
    blocks : int
        블록에서 stacked 블록의 블록 수
    stride1 : int, optional, default=2
        첫 번째 블록의 첫 번째 레이어의 stride.
    groups : int, optional, default=32
        grouped 컨볼루션에 대한 그룹 크기.
    groups : int, optional
        [description], by default 32
    name : [type], optional, default=None
        스택 라벨

    Returns
    -------
    [type]
        stacked 블록에 대한 출력 텐서.
    """
    x = block3(x, filters, stride=stride1, groups=groups, name=name + "_block1")
    for i in range(2, blocks + 1):
        x = block3(
            x,
            filters,
            groups=groups,
            conv_shortcut=False,
            name=name + "_block" + str(i),
        )
    return x


@keras_export(
    "keras.applications.resnet50.ResNet50",
    "keras.applications.resnet.ResNet50",
    "keras.applications.ResNet50",
)
def ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    **kwargs
):
    """
    ResNet50 아키텍쳐의 인스턴스화.
    """

    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name="conv2")
        x = stack1(x, 128, 4, name="conv3")
        x = stack1(x, 256, 6, name="conv4")
        return stack1(x, 512, 3, name="conv5")

    return ResNet(
        stack_fn,
        False,
        True,
        "resnet50",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        **kwargs
    )


@keras_export("keras.applications.resnet.ResNet101", "keras.applications.ResNet101")
def ResNet101(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    **kwargs
):
    """
    ResNet101 아키텍쳐의 인스턴스화.
    """

    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name="conv2")
        x = stack1(x, 128, 4, name="conv3")
        x = stack1(x, 256, 23, name="conv4")
        return stack1(x, 512, 3, name="conv5")

    return ResNet(
        stack_fn,
        False,
        True,
        "resnet101",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        **kwargs
    )


@keras_export("keras.applications.resnet.ResNet152", "keras.applications.ResNet152")
def ResNet152(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    **kwargs
):
    """
    ResNet152 아키텍쳐의 인스턴스화.
    """

    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name="conv2")
        x = stack1(x, 128, 8, name="conv3")
        x = stack1(x, 256, 36, name="conv4")
        return stack1(x, 512, 3, name="conv5")

    return ResNet(
        stack_fn,
        False,
        True,
        "resnet152",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        **kwargs
    )


@keras_export(
    "keras.applications.resnet50.preprocess_input",
    "keras.applications.resnet.preprocess_input",
)
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode="caffe")


@keras_export(
    "keras.applications.resnet50.decode_predictions",
    "keras.applications.resnet.decode_predictions",
)
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode="",
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_CAFFE,
    error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC,
)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__

DOC = """

선택적으로 ImageNet에서 사전 트레이닝된 가중치를 로드합니다.
모델에서 사용하는 데이터 형식 규칙은 Keras 구성 `~/.keras/keras.json`에 지정된 규칙입니다.

참고 : 각 Keras 애플리케이션에는 특정 종류의 입력 전처리가 필요합니다.
ResNet 경우, 입력을 모델에 전달하기 전에 입력에 대해,
`tf.keras.applications.resnet.preprocess_input`을 호출해야 합니다.

References
----------
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (CVPR 2015)

Parameters
----------
include_top : bool
    네트워크 상단에 있는 완전 연결 레이어를 포함할지 여부
weights : str
    `None`(무작위 초기화), 'imagenet' (ImageNet에 대해 사전 트레이닝) 중 하나 또는 로드할 가중치 파일의 경로입니다.
input_tensor : [type]
    모델의 이미지 입력으로 사용할 선택적 Keras 텐서(즉, `layers.Input()`의 출력).
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

Returns
-------
`keras.Model`
    `keras.Model` 인스턴스.
"""

setattr(ResNet50, "__doc__", ResNet50.__doc__ + DOC)
setattr(ResNet101, "__doc__", ResNet101.__doc__ + DOC)
setattr(ResNet152, "__doc__", ResNet152.__doc__ + DOC)
