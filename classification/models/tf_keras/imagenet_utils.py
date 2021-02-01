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
"""
ImageNet 데이터 전처리 및 예측 디코딩을 위한 유틸리티.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import warnings

import numpy as np

from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.util.tf_export import keras_export


CLASS_INDEX = None
CLASS_INDEX_PATH = (
    "https://storage.googleapis.com/download.tensorflow.org/"
    "data/imagenet_class_index.json"
)


PREPROCESS_INPUT_DOC = """
이미지 배치를 인코딩하는 텐서 또는 Numpy 배열을 전처리합니다.

`applications.MobileNet`을 사용한 사용 예 :

```python
i = tf.keras.layers.Input([None, None, 3], dtype = tf.uint8)
x = tf.cast(i, tf.float32)
x = tf.keras.applications.mobilenet.preprocess_input(x)
core = tf.keras.applications.MobileNet()
x = core(x)
model = tf.keras.Model(inputs=[i], outputs=[x])

image = tf.image.decode_png(tf.io.read_file('file.png'))
result = model(image)
```

Parameters
----------
x : [type]
    부동 소수점 `numpy.array` 또는`tf.Tensor`, 3D 또는 4D (색상 채널 3개 포함), 값은 [0, 255] 범위.
    데이터 유형이 호환되는 경우, 전처리된 데이터가 입력 데이터 위에 기록됩니다.
    이 동작을 피하기 위해, `numpy.copy(x)`를 사용할 수 있습니다.
data_format: [type], optional, default=None
    이미지 텐서/배열의 선택적 데이터 형식입니다.
    기본값은 None이며, 이 경우 전역 설정 `tf.keras.backend.image_data_format()`이 사용됩니다. (변경하지 않는 한 기본값은 "channels_last") {mode}

Returns
-------
전처리된 `numpy.array` 또는 `float32` 타입의 `tf.Tensor`.
    {ret}

Raises
------
    {error}
"""

PREPROCESS_INPUT_MODE_DOC = """
mode: str, default="caffe"
    "caffe", "tf" 또는 "torch"중 하나입니다. 기본값은 "caffe"입니다.
    -caffe : 이미지를 RGB에서 BGR로 변환한 다음, 스케일링 없이, ImageNet 데이터 세트와 관련하여, 각 색상 채널의 중심을 0으로 설정합니다.
    -tf : sample-wise로 -1과 1 사이로 픽셀을 스케일합니다.
    -torch : 0과 1 사이의 픽셀을 스케일 한 다음, ImageNet 데이터 세트와 관련하여 각 채널을 정규화합니다.
"""

PREPROCESS_INPUT_DEFAULT_ERROR_DOC = """
ValueError
    알 수 없는 `mode` 또는 `data_format` 인수의 경우.
"""

PREPROCESS_INPUT_ERROR_DOC = """
ValueError
    알 수 없는 `data_format` 인수의 경우.
"""

PREPROCESS_INPUT_RET_DOC_TF = """
입력 픽셀 값은 sample-wise로 -1에서 1 사이로 스케일됩니다.
"""

PREPROCESS_INPUT_RET_DOC_TORCH = """
입력 픽셀 값은 0과 1 사이에서 스케일되며, 각 채널은 ImageNet 데이터 세트와 관련하여 정규화됩니다.
"""

PREPROCESS_INPUT_RET_DOC_CAFFE = """
이미지가 RGB에서 BGR로 변환된 다음, 각 색상 채널은, 스케일링없이, ImageNet 데이터 세트에 대해 중심을 0으로 설정합니다.
"""


@keras_export("keras.applications.imagenet_utils.preprocess_input")
def preprocess_input(x, data_format=None, mode="caffe"):
    """
    이미지 배치를 인코딩하는 텐서 또는 Numpy 배열을 전처리합니다.
    """
    if mode not in {"caffe", "tf", "torch"}:
        raise ValueError("알 수 없는 mode " + str(mode))

    if data_format is None:
        data_format = backend.image_data_format()
    elif data_format not in {"channels_first", "channels_last"}:
        raise ValueError("알 수 없는 data_format " + str(data_format))

    if isinstance(x, np.ndarray):
        return _preprocess_numpy_input(x, data_format=data_format, mode=mode)
    else:
        return _preprocess_symbolic_input(x, data_format=data_format, mode=mode)


preprocess_input.__doc__ = PREPROCESS_INPUT_DOC.format(
    mode=PREPROCESS_INPUT_MODE_DOC, ret="", error=PREPROCESS_INPUT_DEFAULT_ERROR_DOC
)


@keras_export("keras.applications.imagenet_utils.decode_predictions")
def decode_predictions(preds, top=5):
    """
    ImageNet 모델의 예측을 디코딩합니다.

    Parameters
    ----------
    preds : [type]
        예측의 배치를 인코딩하는 Numpy 배열입니다.
    top : int, optional, default=5
        리턴할 top-guesses 수

    Returns
    -------
    [type]
        top 클래스 예측 튜플 `(class_name, class_description, score)` 리스트들의 리스트입니다.
        배치 입력에서 샘플 당 하나의 튜플 리스트.

    Raises
    ------
    ValueError
        `pred` 배열의 shape이 잘못된 경우, (2D여야 함)
    """
    global CLASS_INDEX

    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError(
            "`decode_predictions`는 예측의 배치(즉, shape의 2D 배열 (samples, 1000))를 예상합니다."
            "shape의 배열을 찾았습니다. : " + str(preds.shape)
        )
    if CLASS_INDEX is None:
        fpath = data_utils.get_file(
            "imagenet_class_index.json",
            CLASS_INDEX_PATH,
            cache_subdir="models",
            file_hash="c2c37ea517e94d9795004a39431a14cb",
        )
        with open(fpath) as f:
            CLASS_INDEX = json.load(f)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results


def _preprocess_numpy_input(x, data_format, mode):
    """
    이미지 배치를 인코딩하는 Numpy 배열을 전처리합니다.

    Parameters
    ----------
    x : [type]
        입력 배열, 3D 또는 4D.
    data_format : [type]
        이미지 배열의 데이터 형식입니다.
    mode : [type]
        "caffe", "tf" 또는 "torch"중 하나입니다.
        -caffe : 이미지를 RGB에서 BGR로 변환한 다음, 스케일링 없이, ImageNet 데이터 세트와 관련하여, 각 색상 채널의 중심을 0으로 설정합니다.
        -tf : sample-wise로 -1과 1 사이로 픽셀을 스케일합니다.
        -torch : 0과 1 사이의 픽셀을 스케일 한 다음, ImageNet 데이터 세트와 관련하여 각 채널을 정규화합니다.

    Returns
    -------
    [type]
        전처리 된 Numpy 배열.
    """
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(backend.floatx(), copy=False)

    if mode == "tf":
        x /= 127.5
        x -= 1.0
        return x
    elif mode == "torch":
        x /= 255.0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        if data_format == "channels_first":
            # 'RGB'->'BGR'
            if x.ndim == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        std = None

    # Zero-center by mean pixel
    if data_format == "channels_first":
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]
    return x


def _preprocess_symbolic_input(x, data_format, mode):
    """
    이미지 배치를 인코딩하는 텐서를 전처리합니다.

    Parameters
    ----------
    x : [type]
        입력 배열, 3D 또는 4D.
    data_format : [type]
        이미지 텐서의 데이터 형식입니다.
    mode : [type]
        "caffe", "tf" 또는 "torch" 중 하나입니다.
        -caffe : 이미지를 RGB에서 BGR로 변환한 다음, 스케일링 없이, ImageNet 데이터 세트와 관련하여, 각 색상 채널의 중심을 0으로 설정합니다.
        -tf : sample-wise로 -1과 1 사이로 픽셀을 스케일합니다.
        -torch : 0과 1 사이의 픽셀을 스케일 한 다음, ImageNet 데이터 세트와 관련하여 각 채널을 정규화합니다.

    Returns
    -------
    [type]
        전처리 된 텐서
    """
    if mode == "tf":
        x /= 127.5
        x -= 1.0
        return x
    elif mode == "torch":
        x /= 255.0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        if data_format == "channels_first":
            # 'RGB'->'BGR'
            if backend.ndim(x) == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        std = None

    mean_tensor = backend.constant(-np.array(mean))

    # Zero-center by mean pixel
    if backend.dtype(x) != backend.dtype(mean_tensor):
        x = backend.bias_add(
            x, backend.cast(mean_tensor, backend.dtype(x)), data_format=data_format
        )
    else:
        x = backend.bias_add(x, mean_tensor, data_format)
    if std is not None:
        x /= std
    return x


def obtain_input_shape(
    input_shape, default_size, min_size, data_format, require_flatten, weights=None
):
    """
    모델의 입력 shape을 계산/검증하는 내부 유틸리티입니다.

    Parameters
    ----------
    input_shape : [type]
        None (디폴트 네트워크 입력 shape 반환), 또는 검증을 위해 사용자가 제공한 shape 중 하나입니다.
    default_size : [type]
        모델의 기본 입력 너비/높이입니다.
    min_size : [type]
        모델에서 허용하는 최소 입력 너비/높이입니다.
    data_format : [type]
        사용할 이미지 데이터 형식입니다.
    require_flatten : [type]
        모델이 Flatten 레이어를 통해, 분류기에 연결될 것으로 기대되는지 여부입니다.
    weights : [type], optional, default=None
        `None`(무작위 초기화) 또는 'imagenet'(ImageNet에 대해 사전 트레이닝됨) 중 하나입니다.
        weights='imagenet'인 경우, 입력 채널은 3 이어야 합니다.

    Returns
    -------
    [type]
        integer shape 튜플. (None 항목을 포함할 수 있음)

    Raises
    ------
    ValueError
        잘못된 인수 값의 경우.
    """
    if weights != "imagenet" and input_shape and len(input_shape) == 3:
        if data_format == "channels_first":
            if input_shape[0] not in {1, 3}:
                warnings.warn(
                    "이 모델은 일반적으로 1 또는 3의 입력 채널을 예상합니다. "
                    "하지만, 다음 입력 채널의 input_shape로 전달되었습니다. "
                    + str(input_shape[0])
                    + " 입력 채널."
                )
            default_shape = (input_shape[0], default_size, default_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warnings.warn(
                    "이 모델은 일반적으로 1 또는 3의 입력 채널을 예상합니다. "
                    "하지만, 다음의 input_shape로 전달되었습니다. " + str(input_shape[-1]) + " 입력 채널."
                )
            default_shape = (default_size, default_size, input_shape[-1])
    else:
        if data_format == "channels_first":
            default_shape = (3, default_size, default_size)
        else:
            default_shape = (default_size, default_size, 3)
    if weights == "imagenet" and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError(
                    "`include_top=True`를 설정하고, `imagenet` 가중치를 로드할 때, "
                    "`input_shape`는 " + str(default_shape) + "가 되어야 합니다."
                )
        return default_shape
    if input_shape:
        if data_format == "channels_first":
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError("`input_shape`은 세 정수의 튜플이 되어야 합니다.")
                if input_shape[0] != 3 and weights == "imagenet":
                    raise ValueError(
                        "입력에는 3개의 채널이 있어야 합니다.; 다음을 받았습니다. "
                        "`input_shape=" + str(input_shape) + "`"
                    )
                if (input_shape[1] is not None and input_shape[1] < min_size) or (
                    input_shape[2] is not None and input_shape[2] < min_size
                ):
                    raise ValueError(
                        "Input size는 적어도 "
                        + str(min_size)
                        + "x"
                        + str(min_size)
                        + "가 되어야 합니다.; 다음을 받았습니다. `input_shape="
                        + str(input_shape)
                        + "`"
                    )
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError("`input_shape`은 세 정수의 튜플이 되어야 합니다.")
                if input_shape[-1] != 3 and weights == "imagenet":
                    raise ValueError(
                        "입력에는 3개의 채널이 있어야 합니다.; 다음을 받았습니다. "
                        "`input_shape=" + str(input_shape) + "`"
                    )
                if (input_shape[0] is not None and input_shape[0] < min_size) or (
                    input_shape[1] is not None and input_shape[1] < min_size
                ):
                    raise ValueError(
                        "Input size는 적어도 "
                        + str(min_size)
                        + "x"
                        + str(min_size)
                        + "가 되어야 합니다.; 다음을 받았습니다. `input_shape="
                        + str(input_shape)
                        + "`"
                    )
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == "channels_first":
                input_shape = (3, None, None)
            else:
                input_shape = (None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError(
                "`include_top`이 True면, static `input_shape`를 지정해야 합니다. "
                "Got `input_shape=" + str(input_shape) + "`"
            )
    return input_shape


def correct_pad(inputs, kernel_size):
    """
    다운샘플링을 사용하는 2D 컨볼루션의 제로 패딩을 위한 튜플을 반환합니다.

    Parameters
    ----------
    inputs : [type]
        입력 텐서
    kernel_size : [type]
        정수 또는 2개 정수의 튜플/리스트.

    Returns
    -------
    [type]
        튜플
    """
    img_dim = 2 if backend.image_data_format() == "channels_first" else 1
    input_size = backend.int_shape(inputs)[img_dim : (img_dim + 2)]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return ((correct[0] - adjust[0], correct[0]), (correct[1] - adjust[1], correct[1]))


def validate_activation(classifier_activation, weights):
    """
    `classifer_activation`이 가중치와 호환되는지 검증합니다.

    Parameters
    ----------
    classifier_activation : str 또는 callable
        활성화 함수
    weights : [type]
        로드할 사전 트레이닝된 가중치입니다.

    Raises
    ------
    ValueError
        사전 트레이닝된 가중치와 함께 `None` 또는 `softmax` 이외의 활성화가 사용되는 경우.
    """
    if weights is None:
        return

    classifier_activation = activations.get(classifier_activation)
    if classifier_activation not in {activations.get("softmax"), activations.get(None)}:
        raise ValueError(
            "`include_top=True`로 사전 트레이닝된 가중치를 사용할 때, "
            "`classifier_activation` 인수에 `None` 및`softmax` 활성화만 허용됩니다."
        )
