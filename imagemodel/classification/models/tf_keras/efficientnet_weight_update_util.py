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
r"""
Keras 용 EfficientNet 모델 용 유틸리티.

모델의 keras 구현을 위해 [원래 repo](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)의 ckpt 파일의 가중치를 h5 파일에 씁니다.

사용:

# 체크 포인트 efficientnet-b0/model.ckpt(https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b0.tar.gz 에서 다운로드 가능)를 사용하여, 
# 최상위 레이어없이 가중치를 업데이트하고 efficientnetb0_notop.h5로 저장합니다.

python efficientnet_weight_update_util.py --model b0 --notop \
    --ckpt efficientnet-b0/model.ckpt --o efficientnetb0_notop.h5

# 체크 포인트 noisy_student_efficientnet-b3/model.ckpt(b3에 대한 개선된 결과 제공, https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/noisystudent/noisy_student_efficientnet-b3.tar.gz 에서 다운로드 가능)를 사용하여 
# 최상위 레이어의 가중치를 업데이트하고, efficientnetb3_new.h5에 저장합니다.

python efficientnet_weight_update_util.py --model b3 --notop \
    --ckpt noisy_student_efficientnet-b3/model.ckpt --o efficientnetb3_new.h5

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import warnings

import tensorflow as tf
from tensorflow.keras.applications import efficientnet


def write_ckpt_to_h5(path_h5, path_ckpt, keras_model, use_ema=True):
    """
    체크포인트 파일(tf)의 가중치를 h5 파일(keras)에 매핑합니다.

    Parameters
    ----------
    path_h5 : str
        ckpt 파일로부터 로드된 가중치를 쓰기 위한, 출력 hdf5 파일 경로입니다.
    path_ckpt : str
        원본 저장소(https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)로부터 efficientnet 가중치를 기록하는 ckpt 파일(예 : 'efficientnet-b0/model.ckpt')의 경로
    keras_model : [type]
        keras.applications efficientnet 함수에서 빌드된 keras 모델 (예 : EfficientNetB0)
    use_ema : bool, optional, default=True
        ExponentialMovingAverage 결과 사용 여부
    """
    model_name_keras = keras_model.name
    model_name_tf = model_name_keras.replace("efficientnet", "efficientnet-")

    keras_weight_names = [w.name for w in keras_model.weights]
    tf_weight_names = get_variable_names_from_ckpt(path_ckpt)

    keras_blocks = get_keras_blocks(keras_weight_names)
    tf_blocks = get_tf_blocks(tf_weight_names)

    print("check variables match in each block")
    for keras_block, tf_block in zip(keras_blocks, tf_blocks):
        check_match(
            keras_block, tf_block, keras_weight_names, tf_weight_names, model_name_tf
        )
        print("{} and {} match.".format(tf_block, keras_block))

    block_mapping = {x[0]: x[1] for x in zip(keras_blocks, tf_blocks)}

    changed_weights = 0
    for w in keras_model.weights:
        if "block" in w.name:
            # example: 'block1a_dwconv/depthwise_kernel:0' -> 'block1a'
            keras_block = w.name.split("/")[0].split("_")[0]
            tf_block = block_mapping[keras_block]
            tf_name = keras_name_to_tf_name_block(
                w.name,
                keras_block=keras_block,
                tf_block=tf_block,
                use_ema=use_ema,
                model_name_tf=model_name_tf,
            )
        elif any([x in w.name for x in ["stem", "top", "predictions", "probs"]]):
            tf_name = keras_name_to_tf_name_stem_top(
                w.name, use_ema=use_ema, model_name_tf=model_name_tf
            )
        elif "normalization" in w.name:
            print(
                "skipping variable {}: normalization is a layer"
                "in keras implementation, but preprocessing in "
                "TF implementation.".format(w.name)
            )
            continue
        else:
            raise ValueError("{} failed to parse.".format(w.name))

        try:
            w_tf = tf.train.load_variable(path_ckpt, tf_name)
            if (w.value().numpy() != w_tf).any():
                w.assign(w_tf)
                changed_weights += 1
        except ValueError as e:
            if any([x in w.name for x in ["top", "predictions", "probs"]]):
                warnings.warn(
                    "Fail to load top layer variable {}"
                    "from {} because of {}.".format(w.name, tf_name, e)
                )
            else:
                raise ValueError("Fail to load {} from {}".format(w.name, tf_name))

    total_weights = len(keras_model.weights)
    print("{}/{} weights updated".format(changed_weights, total_weights))
    keras_model.save_weights(path_h5)


def get_variable_names_from_ckpt(path_ckpt, use_ema=True):
    """
    체크 포인트에서 텐서 이름 목록을 가져옵니다.

    Parameters
    ----------
    path_ckpt : str
        ckpt 파일 경로
    use_ema : bool, optional, default=True
        ExponentialMovingAverage 결과 사용 여부

    Returns
    -------
    [type]
        체크 포인트의 변수 이름 목록입니다.
    """
    v_all = tf.train.list_variables(path_ckpt)

    # keep name only
    v_name_all = [x[0] for x in v_all]

    if use_ema:
        v_name_all = [x for x in v_name_all if "ExponentialMovingAverage" in x]
    else:
        v_name_all = [x for x in v_name_all if "ExponentialMovingAverage" not in x]

    # remove util variables used for RMSprop
    v_name_all = [x for x in v_name_all if "RMS" not in x]
    return v_name_all


def get_tf_blocks(tf_weight_names):
    """
    전체 가중치 이름 목록에서 블록 이름을 추출합니다.
    """
    # Example: 'efficientnet-b0/blocks_0/conv2d/kernel' -> 'blocks_0'
    tf_blocks = {x.split("/")[1] for x in tf_weight_names if "block" in x}
    # sort by number
    tf_blocks = sorted(tf_blocks, key=lambda x: int(x.split("_")[1]))
    return tf_blocks


def get_keras_blocks(keras_weight_names):
    """
    전체 가중치 이름 목록에서 블록 이름을 추출합니다.
    """
    # example: 'block1a_dwconv/depthwise_kernel:0' -> 'block1a'
    keras_blocks = {x.split("_")[0] for x in keras_weight_names if "block" in x}
    return sorted(keras_blocks)


def keras_name_to_tf_name_stem_top(
    keras_name, use_ema=True, model_name_tf="efficientnet-b0"
):
    """
    h5의 이름을 stem 또는 top(헤드)에 있는 ckpt에 매핑합니다.

    h5 파일의 가중치를 가리키는 이름 keras_name을 ckpt 파일의 가중치 이름에 매핑합니다.

    Parameters
    ----------
    keras_name : str
        keras 구현의 h5 파일에 있는 가중치 이름
    use_ema : bool, optional, default=True
        ckpt에서 ExponentialMovingAverage 결과 사용 여부
    model_name_tf : str, optional, default="efficientnet-b0"
        ckpt의 모델 이름.

    Returns
    -------
    [type]
        ckpt 파일에서와 같이, 가중치 이름에 대한 문자열입니다.

    Raises
    ------
    KeyError
        keras_name을 파싱할 수 없는 경우.
    """
    if use_ema:
        ema = "/ExponentialMovingAverage"
    else:
        ema = ""

    stem_top_dict = {
        "probs/bias:0": "{}/head/dense/bias{}",
        "probs/kernel:0": "{}/head/dense/kernel{}",
        "predictions/bias:0": "{}/head/dense/bias{}",
        "predictions/kernel:0": "{}/head/dense/kernel{}",
        "stem_conv/kernel:0": "{}/stem/conv2d/kernel{}",
        "top_conv/kernel:0": "{}/head/conv2d/kernel{}",
    }
    for x in stem_top_dict:
        stem_top_dict[x] = stem_top_dict[x].format(model_name_tf, ema)

    # stem batch normalization
    for bn_weights in ["beta", "gamma", "moving_mean", "moving_variance"]:
        tf_name = "{}/stem/tpu_batch_normalization/{}{}".format(
            model_name_tf, bn_weights, ema
        )
        stem_top_dict["stem_bn/{}:0".format(bn_weights)] = tf_name

    # top / head batch normalization
    for bn_weights in ["beta", "gamma", "moving_mean", "moving_variance"]:
        tf_name = "{}/head/tpu_batch_normalization/{}{}".format(
            model_name_tf, bn_weights, ema
        )
        stem_top_dict["top_bn/{}:0".format(bn_weights)] = tf_name

    if keras_name in stem_top_dict:
        return stem_top_dict[keras_name]
    raise KeyError("{} from h5 file cannot be parsed".format(keras_name))


def keras_name_to_tf_name_block(
    keras_name,
    keras_block="block1a",
    tf_block="blocks_0",
    use_ema=True,
    model_name_tf="efficientnet-b0",
):
    """
    h5의 이름을 블록에 속한 ckpt에 매핑합니다.

    h5 파일의 가중치를 가리키는 이름 keras_name을 ckpt 파일의 가중치 이름에 매핑합니다.

    Parameters
    ----------
    keras_name : str
        keras 구현의 h5 파일에 있는 가중치 이름
    keras_block : str, optional, default="block1a"
        keras 구현을 위한 블록 이름 (예 : 'block1a')
    tf_block : str, optional, default="blocks_0"
        tf 구현을 위한 블록 이름 (예 : 'blocks_0')
    use_ema : bool, optional, default=True
        ckpt에서 ExponentialMovingAverage 결과 사용 여부
    model_name_tf : str, optional, default="efficientnet-b0"
        ckpt의 모델 이름.

    Returns
    -------
    [type]
        ckpt 파일에서와 같이 가중치 이름에 대한 문자열입니다.

    Raises
    ------
    ValueError
        keras_block이 keras_name에 표시되지 않는 경우
    """
    if keras_block not in keras_name:
        raise ValueError(
            "block name {} not found in {}".format(keras_block, keras_name)
        )

    # all blocks in the first group will not have expand conv and bn
    is_first_blocks = keras_block[5] == "1"

    tf_name = [model_name_tf, tf_block]

    # depthwide conv
    if "dwconv" in keras_name:
        tf_name.append("depthwise_conv2d")
        tf_name.append("depthwise_kernel")

    # conv layers
    if is_first_blocks:
        # first blocks only have one conv2d
        if "project_conv" in keras_name:
            tf_name.append("conv2d")
            tf_name.append("kernel")
    else:
        if "project_conv" in keras_name:
            tf_name.append("conv2d_1")
            tf_name.append("kernel")
        elif "expand_conv" in keras_name:
            tf_name.append("conv2d")
            tf_name.append("kernel")

    # squeeze expansion layers
    if "_se_" in keras_name:
        if "reduce" in keras_name:
            tf_name.append("se/conv2d")
        elif "expand" in keras_name:
            tf_name.append("se/conv2d_1")

        if "kernel" in keras_name:
            tf_name.append("kernel")
        elif "bias" in keras_name:
            tf_name.append("bias")

    # batch normalization layers
    if "bn" in keras_name:
        if is_first_blocks:
            if "project" in keras_name:
                tf_name.append("tpu_batch_normalization_1")
            else:
                tf_name.append("tpu_batch_normalization")
        else:
            if "project" in keras_name:
                tf_name.append("tpu_batch_normalization_2")
            elif "expand" in keras_name:
                tf_name.append("tpu_batch_normalization")
            else:
                tf_name.append("tpu_batch_normalization_1")

        for x in ["moving_mean", "moving_variance", "beta", "gamma"]:
            if x in keras_name:
                tf_name.append(x)
    if use_ema:
        tf_name.append("ExponentialMovingAverage")
    return "/".join(tf_name)


def check_match(
    keras_block, tf_block, keras_weight_names, tf_weight_names, model_name_tf
):
    """
    h5와 ckpt의 가중치가 일치하는지 확인합니다.

    keras_block에 있는 keras_weight_names의 각 이름을 일치시키고,
    tf_block에 있는 tf_weight_names의 이름에 일대일 대응이 되는지 확인합니다.

    Parameters
    ----------
    keras_block : str
        keras 구현을 위한 블록 이름 (예 : 'block1a')
    tf_block : str
        tf 구현을 위한 블록 이름 (예 : 'blocks_0')
    keras_weight_names : List[str]
        keras 구현의 가중치 이름
    tf_weight_names : List[str]
        tf 구현의 가중치 이름
    model_name_tf : str
        ckpt의 모델 이름.
    """
    names_from_keras = set()
    for x in keras_weight_names:
        if keras_block in x:
            y = keras_name_to_tf_name_block(
                x,
                keras_block=keras_block,
                tf_block=tf_block,
                model_name_tf=model_name_tf,
            )
            names_from_keras.add(y)

    names_from_tf = set()
    for x in tf_weight_names:
        if tf_block in x and x.split("/")[1].endswith(tf_block):
            names_from_tf.add(x)

    names_missing = names_from_keras - names_from_tf
    if names_missing:
        raise ValueError(
            "{} variables not found in checkpoint file: {}".format(
                len(names_missing), names_missing
            )
        )

    names_unused = names_from_tf - names_from_keras
    if names_unused:
        warnings.warn(
            "{} variables from checkpoint file are not used: {}".format(
                len(names_unused), names_unused
            )
        )


if __name__ == "__main__":
    arg_to_model = {
        "b0": efficientnet.EfficientNetB0,
        "b1": efficientnet.EfficientNetB1,
        "b2": efficientnet.EfficientNetB2,
        "b3": efficientnet.EfficientNetB3,
        "b4": efficientnet.EfficientNetB4,
        "b5": efficientnet.EfficientNetB5,
        "b6": efficientnet.EfficientNetB6,
        "b7": efficientnet.EfficientNetB7,
    }

    p = argparse.ArgumentParser(description="write weights from checkpoint to h5")
    p.add_argument(
        "--model",
        required=True,
        type=str,
        help="name of efficient model",
        choices=arg_to_model.keys(),
    )
    p.add_argument(
        "--notop", action="store_true", help="do not include top layers", default=False
    )
    p.add_argument("--ckpt", required=True, type=str, help="checkpoint path")
    p.add_argument(
        "--output", "-o", required=True, type=str, help="output (h5) file path"
    )
    args = p.parse_args()

    include_top = not args.notop

    model = arg_to_model[args.model](include_top=include_top)
    write_ckpt_to_h5(args.output, args.ckpt, keras_model=model)
