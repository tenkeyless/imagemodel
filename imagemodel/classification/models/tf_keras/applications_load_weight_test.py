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
"""
Keras applications을 위한 통합 테스트.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import parameterized
import numpy as np

from tensorflow.python.keras.applications import densenet
from tensorflow.python.keras.applications import efficientnet
from tensorflow.python.keras.applications import inception_resnet_v2
from tensorflow.python.keras.applications import inception_v3
from tensorflow.python.keras.applications import mobilenet
from tensorflow.python.keras.applications import mobilenet_v2
from tensorflow.python.keras.applications import mobilenet_v3
from tensorflow.python.keras.applications import nasnet
from tensorflow.python.keras.applications import resnet
from tensorflow.python.keras.applications import resnet_v2
from tensorflow.python.keras.applications import vgg16
from tensorflow.python.keras.applications import vgg19
from tensorflow.python.keras.applications import xception
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.platform import test


ARG_TO_MODEL = {
    "resnet": (resnet, [resnet.ResNet50, resnet.ResNet101, resnet.ResNet152]),
    "resnet_v2": (
        resnet_v2,
        [resnet_v2.ResNet50V2, resnet_v2.ResNet101V2, resnet_v2.ResNet152V2],
    ),
    "vgg16": (vgg16, [vgg16.VGG16]),
    "vgg19": (vgg19, [vgg19.VGG19]),
    "xception": (xception, [xception.Xception]),
    "inception_v3": (inception_v3, [inception_v3.InceptionV3]),
    "inception_resnet_v2": (
        inception_resnet_v2,
        [inception_resnet_v2.InceptionResNetV2],
    ),
    "mobilenet": (mobilenet, [mobilenet.MobileNet]),
    "mobilenet_v2": (mobilenet_v2, [mobilenet_v2.MobileNetV2]),
    "mobilenet_v3_small": (mobilenet_v3, [mobilenet_v3.MobileNetV3Small]),
    "mobilenet_v3_large": (mobilenet_v3, [mobilenet_v3.MobileNetV3Large]),
    "densenet": (
        densenet,
        [densenet.DenseNet121, densenet.DenseNet169, densenet.DenseNet201],
    ),
    "nasnet_mobile": (nasnet, [nasnet.NASNetMobile]),
    "nasnet_large": (nasnet, [nasnet.NASNetLarge]),
    "efficientnet": (
        efficientnet,
        [
            efficientnet.EfficientNetB0,
            efficientnet.EfficientNetB1,
            efficientnet.EfficientNetB2,
            efficientnet.EfficientNetB3,
            efficientnet.EfficientNetB4,
            efficientnet.EfficientNetB5,
            efficientnet.EfficientNetB6,
            efficientnet.EfficientNetB7,
        ],
    ),
}

TEST_IMAGE_PATH = (
    "https://storage.googleapis.com/tensorflow/" "keras-applications/tests/elephant.jpg"
)
_IMAGENET_CLASSES = 1000

# 테스트 할 application 모듈 파일을 정의하는 플래그를 추가하십시오.
# 모듈 파일이 수정된 경우에만, 모듈의 application 모델 테스트를 트리거하도록 보장하기 위한, 빌드 타겟에서 'arg'로 설정됩니다.
FLAGS = flags.FLAGS
flags.DEFINE_string("module", None, "Application module used in this test.")


def _get_elephant(target_size):
    # Flatten 단계를 포함하지 않는 모델의 경우,
    # 기본값은 ImageNet 가중치를 로드할 때에도 가변 크기 입력을 허용하는 것입니다 (가능하므로).
    # 이 경우 기본값은 299x299입니다.
    if target_size[0] is None:
        target_size = (299, 299)
    test_image = data_utils.get_file("elephant.jpg", TEST_IMAGE_PATH)
    img = image.load_img(test_image, target_size=tuple(target_size))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


class ApplicationsLoadWeightTest(test.TestCase, parameterized.TestCase):
    def assertShapeEqual(self, shape1, shape2):
        if len(shape1) != len(shape2):
            raise AssertionError(
                "Shapes are different rank: %s vs %s" % (shape1, shape2)
            )
        if shape1 != shape2:
            raise AssertionError("Shapes differ: %s vs %s" % (shape1, shape2))

    def test_application_pretrained_weights_loading(self):
        app_module = ARG_TO_MODEL[FLAGS.module][0]
        apps = ARG_TO_MODEL[FLAGS.module][1]
        for app in apps:
            model = app(weights="imagenet")
            self.assertShapeEqual(model.output_shape, (None, _IMAGENET_CLASSES))
            x = _get_elephant(model.input_shape[1:3])
            x = app_module.preprocess_input(x)
            preds = model.predict(x)
            names = [p[1] for p in app_module.decode_predictions(preds)[0]]
            # Test correct label is in top 3 (weak correctness test).
            self.assertIn("African_elephant", names[:3])


if __name__ == "__main__":
    test.main()
