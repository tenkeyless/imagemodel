# Based on https://www.tensorflow.org/tutorials/images/segmentation

# 1. Setup
# -----
# $ pip install tensorflow-metadata
# $ pip install -q git+https://github.com/tensorflow/examples.git

# 2. Code example displaying prediction results
# ------------------------------------------

# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import cv2


# def display(display_list):
#     plt.figure(figsize=(15, 15))

#     title = ["Input Image", "True Mask", "Predicted Mask"]

#     for i in range(len(display_list)):
#         plt.subplot(1, len(display_list), i + 1)
#         plt.title(title[i])
#         plt.imshow(display_list[i])
#         plt.axis("off")
#     plt.show()


# img_name = "american_bulldog_91.npy"
# img_name = img_name[: img_name.rfind(".")]

# image_folder = "/Users/tklee/tensorflow_datasets/downloads/extracted/TAR_GZ.robots.ox.ac.uk_vgg_pets_imagesZxlcXhwB8atfm2pdIrjCelgNiW7ORYkX5h1Fkzf6MY0.tar.gz/images"
# true_label_folder = "/Users/tklee/tensorflow_datasets/downloads/extracted/TAR_GZ.robots.ox.ac.uk_vgg_pets_annotationsUkJftt5cQklCt2JrQoZW_L15jblwqTffYXUMDx01jpE.tar.gz/annotations/trimaps"
# predicted_result_folder = "x"

# img = cv2.imread(os.path.join(image_folder, img_name + ".jpg"))
# resized_img = cv2.resize(img, (128, 128))
# true_label = cv2.imread(
#     os.path.join(true_label_folder, img_name + ".png"),
#     cv2.IMREAD_GRAYSCALE,
# )
# resized_result_img = cv2.resize(true_label, (128, 128))
# predicted_img = np.load(os.path.join(predicted_result_folder, img_name + ".npy"))

# display([resized_img, resized_result_img, predicted_img])


import os
from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from imagemodel.binary_segmentations.models.model_interface import ModelInterface
from tensorflow.keras.models import Model
from tensorflow_examples.models.pix2pix import pix2pix
from typing_extensions import TypedDict
from imagemodel.common.utils.function import get_default_args
from imagemodel.common.utils.optional import optional_map

base_model = tf.keras.applications.MobileNetV2(
    input_shape=[128, 128, 3], include_top=False
)

# Use the activations of these layers
layer_names = [
    "block_1_expand_relu",  # 64x64
    "block_3_expand_relu",  # 32x32
    "block_6_expand_relu",  # 16x16
    "block_13_expand_relu",  # 8x8
    "block_16_project",  # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),  # 32x32 -> 64x64
]


def unet_based_mobilenetv2(output_channels):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2, padding="same"
    )  # 64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


class UNetBasedMobilenetv2ArgumentsDict(TypedDict):
    output_channels: Optional[int]


class UNetBasedMobilenetv2Model(ModelInterface[UNetBasedMobilenetv2ArgumentsDict]):
    __default_args = get_default_args(unet_based_mobilenetv2)

    def func(self):
        return unet_based_mobilenetv2

    def get_model(self, option_dict: UNetBasedMobilenetv2ArgumentsDict) -> Model:
        return unet_based_mobilenetv2(
            output_channels=option_dict.get("output_channels")
            or self.__default_args["output_channels"]
        )

    def convert_str_model_option_dict(
        self, option_dict: Dict[str, str]
    ) -> UNetBasedMobilenetv2ArgumentsDict:
        # output channels
        output_channels_optional_str: Optional[str] = option_dict.get("output_channels")
        output_channel_optional: Optional[int] = optional_map(
            output_channels_optional_str, eval
        )

        return UNetBasedMobilenetv2ArgumentsDict(
            output_channels=output_channel_optional
        )

    def post_processing(self, predicted_result):
        def create_mask(pred_mask):
            pred_mask = tf.argmax(pred_mask, axis=-1)
            pred_mask = pred_mask[..., tf.newaxis]
            return pred_mask

        return create_mask(predicted_result)

    def save_post_processed_result(self, filename: str, result):
        foldername_only: str = os.path.dirname(filename)
        filename_only: str = os.path.basename(filename)
        filename_without_extension: str = filename_only[: filename_only.rfind(".")]
        new_filename: str = os.path.join(
            foldername_only, "{}.npy".format(filename_without_extension)
        )
        np.save(new_filename, result)
