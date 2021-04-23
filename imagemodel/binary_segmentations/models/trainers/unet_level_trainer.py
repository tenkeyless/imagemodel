from argparse import ArgumentParser, RawTextHelpFormatter
from typing import List, Callable

import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics

import _path  # noqa
from imagemodel.binary_segmentations.datasets.bs_augmenter import BSAugmenter
from imagemodel.binary_segmentations.datasets.bs_augmenter_helper import BSAugmenterInputHelper, BSAugmenterOutputHelper
from imagemodel.binary_segmentations.datasets.oxford_iiit_pet import feeder
from imagemodel.binary_segmentations.datasets.pipeline import BSPipeline
from imagemodel.binary_segmentations.models.common_compile_options import CompileOptions
from imagemodel.binary_segmentations.models.trainers._trainer import Trainer
from imagemodel.binary_segmentations.models.unet_level import UNetLevelModelManager
from imagemodel.common.datasets.manipulator.manipulator import SupervisedManipulator


class FlipBSAugmenter(BSAugmenter):
    def __init__(self, manipulator: SupervisedManipulator):
        self._inout_helper = FlipBSAugmenterInOutHelper(
            input_datasets=manipulator.get_input_dataset(),
            output_datasets=manipulator.get_output_dataset(),
        )

    @property
    def input_helper(self) -> BSAugmenterInputHelper:
        return self._inout_helper

    @property
    def output_helper(self) -> BSAugmenterOutputHelper:
        return self._inout_helper


class FlipBSAugmenterInOutHelper(BSAugmenterInputHelper, BSAugmenterOutputHelper):
    def __init__(
            self,
            input_datasets: List[tf.data.Dataset],
            output_datasets: List[tf.data.Dataset],
    ):
        self._input_datasets: List[tf.data.Dataset] = input_datasets
        self._output_datasets: List[tf.data.Dataset] = output_datasets

    def get_image_dataset(self) -> tf.data.Dataset:
        return self._input_datasets[0]

    def image_augment_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        return [lambda img: tf.image.random_flip_left_right(img, 42)]

    def get_mask_dataset(self) -> tf.data.Dataset:
        return self._output_datasets[0]

    def mask_augment_func(self) -> List[Callable[[tf.Tensor], tf.Tensor]]:
        return [lambda img: tf.image.random_flip_left_right(img, 42)]


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser(
        description="Arguments for U-Net Level model in Binary Semantic Segmentation",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("--unet_level", type=int)
    args = parser.parse_args()
    unet_level: int = args.unet_level or 4

    manager = UNetLevelModelManager(level=unet_level, input_shape=(256, 256, 3))
    helper = CompileOptions(
        optimizer=optimizers.Adam(lr=1e-4),
        loss_functions=[losses.BinaryCrossentropy()],
        metrics=[metrics.BinaryAccuracy()])
    training_feeder = feeder.BSOxfordIIITPetTrainingFeeder()
    bs_training_pipeline = BSPipeline(training_feeder, augmenter_func=FlipBSAugmenter)
    validation_feeder = feeder.BSOxfordIIITPetValidationFeeder()
    bs_validation_pipeline = BSPipeline(validation_feeder)

    trainer = Trainer(
        model_manager=manager,
        compile_helper=helper,
        training_pipeline=bs_training_pipeline,
        training_batch_size=4,
        validation_pipeline=bs_validation_pipeline,
        validation_batch_size=4)
    trainer.fit()
