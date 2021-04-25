from argparse import ArgumentParser, RawTextHelpFormatter

from tensorflow.keras import optimizers, losses, metrics

import _path  # noqa
from imagemodel.binary_segmentations.datasets.bs_augmenter import BaseBSAugmenter
from imagemodel.binary_segmentations.datasets.oxford_iiit_pet import feeder
from imagemodel.binary_segmentations.datasets.pipeline import BSPipeline
from imagemodel.binary_segmentations.models.common_compile_options import CompileOptions
from imagemodel.binary_segmentations.models.unet_based_mobilenetv2 import UNetBasedMobileNetV2ModelManager
from imagemodel.binary_segmentations.run.common import get_run_id
from imagemodel.common.reporter import Reporter
from imagemodel.common.setup import ExperimentSetup
from imagemodel.common.trainer import Trainer

# noinspection DuplicatedCode
if __name__ == "__main__":
    """
    Examples
    --------
    >>> docker run \
    ...     --gpus all \
    ...     -it \
    ...     --rm \
    ...     -u $(id -u):$(id -g) \
    ...     -v /etc/localtime:/etc/localtime:ro \
    ...     -v $(pwd):/imagemodel \
    ...     -v ~/binary_segmentations_results:/binary_segmentations_results \
    ...     -v /data/tensorflow_datasets:/tensorflow_datasets \
    ...     -p 6006:6006 \
    ...     --workdir="/imagemodel" \
    ...     imagemodel/tkl:1.0
    >>> python imagemodel/binary_segmentations/models/trainers/unet_based_mobilenetv2_trainer.py \
    ...     --model_name unet_based_mobilenetv2 \
    ...     --result_base_folder /binary_segmentations_results \
    ...     --training_epochs 5 \
    ...     --validation_freq 1 \
    ...     --run_id binary_segmentations__unet_based_mobilenetv2_test__20210425_153158 \
    ...     --without_early_stopping
    """
    # Argument Parsing
    parser: ArgumentParser = ArgumentParser(
            description="Arguments for U-Net based MobileNetV2 model in Binary Semantic Segmentation",
            formatter_class=RawTextHelpFormatter)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--result_base_folder", type=str, required=True)
    parser.add_argument("--training_epochs", type=int)
    parser.add_argument("--validation_freq", type=int)
    parser.add_argument("--run_id", type=str)
    parser.add_argument("--without_early_stopping", action="store_true")

    args = parser.parse_args()
    model_name: str = args.model_name
    result_base_folder: str = args.result_base_folder
    training_epochs: int = args.training_epochs or 200
    validation_freq: int = args.validation_freq or 1
    run_id: str = args.run_id or get_run_id()
    without_early_stopping: bool = args.without_early_stopping
    run_id = run_id.replace(" ", "_")  # run id without space
    training_id: str = "training__model_{}__run_{}".format(model_name, run_id)

    # Experiment Setup
    experiment_setup = ExperimentSetup(result_base_folder, training_id, run_id)
    callback_list = experiment_setup.setup_callbacks(
            training_epochs=training_epochs,
            without_early_stopping=without_early_stopping,
            validation_freq=validation_freq)

    # Dataset, Model Setup
    manager = UNetBasedMobileNetV2ModelManager(input_shape=(256, 256, 3))
    helper = CompileOptions(
            optimizer=optimizers.Adam(lr=1e-4),
            loss_functions=[losses.BinaryCrossentropy()],
            metrics=[metrics.BinaryAccuracy()])
    training_feeder = feeder.BSOxfordIIITPetTrainingFeeder()
    bs_training_pipeline = BSPipeline(training_feeder, augmenter_func=BaseBSAugmenter)
    validation_feeder = feeder.BSOxfordIIITPetValidationFeeder()
    bs_validation_pipeline = BSPipeline(validation_feeder)

    # Trainer Setup
    trainer = Trainer(
            model_manager=manager,
            compile_helper=helper,
            training_pipeline=bs_training_pipeline,
            training_batch_size=4,
            validation_pipeline=bs_validation_pipeline,
            validation_batch_size=4,
            validation_freq=validation_freq)

    # Report
    reporter = Reporter(setup=experiment_setup, trainer=trainer)
    reporter.report()
    reporter.plotmodel()

    # Training
    trainer.fit(training_epochs=training_epochs, callbacks=callback_list)
