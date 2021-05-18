from argparse import ArgumentParser, RawTextHelpFormatter
from typing import Optional

import tensorflow as tf
from tensorflow.keras import losses, metrics, optimizers

import _path  # noqa
from imagemodel.binary_segmentations.configs.datasets import Datasets
from imagemodel.binary_segmentations.models.deeplab_v3 import DeeplabV3ModelManager
from imagemodel.binary_segmentations.run.common import get_run_id
from imagemodel.common.datasets.pipeline import Pipeline
from imagemodel.common.models.common_compile_options import CompileOptions
from imagemodel.common.reporter import TrainerReporter
from imagemodel.common.setup import TrainingExperimentSetup
from imagemodel.common.trainer import Trainer
from imagemodel.common.utils.optional import optional_map

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
    >>> python imagemodel/binary_segmentations/models/trainers/deeplab_v3_trainer.py \
    ...     --model_name unet \
    ...     --result_base_folder /binary_segmentations_results \
    ...     --training_epochs 20 \
    ...     --validation_freq 1 \
    ...     --training_pipeline bs_oxford_iiit_pet_v3_training_1 \
    ...     --validation_pipeline bs_oxford_iiit_pet_v3_validation_1 \
    ...     --run_id binary_segmentations__unet_test__20210424_163658 \
    ...     --without_early_stopping \
    ...     --batch_size 2
    """
    # Argument Parsing
    parser: ArgumentParser = ArgumentParser(
            description="Arguments for Deeplab V3 model in Binary Semantic Segmentation",
            formatter_class=RawTextHelpFormatter)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--result_base_folder", type=str, required=True)
    parser.add_argument("--training_epochs", type=int)
    parser.add_argument("--validation_freq", type=int)
    parser.add_argument("--training_pipeline", type=str, required=True)
    parser.add_argument("--validation_pipeline", type=str)
    parser.add_argument("--run_id", type=str)
    parser.add_argument("--without_early_stopping", action="store_true")
    parser.add_argument("--batch_size", type=int)
    
    args = parser.parse_args()
    model_name: str = args.model_name
    result_base_folder: str = args.result_base_folder
    training_epochs: int = args.training_epochs or 200
    validation_freq: int = args.validation_freq or 1
    run_id: str = args.run_id or get_run_id()
    without_early_stopping: bool = args.without_early_stopping
    training_pipeline: str = args.training_pipeline
    validation_pipeline: Optional[str] = args.validation_pipeline
    batch_size: int = args.batch_size or 4
    
    # Experiment Setup
    experiment_setup = TrainingExperimentSetup(result_base_folder, model_name, run_id)
    callback_list = experiment_setup.setup_callbacks(
            training_epochs=training_epochs,
            without_early_stopping=without_early_stopping,
            validation_freq=validation_freq)
    
    # Dataset, Model Setup
    manager = DeeplabV3ModelManager(input_shape=(256, 256, 3))
    helper = CompileOptions(
            optimizer=optimizers.Adam(lr=1e-4),
            loss_functions=[losses.BinaryCrossentropy()],
            metrics=[metrics.BinaryAccuracy()])
    
    bs_training_pipeline: Pipeline = Datasets(training_pipeline).get_pipeline(resize_to=(256, 256))
    bs_training_dataset: tf.data.Dataset = bs_training_pipeline.get_zipped_dataset()
    bs_training_dataset_description: str = bs_training_pipeline.data_description
    bs_validation_pipeline: Optional[Pipeline] = optional_map(
            validation_pipeline,
            lambda el: Datasets(el).get_pipeline(resize_to=(256, 256)))
    bs_validation_dataset: Optional[tf.data.Dataset] = optional_map(
            bs_validation_pipeline,
            lambda el: el.get_zipped_dataset())
    bs_validation_description: Optional[str] = optional_map(bs_validation_pipeline, lambda el: el.data_description)
    
    # Trainer Setup
    trainer = Trainer(
            model_manager=manager,
            compile_helper=helper,
            training_dataset=bs_training_dataset,
            training_dataset_description=bs_training_dataset_description,
            training_batch_size=batch_size,
            validation_dataset=bs_validation_dataset,
            validation_dataset_description=bs_validation_description,
            validation_batch_size=batch_size,
            validation_freq=validation_freq)
    
    # Report
    reporter = TrainerReporter(setup=experiment_setup, trainer=trainer)
    reporter.report()
    reporter.plotmodel()
    
    # Training
    trainer.fit(training_epochs=training_epochs, callbacks=callback_list)
