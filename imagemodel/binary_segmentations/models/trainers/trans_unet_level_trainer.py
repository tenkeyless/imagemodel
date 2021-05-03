from argparse import ArgumentParser, RawTextHelpFormatter
from typing import Optional

from tensorflow.keras import losses, metrics, optimizers

import _path  # noqa
from imagemodel.binary_segmentations.configs.datasets import Datasets
from imagemodel.binary_segmentations.models.common_compile_options import CompileOptions
from imagemodel.binary_segmentations.models.trans_unet_level import TransUNetLevelModelManager
from imagemodel.binary_segmentations.run.common import get_run_id
from imagemodel.common.reporter import Reporter
from imagemodel.common.setup import ExperimentSetup
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
    >>> python imagemodel/binary_segmentations/models/trainers/trans_unet_level_trainer.py \
    ...     --unet_level 4 \
    ...     --model_name trans_unet_level \
    ...     --result_base_folder /binary_segmentations_results \
    ...     --training_epochs 100 \
    ...     --validation_freq 1 \
    ...     --training_pipeline bs_gs_cell_tracking_training_1 \
    ...     --validation_pipeline bs_gs_cell_tracking_validation_1 \
    ...     --run_id binary_segmentations__20210503_170519 \
    ...     --without_early_stopping \
    ...     --batch_size 2
    """
    # Argument Parsing
    parser: ArgumentParser = ArgumentParser(
            description="Arguments for Trans U-Net Level model in Binary Semantic Segmentation",
            formatter_class=RawTextHelpFormatter)
    parser.add_argument("--unet_level", type=int)
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
    unet_level: int = args.unet_level or 4
    model_name: str = args.model_name
    result_base_folder: str = args.result_base_folder
    training_epochs: int = args.training_epochs or 200
    validation_freq: int = args.validation_freq or 1
    run_id: str = args.run_id or get_run_id()
    without_early_stopping: bool = args.without_early_stopping
    run_id = run_id.replace(" ", "_")  # run id without space
    training_id: str = "training__model_{}__run_{}".format(model_name, run_id)
    training_pipeline: str = args.training_pipeline
    validation_pipeline: Optional[str] = args.validation_pipeline
    batch_size: int = args.batch_size or 4
    
    # Experiment Setup
    experiment_setup = ExperimentSetup(result_base_folder, training_id, run_id)
    callback_list = experiment_setup.setup_callbacks(
            training_epochs=training_epochs,
            without_early_stopping=without_early_stopping,
            validation_freq=validation_freq)
    
    # Dataset, Model Setup
    manager = TransUNetLevelModelManager(level=unet_level, input_shape=(256, 256, 3))
    helper = CompileOptions(
            optimizer=optimizers.Adam(lr=1e-4),
            loss_functions=[losses.BinaryCrossentropy()],
            metrics=[metrics.BinaryAccuracy()])
    bs_training_pipeline = Datasets(training_pipeline).get_pipeline(resize_to=(256, 256))
    bs_validation_pipeline = optional_map(
            validation_pipeline,
            lambda el: Datasets(el).get_pipeline(resize_to=(256, 256)))
    
    # Trainer Setup
    trainer = Trainer(
            model_manager=manager,
            compile_helper=helper,
            training_pipeline=bs_training_pipeline,
            training_batch_size=batch_size,
            training_shuffle_in_buffer=False,
            training_shuffle_buffer_size=None,
            validation_pipeline=bs_validation_pipeline,
            validation_batch_size=batch_size,
            validation_freq=validation_freq)
    
    # Report
    reporter = Reporter(setup=experiment_setup, trainer=trainer)
    reporter.report()
    reporter.plotmodel()
    
    # Training
    trainer.fit(training_epochs=training_epochs, callbacks=callback_list)
