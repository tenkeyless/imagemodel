from argparse import ArgumentParser, RawTextHelpFormatter
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import losses, metrics, optimizers
from tensorflow.python.distribute.tpu_strategy import TPUStrategy

import _path  # noqa
from imagemodel.binary_segmentations.models.unet_level import UNetLevelModelManager
from imagemodel.common.models.common_compile_options import CompileOptions
from imagemodel.common.reporter import TrainerReporter
from imagemodel.common.setup import TrainingExperimentSetup
from imagemodel.common.trainer import Trainer
from imagemodel.common.utils.common_tpu import create_tpu, delete_tpu, tpu_initialize
from imagemodel.common.utils.optional import optional_map
from imagemodel.experimental.reference_tracking.dataset_providers.cell_tracking_dataset. \
    rt_cell_tracking_provider_t import RTCellTrackingProviderT
from imagemodel.experimental.reference_tracking.dataset_providers.rt_provider import RTProviderT
from imagemodel.experimental.reference_tracking.models.ref_local_tracking_model_031_mh2 import \
    RefLocalTrackingModel031MH2Manager

# noinspection DuplicatedCode
if __name__ == "__main__":
    """
    Examples
    --------
    # With CPU
    >>> docker run \
    ...     -it \
    ...     --rm \
    ...     -u $(id -u):$(id -g) \
    ...     -v /etc/localtime:/etc/localtime:ro \
    ...     -v $(pwd):/imagemodel \
    ...     -v /data:/data \
    ...     -v ~/reference_tracking_results:/reference_tracking_results \
    ...     -v /data/tensorflow_datasets:/tensorflow_datasets \
    ...     --workdir="/imagemodel" \
    ...     imagemodel/tkl:1.2
    >>> python imagemodel/experimental/reference_tracking/models/trainers/\
    ... ref_local_tracking_model_031_mh2_provider_trainer.py \
    ...     --model_name ref_local_tracking_model_031_mh2 \
    ...     --head_num 4 \
    ...     --aggregation_mode max \
    ...     --result_base_folder /reference_tracking_results \
    ...     --training_epochs 100 \
    ...     --validation_freq 1 \
    ...     --training_base_folder /data/tracking_training \
    ...     --validation_base_folder /data/tracking_validation \
    ...     --run_id reference_tracking__20210510_215226 \
    ...     --without_early_stopping \
    ...     --batch_size 2
    
    # With GPU
    >>> docker run \
    ...     --gpus all \
    ...     -it \
    ...     --rm \
    ...     -u $(id -u):$(id -g) \
    ...     -v /etc/localtime:/etc/localtime:ro \
    ...     -v $(pwd):/imagemodel \
    ...     -v /data:/data \
    ...     -v ~/reference_tracking_results:/reference_tracking_results \
    ...     -v /data/tensorflow_datasets:/tensorflow_datasets \
    ...     --workdir="/imagemodel" \
    ...     imagemodel/tkl:1.2
    >>> python imagemodel/experimental/reference_tracking/models/trainers/\
    ... ref_local_tracking_model_031_mh2_provider_trainer.py \
    ...     --model_name ref_local_tracking_model_031_mh2 \
    ...     --head_num 4 \
    ...     --aggregation_mode conv \
    ...     --result_base_folder /reference_tracking_results \
    ...     --training_epochs 100 \
    ...     --validation_freq 1 \
    ...     --training_base_folder /data/tracking_training \
    ...     --validation_base_folder /data/tracking_validation \
    ...     --run_id reference_tracking__20210520_190144 \
    ...     --without_early_stopping \
    ...     --batch_size 2
    
    # With TPU
    >>> docker run \
    ...     -it \
    ...     --rm \
    ...     -u $(id -u):$(id -g) \
    ...     -v /etc/localtime:/etc/localtime:ro \
    ...     -v ~/.config:/.config \
    ...     -v ~/.local:/.local \
    ...     -v $(pwd):/imagemodel \
    ...     --workdir="/imagemodel" \
    ...     imagemodel_tpu/tkl:1.4
    >>> python imagemodel/experimental/reference_tracking/models/trainers/\
    ... ref_local_tracking_model_031_mh2_provider_trainer.py \
    ...     --model_name ref_local_tracking_model_031_mh2 \
    ...     --head_num 4 \
    ...     --aggregation_mode max \
    ...     --result_base_folder gs://cell_dataset \
    ...     --training_epochs 100 \
    ...     --validation_freq 1 \
    ...     --training_base_folder gs://cell_dataset/dataset/tracking_training \
    ...     --validation_base_folder gs://cell_dataset/dataset/tracking_validation \
    ...     --run_id reference_tracking__20210510_215226 \
    ...     --without_early_stopping \
    ...     --batch_size 8 \
    ...     --ctpu_zone us-central1-b \
    ...     --tpu_name leetaekyu-1-trainer
    """
    # Argument Parsing
    parser: ArgumentParser = ArgumentParser(
            description="Arguments for Ref Local Tracking model 031 in Reference Tracking",
            formatter_class=RawTextHelpFormatter)
    # model related
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_color_image", action="store_true")
    parser.add_argument("--head_num", type=int)
    parser.add_argument("--aggregation_mode", type=str, help="'average' or 'max' or 'conv'")
    # training related
    parser.add_argument("--training_epochs", type=int)
    parser.add_argument("--validation_freq", type=int)
    parser.add_argument("--run_id", type=str)
    parser.add_argument("--without_early_stopping", action="store_true")
    parser.add_argument("--result_base_folder", type=str, required=True)
    # dataset related
    parser.add_argument("--training_base_folder", type=str, required=True)
    parser.add_argument("--validation_base_folder", type=str)
    parser.add_argument("--batch_size", type=int)
    # tpu related
    parser.add_argument("--ctpu_zone", type=str, help="VM, TPU zone. ex) 'us-central1-b'")
    parser.add_argument("--tpu_name", type=str, help="TPU name. ex) 'leetaekyu-1-trainer'")
    
    args = parser.parse_args()
    # model related
    model_name: str = args.model_name
    input_color_image: bool = args.input_color_image
    head_num: int = args.head_num
    aggregation_mode: Optional[str] = args.aggregation_mode
    # training related
    training_epochs: int = args.training_epochs or 200
    validation_freq: int = args.validation_freq or 1
    run_id: Optional[str] = args.run_id
    without_early_stopping: bool = args.without_early_stopping
    result_base_folder: str = args.result_base_folder
    # dataset related
    training_base_folder: str = args.training_base_folder
    validation_base_folder: Optional[str] = args.validation_base_folder
    batch_size: int = args.batch_size or 4
    # tpu related
    ctpu_zone: str = args.ctpu_zone or "us-central1-b"
    tpu_name_optional: Optional[str] = args.tpu_name
    
    # TPU
    strategy_optional: Optional[TPUStrategy] = None
    if tpu_name_optional:
        create_tpu(tpu_name=tpu_name_optional, ctpu_zone=ctpu_zone)
        strategy_optional = tpu_initialize(tpu_address=tpu_name_optional, tpu_zone=ctpu_zone)
    
    # Experiment Setup
    experiment_setup = TrainingExperimentSetup(result_base_folder, model_name, run_id)
    callback_list = experiment_setup.setup_callbacks(
            training_epochs=training_epochs,
            without_early_stopping=without_early_stopping,
            validation_freq=validation_freq)
    
    # Model Setup
    input_shape: Tuple[int, int, int] = (256, 256, 3) if input_color_image else (256, 256, 1)
    if tpu_name_optional:
        with strategy_optional.scope():
            u_net_model = UNetLevelModelManager.unet_level(input_shape=input_shape)
            # u_net_model2 = tf.keras.models.clone_model(u_net_model)
            # u_net_model2.set_weights(u_net_model.get_weights())
    else:
        u_net_model = UNetLevelModelManager.unet_level(input_shape=input_shape)
        # u_net_model2 = tf.keras.models.clone_model(u_net_model)
        # u_net_model2.set_weights(u_net_model.get_weights())
    manager = RefLocalTrackingModel031MH2Manager(
            unet_l4_model_main=u_net_model,
            unet_l4_model_ref=u_net_model,
            bin_num=30,
            input_main_image_shape=(256, 256, 1),
            input_ref_image_shape=(256, 256, 1),
            input_ref_bin_label_shape=(256, 256, 30),
            head_num=head_num,
            aggregation_mode=aggregation_mode)
    # Output - [Main BW Mask, Ref BW Mask, Main Color Bin Label]
    if tpu_name_optional:
        with strategy_optional.scope():
            helper = CompileOptions(
                    optimizer=optimizers.Adam(lr=1e-4),
                    loss_functions=[losses.BinaryCrossentropy(), losses.BinaryCrossentropy(),
                                    losses.CategoricalCrossentropy()],
                    # loss_weights_optional=[0.1, 0.1, 0.8],
                    loss_weights_optional=[0.25, 0.25, 0.5],
                    metrics=[[metrics.BinaryAccuracy()], [metrics.BinaryAccuracy()], [metrics.CategoricalAccuracy()]])
    else:
        helper = CompileOptions(
                optimizer=optimizers.Adam(lr=1e-4),
                loss_functions=[losses.BinaryCrossentropy(), losses.BinaryCrossentropy(),
                                losses.CategoricalCrossentropy()],
                # loss_weights_optional=[0.1, 0.1, 0.8],
                loss_weights_optional=[0.25, 0.25, 0.5],
                metrics=[[metrics.BinaryAccuracy()], [metrics.BinaryAccuracy()], [metrics.CategoricalAccuracy()]])
    
    # Dataset Setup
    rt_training_provider: RTProviderT = RTCellTrackingProviderT(
            base_folder=training_base_folder,
            shuffle_for_trainer=False,
            shuffle=True,
            random_seed=42,
            bin_size=30,
            resize_to=(256, 256))
    rt_training_dataset: tf.data.Dataset = rt_training_provider.get_output_dataset()
    rt_training_dataset_description: str = rt_training_provider.data_description
    rt_validation_provider_optional: Optional[RTProviderT] = RTCellTrackingProviderT(
            base_folder=validation_base_folder,
            shuffle_for_trainer=False,
            shuffle=False,
            random_seed=42,
            bin_size=30,
            resize_to=(256, 256)) if validation_base_folder else None
    rt_validation_dataset: Optional[tf.data.Dataset] = optional_map(
            rt_validation_provider_optional,
            lambda el: el.get_output_dataset())
    rt_validation_description: Optional[str] = optional_map(
            rt_validation_provider_optional,
            lambda el: el.data_description)
    
    # Trainer Setup
    trainer = Trainer(
            model_manager=manager,
            compile_helper=helper,
            strategy_optional=strategy_optional,
            training_dataset=rt_training_dataset,
            training_dataset_description=rt_training_dataset_description,
            training_batch_size=batch_size,
            validation_dataset=rt_validation_dataset,
            validation_dataset_description=rt_validation_description,
            validation_batch_size=batch_size,
            validation_freq=validation_freq)
    
    # Report
    reporter = TrainerReporter(setup=experiment_setup, trainer=trainer)
    reporter.report()
    reporter.plotmodel()
    
    # Training
    trainer.fit(training_epochs=training_epochs, callbacks=callback_list)
    
    if tpu_name_optional:
        delete_tpu(tpu_name=tpu_name_optional, ctpu_zone=ctpu_zone)
