import os
import sys

sys.path.append(os.getcwd())

from argparse import ArgumentParser, RawTextHelpFormatter
from typing import List, Optional, Tuple, TypeVar

import tensorflow as tf
from image_keras.supports.folder import create_folder_if_not_exist
from keras.utils import plot_model
from segmentations.configs.datasets import Datasets
from segmentations.configs.losses import Losses
from segmentations.configs.metrics import Metrics
from segmentations.configs.optimizers import Optimizers
from segmentations.models.model import Models
from segmentations.run.common import (
    get_run_id,
    loss_coords,
    model_option_coords,
    setup_continuous_training,
)
from tensorflow.keras.callbacks import Callback, History, TensorBoard
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils.list import check_all_exists_or_not, check_exists_or_not, sublist
from utils.modules import load_module
from utils.optional import optional_map

if __name__ == "__main__":
    # 1. Variables --------
    # 1-1) Variables with Parser
    parser: ArgumentParser = ArgumentParser(
        description="Arguments for Semantic Segmentation",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Should be one of models in `Datasets` of `segmentations/configs/datasets.py`. \n"
        "ex) 'oxford_iiit_pet_v3'",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Should be one of models in `Models` of `segmentations/models/model.py`. \n"
        "ex) 'unet'",
    )
    parser.add_argument(
        "--model_option",
        type=model_option_coords,
        action="append",
        help="Model Options. Hyper parameters. \n"
        '- input shape  : `--model_option "input_shape@(256, 256, 3)"`',
    )
    parser.add_argument(
        "--result_base_folder",
        type=str,
        required=True,
        help="Name of base folder. 'segmentations/training', '/user/name/path/folder'.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for training and batch. Defaults to 8. \n" "ex) 8",
    )
    parser.add_argument(
        "--training_epochs",
        type=int,
        help="Number of epochs for training. Defaults to 200. \n" "ex) 200",
    )
    parser.add_argument(
        "--val_freq",
        type=int,
        help="Validate every this value. Defaults to 1. (validate every epoch) \n"
        "ex) 1",
    )
    parser.add_argument(
        "--continuous_model_name",
        type=str,
        help="Training will be continue for this `model`. Full path of TF model. \n"
        "ex) '/my_folder/weights/training__model_unet__run_20210108_221742.epoch_78'",
    )
    parser.add_argument(
        "--continuous_epoch",
        type=int,
        help="Training will be continue from this `epoch`. \n"
        "If model trained during 12 epochs, this will be 12. \n"
        "ex) 12",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        help="(Without space) Run with custom id. \n"
        "Be careful not to use duplicate IDs. If not specified, timestamp will be ID. \n"
        "ex) '210108_185302'",
    )
    parser.add_argument(
        "--plot_sample",
        action="store_true",
        help="With this option, we will plot sample images on training result folder.",
    )
    parser.add_argument(
        "--without_early_stopping",
        action="store_true",
        help="With this option, training will not early stop.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="Loss should be exist in `segmentations.configs.optimizers`. \n"
        "ex) 'adam1'",
    )
    parser.add_argument(
        "--losses",
        type=loss_coords,
        action="append",
        help="Loss and weight pair. \n"
        "Loss should be exist in `segmentations.configs.losses`. \n"
        "- Case 1. 1 output  : `--losses 'categorical_crossentropy',1.0`\n"
        "- Case 2. 2 outputs : `--losses 'categorical_crossentropy',0.8 --losses 'weighted_cce',0.2`\n",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        action="append",
        help="Metrics. \n"
        "Metric should be exist in `ref_local_tracking_with_aux.configs.metrics`. \n"
        "- Case 1. 1 output, 1 metric  : `--metrics 'categorical_accuracy'` \n"
        "- Case 2. 1 output, 2 metric  : `--metrics 'categorical_accuracy' 'categorical_accuracy'` \n"
        "- Case 3. 2 output, (1, 1) metric  : `--metrics 'categorical_accuracy' --metrics 'categorical_accuracy'` \n"
        "- Case 4. 2 output, (1, 0) metric  : `--metrics categorical_accuracy --metrics none` \n"
        "- Case 5. 2 output, (2, 0) metric  : `--metrics categorical_accuracy categorical_accuracy --metrics none` \n",
    )
    args = parser.parse_args()

    # 1-2) Set variables
    # required
    model_name: str = args.model_name
    result_base_folder: str = args.result_base_folder
    model_option_tuple_list: List[Tuple[str, str]] = args.model_option

    # optional with default value
    dataset: str = args.dataset or Datasets.get_default()
    val_freq: int = args.val_freq or 1
    batch_size: int = args.batch_size or 8
    training_epochs: int = args.training_epochs or 200
    run_id: str = args.run_id or get_run_id()

    # flag
    without_early_stopping: bool = args.without_early_stopping
    plot_sample: bool = args.plot_sample

    # optimizer, losses, metrics
    optimizer: str = args.optimizer or Optimizers.get_default()
    loss_weight_tuple_list: List[Tuple[str, float]] = args.losses or [
        (Losses.get_default(), 1.0)
    ]
    metrics_list: List[List[str]] = args.metrics or [[Metrics.get_default()]]

    # processing
    training_batch_size: int = batch_size
    val_batch_size: int = batch_size
    run_id = run_id.replace(" ", "_")  # run id without space
    training_id: str = "training__model_{}__run_{}".format(model_name, run_id)

    # 1-3) continuous
    continuous_model_name: Optional[str] = args.continuous_model_name
    continuous_epoch: Optional[int] = args.continuous_epoch

    # continuous parameter check
    if not check_all_exists_or_not([continuous_model_name, continuous_epoch]):
        raise RuntimeError(
            "`continuous_model_name` and `continuous_epoch` should both exists or not."
        )
    training_id = (
        setup_continuous_training(continuous_model_name, continuous_epoch)
        or training_id
    )

    # 1-4) Variable Check
    models = Models(model_name).get_model()
    models.check_dict_option_key_and_raise(dict(model_option_tuple_list))
    model = models.get_model_from_str_model_option(dict(model_option_tuple_list))

    # 2. Setup --------
    # 2-1) Setup folders for Result and TensorBoard Log.
    # data folder
#     base_data_folder: str = os.path.join(result_base_folder, "data")
#     training_result_folder: str = os.path.join(base_data_folder, training_id)
#     create_folder_if_not_exist(training_result_folder)

#     # save folder
#     base_save_folder: str = os.path.join(result_base_folder, "save")
#     save_models_folder: str = os.path.join(base_save_folder, "models")
#     save_weights_folder: str = os.path.join(base_save_folder, "weights")
#     save_tf_log_folder: str = os.path.join(base_save_folder, "tf_logs")
#     tf_run_log_dir: str = os.path.join(save_tf_log_folder, training_id)
#     for folder in [save_models_folder, save_weights_folder, tf_run_log_dir]:
#         create_folder_if_not_exist(folder)

#     # 2-2) Setup for dataset.
#     datasets = Datasets(dataset).get_dataset()
#     training_dataset = datasets.get_training_dataset(
#         batch_size_optional=training_batch_size
#     )
#     val_dataset = datasets.get_validation_dataset(batch_size_optional=val_batch_size)

#     # 2-3) Report setup results
#     info: str = """
# # Information ---------------------------
# Training ID: {}
# Training Dataset: {}
# Validation Dataset: {}
# Tensorboard Log Folder: {}
# Training Data Folder: {}/{}
# -----------------------------------------
# """.format(
#         training_id,
#         training_dataset.value,
#         val_dataset.value,
#         tf_run_log_dir,
#         base_data_folder,
#         training_id,
#     )
#     print(info)
#     tmp_info = os.path.join(training_result_folder, "info_{}.txt".format(run_id))
#     f = open(tmp_info, "w")
#     f.write(info)
#     f.close()

#     # 3. Model compile --------

#     ref_tracking_model_module = load_module(
#         module_name=model_name,
#         file_path="segmentations/models/{}.py".format(model_name),
#     )
#     if pretrained_unet_path is None:
#         unet_model = unet_l4(input_name="unet_input", output_name="unet_output")
#     else:
#         unet_model = tf.keras.models.load_model(
#             pretrained_unet_path,
#             custom_objects={"binary_class_mean_iou": binary_class_mean_iou},
#         )

#     if not with_shared_unet:
#         unet_model2 = tf.keras.models.clone_model(unet_model)
#         unet_model2.set_weights(unet_model.get_weights())

#         model = getattr(ref_tracking_model_module, model_name)(
#             unet_l4_model_main=unet_model,
#             unet_l4_model_ref=unet_model2,
#             bin_num=bin_size,
#         )
#     else:
#         model = getattr(ref_tracking_model_module, model_name)(
#             pre_trained_unet_l4_model=unet_model, bin_num=bin_size
#         )

#     # continue setting (weights)
#     if continuous_model_name is not None:
#         model = tf.keras.models.load_model(continuous_model_name)

#     model_optimizer = Optimizers(optimizer).get_optimizer()
#     model_loss_list = list(
#         map(lambda el: Losses(el[0]).get_loss(), loss_weight_tuple_list)
#     )
#     model_loss_weight_list = list(map(lambda el: el[1], loss_weight_tuple_list))
#     model_metrics_list = [
#         list(filter(lambda v: v, [Metrics(metric).get_metric() for metric in metrics]))
#         for metrics in metrics_list
#     ]
#     output_keys = model.output_names
#     if len(loss_weight_tuple_list) != len(output_keys):
#         raise ValueError(
#             "Number of `--losses` option(s) should be {}.".format(len(output_keys))
#         )
#     if len(metrics_list) != len(output_keys):
#         raise ValueError(
#             "Number of `--metrics` option(s) should be {}.".format(len(output_keys))
#         )
#     model_loss_dict = dict(zip(output_keys, model_loss_list))
#     model_loss_weight_dict = dict(zip(output_keys, model_loss_weight_list))
#     model_metrics_dict = dict(zip(output_keys, model_metrics_list))
#     model.compile(
#         optimizer=model_optimizer,
#         loss=model_loss_dict,
#         loss_weights=model_loss_weight_dict,
#         metrics=model_metrics_dict,
#     )
#     tmp_plot_model_img_path = "/tmp/model.png"
#     plot_model(
#         model,
#         show_shapes=True,
#         to_file=tmp_plot_model_img_path,
#         expand_nested=True,
#         dpi=144,
#     )
#     upload_blob(
#         bucket_name,
#         tmp_plot_model_img_path,
#         os.path.join(
#             training_result_folder_without_gs,
#             os.path.basename(tmp_plot_model_img_path),
#         ),
#     )
#     model.summary()

#     # 4. Dataset --------
#     # 4-1) Training dataset
#     training_dataset = make_preprocessed_tf_dataset(
#         batch_size=training_batch_size,
#         inout_folder_tuple=training_inout_datasets,
#         bin_size=bin_size,
#     )
#     training_samples = len(training_dataset) * training_batch_size
#     if plot_sample:
#         plot_and_upload_dataset(
#             dataset=training_dataset,
#             batch_size=training_batch_size,
#             bin_size=bin_size,
#             bucket_name=bucket_name,
#             upload_gs_folder=training_result_folder_without_gs,
#         )

#     # 4-2) Validation dataset
#     val_dataset = make_preprocessed_tf_dataset(
#         batch_size=val_batch_size,
#         inout_folder_tuple=val_inout_datasets,
#         bin_size=bin_size,
#     )
#     val_samples = len(val_dataset) * val_batch_size

#     # 5. Training --------
#     # 5-1) Parameters
#     training_steps_per_epoch: int = training_samples // training_batch_size
#     val_steps: int = val_samples // val_batch_size

#     # callbacks
#     model_checkpoint: Callback = ModelCheckpoint(
#         os.path.join(save_weights_folder, training_id + ".epoch_{epoch:02d}"),
#         verbose=1,
#     )
#     early_stopping_patience: int = training_epochs // (10 * val_freq)
#     early_stopping: Callback = EarlyStopping(
#         patience=early_stopping_patience, verbose=1
#     )
#     tensorboard_cb: Callback = TensorBoard(log_dir=tf_run_log_dir)
#     callback_list: List[Callback] = [tensorboard_cb, model_checkpoint]
#     if not without_early_stopping:
#         callback_list.append(early_stopping)

#     # continue setting (initial epoch)
#     initial_epoch = 0
#     if check_exists_or_not(continuous_model_name, continuous_epoch):
#         assert isinstance(continuous_epoch, int)
#         initial_epoch = continuous_epoch

#     # 5-2) Training
#     history: History = model.fit(
#         training_dataset,
#         epochs=training_epochs,
#         verbose=1,
#         callbacks=callback_list,
#         validation_data=val_dataset,
#         shuffle=True,
#         initial_epoch=initial_epoch,
#         steps_per_epoch=training_steps_per_epoch,
#         validation_steps=val_steps,
#         validation_freq=val_freq,
#         max_queue_size=10,
#         workers=8,
#         use_multiprocessing=True,
#     )
