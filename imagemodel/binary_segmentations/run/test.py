import os
import sys

sys.path.append(os.getcwd())

import platform
from argparse import ArgumentParser, RawTextHelpFormatter
from typing import List, Tuple

import tensorflow as tf
from image_keras.supports.folder import create_folder_if_not_exist
from keras.utils import plot_model
from imagemodel.binary_segmentations.configs.datasets import Datasets
from imagemodel.binary_segmentations.configs.losses import Losses
from imagemodel.binary_segmentations.configs.metrics import Metrics
from imagemodel.binary_segmentations.configs.optimizers import Optimizers
from imagemodel.binary_segmentations.run.common import get_run_id, loss_coords

if __name__ == "__main__":
    # 1. Variables --------
    # 1-1) Variables with Parser
    parser: ArgumentParser = ArgumentParser(
        description="Arguments for Test on Semantic Segmentation",
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
        "--result_base_folder",
        type=str,
        required=True,
        help="Name of base folder. 'segmentations/training', '/user/name/path/folder'.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for test. Defaults to 8. \
            ex) 8",
    )
    parser.add_argument(
        "--model_weight_path",
        required=True,
        type=str,
        help="Model to be tested. Full path of TF model which is accessable on cloud bucket. ex) 'gs://cell_dataset/save/weights/training__model_unet_l4__run_leetaekyu_20210108_221742.epoch_78-val_loss_0.179-val_accuracy_0.974'",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        help="(Without space) Run with custom id. Be careful not to use duplicate IDs. If not specified, time is used as ID. ex) 'leetaekyu_210108_185302'",
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

    command_str = list(map(lambda el: str(el), sys.argv[:]))
    list_to_command_str = " ".join(map(str, command_str))
    parsed_args_str = ""
    for arg in list(vars(args))[:-1]:
        parsed_args_str += "- {}: {}\n".format(arg, getattr(args, arg))
    parsed_args_str += "- {}: {}".format(
        list(vars(args))[-1], getattr(args, list(vars(args))[-1])
    )

    # 1-2) Get variables
    # required
    model_name: str = args.model_name
    result_base_folder: str = args.result_base_folder
    model_weight_path: str = args.model_weight_path

    # optional with default value
    dataset: str = args.dataset or Datasets.get_default()
    batch_size: int = args.batch_size or 8
    run_id: str = args.run_id or get_run_id()

    # optimizer, losses, metrics
    optimizer: str = args.optimizer or Optimizers.get_default()
    loss_weight_tuple_list: List[Tuple[str, float]] = args.losses or [
        (Losses.get_default(), 1.0)
    ]
    metrics_list: List[List[str]] = args.metrics or [[Metrics.get_default()]]

    # processing
    test_batch_size: int = batch_size
    run_id = run_id.replace(" ", "_")
    test_id: str = "test__model_{}__run_{}".format(model_name, run_id)

    # 2. Setup --------
    # 2-1) Setup folders for Result and TensorBoard Log.
    # data folder
    base_data_folder: str = os.path.join(result_base_folder, "data")
    test_result_folder: str = os.path.join(base_data_folder, test_id)
    create_folder_if_not_exist(test_result_folder)

    # 2-2) Setup for dataset.
    datasets = Datasets(dataset)
    dataset_interface = datasets.get_dataset()
    if dataset_interface is None:
        raise ValueError("`dataset` should be exist.")
    test_dataset = dataset_interface.get_test_dataset(
        batch_size_optional=test_batch_size
    )

    # 2-3) Report setup results
    info: str = """
# Information ---------------------------
Hostname: {}
Test ID: {}
Test Dataset: {}
Test Data Folder: {}/{}
-----------------------------------------

# Command -------------------------------
{}
-----------------------------------------

# Parsed Arguments ----------------------
{}
-----------------------------------------
""".format(
        platform.node(),
        test_id,
        datasets.value,
        base_data_folder,
        test_id,
        list_to_command_str,
        parsed_args_str,
    )
    print(info)
    tmp_info = os.path.join(test_result_folder, "info_{}.txt".format(run_id))
    f = open(tmp_info, "w")
    f.write(info)
    f.close()

    # 3. Model compile --------
    model = tf.keras.models.load_model(model_weight_path)

    model_optimizer = Optimizers(optimizer).get_optimizer()
    model_loss_list = list(
        map(lambda el: Losses(el[0]).get_loss(), loss_weight_tuple_list)
    )
    model_loss_weight_list = list(map(lambda el: el[1], loss_weight_tuple_list))
    model_metrics_list = [
        list(filter(lambda v: v, [Metrics(metric).get_metric() for metric in metrics]))
        for metrics in metrics_list
    ]
    output_keys = model.output_names
    if len(loss_weight_tuple_list) != len(output_keys):
        raise ValueError(
            "Number of `--losses` option(s) should be {}.".format(len(output_keys))
        )
    if len(metrics_list) != len(output_keys):
        raise ValueError(
            "Number of `--metrics` option(s) should be {}.".format(len(output_keys))
        )
    model_loss_dict = dict(zip(output_keys, model_loss_list))
    model_loss_weight_dict = dict(zip(output_keys, model_loss_weight_list))
    model_metrics_dict = dict(zip(output_keys, model_metrics_list))
    model.compile(
        optimizer=model_optimizer,
        loss=model_loss_dict,
        loss_weights=model_loss_weight_dict,
        metrics=model_metrics_dict,
    )

    # model plot
    tmp_plot_model_img_path = os.path.join(test_result_folder, "model.png")
    plot_model(
        model,
        show_shapes=True,
        to_file=tmp_plot_model_img_path,
        dpi=144,
    )

    # model plot
    tmp_plot_model_img_path = os.path.join(test_result_folder, "model_nested.png")
    plot_model(
        model,
        show_shapes=True,
        to_file=tmp_plot_model_img_path,
        expand_nested=True,
        dpi=144,
    )

    # model summary
    tmp_plot_model_txt_path = os.path.join(test_result_folder, "model.txt")
    with open(tmp_plot_model_txt_path, "w") as fh:
        model.summary(print_fn=lambda x: fh.write(x + "\n"))

    # 4. Test --------
    # ValueError: too many values to unpack (expected 2)
    test_loss, test_acc = model.evaluate(
        test_dataset, workers=8, use_multiprocessing=True
    )

    result: str = """
# Result ---------------------------
Test Loss: {}
Test Accuracy: {}
-----------------------------------------
""".format(
        test_loss, test_acc
    )
    print(result)
    tmp_info = os.path.join(test_result_folder, "result_{}.txt".format(run_id))
    f = open(tmp_info, "w")
    f.write(result)
    f.close()
