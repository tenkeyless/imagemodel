import os
import sys

sys.path.append(os.getcwd())

import platform
from argparse import ArgumentParser, RawTextHelpFormatter

import tensorflow as tf
from image_keras.supports.folder import create_folder_if_not_exist
from keras.utils import plot_model
from segmentations.configs.datasets import Datasets
from segmentations.models.model import Models
from segmentations.run.common import get_run_id

if __name__ == "__main__":
    # 1. Variables --------
    # 1-1) Variables with Parser
    parser: ArgumentParser = ArgumentParser(
        description="Arguments for Prediction dataset on Semantic Segmentation",
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
    batch_size: int = args.batch_size or 1
    run_id: str = args.run_id or get_run_id()

    # processing
    predict_testset_batch_size: int = batch_size
    run_id = run_id.replace(" ", "_")
    predict_testset_id: str = "predict_testset__model_{}__run_{}".format(
        model_name, run_id
    )

    # 1-3) Variable Check
    models = Models(model_name).get_model()
    post_processing = models.post_processing
    saved_post_processed_result = models.saved_post_processed_result

    # 2. Setup --------
    # 2-1) Setup folders for Result.
    # data folder
    base_data_folder: str = os.path.join(result_base_folder, "data")
    predict_testset_result_folder: str = os.path.join(
        base_data_folder, predict_testset_id
    )
    predict_testset_result_image_folder: str = os.path.join(
        predict_testset_result_folder, "images"
    )
    create_folder_if_not_exist(predict_testset_result_image_folder)

    # 2-2) Setup for dataset.
    datasets = Datasets(dataset)
    dataset_interface = datasets.get_dataset()
    test_dataset = dataset_interface.get_test_dataset(
        batch_size_optional=predict_testset_batch_size
    )
    test_dataset_filenames = dataset_interface.get_test_dataset_filenames()

    # 2-3) Report setup results
    info: str = """
# Information ---------------------------
Hostname: {}
Predict testset ID: {}
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
        predict_testset_id,
        datasets.value,
        base_data_folder,
        predict_testset_id,
        list_to_command_str,
        parsed_args_str,
    )
    print(info)
    tmp_info = os.path.join(predict_testset_result_folder, "info_{}.txt".format(run_id))
    f = open(tmp_info, "w")
    f.write(info)
    f.close()

    # 3. Model compile --------
    model = tf.keras.models.load_model(model_weight_path)

    # model plot
    tmp_plot_model_img_path = os.path.join(predict_testset_result_folder, "model.png")
    plot_model(
        model,
        show_shapes=True,
        to_file=tmp_plot_model_img_path,
        dpi=144,
    )

    # model plot
    tmp_plot_model_img_path = os.path.join(
        predict_testset_result_folder, "model_nested.png"
    )
    plot_model(
        model,
        show_shapes=True,
        to_file=tmp_plot_model_img_path,
        expand_nested=True,
        dpi=144,
    )

    # model summary
    tmp_plot_model_txt_path = os.path.join(predict_testset_result_folder, "model.txt")
    with open(tmp_plot_model_txt_path, "w") as fh:
        model.summary(print_fn=lambda x: fh.write(x + "\n"))

    # 4. Predict --------
    predicted = model.predict(
        test_dataset,
        batch_size=predict_testset_batch_size,
        verbose=1,
        max_queue_size=1,
    )

    filename_predicted_list = tuple(zip(list(test_dataset_filenames), predicted))
    total_length = len(filename_predicted_list)
    zpad = len(str(total_length))
    for index, filename_predicted in enumerate(filename_predicted_list):
        filename = filename_predicted[0].numpy().decode()
        predicted = filename_predicted[1]

        print(
            "({}/{}) Post processing and save {}... \r".format(
                str(index + 1).zfill(zpad), str(total_length).zfill(zpad), filename
            ),
            end="",
        )

        post_processed_result = post_processing(predicted)
        saved_post_processed_result(
            os.path.join(predict_testset_result_image_folder, filename),
            post_processed_result,
        )
    print("\n")
