from argparse import ArgumentParser, RawTextHelpFormatter
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.distribute.tpu_strategy import TPUStrategy

import _path  # noqa
from imagemodel.common.reporter import PredictorReporter
from imagemodel.common.setup import PredictExperimentSetup, predict_experiment_id
from imagemodel.common.utils.common_tpu import create_tpu, delete_tpu, tpu_initialize
from imagemodel.common.utils.gpu_check import check_first_gpu
from imagemodel.experimental.reference_tracking.dataset_providers.cell_tracking_dataset. \
    rt_cell_tracking_provider_p import RTCellTrackingProviderP
from imagemodel.experimental.reference_tracking.dataset_providers.rt_provider import RTProviderP
from imagemodel.experimental.reference_tracking.models.testers.rt_provider_predictor import RTProviderPredictor

check_first_gpu()

# noinspection DuplicatedCode
if __name__ == "__main__":
    """
    Examples
    --------
    # With CPU (O)
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
    >>> python imagemodel/experimental/reference_tracking/models/testers/provider_predict.py \
    ...     --model_name ref_local_tracking_model_031_mh \
    ...     --model_weight_path saved/\
    ... training__model_ref_local_tracking_model_031_mh__run_reference_tracking__20210511_063754.epoch_23 \
    ...     --run_id reference_tracking__20210513_050631 \
    ...     --result_base_folder /reference_tracking_results \
    ...     --predict_pipeline rt_cell_sample_2_test_1 \
    ...     --batch_size 4
    
    # With GPU (O)
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
    >>> python imagemodel/experimental/reference_tracking/models/testers/provider_predict.py \
    ...     --model_name ref_local_tracking_model_031 \
    ...     --model_weight_path /reference_tracking_results/save/weights/\
    ... training__model_ref_local_tracking_model_031__run_reference_tracking__20210517_163254.epoch_02 \
    ...     --run_id reference_tracking__20210517_170230 \
    ...     --result_base_folder /reference_tracking_results \
    ...     --predict_base_folder /data/tracking_test2 \
    ...     --predict_filename_folder /data/tracking_test2/framed_sample \
    ...     --batch_size 2

    # With TPU (X)
    
    >>> docker run \
    ...     -it \
    ...     --rm \
    ...     -u $(id -u):$(id -g) \
    ...     -v /etc/localtime:/etc/localtime:ro \
    ...     -v ~/.config:/.config \
    ...     -v ~/.local:/.local \
    ...     -v $(pwd):/imagemodel \
    ...     --workdir="/imagemodel" \
    ...     imagemodel_tpu/tkl:1.0
    >>> python imagemodel/experimental/reference_tracking/models/testers/predict.py \
    ...     --model_name ref_local_tracking_model_031_mh \
    ...     --model_weight_path gs://cell_dataset/save/weights/\
    ... training__model_ref_local_tracking_model_031_mh__run_reference_tracking__20210511_063754.epoch_23 \
    ...     --run_id reference_tracking__20210513_052931 \
    ...     --result_base_folder gs://cell_dataset \
    ...     --predict_pipeline rt_gs_cell_sample_2_test_1 \
    ...     --batch_size 4 \
    ...     --ctpu_zone us-central1-b \
    ...     --tpu_name leetaekyu-1-trainer
    """
    # Argument Parsing
    parser: ArgumentParser = ArgumentParser(
            description="Arguments predicts in Reference Tracking",
            formatter_class=RawTextHelpFormatter)
    # model related
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_color_image", action="store_true")
    # trained model related
    parser.add_argument("--model_weight_path", required=True, type=str)
    # predict related
    parser.add_argument("--run_id", type=str)
    parser.add_argument("--result_base_folder", type=str, required=True)
    # dataset related
    parser.add_argument("--predict_filename_folder", type=str)
    parser.add_argument("--predict_base_folder", type=str, required=True)
    parser.add_argument("--batch_size", type=int)
    # tpu related
    parser.add_argument("--ctpu_zone", type=str, help="VM, TPU zone. ex) 'us-central1-b'")
    parser.add_argument("--tpu_name", type=str, help="TPU name. ex) 'leetaekyu-1-trainer'")
    
    args = parser.parse_args()
    # model related
    model_name: str = args.model_name
    input_color_image: bool = args.input_color_image
    # trained model related
    model_weight_path: str = args.model_weight_path
    # predict related
    run_id: Optional[str] = args.run_id
    result_base_folder: str = args.result_base_folder
    predict_filename_folder: Optional[str] = args.predict_filename_folder
    # dataset related
    predict_base_folder: str = args.predict_base_folder
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
    experiment_setup = PredictExperimentSetup(
            result_base_folder=result_base_folder,
            model_name=model_name,
            run_id=run_id,
            experiment_id_generator=predict_experiment_id)
    
    # Model Setup
    input_shape: Tuple[int, int, int] = (256, 256, 3) if input_color_image else (256, 256, 1)
    if tpu_name_optional:
        with strategy_optional.scope():
            model: Model = tf.keras.models.load_model(model_weight_path)
    else:
        model: Model = tf.keras.models.load_model(model_weight_path)
    
    # Dataset Setup
    rt_predict_provider: RTProviderP = RTCellTrackingProviderP(
            filename_folder=predict_filename_folder,
            base_folder=predict_base_folder,
            shuffle=False,
            random_seed=42,
            bin_size=30,
            resize_to=(256, 256))
    rt_predict_dataset: tf.data.Dataset = rt_predict_provider.get_output_dataset()
    rt_predict_dataset_description: str = rt_predict_provider.data_description
    
    
    def combine_folder_file(a, b):
        return a + "/" + b
    
    
    def post_processing(predicted_current_bin_label: tf.Tensor, bin_color_map: tf.Tensor, current_filenames: tf.Tensor):
        current_arg_max_bin = tf.argmax(predicted_current_bin_label, axis=-1)
        current_label = tf.gather(bin_color_map, current_arg_max_bin, axis=1, batch_dims=1)
        
        for index, current_filename in enumerate(current_filenames):
            current_folder_filename = combine_folder_file(experiment_setup.save_result_images_folder, current_filename)
            img = tf.image.encode_png(tf.cast(current_label[index], tf.uint8))
            tf.io.write_file(current_folder_filename, img)
    
    
    # Trainer Setup
    predictor = RTProviderPredictor(
            model=model,
            predict_dataset=rt_predict_dataset,
            predict_dataset_description=rt_predict_dataset_description,
            predict_batch_size=batch_size,
            strategy_optional=strategy_optional,
            post_processing=post_processing)
    
    # Report
    reporter = PredictorReporter(setup=experiment_setup, predictor=predictor)
    reporter.report()
    reporter.plotmodel()
    
    # Predict
    predictor.predict()
    
    if tpu_name_optional:
        delete_tpu(tpu_name=tpu_name_optional, ctpu_zone=ctpu_zone)
