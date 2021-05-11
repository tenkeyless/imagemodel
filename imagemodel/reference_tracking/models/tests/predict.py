from argparse import ArgumentParser, RawTextHelpFormatter
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.distribute.tpu_strategy import TPUStrategy

import _path  # noqa
from imagemodel.common.predictor import Predictor
from imagemodel.common.reporter import PredictorReporter
from imagemodel.common.setup import PredictExperimentSetup, predict_experiment_id
from imagemodel.common.utils.common_tpu import create_tpu, delete_tpu, tpu_initialize
from imagemodel.reference_tracking.configs.datasets import Datasets

# noinspection DuplicatedCode
if __name__ == "__main__":
    """
    Examples
    --------
    # With CPU, GPU
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
    ...     imagemodel/tkl:1.0
    >>> python imagemodel/reference_tracking/models/tests/predict.py \
    ...     --model_name ref_local_tracking_model_031 \
    ...     --model_weight_path saved/training__model_ref_local_tracking_model_031_mh__run_reference_tracking__20210511_063754.epoch_12 \
    ...     --run_id reference_tracking__20210511_133606 \
    ...     --result_base_folder gs://cell_dataset \
    ...     --predict_pipeline rt_gs_cell_sample_test_1 \
    ...     --batch_size 1

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
    ...     imagemodel_tpu/tkl:1.0
    >>> python imagemodel/reference_tracking/models/tests/predict.py \
    ...     --model_name ref_local_tracking_model_031 \
    ...     --model_weight_path saved/training__model_ref_local_tracking_model_031_mh__run_reference_tracking__20210511_063754.epoch_12 \
    ...     --run_id reference_tracking__20210511_133606 \
    ...     --result_base_folder gs://cell_dataset \
    ...     --predict_pipeline rt_gs_cell_sample_test_1 \
    ...     --batch_size 1
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
    parser.add_argument("--predict_pipeline", type=str, required=True)
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
    # training related
    run_id: Optional[str] = args.run_id
    result_base_folder: str = args.result_base_folder
    # dataset related
    predict_pipeline: str = args.predict_pipeline
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
    rt_predict_pipeline = Datasets(predict_pipeline).get_pipeline(resize_to=(256, 256))
    
    # Trainer Setup
    predictor = Predictor(
            model=model,
            predict_pipeline=rt_predict_pipeline,
            predict_batch_size=batch_size,
            strategy_optional=strategy_optional)
    
    # Report
    reporter = PredictorReporter(setup=experiment_setup, predictor=predictor)
    reporter.report()
    reporter.plotmodel()
    
    # Predict
    predictor.predict()
    
    if tpu_name_optional:
        delete_tpu(tpu_name=tpu_name_optional, ctpu_zone=ctpu_zone)
