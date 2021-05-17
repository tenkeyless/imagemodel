from argparse import ArgumentParser, RawTextHelpFormatter
from typing import Callable, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import losses, metrics, optimizers
from tensorflow.keras.models import Model
from tensorflow.python.distribute.tpu_strategy import TPUStrategy

import _path  # noqa
from imagemodel.common.datasets.pipeline import Pipeline
from imagemodel.common.models.common_compile_options import CompileOptions
from imagemodel.common.setup import TestExperimentSetup, test_experiment_id
from imagemodel.common.utils.common_tpu import create_tpu, delete_tpu, tpu_initialize
from imagemodel.common.utils.gpu_check import check_first_gpu
from imagemodel.reference_tracking.configs.datasets import Datasets
from imagemodel.reference_tracking.datasets.cell_tracking.preprocessor import RTCellTrackingPreprocessor
from imagemodel.reference_tracking.datasets.pipeline import RTPipeline
from imagemodel.reference_tracking.datasets.rt_augmenter import RTAugmenter
from imagemodel.reference_tracking.datasets.rt_preprocessor import RTPreprocessor
from imagemodel.reference_tracking.datasets.rt_regularizer import BaseRTRegularizer, RTRegularizer
from imagemodel.reference_tracking.models.testers.rt_test_reporter import RTTestReporter
from imagemodel.reference_tracking.models.testers.tester import Tester

check_first_gpu()

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
    >>> python imagemodel/reference_tracking/models/testers/test.py \
    ...     --model_name ref_local_tracking_model_031_mh \
    ...     --result_base_folder /reference_tracking_results \
    ...     --training_epochs 100 \
    ...     --validation_freq 1 \
    ...     --training_pipeline rt_cell_tracking_training_2 \
    ...     --validation_pipeline rt_cell_tracking_validation_2 \
    ...     --run_id binary_segmentations__unet_level_test__20210510_2120915 \
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
    >>> python imagemodel/reference_tracking/models/testers/test.py \
    ...     --model_name ref_local_tracking_model_031_mh \
    ...     --model_weight_path /reference_tracking_results/save/weights/\
    ... training__model_ref_local_tracking_model_031_mh__run_reference_tracking__20210513_194150.epoch_23 \
    ...     --run_id reference_tracking__20210513_101705 \
    ...     --result_base_folder /reference_tracking_results \
    ...     --test_pipeline rt_cell_sample_test_1 \
    ...     --batch_size 2
# training__model_ref_local_tracking_model_031_mh__run_reference_tracking__20210514_025537.epoch_14
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
    >>> python imagemodel/reference_tracking/models/tester/test.py \
    ...     --model_name ref_local_tracking_model_031_mh \
    ...     --result_base_folder gs://cell_dataset \
    ...     --training_epochs 100 \
    ...     --validation_freq 1 \
    ...     --training_pipeline rt_gs_cell_tracking_training_2 \
    ...     --validation_pipeline rt_gs_cell_tracking_validation_2 \
    ...     --run_id reference_tracking__20210511_055305 \
    ...     --without_early_stopping \
    ...     --batch_size 8 \
    ...     --ctpu_zone us-central1-b \
    ...     --tpu_name leetaekyu-1-trainer
    """
    # Argument Parsing
    parser: ArgumentParser = ArgumentParser(
            description="Arguments for Test in Reference Tracking",
            formatter_class=RawTextHelpFormatter)
    # model related
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_color_image", action="store_true")
    # trained model related
    parser.add_argument("--model_weight_path", required=True, type=str)
    # test related
    parser.add_argument("--run_id", type=str)
    parser.add_argument("--result_base_folder", type=str, required=True)
    # dataset related
    parser.add_argument("--test_pipeline", type=str, required=True)
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
    # test related
    run_id: Optional[str] = args.run_id
    result_base_folder: str = args.result_base_folder
    # dataset related
    test_pipeline: str = args.test_pipeline
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
    experiment_setup = TestExperimentSetup(
            result_base_folder=result_base_folder,
            model_name=model_name,
            run_id=run_id,
            experiment_id_generator=test_experiment_id)
    
    # Model Setup
    input_shape: Tuple[int, int, int] = (256, 256, 3) if input_color_image else (256, 256, 1)
    if tpu_name_optional:
        with strategy_optional.scope():
            model: Model = tf.keras.models.load_model(model_weight_path)
    else:
        model: Model = tf.keras.models.load_model(model_weight_path)
    
    # Output - [Main BW Mask, Ref BW Mask, Main Color Bin Label]
    if tpu_name_optional:
        with strategy_optional.scope():
            helper = CompileOptions(
                    optimizer=optimizers.Adam(lr=1e-4),
                    loss_functions=[losses.BinaryCrossentropy(name="main_u-net_binary_crossentropy"),
                                    losses.BinaryCrossentropy(name="ref_u-net_binary_crossentropy"),
                                    losses.CategoricalCrossentropy(name="main_label_categorical_crossentropy")],
                    # loss_weights_optional=[0.1, 0.1, 0.8],
                    loss_weights_optional=[0.25, 0.25, 0.5],
                    metrics=[[metrics.BinaryAccuracy(name="main_u-net_binary_accuracy")],
                             [metrics.BinaryAccuracy(name="ref_u-net_binary_accuracy")],
                             [metrics.CategoricalAccuracy(name="main_label_categorical_accuracy")]])
    else:
        helper = CompileOptions(
                optimizer=optimizers.Adam(lr=1e-4),
                loss_functions=[losses.BinaryCrossentropy(name="main_u-net_binary_crossentropy"),
                                losses.BinaryCrossentropy(name="ref_u-net_binary_crossentropy"),
                                losses.CategoricalCrossentropy(name="main_label_categorical_crossentropy")],
                # loss_weights_optional=[0.1, 0.1, 0.8],
                loss_weights_optional=[0.25, 0.25, 0.5],
                metrics=[[metrics.BinaryAccuracy(name="main_u-net_binary_accuracy")],
                         [metrics.BinaryAccuracy(name="ref_u-net_binary_accuracy")],
                         [metrics.CategoricalAccuracy(name="main_label_categorical_accuracy")]])
    
    # Dataset Setup
    # rt_test_pipeline = Datasets(test_pipeline).get_pipeline(resize_to=(256, 256))
    
    # Dataset Setup
    feeder = Datasets(test_pipeline).get_feeder()
    regularizer_func: Callable[[RTAugmenter], RTRegularizer] = \
        lambda el_bs_augmenter: BaseRTRegularizer(el_bs_augmenter, (256, 256))
    preprocessor_func: Callable[[RTRegularizer], RTPreprocessor] = \
        lambda el_rt_augmenter: RTCellTrackingPreprocessor(el_rt_augmenter, bin_size=30, cache_inout=False)
    rt_test_pipeline: Pipeline = RTPipeline(
            feeder,
            regularizer_func=regularizer_func,
            preprocessor_func=preprocessor_func)
    rt_test_dataset: tf.data.Dataset = rt_test_pipeline.get_zipped_dataset()
    rt_test_dataset_description: str = rt_test_pipeline.data_description
    
    # Trainer Setup
    tester = Tester(
            model=model,
            compile_helper=helper,
            strategy_optional=strategy_optional,
            test_dataset=rt_test_dataset,
            test_dataset_description=rt_test_dataset_description,
            test_batch_size=batch_size)
    
    # Report
    reporter = RTTestReporter(setup=experiment_setup, tester=tester)
    reporter.report()
    reporter.plotmodel()
    
    # Test
    test_result = tester.test(callbacks=[])
    
    reporter.report_result(test_result)
    
    if tpu_name_optional:
        delete_tpu(tpu_name=tpu_name_optional, ctpu_zone=ctpu_zone)
