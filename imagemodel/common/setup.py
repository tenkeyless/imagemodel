import os
import time
from typing import Callable, List, Optional

from image_keras.supports import create_folder_if_not_exist
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard

from imagemodel.common.utils.optional import optional_map


def training_experiment_id(model_name: str, run_id: str) -> str:
    return "training__model_{}__run_{}".format(model_name, run_id)


def test_experiment_id(model_name: str, run_id: str) -> str:
    return "test__model_{}__run_{}".format(model_name, run_id)


def predict_experiment_id(model_name: str, run_id: str) -> str:
    return "predict__model_{}__run_{}".format(model_name, run_id)


class ExperimentSetup:
    def __init__(
            self,
            result_base_folder: str,
            model_name: str,
            run_id: Optional[str],
            experiment_id_generator: Callable[[str, str], str]):
        self.result_base_folder: str = result_base_folder
        self.run_id: str = optional_map(run_id, lambda el: el.replace(" ", "_")) or ExperimentSetup.__get_run_id()
        self.experiment_id: str = experiment_id_generator(model_name, self.run_id)
    
    @staticmethod
    def __get_run_id() -> str:
        os.environ["TZ"] = "Asia/Seoul"
        time.tzset()
        return time.strftime("%Y%m%d_%H%M%S")


class TrainingExperimentSetup(ExperimentSetup):
    def __init__(
            self,
            result_base_folder: str,
            model_name: str,
            run_id: Optional[str],
            experiment_id_generator: Callable[[str, str], str] = training_experiment_id):
        super().__init__(
                result_base_folder=result_base_folder,
                model_name=model_name,
                run_id=run_id,
                experiment_id_generator=experiment_id_generator)
        
        # data folder
        self.base_data_folder: str = os.path.join(self.result_base_folder, "data")
        self.training_result_folder: str = os.path.join(self.base_data_folder, self.experiment_id)
        # create folder not gs
        if not self.training_result_folder.startswith("gs://"):
            create_folder_if_not_exist(self.training_result_folder)
        
        # save folder
        self.base_save_folder: str = os.path.join(self.result_base_folder, "save")
        save_models_folder: str = os.path.join(self.base_save_folder, "models")
        self.save_weights_folder: str = os.path.join(self.base_save_folder, "weights")
        save_tf_log_folder: str = os.path.join(self.base_save_folder, "tf_logs")
        self.tf_run_log_dir: str = os.path.join(save_tf_log_folder, self.experiment_id)
        # create folder not gs
        if not self.training_result_folder.startswith("gs://"):
            for folder in [save_models_folder, self.save_weights_folder, self.tf_run_log_dir]:
                create_folder_if_not_exist(folder)
    
    def setup_callbacks(
            self, training_epochs: int, without_early_stopping: bool, validation_freq: int) -> List[Callback]:
        model_checkpoint_cb: Callback = ModelCheckpoint(
                os.path.join(self.save_weights_folder, self.experiment_id + ".epoch_{epoch:02d}"),
                verbose=1)
        
        early_stopping_patience: int = training_epochs // (10 * validation_freq)
        early_stopping_cb: Callback = EarlyStopping(patience=early_stopping_patience, verbose=1)
        
        tensorboard_cb: Callback = TensorBoard(log_dir=self.tf_run_log_dir)
        
        callback_list: List[Callback] = [tensorboard_cb, model_checkpoint_cb]
        if not without_early_stopping:
            callback_list.append(early_stopping_cb)
        
        return callback_list


class PredictExperimentSetup(ExperimentSetup):
    def __init__(
            self,
            result_base_folder: str,
            model_name: str,
            run_id: Optional[str],
            experiment_id_generator: Callable[[str, str], str] = predict_experiment_id):
        super().__init__(
                result_base_folder=result_base_folder,
                model_name=model_name,
                run_id=run_id,
                experiment_id_generator=experiment_id_generator)
        
        # data folder
        self.base_data_folder: str = os.path.join(self.result_base_folder, "data")
        self.predict_result_folder: str = os.path.join(self.base_data_folder, self.experiment_id)
        self.save_result_images_folder: str = os.path.join(self.predict_result_folder, "images")
        
        # create folder not gs
        if not self.predict_result_folder.startswith("gs://"):
            for folder in [self.predict_result_folder, self.save_result_images_folder]:
                create_folder_if_not_exist(folder)
