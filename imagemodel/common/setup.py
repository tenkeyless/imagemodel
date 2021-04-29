import os
from typing import List

from image_keras.supports import create_folder_if_not_exist
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard


class ExperimentSetup:
    def __init__(self, result_base_folder: str, training_id: str, run_id: str):
        self.result_base_folder = result_base_folder
        self.training_id = training_id
        self.run_id = run_id

        # data folder
        self.base_data_folder: str = os.path.join(self.result_base_folder, "data")
        self.training_result_folder: str = os.path.join(self.base_data_folder, self.training_id)
        create_folder_if_not_exist(self.training_result_folder)

        # save folder
        self.base_save_folder: str = os.path.join(self.result_base_folder, "save")
        save_models_folder: str = os.path.join(self.base_save_folder, "models")
        self.save_weights_folder: str = os.path.join(self.base_save_folder, "weights")
        save_tf_log_folder: str = os.path.join(self.base_save_folder, "tf_logs")
        self.tf_run_log_dir: str = os.path.join(save_tf_log_folder, self.training_id)
        for folder in [save_models_folder, self.save_weights_folder, self.tf_run_log_dir]:
            create_folder_if_not_exist(folder)

    def setup_callbacks(
            self,
            training_epochs: int,
            without_early_stopping: bool,
            validation_freq: int) -> List[Callback]:
        model_checkpoint_cb: Callback = ModelCheckpoint(
            os.path.join(self.save_weights_folder, self.training_id + ".epoch_{epoch:02d}"),
            verbose=1)

        early_stopping_patience: int = training_epochs // (10 * validation_freq)
        early_stopping_cb: Callback = EarlyStopping(
            patience=early_stopping_patience, verbose=1
        )

        tensorboard_cb: Callback = TensorBoard(log_dir=self.tf_run_log_dir)

        callback_list: List[Callback] = [tensorboard_cb, model_checkpoint_cb]
        if not without_early_stopping:
            callback_list.append(early_stopping_cb)

        return callback_list
