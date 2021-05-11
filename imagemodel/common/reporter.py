import io
import os
import platform
from typing import TypeVar

from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

from imagemodel.common.predictor import Predictor
from imagemodel.common.setup import ExperimentSetup, PredictExperimentSetup, TrainingExperimentSetup
from imagemodel.common.trainer import Trainer
from imagemodel.common.utils.gc_storage import upload_blob
from imagemodel.common.utils.optional import optional_map

SE = TypeVar('SE', bound=ExperimentSetup)


class Reporter:
    def __init__(self, setup: SE):
        self.setup = setup
    
    @staticmethod
    def __save_str_to_file(path_filename: str, contents: str):
        f = open(path_filename, "w")
        f.write(contents)
        f.close()
    
    @staticmethod
    def __upload_file_to_google_storage(gs_folder_name: str, path_filename: str):
        bucket_name = gs_folder_name.replace("gs://", "").split("/")[0]
        folder_without_gs = gs_folder_name.replace("gs://", "")[gs_folder_name.replace("gs://", "").find("/") + 1:]
        upload_blob(bucket_name, path_filename, os.path.join(folder_without_gs, os.path.basename(path_filename)))
    
    @staticmethod
    def _get_model_summary(model: Model):
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_string = stream.getvalue()
        stream.close()
        return summary_string
    
    def _save_txt_gs_or_local(self, folder_name: str, tmp_path_filename: str, local_path_filename: str, content: str):
        # Upload to gs bucket
        if folder_name.startswith("gs://"):
            self.__save_str_to_file(tmp_path_filename, content)
            self.__upload_file_to_google_storage(folder_name, tmp_path_filename)
        # Save on local
        else:
            self.__save_str_to_file(local_path_filename, content)
    
    def report(self):
        pass
    
    def _plotmodel(self, model: Model, result_folder: str):
        # Model Structure
        # Upload to gs bucket
        if result_folder.startswith("gs://"):
            tmp_plot_model_img_path = "/tmp/model_{}.png".format(self.setup.run_id)
            plot_model(model, show_shapes=True, to_file=tmp_plot_model_img_path, dpi=144)
            self.__upload_file_to_google_storage(result_folder, tmp_plot_model_img_path)
        # Save on local
        else:
            tmp_plot_model_img_path = os.path.join(result_folder, "model.png")
            plot_model(model, show_shapes=True, to_file=tmp_plot_model_img_path, dpi=144)
        
        # Model Nested Structure
        # Upload to gs bucket
        if result_folder.startswith("gs://"):
            tmp_plot_model_nested_img_path = "/tmp/model_nested_{}.png".format(self.setup.run_id)
            plot_model(
                    model,
                    show_shapes=True,
                    to_file=tmp_plot_model_nested_img_path,
                    expand_nested=True,
                    dpi=144)
            self.__upload_file_to_google_storage(result_folder, tmp_plot_model_nested_img_path)
        # Save on local
        else:
            tmp_plot_model_img_path = os.path.join(result_folder, "model_nested.png")
            plot_model(
                    model,
                    show_shapes=True,
                    to_file=tmp_plot_model_img_path,
                    expand_nested=True,
                    dpi=144)
    
    def plotmodel(self):
        pass


class TrainerReporter(Reporter):
    def __init__(self, setup: TrainingExperimentSetup, trainer: Trainer):
        super().__init__(setup=setup)
        self.trainer = trainer
    
    def report(self):
        # Info
        info: str = """
# Information ---------------------------
Hostname: {}
Training ID: {}
Training Dataset: {}
Validation Dataset: {}
Tensorboard Log Folder: {}
Training Data Folder: {}/{}
-----------------------------------------
        """.format(
                platform.node(),
                self.setup.experiment_id,
                self.trainer.training_pipeline.data_description,
                optional_map(self.trainer.validation_pipeline_optional, lambda vp: vp.data_description) or "",
                self.setup.tf_run_log_dir,
                self.setup.base_data_folder,
                self.setup.experiment_id)
        print(info)
        tmp_info = "/tmp/info_{}.txt".format(self.setup.run_id)
        tmp_info_local = os.path.join(self.setup.training_result_folder, "info_{}.txt".format(self.setup.run_id))
        self._save_txt_gs_or_local(self.setup.training_result_folder, tmp_info, tmp_info_local, info)
        
        # Model
        model_summary_content: str = self._get_model_summary(self.trainer.model)
        tmp_model_summary = "/tmp/model_summary_{}.txt".format(self.setup.run_id)
        tmp_model_summary_local = os.path.join(
                self.setup.training_result_folder,
                "model_summary_{}.txt".format(self.setup.run_id))
        self._save_txt_gs_or_local(
                self.setup.training_result_folder,
                tmp_model_summary,
                tmp_model_summary_local,
                model_summary_content)
    
    def plotmodel(self):
        self._plotmodel(self.trainer.model, self.setup.training_result_folder)


class PredictorReporter(Reporter):
    def __init__(self, setup: PredictExperimentSetup, predictor: Predictor):
        super().__init__(setup=setup)
        self.predictor = predictor
    
    def report(self):
        # Info
        info: str = """
# Information ---------------------------
Hostname: {}
Predict ID: {}
Predict Dataset: {}
Predict Data Folder: {}/{}
-----------------------------------------
        """.format(
                platform.node(),
                self.setup.experiment_id,
                self.predictor.predict_pipeline.data_description,
                self.setup.base_data_folder,
                self.setup.experiment_id)
        print(info)
        tmp_info = "/tmp/info_{}.txt".format(self.setup.run_id)
        tmp_info_local = os.path.join(self.setup.predict_result_folder, "info_{}.txt".format(self.setup.run_id))
        self._save_txt_gs_or_local(self.setup.predict_result_folder, tmp_info, tmp_info_local, info)
        
        # Model
        model_summary_content: str = self._get_model_summary(self.predictor.model)
        tmp_model_summary = "/tmp/model_summary_{}.txt".format(self.setup.run_id)
        tmp_model_summary_local = os.path.join(
                self.setup.predict_result_folder,
                "model_summary_{}.txt".format(self.setup.run_id))
        self._save_txt_gs_or_local(
                self.setup.predict_result_folder,
                tmp_model_summary,
                tmp_model_summary_local,
                model_summary_content)
    
    def plotmodel(self):
        self._plotmodel(self.predictor.model, self.setup.predict_result_folder)
