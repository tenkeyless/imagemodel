import os
import platform

from tensorflow.python.keras.utils.vis_utils import plot_model

from imagemodel.common.setup import ExperimentSetup
from imagemodel.common.trainer import Trainer
from imagemodel.common.utils.optional import optional_map


class Reporter:
    def __init__(self, setup: ExperimentSetup, trainer: Trainer):
        self.setup = setup
        self.trainer = trainer

    def report(self):
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
            self.setup.training_id,
            self.trainer.training_pipeline.data_description,
            optional_map(self.trainer.validation_pipeline_optional, lambda vp: vp.data_description) or "",
            self.setup.tf_run_log_dir,
            self.setup.base_data_folder,
            self.setup.training_id
        )
        print(info)
        tmp_info = os.path.join(self.setup.training_result_folder, "info_{}.txt".format(self.setup.run_id))
        f = open(tmp_info, "w")
        f.write(info)
        f.close()

    def plotmodel(self):
        tmp_plot_model_img_path = os.path.join(self.setup.training_result_folder, "model.png")
        plot_model(self.trainer.model, show_shapes=True, to_file=tmp_plot_model_img_path, dpi=144)

        tmp_plot_model_img_path = os.path.join(self.setup.training_result_folder, "model_nested.png")
        plot_model(self.trainer.model, show_shapes=True, to_file=tmp_plot_model_img_path, expand_nested=True, dpi=144)
