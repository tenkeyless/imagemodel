from typing import Optional

import tensorflow as tf
from keras.callbacks import History
from tensorflow.keras.models import Model

from imagemodel.binary_segmentations.models.common_compile_options import CompileOptions
from imagemodel.binary_segmentations.models.common_model_manager import CommonModelManager
from imagemodel.common.datasets.pipeline import Pipeline
from imagemodel.common.utils.optional import optional_map


class Trainer:
    def __init__(
            self,
            unet_level_model_manager: CommonModelManager,
            compile_helper: CompileOptions,
            training_pipeline: Pipeline,
            validation_pipeline: Optional[Pipeline] = None):
        self.model_manager: CommonModelManager = unet_level_model_manager
        self.compile_helper: CompileOptions = compile_helper
        self.training_pipeline: Pipeline = training_pipeline
        self.validation_pipeline: Optional[Pipeline] = validation_pipeline

        self.model: Model = self.model_manager.setup_model()
        self.training_dataset: tf.data.Dataset = self.training_pipeline.get_zipped_dataset().batch(4)
        self.validation_dataset_optional: Optional[tf.data.Dataset] = optional_map(
            self.validation_pipeline,
            lambda el: el.get_zipped_dataset().batch(4))

        self.model.compile(
            optimizer=self.compile_helper.optimizer,
            loss=self.compile_helper.loss_functions,
            metrics=self.compile_helper.metrics)

    def fit(self) -> History:
        return self.model.fit(
            self.training_dataset,
            epochs=10,
            verbose=1,
            validation_data=self.validation_dataset_optional,
            shuffle=True,
            initial_epoch=0,
            steps_per_epoch=500,
            validation_steps=500,
            validation_freq=1,
            max_queue_size=10,
            workers=8,
            use_multiprocessing=True)
