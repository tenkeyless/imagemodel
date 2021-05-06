from typing import List, Optional

import tensorflow as tf
from tensorflow.keras.callbacks import Callback, History
from tensorflow.keras.models import Model
from tensorflow.python.distribute.tpu_strategy import TPUStrategy

from imagemodel.common.models.common_compile_options import CompileOptions
from imagemodel.common.models.common_model_manager import CommonModelManager
from imagemodel.common.datasets.pipeline import Pipeline
from imagemodel.common.utils.optional import optional_map


class Trainer:
    def __init__(
            self,
            model_manager: CommonModelManager,
            compile_helper: CompileOptions,
            training_pipeline: Pipeline,
            training_batch_size: int,
            strategy_optional: Optional[TPUStrategy] = None,
            training_shuffle_in_buffer: bool = False,
            training_shuffle_buffer_size: Optional[int] = None,
            training_shuffle_buffer_seed: int = 42,
            validation_pipeline: Optional[Pipeline] = None,
            validation_batch_size: int = 4,
            validation_freq: int = 1):
        self.model_manager: CommonModelManager = model_manager
        self.compile_helper: CompileOptions = compile_helper
        self.training_pipeline: Pipeline = training_pipeline
        self.training_batch_size: int = training_batch_size
        self.strategy_optional: Optional[TPUStrategy] = strategy_optional
        self.validation_pipeline_optional: Optional[Pipeline] = validation_pipeline
        self.validation_batch_size: int = validation_batch_size
        self.validation_freq: int = validation_freq
        
        self.training_dataset: tf.data.Dataset = self.training_pipeline.get_zipped_dataset()
        self.training_dataset_num: int = len(self.training_dataset)
        if training_shuffle_in_buffer:
            training_shuffle_buffer_size = training_shuffle_buffer_size or self.training_dataset_num
            self.training_dataset = self.training_dataset.shuffle(
                    buffer_size=training_shuffle_buffer_size,
                    seed=training_shuffle_buffer_seed,
                    reshuffle_each_iteration=True)
        self.training_dataset = self.training_dataset.repeat().batch(self.training_batch_size, drop_remainder=True)
        self.training_dataset = self.training_dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)
        
        self.validation_dataset_optional: Optional[tf.data.Dataset] = optional_map(
                self.validation_pipeline_optional, lambda el: el.get_zipped_dataset())
        self.validation_dataset_num: int = optional_map(self.validation_dataset_optional, len) or 0
        self.validation_dataset_optional = optional_map(
                self.validation_dataset_optional,
                lambda el: el.batch(self.validation_batch_size, drop_remainder=True))
        self.validation_dataset_optional = optional_map(
                self.validation_dataset_optional,
                lambda el: el.cache().prefetch(tf.data.experimental.AUTOTUNE))
        
        if self.strategy_optional:
            with self.strategy_optional.scope():
                self.model: Model = self.model_manager.setup_model()
                self.model.compile(
                        optimizer=self.compile_helper.optimizer,
                        loss=self.compile_helper.loss_functions,
                        metrics=self.compile_helper.metrics)
        else:
            self.model: Model = self.model_manager.setup_model()
            self.model.compile(
                    optimizer=self.compile_helper.optimizer,
                    loss=self.compile_helper.loss_functions,
                    metrics=self.compile_helper.metrics)
    
    def fit(self, training_epochs: int, callbacks: List[Callback]) -> History:
        return self.model.fit(
                self.training_dataset,
                epochs=training_epochs,
                verbose=1,
                callbacks=callbacks,
                validation_data=self.validation_dataset_optional,
                shuffle=True,
                initial_epoch=0,
                steps_per_epoch=self.training_dataset_num // self.training_batch_size,
                validation_steps=self.validation_dataset_num // self.validation_batch_size,
                validation_freq=self.validation_freq,
                max_queue_size=10,
                workers=8,
                use_multiprocessing=True)
