from typing import Dict, List, Optional

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.python.distribute.tpu_strategy import TPUStrategy

from imagemodel.common.models.common_compile_options import CompileOptions


class Tester:
    def __init__(
            self,
            model: Model,
            compile_helper: CompileOptions,
            test_dataset: tf.data.Dataset,
            test_dataset_description: str,
            test_batch_size: int,
            strategy_optional: Optional[TPUStrategy] = None):
        self.model: Model = model
        self.compile_helper: CompileOptions = compile_helper
        self.test_dataset: tf.data.Dataset = test_dataset
        self.test_dataset_description: str = test_dataset_description
        self.test_batch_size: int = test_batch_size
        self.strategy_optional: Optional[TPUStrategy] = strategy_optional
        
        self.test_dataset = self.test_dataset.batch(self.test_batch_size, drop_remainder=True)
        self.test_dataset = self.test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        if self.strategy_optional:
            with self.strategy_optional.scope():
                self.model.compile(
                        optimizer=self.compile_helper.optimizer,
                        loss=self.compile_helper.loss_functions,
                        loss_weights=self.compile_helper.loss_weights_optional,
                        metrics=self.compile_helper.metrics)
        else:
            self.model.compile(
                    optimizer=self.compile_helper.optimizer,
                    loss=self.compile_helper.loss_functions,
                    loss_weights=self.compile_helper.loss_weights_optional,
                    metrics=self.compile_helper.metrics)
    
    def test(self, callbacks: List[Callback]) -> Dict[str, float]:
        return self.model.evaluate(
                self.test_dataset,
                batch_size=self.test_batch_size,
                verbose=1,
                callbacks=callbacks,
                max_queue_size=4,
                workers=4,
                use_multiprocessing=True,
                return_dict=True)
