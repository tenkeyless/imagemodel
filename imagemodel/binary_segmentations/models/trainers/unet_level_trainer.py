from tensorflow.keras import optimizers, losses, metrics

import _path  # noqa
from imagemodel.binary_segmentations.datasets.oxford_iiit_pet import feeder
from imagemodel.binary_segmentations.datasets.pipeline import BSPipeline
from imagemodel.binary_segmentations.models.common_compile_options import CompileOptions
from imagemodel.binary_segmentations.models.trainers._trainer import Trainer
from imagemodel.binary_segmentations.models.unet_level import UNetLevelModelManager

if __name__ == "__main__":
    manager = UNetLevelModelManager(level=2, input_shape=(256, 256, 3))
    helper = CompileOptions(
        optimizer=optimizers.Adam(lr=1e-4),
        loss_functions=[losses.BinaryCrossentropy()],
        metrics=[metrics.BinaryAccuracy()])
    training_feeder = feeder.BSOxfordIIITPetTrainingFeeder()
    bs_pipeline = BSPipeline(training_feeder)

    trainer = Trainer(unet_level_model_manager=manager, compile_helper=helper, training_pipeline=bs_pipeline)
    trainer.fit()
