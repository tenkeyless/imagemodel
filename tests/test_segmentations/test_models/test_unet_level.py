import tensorflow as tf
from segmentations.models.unet import unet
from segmentations.models.unet_level import unet_level


class UnetLevelTest(tf.test.TestCase):
    def testUNetLevelCreation(self):
        unet_model = unet()
        unet_level_model = unet_level(level=5)

        self.assertEqual(len(unet_model.layers), len(unet_level_model.layers))
