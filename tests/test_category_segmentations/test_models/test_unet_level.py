import tensorflow as tf
from imagemodel.category_segmentations.models.unet import unet
from imagemodel.category_segmentations.models.unet_level import unet_level


class UnetLevelTest(tf.test.TestCase):
    def testUNetLevelCreation(self):
        unet_model = unet()
        unet_level_model = unet_level(level=5)

        self.assertEqual(len(unet_model.layers), len(unet_level_model.layers))
