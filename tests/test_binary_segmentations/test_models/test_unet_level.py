import tensorflow as tf
from tensorflow.keras import Model

from imagemodel.binary_segmentations.models.unet_level import UNetLevelModelManager
from imagemodel.common.utils.tf_model import model_is_equal


class UnetLevelTest(tf.test.TestCase):
    def setUp(self):
        base_model = UNetLevelModelManager(
            level=3,
            input_shape=(256, 256, 3),
            input_name="unet model_input",
            output_name="unet_model_output",
            base_filters=16)
        self.base_tf_model: Model = base_model.setup_model()

    def testUNetLevelCreation(self):
        unet_model = UNetLevelModelManager(
            level=3,
            input_shape=(256, 256, 3),
            input_name="unet model_input",
            output_name="unet_model_output",
            base_filters=16)
        tf_model: Model = unet_model.setup_model()
        self.assertTrue(model_is_equal(self.base_tf_model, tf_model))

    def testUNetLevelCreationStrDict(self):
        str_dict = {
            'level': '3', 'input_shape': '(256, 256, 3)', 'input_name': "unet model_input",
            'output_name': "unet_model_output", 'base_filters': '16'
        }
        unet_model = UNetLevelModelManager.init_with_str_dict(str_dict)
        tf_model: Model = unet_model.setup_model()
        self.assertTrue(model_is_equal(self.base_tf_model, tf_model))

    def testUNetLevelCreationStrDictDiff(self):
        str_dict = {
            'level': '2', 'input_shape': '(256, 256, 1)', 'input_name': "unet model_input",
            'output_name': "unet_model_output", 'base_filters': '4'
        }
        unet_model = UNetLevelModelManager.init_with_str_dict(str_dict)
        tf_model: Model = unet_model.setup_model()
        self.assertFalse(model_is_equal(self.base_tf_model, tf_model))

    def testUNetLevelCreationStrDictFailure(self):
        str_dict = {
            'level': '2', 'input_shape': '(256, 256)', 'input_name': "unet model_input",
            'output_name': "unet_model_output", 'base_filters': '4'
        }

        with self.assertRaises(ValueError) as context:
            UNetLevelModelManager.init_with_str_dict(str_dict)

        self.assertTrue("'input_shape' should be tuple of 3 ints." in str(context.exception))
