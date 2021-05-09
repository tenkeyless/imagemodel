import numpy as np
import tensorflow as tf

from imagemodel.reference_tracking.layers.extract_patch_layer import ExtractPatchLayer
from imagemodel.reference_tracking.layers.extract_patch_layer3 import ExtractPatchLayer3


class TestExtractPatchLayer3(tf.test.TestCase):
    def test_extract_patch_layer3(self):
        batch_size = 2
        hw_size = 2
        channel_size = 3
        k_size = 3
        
        input_feature = tf.ones([batch_size, hw_size, hw_size, channel_size])
        output = ExtractPatchLayer3(k_size=k_size)(input_feature)
        
        # assertion
        tf.debugging.assert_shapes([(output, (batch_size, hw_size, hw_size, k_size * k_size * channel_size))])
    
    def test_same_with_extract_patch_layer(self):
        batch_size = 4
        hw_size = 32
        channel_size = 3
        k_size = 5
        
        input_feature = tf.constant(
                np.arange(batch_size * hw_size * hw_size * channel_size).reshape(
                        [batch_size, hw_size, hw_size, channel_size]))
        epl3 = ExtractPatchLayer3(k_size=k_size)(input_feature)
        epl = ExtractPatchLayer(k_size=k_size)(input_feature)
        tf.debugging.assert_equal(epl3, epl)
