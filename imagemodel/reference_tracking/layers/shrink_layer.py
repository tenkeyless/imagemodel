import tensorflow as tf
from tensorflow.keras.layers import Layer

from imagemodel.common.utils.tf_images import tf_image_shrink


class ShrinkLayer(Layer):
    def __init__(self, bin_num: int, resize_by_power_of_two: int, **kwargs):
        super().__init__(**kwargs)
        self.bin_num: int = bin_num
        self.resize_by_power_of_two: int = resize_by_power_of_two
        
        # Layers
        self.shrink_layer = None
        self.reshape_layer = None
    
    def build(self, input_shape):
        ratio = 2 ** self.resize_by_power_of_two
        img_wh = input_shape[1] // ratio
        # Layers
        self.shrink_layer = lambda x: tf_image_shrink(x, self.bin_num, self.resize_by_power_of_two)
        self.reshape_layer = lambda x: tf.reshape(x, (-1, img_wh, img_wh, self.bin_num))
        # self.reshape_layer = Reshape((-1, img_wh, img_wh, self.bin_num))
        # self.reshape_layer = lambda x: backend.squeeze(x, 1)
    
    def get_config(self):
        config = super().get_config()
        config.update({"bin_num": self.bin_num, "resize_by_power_of_two": self.resize_by_power_of_two})
        return config
    
    def call(self, inputs, **kwargs):
        r = self.shrink_layer(inputs)
        r = self.reshape_layer(r)
        return r
