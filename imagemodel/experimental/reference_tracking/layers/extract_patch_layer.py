import tensorflow as tf
from tensorflow.keras.layers import Layer


class ExtractPatchLayer(Layer):
    def __init__(self, k_size: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.k_size = k_size
    
    def get_config(self):
        config = super().get_config()
        config.update({"k_size": self.k_size})
        return config
    
    def call(self, images, **kwargs):
        return tf.image.extract_patches(
                images=images,
                sizes=[1, self.k_size, self.k_size, 1],
                strides=[1, 1, 1, 1],
                rates=[1, 1, 1, 1],
                padding="SAME")
