import tensorflow as tf
from tensorflow.keras.layers import Layer


# noinspection DuplicatedCode
class ExtractPatchLayer3(Layer):
    def __init__(self, k_size: int = 5, **kwargs):
        super().__init__(**kwargs)
        if (k_size % 2) == 0:
            raise RuntimeError("`k_size` should be odd number.")
        self.k_size = k_size
        self.tf_extract_patches_layer = None
    
    def build(self, input_shape):
        batch_size = input_shape[0]
        img_wh = input_shape[-3]
        channel = input_shape[-1]
        
        def tf_extract_patches(tf_array, _k_size: int, _img_wh: int, _channel: int, _batch_size: int):
            padding_size = max((_k_size - 1), 0) // 2
            zero_padded_image = tf.keras.layers.ZeroPadding2D((padding_size, padding_size))(tf_array)
            
            wh_indices = tf.range(_k_size) + tf.range(_img_wh)[:, tf.newaxis]
            
            a1 = tf.repeat(tf.repeat(wh_indices, _k_size, axis=1), _img_wh, axis=0)
            a2 = tf.tile(wh_indices, (_img_wh, _k_size))
            m = tf.stack([a1, a2], axis=-1)
            m = tf.expand_dims(m, axis=0)

            m1 = tf.repeat(m, _batch_size, axis=0)
            m2 = tf.reshape(m1, (-1, _img_wh, _img_wh, _k_size, _k_size, 2))
            
            gg = tf.gather_nd(zero_padded_image, m2, batch_dims=1)
            gg2 = tf.reshape(gg, (-1, _img_wh, _img_wh, _k_size * _k_size * _channel))
            
            return gg2
        
        self.tf_extract_patches_layer = lambda x: tf_extract_patches(
                x,
                _k_size=self.k_size,
                _img_wh=img_wh,
                _channel=channel,
                _batch_size=batch_size)
    
    def get_config(self):
        config = super().get_config()
        config.update({"k_size": self.k_size})
        return config
    
    def call(self, images, **kwargs):
        return self.tf_extract_patches_layer(images)
