from typing import Optional

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import Concatenate, Conv2D, Layer, Reshape, Softmax

# from imagemodel.reference_tracking.layers.extract_patch_layer import ExtractPatchLayer
from imagemodel.reference_tracking.layers.extract_patch_layer2 import ExtractPatchLayer2 as ExtractPatchLayer
# from imagemodel.reference_tracking.layers.extract_patch_layer3 import ExtractPatchLayer3 as ExtractPatchLayer


class RefLocal7(Layer):
    def __init__(self, intermediate_dim: Optional[int] = None, k_size: int = 5, mode: str = "dot", **kwargs):
        super().__init__(**kwargs)
        self.input_intermediate_dim = intermediate_dim
        self.k_size = k_size
        self.mode = mode
        
        # Layers
        self.conv_main_layer = None
        self.conv_ref_layer = None
        self.ref_extracted_reshape_layer = None
        self.main_expanded_layer = None
        self.einsum_layer = None
    
    def build(self, input_shape):
        input_main_h_size = input_shape[0][1]
        input_main_w_size = input_shape[0][2]
        input_main_channels_size = input_shape[0][-1]
        self.input_intermediate_dim = self.input_intermediate_dim or input_main_channels_size
        
        # Layers
        self.conv_main_layer = Conv2D(
                filters=self.input_intermediate_dim,
                kernel_size=1,
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal")
        self.conv_ref_layer = Conv2D(
                filters=self.input_intermediate_dim,
                kernel_size=1,
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal")
        self.ref_extracted_reshape_layer = Reshape(
                (input_main_h_size,
                 input_main_w_size,
                 self.k_size * self.k_size,
                 self.input_intermediate_dim))
        self.main_expanded_layer = lambda x: backend.expand_dims(x, -2)
        self.einsum_layer = lambda x, y: tf.einsum("bhwc,bhwkc->bhwk", x, y)
    
    def get_config(self):
        config = super().get_config()
        config.update({"intermediate_dim": self.input_intermediate_dim, "k_size": self.k_size, "mode": self.mode})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs, **kwargs):
        input_main = inputs[0]
        input_ref = inputs[1]
        
        conv_main = self.conv_main_layer(input_main)
        conv_ref = self.conv_ref_layer(input_ref)
        
        # Attention
        ref_stacked = ExtractPatchLayer(k_size=self.k_size)(conv_ref)
        ref_stacked = self.ref_extracted_reshape_layer(ref_stacked)
        
        if self.mode == "dot":
            expanded_main = self.main_expanded_layer(conv_main)
            ref_with_main = Concatenate(axis=-2)([ref_stacked, expanded_main])
            attn = self.einsum_layer(conv_main, ref_with_main)
            attn = Softmax()(attn)
        elif self.mode == "norm_dot":
            raise NotImplementedError("norm_dot model has not been implemented yet")
        elif self.mode == "gaussian":
            raise NotImplementedError("gaussian model has not been implemented yet")
        else:
            raise ValueError("`mode` value is not valid. Should be one of 'dot', 'norm_dot', 'gaussian'.")
        
        return attn
