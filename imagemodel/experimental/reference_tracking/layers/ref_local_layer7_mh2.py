from typing import Optional

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import Concatenate, Conv2D, Layer, Reshape, Softmax, concatenate

from imagemodel.experimental.reference_tracking.layers.extract_patch_layer2 import ExtractPatchLayer2 as ExtractPatchLayer


class RefLocal7MH(Layer):
    def __init__(
            self,
            intermediate_dim: Optional[int] = None,
            k_size: int = 5,
            mode: str = "dot",
            head_num: int = 1,
            aggregation_mode: str = "average",
            **kwargs):
        super().__init__(**kwargs)
        self.input_intermediate_dim = intermediate_dim
        self.k_size = k_size
        self.mode = mode
        self.head_num = head_num
        self.aggregation_mode = aggregation_mode
        
        # Layers
        self.conv_main_layers = []
        self.conv_ref_layers = []
        self.ref_extracted_reshape_layers = []
        self.aggregation_conv_layer = None
    
    def build(self, input_shape):
        input_main_h_size = input_shape[0][1]
        input_main_w_size = input_shape[0][2]
        input_main_channels_size = input_shape[0][-1]
        self.input_intermediate_dim = self.input_intermediate_dim or input_main_channels_size
        
        # Layers
        for head_index in range(self.head_num):
            conv_main_layer = Conv2D(
                    filters=self.input_intermediate_dim // self.head_num,
                    kernel_size=1,
                    padding="same",
                    use_bias=False,
                    kernel_initializer="he_normal")
            self.conv_main_layers.append(conv_main_layer)
            
            conv_ref_layer = Conv2D(
                    filters=self.input_intermediate_dim // self.head_num,
                    kernel_size=1,
                    padding="same",
                    use_bias=False,
                    kernel_initializer="he_normal")
            self.conv_ref_layers.append(conv_ref_layer)
            
            ref_extracted_reshape_layer = Reshape(
                    (input_main_h_size,
                     input_main_w_size,
                     self.k_size * self.k_size,
                     self.input_intermediate_dim // self.head_num))
            self.ref_extracted_reshape_layers.append(ref_extracted_reshape_layer)
        
        self.aggregation_conv_layer = Conv2D(
                filters=(self.k_size * self.k_size) + 1,
                kernel_size=1,
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal")
    
    def get_config(self):
        config = super().get_config()
        config.update({"intermediate_dim": self.input_intermediate_dim, "k_size": self.k_size, "mode": self.mode})
        return config
    
    def call(self, inputs, **kwargs):
        main = inputs[0]
        ref = inputs[1]
        
        attns = []
        for head_index in range(self.head_num):
            conv_main = self.conv_main_layers[head_index](main)
            conv_ref = self.conv_ref_layers[head_index](ref)
            
            # Attention
            ref_stacked = ExtractPatchLayer(k_size=self.k_size)(conv_ref)
            ref_stacked = self.ref_extracted_reshape_layers[head_index](ref_stacked)
            
            if self.mode == "dot":
                expanded_main = backend.expand_dims(conv_main, -2)
                ref_with_main = Concatenate(axis=-2)([ref_stacked, expanded_main])
                attn = tf.einsum("bhwc,bhwkc->bhwk", conv_main, ref_with_main)
                attn = Softmax()(attn)
            elif self.mode == "norm_dot":
                raise NotImplementedError("norm_dot model has not been implemented yet")
            elif self.mode == "gaussian":
                raise NotImplementedError("gaussian model has not been implemented yet")
            else:
                raise ValueError("`mode` value is not valid. Should be one of 'dot', 'norm_dot', 'gaussian'.")
            
            attns.append(attn)
        
        result = attns
        if self.aggregation_mode == "average":
            result = tf.keras.layers.Average()(result)
        elif self.aggregation_mode == "max":
            result = tf.keras.layers.Maximum()(result)
            result = Softmax()(result)
        elif self.aggregation_mode == "conv":
            result = concatenate(result)
            result = self.aggregation_conv_layer(result)
            result = Softmax()(result)
        
        return result
