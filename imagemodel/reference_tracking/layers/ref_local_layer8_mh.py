from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Layer, Reshape, Softmax

from imagemodel.reference_tracking.layers.extract_patch_layer2 import ExtractPatchLayer2 as ExtractPatchLayer


class RefLocal8MH(Layer):
    def __init__(
            self,
            intermediate_dim: Optional[int] = None,
            k_size: int = 5,
            mode: str = "dot",
            head_num: int = 1,
            **kwargs):
        super().__init__(**kwargs)
        self.input_intermediate_dim = intermediate_dim
        self.k_size = k_size
        self.mode = mode
        self.head_num = head_num
        
        # Layers
        self.conv_main_layers = []
        self.conv_ref_layers = []
        self.ref_extracted_reshape_layers = []
    
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
    
    def get_config(self):
        config = super().get_config()
        config.update(
                {
                    "intermediate_dim": self.input_intermediate_dim,
                    "k_size": self.k_size,
                    "mode": self.mode,
                    "head_num": self.head_num})
        return config
    
    def call(self, inputs, **kwargs):
        main = inputs[0]
        ref = inputs[1]
        
        attns = []
        for head_index in range(self.head_num):
            conv_main = self.conv_main_layers[head_index](main)
            conv_ref = self.conv_ref_layers[head_index](ref)
            
            ref_stacked = ExtractPatchLayer(k_size=self.k_size)(conv_ref)
            ref_stacked = self.ref_extracted_reshape_layers[head_index](ref_stacked)
            
            if self.mode == "dot":
                attn = tf.einsum("bhwc,bhwkc->bhwk", conv_main, ref_stacked)
                attn = Softmax()(attn)
            elif self.mode == "norm_dot":
                raise NotImplementedError("norm_dot model has not been implemented yet")
            elif self.mode == "gaussian":
                raise NotImplementedError("gaussian model has not been implemented yet")
            else:
                raise ValueError("`mode` value is not valid. Should be one of 'dot', 'norm_dot', 'gaussian'.")
            
            attns.append(attn)
        
        return tf.keras.layers.Average()(attns)
