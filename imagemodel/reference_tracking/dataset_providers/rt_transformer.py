from abc import ABCMeta, abstractmethod
from typing import Tuple

import tensorflow as tf

from imagemodel.common.dataset_providers.transformer import TransformerP, TransformerT


class RTTransformerT(TransformerT, metaclass=ABCMeta):
    @property
    @abstractmethod
    def bin_size(self) -> int:
        pass
    
    def __augment(
            self,
            main_img: tf.Tensor,
            ref_img: tf.Tensor,
            main_label: tf.Tensor,
            ref_label: tf.Tensor,
            main_bw_label: tf.Tensor,
            ref_bw_label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        pass
    
    def __resize(
            self,
            main_img: tf.Tensor,
            ref_img: tf.Tensor,
            main_label: tf.Tensor,
            ref_label: tf.Tensor,
            main_bw_label: tf.Tensor,
            ref_bw_label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        pass
    
    def __color_extract(
            self,
            main_img: tf.Tensor,
            ref_img: tf.Tensor,
            main_label: tf.Tensor,
            ref_label: tf.Tensor,
            main_bw_label: tf.Tensor,
            ref_bw_label: tf.Tensor) -> \
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Tuple[tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor, tf.Tensor]:
        pass
    
    def __color_to_bin(
            self,
            main_img: tf.Tensor,
            ref_img: tf.Tensor,
            main_label: tf.Tensor,
            ref_label_color_map: tf.Tensor,
            ref_label: tf.Tensor,
            main_bw_label: tf.Tensor,
            ref_bw_label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        pass
    
    def __apply_filter(
            self,
            main_img: tf.Tensor,
            ref_img: tf.Tensor,
            bin_main_label: tf.Tensor,
            bin_ref_label: tf.Tensor,
            main_bw_label: tf.Tensor,
            ref_bw_label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        pass
    
    def __norm_data(
            self,
            main_img: tf.Tensor,
            ref_img: tf.Tensor,
            bin_main_label: tf.Tensor,
            bin_ref_label: tf.Tensor,
            main_bw_label: tf.Tensor,
            ref_bw_label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        pass
    
    def __zip_dataset(
            self,
            main_img: tf.Tensor,
            ref_img: tf.Tensor,
            bin_main_label: tf.Tensor,
            bin_ref_label: tf.Tensor,
            main_bw_label: tf.Tensor,
            ref_bw_label: tf.Tensor) -> \
            Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        pass


class RTTransformerP(TransformerP, metaclass=ABCMeta):
    @property
    @abstractmethod
    def bin_size(self) -> int:
        pass
    
    def __resize(self, filename: tf.Tensor, main_img: tf.Tensor, ref_img: tf.Tensor, ref_label: tf.Tensor) -> \
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        pass
    
    def __color_extract(self, filename: tf.Tensor, main_img: tf.Tensor, ref_img: tf.Tensor, ref_label: tf.Tensor) -> \
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        pass
    
    def __color_to_bin(
            self,
            filename: tf.Tensor,
            main_img: tf.Tensor,
            ref_img: tf.Tensor,
            ref_label: tf.Tensor,
            ref_label_color_map: Tuple[tf.Tensor, tf.Tensor]) -> \
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        pass
    
    def __apply_filter(
            self,
            filename: tf.Tensor,
            main_img: tf.Tensor,
            ref_img: tf.Tensor,
            bin_ref_label: tf.Tensor,
            ref_label_color_map: Tuple[tf.Tensor, tf.Tensor]) -> \
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        pass
    
    def __norm_data(
            self,
            filename: tf.Tensor,
            main_img: tf.Tensor,
            ref_img: tf.Tensor,
            bin_ref_label: tf.Tensor,
            ref_label_color_map: Tuple[tf.Tensor, tf.Tensor]) -> \
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        pass
    
    def __zip_dataset(
            self,
            filename: tf.Tensor,
            main_img: tf.Tensor,
            ref_img: tf.Tensor,
            bin_ref_label: tf.Tensor,
            ref_label_color_map: Tuple[tf.Tensor, tf.Tensor]) -> \
            Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]]:
        pass
