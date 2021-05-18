from abc import ABCMeta
from typing import Optional, Tuple

import tensorflow as tf

from imagemodel.common.dataset_providers.drafter import DrafterP, DrafterT
from imagemodel.common.dataset_providers.transformer import Transformer


class Provider(metaclass=ABCMeta):
    def get_output_dataset(self) -> tf.data.Dataset:
        return self.get_drafted_transformer().out_dataset
    
    def plot_output_dataset(self, sample_num: int, target_base_folder: str):
        self.get_drafted_transformer().plot_out_dataset(sample_num=sample_num, target_base_folder=target_base_folder)
    
    def get_transformer(self, dataset: tf.data.Dataset, resize_to: Tuple[int, int]) -> Transformer:
        pass
    
    def get_drafted_transformer(self) -> Transformer:
        pass
    
    @property
    def data_description(self):
        return "Data Provider"


class ProviderT(Provider, metaclass=ABCMeta):
    def get_drafter(self, shuffle_for_trainer: bool, shuffle: bool, random_seed: Optional[int] = 42) -> DrafterT:
        pass


class ProviderP(Provider, metaclass=ABCMeta):
    def get_drafter(self, shuffle: bool, random_seed: Optional[int] = 42) -> DrafterP:
        pass
