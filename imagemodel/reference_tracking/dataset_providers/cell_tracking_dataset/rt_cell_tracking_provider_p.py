import os
from typing import Tuple

import tensorflow as tf

from imagemodel.reference_tracking.dataset_providers.cell_tracking_dataset.rt_cell_tracking_drafter import \
    RTCellTrackingDrafterP
from imagemodel.reference_tracking.dataset_providers.rt_drafter import RTDrafterP
from imagemodel.reference_tracking.dataset_providers.rt_provider import RTProviderP
from imagemodel.reference_tracking.dataset_providers.rt_transformer import RTTransformerP
from imagemodel.reference_tracking.dataset_providers.transformer.base_rt_transformer import BaseRTTransformerP


class RTCellTrackingProviderP(RTProviderP):
    """
    Examples
    --------
    >>> from imagemodel.reference_tracking.dataset_providers.cell_tracking_dataset.\
    ...     rt_cell_tracking_provider_p import RTCellTrackingProviderP
    >>> provider = RTCellTrackingProviderP(
    ...     base_folder="/data/tracking_training",
    ...     shuffle=False,
    ...     random_seed=42,
    ...     bin_size=30,
    ...     resize_to=(256,256))
    >>> provider.plot_output_dataset()
    # <ParallelMapDataset
    #     shapes: (((256, 256, 1), (256, 256, 1), (256, 256, 30)), ((), ((None,), (None, None)))),
    #     types: ((tf.float32, tf.float32, tf.float32), (tf.string, (tf.float32, tf.float32)))>
    >>> provider.plot_output_dataset(sample_num=4, target_base_folder="/reference_tracking_results/test6")
    """
    
    def __init__(
            self,
            base_folder: str,
            bin_size: int,
            shuffle: bool = False,
            random_seed: int = 42,
            resize_to: Tuple[int, int] = (256, 256)):
        main_image_folder: str = os.path.join(base_folder, "framed_image", "zero")
        ref_image_folder: str = os.path.join(base_folder, "framed_image", "p1")
        ref_label_folder: str = os.path.join(base_folder, "framed_label", "p1")
        self.folders: Tuple[str, str, str] = (main_image_folder, ref_image_folder, ref_label_folder)
        self.shuffle: bool = shuffle
        self.random_seed: int = random_seed
        self.bin_size: int = bin_size
        self.resize_to: Tuple[int, int] = resize_to
    
    def get_drafter(self, shuffle: bool, random_seed: int = 42) -> RTDrafterP:
        return RTCellTrackingDrafterP(self.folders, shuffle=shuffle, random_seed=random_seed)
    
    def get_transformer(self, dataset: tf.data.Dataset, resize_to: Tuple[int, int]) -> RTTransformerP:
        return BaseRTTransformerP(dataset, resize_to, self.bin_size)
    
    def get_drafted_transformer(self) -> RTTransformerP:
        drafter: RTDrafterP = self.get_drafter(shuffle=self.shuffle, random_seed=self.random_seed)
        return self.get_transformer(drafter.out_dataset, resize_to=self.resize_to)
    
    @property
    def data_description(self):
        return "Reference Tracking Predict provider for Cell Tracking Dataset, `BaseRTTransformerP`."
