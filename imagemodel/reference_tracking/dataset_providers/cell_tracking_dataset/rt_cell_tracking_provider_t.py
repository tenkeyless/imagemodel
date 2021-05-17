import os
from typing import Optional, Tuple

import tensorflow as tf

from imagemodel.reference_tracking.dataset_providers.cell_tracking_dataset.rt_cell_tracking_drafter import \
    RTCellTrackingDrafterT
from imagemodel.reference_tracking.dataset_providers.rt_drafter import RTDrafterT
from imagemodel.reference_tracking.dataset_providers.rt_provider import RTProviderT
from imagemodel.reference_tracking.dataset_providers.rt_transformer import RTTransformerT
from imagemodel.reference_tracking.dataset_providers.transformer.base_rt_transformer import BaseRTTransformerT


class RTCellTrackingProviderT(RTProviderT):
    """
    Examples
    --------
    >>> from imagemodel.reference_tracking.dataset_providers.cell_tracking_dataset.\
    ...     rt_cell_tracking_provider_t import RTCellTrackingProviderT
    >>> provider = RTCellTrackingProviderT(
    ...     base_folder="/data/tracking_training",
    ...     shuffle_for_trainer=True,
    ...     shuffle=True,
    ...     random_seed=None,
    ...     bin_size=30,
    ...     resize_to=(256,256))
    >>> provider.get_output_dataset()
    # <ParallelMapDataset
    #     shapes: (((256, 256, 1), (256, 256, 1), (256, 256, 30)), ((256, 256, 1), (256, 256, 1), (256, 256, 30, 1))),
    #     types: ((tf.float32, tf.float32, tf.float32), (tf.float32, tf.float32, tf.float32))>
    >>> provider.plot_output_dataset(sample_num=8, target_base_folder="/reference_tracking_results/test5")
    """
    
    def __init__(
            self,
            base_folder: str,
            shuffle_for_trainer: bool,
            shuffle: bool,
            random_seed: Optional[int],
            bin_size: int,
            resize_to: Tuple[int, int] = (256, 256)):
        main_image_folder: str = os.path.join(base_folder, "framed_image", "zero")
        main_label_folder: str = os.path.join(base_folder, "framed_label", "zero")
        main_bw_label_folder: str = os.path.join(base_folder, "framed_bw_label", "zero")
        ref_image_folder: str = os.path.join(base_folder, "framed_image", "p1")
        ref_label_folder: str = os.path.join(base_folder, "framed_label", "p1")
        ref_bw_label_folder: str = os.path.join(base_folder, "framed_bw_label", "p1")
        self.folders: Tuple[str, str, str, str, str, str] = (
            main_image_folder,
            ref_image_folder,
            main_label_folder,
            ref_label_folder,
            main_bw_label_folder,
            ref_bw_label_folder)
        
        self.base_folder: str = base_folder
        self.shuffle_for_trainer: bool = shuffle_for_trainer
        self.shuffle: bool = shuffle
        self.random_seed: Optional[int] = random_seed
        self.bin_size: int = bin_size
        self.resize_to: Tuple[int, int] = resize_to
    
    def get_drafter(self, shuffle_for_trainer: bool, shuffle: bool, random_seed: Optional[int] = 42) -> RTDrafterT:
        return RTCellTrackingDrafterT(
                self.folders,
                shuffle_for_trainer=shuffle_for_trainer,
                shuffle=shuffle,
                random_seed=random_seed)
    
    def get_transformer(self, dataset: tf.data.Dataset, resize_to: Tuple[int, int]) -> RTTransformerT:
        return BaseRTTransformerT(dataset, resize_to, self.bin_size)
    
    def get_drafted_transformer(self) -> RTTransformerT:
        drafter: RTDrafterT = self.get_drafter(
                self.shuffle_for_trainer,
                shuffle=self.shuffle,
                random_seed=self.random_seed)
        return self.get_transformer(drafter.out_dataset, resize_to=self.resize_to)
    
    @property
    def data_description(self):
        return "Reference Tracking Training, Tester provider for Cell Tracking Dataset, `BaseRTTransformerT`. " \
               "Base folder is {}.".format(self.base_folder)
