from typing import Callable, Tuple

import tensorflow as tf

from imagemodel.common.datasets.pipeline import Pipeline
from imagemodel.reference_tracking.datasets.cell_tracking.preprocessor import RTCellTrackingPreprocessor
from imagemodel.reference_tracking.datasets.rt_augmenter import BaseRTAugmenter, RTAugmenter
from imagemodel.reference_tracking.datasets.rt_feeder import RTFeeder
from imagemodel.reference_tracking.datasets.rt_preprocessor import RTPreprocessor
from imagemodel.reference_tracking.datasets.rt_regularizer import BaseRTRegularizer, RTRegularizer


class RTPipeline(Pipeline[RTFeeder, RTAugmenter, RTRegularizer, RTPreprocessor]):
    def __init__(
            self,
            feeder: RTFeeder,
            augmenter_func: Callable[[RTFeeder], RTAugmenter] = BaseRTAugmenter,
            regularizer_func: Callable[[RTAugmenter], RTRegularizer] = (
                    lambda el_rt_augmenter: BaseRTRegularizer(el_rt_augmenter, (256, 256))),
            preprocessor_func: Callable[[RTRegularizer], RTPreprocessor] = (
                    lambda el_rt_augmenter: RTCellTrackingPreprocessor(el_rt_augmenter, 30))):
        """
        Pipeline for Binary Segmentation.

        Parameters
        ----------
        feeder: BSFeeder
        augmenter_func: Callable[[BSFeeder], BSAugmenter], default=BaseBSAugmenter
        regularizer_func: Callable[[BSAugmenter], BSRegularizer], default=BaseBSRegularizer
        preprocessor_func: Callable[[BSRegularizer], BSPreprocessor], default=BaseBSPreprocessor

        Examples
        --------
        >>> from imagemodel.reference_tracking.datasets.cell_tracking.feeder import RTGSCellTrackingTrainingFeeder
        >>> training_feeder = RTGSCellTrackingTrainingFeeder()
        >>> from imagemodel.reference_tracking.datasets.pipeline import RTPipeline
        >>> rt_pipeline = RTPipeline(training_feeder)
        >>> rt_pipeline.get_zipped_dataset()
        <ZipDataset shapes: (((256, 256, 1), (256, 256, 1), (256, 256, 30)),
                             ((256, 256, 1), (256, 256, 1), (256, 256, 30))),
                    types: ((tf.float32, tf.float32, tf.float32), (tf.float32, tf.float32, tf.float32))>
        >>> for d in rt_pipeline.get_zipped_dataset().take(1):
        ...     ref_img = d[0][0]
        ...     main_img = d[0][1]
        ...     ref_bin_label = d[0][2]
        ...     rev_bw_label = d[1][0]
        ...     main_bw_label = d[1][1]
        ...     main_bin_label = d[1][2]
        ...
        >>> import cv2
        >>> cv2.imwrite("ref_img.png", ref_img.numpy()*255)
        >>> cv2.imwrite("main_img.png", main_img.numpy()*255)
        >>> cv2.imwrite("rev_bw_label.png", rev_bw_label.numpy()*255)
        >>> cv2.imwrite("main_bw_label.png", main_bw_label.numpy()*255)
        >>> for index in range(ref_bin_label.shape[-1]):
        ...     a = ref_bin_label[...,index]
        ...     b = a.numpy()
        ...     cv2.imwrite("ref_bin_label_{:02d}.png".format(index), b*255)
        ...
        >>> for index in range(main_bin_label.shape[-1]):
        ...     a = main_bin_label[...,index]
        ...     b = a.numpy()
        ...     cv2.imwrite("main_bin_label_{:02d}.png".format(index), b*255)
        ...
        """
        super().__init__(feeder, augmenter_func, regularizer_func, preprocessor_func)
    
    def get_input_zipped_dataset(self) -> tf.data.Dataset:
        return self.get_input_zipped_dataset_filled(filled_with=(255, 255, 255))
    
    def get_input_zipped_dataset_filled(self, filled_with: Tuple[int, int, int]) -> tf.data.Dataset:
        zipped_dataset = super().get_input_zipped_dataset()
        color_info_dataset = self.preprocessor.input_helper.ref_random_color_info_map(
                dataset=self.preprocessor.input_helper.get_ref_color_label_dataset())
        color_info_dataset = color_info_dataset.map(lambda img, color_info: color_info)
        
        def _color_fill(_color, _color_index, fill_with: Tuple[int, int, int]):
            fill_empty_with = tf.repeat([fill_with], repeats=tf.shape(_color_index)[-1], axis=0)
            fill_empty_with = tf.cast(fill_empty_with, tf.float32)
            filled_bin = tf.concat([_color, fill_empty_with], axis=0)
            filled_bin = filled_bin[:tf.shape(_color_index)[-1], :]
            return filled_bin
        
        color_info_dataset = color_info_dataset.map(
                lambda color_index, color: (color_index, _color_fill(color, color_index, filled_with)))
        if self.feeder.filename_optional is not None:
            filename_dataset = self.feeder.filename_optional
            result_dataset = tf.data.Dataset.zip((zipped_dataset, color_info_dataset, filename_dataset))
        else:
            result_dataset = tf.data.Dataset.zip((zipped_dataset, color_info_dataset))
        return result_dataset
    
    def get_zipped_dataset(self) -> tf.data.Dataset:
        return self.preprocessor.get_zipped_dataset()
    
    @property
    def data_description(self):
        return self.feeder.feeder_data_description
