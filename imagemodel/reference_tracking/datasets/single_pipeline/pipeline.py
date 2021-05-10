from typing import Callable

import tensorflow as tf

from imagemodel.common.datasets.pipeline import Pipeline
from imagemodel.reference_tracking.datasets.single_pipeline.rt_augmenter import (
    BaseRTAugmenter,
    RTAugmenter
)
from imagemodel.reference_tracking.datasets.single_pipeline.cell_tracking.rt_clahe_preprocessor import \
    RTCellTrackingPreprocessor
from imagemodel.reference_tracking.datasets.single_pipeline.rt_preprocessor import RTPreprocessor
from imagemodel.reference_tracking.datasets.single_pipeline.rt_regularizer import (
    BaseRTRegularizer,
    RTRegularizer
)
from imagemodel.reference_tracking.datasets.rt_feeder import RTFeeder


class RTSinglePipeline(Pipeline[RTFeeder, RTAugmenter, RTRegularizer, RTPreprocessor]):
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
        >>> from imagemodel.reference_tracking.datasets.cell_tracking.feeder import RTCellTrackingTrainingFeeder
        >>> training_feeder = RTCellTrackingTrainingFeeder()
        >>> from imagemodel.reference_tracking.datasets.cell_tracking.pipeline import RTSinglePipeline
        >>> rt_pipeline = RTSinglePipeline(training_feeder)
        >>> rt_pipeline.get_zipped_dataset()
        <ZipDataset shapes: (((256, 256, 1), (256, 256, 1), (256, 256, 30)),
                             ((256, 256, 1), (256, 256, 1), (256, 256, 30))),
                    types: ((tf.float32, tf.float32, tf.float32), (tf.float32, tf.float32, tf.float32))>
        >>> import cv2
        >>> for i, d in enumerate(rt_pipeline.get_zipped_dataset().take(10)):
        ...     ref_img = d[0][0]
        ...     cv2.imwrite("{}_ref_img.png".format(i), ref_img.numpy()*255)
        ...     main_img = d[0][1]
        ...     cv2.imwrite("{}_main_img.png".format(i), main_img.numpy()*255)
        ...     ref_bin_label = d[0][2]
        ...     rev_bw_label = d[1][0]
        ...     cv2.imwrite("{}_rev_bw_label.png".format(i), rev_bw_label.numpy()*255)
        ...     main_bw_label = d[1][1]
        ...     cv2.imwrite("{}_main_bw_label.png".format(i), main_bw_label.numpy()*255)
        ...     main_bin_label = d[1][2]
        ...
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
    
    def get_zipped_dataset(self) -> tf.data.Dataset:
        input_dataset = self.preprocessor.get_input_dataset()[0]
        output_dataset = self.preprocessor.get_output_dataset()[0]
        return tf.data.Dataset.zip((input_dataset, output_dataset))
    
    @property
    def data_description(self):
        return self.feeder.feeder_data_description
