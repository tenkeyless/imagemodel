import os
from typing import List, Tuple

import cv2
import tensorflow as tf
import tf_clahe
from image_keras.tf.utils.images import (
    tf_change_order,
    tf_generate_random_color_map,
    tf_image_detach_with_id_color_list
)
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from imagemodel.common.reporter import Reporter
from imagemodel.common.utils.tf_images import tf_shrink3D
from imagemodel.reference_tracking.dataset_providers.rt_transformer import RTTransformerP, RTTransformerT


def check_in_dataset(dataset: tf.data.Dataset, target_dataset_length: int):
    dataset_length = len(dataset.element_spec)
    if dataset_length != target_dataset_length:
        return False


class BaseRTTransformerT(RTTransformerT):
    """
    Examples
    --------
    >>> import os
    >>> base_folder = "/data/tracking_training"
    >>> main_image_folder: str = os.path.join(base_folder, "framed_image", "zero")
    >>> main_label_folder: str = os.path.join(base_folder, "framed_label", "zero")
    >>> main_bw_label_folder: str = os.path.join(base_folder, "framed_bw_label", "zero")
    >>> ref_image_folder: str = os.path.join(base_folder, "framed_image", "p1")
    >>> ref_label_folder: str = os.path.join(base_folder, "framed_label", "p1")
    >>> ref_bw_label_folder: str = os.path.join(base_folder, "framed_bw_label", "p1")
    >>> folders = (
    ...     main_image_folder,
    ...     ref_image_folder,
    ...     main_label_folder,
    ...     ref_label_folder,
    ...     main_bw_label_folder,
    ...     ref_bw_label_folder)
    ...
    >>> from imagemodel.reference_tracking.dataset_providers.cell_tracking_dataset.\
    ...     rt_cell_tracking_drafter import RTCellTrackingDrafterT
    >>> dt = RTCellTrackingDrafterT(folders, shuffle_for_trainer=True, shuffle=True, random_seed=42)
    >>> from imagemodel.reference_tracking.dataset_providers.transformer.base_rt_transformer import BaseRTTransformerT
    >>> brt = BaseRTTransformerT(dt.out_dataset, (256, 256), 30)
    >>> brt.plot_out_dataset(10, "/reference_tracking_results/test3")
    >>> for d in brt.out_dataset.take(1):
    ...     print(d)
    ...
    """
    
    def __init__(self, in_dataset: tf.data.Dataset, resize_to: Tuple[int, int], bin_size: int):
        self.__in_dataset: tf.data.Dataset = in_dataset
        self.__in_dataset_length: int = 6
        check_in_dataset(self.in_dataset, self.__in_dataset_length)
        
        self.__resize_to: Tuple[int, int] = resize_to
        self.__bin_size: int = bin_size
    
    @property
    def resize_to(self) -> Tuple[int, int]:
        return self.__resize_to
    
    @property
    def bin_size(self) -> int:
        return self.__bin_size
    
    @property
    def in_dataset(self) -> tf.data.Dataset:
        return self.__in_dataset
    
    @property
    def out_dataset(self) -> tf.data.Dataset:
        # Augment
        augmented_dataset = self.in_dataset.map(self.__augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # Regularize
        resized_dataset = augmented_dataset.map(self.__resize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # Preprocess
        color_extracted_dataset = resized_dataset.map(
                self.__color_extract,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        color_to_bin_dataset = color_extracted_dataset.map(
                self.__color_to_bin,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        filtered_dataset = color_to_bin_dataset.map(
                self.__apply_filter,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        normed_dataset = filtered_dataset.map(self.__norm_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # Zip
        zipped_dataset = normed_dataset.map(self.__zip_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return zipped_dataset
    
    def plot_out_dataset(self, sample_num: int, target_base_folder: str):
        dataset = self.out_dataset
        
        files: List[str] = []
        _base_folder: str = target_base_folder
        
        if target_base_folder.startswith("gs://"):
            _base_folder = "/tmp"
        
        for i, d in enumerate(dataset.take(sample_num)):
            inputs = d[0]
            
            main_img_file_name = "{}_RT_input_main_img.png".format(i)
            main_img_fullpath = os.path.join(_base_folder, main_img_file_name)
            cv2.imwrite(main_img_fullpath, inputs[0].numpy() * 255)
            files.append(main_img_fullpath)
            
            ref_img_file_name = "{}_RT_input_ref_img.png".format(i)
            ref_img_fullpath = os.path.join(_base_folder, ref_img_file_name)
            cv2.imwrite(ref_img_fullpath, inputs[1].numpy() * 255)
            files.append(ref_img_fullpath)
            
            for b in range(inputs[2].shape[-1]):
                ref_bin_file_name = "{}_RT_bin_{:02d}_input_ref.png".format(i, b)
                ref_bin_fullpath = os.path.join(_base_folder, ref_bin_file_name)
                cv2.imwrite(ref_bin_fullpath, inputs[2][..., b:b + 1].numpy() * 255)
                files.append(ref_bin_fullpath)
            
            outputs = d[1]
            
            main_bw_label_file_name = "{}_RT_output_main_bw_label.png".format(i)
            main_bw_label_fullpath = os.path.join(_base_folder, main_bw_label_file_name)
            cv2.imwrite(main_bw_label_fullpath, outputs[0].numpy() * 255)
            files.append(main_bw_label_fullpath)
            
            ref_bw_label_file_name = "{}_RT_output_ref_bw_label.png".format(i)
            ref_bw_label_fullpath = os.path.join(_base_folder, ref_bw_label_file_name)
            cv2.imwrite(ref_bw_label_fullpath, outputs[1].numpy() * 255)
            files.append(ref_bw_label_fullpath)
            
            for b in range(outputs[2].shape[-2]):
                main_bin_file_name = "{}_RT_bin_{:02d}_output_main.png".format(i, b)
                main_bin_fullpath = os.path.join(_base_folder, main_bin_file_name)
                cv2.imwrite(main_bin_fullpath, outputs[2][..., b:b + 1, 0].numpy() * 255)
                files.append(main_bin_fullpath)
        
        if target_base_folder.startswith("gs://"):
            for file in files:
                Reporter.upload_file_to_google_storage(target_base_folder, file)
    
    def __augment(
            self,
            main_img: tf.Tensor,
            ref_img: tf.Tensor,
            main_label: tf.Tensor,
            ref_label: tf.Tensor,
            main_bw_label: tf.Tensor,
            ref_bw_label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        return main_img, ref_img, main_label, ref_label, main_bw_label, ref_bw_label
    
    def __resize(
            self,
            main_img: tf.Tensor,
            ref_img: tf.Tensor,
            main_label: tf.Tensor,
            ref_label: tf.Tensor,
            main_bw_label: tf.Tensor,
            ref_bw_label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        @tf.autograph.experimental.do_not_convert
        def __resize(img: tf.Tensor) -> tf.Tensor:
            return tf.image.resize(img, self.resize_to)
        
        @tf.autograph.experimental.do_not_convert
        def __resize_nn(img: tf.Tensor) -> tf.Tensor:
            return tf.image.resize(img, self.resize_to, method=ResizeMethod.NEAREST_NEIGHBOR)
        
        return (
            __resize(main_img),
            __resize(ref_img),
            __resize_nn(main_label),
            __resize_nn(ref_label),
            __resize(main_bw_label),
            __resize(ref_bw_label))
    
    def __color_extract(
            self,
            main_img: tf.Tensor,
            ref_img: tf.Tensor,
            main_label: tf.Tensor,
            ref_label: tf.Tensor,
            main_bw_label: tf.Tensor,
            ref_bw_label: tf.Tensor) -> \
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Tuple[tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor, tf.Tensor]:
        @tf.autograph.experimental.do_not_convert
        def tf_color_to_random_map(ref_label_img, bin_size, exclude_first=1):
            return tf_generate_random_color_map(ref_label_img, bin_size=bin_size, shuffle_exclude_first=exclude_first)
        
        ref_label_color_map = tf_color_to_random_map(ref_label, self.bin_size, 1)
        return main_img, ref_img, main_label, ref_label_color_map, ref_label, main_bw_label, ref_bw_label
    
    def __color_to_bin(
            self,
            main_img: tf.Tensor,
            ref_img: tf.Tensor,
            main_label: tf.Tensor,
            ref_label_color_map: Tuple[tf.Tensor, tf.Tensor],
            ref_label: tf.Tensor,
            main_bw_label: tf.Tensor,
            ref_bw_label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        @tf.autograph.experimental.do_not_convert
        def tf_image_detach_with_id_color_probability_list(
                color_img,
                id_color_list,
                bin_num: int,
                resize_by_power_of_two: int = 0):
            result = tf_image_detach_with_id_color_list(color_img, id_color_list, bin_num, 1.0)
            ratio = 2 ** resize_by_power_of_two
            result2 = tf_shrink3D(result, tf.shape(result)[-3] // ratio, tf.shape(result)[-2] // ratio, bin_num)
            result2 = tf.divide(result2, ratio ** 2)
            return result2
        
        @tf.autograph.experimental.do_not_convert
        def __tf_input_ref_label_preprocessing_function(label, color_info, bin_size):
            result = tf_image_detach_with_id_color_probability_list(label, color_info, bin_size, 0)
            result = tf.reshape(result, (256 // (2 ** 0), 256 // (2 ** 0), bin_size))
            result = tf_change_order(result, color_info[0])
            result = tf.squeeze(result)
            return result
        
        @tf.autograph.experimental.do_not_convert
        def __tf_output_label_processing(label, color_info, bin_size):
            result = tf_image_detach_with_id_color_probability_list(label, color_info, bin_size, 0)
            result = tf.reshape(result, (256 // (2 ** 0), 256 // (2 ** 0), bin_size))
            result = tf_change_order(result, color_info[0])
            return result
        
        return (
            main_img,
            ref_img,
            __tf_output_label_processing(main_label, ref_label_color_map, self.bin_size),
            __tf_input_ref_label_preprocessing_function(ref_label, ref_label_color_map, self.bin_size),
            main_bw_label,
            ref_bw_label)
    
    def __apply_filter(
            self,
            main_img: tf.Tensor,
            ref_img: tf.Tensor,
            bin_main_label: tf.Tensor,
            bin_ref_label: tf.Tensor,
            main_bw_label: tf.Tensor,
            ref_bw_label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        @tf.autograph.experimental.do_not_convert
        def __clahe(img: tf.Tensor) -> tf.Tensor:
            return tf_clahe.clahe(img, tile_grid_size=[8, 8], clip_limit=2.0)
        
        @tf.autograph.experimental.do_not_convert
        def __reshape(img: tf.Tensor) -> tf.Tensor:
            return tf.reshape(img, (256, 256, 1))
        
        return (
            __reshape(__clahe(main_img)),
            __reshape(__clahe(ref_img)),
            bin_main_label,
            bin_ref_label,
            main_bw_label,
            ref_bw_label)
    
    def __norm_data(
            self,
            main_img: tf.Tensor,
            ref_img: tf.Tensor,
            bin_main_label: tf.Tensor,
            bin_ref_label: tf.Tensor,
            main_bw_label: tf.Tensor,
            ref_bw_label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        @tf.autograph.experimental.do_not_convert
        def __cast_norm(img: tf.Tensor) -> tf.Tensor:
            return tf.cast(img, tf.float32) / 255.0
        
        @tf.autograph.experimental.do_not_convert
        def __greater_cast(img: tf.Tensor) -> tf.Tensor:
            return tf.cast(tf.greater(tf.cast(img, tf.float32), 0.5), tf.float32)
        
        return (
            __cast_norm(main_img),
            __cast_norm(ref_img),
            bin_main_label,
            bin_ref_label,
            __greater_cast(main_bw_label),
            __greater_cast(ref_bw_label))
    
    def __zip_dataset(
            self,
            main_img: tf.Tensor,
            ref_img: tf.Tensor,
            bin_main_label: tf.Tensor,
            bin_ref_label: tf.Tensor,
            main_bw_label: tf.Tensor,
            ref_bw_label: tf.Tensor) -> \
            Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        return (main_img, ref_img, bin_ref_label), (main_bw_label, ref_bw_label, bin_main_label)


class BaseRTTransformerP(RTTransformerP):
    """
    Examples
    --------
    >>> import os
    >>> base_folder = "/data/tracking_training"
    >>> main_image_folder: str = os.path.join(base_folder, "framed_image", "zero")
    >>> ref_image_folder: str = os.path.join(base_folder, "framed_image", "p1")
    >>> ref_label_folder: str = os.path.join(base_folder, "framed_label", "p1")
    >>> folders = (
    ...     main_image_folder,
    ...     ref_image_folder,
    ...     ref_label_folder)
    ...
    >>> from imagemodel.reference_tracking.dataset_providers.cell_tracking_dataset.\
    ...     rt_cell_tracking_drafter import RTCellTrackingDrafterP
    >>> dt = RTCellTrackingDrafterP(None, folders, shuffle=True, random_seed=42)
    >>>
    >>> from imagemodel.reference_tracking.dataset_providers.transformer.base_rt_transformer import BaseRTTransformerP
    >>> brt = BaseRTTransformerP(dt.out_dataset, (256, 256), 30)
    >>>
    >>> brt.plot_out_dataset(10, "/reference_tracking_results/test3")
    >>>
    >>> for d in brt.out_dataset.take(1):
    ...     print(d)
    ...
    """
    
    def __init__(
            self,
            in_dataset: tf.data.Dataset,
            resize_to: Tuple[int, int],
            bin_size: int,
            fill_with: Tuple[int, int, int] = (255, 255, 255)):
        self.__in_dataset: tf.data.Dataset = in_dataset
        self.__in_dataset_length: int = 4
        check_in_dataset(self.in_dataset, self.__in_dataset_length)
        
        self.__resize_to: Tuple[int, int] = resize_to
        self.__bin_size: int = bin_size
        self.fill_with: Tuple[int, int, int] = fill_with
    
    @property
    def resize_to(self) -> Tuple[int, int]:
        return self.__resize_to
    
    @property
    def bin_size(self) -> int:
        return self.__bin_size
    
    @property
    def in_dataset(self) -> tf.data.Dataset:
        return self.__in_dataset
    
    def __resize(self, filename: tf.Tensor, main_img: tf.Tensor, ref_img: tf.Tensor, ref_label: tf.Tensor) -> \
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        @tf.autograph.experimental.do_not_convert
        def __resize(img: tf.Tensor) -> tf.Tensor:
            return tf.image.resize(img, self.resize_to)
        
        @tf.autograph.experimental.do_not_convert
        def __resize_nn(img: tf.Tensor) -> tf.Tensor:
            return tf.image.resize(img, self.resize_to, method=ResizeMethod.NEAREST_NEIGHBOR)
        
        return (
            filename,
            __resize(main_img),
            __resize(ref_img),
            __resize_nn(ref_label))
    
    def __color_extract(self, filename: tf.Tensor, main_img: tf.Tensor, ref_img: tf.Tensor, ref_label: tf.Tensor) -> \
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        @tf.autograph.experimental.do_not_convert
        def tf_color_to_random_map(ref_label_img, bin_size, exclude_first=1):
            return tf_generate_random_color_map(ref_label_img, bin_size=bin_size, shuffle_exclude_first=exclude_first)
        
        ref_label_color_map = tf_color_to_random_map(ref_label, self.bin_size, 1)
        return filename, main_img, ref_img, ref_label, ref_label_color_map
    
    def __color_to_bin(
            self,
            filename: tf.Tensor,
            main_img: tf.Tensor,
            ref_img: tf.Tensor,
            ref_label: tf.Tensor,
            ref_label_color_map: Tuple[tf.Tensor, tf.Tensor]) -> \
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        @tf.autograph.experimental.do_not_convert
        def tf_image_detach_with_id_color_probability_list(
                color_img,
                id_color_list,
                bin_num: int,
                resize_by_power_of_two: int = 0):
            result = tf_image_detach_with_id_color_list(color_img, id_color_list, bin_num, 1.0)
            ratio = 2 ** resize_by_power_of_two
            result2 = tf_shrink3D(result, tf.shape(result)[-3] // ratio, tf.shape(result)[-2] // ratio, bin_num)
            result2 = tf.divide(result2, ratio ** 2)
            return result2
        
        @tf.autograph.experimental.do_not_convert
        def __tf_input_ref_label_preprocessing_function(label, color_info, bin_size):
            result = tf_image_detach_with_id_color_probability_list(label, color_info, bin_size, 0)
            result = tf.reshape(result, (256 // (2 ** 0), 256 // (2 ** 0), bin_size))
            result = tf_change_order(result, color_info[0])
            result = tf.squeeze(result)
            return result
        
        def generate_filled_color_map(color_map):
            @tf.autograph.experimental.do_not_convert
            def _color_fill(_color, _color_index, fill_with: Tuple[int, int, int]):
                fill_empty_with = tf.repeat([fill_with], repeats=tf.shape(_color_index)[-1], axis=0)
                fill_empty_with = tf.cast(fill_empty_with, tf.float32)
                filled_bin = tf.concat([_color, fill_empty_with], axis=0)
                filled_bin = filled_bin[:tf.shape(_color_index)[-1], :]
                result = tf.gather(filled_bin, tf.cast(_color_index, tf.int32), axis=0)
                return result
            
            return _color_fill(color_map[1], color_map[0], self.fill_with)
        
        ref_label_color_map_filled = generate_filled_color_map(ref_label_color_map)
        return (
            filename,
            main_img,
            ref_img,
            __tf_input_ref_label_preprocessing_function(ref_label, ref_label_color_map, self.bin_size),
            ref_label_color_map_filled)
    
    def __apply_filter(
            self,
            filename: tf.Tensor,
            main_img: tf.Tensor,
            ref_img: tf.Tensor,
            bin_ref_label: tf.Tensor,
            ref_label_color_map: tf.Tensor) -> \
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        @tf.autograph.experimental.do_not_convert
        def __clahe(img: tf.Tensor) -> tf.Tensor:
            return tf_clahe.clahe(img, tile_grid_size=[8, 8], clip_limit=2.0)
        
        @tf.autograph.experimental.do_not_convert
        def __reshape(img: tf.Tensor) -> tf.Tensor:
            return tf.reshape(img, (256, 256, 1))
        
        return (
            filename,
            __reshape(__clahe(main_img)),
            __reshape(__clahe(ref_img)),
            bin_ref_label,
            ref_label_color_map)
    
    def __norm_data(
            self,
            filename: tf.Tensor,
            main_img: tf.Tensor,
            ref_img: tf.Tensor,
            bin_ref_label: tf.Tensor,
            ref_label_color_map: tf.Tensor) -> \
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        @tf.autograph.experimental.do_not_convert
        def __cast_norm(img: tf.Tensor) -> tf.Tensor:
            return tf.cast(img, tf.float32) / 255.0
        
        return (
            filename,
            __cast_norm(main_img),
            __cast_norm(ref_img),
            bin_ref_label,
            ref_label_color_map)
    
    def __zip_dataset(
            self,
            filename: tf.Tensor,
            main_img: tf.Tensor,
            ref_img: tf.Tensor,
            bin_ref_label: tf.Tensor,
            ref_label_color_map: tf.Tensor) -> \
            Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
        return (main_img, ref_img, bin_ref_label), (filename, ref_label_color_map)
    
    @property
    def out_dataset(self) -> tf.data.Dataset:
        # Regularize
        resized_dataset = self.in_dataset.map(self.__resize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # Preprocess
        color_extracted_dataset = resized_dataset.map(
                self.__color_extract,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        color_to_bin_dataset = color_extracted_dataset.map(
                self.__color_to_bin,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        filtered_dataset = color_to_bin_dataset.map(
                self.__apply_filter,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        normed_dataset = filtered_dataset.map(self.__norm_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # Zip
        zipped_dataset = normed_dataset.map(self.__zip_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return zipped_dataset
    
    def plot_out_dataset(self, sample_num: int, target_base_folder: str):
        dataset = self.out_dataset
        
        files: List[str] = []
        _base_folder: str = target_base_folder
        
        if target_base_folder.startswith("gs://"):
            _base_folder = "/tmp"
        
        for i, d in enumerate(dataset.take(sample_num)):
            assistants = d[1]
            filename = assistants[0]
            filename = filename.numpy().decode()
            filename = filename[:filename.rfind(".")]
            color_map = assistants[1]
            
            inputs = d[0]
            main_img_file_name = "predict_{}_main_img.png".format(filename)
            main_img_fullpath = os.path.join(_base_folder, main_img_file_name)
            cv2.imwrite(main_img_fullpath, inputs[0].numpy() * 255)
            files.append(main_img_fullpath)
            
            ref_img_file_name = "predict_{}_ref_img.png".format(filename)
            ref_img_fullpath = os.path.join(_base_folder, ref_img_file_name)
            cv2.imwrite(ref_img_fullpath, inputs[1].numpy() * 255)
            files.append(ref_img_fullpath)
            
            for b in range(inputs[2].shape[-1]):
                ref_bin_file_name = "predict_{}_RT_ref_bin_{:02d}.png".format(filename, b)
                ref_bin_fullpath = os.path.join(_base_folder, ref_bin_file_name)
                cv2.imwrite(ref_bin_fullpath, inputs[2][..., b:b + 1].numpy() * 255)
                files.append(ref_bin_fullpath)
            
            ref_arg_max_bin = tf.argmax(inputs[2], axis=-1)
            ref_label = tf.gather(color_map, ref_arg_max_bin, axis=0, batch_dims=1)
            ref_label_file_name = "predict_{}_ref_color_label_argmaxed.png".format(filename)
            ref_label_fullpath = os.path.join(_base_folder, ref_label_file_name)
            img = tf.image.encode_png(tf.cast(ref_label, tf.uint8))
            tf.io.write_file(ref_label_fullpath, img)
        
        if target_base_folder.startswith("gs://"):
            for file in files:
                Reporter.upload_file_to_google_storage(target_base_folder, file)
