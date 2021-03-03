from unittest import TestCase

import tensorflow as tf
from imagemodel.common.losses.categorical_cross_entropy import categorical_crossentropy
from imagemodel.common.metrics.categorical_cross_entropy import CategoricalCrossentropy


class TestCategoricalCrossEntropy(TestCase):
    def test_categorical_cross_entropy(self):
        y_true = [[0, 1, 0], [0, 0, 1]]
        y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
        sample_weight = [0.3, 0.7]

        # target 1
        ground_truth1 = 1.1769392

        # target 2
        ground_truth2 = 1.6271976

        # calculate 1
        categorical_cross_entropy = CategoricalCrossentropy()
        categorical_cross_entropy.update_state(y_true, y_pred)
        result1 = categorical_cross_entropy.result()

        # calculate 2
        categorical_cross_entropy.reset_states()
        categorical_cross_entropy.update_state(y_true, y_pred, sample_weight)
        result2 = categorical_cross_entropy.result()

        # assertion
        tf.debugging.assert_equal(ground_truth1, result1)
        tf.debugging.assert_equal(ground_truth2, result2)

    def test_categorical_cross_entropy_method(self):
        y_true = [[0, 1, 0], [0, 0, 1]]
        y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]

        # target 1
        ground_truth1 = 1.1769392

        # target 2
        ground_truth2 = 1.1769392

        # calculate 1
        cce = categorical_crossentropy(y_true, y_pred)
        result = tf.reduce_mean(cce)

        # calculate 2
        EPSILON = 1e-7
        # y_pred_2 = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
        y_pred_2 = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
        xent = -tf.reduce_sum(y_true * tf.math.log(y_pred_2), axis=-1)
        reduced_xent = tf.reduce_mean(xent)

        # assertion
        tf.debugging.assert_equal(ground_truth1, result)
        tf.debugging.assert_equal(ground_truth2, reduced_xent)
