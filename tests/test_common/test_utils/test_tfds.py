from unittest import TestCase

from imagemodel.common.utils.tfds import append_tfds_str_range, int_range_to_string


class TFDSTest(TestCase):
    def testIntRangeToString(self):
        self.assertEqual(int_range_to_string(begin_optional=30, unit="%"), "30%:")
        self.assertEqual(int_range_to_string(end_optional=30, unit="%"), ":30%")
        self.assertEqual(int_range_to_string(unit="%"), None)
        self.assertEqual(
            int_range_to_string(begin_optional=10, end_optional=50, unit="%"), "10%:50%"
        )
        self.assertEqual(int_range_to_string(begin_optional=10), "10:")
        self.assertEqual(
            int_range_to_string(begin_optional=10, default_end_value="c"), "10:c"
        )

    def testAppendTfdsStrRange(self):
        self.assertEqual(append_tfds_str_range(option_string="train"), "train")
        self.assertEqual(
            append_tfds_str_range(option_string="train", begin_optional=30),
            "train[30%:]",
        )
