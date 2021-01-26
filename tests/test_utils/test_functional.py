from unittest import TestCase

from utils.functional import compose, compose_left


class FunctionalTest(TestCase):
    def testCompose(self):
        def f1(value: int):
            return value + 1

        def f2(value: int):
            return value * 2

        def f3(value: int):
            return value * 3

        result = f1(1)
        result = f2(result)
        result = f3(result)

        self.assertEqual(compose(f3, f2, f1)(1), result)
        self.assertEqual(compose_left(f1, f2, f3)(1), compose(f3, f2, f1)(1))
