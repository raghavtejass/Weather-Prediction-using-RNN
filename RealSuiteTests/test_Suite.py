from RealSuite import Suite
import unittest
import numpy as np
import pandas as pd


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.onedim = pd.DataFrame([x for x in range(1, 11)])

        self.twodim = pd.DataFrame([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])

    def test_BasicSplit(self):
        first, second = Suite.split_data(self.onedim, ratio=0.9)
        first_test = pd.DataFrame([x for x in range(1, 10)])
        second_test = pd.DataFrame([10])
        second = second.reset_index(drop=True)

        self.assertTrue(first.equals(first_test))
        self.assertTrue(second.equals(second_test))

    def test_BasicSplit_2(self):
        first, second = Suite.split_data(self.onedim, ratio=0.6)
        first_test = pd.DataFrame([x for x in range(1, 7)])
        second_test = pd.DataFrame([7, 8, 9, 10])
        second = second.reset_index(drop=True)

        self.assertTrue(first.equals(first_test))
        self.assertTrue(second.equals(second_test))

    def test_AdvancedSplit(self):
        first, second = Suite.split_data(self.twodim, ratio=0.6)
        first_test = pd.DataFrame([[1, 2], [2, 3], [3, 4]])
        second_test = pd.DataFrame([[4, 5], [5, 6]])
        second = second.reset_index(drop=True)

        self.assertTrue(first.equals(first_test))
        self.assertTrue(second.equals(second_test))

    def test_AdvancedSplit_2(self):
        first, second = Suite.split_data(self.twodim, ratio=0.8)
        first_test = pd.DataFrame([[1, 2], [2, 3], [3, 4], [4, 5]])
        second_test = pd.DataFrame([[5, 6]])
        second = second.reset_index(drop=True)

        self.assertTrue(first.equals(first_test))
        self.assertTrue(second.equals(second_test))


class rolling_window_test(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame(
            zip(range(1, 10), range(101, 110)), columns=["One", "Hundred"]
        )

    def test_basic(self):
        x, y = Suite.adv_rolling_window(
            self.data, feature_idx=0, input_size=3, output_size=3
        )

        a = np.array(
            [[[1], [2], [3]], [[2], [3], [4]], [[3], [4], [5]], [[4], [5], [6]]]
        )

        b = np.array(
            [[[4], [5], [6]], [[5], [6], [7]], [[6], [7], [8]], [[7], [8], [9]]]
        )

        self.assertTrue(np.array_equiv(x, a))
        self.assertTrue(np.array_equiv(y, b))

    def test_basic_2(self):
        x, y = Suite.adv_rolling_window(
            self.data, feature_idx=1, input_size=3, output_size=3
        )

        a = (
            np.array(
                [[[1], [2], [3]], [[2], [3], [4]], [[3], [4], [5]], [[4], [5], [6]]]
            )
            + 100
        )

        b = (
            np.array(
                [[[4], [5], [6]], [[5], [6], [7]], [[6], [7], [8]], [[7], [8], [9]]]
            )
            + 100
        )

        self.assertTrue(np.array_equiv(x, a))
        self.assertTrue(np.array_equiv(y, b))


