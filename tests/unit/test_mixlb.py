"""
Tests the MIXLB class to ensure it is constructed correctly and that all
methods work as expected.
"""
import sys
import unittest

import torch
import numpy as np

from src.models.mixlb import MIXLB


class InputMixlTests(unittest.TestCase):
    """
    Unit test class for storing the various tests of the MIXLB class.
    """
    def test_get_std_deviation(self):
        """
        Ensures that expected standard deviation values are returned and
        appropriate errors are raised.
        """
        model = MIXLB()
        func = model._get_std_deviation
        func_args = ['neg_price_over_log_income',
                     'luggage_space',
                     'vehicle_size_over_10']
        expected_results =\
            [model.log_normal_std, torch.tensor(1), torch.tensor(0)]


        for column_name, expected_result in zip(func_args, expected_results):
            func_result = func(column_name)
            self.assertEqual(expected_result.item(), func_result.item())

        with self.assertRaises(ValueError):
            func('fake_column')
        return None
