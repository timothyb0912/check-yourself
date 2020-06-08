"""
Tests the MIXLB class to ensure it is constructed correctly and that all
methods work as expected.
"""
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

    def test_get_generated_coefs(self):
        """
        Ensures expected creation of heterogeneous coefficients.
        """
        model = MIXLB()
        model.double()
        func = model._get_generated_coefs
        func_args =\
            [('luggage_space',
              7,
              torch.tensor(2),
              torch.tensor([0.125, 0.25, 0.5]).double()),
             ('range_over_100',
              1,
              torch.tensor(3),
              torch.from_numpy((np.log([7.25, 7.5, 8]) - 1) / 3).double()
             ),
            ]
        expected_results =\
            [torch.tensor([7.25, 7.5, 8]).double(),
             torch.tensor([7.25, 7.5, 8]).double()]

        for args, expected_result in zip(func_args, expected_results):
            func_result = func(*args)
            self.assertTrue(torch.allclose(expected_result, func_result))
        return None
