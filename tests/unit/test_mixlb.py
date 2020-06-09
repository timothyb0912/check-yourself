"""
Tests the MIXLB class to ensure it is constructed correctly and that all
methods work as expected.
"""
import unittest

import torch
import numpy as np

from src.models.mixlb import MIXLB


class MixlBTests(unittest.TestCase):
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

    def test_create_coef_tensor(self):
        """
        Ensures that at least in the trivial case of random variates that are
        all zeros, that we return the expected values. Not the strongest test,
        but ensures the function can return results of the correct shape and,
        at least trivially, indexes the column names correctly.
        """
        # Create model object and set its dtype to double.
        model = MIXLB()
        model.double()
        # Extract needed information from the model.
        num_design_columns = model.means.size()[0]
        num_mixing_vars = model.design_info.num_mixing_vars
        # Set needed constants for test argument creation
        num_draws = 3
        num_decision_makers = 2
        # Alias the function being tested
        func = model.create_coef_tensor
        # Note the expected results for the test
        # - coef_tensor of shape (2, num_predictors, 3 draws)
        expected_results =\
            [((torch.arange(num_design_columns,
                            dtype=torch.double)[None, :] *
               torch.ones(num_decision_makers)[:, None]
               )[:, :, None] *
              torch.ones(num_draws, dtype=torch.double)[None, None, :]
             ),
            ]
        # Account for the lognormal coefficients
        lognormal_indices =\
            [model.design_info.mixing_to_design_cols[x] for x in
             model.design_info.lognormal_coef_names]
        expected_results[0][:, lognormal_indices, :] =\
            torch.exp(expected_results[0][:, lognormal_indices, :])
        # Create function arguments that should lead to the desired results.
        fake_design =\
            torch.ones((num_decision_makers, num_design_columns),
                       dtype=torch.double)
        # corresponds to an identity matrix of size 2x2
        fake_mapping =\
            torch.sparse.FloatTensor(torch.LongTensor([[0, 1], [0, 1]]),
                                     torch.ones(2, dtype=torch.double),
                                     torch.Size([2, 2]))
        fake_rvs_list =\
            [torch.zeros(num_decision_makers, num_draws, dtype=torch.double)
             for x in range(num_mixing_vars)]
        func_args =\
            [(fake_design, fake_mapping, fake_rvs_list),]
        # Test the function.
        for args, expected_result in zip(func_args, expected_results):
            func_result = func(*args)
            self.assertTrue(torch.allclose(expected_result, func_result))

    def test_calc_systematic_utilities(self):
        # Create model object and set its dtype to double.
        model = MIXLB()
        model.double()
        # Create the fake arguments
        fake_design = torch.ones((3, 5), dtype=torch.double)
        fake_coefs =\
            (torch.arange(start=1, end=4, dtype=torch.double)[:, None, None] *
             torch.ones((3, 5, 4), dtype=torch.double))
        # Create / note the expected results
        expected_result =\
            (torch.tensor([5, 10, 15], dtype=torch.double)[:, None] *
             torch.ones((3, 4), dtype=torch.double))
        # Alias the function to be tested
        func = model._calc_systematic_utilities
        # Perform the desired test
        func_result = func(fake_design, fake_coefs)
        self.assertTrue(torch.allclose(expected_result, func_result))

    def test_calc_probs_per_draw(self):
        # Create model object and set its dtype to double.
        model = MIXLB()
        model.double()
        # Create the fake arguments
        numpy_utilities =\
            np.log(np.concatenate((np.arange(start=1, stop=7)[:, None],
                                   2 * np.arange(start=1, stop=7)[:, None]),
                                  axis=1))
        fake_utilities = torch.from_numpy(numpy_utilities)
        fake_mapping =\
            torch.sparse.FloatTensor(
                torch.LongTensor([[0, 1, 2, 3, 4, 5],
                                  [0, 0, 0, 1, 1, 1]]),
                torch.ones(6, dtype=torch.double),
                torch.Size([6, 2]))
        # Create / note the expected results
        expected_result =\
            torch.tensor(
                [[1 / 6, 2 / 6, 3 / 6, 4 / 15, 5 / 15, 6 / 15],
                 [2 / 12, 4 / 12, 6 / 12, 8 / 30, 10 / 30, 12 / 30]],
                dtype=torch.double)
        expected_result = torch.transpose(expected_result, 0, 1)
        # Alias the function to be tested
        func = model._calc_probs_per_draw
        # Perform the desired test
        func_result = func(fake_utilities, fake_mapping)
        self.assertTrue(torch.allclose(expected_result, func_result))
