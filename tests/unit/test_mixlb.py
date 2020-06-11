"""
Tests the MIXLB class to ensure it is constructed correctly and that all
methods work as expected.
"""
import unittest
from unittest.mock import patch

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

    @patch.object(MIXLB, 'create_coef_tensor')
    @patch.object(MIXLB, '_calc_systematic_utilities')
    @patch.object(MIXLB, '_calc_probs_per_draw')
    def test_forward_method(self, mock_calc_probs, mock_utilities, mock_coefs):
        """
        Ensures that the forward method takes averages of the simulated
        probabilities, as expected. Other component methods are tested above.
        """
        # Create model object and set its dtype to double.
        model = MIXLB()
        model.double()
        # Create fake return values for _calc_probs_per_draw
        fake_probs_per_draw_np =\
            (np.repeat(np.arange(start=1, stop=4, dtype=np.float32)[None, :],
                       3,
                       axis=0
                       ) *
             np.arange(start=1, stop=4, dtype=np.float32)[:, None]
            )
        fake_probs_per_draw = torch.from_numpy(fake_probs_per_draw_np).double()
        fake_systematic_utilities = torch.ones((3, 3), dtype=torch.double)
        fake_coefs = torch.ones((3, 3, 3), dtype=torch.double)
        # Mock all the internal method calls except the last since those calls
        # have their own separate tests.
        mock_calc_probs.return_value = fake_probs_per_draw
        mock_utilities.return_value = fake_systematic_utilities
        mock_coefs.return_value = fake_coefs
        # Note the expected results
        expected_result = torch.tensor([2, 4, 6], dtype=torch.double)
        # Create necessary arguments for model.forward
        fake_design =\
            torch.ones(fake_probs_per_draw_np.shape, dtype=torch.double)
        # corresponds to an identity matrix of size 2x2
        fake_mapping =\
            torch.sparse.FloatTensor(torch.LongTensor([[0, 1], [0, 1]]),
                                     torch.ones(2, dtype=torch.double),
                                     torch.Size([2, 2]))
        fake_mapping_2 = fake_mapping.clone()
        fake_rvs_list =\
            [torch.zeros(fake_coefs.size()[0],
                         fake_coefs.size()[2],
                         dtype=torch.double)
             for x in range(fake_coefs.size()[1])]
        # Alias the function to be tested
        func = model.forward
        # Perform the desired test
        func_result =\
            func(fake_design, fake_mapping, fake_mapping_2, fake_rvs_list)
        self.assertTrue(torch.allclose(expected_result, func_result))

    def test_set_params_numpy(self):
        """
        Ensures that we can correctly set the parameters on the module.
        """
        # Create model object and set its dtype to double.
        model = MIXLB()
        model.double()
        # Create the new parameters, i.e. the function arguments
        new_means = 2 * model.means.detach().numpy().astype(np.float32)
        new_std_dev =\
            2 * model.std_deviations.detach().numpy().astype(np.float32)
        new_param_array = np.concatenate((new_means, new_std_dev), axis=0)
        new_param_array = new_param_array.astype(np.float32)
        # Create the expected results
        expected_means = torch.from_numpy(new_means).double()
        expected_std_dev = torch.from_numpy(new_std_dev).double()
        # Alias the function being tested
        func = model.set_params_numpy
        # Perform the desired test
        func(new_param_array)
        self.assertTrue(torch.allclose(expected_means, model.means))
        self.assertTrue(torch.allclose(expected_std_dev, model.std_deviations))
