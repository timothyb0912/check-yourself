"""
Tests the InputMixl class to ensure it is constructed correctly and that
input validation works as expected.
"""
import unittest

import numpy as np

from src.models.base_model_inputs import InputMixl

class InputMixlTests(unittest.TestCase):
    """
    Unit test class for storing the various tests of the InputMixl class.
    """
    def setUp(self):
        # Set a seed for reproducibility
        np.random.seed(456)

        # Specify testing constants
        self.num_draws = 10

        # Store the fake data needed for the tests
        self.fake_alt_ids = np.array([1, 2, 1, 2], dtype=int)
        self.fake_obs_ids = np.array([1, 1, 2, 2], dtype=int)
        self.fake_design = np.arange(1, 5)[:, None]

        # Note the number of observations
        self.num_obs = np.unique(self.fake_obs_ids).size
        self.fake_normal_rvs_list =\
            [np.random.normal(size=(self.num_obs, self.num_draws))]

        return None

    def test_input_mixl_constructor(self):
        # Initialize the input object
        input_object =\
            InputMixl(self.fake_design,
                      self.fake_alt_ids,
                      self.fake_obs_ids,
                      self.fake_normal_rvs_list)

        # Test that the object is of the correct type

        # Test that the instance has the correct attributes

        # Test that the values of the attribute instances have the correct id

        return None
