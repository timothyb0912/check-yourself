# -*- coding: utf-8 -*-
"""
Contains the PyTorch implementation of "Mixed Logit B" from

Brownstone, David, and Kenneth Train. "Forecasting new product
penetration with flexible substitution patterns." Journal of
econometrics 89.1-2 (1998): 109-129.

To Do:
- Add validators to attributes in DesignInfoMixlB
  - Should define a validation function that ensures that the length of
    the list of normally distributed random variables equals the number
    of standard deviation parameters
  - Should have a second validation function that makes sure all values
    in these two lists are < the length of parameter means.
- Add unit tests for MIXLB methods:
  - _get_std_deviation
  - _get_generated_coefs
  - create_coef_tensor
"""
# PyTorch is used for numeric computation and automatic differentiation
import torch
# Used to access various pytorch utilities
import torch.nn as nn
# Used to get sparse matrices
import torch.sparse
# Used for numeric computation
import numpy as np
# Use attrs for boilerplate free creation of classes
import attr
# Used for type hinting
from typing import List


# List the parameter names for the design matrix columns
DESIGN_COLUMN_NAMES =\
    ["neg_price_over_log_income",
     "range_over_100",
     "neg_acceleration_over_10",
     "top_speed_over_100",
     "neg_pollution",
     "vehicle_size_over_10",
     "big_enough",
     "luggage_space",
     "neg_tens_of_cents_per_mile",
     "station_availability",
     "sports_utility_vehicle",
     "sports_car",
     "station_wagon",
     "truck",
     "van",
     "electric",
     "electric_commute_lte_5mi",
     "electric_and_college",
     "compressed_natural_gas",
     "methanol",
     "methanol_and_college",
     "non_ev",
     "non_cng",
    ]

DESIGN_TO_DISPLAY_DICT =\
    dict([("neg_price_over_log_income", 'Neg Price over log(income)'),
          ('range_over_100', 'Range (units: 100mi)'),
          ("neg_acceleration_over_10", 'Neg Acceleration (units: 0.1sec)'),
          ('top_speed_over_100', 'Neg Top speed (units: 0.01mph)'),
          ("neg_pollution", 'Neg Pollution'),
          ('vehicle_size_over_10', 'Size'),
          ('big_enough', 'Big enough'),
          ('luggage_space', 'Luggage space'),
          ("neg_tens_of_cents_per_mile", 'Neg Operation cost'),
          ('station_availability', 'Station availability'),
          ('sports_utility_vehicle', 'Sports utility vehicle'),
          ('sports_car', 'Sports car'),
          ('station_wagon', 'Station wagon'),
          ('truck', 'Truck'),
          ('van', 'Van'),
          ('electric', 'EV'),
          ('electric_commute_lte_5mi', 'Commute < 5 & EV'),
          ('electric_and_college', 'College & EV'),
          ('compressed_natural_gas', 'CNG'),
          ('methanol', 'Methanol'),
          ('methanol_and_college', 'College & Methanol'),
          ('non_ev', 'Non Electric-Vehicle'),
          ('non_cng', 'Non Compressed Natural Gas')])

MIXING_VARIABLES =\
    ['neg_price_over_log_income',
     'neg_acceleration_over_10',
     'neg_pollution',
     'neg_tens_of_cents_per_mile',
     'range_over_100',
     'top_speed_over_100',
     'big_enough',
     'station_availability',
     'vehicle_size_over_10',
     'luggage_space',
     'non_ev',
     'non_cng']

@attr.s
class DesignInfoMixlB(object):
    ####
    # To do: Add validators to objects
    #   - Should define a validation function that ensures that the length of
    #     the list of normally distributed random variables equals the number
    #     of standard deviation parameters
    #   - Should have a second validation function that makes sure all values
    #     in these two lists are < the length of parameter means.
    ####
    # Need to know the various columns of the design matrix, in order.
    column_names =\
        attr.ib(init=False, default=list(DESIGN_TO_DISPLAY_DICT.keys()))
    # Need to know the display names of the design matrix's index coefficients.
    design_to_display_dict =\
        attr.ib(init=False, default=DESIGN_TO_DISPLAY_DICT)
    # Need to know, in order, columns with randomly distributed coefficients
    mixing_variable_names = attr.ib(init=False, default=MIXING_VARIABLES)
    # Which columns correspond to log-normally distributed variables?
    lognormal_coef_names = attr.ib(init=False, default=MIXING_VARIABLES[:-4])
    # Which columns correspond to normally distributed variables?
    normal_coef_names = attr.ib(init=False, default=MIXING_VARIABLES[-4:])

    def __attrs_post_init__(self):
        # How many columns have randomly distributed coefficients?
        self.num_mixing_vars = len(self.mixing_variable_names)
        # Which design columns correspond to the mixing variable names?
        self.mixing_to_design_cols =\
            {x: self.column_names.index(x) for x in self.mixing_variable_names}
        # Map normally distributed coefficients names to their indices
        self.mixing_to_normal_indices =\
            dict(zip(self.normal_coef_names,
                     range(len(self.normal_coef_names))))


# eq=False enables nn.Module hashing and thereby internal C++ usage for pytorch
# repr=False ensures we don't overwrite the nn.Module string representation.
# For more info, see https://stackoverflow.com/questions/57291307/
# pytorch-module-with-attrs-cannot-get-parameter-list
@attr.s(eq=False, repr=False)
class MIXLB(nn.Module):
    """"
    PyTorch implementation of `Mixed Logit B` in [1].

    References
    ----------
    [1] Brownstone, David, and Kenneth Train. "Forecasting new product
    penetration with flexible substitution patterns." Journal of
    econometrics 89.1-2 (1998): 109-129.
    """
    # Should store all needed information required to specifcy the
    # computational steps needed to calculate the probability function
    # corresponding to `Mixed Logit B`

    # Info denoting the design columns, the indices for normal and lognormally
    # distributed parameters, etc.
    design_info = attr.ib(init=False, default=DesignInfoMixlB())

    # Standard deviation constant for lognormally distributed values
    log_normal_std = attr.ib(init=False, default=torch.tensor(0.8326))

    ####
    # Generically needed constants for the model
    ####
    num_alternatives = attr.ib(init=False, default=torch.tensor(6))
    num_design_columns = attr.ib(init=False, default=torch.tensor(23))

    ####
    # Needed constants for numerical stability
    ####
    # Minimum and maximum value that should be exponentiated
    min_exponent_val = attr.ib(init=False, default=torch.tensor(-700))
    max_exponent_val = attr.ib(init=False, default=torch.tensor(700))
    # Maximum computational value before overflow is likely.
    max_comp_value = attr.ib(init=False, default=torch.tensor(1e300))
    # MNL models and generalizations only have probability = 1
    # when the linear predictor = infinity
    max_prob_value = attr.ib(init=False, default=torch.tensor(1-1e16))
    # MNL models and generalizations only have probability = 0
    # when the linear predictor = -infinity
    min_comp_value = attr.ib(init=False, default=torch.tensor(1e-300))

    def __attrs_post_init__(self):
        # Make sure that we call the constructor method of nn.Module to
        # initialize pytorch specific parameters, as they are not automatically
        # initialized by attrs
        super().__init__()

        # Needed paramters tensors for the module:
        #   - parameter "means" and "standard deviations"
        self.means =\
            nn.Parameter(torch.arange(len(self.design_info.column_names),
                                      dtype=torch.double))
        self.std_deviations =\
            nn.Parameter(torch.arange(len(self.design_info.normal_coef_names),
                                      dtype=torch.double))

    def forward(self):
        # Should specify the computational steps for calculating the
        # probability function corresponding to `Mixed Logit B`.
        raise NotImplementedError()

    def _get_std_deviation(self, col_name: str) -> torch.Tensor:
        if col_name in self.design_info.normal_coef_names:
            mixing_position_idx =\
                self.design_info.mixing_to_normal_indices[col_name]
            return self.std_deviations[mixing_position_idx]
        if col_name in self.design_info.lognormal_coef_names:
            return self.log_normal_std
        msg =\
            ('`col_name`: {} MUST be in '.format(col_name) +
             '`self.design_info.mixing_variable_names`')
        raise ValueError(msg)

    def _get_generated_coefs(self,
                             col_name: str,
                             design_column_idx: int,
                             std_deviation: torch.Tensor,
                             current_rvs: torch.Tensor) -> torch.Tensor:
        generated_coefs =\
            self.means[design_column_idx] + std_deviation * current_rvs
        if col_name in self.design_info.lognormal_coef_names:
            generated_coefs =\
                torch.exp(torch.clamp(generated_coefs,
                                      self.min_exponent_val,
                                      self.max_exponent_val))
        return generated_coefs

    def create_coef_tensor(
        self,
        design_2d: torch.Tensor,
        rows_to_mixers: torch.sparse.FloatTensor,
        normal_rvs_list: List[torch.Tensor]) -> torch.Tensor:
        # Determine the number of draws for each set of randomly drawn values
        num_draws = normal_rvs_list[0].shape[1]
        # Initialize the coefficients as if they were all homogenous. Shape =
        # (design_2d.shape[0], design_2d.shape[1], normal_rvs_list[0].shape[1])
        coef_shape = (design_2d.shape[0], design_2d.shape[1], num_draws)
        coef_tensor =\
            torch.ones(coef_shape) * self.means[None, :, None]

        # Assign the randomly distributed coefficients
        for mixing_pos, col_name in self.design_info.mixing_variable_names:
            # Get the design column and position in `normal_rvs_list`
            design_column_idx =\
                self.design_info.mixing_to_design_cols[col_name]
            # Get the randomly distributed coefficient's standard deviation
            current_std_deviation = self._get_std_deviation(col_name)
            # Get the current standard normal randomly generated values
            current_rvs = normal_rvs_list[mixing_pos]
            # Get the 2D array of randomly generated coefficients, across all
            # decision makers, and across all draws.
            generated_coefs =\
                self._get_generated_coefs(col_name,
                                          design_column_idx,
                                          current_std_deviation,
                                          current_rvs)
            # Get the 2D array of randomly generated coefficients, across all
            # design matrix rows, and across all draws.
            # shape = (design_2d.shape[0], num_draws)
            generated_coefs_for_design =\
                torch.sparse.mm(rows_to_mixers, generated_coefs)
            # Assign the generated coefficients
            coef_tensor[:, design_column_idx, :] = generated_coefs_for_design

        # Return the final 3D coefficient Tensor.
        return coef_tensor
