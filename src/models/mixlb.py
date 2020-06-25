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
"""
# Used for type hinting
from typing import Tuple, List, Dict, Optional

# Numpy is used for numeric computation
import numpy as np
# PyTorch is used for numeric computation and automatic differentiation
import torch
# Used to access various pytorch utilities
import torch.nn as nn
# Used to get sparse matrices
import torch.sparse
# Used to access convenience functions for torch to numpy optimization
import botorch.optim.numpy_converter as numpy_converter
import botorch.optim.utils as optim_utils
from botorch.optim.numpy_converter import TorchAttr
# Use attrs for boilerplate free creation of classes
import attr


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
class DesignInfoMixlB:
    """
    Data storage class providing information about the design matrix for MIXLB.
    """
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
    # Needed constants for numerical stability
    ####
    # Minimum and maximum value that should be exponentiated
    min_exponent_val = attr.ib(init=False, default=torch.tensor(-700))
    max_exponent_val = attr.ib(init=False, default=torch.tensor(700))
    # MNL models and generalizations only have probability = 1
    # when the linear predictor = infinity
    max_prob_value = attr.ib(init=False, default=torch.tensor(1-1e-16))
    # MNL models and generalizations only have probability = 0
    # when the linear predictor = -infinity
    min_prob_value = attr.ib(init=False, default=torch.tensor(1e-40))

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
        # Enforce parameter constraints
        self.constrain_means()

    def constrain_means(self):
        """
        Ensures that we don't compute the gradients for mean parameters that
        should be constrained to zero.
        """
        # Note that parameters 21 (non_ev) and 22 (non_cng) are constrained to
        # zero because those columns are just for random effects, not means.
        self.constrained_means =\
            optim_utils.fix_features(self.means, {21: None, 22: None})

    def forward(self,
                design_2d: torch.Tensor,
                rows_to_obs: torch.sparse.FloatTensor,
                rows_to_mixers: torch.sparse.FloatTensor,
                normal_rvs_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the probabilities for `Mixed Logit B`.

        Parameters
        ----------
        design_2d : 2D torch.Tensor.
            Denotes the design matrix whose coefficients are to be computed.
        rows_to_obs : 2D torch.sparse.FloatTensor.
            Denotes the mapping between rows of `design_2d` and the
            choice observations the probabilities are being computed for.
        rows_to_mixers : 2D torch.sparse.FloatTensor.
            Denotes the mapping between rows of `design_2d` and the
            decision-makers the coefficients are randomly distributed over.
        normal_rvs_list : list of 2D torch.Tensor.
            Should have length `self.design_info.num_mixing_vars`. Each element
            of the list should be of shape `(rows_to_mixers.size()[1],
            num_draws)`. Each element should represent a draw from a standard
            normal distribution.

        Returns
        -------
        average_probabilities : 1D torch.Tensor
            Denotes the average of the probabilities for each alternative for
            each choice situation, across all random draws of the model
            coefficients.
        """
        # Get the coefficient tensor for all observations
        # This will be a 3D tensor
        coefficients =\
            self.create_coef_tensor(design_2d, rows_to_mixers, normal_rvs_list)
        # Compute the long-format systematic utilities per row and random draw.
        # This will be a 2D tensor
        systematic_utilities =\
            self._calc_systematic_utilities(design_2d, coefficients)
        # Compute the long-format probabilities for each row and random draw.
        # This will be a 2D tensor
        probabilities_per_draw =\
            self._calc_probs_per_draw(systematic_utilities, rows_to_obs)
        # Compute the long-format, average probabilities across draws.
        # This will be a 1D tensor.
        average_probabilities = torch.mean(probabilities_per_draw, 1)
        return average_probabilities

    def _get_std_deviation(self, col_name: str) -> torch.Tensor:
        """
        Parameters
        ----------
        col_name : str.
            The name of the column whose coefficient is being treated as
            randomly distributed across decision makers.

        Returns
        -------
        std_deviation : scalar torch.Tensor
            Denotes the current standard deviation of the normally or
            log-normally distributed random variable.
        """
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
        """
        Parameters
        ----------
        col_name : str.
            The name of the column whose coefficient is being treated as
            randomly distributed across decision makers.
        design_column_idx : int.
            The column index in the design matrix that matching `col_name`.
        std_deviation : scalar torch.Tensor.
            Denotes the standard deviation for the underlying normal random
            variates that are to be transformed into the distribution being
            assumed for the `col_name` coefficient.
        current_rvs : 2D torch.Tensor.
            Denotes the random variates for each decision maker (rows) and for
            each draw per decision maker (columns).

        Returns
        -------
        generated_coefs : 2D torch.Tensor.
            Normally or log-normally (based on `col_name`) distributed
            coefficients for each decision maker for each draw.
        """
        generated_coefs =\
            (self.constrained_means[design_column_idx] +
             std_deviation * current_rvs)
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
            normal_rvs_list: List[torch.Tensor]
        ) -> torch.Tensor:
        """
        Creates a 3D tensor of coefficients. These coefficients are to be
        element-wise multiplied with the design matrix to calculate the
        systematic utilities for each draw from the distributions of the
        randomly generated coefficients.

        Parameters
        ----------
        design_2d : 2D torch.Tensor.
            Denotes the design matrix whose coefficients are to be computed.
        rows_to_mixers : 2D torch.sparse.FloatTensor.
            Denotes the mapping between rows of `design_2d` and the
            decision-makers the coefficients are randomly distributed over.
        normal_rvs_list : list of 2D torch.Tensor.
            Should have length `self.design_info.num_mixing_vars`. Each element
            of the list should be of shape `(rows_to_mixers.size()[1],
            num_draws)`. Each element should represent a draw from a standard
            normal distribution.

        Returns
        -------
        coef_tensor : 3D torch.Tensor.
            Denotes the design coefficients for each decision maker and draw
            from the random coeffient distributions, in a shape amenable to
            element-wise multiplication with the design marix.
        """
        # Determine the number of draws for each set of randomly drawn values
        num_draws = normal_rvs_list[0].shape[1]
        # Initialize the coefficients as if they were all homogenous. Shape =
        # (design_2d.shape[0], design_2d.shape[1], normal_rvs_list[0].shape[1])
        coef_shape = (design_2d.shape[0], design_2d.shape[1], num_draws)
        coef_tensor =\
            torch.ones(coef_shape) * self.constrained_means[None, :, None]

        # Assign the randomly distributed coefficients
        mixing_var_iterable = enumerate(self.design_info.mixing_variable_names)
        for mixing_pos, col_name in mixing_var_iterable:
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

    def _calc_systematic_utilities(self,
                                   design_2d: torch.Tensor,
                                   coefs: torch.Tensor) -> torch.Tensor:
        """
        Calcualtes the "systematic utility", i.e. the deterministic function of
        model coefficients and the design matrix that the model's probabilities
        are based on.

        Parameters
        ----------
        design_2d : 2D torch.Tensor.
            Denotes the design matrix whose coefficients are to be computed.
        coef_tensor : 3D torch.Tensor.
            Denotes the design coefficients for each decision maker and draw
            from the random coeffient distributions, in a shape amenable to
            element-wise multiplication with the design marix.

        Returns
        -------
        sys_utilities : 2D torch.Tensor.
            The systematic utilities for each row and each draw.
        """
        sys_utilities =\
            torch.sum(design_2d[:, :, None] * coefs, 1, dtype=torch.double)
        safe_sys_utilities =\
            torch.clamp(sys_utilities,
                        self.min_exponent_val,
                        self.max_exponent_val)
        return safe_sys_utilities

    def _calc_probs_per_draw(
            self,
            sys_utilities: torch.Tensor,
            rows_to_obs: torch.sparse.FloatTensor
        ) -> torch.Tensor:
        """
        Calculates the probabilities for each row's alternative being chosen,
        given the coefficients for the current individual and random draw.

        Parameters
        ----------
        sys_utilities : 2D torch.Tensor.
            The systematic utilities for each row and each draw.
        rows_to_obs : 2D torch.sparse.FloatTensor.
            Denotes the mapping between rows of `design_2d` and the
            choice observations the probabilities are being computed for.

        Returns
        -------
        long_probs : 2D torch.Tensor
            The probabilities of each row's alternative being chosen for the
            given choice situation, based on each random draw of model
            coefficients (one draw per column).
        """
        # Compute exp(V)
        exponentiated_sys_utilities = torch.exp(sys_utilities)
        # Get denominators to compute probabilities. One row per observation.
        denominators_by_obs =\
            torch.sparse.mm(rows_to_obs.transpose(0, 1),
                            exponentiated_sys_utilities)
        # Convert denominators into a tensor of the same size as sys_utilities.
        long_denominators = torch.sparse.mm(rows_to_obs, denominators_by_obs)
        # Note we use clamp to guard against against zero probabilities.
        long_probs =\
            torch.clamp(exponentiated_sys_utilities / long_denominators,
                        min=self.min_prob_value,
                        max=self.max_prob_value)
        return long_probs

    def get_params_numpy(self) -> Tuple[
            np.ndarray, Dict[str, TorchAttr], Optional[np.ndarray]]:
        """
        Syntatic sugar for `botorch.optim.numpy_converter.module_to_array`.

        Returns
        -------
        param_array : 1D np.ndarray
            Model parameters values.
        param_dict : dict.
            String representations of parameter names are keys, and the values
            are TorchAttr objects containing shape, dtype, and device
            information about the correpsonding pytorch tensors.
        bounds : optional, np.ndarray or None.
            If at least one parameter has bounds, then these are returned as a
            2D ndarray representing the bounds for each paramaeter. Otherwise
            None.
        """
        return numpy_converter.module_to_array(self)

    def set_params_numpy(self, new_param_array: torch.Tensor) -> None:
        """
        Sets the model's parameters using the values in `new_param_array`.

        Parameters
        ----------
        new_param_array : 1D ndarray.
            Should have one element for each element of the tensors in
            `self.parameters`.

        Returns
        -------
        None.
        """
        # Get the property dictionary for this module
        _, property_dict, _ = self.get_params_numpy()
        # Set the parameters
        numpy_converter.set_params_with_array(
            self, new_param_array, property_dict)
        # Constrain the parameters
        self.constrain_means()

    def get_grad_numpy(self) -> None:
        """
        Returns the gradient of the model parameters as a 1D numpy array.
        """
        grad =\
            np.concatenate(list(x.grad.data.numpy().ravel()
                                for x in self.parameters()),
                           axis=0)
        return grad
