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
from Typing import List


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

    # Needed attributes:
    # - Module parameter tensors:
    #   - parameter "means"
    #   - parameter "standard deviations"
    # - List denoting the design column indices for normal and lognormally
    #   distributed parameters
    #   - Should define a validation function that ensures that the length of
    #     the list of normally distributed random variables equals the number
    #     of standard deviation parameters
    #   - Should have a second validation function that makes sure all values
    #     in these two lists are < the length of parameter means.
    # - Standard deviation constant for lognormally distributed values
    # - Generically needed constants for the model:
    #   - number of alternatives
    #   - number of design columns
    # - Needed constants for numerical stability
    #   - min_exponent_val (minimum value that should be exponentiated)
    #   - max_exponent_val (maximum value that should be exponentiated)
    #   - min_prob_value (minimum probability value before value looks like 0)
    #   - max_prob_value (maximum probability value before value looks like 1)
    #   - max_comp_value (maximum value before overflow is likely to occur)

    def __attrs_post_init__(self):
        # Make sure that we call the constructor method of nn.Module to
        # initialize pytorch specific parameters, as they are not automatically
        # initialized by attrs
        super().__init__()

    def forward(self):
        # Should specify the computational steps for calculating the
        # probability function corresponding to `Mixed Logit B`.
        raise NotImplementedError()

    def create_coef_tensor(
        self,
        design_2d: torch.Tensor,
        rows_to_mixers: torch.sparse.FloatTensor,
        normal_rvs_list: List[torch.Tensor]) -> torch.Tensor:
        # Initialize the Tensor of coefficients to be created
        # Should have shape
        # (design_2d.shape[0], design_2d.shape[1], normal_rvs_list[0].shape[1])
        coef_tensor = None

        # Assign the values that are for coefficients not being integrated over

        # Access the list of column indices corresponding to normally
        # distributed coefficients.

        # Assign coefficients for each of the 'normally distributed' variables

        # Access the list of column indices corresponding to log-normally
        # distributed coefficients

        # Assign coefficients for each log-normally distributed variable.

        # Return the final 3D coefficient Tensor.
        return coef_tensor
