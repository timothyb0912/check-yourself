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
        # How many columns with randomly distributed coefficients are there?
        self.num_mixing_vars = len(self.mixing_variable_names)
        # Which design columns have log-normal coefficients?
        self.lognormal_design_cols =\
            [self.column_names.index(x) for x in self.lognormal_coefs]
        # Which design columns have normal coefficients?
        self.normal_design_cols =\
            [self.column_names.index(x) for x in self.normal_coefs]
        # What are the indices of the randomly distributed coefficients that
        # are log-normal and normal, respectively?
        self.lognormal_mixing_indices =\
            [self.mixing_variable_names.index(x) for x in self.lognormal_coefs]
        self.normal_mixing_indices =\
            [self.mixing_variable_names.index(x) for x in self.normal_coefs]


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
    log_normal_std = attr.ib(init=False, default=0.8326)

    ####
    # Generically needed constants for the model
    ####
    num_alternatives = attr.ib(init=False, default=6)
    num_design_columns = attr.ib(init=False, default=23)

    ####
    # Needed constants for numerical stability
    ####
    # Minimum and maximum value that should be exponentiated
    min_exponent_val = attr.ib(init=False, default=-700)
    max_exponent_val = attr.ib(init=False, default=700)
    # Maximum computational value before overflow is likely.
    max_comp_value = attr.ib(init=False, default=1e300)
    # MNL models and generalizations only have probability = 1
    # when the linear predictor = infinity
    max_prob_value = attr.ib(init=False, default=1-1e16)
    # MNL models and generalizations only have probability = 0
    # when the linear predictor = -infinity
    min_comp_value = attr.ib(init=False, default=1e-300)

    def __attrs_post_init__(self):
        # Make sure that we call the constructor method of nn.Module to
        # initialize pytorch specific parameters, as they are not automatically
        # initialized by attrs
        super().__init__()

        # Needed paramters to the module:
        # - Module parameter tensors:
        #   - parameter "means"
        #   - parameter "standard deviations"
        means = nn.Parameter(torch.ones(len(self.design_info.column_names)))
        std_deviations =\
            nn.Parameter(torch.ones(self.design_info.num_mixing_vars))

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
