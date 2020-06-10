# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Purpose
# The purpose of this notebook is to replicate notebook `_09-tb-Calculate-lognormal-MIXL-probs` using the pytorch implementation of MIXLB.

# +
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.sparse as sparse
import pandas as pd

import pylogit as pl
import pylogit.mixed_logit_calcs as mlc
import pylogit.choice_tools as ct

sys.path.insert(0, '../../')
import src.models.mixlb as mixlb
# -
# # Load needed data

car_df = pd.read_csv("../../data/processed/model_ready_car_data.csv")
forecast_df = pd.read_csv("../../data/processed/forecast_car_data.csv")

# # Estimate the MNL model

# +
# Create specification and name dictionaries
mnl_spec, mnl_names = OrderedDict(), OrderedDict()

orig_cols_and_display_names =\
    [("neg_price_over_log_income", 'Neg Price over log(income)'),
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
     ('non_cng', 'Non Compressed Natural Gas')]

for col, display_name in orig_cols_and_display_names:
    mnl_spec[col] = 'all_same'
    mnl_names[col] = display_name

# +
# Estimate an mnl with the same basic specification as the MIXL.
mnl_model =\
    pl.create_choice_model(data=car_df,
                           alt_id_col='alt_id',
                           obs_id_col='obs_id',
                           choice_col='choice',
                           specification=mnl_spec,
                           model_type='MNL',
                           names=mnl_names)

mnl_model.fit_mle(np.zeros(len(mnl_names)),
                  constrained_pos=[-2, -1])

# Look at the estimation results
mnl_model.get_statsmodels_summary()
# -

# # Initialize the MIXL model

# +
# Instantiate the model
mixl_model = mixlb.MIXLB()

# Set the model parameters to the final estimates from Brownstone & Train (1998),
# taking care of the typo from the published paper.
mean_array =\
    np.array([-1.5983748481622846, #-5.999,
              -0.877,
              -0.302,
              -1.364,
              -0.711,
               1.541,
              -1.748,
               1.563,
              -0.071,
              -0.741,
               0.897,
               0.698,
              -1.508,
              -1.094,
              -0.819,
              -0.905,
               0.359,
               0.770,
               0.621,
               0.476,
               0.335,
               0,
               0])

std_dev_array =\
    np.array([6.808, 5.380, 2.289, 0.971])

paper_estimates_array =\
    np.concatenate((mean_array, std_dev_array), axis=0)
mixl_model.set_params_numpy(paper_estimates_array)


# -

# # Create arguments to calculate model probabilities

def create_sparse_mapping_torch(id_array):
    mapping_scipy =\
        ct.create_sparse_mapping(id_array).tocoo()
    torch_mapping_indices =\
        torch.LongTensor(torch.from_numpy(
            np.concatenate((mapping_scipy.row[None, :],
                            mapping_scipy.col[None, :]),
                           axis=0).astype(np.int_)))
    torch_mapping_values =\
        (torch.from_numpy(mapping_scipy.data.astype(np.float32))
              .double())
    num_rows = mapping_scipy.data.size
    num_cols = ct.get_original_order_unique_ids(id_array).size
    mapping_torch =\
        sparse.FloatTensor(
            torch_mapping_indices,
            torch_mapping_values,
            torch.Size([num_rows, num_cols]))
    return mapping_torch


# +
# Get the design matrix
orig_design_matrix_np = mnl_model.design
orig_design_matrix = torch.tensor(orig_design_matrix_np)

# Get the rows_to_obs and rows_to_mixers matrices.
rows_to_obs =\
    create_sparse_mapping_torch(car_df[mnl_model.obs_id_col].values)
rows_to_mixers =\
    create_sparse_mapping_torch(car_df[mnl_model.obs_id_col].values)



####
# Get the normal random variates.
####
# Determine the number of draws being used for the mixed logit
num_draws = 250
# Determine the number of observations with randomly distributed
# sensitivities
num_mixers = car_df.obs_id.unique().size

# Get the random draws needed for the draws of each coeffcient
# Each element in the list will be a 2D ndarray of shape
# num_mixers by num_draws
normal_rvs_list_np =\
    mlc.get_normal_draws(num_mixers,
                         num_draws,
                         mixl_model.design_info.num_mixing_vars,
                         seed=601)
normal_rvs_list = [torch.from_numpy(x).double() for x in normal_rvs_list_np]
# -

# # Compare MIXL probabilities to MNL

# Compute the MIXL probabilities
mixl_probs =\
    mixl_model.forward(design_2d=orig_design_matrix,
                       rows_to_obs=rows_to_obs,
                       rows_to_mixers=rows_to_mixers,
                       normal_rvs_list=normal_rvs_list)
# Compute the MIXL log-likelihood
torch_choices =\
    torch.from_numpy(mnl_model.choices.astype(np.float32)).double()
mixl_log_likelihood =\
    torch.sum(torch_choices * torch.log(mixl_probs))
mixl_log_likelihood

# Compare the MIXL to MNL log-likelihoods
msg = 'MIXL: {:,.2f}\nMNL:  {:,.2f}'
print(msg.format(mixl_log_likelihood.item(), mnl_model.llf))
