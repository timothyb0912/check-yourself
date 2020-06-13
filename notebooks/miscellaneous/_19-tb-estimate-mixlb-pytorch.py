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
# The purpose of this notebook is estimate the "Mixed Logit B" model of Brownstone and Train (1998) using pytorch.

# # Notebook parameters

# +
# Declare paths to where data is or should be stored
DATA_PATH =\
    "../../data/processed/model_ready_car_data.csv"

OUTPUT_PARAM_PATH =\
    "../../models/estimated_mixlb_parameters.csv"

OUTPUT_GRADIENT_PATH =\
    "../../models/estimated_mixlb_gradient.csv"

OUTPUT_HESSIAN_PATH =\
    "../../models/estimated_mixlb_hessian.csv"

# Note needed column names
ALT_ID_COLUMN = 'alt_id'
OBS_ID_COLUMN = 'obs_id'
CHOICE_COLUMN = 'choice'
# -

# # Import needed libraries

# +
# Built-in modules
import sys
import time
from collections import OrderedDict

# Third-party modules
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import pylogit as pl
import pylogit.mixed_logit_calcs as mlc

# Local modules
sys.path.insert(0, '../../')
import src.models.mixlb as mixlb
import src.models.torch_utils as utils
from src.hessian import hessian
# -
# # Load needed data

car_df = pd.read_csv(DATA_PATH)

# # Create the corresponding MNL model
# Create a MNL model that has the same design matrix as Mixed Logit B.

# +
# Create specification and name dictionaries
mnl_spec, mnl_names = OrderedDict(), OrderedDict()

for col, display_name in mixlb.DESIGN_TO_DISPLAY_DICT.items():
    mnl_spec[col] = 'all_same'
    mnl_names[col] = display_name

# +
# Instantiate a MNL with the same design matirx as the MIXL.
mnl_model =\
    pl.create_choice_model(data=car_df,
                           alt_id_col=ALT_ID_COLUMN,
                           obs_id_col=OBS_ID_COLUMN,
                           choice_col=CHOICE_COLUMN,
                           specification=mnl_spec,
                           model_type='MNL',
                           names=mnl_names)

# Estimate the model to note reproduction of
# Brownstone & Train's MNL
mnl_model.fit_mle(np.zeros(len(mnl_names)),
                  constrained_pos=[-2, -1])

# Look at the estimation results
mnl_model.get_statsmodels_summary()
# -

# # Initialize the MIXL model

# Instantiate the model
mixl_model = mixlb.MIXLB()

# # Create objects for probability and loss calculations

# +
# Create target variables for the loss function
torch_choices =\
    torch.from_numpy(mnl_model.choices.astype(np.float32)).double()

# Get the design matrix from the original and forecast data
orig_design_matrix_np = mnl_model.design
orig_design_matrix =\
    torch.tensor(orig_design_matrix_np.astype(np.float32))

# Get the rows_to_obs and rows_to_mixers matrices.
rows_to_obs =\
    utils.create_sparse_mapping_torch(car_df[mnl_model.obs_id_col].values)
rows_to_mixers =\
    utils.create_sparse_mapping_torch(car_df[mnl_model.obs_id_col].values)

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

# # Create the objective function
# Create the function to be used by `scipy.optimize.minimize`.

# +
def make_scipy_closure(
        design,
        obs_mapping,
        mixers_mapping,
        normal_draws,
        targets,
        model,
        loss_func,
        ):
    def closure(params):
        # params -> loss, grad
        # Load the parameters onto the model
        model.set_params_numpy(params)
        # Ensure the gradients are summed starting from zero
        model.zero_grad()
        # Calculate the probabilities
        probabilities =\
            model(design_2d=design,
                  rows_to_obs=obs_mapping,
                  rows_to_mixers=mixers_mapping,
                  normal_rvs_list=normal_draws)
        # Calculate the loss
        loss = loss_func(probabilities, targets)
        # Compute the gradients
        loss.backward()
        # Get the gradient.
        grad = model.get_grad_numpy()
        # Get a value of the loss to pass around.
        loss_val = loss.item()
        return loss_val, grad
    return closure

scipy_objective =\
    make_scipy_closure(orig_design_matrix,
                       rows_to_obs,
                       rows_to_mixers,
                       normal_rvs_list,
                       torch_choices,
                       mixl_model,
                       utils.log_loss)
# -

# # Estimate MIXLB

# +
####
# Initialize parameters
####
# Initialize the model parameters to the final estimates from Brownstone & Train (1998),
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

# Set the parameters on the model
mixl_model.set_params_numpy(paper_estimates_array)
# -

####
# Compute initial log-likelihood
####
with torch.no_grad():
    # Compute the MIXL probabilities
    initial_mixl_probs =\
        mixl_model.forward(design_2d=orig_design_matrix,
                           rows_to_obs=rows_to_obs,
                           rows_to_mixers=rows_to_mixers,
                           normal_rvs_list=normal_rvs_list)

    # Compute the MIXL log-likelihood
    initial_mixl_log_likelihood =\
        -1 * utils.log_loss(initial_mixl_probs, torch_choices)

    # Compare the MIXL to MNL log-likelihoods
    msg = 'Initial MIXL: {:,.2f}\nMNL:          {:,.2f}'
    print(msg.format(initial_mixl_log_likelihood.item(), mnl_model.llf))

# +
# Perform the optimization
start_time = time.time()

optimization_results =\
    minimize(scipy_objective,
             paper_estimates_array,
             jac=True,
             method='bfgs')

end_time = time.time()
duration_sec = end_time - start_time
duration_mins = duration_sec / 60.

print('Estimation Time: {:.1f} minutes'.format(duration_mins))
# -

print('Initial Log-likelihood: {:,.2f}'.format(initial_mixl_log_likelihood))
print('Final Log-Likelihood:    {:,.2f}'.format(optimization_results['fun']))

# Look at the gradient at the final parameters
optimization_results['jac']

# Compare the final parameters to their starting values
estimates_df =\
    pd.DataFrame({'initial': paper_estimates_array,
                  'final': optimization_results['x']})
estimates_df

# # Compute the hessian

# +
# Get rid of old gradient computations
mixl_model.zero_grad()

# Compute final probabilities
final_mixl_probs =\
    mixl_model(design_2d=orig_design_matrix,
               rows_to_obs=rows_to_obs,
               rows_to_mixers=rows_to_mixers,
               normal_rvs_list=normal_rvs_list)

# Compute final loss
final_log_likelihood =\
    utils.log_loss(final_mixl_probs, torch_choices)

# Compute the hessian of the loss
hess_start_time = time.time()
final_mixlb_hessian =\
    hessian(final_log_likelihood,
            mixl_model.parameters())
hess_end_time = time.time()
hess_duration_sec = hess_end_time - hess_start_time
hess_duration_mins = hess_duration_sec / 60.
print('Hessian Computation: {:.1f} minutes'.format(hess_duration_mins))

# Get the numpy array corresponding to the hessian
final_mixlb_hessian_array = final_mixlb_hessian.numpy()

# Extract the hessian that excludes the rows and columns
# for the two constrained parameters
desired_rows =\
    np.concatenate((np.arange(0, 21), np.arange(23, 27)), axis=0)
final_mixlb_hessian_core =\
    final_mixlb_hessian_array[np.ix_(desired_rows, desired_rows)]
final_mixlb_hessian_core.shape
# -

# # Save the results

# Save the final parameters
estimates_df.final.to_csv(OUTPUT_PARAM_PATH, index=False)
# Save the final gradient
(pd.Series(optimization_results['jac'])
   .to_csv(OUTPUT_GRADIENT_PATH, index=False))
# Save the final hessian
(pd.DataFrame(final_mixlb_hessian_array)
   .to_csv(OUTPUT_HESSIAN_PATH, index=False, header=False))

# # Findings
# 1. The most unexpected finding was that none of the parameter gradients was near zero when using the parameter values from the published article.
#
# 2. When optimizing the model to get to a true local maximum of the log-likelihood function, there does not seem to be a huge difference in final results.
# The largest parameter change is the increase in the variance of non-EV utility functions.
#
# 3. Computing the hessian of the estimated parameters takes a **very** long time.
#
