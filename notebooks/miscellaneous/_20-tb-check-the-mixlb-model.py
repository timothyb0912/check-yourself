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
# This notebook's purpose is to produce the model checking plots of the MIXLB model that will be needed in the 3rd version of the paper for ArXiv.

# # Notebook parameters

# +
# Declare paths to where data is or should be stored
DATA_PATH =\
    "../../data/processed/model_ready_car_data.csv"

PARAM_PATH =\
    "../../models/estimated_mixlb_parameters.csv"

HESSIAN_PATH =\
    "../../models/estimated_mixlb_hessian.csv"

FIGURE_DIR =\
    "../../reports/figures/mixlb"

# Note needed column names
ALT_ID_COLUMN = 'alt_id'
OBS_ID_COLUMN = 'obs_id'
CHOICE_COLUMN = 'choice'

# Note the number of samples being drawn for the model checking
NUM_SAMPLES = 200
# -

# # Import modules

# +
# Built-in modules
import sys
import time
import pathlib
from collections import OrderedDict

# Third-party modules
import torch
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

# Local modules
sys.path.insert(0, '../../')
import src.models.mixlb as mixlb
# -

# # Load needed data

car_df = pd.read_csv(DATA_PATH)
estimated_params =\
    np.loadtxt(PARAM_PATH, delimiter=',', skiprows=1)
estimated_hessian = np.loadtxt(HESSIAN_PATH, delimiter=',')

# # Create sampling distribution

# Extract the portion of the hessian pertaining to estimated
# parameters, excluding rows and columns for the fixed parameters
desired_rows =\
    np.concatenate((np.arange(0, 21), np.arange(23, 27)), axis=0)
hessian_core =\
    estimated_hessian[np.ix_(desired_rows, desired_rows)]

# Note we don't multiply by -1 because this is the hessian
# of the log-loss as opposed to the log-likelihood. The
# -1 is already included in the log-loss definition.
asymptotic_cov = np.linalg.inv(hessian_core)

asymptotic_sampling_dist =\
    multivariate_normal(mean=estimated_params[desired_rows],
                        cov=asymptotic_cov)

# # Sample $\left( y, \theta \right)$ from posterior distribution



# # Produce desired plots




















