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

# The purpose of this notebook is two-fold. In it, I aim to:
# 1. Reproduce the MNL model used in "Brownstone, Davide and Train, Kenneth (1999). 'Forecasting new product penetration with flexible substitution patterns'. Journal of Econometrics 89: 109-129." (p. 121).
# 2. 'Check' the MNL model for lack-of-fit between observable features of the data and predictions from the model.

# +
import sys
from collections import OrderedDict

import scipy.stats
import pandas as pd
import numpy as np
import pylogit as pl

sys.path.insert(0, '../src/')
from visualization import predictive_viz as viz

# %matplotlib inline
# -

# # Load the car data

car_df = pd.read_csv("../data/interim/car_long_format.csv")

# # Create the necessary variables

car_df.head().T

# +
# Create the 'big_enough' variable
car_df['big_enough'] =\
    (car_df['hsg2'] & (car_df['vehicle_size'] == 3)).astype(int)

# Determine the type of car
car_df['sports_utility_vehicle'] =\
    (car_df['body_type'] == 'sportuv').astype(int)

car_df['sports_car'] =\
    (car_df['body_type'] == 'sportcar').astype(int)
    
car_df['station_wagon'] =\
    (car_df['body_type'] == 'stwagon').astype(int)

car_df['truck'] =\
    (car_df['body_type'] == 'truck').astype(int)

car_df['van'] =\
    (car_df['body_type'] == 'van').astype(int)

# Determine the car's fuel usage
car_df['electric'] =\
    (car_df['fuel_type'] == 'electric').astype(int)

car_df['compressed_natural_gas'] =\
    (car_df['fuel_type'] == 'cng').astype(int)

car_df['methanol'] =\
    (car_df['fuel_type'] == 'methanol').astype(int)

# Determine if this is an electric vehicle with a small commute
car_df['electric_commute_lte_5mi'] =\
    (car_df['electric'] & car_df['coml5']).astype(int)

# See if this is an electric vehicle for a college educated person
car_df['electric_and_college'] =\
    (car_df['electric'] & car_df['college']).astype(int)

# See if this is a methanol vehicle for a college educated person
car_df['methanol_and_college'] =\
    (car_df['methanol'] & car_df['college']).astype(int)
    
# Scale the range and acceleration variables
car_df['range_over_100'] = car_df['range'] / 100.0
car_df['acceleration_over_10'] = car_df['acceleration'] / 10.0
car_df['top_speed_over_100'] = car_df['top_speed'] / 100.0
car_df['vehicle_size_over_10'] = car_df['vehicle_size'] / 10.0
car_df['tens_of_cents_per_mile'] = car_df['cents_per_mile'] / 10.0
# -

car_df.loc[car_df.choice == 1, 'fuel_type'].value_counts()

# # Create the utility specification

# +
car_mnl_spec, car_mnl_names = OrderedDict(), OrderedDict()

cols_and_display_names =\
    [('price_over_log_income', 'Price over log(income)'),
     ('range_over_100', 'Range (units: 100mi)'),
     ('acceleration_over_10', 'Acceleration (units: 0.1sec)'),
     ('top_speed_over_100', 'Top speed (units: 0.01mph)'),
     ('pollution', 'Pollution'),
     ('vehicle_size_over_10', 'Size'),
     ('big_enough', 'Big enough'),
     ('luggage_space', 'Luggage space'),
     ('tens_of_cents_per_mile', 'Operation cost'),
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
     ('methanol_and_college', 'College & Methanol')]
    
for col, display_name in cols_and_display_names:
    car_mnl_spec[col] = 'all_same'
    car_mnl_names[col] = display_name

# -

# # Estimate the MNL model

# +
# Initialize the mnl model
car_mnl = pl.create_choice_model(data=car_df,
                                 alt_id_col='alt_id',
                                 obs_id_col='obs_id',
                                 choice_col='choice',
                                 specification=car_mnl_spec,
                                 model_type='MNL',
                                 names=car_mnl_names)

# Create the initial variables for model estimation
num_vars = len(car_mnl_names)
initial_vals = np.zeros(num_vars)

# Estimate the mnl model
car_mnl.fit_mle(initial_vals, method='BFGS')

# Look at the estimation results
car_mnl.get_statsmodels_summary()
# -

# # Replication Results
#
# The original modeling results cannot be replicated. When using the same model specification as the original authors, my coefficient estimates are different than those obtained in the original study.
#
# The major differences seem to be with the various fuel type variables and their interactions. I am not sure why.
#
# Using the coefficient estimates from the paper does not work either.
#
# My suspicion is that my variables are not defined the same way as in the paper.

# ### See if paper results can be replicated:

# +
paper_vals =\
    np.array([-0.185,
               0.350,
              -0.716,
               0.261,
              -0.444,
               0.935,
               0.143,
               0.501,
              -0.768,
               0.413,
               0.820,
               0.637,
              -1.437,
              -1.017,
              -0.799,
              -0.179,
               0.198,
               0.443,
               0.345,
               0.313,
               0.228])
    
np.log(car_mnl.predict(car_df,
                       param_list=[paper_vals, None, None, None],
                       return_long_probs=False,
                       choice_col='choice')).sum()
# -

# The answer appears to be no.
#
# The results from "Brownstone, Davide and Train, Kenneth (1999). 'Forecasting new product penetration with flexible substitution patterns'. Journal of Econometrics 89: 109-129." cannot be directly reproduced using the data in `car_long_format.csv`.
