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

# The purpose of this notebook is to diagnose why I cannot straight-forwardly reproduce the MNL model used in "Brownstone, Davide and Train, Kenneth (1999). 'Forecasting new product penetration with flexible substitution patterns'. Journal of Econometrics 89: 109-129." (p. 121).

# +
import sys
from collections import OrderedDict

import scipy.stats
import pandas as pd
import numpy as np
import pylogit as pl

sys.path.insert(0, '../src')
from visualization import predictive_viz as viz

# %matplotlib inline
# -

car_df = pd.read_csv("../data/interim/car_long_format.csv")

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
    
car_df['non_electric'] =\
    (car_df['fuel_type'] != 'electric').astype(int)

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
    
car_df['methanol_commute_lte_5mi'] =\
    (car_df['methanol'] & car_df['coml5']).astype(int)
    
# Scale the range and acceleration variables
car_df['range_over_100'] = car_df['range'] / 100.0
car_df['acceleration_over_10'] = car_df['acceleration'] / 10.0
car_df['top_speed_over_100'] = car_df['top_speed'] / 100.0
car_df['vehicle_size_over_10'] = car_df['vehicle_size'] / 10.0
car_df['tens_of_cents_per_mile'] = car_df['cents_per_mile'] / 10.0

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
     ('methanol', 'EV'),
     ('methanol_commute_lte_5mi', 'Commute < 5 & EV'),
     ('methanol_and_college', 'College & EV'),
     ('compressed_natural_gas', 'CNG'),
     ('electric', 'Methanol'),
     ('electric_and_college', 'College & Methanol')]
    
for col, display_name in cols_and_display_names:
    car_mnl_spec[col] = 'all_same'
    car_mnl_names[col] = display_name


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

# Failed attempts:
# 1. Replace 'EV' with a non-EV dummy variable
#
# Successful attempts:
# 1. Switch electric and methanol fuel type labels. The idea occurred to me because my original estimated methanol dummy had a similar coefficient to the estimated electric dummy variable in the paper.
#
# # Wow. A Semantic Error?
# Somehow the 'methanol' and 'electric' columns were erroneously swapped.
#
# When I estimate the model, using 'methanol' as if it were 'electric' and vice-versa, I can recreate the model estimation results exactly.

# # Check raw data from Brownstone and Train
#
# There seems to have been an error going from their raw data to the data in mlogit. The dataset values don't match.

t_ = pd.read_table("../data/raw/mcfadden_train_2000_raw/mt-data/xmat.txt",
                   sep=r'\s*',
                   names=['col{}'.format(x) for x in range(1, 5)])

raw_df = pd.DataFrame(t_.values.reshape((4654, 156)))

# Look at the 'methanol' vehicles according to the original brownstone and train data
raw_df.iloc[0, 114:120]

# Look at the fuel types for the first six long-format rows.
# From above, the third and fourth entry should be methanol vehicles
# NOT electric vehicles.
car_df['fuel_type'].iloc[:6]
