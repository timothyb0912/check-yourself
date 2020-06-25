# -*- coding: utf-8 -*-
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

# The purpose of this notebook is to convert the wide-format car data to long-format. The car data comes from the mlogit package. The data description is reproduced below. Note the data originally comes from McFadden and Train (2000).
#
# #### Description
# - Cross-Sectional Dataset
# - Number of Observations: 4,654
# - Unit of Observation: Individual
# - Country: United States
#
# #### Format
# A dataframe containing :
# - choice: choice of a vehicule amoung 6 propositions
# - college: college education?
# - hsg2: size of household greater than 2?
# - coml5: commulte lower than 5 miles a day?
# - typez: body type, one of regcar (regular car), sportuv (sport utility vehicule), sportcar, stwagon (station wagon), truck, van, for each proposition z from 1 to 6
# - fuelz: fuel for proposition z, one of gasoline, methanol, cng (compressed natural gas), electric. pricez price of vehicule divided by the logarithme of income
# - rangez: hundreds of miles vehicule can travel between refuelings/rechargings
# - accz: acceleration, tens of seconds required to reach 30 mph from stop
# - speedz: highest attainable speed in hundreds of mph
# - pollutionz: tailpipe emissions as fraction of those for new gas vehicule
# - sizez: 0 for a mini, 1 for a subcompact, 2 for a compact and 3 for a mid–size or large vehicule
# - spacez: fraction of luggage space in comparable new gas vehicule
# - costz: cost per mile of travel(tens of cents). Either cost of home recharging for electric vehicule or the cost of station refueling otherwise
# - stationz: fraction of stations that can refuel/recharge vehicule
#
# #### Source
# McFadden, Daniel and Kenneth Train (2000) “Mixed MNL models for discrete response”, Journal of Applied Econometrics, 15(5), 447–470.
#
#
# Journal of Applied Econometrics data archive : http://jae.wiley.com/jae/

import pandas as pd
import numpy as np
import pylogit as pl

# # Load the Car data

wide_car = pd.read_csv("../data/raw/car_wide_format.csv")
wide_car.head().T

# # Convert the Car dataset to long-format

# Look at the columns of the car data
print(wide_car.columns.tolist())

# +
# Create the list of individual specific variables
ind_variables = wide_car.columns.tolist()[1:4]

# Specify the variables that vary across individuals and some or all alternatives
# The keys are the column names that will be used in the long format dataframe.
# The values are dictionaries whose key-value pairs are the alternative id and
# the column name of the corresponding column that encodes that variable for
# the given alternative. Examples below.
new_name_to_old_base = {'body_type': 'type{}',
                        'fuel_type': 'fuel{}',
                        'price_over_log_income': 'price{}',
                        'range': 'range{}',
                        'acceleration': 'acc{}',
                        'top_speed': 'speed{}',
                        'pollution': 'pollution{}',
                        'vehicle_size': 'size{}',
                        'luggage_space': 'space{}',
                        'cents_per_mile': 'cost{}',
                        'station_availability': 'station{}'}

alt_varying_variables =\
    {k: dict([(x, v.format(x)) for x in range(1, 7)])
     for k, v in list(new_name_to_old_base.items())}

# Specify the availability variables
# Note that the keys of the dictionary are the alternative id's.
# The values are the columns denoting the availability for the
# given mode in the dataset.
availability_variables =\
    {x: 'avail_{}'.format(x) for x in range(1, 7)}
for col in availability_variables.values():
    wide_car[col] = 1

##########
# Determine the columns for: alternative ids, the observation ids and the choice
##########
# The 'custom_alt_id' is the name of a column to be created in the long-format data
# It will identify the alternative associated with each row.
custom_alt_id = "alt_id"

# Create a custom id column that ignores the fact that this is a 
# panel/repeated-observations dataset. Note the +1 ensures the id's start at one.
obs_id_column = "obs_id"
wide_car[obs_id_column] =\
    np.arange(1, wide_car.shape[0] + 1, dtype=int)


# Create a variable recording the choice column
choice_column = "choice"
# Store the original choice column in a new variable
wide_car['orig_choices'] = wide_car['choice'].values
# Alter the original choice column
choice_str_to_value = {'choice{}'.format(x): x for x in range(1, 7)}
wide_car[choice_column] =\
    wide_car[choice_column].map(choice_str_to_value)

# Convert the wide-format data to long format
long_car =\
    pl.convert_wide_to_long(wide_data=wide_car,
                            ind_vars=ind_variables,
                            alt_specific_vars=alt_varying_variables,
                            availability_vars=availability_variables,
                            obs_id_col=obs_id_column,
                            choice_col=choice_column,
                            new_alt_id_name=custom_alt_id)

long_car.head().T
# -

# Save the long-format data
long_car.to_csv("../data/interim/car_long_format.csv",
                index=False)
