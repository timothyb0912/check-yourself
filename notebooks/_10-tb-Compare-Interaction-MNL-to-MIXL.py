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
# The purpose of this notebook is to compare the case study predictive results for the MNL model with systematic heterogeneity to the log-normal MIXL model from Brownstone and Train (1998).

# +
import sys
from copy import deepcopy
from collections import OrderedDict
from collections import defaultdict

import scipy.stats
import pandas as pd
import numpy as np

import pylogit as pl

sys.path.insert(0, '../src/')
from visualization import predictive_viz as viz

# %matplotlib inline
# -

# # Load the car data

car_df = pd.read_csv("../data/processed/model_ready_car_data.csv")
forecast_df = pd.read_csv("../data/processed/forecast_car_data.csv")


# # Create the model specification

def create_specification_dict(list_of_cols_and_names):
    # Create interaction variables for the various body types
    body_names = ['sports_utility_vehicle',
                  'sports_car',
                  'station_wagon',
                  'truck',
                  'van']

    non_body_or_fuel_vars = ['price_over_log_income',
                             'price_over_log_income_lte_3',
                             'price_over_log_income_gt_3',
                             'range_over_100',
                             'acceleration_over_10',
                             'top_speed_over_100',
                             'pollution',
                             'vehicle_size_over_10',
                             'tens_of_cents_per_mile']

    body_interactions = defaultdict(lambda : [])

    for body in body_names:
        for interaction_var in non_body_or_fuel_vars:
            new_name = interaction_var + "_for_" + body
            # Store the new variable name
            body_interactions[interaction_var].append(new_name)

    # Create interaction variables for the various fuel types
    fuel_names = ['electric',
                  'compressed_natural_gas',
                  'methanol']

    fuel_interaction_vars = ['price_over_log_income',
                             'price_over_log_income_lte_3',
                             'price_over_log_income_gt_3',
                             'range_over_100',
                             'top_speed_over_100',
                             'pollution',
                             'vehicle_size_over_10',
                             'tens_of_cents_per_mile']

    fuel_interactions = defaultdict(lambda : [])

    for fuel in fuel_names:
        for interaction_var in fuel_interaction_vars:
            new_name = interaction_var + "_for_" + fuel
            # Store the new variable name
            fuel_interactions[interaction_var].append(new_name)
            
    # Create specification and name objects
    spec_dict, name_dict = OrderedDict(), OrderedDict()
            
    for col, display_name in list_of_cols_and_names:
        if col in body_interactions:
            for interaction_col in body_interactions[col]:
                suffix = interaction_col[interaction_col.rfind("for_") + 4:]
                new_display_name = display_name + " ({})".format(suffix)

                if car_df[interaction_col].unique().size == 1:
                    continue

                spec_dict[interaction_col] = 'all_same'
                name_dict[interaction_col] = new_display_name

            for interaction_col in fuel_interactions[col]:
                suffix = interaction_col[interaction_col.rfind("for_") + 4:]
                new_display_name = display_name + "({})".format(suffix)

                if car_df[interaction_col].unique().size == 1:
                    continue

                spec_dict[interaction_col] = 'all_same'
                name_dict[interaction_col] = new_display_name

        spec_dict[col] = 'all_same'
        name_dict[col] = display_name
        
    return spec_dict, name_dict



# +
orig_cols_and_display_names =\
    [('price_over_log_income_lte_3', 'Price over log(income) <= 3'),
     ('price_over_log_income_gt_3', 'Price over log(income) > 3'),
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
    
interaction_mnl_spec, interaction_mnl_names =\
    create_specification_dict(orig_cols_and_display_names)
# -

# # Estimate the expanded and original MNL models

# +
# Determine the number of index coefficients for the interaction MNL
num_index_coefs = len(interaction_mnl_names)

# Initialize the interaction mnl model object
interaction_model =\
    pl.create_choice_model(data=car_df,
                           alt_id_col='alt_id',
                           obs_id_col='obs_id',
                           choice_col='choice',
                           specification=interaction_mnl_spec,
                           model_type='MNL',
                           names=interaction_mnl_names)
    
interaction_model.fit_mle(np.zeros(num_index_coefs))

interaction_model.get_statsmodels_summary()

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

# Initialize the mnl model
simple_mnl = pl.create_choice_model(data=car_df,
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
fit_vals = simple_mnl.fit_mle(initial_vals,
                              method='L-BFGS-B',
                              just_point=True)['x']
# Note ridge=1e-7 produces the same results as non-regularized MLE
simple_mnl.fit_mle(fit_vals, method='BFGS')

# Look at the estimation results
simple_mnl.get_statsmodels_summary()
# -

# # Make predictions

# Create a set of values to use for grouping
grouping_series = [forecast_df.vehicle_size,
                   forecast_df.fuel_type,
                   forecast_df.body_type]

# +
# Get forecast probabilities using the interaction MNL
mnl_forecast_probs =\
    pd.Series(interaction_model.predict(forecast_df))
    
# Get forecast probabilities using the log-normal MIXL model
mixl_forecast_probs =\
    (pd.read_csv("../data/processed/lognormal_mixl_probs_mle_forecast.csv",
                 header=None)
       .iloc[:, 0])
    
# Get the original probabilities using the interaction MNL
mnl_original_probs =\
    pd.Series(interaction_model.long_fitted_probs)

# Get the original probabilities using the log-normal MIXL
mixl_original_probs =\
    pd.read_csv("../data/processed/lognormal_mixl_probs_mle.csv",
                header=None).iloc[:, 0]
    
# Get forecast probabilities using the simple MNL
simple_mnl_forecast_probs =\
    pd.Series(simple_mnl.predict(forecast_df))

# Get the original probabilities using the simple MNL
simple_mnl_original_probs =\
    pd.Series(simple_mnl.long_fitted_probs)
# -

# Ensure the forecast probabilities for large gas cars are
# higher than the original probabilities for large gas cars
large_gas_car_idx = ((car_df['body_type'] == 'regcar') &
                     (car_df['vehicle_size'] == 3) &
                     (car_df['fuel_type'] == 'gasoline')).values
num_stupid_forecasts =\
    ((mixl_forecast_probs >
      mixl_original_probs)[large_gas_car_idx]).sum()
print("{:,} stupid forecasts".format(num_stupid_forecasts))

# Look at the total number of forecasted observations
# choosing large gas cars under the baseline and increased
# price scenarios with Brownstone and Train's Mixed Logit B
(mixl_original_probs[large_gas_car_idx].sum(),
 mixl_forecast_probs[large_gas_car_idx].sum())

# Look at the total number of forecasted observations
# choosing large gas cars under the baseline and increased
# price scenarios with the new expanded MNL model.
(mnl_original_probs[large_gas_car_idx].sum(),
 mnl_forecast_probs[large_gas_car_idx].sum())


# Create a function that will calculate the desired percent
# changes in the predicted mode share
def calc_mode_share_change(orig_prob_series,
                           new_prob_series,
                           grouping_series,
                           num_obs,
                           name=None):
    """
    Calculate the relative change in predicted shares by group.
    """
    new_shares =\
        (new_prob_series.groupby(grouping_series)
                        .agg(np.sum) / num_obs)

    orig_shares =\
        (orig_prob_series.groupby(grouping_series)
                        .agg(np.sum) / num_obs)

    change_in_shares = new_shares - orig_shares
    
    relative_change = change_in_shares / orig_shares

    if isinstance(name, str):
        relative_change.name = name
    
    return relative_change



# +
# Calculate the relative change using the interaction MNL and the
# log-normal MIXL model.
num_obs = interaction_model.nobs
relative_change_mnl =\
    calc_mode_share_change(mnl_original_probs,
                           mnl_forecast_probs,
                           grouping_series,
                           num_obs,
                           name='interaction_mnl')

relative_change_mixl =\
    calc_mode_share_change(mixl_original_probs,
                           mixl_forecast_probs,
                           grouping_series,
                           num_obs,
                           name='lognormal-mixl')

relative_change_simple_mnl =\
    calc_mode_share_change(simple_mnl_original_probs,
                           simple_mnl_forecast_probs,
                           grouping_series,
                           num_obs,
                           name='simple_mnl')

# +
big_change =\
    (((relative_change_mnl >= 2 * relative_change_mixl) & (relative_change_mixl > 0)) |
     ((relative_change_mnl <= 0.5 * relative_change_mixl) & (relative_change_mixl > 0)) |
     ((relative_change_mnl <= 2 * relative_change_mixl) & (relative_change_mixl < 0)) |
     ((relative_change_mnl >= 0.5 * relative_change_mixl) & (relative_change_mixl < 0)))
    
differences =\
    pd.concat([relative_change_mnl.loc[big_change],
               relative_change_mixl.loc[big_change],
               relative_change_simple_mnl.loc[big_change]],
              axis=1)
# -

differences

relative_change_mixl.get_value((3, 'gasoline', 'regcar'))

relative_change_mnl.get_value((3, 'gasoline', 'regcar'))

sep = "="
print("Log-normal Mixed Logit")
for size in range(4):
    print("Size {}".format(size))
    print(relative_change_mixl[size])
    print(sep*20)


sep = "="
print("Expanded MNL")
for size in range(4):
    print("Size {}".format(size))
    print(relative_change_mnl[size])
    print(sep*20)


relative_change_mixl.sort_values(ascending=False).iloc[:10]

relative_change_mnl.sort_values(ascending=False).iloc[:10]

relative_change_simple_mnl.sort_values(ascending=False).iloc[:10]


