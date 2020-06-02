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


# # Create the new model specification

# #### Create variables for the new model
#
# I need 3 things:
# 1. Interactions for the separate categories of body type with respect to all variables (except maybe luggage space)
# 2. Interactions for the separate categories of fuel type with respect to:
#    1. price
#    2. range
#    3. top speed
#    4. pollution
#    5. operating costs
# 3. Piecewise linear specifications of:
#    1. price
#    2. range
#    3. acceleration
#    4. top speed
#    5. pollution
#    6. operating costs
# 4. <strike>Change specification of price to [x, ln(x)]</strike>
#    1. (Not done to reduce model complexity.)

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

# # Try an mnl with body and fuel interactions and piecewise linear price

# +
# Determine the number of index coefficients for the full interaction MNL
num_index_coefs = len(interaction_mnl_names)

# Initialize the full interaction mnl model object
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
# -

# # Check the new model

# +
# Create the sampling distribution object
cov_matrix = interaction_model.cov
mnl_sampling_dist =\
    scipy.stats.multivariate_normal(
        mean=interaction_model.params.values, cov=cov_matrix)

# Take random draws from the sampling distribution
num_draws = 1000
np.random.seed(325)
simulated_coefs = mnl_sampling_dist.rvs(num_draws)
simulated_coefs.shape

# Predict the model probabilities
simulated_probs =\
    interaction_model.predict(car_df,
                              param_list=[simulated_coefs.T,
                                          None, None, None])

# Simulate y from the sampling distribution
posterior_simulated_y =\
    viz.simulate_choice_vector(simulated_probs,
                               car_df['obs_id'].values,
                               rseed=1122018)
# -

# Check the price variable for all of the various modes.
for body in np.sort(car_df.body_type.unique()):
    filter_row = car_df.body_type == body
    
    current_title = 'CDF of Price/log(income) for Body={}'
    
    viz.plot_simulated_cdf_traces(posterior_simulated_y,
                                  car_df,
                                  filter_row,
                                  'price_over_log_income',
                                  'choice',
                                  label='Simulated',
                                  title=current_title.format(body),
                                  figsize=(10, 6))
    
    title_2 = 'KDE of Price/log(income) for Body={}'
    viz.plot_simulated_kde_traces(posterior_simulated_y,
                                  car_df,
                                  filter_row,
                                  'price_over_log_income',
                                  'choice',
                                  label='Simulated',
                                  title=title_2.format(body),
                                  figsize=(10, 6))
    

# From above, we see that we still have "mild" problems with the price variable. I'll move on though because there isn't much time.

# # Key Takeaways
#
# Accounting for systematic heterogeneity improved the model fit more than accounting for unobserved heterogeneity.
#
# Accounting for sytematic heterogeneity <i>may</i> have improved the model fit on substantive features more than accounting for unobserved heterogeneity.
#
#
