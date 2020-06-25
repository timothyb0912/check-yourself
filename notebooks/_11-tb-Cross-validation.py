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
import time
from collections import OrderedDict, defaultdict

import scipy.stats
import pandas as pd
import numpy as np

import pylogit as pl

from tqdm import tqdm

sys.path.insert(0, '../src')
from visualization import predictive_viz as viz

# %matplotlib inline
# -

# # Load the car data

car_df = pd.read_csv("../data/processed/model_ready_car_data.csv")


# # Create specification and name dictionaries

def create_interaction_spec_and_names(list_of_cols_and_names):
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
# Create the specification and names for the original MNL model
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


# +
# Create the specification and names for the interaction MNL model
interaction_mnl_spec_full, interaction_mnl_names_full =\
    OrderedDict(), OrderedDict()

interaction_cols_and_display_names =\
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
    create_interaction_spec_and_names(interaction_cols_and_display_names)
# -

# # Set up the cross-validation

# +
# Determine the number of folds
n_folds = 10

# Set a random seed for reproducibility
np.random.seed(910)

# Shuffle the original observation ids
shuffled_obs_ids = np.sort(car_df.obs_id.unique())
np.random.shuffle(shuffled_obs_ids)

# Determine the number of observations for each fold
n_obs = shuffled_obs_ids.shape[0]
n_obs_per_fold = viz._determine_bin_obs(n_obs, n_folds)

# Initialize a list to store the fold assignments
obs_in_fold = []
# Initialize the count of assigned observations
assigned_obs = 0

# Determine the observations for each fold
for i in xrange(n_folds):
    # Get the number of observations for the current fold
    current_n_obs = n_obs_per_fold[i]
    # Determine the start and end positions to slice the
    # vector of observation ids at
    idx_start = assigned_obs
    idx_end = assigned_obs + current_n_obs
    # Select and store the observations for the i'th fold
    obs_in_fold.append(shuffled_obs_ids[idx_start:idx_end])
    # Increment the count of assigned observations
    assigned_obs += current_n_obs

# Perform a sanity check to make sure all is well
assert all([n_obs_per_fold[i] == obs_in_fold[i].shape[0]
            for i in xrange(n_folds)])


# -

# # Perform the cross-validation

def create_and_estimate_mnl(data,
                            spec,
                            names,
                            alt_col='alt_id',
                            obs_col='obs_id',
                            choice_col='choice'):
    # Initialize the mnl model object
    car_mnl = pl.create_choice_model(data=data,
                                     alt_id_col=alt_col,
                                     obs_id_col=obs_col,
                                     choice_col=choice_col,
                                     specification=spec,
                                     model_type='MNL',
                                     names=names)

    # Create the initial variables for model estimation
    num_vars = len(names)
    initial_vals = np.zeros(num_vars)

    # Estimate the mnl model
    car_mnl.fit_mle(initial_vals,
                    method='BFGS',
                    print_res=False)
    
    return car_mnl


# +
# Initialize an array to hold the
# log-likelihoods on the held-out folds
test_log_likelihoods = np.empty((n_folds, 2), dtype=float)

# Populate the array
for test_fold in tqdm(xrange(n_folds), desc='Cross-validating'):
    # Get the test observation ids
    test_obs_ids = obs_in_fold[test_fold]

    # Generate the test and training datasets
    test_df = car_df.loc[car_df.obs_id.isin(test_obs_ids)]
    train_df = car_df.loc[~car_df.obs_id.isin(test_obs_ids)]

    # Estimate the original and interaction MNL models
    orig_mnl = create_and_estimate_mnl(train_df,
                                       car_mnl_spec,
                                       car_mnl_names)

    interaction_mnl =\
        create_and_estimate_mnl(train_df,
                                interaction_mnl_spec,
                                interaction_mnl_names)

    # Make predictions on the held-out data
    orig_predictions = orig_mnl.predict(test_df)
    interaction_predictions = interaction_mnl.predict(test_df)
    
    # Isolate the test outcomes
    test_y = test_df.choice.values
    
    # Calculate test log-likelihoods
    orig_log_likelihood = test_y.dot(np.log(orig_predictions))
    interaction_log_likelihood =\
        test_y.dot(np.log(interaction_predictions))
    
    # Store the test log-likelihoods
    test_log_likelihoods[test_fold] =\
        [orig_log_likelihood, interaction_log_likelihood]
        
# Create a dataframe of the cross-validation performance
cv_df = pd.DataFrame(test_log_likelihoods, columns=['Original', 'Expanded'])
print('Individual cross-validation results:')
print(cv_df)

print('\nAverage cross-validation results:')
print(cv_df.mean())
