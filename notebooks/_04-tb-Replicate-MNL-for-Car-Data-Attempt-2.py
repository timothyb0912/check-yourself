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
# The purpose of this notebook is two-fold. In it, I aim to:
# 1. Reproduce the MNL model used in "Brownstone, Davide and Train, Kenneth (1999). 'Forecasting new product penetration with flexible substitution patterns'. Journal of Econometrics 89: 109-129." (p. 121).
# 2. 'Check' the MNL model for lack-of-fit between observable features of the data and predictions from the model.
#
# # Main Findings
# 1. Categorical or piecewise-linear specifications should be used for many variables that are currently treated as continuous in the dataset. Categorical specifications should be used if we don't wish to make predictions for values of the variable outside the currently observed vaues. Piecewise-linear specifications should be used if we will make predictions for values outside the data. Specific variables that should get this treatment include:
#    1. price
#    2. range
#    3. acceleration
#    4. top speed
#    5. pollution
#    6. operating costs
# 2. There is a lack of systematic heterogeneity with respect to vehicle body type. All variables should be interacted with vehicle body type.
# 3. There is a lack of systematic heterogeneity with respect to vehicle fuel type. The following variables should be interacted with vehicle fuel type:
#    1. price
#    2. range
#    3. top speed
#    5. pollution
#    6. operating costs

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

# # Load the final modeling dataset

car_df =\
    pd.read_csv("../data/processed/model_ready_car_data.csv")

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
# The original modeling results have been replicated. To do so, I needed to change the vehicle fuel types in the mlogit data to correct a likely transcription error.

# # MNL Model Checking

# Simulate values from the sampling distribution of coefficients
cov_matrix = np.linalg.inv(-1 * car_mnl.hessian)
mnl_sampling_dist =\
    scipy.stats.multivariate_normal(mean=car_mnl.params.values,
                                    cov=cov_matrix)

# Take Draws from the sampling distribution
num_draws = 500
simulated_coefs = mnl_sampling_dist.rvs(num_draws)
simulated_coefs.shape

# +
# Predict the model probabilities
simulated_probs =\
    car_mnl.predict(car_df,
                    param_list=[simulated_coefs.T, None, None, None])

# Simulate y from the sampling distribution
likelihood_sim_y =\
    viz.simulate_choice_vector(simulated_probs,
                               car_df['obs_id'].values,
                               rseed=1122018)
# -

# ## Price/log(income)

# Check the price variable for all of the various modes.
for body in np.sort(car_df.body_type.unique()):
    filter_row = car_df.body_type == body
    
    current_title = 'CDF of Price/log(income) for Body={}'
    
    viz.plot_simulated_cdf_traces(likelihood_sim_y,
                                  car_df,
                                  filter_row,
                                  'price_over_log_income',
                                  'choice',
                                  label='Simulated',
                                  title=current_title.format(body),
                                  figsize=(10, 6))
    
    title_2 = 'KDE of Price/log(income) for Body={}'
    viz.plot_simulated_kde_traces(likelihood_sim_y,
                                  car_df,
                                  filter_row,
                                  'price_over_log_income',
                                  'choice',
                                  label='Simulated',
                                  title=title_2.format(body),
                                  figsize=(10, 6))
    

# From the plots above, we can see that having a single price variable is not sufficient to capture the observed relationships between price and chosen body type of one's vehicle. In particular, we frequently:
#
# 1. overpredict the choice of SUV as a function of price.
# 2. underpredict the choice of station wagon, truck, and van as a function of price.

# Check the price variable for all of the various modes.
for fuel in np.sort(car_df.fuel_type.unique()):
    filter_row = car_df.fuel_type == fuel
    current_title = 'CDF of Price/log(income) for Fuel={}'
    
    viz.plot_simulated_cdf_traces(likelihood_sim_y,
                                  car_df,
                                  filter_row,
                                  'price_over_log_income',
                                  'choice',
                                  title=current_title.format(fuel),
                                  figsize=(10, 6))
    
    title_2 = 'KDE of Price/log(income) for Fuel={}'
    viz.plot_simulated_kde_traces(likelihood_sim_y,
                                  car_df,
                                  filter_row,
                                  'price_over_log_income',
                                  'choice',
                                  title=current_title.format(fuel),
                                  figsize=(10, 6))

# From the plots above we can see that our model does not adequately capture the relationship between price/log(income) and fuel type.

# ## Range

# Check the price variable for all of the various modes.
for body in np.sort(car_df.body_type.unique()):
    filter_row = car_df.body_type == body
    
    current_title = 'Num Observations by Range for Body={}'
    
    viz.plot_categorical_predictive_densities(
        car_df,
        None,
        likelihood_sim_y,
        'range',
        filter_row,
        car_mnl.choices,
        title=current_title.format(body),
        figsize=(10, 6))
    

car_df.loc[car_df.choice == 1, 'range'].value_counts()

# As seen above, range is a categorical variable.
#
# From examining the plots of the simulated y-values from the sampling distribution, we can see that our fitted model leads to severe misfit between various features of the range variable:
#
# 1. we underpredict the number of station wagons with range in {50, 125, 150, 200}
# 2. we overpredict the number of station wagons with range in {400}.
# 3. we overpredict the number of trucks with range in {50}.
# 4. we underpredict the number of trucks with range in {150}.
# 5. we underpredict the number of vans with range in {150}.
# 6. we overpredict the number of vans with range in {200}.

# Check the price variable for all of the various modes.
for fuel in np.sort(car_df.fuel_type.unique()):
    filter_row = car_df.fuel_type == fuel
    current_title = 'Num Observations by Range for Fuel={}'

    viz.plot_categorical_predictive_densities(
        car_df,
        None,
        likelihood_sim_y,
        'range',
        filter_row,
        car_mnl.choices,
        title=current_title.format(fuel),
        figsize=(10, 6))

# Based on the plots above, we can see that our model systematically:
# 1. over predicts compressed natural gas vehicles with range = 150
# 2. over predicts methanol vehicles with range = 250
# 3. under predicts methanol vehicles with range = 300

# ## Acceleration

car_df.loc[car_df.choice == 1, 'acceleration'].value_counts()

# Check the price variable for all of the various modes.
for body in np.sort(car_df.body_type.unique()):
    filter_row = car_df.body_type == body
    
    current_title = 'Num Observations by Acceleration for Body={}'
    
    viz.plot_categorical_predictive_densities(
        car_df,
        None,
        likelihood_sim_y,
        'acceleration',
        filter_row,
        car_mnl.choices,
        title=current_title.format(body),
        figsize=(10, 6))
    

# From above, we see that we:
# 1. underpredict the number of sports cars with Acceleration = 2.5
# 2. underpredict the number of station wagons with Acceleration = 4.0
# 3. underpredict the number of vans with Acceleration = 4.0
# 4. overpredict the number of vans with Acceleration = 6.0

# Check the price variable for all of the various modes.
for fuel in np.sort(car_df.fuel_type.unique()):
    filter_row = car_df.fuel_type == fuel
    current_title = 'Num Observations by Acceleration for Fuel={}'

    viz.plot_categorical_predictive_densities(
        car_df,
        None,
        likelihood_sim_y,
        'acceleration',
        filter_row,
        car_mnl.choices,
        title=current_title.format(fuel),
        figsize=(10, 6))

# With respect to acceleration by vehicle fuel type, our model appears to adequately capture the observed relationships.

# ## Top speed

car_df.loc[car_df.choice == 1, 'top_speed'].value_counts()

# Check the price variable for all of the various modes.
for body in np.sort(car_df.body_type.unique()):
    filter_row = car_df.body_type == body
    
    current_title = 'Num Observations by Top Speed for Body={}'
    
    viz.plot_categorical_predictive_densities(
        car_df,
        None,
        likelihood_sim_y,
        'top_speed',
        filter_row,
        car_mnl.choices,
        title=current_title.format(body),
        figsize=(10, 6))
    

# What we can see is that we:
# 1. under predict regular car with top speed = 100
# 2. over predict sports cars with top speeds = 55
# 3. under predict sports car with top speed = 95
# 4. over predict sports utility vehicles with top speeds = 55
# 5. under predict sports utility vehicles with top speed = 85
# 6. under predict station wagons with top speed = 65
# 7. under predict station wagons with top speed = 100
# 8. over predict trucks with top speeds = 100
# 9. under predict trucks with top speeds = 110

# Check the price variable for all of the various modes.
for fuel in np.sort(car_df.fuel_type.unique()):
    filter_row = car_df.fuel_type == fuel
    current_title = 'Num Observations by Top Speed for Fuel={}'

    viz.plot_categorical_predictive_densities(
        car_df,
        None,
        likelihood_sim_y,
        'top_speed',
        filter_row,
        car_mnl.choices,
        title=current_title.format(fuel),
        figsize=(10, 6))

# From the plots above, we see that our model systematically:
# 1. under predicts CNG vehicles with top speed = 95
# 2. over predicts electric vehicles with top speed = 55
# 3. under predicts electric vehicles with top speed = 85
# 4. under predicts gasoline vehicles with top speed = 85
# 5. over predicts gasoline vehicles with top speed = 95

# ## Pollution

car_df.loc[car_df.choice == 1,
           'pollution'].value_counts().sort_index()

# Check the price variable for all of the various modes.
for body in np.sort(car_df.body_type.unique()):
    filter_row = car_df.body_type == body
    
    current_title = 'Num Observations by Pollution for Body={}'
    
    viz.plot_categorical_predictive_densities(
        car_df,
        None,
        likelihood_sim_y,
        'pollution',
        filter_row,
        car_mnl.choices,
        title=current_title.format(body),
        figsize=(10, 6))
    

# What we see is that we:
# 1. over predict sports cars with pollution in {0, 0.75}
# 2. under predict sports cars with pollution in {0.1, 0.5}
# 3. under predict station wagons with pollution in {0, 0.1}
# 4. over predict station wagons with pollution = 0.6
# 5. over predict trucks with pollution = 0
# 6. over predict vans with pollution = 0.5
# 7. under predict vans with pollution = 0.6

# Check the price variable for all of the various modes.
for fuel in np.sort(car_df.fuel_type.unique()):
    filter_row = car_df.fuel_type == fuel
    current_title = 'Num Observations by Pollution for Fuel={}'

    viz.plot_categorical_predictive_densities(
        car_df,
        None,
        likelihood_sim_y,
        'pollution',
        filter_row,
        car_mnl.choices,
        title=current_title.format(fuel),
        figsize=(10, 6))

# From the plots above, we see that our model systematically:
# 1. over predicts gasoline vehicles with pollution = 0.75

# ## Vehicle Size

car_df.loc[car_df.choice == 1, 'vehicle_size'].value_counts()

# Check the price variable for all of the various modes.
for body in np.sort(car_df.body_type.unique()):
    filter_row = car_df.body_type == body
    current_title = 'Num Observations by Vehicle Size for Body={}'

    viz.plot_categorical_predictive_densities(
        car_df,
        None,
        likelihood_sim_y,
        'vehicle_size',
        filter_row,
        car_mnl.choices,
        title=current_title.format(body),
        figsize=(10, 6))

# What we can see is that we systematically:
# 1. under predict regular cars with vehicle size = 0
# 2. under predict sports cars with vehicle size = 0
# 3. over predict sports cars with vehicle size = 3
# 4. over predict sports utility vehicle with vehicle size = 0
# 5. under predict sports utility vehicle with vehicle size = 3
# 6. over predict trucks with vehicle size = 0
# 7. under predict trucks with vehicle size = 1
# 8. over predict vans with vehicle size = 0
# 9. over predict vans with vehicle size = 1

# Check the price variable for all of the various modes.
for fuel in np.sort(car_df.fuel_type.unique()):
    filter_row = car_df.fuel_type == fuel
    current_title = 'Num Observations by Vehicle Size for Fuel={}'

    viz.plot_categorical_predictive_densities(
        car_df,
        None,
        likelihood_sim_y,
        'vehicle_size',
        filter_row,
        car_mnl.choices,
        title=current_title.format(fuel))

# From the plots above, we can see that our model appears to adequately represent the relationship between vehicle size and fuel type.

# ## Luggage Space

car_df.loc[car_df.choice == 1, 'luggage_space'].value_counts()

# Check the price variable for all of the various modes.
for body in np.sort(car_df.body_type.unique()):
    filter_row = car_df.body_type == body
    current_title = 'Num Observations by Luggage Space for Body={}'

    viz.plot_categorical_predictive_densities(
        car_df,
        None,
        likelihood_sim_y,
        'luggage_space',
        filter_row,
        car_mnl.choices,
        title=current_title.format(body),
        figsize=(10, 6))

# We can see that we systematically:
#
# 1. under predict regular cars with luggage space = 0.7
# 2. under predict station wagons with luggage space = 0.7
#

# Check the price variable for all of the various modes.
for fuel in np.sort(car_df.fuel_type.unique()):
    filter_row = car_df.fuel_type == fuel
    current_title = 'Num Observations by Luggage Space for Fuel={}'

    viz.plot_categorical_predictive_densities(
        car_df,
        None,
        likelihood_sim_y,
        'luggage_space',
        filter_row,
        car_mnl.choices,
        title=current_title.format(fuel),
        figsize=(10, 6))

# With respect to vehicle fuel type, the observed distribution of luggage space seems adequately captured.

# # Operating Cost

car_df.loc[car_df.choice == 1,
           'cents_per_mile'].value_counts().sort_index()

# Check the price variable for all of the various modes.
for body in np.sort(car_df.body_type.unique()):
    filter_row = car_df.body_type == body
    current_title = 'Num Observations by Cents per Mile for Body={}'

    viz.plot_categorical_predictive_densities(
        car_df,
        None,
        likelihood_sim_y,
        'cents_per_mile',
        filter_row,
        car_mnl.choices,
        title=current_title.format(body),
        figsize=(10, 6))

# From the plots above, we see that the model systematically:
# 1. under predict regular car with cents_per_mile = 2
# 2. over predict regular car with cents_per_mile = 6
# 3. over predict sports car with cents_per_mile = 1
# 4. under predict sports car with cents_per_mile = 6
# 5. over predict sports utility vehicle with cents per mile = 1
# 6. under predict sports utility vehicle with cents per mile = 8
# 7. under predict station wagon with cents per mile = 1
# 8. under predict station wagon with cents per mile = 2
# 9. over predict station wagon with cents per mile = 6
# 10. over predict truck with cents per mile = 1
# 11. under predict truck with cents per mile = 8

# Check the price variable for all of the various modes.
for fuel in np.sort(car_df.fuel_type.unique()):
    filter_row = car_df.fuel_type == fuel
    current_title = 'Num Observations by Cents per Mile for Fuel={}'

    viz.plot_categorical_predictive_densities(
        car_df,
        None,
        likelihood_sim_y,
        'cents_per_mile',
        filter_row,
        car_mnl.choices,
        title=current_title.format(fuel),
        figsize=(10, 6))

# # Look at model reliability

# Check the price variable for all of the various modes.
for body in np.sort(car_df.body_type.unique()):
    filter_idx = np.where((car_df.body_type == body).values)[0]
    current_probs = simulated_probs[filter_idx, :]
    current_choices = car_mnl.choices[filter_idx]
    current_sim_y = likelihood_sim_y[filter_idx, :]
    current_line_label = 'Observed vs Predicted ({})'.format(body)
    current_sim_label = 'Simulated vs Predicted ({})'.format(body)

    viz.plot_binned_reliability(
        current_probs,
        current_choices,
        alpha=0.5,
        sim_y=current_sim_y,
        line_label=current_line_label,
        sim_label=current_sim_label,
        figsize=(10, 6),
        ref_line=True)

# Check the price variable for all of the various modes.
for fuel in np.sort(car_df.fuel_type.unique()):
    filter_idx = np.where((car_df.fuel_type == fuel).values)[0]
    current_probs = simulated_probs[filter_idx, :]
    current_choices = car_mnl.choices[filter_idx]
    current_sim_y = likelihood_sim_y[filter_idx, :]
    current_line_label = 'Observed vs Predicted ({})'.format(fuel)
    current_sim_label = 'Simulated vs Predicted ({})'.format(fuel)

    viz.plot_binned_reliability(
        current_probs,
        current_choices,
        alpha=0.5,
        sim_y=current_sim_y,
        line_label=current_line_label,
        sim_label=current_sim_label,
        figsize=(10, 6),
        ref_line=True)

# From the plots above, we see that the model systematically:
# 1. over predict compressed natural gas vehicles with cents per mile == 6.
# 2. over predict electric vehicles with cents_per_mile = 1
# 3. under predict electric vehicles with cents_per_mile = 6
# 4. under predict gas vehicles with cents_per_mile = 2
# 5. over predict gas vehicles with cents per mile = 4
# 6. under predict methanol vehicles with cents per mile = 2
# 7. over predict methanol vehicles with cents per mile = 6
# 8. produces uncalibrated probabilities with respect to all body types except sport utility vehicles
# 9. produces uncalibrated probabilities with respect to all fuel types


