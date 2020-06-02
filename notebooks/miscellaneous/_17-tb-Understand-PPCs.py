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
# 3. Produce 7 plots to illustrate the general model-checking procedure.
#

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

# # Load the car data

car_df = pd.read_csv("../data/processed/model_ready_car_data.csv")

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
fit_vals = car_mnl.fit_mle(initial_vals,
                           method='L-BFGS-B',
                           just_point=True)['x']
# Note ridge=1e-7 produces the same results as non-regularized MLE
car_mnl.fit_mle(fit_vals, method='BFGS')

# Look at the estimation results
car_mnl.get_statsmodels_summary()
# -

print(np.round(pd.concat([car_mnl.params,
                    car_mnl.standard_errors],
                   axis=1),
               decimals=3).to_latex())

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
num_draws = 1001
np.random.seed(325)
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

# # Make the seven desired plots

import seaborn as sbn
import matplotlib.pyplot as plt

# ### 1. Log-Likelihood plot

# +
reload(viz)

sim_log_likes =\
    viz.compute_prior_predictive_log_likelihoods(likelihood_sim_y,
                                                 car_df,
                                                 "choices",
                                                 car_mnl)

log_like_path =\
    None #'../reports/figures/log-predictive-vehicle-choice-mnl.pdf'
viz.plot_predicted_log_likelihoods(sim_log_likes,
                                   car_mnl.llf,
                                   output_file=log_like_path)

# +
body = 'regcar'
filter_idx = (car_df.body_type == body).values

current_df = car_df.loc[filter_idx]
current_probs = car_mnl.long_fitted_probs[filter_idx]
current_choices = car_mnl.choices[filter_idx]
current_sim_y = likelihood_sim_y[filter_idx, :]

current_sim_log_likes =\
    current_sim_y.T.dot(np.log(current_probs))
current_obs_log_likes =\
    current_choices.T.dot(np.log(current_probs))

current_log_like_path =\
    None #'../reports/figures/original_mnl_log_likelihood_plot_regcar.jpeg'
current_x_label = 'Log-Likelihood for individuals choosing regular cars'
viz.plot_predicted_log_likelihoods(current_sim_log_likes,
                                   current_obs_log_likes,
                                   x_label=current_x_label,
                                   fontsize=14,
                                   output_file=current_log_like_path)

# +
body = 'regcar'
filter_idx = (car_df.body_type == body).values

current_df = car_df.loc[filter_idx]
current_probs = car_mnl.long_fitted_probs[filter_idx]
current_choices = likelihood_sim_y[filter_idx, 0]
current_sim_y = likelihood_sim_y[filter_idx, 1:]

current_sim_log_likes =\
    current_sim_y.T.dot(np.log(current_probs))
current_obs_log_likes =\
    current_choices.T.dot(np.log(current_probs))

current_log_like_path =\
    None #'../reports/figures/original_mnl_log_likelihood_plot_regcar.jpeg'
current_x_label = 'Log-Likelihood for FAKE individuals choosing regular cars'
viz.plot_predicted_log_likelihoods(current_sim_log_likes,
                                   current_obs_log_likes,
                                   x_label=current_x_label,
                                   fontsize=14,
                                   output_file=current_log_like_path)

# +
reload(viz)

for body in ['regcar', 'sportcar', 'stwagon', 'van', 'truck']:
    filter_idx = (car_df.body_type == body).values

    current_df = car_df.loc[filter_idx]
    current_probs = car_mnl.long_fitted_probs[filter_idx]
    current_choices = car_mnl.choices[filter_idx]
    current_sim_y = likelihood_sim_y[filter_idx, :]

    current_sim_log_likes =\
        current_sim_y.T.dot(np.log(current_probs))
    current_obs_log_likes =\
        current_choices.T.dot(np.log(current_probs))

    current_log_like_path = None
    viz.plot_predicted_log_likelihoods(current_sim_log_likes,
                                       current_obs_log_likes,
                                       x_label='Log-Likelihood for {}'.format(body),
                                       output_file=current_log_like_path)

# +
reload(viz)

for fuel in ['cng', 'methanol', 'gasoline', 'electric']:
    filter_idx = (car_df.fuel_type == fuel).values

    current_df = car_df.loc[filter_idx]
    current_probs = car_mnl.long_fitted_probs[filter_idx]
    current_choices =\
        likelihood_sim_y[filter_idx, 5] # car_mnl.choices[filter_idx]
    current_sim_y = likelihood_sim_y[filter_idx, 1:]

    current_sim_log_likes =\
        current_sim_y.T.dot(np.log(current_probs))
    current_obs_log_likes =\
        current_choices.T.dot(np.log(current_probs))

    current_log_like_path = None
    viz.plot_predicted_log_likelihoods(current_sim_log_likes,
                                       current_obs_log_likes,
                                       x_label='Log-Likelihood for {}'.format(fuel),
                                       output_file=current_log_like_path)

# +
num_choices = car_mnl.choices.sum()

sim_mse = (likelihood_sim_y * 
           (likelihood_sim_y - 
            car_mnl.long_fitted_probs[:, None])**2).sum(axis=0) / num_choices

obs_mse = (car_mnl.choices * 
           (car_mnl.choices - 
            car_mnl.long_fitted_probs)**2).sum() / num_choices

mse_path = None
viz.plot_predicted_log_likelihoods(sim_mse,
                                   obs_mse,
                                   x_label='Mean Squared Error',
                                   output_file=mse_path)
# -

for body in ['regcar', 'sportcar', 'stwagon', 'van', 'truck']:
    filter_idx = (car_df.body_type == body).values

    current_df = car_df.loc[filter_idx]
    current_probs = car_mnl.long_fitted_probs[filter_idx]
    current_choices = car_mnl.choices[filter_idx]
    current_sim_y = likelihood_sim_y[filter_idx, :]
    
    sim_mse = (current_sim_y *
               ((current_sim_y - 
                 current_probs[:, None])**2)).sum(axis=0) / num_choices

    obs_mse = (current_choices *
               ((current_choices - 
                 current_probs)**2)).sum() / num_choices

    mse_path = None
    viz.plot_predicted_log_likelihoods(sim_mse,
                                       obs_mse,
                                       x_label='Mean Squared Error for {}'.format(body),
                                       output_file=mse_path)

for fuel in ['cng', 'methanol', 'gasoline', 'electric']:
    filter_idx = (car_df.fuel_type == fuel).values

    current_df = car_df.loc[filter_idx]
    current_probs = car_mnl.long_fitted_probs[filter_idx]
    current_choices = car_mnl.choices[filter_idx]
    current_sim_y = likelihood_sim_y[filter_idx, :]
    
    sim_mse = (current_sim_y *
               ((current_sim_y - 
                 current_probs[:, None])**2)).sum(axis=0) / num_choices

    obs_mse = (current_choices *
               ((current_choices - 
                 current_probs)**2)).sum() / num_choices

    mse_path = None
    viz.plot_predicted_log_likelihoods(sim_mse,
                                       obs_mse,
                                       x_label='Mean Squared Error for {}'.format(fuel),
                                       output_file=mse_path)

# +
num_choices = car_mnl.choices.sum()

for i in list(range(car_mnl.design.shape[1])):
    design_column = i
    column_name = car_mnl.params.index.values[design_column]

    x_values = car_mnl.design[:, design_column]

    sim_constraint = likelihood_sim_y.T.dot(x_values) / num_choices

    obs_constraint = car_mnl.choices.dot(x_values) / num_choices

    constraint_path = None
    viz.plot_predicted_log_likelihoods(sim_constraint,
                                       obs_constraint,
                                       x_label='Mean {}'.format(column_name),
                                       output_file=constraint_path)

# +
num_choices = car_mnl.choices.sum()

def calc_variance(y, x):
    num_obs = y.sum(axis=0)
    x_mean = y.T.dot(x) / num_obs
    x_squared_mean = y.T.dot(x**2) / num_obs
    x_variance = x_squared_mean - (x_mean)**2
    return x_variance

for i in list(range(car_mnl.design.shape[1])):
    design_column = i
    column_name = car_mnl.params.index.values[design_column]

    x_values = car_mnl.design[:, design_column]

    sim_variance = calc_variance(likelihood_sim_y, x_values)

    obs_variance = calc_variance(car_mnl.choices, x_values)

    variance_path = None
    viz.plot_predicted_log_likelihoods(sim_variance,
                                       obs_variance,
                                       x_label='Variance of {}'.format(column_name),
                                       output_file=variance_path)
# -

(pd.DataFrame(car_mnl.design,
              columns=car_mnl.params.index)
   .describe().T['std']**2)

# ### 2. Outcome Boxplot

# +
reload(viz)
market_path = '../reports/figures/market-share-plot-vehicle-choice-mnl.pdf'

market_dict = dict(cng='compressed_natural_gas')

viz.plot_simulated_market_shares(car_df.fuel_type.values,
                                 likelihood_sim_y,
                                 car_df.choice.values,
                                 x_label='Fuel Type',
                                 y_label='Number\nof times\nchosen',
                                 display_dict=market_dict,
                                 output_file=market_path)
# -

reload(viz)
viz.plot_simulated_market_shares(car_df.body_type.values,
                                 likelihood_sim_y,
                                 car_df.choice.values,
                                 x_label='Body Type',
                                 y_label='Number\nof times\nchosen')

# ### 3. Binned Reliability Plot

# +
reload(viz)
current_fuel = 'methanol'
filter_idx = np.where((car_df.fuel_type == current_fuel).values)[0]
# current_probs = simulated_probs[filter_idx, :]
current_probs = car_mnl.long_fitted_probs[filter_idx]
current_choices = car_mnl.choices[filter_idx]
current_sim_y = likelihood_sim_y[filter_idx, :]
current_line_label = 'Observed vs Predicted ({})'.format(current_fuel)
current_sim_label = 'Simulated vs Predicted ({})'.format(current_fuel)

current_sim_color = '#a6bddb'
current_obs_color = '#045a8d'

# viz.plot_binned_reliability(
#     current_probs,
#     current_choices,
#     sim_y=current_sim_y,
#     line_label=current_line_label,
#     line_color=current_obs_color,
#     sim_label=current_sim_label,
#     sim_line_color=current_sim_color,
#     figsize=(10, 6),
#     ref_line=True,
#     output_file='../reports/figures/reliability-plot-vehicle-choice-mnl-methanol-point.pdf')

viz.plot_binned_reliability(
    current_probs,
    current_choices,
    sim_y=current_sim_y,
    line_label=current_line_label,
    line_color=current_obs_color,
    sim_label=current_sim_label,
    sim_line_color=current_sim_color,
    figsize=(10, 6),
    ref_line=True,
    output_file='../reports/figures/reliability-plot-vehicle-choice-mnl-methanol-point.jpeg')
# -

# ###  4. Binned Marginal Model Plot

# +
current_body = 'sportuv'
selection_idx = (car_df.body_type == current_body).values

num_traces = 500
current_probs = simulated_probs[selection_idx]
current_y = car_df.loc[selection_idx, 'choice'].values
current_x = car_df.loc[selection_idx, 'price_over_log_income'].values
current_sim_y = likelihood_sim_y[selection_idx]

# filename =\
#     '../reports/figures/marginal-model-plot-vehicle-choice-mnl-suv.pdf'
filename =\
    '../reports/figures/marginal-model-plot-vehicle-choice-mnl-suv.jpeg'

viz.make_binned_marginal_model_plot(current_probs,
                                    current_y,
                                    current_x,
                                    partitions=10,
                                    sim_y=current_sim_y,
                                    y_label='Observed P(Y=SUV)',
                                    prob_label='Predicted P(Y=SUV)',
                                    sim_label='Simulated P(Y=SUV)',
                                    x_label='Binned, Mean SUV Price / ln(income)',
                                    alpha=0.5,
                                    figsize=(10, 6),
                                    output_file=filename)
# -

# ### 5. Simulated Histogram

# +
reload(viz)

filter_row = ((car_df.body_type == 'regcar') &
              (car_df.cents_per_mile == 2))
# current_title = 'Num Observations by Cents per Mile for Body = {}'
current_title = ''
# filename =\
#     '../reports/figures/histogram-vehicle-choice-mnl-regcar-operating-costs.pdf'
filename =\
    '../reports/figures/histogram-vehicle-choice-mnl-regcar-operating-costs.jpeg'

viz.plot_categorical_predictive_densities(
    car_df,
    None,
    likelihood_sim_y,
    'cents_per_mile',
    filter_row,
    car_mnl.choices,
    title=current_title.format('Regular Car'),
    filter_name='observations',
    post_color=sbn.color_palette('colorblind')[0],
    figsize=(10, 6),
    legend_loc='upper left',
    output_file=filename)

# +
reload(viz)

filter_row = ((car_df.body_type == 'regcar') &
              (car_df.cents_per_mile == 2))
# current_title = 'Num Observations by Cents per Mile for Body = {}'
current_title = ''
# filename =\
#     '../reports/figures/histogram-vehicle-choice-mnl-regcar-operating-costs.pdf'
filename =\
    '../reports/figures/histogram-vehicle-choice-mnl-regcar-operating-costs_v2.jpeg'

viz.plot_categorical_predictive_densities(
    car_df,
    None,
    likelihood_sim_y,
    'cents_per_mile',
    filter_row,
    car_mnl.choices,
    title=current_title.format('Regular Car'),
    filter_name='regular cars',
    post_color=sbn.color_palette('colorblind')[0],
    figsize=(10, 6),
    legend_loc='upper left',
    output_file=filename)
# -

# ### 6. Simulated KDE

# +
reload(viz)
current_fuel = 'electric'
filter_row = car_df.fuel_type == current_fuel
# current_title = 'KDE of Price/log(income) for {} vehicles'
current_title = ''
filename =\
    '../reports/figures/kde-vehicle-choice-mnl-electric-price.pdf'

viz.plot_simulated_kde_traces(likelihood_sim_y,
                              car_df,
                              filter_row,
                              'price_over_log_income',
                              'choice',
                              title=current_title.format(current_fuel),
                              figsize=(10, 6),
                              label='Simulated',
                              n_traces=500,
                              output_file=filename)
# -

# ### 7. Simulated CDF

# +
reload(viz)
current_body = 'sportuv'
filter_row = car_df.body_type == current_body
# current_title =\
#     'CDF of Price/log(income) for Sport Utility Vehicles'
current_title = ''
# filename =\
#     '../reports/figures/cdf-vehicle-choice-mnl-suv-price.pdf'
filename =\
    '../reports/figures/cdf-vehicle-choice-mnl-suv-price.jpeg'

viz.plot_simulated_cdf_traces(likelihood_sim_y,
                              car_df,
                              filter_row,
                              'price_over_log_income',
                              'choice',
                              label='Simulated',
                              title=current_title,
                              figsize=(10, 6),
                              output_file=filename)
# -

# ## Sandbox:
# Look at additional plots.

for body in ['regcar', 'sportcar', 'stwagon', 'van', 'truck']:
    current_body = body
    selection_idx = (car_df.body_type == current_body).values

    num_traces = 500
    current_probs = simulated_probs[selection_idx]
    current_y = car_df.loc[selection_idx, 'choice'].values
    current_x = car_df.loc[selection_idx, 'price_over_log_income'].values
    current_sim_y = likelihood_sim_y[selection_idx]

    filename = None

    viz.make_binned_marginal_model_plot(current_probs,
                                        current_y,
                                        current_x,
                                        partitions=10,
                                        sim_y=current_sim_y,
                                        y_label='Observed P(Y={})'.format(body),
                                        prob_label='Predicted P(Y={})'.format(body),
                                        sim_label='Simulated P(Y={})'.format(body),
                                        x_label='Binned, Mean {} Price / ln(income)'.format(body),
                                        alpha=0.5,
                                        figsize=(10, 6),
                                        output_file=filename)

for body in ['regcar', 'sportcar', 'stwagon', 'van', 'truck']:
    current_body = body
    filter_row = car_df.body_type == current_body
    # current_title =\
    #     'CDF of Price/log(income) for Sport Utility Vehicles'
    current_title = ''
    filename = None
    viz.plot_simulated_cdf_traces(likelihood_sim_y,
                                  car_df,
                                  filter_row,
                                  'price_over_log_income',
                                  'choice',
                                  label='Simulated ({})'.format(body),
                                  title=current_title,
                                  figsize=(10, 6),
                                  output_file=filename)

for body in ['regcar', 'sportcar', 'stwagon', 'van', 'truck']:
    filter_idx = (car_df.body_type == body).values
    current_probs = car_mnl.long_fitted_probs[filter_idx]
    current_choices = car_mnl.choices[filter_idx]
    current_sim_y = likelihood_sim_y[filter_idx, :]
    current_line_label = 'Observed vs Predicted ({})'.format(body)
    current_sim_label = 'Simulated vs Predicted ({})'.format(body)

    current_sim_color = '#a6bddb'
    current_obs_color = '#045a8d'

    reliability_path = None
    viz.plot_binned_reliability(
        current_probs,
        current_choices,
        sim_y=current_sim_y,
        line_label=current_line_label,
        line_color=current_obs_color,
        sim_label=current_sim_label,
        sim_line_color=current_sim_color,
        figsize=(10, 6),
        ref_line=True,
        output_file=reliability_path)

for fuel in ['cng', 'electric', 'gasoline']:
    current_fuel = fuel
    filter_idx = np.where((car_df.fuel_type == current_fuel).values)[0]
    current_probs = car_mnl.long_fitted_probs[filter_idx]
    current_choices = car_mnl.choices[filter_idx]
    current_sim_y = likelihood_sim_y[filter_idx, :]
    current_line_label = 'Observed vs Predicted ({})'.format(current_fuel)
    current_sim_label = 'Simulated vs Predicted ({})'.format(current_fuel)

    current_sim_color = '#a6bddb'
    current_obs_color = '#045a8d'

    reliability_path = None
    viz.plot_binned_reliability(
        current_probs,
        current_choices,
        sim_y=current_sim_y,
        line_label=current_line_label,
        line_color=current_obs_color,
        sim_label=current_sim_label,
        sim_line_color=current_sim_color,
        figsize=(10, 6),
        ref_line=True,
        output_file=reliability_path)
