# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.5
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

# +
# PS2 - CE264
# GSI: Mustapha Harb - Mengqiao Yu
# Good Reference for this homework:
# https://github.com/timothyb0912/pylogit/blob/master/examples/notebooks/Main%20PyLogit%20Example.ipynb

# importing the requried libraries
from collections import OrderedDict    # For recording the model specification

import pandas as pd                    # For file input/output
import numpy as np                     # For vectorized math operations

import pylogit as pl                   # For MNL model estimation and
                                       # conversion from wide to long format
import warnings
warnings.filterwarnings("ignore")
# -

# # Load raw data

# reading the data file
data_path = '../data/raw/Air_Travel_Survey.csv'
data_01 = pd.read_csv(data_path, sep=",")

#look at the columns and the data
data_01.columns

# ## Overview for binomial logit in python

# ### Step 0: Load the data
# ### Step 1: Define necessary variables and convert the data to long format.
# ### Step 2: Variable creations and transformations
# ### Step 3: Model specification
# ### Step 4: Run the model and analyze the results

# ## Step 1: Define necessary variables and convert the data to long format.
# We need to specify five elements to construct a long format dataset in order to run the model under PyLogit.
#
# (1.1) Individual related variables: the columns in the dataset that are specific to a given individual, regardless of what alternative is being considered. (e.g. gender)
#
# (1.2) Alternative related variables (e.g. travel time).
#
# (1.3) Altervative availabilities.
#
# (1.4) Alternative and observation ids.
#
# (1.5) The choice column.

# (1.1)
# Create the list of individual specific variables
ind_variables = data_01.columns.tolist()[:14]
print("ind_variables are:\n{}".format(ind_variables))

# +
# (1.2)
# Specify the variables that vary across individuals and some or all alternatives
# The keys are the column names that will be used in the long format dataframe.
# The values are dictionaries whose key-value pairs are the alternative id and
# the column name of the corresponding column that encodes that variable for
# the given alternative.

# {key1: value1, key2: value2}

alt_varying_variables = {u'aircraft_type': dict([(1, 'a1aircraft'),
                                                 (2, 'a2aircraft')]),
                          u'departure_time': dict([(1, 'a1departMAM'),
                                                   (2, 'a2departMAM')]),
                          u'connections': dict([(1, 'a1connections'),
                                                (2, 'a2connections')]),
                          u'travel_time': dict([(1, 'a1travtime'),
                                                (2, 'a2travtime')]),
                          u'arrival_time': dict([(1, 'a1arriveMAM'),
                                                 (2, 'a2arriveMAM')]),
                          u'time_diff': dict([(1, 'a1timediff'),
                                              (2, 'a2timediff')]),
                          u'performance': dict([(1, 'a1performance'),
                                                (2, 'a2performance')]),
                          u'fare': dict([(1, 'a1fare'),
                                         (2, 'a2fare')]),
                          u'airline': dict([(1, 'a1airline'),
                                            (2, 'a2airline')])}

# -

# (1.3)
# Specify the availability variables
# Note that the keys of the dictionary are the alternative id's.
# The values are the columns denoting the availability for the
# given alternative in the dataset.
availability_variables = {1: 'a1_AV', 2: 'a2_AV'}

# +
# (1.4)
# Identify the alternative associated with each row.
custom_alt_id = "alternative_id"

# Create a custom id column that ignores the fact that this is a
# panel/repeated-observations dataset.
obs_id_column = "choiceSituationID"
# -

# (1.5)
# Create a variable recording the choice column
choice_column = "choice"

# Perform the conversion to long-format
data_long =\
    pl.convert_wide_to_long(data_01,
                            ind_variables,
                            alt_varying_variables,
                            availability_variables,
                            obs_id_column,
                            choice_column,
                            new_alt_id_name=custom_alt_id)
# Look at the resulting long-format dataframe
data_long.head(5).T

# ## Step 2: Variable creations and transformations

# +
# Scale variables so the estimated coefficients are of similar magnitudes
# Scale travel time by 60 to convert raw units (minutes) to hours
data_long["travel_time_hrs"] = data_long["travel_time"] / 60.0

# Scale the fare column by 100 to convert raw units ($) to 100$
data_long["fare_100$"] = data_long["fare"] / 100.0

# Create dummy variables
data_long["fare_over500$"] = (data_long["fare_100$"] > 500).astype(int)

# data_long["interation_term"] =\
#     data_long["gender"] * data_long["legroom"]

# Create a human-readable airline column
data_long['airline_text'] =\
    data_long['airline'].map(
        {1: 'american',
         2: 'continental',
         3: 'delta',
         4: 'jet_blue',
         5: 'southwest',
         6: 'united',
         7: 'us_airways',
         8: 'other'}
    )

# Determine if the individual has a basic or elite FFP membership
# with each alternative's airline.
airline_text_to_ffp_column =\
    {'american': 'AA_FFP',
     'continental': 'CO_FFP',
     'delta': 'DL_FFP',
     'jet_blue': 'B6_FFP',
     'southwest': 'WN_FFP',
     'united': 'UA_FFP',
     'us_airways': 'US_FFP'
    }

data_long['basic_membership'] =\
    sum((data_long['airline_text'] == key) *
        (data_long[value] == 2)
        for key, value in airline_text_to_ffp_column.items()
       ).astype(int)

data_long['elite_membership'] =\
    sum((data_long['airline_text'] == key) *
        (data_long[value] == 3)
        for key, value in airline_text_to_ffp_column.items()
       ).astype(int)

# Determine if this alternative features a widebody aircraft
data_long['big_plane'] = (data_long['aircraft_type'] == 1).astype(int)

# Determine if this itinerary contains a red-eye flight, defined here
# as a flight departing between 8pm and 6am
data_long['red_eye'] =\
    ((data_long['departure_time'] < (6 * 60)) |
     (data_long['departure_time'] > (20 * 60))).astype(int)
# -

# ## Step 3: Model specification

# +
# specifying the utility equations

# NOTE: - Specification and variable names must be ordered dictionaries.
#       - Keys should be variables within the long format dataframe.
#         The sole exception to this is the "intercept" key.
#       - For the specification dictionary, the values should be lists
#         of integers or or lists of lists of integers. Within a list,
#         or within the inner-most list, the integers should be the
#         alternative ID's of the alternative whose utility specification
#         the explanatory variable is entering. Lists of lists denote
#         alternatives that will share a common coefficient for the variable
#         in question.

basic_specification = OrderedDict()
basic_names = OrderedDict()

# Case A: alternative specific
basic_specification["travel_time_hrs"] = [1, 2]
basic_names["travel_time_hrs"] = ['Travel Time, units:hrs (Alt 1)',
                                  'Travel Time, units:hrs (Alt 2)']

# Case B: generic: hw2
# basic_specification["travel_time_hrs"] = [[1, 2]]
# basic_names["travel_time_hrs"] = ['Travel Time, units:hrs']

# Case C: only for one
# basic_specification["travel_time_hrs"] = [1]
# basic_names["travel_time_hrs"] = ['Travel Time, units:hrs Alternative 1']

basic_specification["fare_100$"] = [1, 2]
basic_names["fare_100$"] =\
    ['Fare, units:hundredth (Alt 1)',
     'Fare, units:hundredth (Alt 2)']

basic_specification['red_eye'] = [1, 2]
basic_names['red_eye'] =\
    ['Red Eye (Alt 1), base:False',
     'Red Eye (Alt 2), base:False']

basic_specification['connections'] = [1, 2]
basic_names['connections'] =\
    ['Number of Connections (Alt 1)',
     'Number of Connections (Alt 2)']

basic_specification['big_plane'] = [1, 2]
basic_names['big_plane'] =\
    ['Widebody Aircraft (Alt 1), base:False',
     'Widebody Aircraft (Alt 2), base:False']

basic_specification['performance'] = [1, 2]
basic_names['performance'] =\
    ['On-Time Performance (%) (Alt 1)',
     'On-Time Performance (%) (Alt 2)']

basic_specification['basic_membership'] = [1, 2]
basic_names['basic_membership'] =\
    ['Basic FFP Membership (Alt 1), base:False',
     'Basic FFP Membership (Alt 2), base:False']

basic_specification['elite_membership'] = [1, 2]
basic_names['elite_membership'] =\
    ['Elite FFP Membership (Alt 1), base:False',
     'Elite FFP Membership (Alt 2), base:False']


#basic_specification["intercept"] = [1, 2]
# basic_names["intercept"] = ['ASC Alternative 1',
#                            'ASC Alternative 2']
# -

# ## Now! Let's estimate the model and show the results

# Estimate the binary logit model (
air_travel_logit =\
    pl.create_choice_model(data=data_long,
                           alt_id_col=custom_alt_id,
                           obs_id_col=obs_id_column,
                           choice_col=choice_column,
                           specification=basic_specification,
                           model_type="MNL",
                           names=basic_names)

basic_names.values()

# +
# Specify the initial values and method for the optimization.
# 4 being the total number of parameters to be estimated
num_params = sum(map(len, basic_names.values()))
initial_values = np.zeros(num_params)

air_travel_logit.fit_mle(initial_values)
# -

# Look at the estimation results
air_travel_logit.get_statsmodels_summary()


# # Prepare oneself for model checking
#
# To use the interactive model checking demonstration, there are two options.
#
# 1. Students can clone or download the `ce264_march10th_2020` branch of the `check-yourself` github repo, place their notebook into this directory, and directly make use of the model checking functions in the `src.visiualization.predictive_viz` module.
#
# 2. Students can copy the function below, run it locally on their computer, and upload the resulting file to Binder for use there.

# +
def make_directory_if_necessary(filename):
    """
    Creates the directories for the given file if they do not already exit.
    """
    import os
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        # Guard against race condition where directories have been created
        # between the os.path.exists call and the os.makedirs call.
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    return None

def package_model_for_binder(df, fitted_model, temp_dir='./temp'):
    import os
    import json
    import shutil
    ####
    # Save all needed objects in the temporary directory
    ####
    # Create a path for the asymptotic covariance matrix
    cov_path = os.path.join(temp_dir, 'cov.csv')
    # Create the temporary directory if needed
    make_directory_if_necessary(cov_path)
    # Save the asymptotic covariance matrix
    fitted_model.cov.to_csv(cov_path, index=True)

    # Save the dataframe used to estimate the model
    df_path = os.path.join(temp_dir, 'df.csv')
    df.to_csv(df_path, index=False)

    # Save the estimated parameters
    param_path = os.path.join(temp_dir, 'params.csv')
    fitted_model.params.to_csv(param_path)

    # Save the alt_id_col, obs_id_col, and choice_col
    col_dict =\
        {'alt_id_col': fitted_model.alt_id_col,
         'obs_id_col': fitted_model.obs_id_col,
         'choice_col': fitted_model.choice_col}
    col_dict_path = os.path.join(temp_dir, 'col_dict.json')
    with open(col_dict_path, 'w') as fpath:
        json.dump(col_dict, fpath)

    # Save the model specification and name dictionaries
    spec_path = os.path.join(temp_dir, 'spec.json')
    with open(spec_path, 'w') as fpath:
        json.dump(fitted_model.specification, fpath)

    name_path = os.path.join(temp_dir, 'names.json')
    with open(name_path, 'w') as fpath:
        json.dump(fitted_model.name_spec, fpath)

    # Zip the temporary directory
    shutil.make_archive('temp', 'zip', root_dir=temp_dir)
    return None


# -

# Test out the function for packaging results for binder
package_model_for_binder(data_long, air_travel_logit)
