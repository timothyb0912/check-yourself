# -*- coding: utf-8 -*-
"""
This file prepares the final modeling data needed to reproduce Brownstone and
Train's (1998) original MNL model, their Mixed Logit results, and the
'expanded' MNL model created as part of this case study. This file is meant to
be run from the src/features directory.
"""
from collections import defaultdict
import pandas as pd
import numpy as np


def interaction_model_transform(df):
    """
    Take a dataframe that has all necessary variables for the
    regular MNL model and ensure that it has the needed variables
    for the interaction MNL model.
    """
    # Create piecewise linear price variables
    df['price_over_log_income_lte_3'] =\
        np.clip(df['price_over_log_income'], None, 3)

    df['price_over_log_income_gt_3'] =\
        (df['price_over_log_income'] > 3) * (df['price_over_log_income'] - 3)

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
            # Create the data for the interaction variable
            df[new_name] = df[body] * df[interaction_var]

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
            # Create the data for the interaction variable
            df[new_name] =\
                df[fuel] * df[interaction_var]

    return df


def create_modeling_df():
    # Print a status message
    print("Starting to create final modeling dataframe.")

    # Load the original long-format data
    car_df = pd.read_csv("../../data/interim/car_long_format.csv")

    # Switch the methanol and electric fuel type variables to account for the
    # data errors in the mlogit package.
    exchange_dict = {'methanol': 'electric',
                 'electric': 'methanol',
                 'cng': 'cng',
                 'gasoline': 'gasoline'}
    car_df['fuel_type'] = car_df['fuel_type'].map(exchange_dict)

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

    # Create variables for the mixed logit models
    car_df['non_ev'] = (~car_df['electric']).astype(int)
    car_df['non_cng'] = (~car_df['compressed_natural_gas']).astype(int)

    # Create variables for Mixed Logit B from Brownstone and Train (1998)
    neg_variables = ['price_over_log_income',
                     'acceleration_over_10',
                     'pollution',
                     'tens_of_cents_per_mile']

    prefix = 'neg_'
    for col in neg_variables:
        new_col = prefix + col
        car_df[new_col] = -1 * car_df[col]

    # Add the piecewise and interaction variables
    car_df = interaction_model_transform(car_df)

    # Write the new dataset to file
    car_df.to_csv("../../data/processed/model_ready_car_data.csv", index=False)

    # Print a status message
    print("Finished creating final modeling dataframe.")
    return None

if __name__ == '__main__':
    create_modeling_df()
