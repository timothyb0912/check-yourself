# -*- coding: utf-8 -*-
"""
This file converts the wide-format car data from the mlogit package into the
long-format needed for use with the PyLogit package. This file is meant to be
run from the src/data directory.
"""
import pandas as pd
import numpy as np
import pylogit as pl


def convert_wide_to_long():
    # Print a status message
    print("Beginning Conversion Process.")

    # Load the wide format car data
    wide_car = pd.read_csv("../../data/raw/car_wide_format.csv")

    # Create the list of individual specific variables
    ind_variables = wide_car.columns.tolist()[1:4]

    # Specify the variables that vary across individuals and some or all
    # alternatives.

    # The keys are the column names that will be used in the long format df.
    # The values are dictionaries whose key-value pairs are the alternative id
    # and the column name of the corresponding column that encodes that
    # variable for the given alternative. Examples below.
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
    # Determine the columns for: alternative ids, the observation ids, and the choice
    ##########
    # The 'custom_alt_id' is the name of a column to be created in the long-
    # format data. It will identify the alternative associated with each row.
    custom_alt_id = "alt_id"

    # Create a custom id column that ignores the fact that this is a
    # panel/repeated-observations dataset. Note the +1 ensures the id's start
    # at one.
    obs_id_column = "obs_id"
    wide_car[obs_id_column] = np.arange(1, wide_car.shape[0] + 1, dtype=int)

    # Create a variable recording the choice column
    choice_column = "choice"
    # Store the original choice column in a new variable
    wide_car['orig_choices'] = wide_car['choice'].values
    # Alter the original choice column
    choice_str_to_value = {'choice{}'.format(x): x for x in range(1, 7)}
    wide_car[choice_column] = wide_car[choice_column].map(choice_str_to_value)

    # Convert the wide-format data to long format
    long_car =\
        pl.convert_wide_to_long(wide_data=wide_car,
                                ind_vars=ind_variables,
                                alt_specific_vars=alt_varying_variables,
                                availability_vars=availability_variables,
                                obs_id_col=obs_id_column,
                                choice_col=choice_column,
                                new_alt_id_name=custom_alt_id)

    # Save the long-format data to file.
    long_car.to_csv("../../data/interim/car_long_format.csv", index=False)

    # Print a status message.
    print("Finished Conversion Process.")
    return None


if __name__ == '__main__':
    convert_wide_to_long()
