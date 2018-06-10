# -*- coding: utf-8 -*-
"""
This file prepares the forecast dataset where the price of large gasoline
vehicles is increased by 20%. This file is meant to be run from the
'src/features' directory.
"""
from __future__ import absolute_import

from collections import defaultdict
import pandas as pd
import numpy as np

from build_features import interaction_model_transform


def create_forecast_df():
    print("Beginning to create the forecast data.")
    # Load the standard data
    df = pd.read_csv("../../data/processed/model_ready_car_data.csv")

    # Find the large gas cars
    large_gas_car_idx = ((df['body_type'] == 'regcar') &
                         (df['vehicle_size'] == 3) &
                         (df['fuel_type'] == 'gasoline'))
    # Increase the price of large gas cars
    df.loc[large_gas_car_idx, 'price_over_log_income'] *= 1.2
    # Overwrite the negative price column
    df['neg_price_over_log_income'] = -1 * df['price_over_log_income']
    # Overwrite the piecewise specification and interaction columns
    df = interaction_model_transform(df)

    # Store the new dataframe on disk
    df.to_csv("../../data/processed/forecast_car_data.csv", index=False)
    print("Finished creating the forecast data.")
    return None


if __name__ == "__main__":
    create_forecast_df()
