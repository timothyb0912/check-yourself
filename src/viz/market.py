# -*- coding: utf-8 -*-
"""
Functions for plotting simulated vs observed market shares of each alternative.
"""
from __future__ import absolute_import

import seaborn as sbn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import progress
from .plot_utils import _label_despine_save_and_show_plot


def _get_objects_for_market_share_plot(x,
                                       sim_y,
                                       obs_y,
                                       x_label,
                                       y_label,
                                       display_dict=None):
    """
    Creates dataframes needed for the market share plot.

    Parameters
    ----------
    x : 1D ndarray.
        Should contain the values of the discrete random value for each
        alternative for each observation.
    sim_y : 2D ndarray of zeros and ones.
        Denotes the simulated choices based on one's model. `sim_y.shape[0]`
        MUST equal `x.shape[0]`. There should be one column per simulation.
    obs_y : 1D ndarray of zeros and ones.
        The observed choices used to estimate one's model.
    x_label, y_label : str, optional.
        Denotes the x-axis and y-axis labels used on the plot. Defaults are 'X'
        and 'Counts'.
    display_dict : dict or None, optional.
        If passed, will be used to override the default xtick-labels. Each key
        should be a unique value in `x`. Each value should be the label to be
        plotted.

    Returns
    -------
    boxplot_df : pandas DataFrame.
        Will contain an `x_label` and `y_label` column. There will be one row
        per unique value in `x_label` per simulation. The `y_label` column will
        contain the counts of the number of times the associated value in the
        `x_label` column was simulated to be chosen.
    obs_df : pandas DataFrame.
        Will contain the same two columns as boxplot_df. There will be one row
        per unique value in `x_label`. The values in the `y_label` column will
        be the number of observations with the row's corresponding `x_label`
        value.
    """
    # Get the positions and counts of the chosen values of x
    unique_pos = np.unique(x, return_index=True)[1]

    # Determine the unique values in the x-array, in their original order.
    unique_vals = x[np.sort(unique_pos)]

    # Get the counts of the chosen values of x
    _val_names, _val_counts = np.unique(x[obs_y == 1], return_counts=True)
    obs_df = pd.DataFrame({x_label: _val_names, y_label: _val_counts})

    # Initialize an array of the simulated number of observations per value
    num_per_value_per_sim =\
        np.empty((unique_vals.size, sim_y.shape[1]))

    # Create the iterable for populating `num_per_value_per_sim`
    iterator = progress(unique_vals, desc='Unique x-values')

    # Populate the created array
    for pos, val in enumerate(iterator):
        # Determine the rows of x that have values of `val`.
        row_idxs = np.where(x == val)[0]
        # Get the simulated y values for the given value.
        current_sim_y = sim_y[row_idxs, :]
        # Store the total simulated number of observations equaling `val`
        num_per_value_per_sim[pos, :] = current_sim_y.sum(axis=0)

    ####
    # Convert the array of simulated counts per value of X to a dataframe
    ####
    # Create an array with 1 row per unique value of x per simulation
    long_vals = np.repeat(unique_vals, sim_y.shape[1])
    # Convert the counts per unique value per simulation into a 1D array
    long_counts = num_per_value_per_sim.ravel()
    # Create a dataframe of the unique values of x and the simulated counts
    boxplot_df = pd.DataFrame({x_label: long_vals, y_label: long_counts})
    # Convert the unique values to names the user wants to display on the plot
    if display_dict is not None:
        boxplot_df[x_label] = boxplot_df[x_label].map(display_dict)
        obs_df[x_label] = obs_df[x_label].map(display_dict)

    # Also make the x_label values the index for the observed dataframe for
    # later sorting.
    obs_df.index = [str(v) for v in obs_df[x_label].values]
    return boxplot_df, obs_df


def plot_simulated_market_shares(x,
                                 sim_y,
                                 obs_y,
                                 x_label='X',
                                 y_label='Counts',
                                 display_dict=None,
                                 fig_and_ax=None,
                                 figsize=(10, 6),
                                 fontsize=12,
                                 title=None,
                                 box_color='white',
                                 obs_color='#045a8d',
                                 obs_marker='*',
                                 obs_size=12,
                                 obs_label='Observed',
                                 output_file=None,
                                 dpi=500,
                                 show=True):
    """
    Makes a 'market share' boxplot of the simulated distributions of a discrete
    random variable versus the observed values of that variable. In particular,
    plots the observed number of observations that had a given value of a
    discrete variable versus the simulated distributions of how many
    observations had the same value.

    Parameters
    ----------
    x : 1D ndarray.
        Should contain the values of the discrete random value for each
        alternative for each observation.
    sim_y : 2D ndarray of zeros and ones.
        Denotes the simulated choices based on one's model. `sim_y.shape[0]`
        MUST equal `x.shape[0]`. There should be one column per simulation.
    obs_y : 1D ndarray of zeros and ones.
        The observed choices used to estimate one's model.
    x_label, y_label : str, optional.
        Denotes the x-axis and y-axis labels used on the plot. Defaults are 'X'
        and 'Counts'.
    display_dict : dict or None, optional.
        If passed, will be used to override the default xtick-labels. Each key
        should be a unique value in `x`. Each value should be the label to be
        plotted.
    fig_and_ax : list of matplotlib figure and axis, or `None`, optional.
        Determines whether a new figure will be created for the plot or whether
        the plot will be drawn on existing axes. If None, a new figure will be
        created. Default is `None`.
    figsize : 2-tuple of positive ints.
        Determines the size of the created figure. Default == (10, 6).
    fontsize : int or None, optional.
        The fontsize to be used in the plot. Default is 12.
    title : string or None, optional.
        Denotes the title to be displayed for the plot. Default is None.
    box_color, obs_color : valid matplotlib color argument, optional.
        Denotes the color of the boxes on the boxplot and the color used to
        plot the observed distribution of `x`. Default == 'white', '#045a8d'.
    obs_marker : valid matplotlib marker argument, optional.
        Determines the marker used to plot the observed distribution of `x`.
        Default is '*'.
    obs_size : int, optional.
        Determines the size of the marker for the observed distribution
        of `x`. Default is 12.
    obs_label : str, optional.
        Denotes the legend label used for the markers of the observed
        distribution of `x`. Default is 'Observed'.
    output_file : str, or None, optional.
        Denotes the relative or absolute filepath (including the file format)
        that is to be used to save the plot. If None, the plot will not be
        saved to file. Default is None.
    dpi : positive int, optional.
        Denotes the number of 'dots per inch' for the saved figure. Will only
        be used if `output_file is not None`. Default == 500.
    show : bool, optional.
        Determines whether the figure is shown after plotting is complete.
        Default == True.

    Returns
    -------
    None
    """
    # Ensure the display dict has all possible values that are in x.
    if display_dict is not None:
        safe_display = {k: k for k in np.unique(x)}
        safe_display.update(display_dict)
    else:
        safe_display = None

    # Get the data needed for the plot
    boxplot_df, obs_df =\
        _get_objects_for_market_share_plot(
            x, sim_y, obs_y, x_label, y_label, display_dict=safe_display)

    # Create or access the figure and axis on which the plot is to be drawn.
    if fig_and_ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
        fig_and_ax = [fig, ax]
    else:
        fig, ax = fig_and_ax

    # Create the desired boxplot plot
    sbn.boxplot(x=x_label, y=y_label, data=boxplot_df, color=box_color, ax=ax)
    # Reorder the observed values according to the order of the plot
    plot_labels = [v.get_text() for v in ax.get_xticklabels()]
    obs_df = obs_df.loc[plot_labels]
    # Add the observed values on top the boxplot
    sbn.stripplot(x=x_label, y=y_label, data=obs_df, ax=ax, color=obs_color,
                  s=obs_size, marker=obs_marker, label=obs_label)

    # Ensure that the xticklabels are of the correct fontsize
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)

    # Draw the legend, ensuring that we only have one entry.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:1], labels[:1], loc='best', fontsize=fontsize)

    # Take care of boilerplate plotting necessities
    _label_despine_save_and_show_plot(
        x_label=x_label, y_label=y_label, fig_and_ax=fig_and_ax,
        fontsize=fontsize, y_rot=0, y_pad=40, title=title,
        output_file=output_file, show=show, dpi=dpi)
    return None
