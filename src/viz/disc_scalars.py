# -*- coding: utf-8 -*-
"""
Functions for plotting simulated vs observed, discrete scalars.
"""
from __future__ import absolute_import

from copy import deepcopy
from numbers import Number

import scipy.stats
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt

from .plot_utils import _label_despine_save_and_show_plot

# Set the plotting style
sbn.set_style('darkgrid')


def _calc_num_simulated_obs_meeting_a_condition(simulated_y, condition):
    """
    Calulates the number of simulated observations where `y == 1` and
    `condition == True`.

    Parameters
    ----------
    simulated_y : 1D ndarray of ints in `{0, 1}`.
        Denotes the simulated outcomes.
    condition : 1D ndarray of booleans or ints in `{0, 1}`.
        Denotes the conditions that need to be met in addition to `y == 1`.

    Returns
    -------
    num : scalar.
        The number observations with `simulated_y == 1 and condition == True`.
    """
    if simulated_y.shape[0] != condition.shape[0]:
        msg = 'simulated_y.shape[0] MUST EQUAL condition.shape[0]'
        raise ValueError(msg)
    return simulated_y.T.dot(condition)


def _get_value_counts_categorical(df, column, alt_filter, ascending=False):
    """
    Count the number of rows in `df` where `alt_filter == True` for each unique
    value in `df.loc[alt_filter, column]`.

    Parameters
    ----------
    df : pandas DataFrame
        The dataframe containing the column that we wish to get item-counts of.
    column : str.
        Should be present in `df.columns`. Should denote the column item-counts
        should be computed on.
    alt_filter : 1D boolean ndarray.
        Denotes the rows of `df` that are to be counted when item-counting.
    ascending : bool, optional.
        Denotes whether the counts are to be returned in ascending order or
        not. Default == False (return the counts from largest to smallest).

    Returns
    -------
    value_counts : pandas Series
        The index will contain the unique values from
        `df.loc[alt_filter, column]`, and the values of the Series will be
        a count of how many times the corresponding index value was in
        `df.loc[alt_filter, column]`.
    """
    # Value count the rows pertaining to the alternative of interest
    value_counts = df.loc[alt_filter, column].value_counts()

    # Sort the value counts in the desired order
    value_counts = value_counts.sort_values(ascending=ascending)
    return value_counts


def _plot_predictive_counts(predictions,
                            color,
                            label,
                            axis,
                            alpha=0.5):
    """
    Plot a histogram of the number of observations having a given range of
    predictions. If we have less than 100 unique predictions, use 1 bar for
    every unique prediction, otherwise, bin the predictions into 100 groups.

    Parameters
    ----------
    predictions : 1D ndarray of ints.
        The predicted number of observations meeting some criterion.
    color : valid color argument to matplotlib Axes.bar
        Should denote the desired color for the bars in the histogram.
    label : str or None.
        Should denote the label for the bars of the histogram.
    axis : matplotlib Axes instance.
        The Axes that the histogram should be plotted on.
    alpha : float in (0.0, 1.0), optional.
        Determines the opacity of the histogram bars. At 0.0, the bars are are
        invisible, and at 1.0, once cannot see through the bars. Higher values
        means lower transparency. Default == 0.5

    Returns
    -------
    None. Draws the desired histogram on `axis`.
    """
    # For fast plotting, use a histogram if np.unique(predictions).size > 100.
    if np.unique(predictions).size > 100:
        # Compute the counts and bin edges when we only want 100 bins
        pred_counts, pred_edges = np.histogram(predictions, bins=100)
        # Compute the bin-widths for plotting
        bin_widths = pred_edges[1:] - pred_edges[:-1]
        # Exclude the right-most bin-edge. It will be implicitly defined when
        # plotting based on the left-edge and the bin-widths.
        pred_edges = pred_edges[:-1]
    else:
        # Calculate the prior-predictive density of the number of
        # decision makers choosing the specified alternative AND
        # meeting the condition on the categorical variable.
        pred_itemfreq = scipy.stats.itemfreq(predictions)
        # Note we subtract 0.5 since we'll make bins of width 1 and
        # (num - 0.5, num + 0.5) captures all values that round to 'num'
        pred_edges = pred_itemfreq[:, 0] - 0.5
        pred_counts = pred_itemfreq[:, 1]
        # Specify the bin-width for plotting. Since each bin corresponds to a
        # single predicted number, the bin-widths will be 1.
        bin_widths = 1

    # Plot the counts directly.
    axis.bar(pred_edges, pred_counts, width=bin_widths, align='edge',
             color=color, alpha=alpha, label=label)
    return None


def plot_discrete_scalars(df,
                          prior_sim_y,
                          post_sim_y,
                          column,
                          alt_filter,
                          orig_choices,
                          top_n=None,
                          min_obs=None,
                          prior_color=None,
                          post_color=None,
                          prior_label='Prior',
                          post_label='Posterior',
                          legend_loc='best',
                          x_label='',
                          filter_name='',
                          title='',
                          show=True,
                          figsize=(5, 3),
                          fontsize=12,
                          output_file='',
                          dpi=500):
    """
    Plots the observed value versus the predictive distribution of the number
    of observations meeting some criteria (`alt_filter`), having `y == 1`, and
    having a particular value of `df.loc[alt_filter, column]`. This function
    allows one to examine `P(column | y == 1 & alt_filter == True)`.

    Parameters
    ----------
    df : pandas DataFrame.
        The dataframe containing the data used to estimate one's model. Should
        have the same number of rows as `prior_sim_y` and `post_sim_y`.
    prior_sim_y : 2D ndarray or None.
        If ndarray, `prior_sim_y` should be the prior predictive simulated
        choices. If None, then no prior predicted distribution will be plotted.
    post_sim_y : 2D ndarray.
        The predictive distribution of outcomes based on one's estimated model.
        There should be one row for every row in `df`. There should be one
        column for each set of simulated outcomes.
    column : str.
        The (categorical or mixed categorical / continuous) column in `df`
        whose distribution is to be examined given that `y == 1`.
    alt_filter : 1D ndarray of booleans.
        Should have the same number of rows as `df`. Will denote the rows
        that should be used when examining the distribution of `column` given
        `y == 1`.
    orig_choices : 1D ndarray of ints in `{0, 1}`.
        Denotes the original outcomes in one's dataset.
    top_n : int or None, optional.
        Only plots the predictive distibutions of the `top_n` most common
        values in `df.loc[alt_filter, column]`. If None, the distributions of
        all the values in `df.loc[alt_filter, column]` are shown.
    min_obs : int or None, optional.
        The minimum number of observations having a given value of
        `df.loc[alt_filter, column]` before the predictive distribution of
        observations having that value will be shown.
    prior_color, post_color : valid 'color' argument for matplotlib, optional.
        The colors that will be used to plot distributions based on
        `prior_sim_y` and 'post_sim_y', respectively. Default is
        `prior_color == sbn.color_palette()[0]` and
        `post_color == sbn.color_palette()[1]`.
    prior_label, post_label : str, optional.
        The name used in the legend for `prior_sim_y` and `post_sim_y`.
        Defaults are 'Prior' and 'Posterior'.
    legend_loc : str or 2-tuple, optional.
        Should be a valid object for the `loc` kwarg in `ax.legend`. This kwarg
        determines the location of the legend on the plot. Default = 'best'.
    x_label : str or None, optional.
        The label for the x-axis. Will be called with `x_label.format(num)`
        where `num` is a unique value from `df.loc[alt_filter, column]`.
        Default is ''.
    filter_name : str, optional.
        If `x_label is None`, `filter_name` will be used in the following
        manner to generate a label for the x-axis. We will use
        `'Number of {} with {} == {}'.format(filter_name, column, num)` where,
        again, `num` is a unique value from `df.loc[alt_filter, column]`.
        Default is ''.
    title : str, optional.
        The title of the plot.
    show : bool, optional.
        Determines whether the figure is shown after plotting is complete.
        Default == True.
    figsize : 2-tuple of ints, optional.
        If a new figure is created for this plot, this kwarg determines the
        width and height of the figure that is created. Default is `(5, 3)`.
    fontsize : int or None, optional.
        The fontsize to be used in the plot. Default is 12.
    output_file : str, or None, optional.
        Denotes the relative or absolute filepath (including the file format)
        that is to be used to save the plot. Will be called with
        `output_file.format(num)` where `num` is a unique value from
        `df.loc[alt_filter, column]`. If None, the plot will not be saved to
        file. Default is None.
    dpi : positive int, optional.
        Denotes the number of 'dots per inch' for the saved figure. Will only
        be used if `output_file is not None`. Default == 500.

    Returns
    -------
    None. Plots the desired predictive densities on a series of matplotlib
    figures. If one wishes to save the figures, use the `output_file` keyword
    argument.
    """
    value_counts = _get_value_counts_categorical(df, column, alt_filter)
    value_counts = value_counts if top_n is None else value_counts.iloc[:top_n]

    if prior_color is None:
        prior_color = sbn.color_palette()[0]
    if post_color is None:
        post_color = sbn.color_palette()[1]
        
    orig_output_file =\
        deepcopy(output_file) if output_file is not None else None

    for num in np.sort(value_counts.index):
        num_condition = (df[column] == num).values
        current_condition = alt_filter & num_condition

        obs_value = (orig_choices & current_condition).sum()

        if min_obs is not None and obs_value < min_obs:
            continue

        # Create a figure and axes for the plot
        fig, ax = plt.subplots(1, figsize=figsize)
        fig_and_ax = [fig, ax]

        # Plot the prior predictions
        if prior_sim_y is not None:
            prior_sim_values =\
                _calc_num_simulated_obs_meeting_a_condition(
                    prior_sim_y, current_condition)

            # Determine the fraction of samples <= observed
            prior_frac_below_obs = (prior_sim_values < obs_value).mean()
            prior_frac_equal_obs = (prior_sim_values == obs_value).mean()

            # Create handle for the plot
            prior_handle_1 = prior_label
            prior_handle_2 = 'P({} samples < actual) = {:.2f}'
            prior_handle_3 = 'P({} samples == actual) = {:.2f}'
            prior_handle_list =\
                [prior_handle_1,
                 prior_handle_2.format(prior_label, prior_frac_below_obs),
                 prior_handle_3.format(prior_label, prior_frac_equal_obs)]
            prior_handle = '\n'.join(prior_handle_list)

            # Plot the prior predicted counts
            _plot_predictive_counts(
                prior_sim_values, prior_color, prior_handle, ax)

        # Get and plot the posterior predictions.
        post_sim_values =\
            _calc_num_simulated_obs_meeting_a_condition(
                post_sim_y, current_condition)

        post_frac_below_obs = (post_sim_values < obs_value).mean()
        post_frac_equal_obs = (post_sim_values == obs_value).mean()

        # Create the handle for the posterior samples
        post_handle_1 = post_label
        post_handle_2 = 'P({} samples < actual) = {:.0%}'
        post_handle_3 = 'P({} samples == actual) = {:.0%}'
        post_handle_list =\
            [post_handle_1,
             post_handle_2.format(post_label, post_frac_below_obs),
             post_handle_3.format(post_label, post_frac_equal_obs)]
        post_handle = '\n'.join(post_handle_list)

        # Plot the posterior predicted counts
        _plot_predictive_counts(post_sim_values, post_color, post_handle, ax)

        # Plot the observed count
        min_y, max_y = ax.get_ylim()

        line_label = ('Observed = {:,.0f}')
        ax.vlines(obs_value, min_y, max_y,
                  linestyle='dashed', label=line_label.format(obs_value))

        # Create the legend
        ax.legend(loc=legend_loc, fontsize=fontsize)

        # Create an x-label.
        if x_label == '':
            value_label =\
                ' == {:.2f}.' if isinstance(num, Number) else ' == {}.'
            value_label = value_label.format(num)
            current_x_label =\
                ('Number of ' + filter_name + ' with ' + column + value_label)
        else:
            current_x_label = deepcopy(x_label)

        # Take care of boilerplate plotting necessities
        title = title.format(num) if title is not None else title
        current_output_file = deepcopy(output_file)
        current_output_file =\
            (current_output_file.format(num) if output_file is not None
             else output_file)
        _label_despine_save_and_show_plot(
            x_label=current_x_label, y_label='Count', fig_and_ax=fig_and_ax,
            fontsize=fontsize, y_rot=0, y_pad=40, title=title,
            output_file=current_output_file, show=show, dpi=dpi)
    return None
