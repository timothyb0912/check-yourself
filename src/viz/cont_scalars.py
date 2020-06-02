# -*- coding: utf-8 -*-
"""
Functions for plotting simulated vs observed, continuous scalars.
"""
from __future__ import absolute_import

import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt

from .plot_utils import _label_despine_save_and_show_plot
from .plot_utils import _plot_single_cdf_on_axis

# Set the plotting style
sbn.set_style('darkgrid')


def plot_continous_scalars(sim_scalars,
                           obs_scalar,
                           kde=True,
                           fig_and_ax=None,
                           figsize=(10, 6),
                           sim_color='#a6bddb',
                           sim_label='Simulated',
                           obs_label='Observed',
                           x_label='Log-Likelihood',
                           y_label='Density',
                           fontsize=12,
                           title=None,
                           output_file=None,
                           dpi=500,
                           show=True):
    """
    For a given continuous scalar, this function plots the distribution of
    predicted scalars versus the observed scalars.

    Parameters
    ----------
    sim_scalars : 1D ndarray of floats.
        The array of simulated scalar values, with 1 value per simulation.
    obs_scalar : int, float, or long.
        The scalar value for one's model with the observed outcomes.
    kde : bool, optional.
        Determines whether a kernel density estimate is plotted. If
        `kde=False`, a cumulative density plot is made.
    fig_and_ax : list of matplotlib figure and axis, or `None`, optional.
        Determines whether a new figure will be created for the plot or whether
        the plot will be drawn on existing axes. If None, a new figure will be
        created. Default is `None`.
    figsize : 2-tuple of positive ints.
        Determines the size of the created figure. Default == (5, 3).
    sim_color : valid 'color' argument for matplotlib, optional.
        The colors that will be used to plot the simulated and observed KDEs,
        respectively. Default is `sim_color == '#a6bddb'`.
    sim_label, obs_label, x_label, y_label, : str, or None, optional.
        Denotes the label to be used for the lines denoting the simulated and
        observed scalars, the x-axis, and the y-axis. Defaults are
        `['Simulated', 'Observed', 'Log-Likelihood', 'Density']`.
    fontsize : int or None, optional.
        The fontsize to be used in the plot. Default is 12.
    title : string or None, optional.
        Denotes the title to be displayed for the plot. Default is None.
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
    None.
    """
    # Create or access the figure and axis on which the plot is to be drawn.
    if fig_and_ax is None:
        fig, axis = plt.subplots(1, figsize=figsize)
        fig_and_ax = [fig, axis]
    else:
        fig, axis = fig_and_ax

    # Plot the distribution of scalars
    if kde:
        # Check to make sure that we have data and no NaNs
        msg = None
        if sim_scalars.size == 0:
            msg = "`sim_scalars` has no rows."
        elif np.isnan(sim_scalars).sum() > 0:
            msg = "`sim_scalars` contains NaNs"
        if msg is not None:
            raise ValueError(msg)
        sbn.kdeplot(sim_scalars, ax=axis, label=sim_label)
    else:
        _plot_single_cdf_on_axis(sim_scalars, axis,
                                 color=sim_color, linestyle='-',
                                 label=sim_label, alpha=1.0)

    # Figure out the axis boundaries
    min_y, max_y = axis.get_ylim()

    # Calculate the percentile corresponding to the observed scalar
    simulated_frac_below_observed =\
        (sim_scalars < obs_scalar).sum() / float(sim_scalars.size)

    # Create the vertical line to show the observed scalar
    line_label = obs_label + '\nP(samples < observed) = {:.0%}'
    axis.vlines(obs_scalar, min_y, max_y, linestyle='dashed',
                label=line_label.format(simulated_frac_below_observed))

    # Create the plot legend
    axis.legend(loc='best', fontsize=fontsize)

    # Take care of boilerplate plotting necessities
    _label_despine_save_and_show_plot(
        x_label=x_label, y_label=y_label, fig_and_ax=fig_and_ax,
        fontsize=fontsize, y_rot=0, y_pad=40, title=title,
        output_file=output_file, show=show, dpi=dpi)
    return None
