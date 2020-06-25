# -*- coding: utf-8 -*-
"""
Functions for plotting marginal model plots: smooths of simulated vs observed
outcomes on the y-axis against a continuous variable on the x-axis.
"""
from __future__ import absolute_import

import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt

from .utils import progress
from .plot_utils import _label_despine_save_and_show_plot
from .smoothers import DiscreteSmoother, ContinuousSmoother, SmoothPlotter

try:
    # in Python 3 range returns an iterator instead of list
    # to maintain backwards compatibility use "old" version of range
    from past.builtins import range
except ImportError:
    pass

# Set the plotting style
sbn.set_style('darkgrid')


def _check_marginal_args(probs, choices, partitions, sim_y):
    """
    Ensures `probs` is a 1D or 2D ndarray, that `choices` is a 1D ndarray, that
    `partitions` is an int, and that `sim_y` is a 2D ndarray.
    """
    if not isinstance(probs, np.ndarray) and probs is not None:
        msg = '`probs` MUST be an ndarray or None.'
        raise ValueError(msg)
    if isinstance(probs, np.ndarray) and probs.ndim not in [1, 2]:
        msg = 'probs` MUST be a 1D or 2D ndarray.'
        raise ValueError(msg)
    if not isinstance(choices, np.ndarray):
        msg = '`choices` MUST be an ndarray.'
        raise ValueError(msg)
    if choices.ndim != 1:
        msg = '`choices` MUST be a 1D ndarray.'
        raise ValueError(msg)
    if not isinstance(partitions, int):
        msg = '`partitions` MUST be an int.'
        raise ValueError(msg)
    if not isinstance(sim_y, np.ndarray) and sim_y is not None:
        msg = '`sim_y` MUST be an ndarray or None.'
        raise ValueError(msg)
    if isinstance(sim_y, np.ndarray) and sim_y.ndim != 2:
        msg = ('`sim_y` MUST be a 2D ndarray')
        raise ValueError(msg)
    return None


def _check_mmplot_ref_vals(probs, ref_vals):
    """
    Checks argument validity for the marginal model plots. Ensures `ref_vals`
    is a 1D ndarray with the same number of rows as `probs`.
    """
    if not isinstance(ref_vals, np.ndarray) or ref_vals.ndim != 1:
        msg = "`ref_vals` MUST be a 1D ndarray."
        raise ValueError(msg)
    elif probs is not None:
        if ref_vals.shape[0] != probs.shape[0]:
            msg = "`ref_vals` MUST have the same number of rows as `probs`."
            raise ValueError(msg)
    return None


def plot_smoothed_marginal(sim_y,
                           choices,
                           ref_vals,
                           probs=None,
                           discrete=True,
                           partitions=10,
                           n_estimators=50,
                           min_samples_leaf=10,
                           random_state=None,
                           y_color='#1f78b4',
                           prob_color='#a6cee3',
                           sim_color='#fb9a99',
                           y_label='Observed',
                           prob_label='Predicted',
                           sim_label='Simulated',
                           y_axis_label='Binned,\nMean\nProbability',
                           x_label='Binned, Mean Reference Values',
                           alpha=None,
                           title=None,
                           fontsize=12,
                           figsize=(5, 3),
                           fig_and_ax=None,
                           legend=True,
                           progress_bar=True,
                           show=True,
                           output_file=None,
                           dpi=500):
    """
    Creates a smoothed marginal model plot based on simulated outcomes,
    observed outcomes, refernce values, and optionally, one's predicted
    probabilities.

    Parameters
    ----------
    sim_y : 2D ndarray.
        An array of simulated choices from one's model. Each element should be
        either a zero or a one. `sim_y` should have the same number of rows as
        `choices` and `ref_vals`.
    choices : 1D ndarray.
        Each element should be either a zero or a one. Elements should denote
        whether the alternative corresponding to the given row was actually
        chosen or not. A 'one' corresponds to an outcome of 'success'.
    ref_vals : 1D ndarray of floats.
        These should be the elements to plot on the x-axis. `ref_vals` should
        represent a continuous variable. Should have the same number of rows as
        `sim_y` and `choices`.
    probs : 1D or 2D ndarray, optional.
        Each element should be in [0, 1]. There should be 1 column for each
        set of predicted probabilities. Plotting a point estimate of the
        predicted probabilities (1D) or a sample of predicted probabilities
        (2D) is useful for visualizing the relationships that one's model
        expects, asymptotically.
    discrete : bool, optional.
        Determines whether discrete smoothing (i.e. binning) will be used or
        whether continuous binning via Extremely Randomized Trees will be used.
        Default is to use discrete binning, so `discrete == True`.
    partitions : positive int, optional.
        Denotes the number of partitions to split one's data into for binning.
        Only used if `discrete is True`. Default == 10.
    n_estimators : positive int, optional.
        Determines the number of trees in the ensemble of Extremely Randomized
        Trees that is used to do continuous smoothing. This parameter controls
        how smooth one's resulting estimate is. The more estimators the
        smoother one's estimated relationship and the lower the variance in
        that estimated relationship. This kwarg is only used if `discrete is
        False`. Default == 50.
    min_samples_leaf : positive int, optional.
        Determines the minimum number of observations allowed in a leaf node in
        any tree in the ensemble. This parameter is conceptually equivalent to
        the bandwidth parameter in a kernel density estimator. This kwarg is
        only used if `discrete is False`. Default == 10.
    random_state : positive int, or None, optional.
        Denotes the random seed to be used when constructing the ensemble of
        Extremely Randomized Trees. This kwarg is only used if `discrete is
        False`. Default is None.
    y_color, prob_color, sim_color : matplotlib color, or `None`, optional.
        Determines the color that is used to plot the observed choices,
        predicted probabilities, and simulated choices versus `ref_vals`.
        Defaults are `'#1f78b4', '#a6cee3', '#fb9a99'`.
    y_label, prob_label, sim_label : str, or None, optional.
        Denotes the label to be used for the lines relating the observed
        choices, predicted probabilities, and simulated choices to the
        `ref_vals`. Default == ['Observed', 'Predicted', 'Simulated'].
    y_axis_label : str, optional.
        The label for the y-axis of the plot. Default is
        'Binned,\nMean\nProbability'.
    x_label : str, optional.
        The label for the x-axis of the plot. Default is 'Binned, Mean
        Reference Values.'
    alpha : positive float in [0.0, 1.0], or `None`, optional.
        Determines the opacity of the simulated and predicted elements drawn on
         the plot. 0.0 == transparent and 1.0 == opaque. Default is `None`.
    title : str, or None, optional.
        Denotes the title to be displayed for the plot. Default is None.
    fontsize : int or None, optional.
        The fontsize to be used in the plot. Default is 12.
    figsize : 2-tuple of positive ints.
        Determines the size of the created figure. Default == (5, 3).
    fig_and_ax : list of matplotlib figure and axis, or `None`, optional.
        Determines whether a new figure will be created for the plot or whether
        the plot will be drawn on existing axes. If None, a new figure will be
        created. Default is `None`.
    legend : bool, optional.
        Determines whether a legend is printed for the plot. Default == True.
    progress_bar : bool, optional.
        Determines whether a progressbar is displayed while making the plot.
        Default == True.
    show : bool, optional.
        Determines whether the figure is shown after plotting is complete.
        Default == True.
    output_file : str, or None, optional.
        Denotes the relative or absolute filepath (including the file format)
        that is to be used to save the plot. If None, the plot will not be
        saved to file. Default is None.
    dpi : positive int, optional.
        Denotes the number of 'dots per inch' for the saved figure. Will only
        be used if `output_file is not None`. Default == 500.

    Returns
    -------
    None.
    """
    # Perform some basic argument checking
    _check_marginal_args(probs, choices, partitions, sim_y)
    _check_mmplot_ref_vals(probs, ref_vals)

    # Sort the arguments, if necesssary
    sort_order = np.argsort(ref_vals)
    ref_vals, sim_y, choices =\
        ref_vals[sort_order], sim_y[sort_order, :], choices[sort_order]

    # Create the figure and axes if need be
    if fig_and_ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
        fig_and_ax = [fig, ax]
    else:
        fig, ax = fig_and_ax

    # Create the progressbar iterator if desired
    if progress_bar:
        sim_iterator =\
            progress(range(sim_y.shape[1]), desc="Plotting Simulations")
    else:
        sim_iterator = range(sim_y.shape[1])

    # Create the desired smoother
    if discrete:
        smoother =\
            DiscreteSmoother(num_obs=ref_vals.shape[0], partitions=partitions)
    else:
        smoother = ContinuousSmoother(n_estimators=n_estimators,
                                      min_samples_leaf=min_samples_leaf,
                                      random_state=random_state)

    # Create the plotter that will plot single smooth curves
    plotter = SmoothPlotter(smoother=smoother, ax=ax)

    # Plot the simulated choices vs reference vals
    for i in sim_iterator:
        current_label = sim_label if i == 0 else None
        plotter.plot(ref_vals,
                     sim_y[:, i],
                     label=current_label,
                     color=sim_color,
                     alpha=alpha)

    # Plot the probabilities versus the ref values.
    if probs is not None:
        # Make probs 2D if necessary
        probs = probs[:, None] if probs.ndim == 1 else probs
        # Sort probs as appropriate
        probs = probs[sort_order, :]
        # Create another progressbar iterator if desired
        if progress_bar:
            prob_iterator = progress(range(probs.shape[1]), desc="Plotting")
        else:
            prob_iterator = range(probs.shape[1])
        # Make the desired plots
        for col in prob_iterator:
            # Get the current line label and probabilities
            current_label = prob_label if col == 0 else None
            current_probs = probs[:, col]
            plotter.plot(ref_vals,
                         current_probs,
                         label=current_label,
                         color=prob_color,
                         alpha=alpha)

    #####
    # Plot observed choices versus ref_vals
    #####
    # Make sure the 'true' relationship is not transparent
    observed_alpha = 1.0
    plotter.plot(ref_vals,
                 choices, label=y_label, color=y_color, alpha=observed_alpha)

    # Make the legend, if desired
    if legend:
        ax.legend(loc='best', fontsize=fontsize)

    # Take care of boilerplate plotting necessities
    _label_despine_save_and_show_plot(
        x_label=x_label, y_label=y_axis_label, fig_and_ax=fig_and_ax,
        fontsize=fontsize, y_rot=0, y_pad=40, title=title,
        output_file=output_file, show=show, dpi=dpi)
    return None
