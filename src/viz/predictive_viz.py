# -*- coding: utf-8 -*-
"""
This file contains functions for visualizing predictive model checks.
"""
import sys
from numbers import Number
from copy import deepcopy
# Use set_trace to set debugging break points
from pdb import set_trace as bp

import scipy.stats
import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm, tqdm_notebook
# Use statsmodels for kernel density estimation
import statsmodels.api as sm
# used to plot empirical CDFs
import statsmodels.tools as sm_tools
import statsmodels.distributions as sm_dist

# # Use binary_lowess to fit a binary lowess curve
# from binary_lowess import lowess as b_lowess
# from binary_lowess import logit, logistic
#
# # Use pyqt to fit the non-parametric regressions
# import pyqt_fit.nonparam_regression as smooth
# from pyqt_fit import npr_methods
#
# # Use isotonic regression as a replacement for lowess
# from sklearn.isotonic import IsotonicRegression

# Alias the lowess function
lowess = sm.nonparametric.lowess

# Alias the empirical cdf function
try:
    ECDF = sm_tools.tools.ECDF
except:
    ECDF = sm_dist.ECDF

# Alias the itemfreq function
itemfreq = scipy.stats.itemfreq

# Set the plotting style
sbn.set_style('darkgrid')


def is_kernel():
    """
    Determines whether or not one's code is executed inside of an ipython
    notebook environment.
    """
    if any([x in sys.modules for x in ['ipykernel', 'IPython']]):
        return True
    else:
        return False

# Create a progressbar iterable based on wehther we are in ipython or not.
def PROGRESS(*args, **kwargs):
    if is_kernel():
        try:
            return tqdm_notebook(*args, **kwargs)
        except:
            return tqdm(*args, **kwargs)
    else:
        return tqdm(*args, **kwargs)


def _prep_categorical_return(truth, description, verbose):
    """
    Return `truth` and `description` if `verbose is True` else return
    `description` by itself.
    """
    if verbose:
        return truth, description
    else:
        return truth


def is_categorical(vector,
                   solo_threshold=0.1,
                   group_threshold=0.5,
                   group_num=10,
                   verbose=False):
    """
    Determines if a given vector of variables is categorical (or mixed
    categorical and continuous) or not.

    Parameters
    ----------
    vector : 1D ndarray.
        Contains the data to be checked for categorical status.
    solo_threshold : float in (0.0, 1.0), optional.
        If a single unique value in `vector` makes up more than or equal to
        this fraction of the values, then the vector is considered to be
        categorical. Default == 0.1.
    group_threshold : float in (0.0, 1.0), optional.
        If a group of `group_num` unique values in `vector` makes up more than
        this fraction of the values, then the vector is considered to be
        categorical. Default == 0.5.
    group_num : int, optional.
        Denotes the size of the group that is used when judging if the vector
        is categorical or not. Default == 10.
    verbose : bool, optional.
        Determines whether the function will return a description of the 'type'
        of categorical variable this vector is deemed to be. Default is False.

    Returns
    -------
    bool, or (bool, str) if `verbose == True`.

    Examples
    --------
    >>> import numpy as np
    >>> is_categorical(np.arange(30))
    False

    >>> is_categorical(np.arange(30), group_num=15, group_threshold=0.5)
    True

    15 values make up 50% of the data so the second example evaluates to True

    >>> x = np.tile(np.array([1, 2, 3]), 10)
    >>> x
    array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2,
           3, 1, 2, 3, 1, 2, 3])

    >>> is_categorical(x, group_num=2, group_threshold=0.75)
    True

    Even though 2 values make up only 2/3 of the data, a single value (e.g. 1)
    makes up 10% of the data, thus reaching the `solo_threshold`

    >>> is_categorical(x, solo_threshold=0.15,
    >>>                group_num=2, group_threshold=0.75)
    False

    Since the `solo_threshold` is now 15%, `x` is no longer deemed categorical.
    """
    # Figure out how many observations are in `vector`
    num_observations = float(vector.shape[0])
    # Get the count of each unique value in `vector`
    item_frequencies = itemfreq(vector)
    # Sort the item frequencies by the second column
    item_frequencies =\
        item_frequencies[np.argsort(item_frequencies[:, 1])[::-1], :]
    # Get the percentage of `vector` made up by each unique value
    individual_percents = item_frequencies[:, 1] / num_observations
    # Get the cumulative density function of `vector`.
    cumulative_percents = np.cumsum(item_frequencies[:, 1]) / num_observations
    # Check for 'categorical' nature of `vector`
    if item_frequencies.shape[0] <= group_num:
        truth = True
        description = 'categorical'
    elif cumulative_percents[group_num] >= group_threshold:
        truth = True
        description = 'group'
    elif (individual_percents > solo_threshold).any():
        truth = True
        description = 'solo'
    else:
        truth = False
        description = None

    return _prep_categorical_return(truth, description, verbose)


def _simulate_wide_binary_choices(predictions, rseed=None):
    """
    Take vectorized random draws over many bernoulli random variables with
    different probabilities of success. This function is faster than using a
    for-loop and repeated calls to `np.random.choice`.
    """
    # Initialize the simulated choices
    choice_vec = np.zeros(predictions.shape, dtype=int)

    # Set the random seed if desired
    if rseed is not None:
        np.random.seed(rseed)

    # Generate uniform random variates
    uniform_draws =\
        np.random.uniform(size=predictions.shape)

    # Determine which predictions led to 'successful' observations
    choice_vec[np.where(uniform_draws <= predictions)] = 1
    return choice_vec


def simulate_choice_vector(predicted_probs,
                           observation_ids,
                           wide_binary=False,
                           rseed=None):
    """
    Simulates choice outcomes based on the predicted probabilities of each
    alternative for each observation.

    Parameters
    ----------
    predicted_probs : 2D ndarray of floats in (0.0, 1.0).
        Each row should correspond to a particular alternative for a particular
        observation. Each column should correspond to a sampled parameter
        vector. Finally, each element should denote the probability of that
        alternative being chosen by that decision maker, given their explanatory
        variables and the sampled model parameters.
    observation_ids : 1D ndarray of ints.
        Each element should represent an obervation id. Should have
        `observation_ids.shape[0] == predicted_probs.shape[0]`.
    wide_binary : bool, optional.
        Denotes whether `predicted_probs` are for a wide-format dataset of
        binary choices or not.
    rseed : positive int or None, optional.
        The random seed used to simulate the choices. Use when one wants to
        reproduce particular simulations. Default is None.

    Returns
    -------
    simulated_y : 2D ndarray of zeros and ones.
        Each row should correspond to a particular alternative for a particular
        observation. Each column should correspond to a sampled parameter
        vector. Finally, each element will be a one if that row's alternative
        was chosen by that row's decision-maker for that columns simulated
        parameter vector. Otherwise, the element will be a zero. When
        `wide_binary == True`, each element in `simulated_y` will indicate
        whether that row's observation had `y == 1` for that simulation or not.
    """
    # Make predicted_probs 2D
    if predicted_probs.ndim == 1:
        predicted_probs = predicted_probs[:, None]
    elif predicted_probs.ndim > 2:
        msg = 'predicted_probs should have 1 or 2 dimensions.'
        raise ValueError(msg)

    # Make the wide-format binary simulations if necessary
    if wide_binary:
        return _simulate_wide_binary_choices(predicted_probs, rseed=rseed)

    # Determine the unique values in observation_ids
    unique_idx = np.sort(np.unique(observation_ids, return_index=True)[1])
    unique_obs = observation_ids[unique_idx]

    # Determine the rows belonging to each observation
    rows_per_obs = {k: np.where(observation_ids == k)[0] for k in unique_obs}

    # Initialize an array of simulated choices
    choice_vec = np.zeros(predicted_probs.shape, dtype=int)

    # Create an index for the columns
    col_idx = np.arange(predicted_probs.shape[1])

    # Set the seed if desired
    if isinstance(rseed, int):
        np.random.seed(rseed)

    # Populate the array
    for obs_id in PROGRESS(unique_obs.tolist(), desc='Simulating Choices'):
        # Get the rows belonging to this observation
        obs_rows = rows_per_obs[obs_id]

        # Get the current probabilities
        current_long_probs = predicted_probs[obs_rows, :]

        # Get the 'cdf' of each alternative
        current_cdf = np.cumsum(current_long_probs, axis=0)

        # Draw random uniform values for each probability vector
        uniform_draws = np.random.uniform(size=predicted_probs.shape[1])

        # Determine which alternative's 'bucket' the random value
        # might have fallen into.
        possible_alts =\
            (np.arange(1, obs_rows.size + 1)[:, None] *
             (current_cdf >= uniform_draws[None, :]))
        # Give a 'big' value to alternatives that are not chosen
        possible_alts[np.where(possible_alts == 0)] = obs_rows.size + 10
        # Figure out the exact rows/alternatives that were chosen
        chosen_pos = np.argmin(possible_alts, axis=0)

        # Store the simulated choice
        choice_vec[obs_rows[chosen_pos], col_idx] = 1

    return choice_vec


def compute_prior_predictive_log_likelihoods(simulated_y,
                                             orig_df,
                                             choice_col,
                                             model_obj):
    """
    Compute the log-likelihood of probability predictions of the point-estimated
    model and original training data, given the simulated outcomes.

    Parameters
    ----------
    simulated_y : 2D ndarray.
        The simulated outcomes. All elements should be zeros or ones. There
        should be one column for every set of simulated outcomes. There should
        be one row for every row of one's dataset.
    orig_df : pandas DataFrame.
        The dataframe containing the data used to estimate one's model. Should
        have the same number of rows as `simulated_y`.
    choice_col : str.
        Should be the column in `orig_df` that contains the original outcomes.
    model_obj : object with a `long_fitted_probs` attribute.
        `model_obj.long_fitted_probs` should be a 1D ndarray point estimate of
        the probability of that each row's outcome is a `1`.

    Returns
    -------
    log_likes : 1D ndarray of floats.
        Will contain the log-likelihoods generated by
        `model_obj.long_fitted_probs` and outcomes in `simulated_y`. There will
        be one element for each column in `simulated_y`.
    """
    # Get the long-fitted probabilities
    long_probs = model_obj.long_fitted_probs
    log_long_probs = np.log(long_probs)

    # Populate the log-likelihood values
    log_likes = simulated_y.T.dot(log_long_probs).ravel()

    return log_likes


def plot_simulated_kde_traces(sim_y,
                              orig_df,
                              filter_idx,
                              col_to_plot,
                              choice_col,
                              sim_color='#a6bddb',
                              orig_color='#045a8d',
                              choice_condition=1,
                              fig_and_ax=None,
                              label='Simulated',
                              title=None,
                              bar_alpha=0.5,
                              bar_color='#fee391',
                              thin_pct=None,
                              n_traces=100,
                              rseed=None,
                              smooth=False,
                              show=True,
                              figsize=(5, 3),
                              fontsize=12,
                              xlim=None,
                              ylim=None,
                              output_file=None,
                              dpi=500,
                              **kwargs):
    """
    Plots an observed kernel density estimate (KDE) versus the simulated
    versions of that same KDE.

    Parameters
    ----------
    sim_y : 2D ndarray.
        The simulated outcomes. All elements should be zeros or ones. There
        should be one column for every set of simulated outcomes. There should
        be one row for every row of one's dataset.
    orig_df : pandas DataFrame.
        The dataframe containing the data used to estimate one's model. Should
        have the same number of rows as `sim_y`.
    filter_idx : 1D ndarray of booleans.
        Should have the same number of rows as `orig_df`. Will denote the rows
        that should be used to compute the KDE if their outcome is
        `choice_condition`.
    col_to_plot : str.
        A column in `orig_df` whose data will be used to compute the KDEs.
    choice_col : str.
        The column in `orig_df` that contains the data on the original outcomes.
    sim_color, orig_color : valid 'color' argument for matplotlib, optional.
        The colors that will be used to plot the simulated and observed KDEs,
        respectively. Default is `sim_color == '#a6bddb'` and
        `orig_color == '#045a8d'`.
    choice_condition : `{0, 1}`, optional.
        Denotes the outcome class that we wish to plot the KDEs for. If
        `choice_condition == 1`, then we will plot the KDEs for those where
        `sim_y == 1` and `filter_idx == True`. If `choice_condition == 0`, we
        will plot the KDEs for those rows where `sim_y == 0` and
        `filter_idx == True`. Default == 1.
    fig_and_ax : list of matplotlib figure and axis, or `None`, optional.
        Determines whether a new figure will be created for the plot or whether
        the plot will be drawn on the passed Axes object. If None, a new figure
        will be created. Default is `None`.
    label : str or None, optional.
        The label for the simulated KDEs. If None, no label will be displayed.
        Default = 'Simulated'.
    title : str or None, optional.
        The plot title. If None, no title will be displayed. Default is None.
    bar_alpha : float in (0.0, 1.0), optional.
        Denotes the opacity of the bar used to denote the proportion of
        simulations where no observations had `sim_y == choice_condition`.
        Higher values lower the bar's transparency. `0` leads to an invisible
        bar. Default == 0.5.
    bar_color : valid 'color' argument for matplotlib, optional.
        The color that will be used to plot the bar that shows the proportion
        of simulations where no observations had `sim_y == choice_condition`.
        Default is '#fee391'.
    thin_pct : float in (0.0, 1.0) or None, optional.
        Determines the percentage of the data (rows) to be used for plotting.
        If None, the full dataset will be used. Default is None.
    n_traces : int or None, optional.
        Should be less than `sim_y.shape[1]`. Denotes the number of simulated
        choices to randomly select for plotting. If None, all columns of `sim_y`
        will be used for plotting. Default == 100.
    rseed : int or None, optional.
        Denotes the random seed to be used when selecting the `n_traces` columns
        for plotting. This is useful for reproducing an exact plot when using
        `n_traces`. If None, no random seed will be set. Default is None.
    smooth : bool, optional.
        Determines whether we will plot lowess smooths (`smooth == True`) or
        kernel density estimates (`smooth == False`). Default is False.
    show : bool, optional.
        Determines whether `fig.show()` will be called after the plots have been
        drawn. Default is True.
    figsize : 2-tuple of ints, optional.
        If a new figure is created for this plot, this kwarg determines the
        width and height of the figure that is created. Default is `(5, 3)`.
    fontsize : int or None, optional.
        The fontsize to be used in the plot. Default is 12.
    xlim, ylim : 2-tuple of ints or None, optional.
        Denotes the extent that will be set on the x-axis and y-axis,
        respectively, of the matplotlib Axes instance that is drawn on. If None,
        then the extent will not be manually altered. Default is None.
    output_file : str, or None, optional.
        Denotes the relative or absolute filepath (including the file format)
        that is to be used to save the plot. If None, the plot will not be
        saved to file. Default is None.
    dpi : positive int, optional.
        Denotes the number of 'dots per inch' for the saved figure. Will only
        be used if `output_file is not None`. Default == 500.

    kwargs : passed to `ax.plot` call in matplotlib.

    Returns
    -------
    None.
    """
    if rseed is not None:
        np.random.seed(rseed)

    if n_traces is not None:
        selected_cols =\
            np.random.choice(sim_y.shape[1], size=n_traces, replace=False)
        sim_y = sim_y[:, selected_cols]

    if thin_pct is not None:
        # Determine the number of rows to select
        num_selected_rows = int(thin_pct * sim_y.shape[0])
        # Randomly choose rows to retain.
        selected_rows =\
            np.random.choice(sim_y.shape[0],
                             size=num_selected_rows,
                             replace=False)
        # Filter the simulated-y, df, and filtering values
        sim_y = sim_y[selected_rows, :]
        orig_df = orig_df.iloc[selected_rows, :]
        filter_idx = filter_idx[selected_rows]

    sample_iterator = PROGRESS(xrange(sim_y.shape[1]), desc='Calculating KDEs')

    # Create a function to evaluate the choice condition:
    def choice_evaluator(choice_array, choice_condition):
        if choice_condition in [0.0, 1.0]:
            return choice_array == choice_condition
        else:
            return choice_array

    # Get the original values
    orig_choices = orig_df[choice_col].values

    orig_plotting_idx =\
        filter_idx & choice_evaluator(orig_choices, choice_condition)
    orig_plotting_rows = np.where(orig_plotting_idx)
    orig_plotting_vals = orig_df.loc[orig_plotting_idx, col_to_plot].values

    if fig_and_ax is None:
        fig, axis = plt.subplots(1, figsize=figsize)
        fig_and_ax = [fig, axis]
    else:
        fig, axis = fig_and_ax

    # This will track how many simulated datasets had
    # no outcome meeting the choice condition and the
    # filter condition.
    num_null_choices = 0

    # Calculate an interpolation threshold
    delta = None

    # store the minimum and maximum x-values
    min_x = orig_plotting_vals.min()
    max_x = orig_plotting_vals.max()

    for i in sample_iterator:
        current_choices = sim_y[:, i]

        current_num_pos =\
            current_choices[np.where(filter_idx)].sum()

        if current_num_pos == 0:
            num_null_choices += 1
            continue

        current_choice_validity =\
            choice_evaluator(current_choices,
                             choice_condition)

        # Determine the final rows to use for plotting
        plotting_idx =\
            filter_idx & current_choice_validity
        plotting_rows = np.where(plotting_idx)

        # Get the values for plotting
        current_plotting_vals = orig_df.loc[plotting_idx, col_to_plot].values
        assert current_plotting_vals.size > 0

        # Update the plot extents
        min_x = min(current_plotting_vals.min(), min_x)
        max_x = max(current_plotting_vals.max(), max_x)

        if smooth:
            plot_lowess_smooth(current_plotting_vals,
                               current_choices[plotting_rows],
                               show=False,
                               plot_mean=False,
                               delta=0.01,
                               fig_and_ax=fig_and_ax,
                               color=sim_color)
        else:
            sbn.kdeplot(current_plotting_vals,
                        ax=axis,
                        color=sim_color,
                        alpha=0.5,
                        **kwargs)

    # Plot the originally observed relationship
    if smooth:
        plot_lowess_smooth(orig_plotting_vals,
                           orig_choices[orig_plotting_rows],
                           show=False,
                           plot_mean=False,
                           delta=delta,
                           fig_and_ax=fig_and_ax,
                           label='Observed',
                           color=orig_color)
    else:
        sbn.kdeplot(orig_plotting_vals,
                    ax=axis,
                    color=orig_color,
                    label='Observed',
                    **kwargs)

    if num_null_choices > 0:
        num_null_pct = num_null_choices / float(sim_y.shape[1])
        null_pct_density_equivalent = axis.get_ylim()[1] * num_null_pct
        axis.bar([0],
                 [null_pct_density_equivalent],
                 width=0.1 * np.ptp(orig_plotting_vals),
                 align='edge',
                 alpha=bar_alpha,
                 color=bar_color,
                 label="'No Obs' Simulations: {:.2%}".format(num_null_pct))

    if label is not None:
        _patch = mpatches.Patch(color=sim_color, label=label)
        current_handles, current_labels = axis.get_legend_handles_labels()
        current_handles.append(_patch)
        current_labels.append(label)

        axis.legend(current_handles,
                    current_labels,
                    # loc=(1, 0.75),
                    loc='best',
                    fontsize=fontsize)

    # set the plot extents
    if xlim is None:
        axis.set_xlim((min_x, max_x))
    else:
        axis.set_xlim(xlim)

    if ylim is not None:
        axis.set_ylim(ylim)

    # Despine the plot
    sbn.despine()
    # Make plot labels
    axis.set_xlabel(col_to_plot, fontsize=fontsize)
    axis.set_ylabel('Density', fontsize=fontsize, rotation=0, labelpad=40)
    # Create the title
    if title is not None and title != '':
        if not isinstance(title, basestring):
            msg = "`title` MUST be a string."
            raise TypeError(msg)
        axis.set_title(title, fontsize=fontsize)

    # Save the plot if desired
    if output_file is not None:
        fig.tight_layout()
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')

    if show:
        fig.show()

    return None


# Create a function to calculate the number of cyclists with 2 kids
def calc_num_simulated_obs_meeting_a_condition(simulated_y, condition):
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
        The number observations with `simulated_y == 1` and `condition == True`.
    """
    if simulated_y.shape[0] != condition.shape[0]:
        msg = 'simulated_y.shape[0] MUST EQUAL condition.shape[0]'
        raise ValueError(msg)
    return simulated_y.T.dot(condition)


def plot_lowess_smooth(explanatory_array,
                       outcome_array,
                       x_name='',
                       y_name='',
                       label=None,
                       title=None,
                       scatter=False,
                       show=True,
                       output_file=None,
                       dpi=500,
                       fontsize=11,
                       color=None,
                       alpha=None,
                       fig_and_ax=None,
                       delta=None,
                       frac=2.0/3.0,
                       plot_mean=False,
                       legend_title=None,
                       binary=True,
                       isotonic=False,
                       adjust=False,
                       pyqt=False):
    # Essenstially skipping this section due to laziness
    robust_array, robust_outcome = explanatory_array, outcome_array

    # Create the figure and axis on which the plot will be drawn
    if fig_and_ax is None:
        fig, ax = plt.subplots(1)
    else:
        fig, ax = fig_and_ax

    # Calculate the default threshold for linear interpolation between points.
    if delta is None:
        delta = 0.01 * np.ptp(robust_array)
    # Compute the lowess values
    # if isotonic:
    #     regressor = IsotonicRegression(y_min=0.0, y_max=1.0)
    #     y_lowess = regressor.fit_transform(robust_array, robust_outcome)
    #     x_lowess = robust_array
    # elif pyqt:
    #     # sort the data
    #     sort_idx = np.argsort(robust_array)
    #
    #     if binary:
    #         y_for_fit = logit(robust_outcome)[sort_idx]
    #     else:
    #         y_for_fit = robust_outcome[sort_idx]
    #
    #     x_lowess = robust_array[sort_idx]
    #
    #     regressor =\
    #         smooth.NonParamRegression(x_lowess,
    #                                   y_for_fit,
    #                                   method=npr_methods.SpatialAverage())
    #     regressor.fit()
    #
    #     if binary:
    #         y_lowess = logistic(regressor(robust_array))
    #     else:
    #         y_lowess = regressor(robust_array)

    else:
        x_lowess, y_lowess =\
            lowess(robust_outcome, robust_array,
                   delta=delta, frac=frac).T

    if adjust and binary:
        orig_y_mean = robust_outcome.mean()
        lowess_mean = y_lowess.mean()
        multiplier = orig_y_mean / lowess_mean
        y_lowess *= multiplier

    assert (y_lowess >= 0).all()
    assert (y_lowess <= 1.0).all()

    # Get the current plot color
    if color is None:
        current_color = sbn.color_palette()[0]
    else:
        current_color = color

    # Manually create the lowess plot
    ax.plot(x_lowess, y_lowess, label=label, color=current_color, alpha=alpha)

    if scatter:
        # Add jitter to the y-values
        jittered_y = (robust_outcome +
                      np.random.uniform(low=-0.03,
                                        high=0.03,
                                        size=y_lowess.size))
        ax.scatter(robust_array, jittered_y,
                   c=np.array(current_color)[None, :])

    if plot_mean:
        x_low, x_high = ax.get_xlim()
        ax.hlines(robust_outcome.mean(),
                  x_low, x_high,
                  linestyle='dashed',
                  label='E({})'.format(y_name))

    # Set the x- and y-labels
    ax.set_xlabel(x_name, fontsize=fontsize)

    y_rot = 0 + 90 * (len(y_name) >= 15) * ("\n" not in y_name)
    y_pad = 40 * (y_rot == 0) + 20 * (y_rot != 0)
    ax.set_ylabel(y_name,
                  fontsize=fontsize,
                  rotation=y_rot,
                  labelpad=y_pad)

    # Set the title if desired
    if title is not None:
        if not isinstance(title, basestring):
            msg = "`title` MUST be a string."
            raise TypeError(msg)
        ax.set_title(title, fontsize=fontsize)

    # Plot the legend if desired
    if ((label is not None) and (label != '')) or plot_mean:
        if plot_mean and (robust_outcome.mean() > y_lowess.max()):
            loc = 'center right'
        else:
            loc = 'best'
        ax.legend(loc=loc, title=legend_title, frameon=True, framealpha=1.0)

    # Save the plot if desired
    if isinstance(output_file, basestring) and output_file != '':
        # Make the plot have a tight_layout
        fig.tight_layout()
        # Save the plot
        fig.savefig(output_file,
                    dpi=dpi,
                    bbox_inches='tight')

    # Show the plot if desired
    if show:
        fig.show()
    return None


def get_value_counts_categorical(df, column, alt_filter, ascending=False):
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
        Denotes whether the counts are to be returned in ascending order or not.
        Default == False (return the counts from largest to smallest).

    Returns
    -------
    value_counts : pandas Series
        The index will contain the unique values from
        `df.loc[alt_filter, column]`, and the values of the Series will be count
        of how many times the corresponding index value was in
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
                            alpha=0.5,
                            debug=False):
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
    debug : bool, optional.
        Determines wehether or not to execute this function with python debugger
        break-points for function introspection.

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

    # Set a trace for debugging purposes
    if debug:
        bp()

    # Plot the counts directly.
    axis.bar(pred_edges,
             pred_counts,
             width=bin_widths,
             align='edge',
             color=color,
             alpha=alpha,
             label=label)

    return None


def plot_categorical_predictive_densities(df,
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
                                          dpi=500,
                                          debug=False):
    """
    Plots the observed value versus the predictive distribution of the number of
    observations meeting some criteria (`alt_filter`), having `y == 1`, and
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
        The (categorical or mixed categorical / continuous) column in `df` whose
        distribution is to be examined given that a row's predicted `y == 1`.
    alt_filter : 1D ndarray of booleans.
        Should have the same number of rows as `df`. Will denote the rows
        that should be used when examining the distribution of `column` given
        `y == 1`.
    orig_choices : 1D ndarray of ints in `{0, 1}`.
        Denotes the original outcomes in one's dataset.
    top_n : int or None, optional.
        Only plots the predictive distibutions of the `top_n` most common values
        in `df.loc[alt_filter, column]`. If None, the predictive distribution of
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
        where `num` is a unique value from `df.loc[alt_filter, column]`. Default
        is ''.
    filter_name : str, optional.
        If `x_label is None`, `filter_name` will be used in the following manner
        to generate a label for the x-axis. We will use
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
    debug : bool, optional.
        Determines wehether or not to execute this function with python debugger
        break-points for function introspection.

    Returns
    -------
    None. Plots the desired predictive densities on a series of matplotlib
    figures. If one wishes to save the figures, use the `output_file` keyword
    argument.
    """
    value_counts = get_value_counts_categorical(df, column, alt_filter)
    value_counts = value_counts if top_n is None else value_counts.iloc[:top_n]

    if prior_color is None:
        prior_color = sbn.color_palette()[0]
    if post_color is None:
        post_color = sbn.color_palette()[1]

    for num in np.sort(value_counts.index):
        num_condition = df[column] == num
        current_condition = alt_filter & num_condition

        obs_value = (orig_choices & current_condition).sum()

        if min_obs is not None and obs_value < min_obs:
            continue

        # Create a figure and axes for the plot
        fig, ax = plt.subplots(1, figsize=figsize)

        # Plot the prior predictions
        if prior_sim_y is not None:
            prior_sim_values =\
                calc_num_simulated_obs_meeting_a_condition(
                    prior_sim_y, current_condition)

            # Determine the fraction of samples <= observed
            frac_prior_samples_below_obs =\
                (prior_sim_values < obs_value).mean()

            frac_prior_samples_equal_obs =\
                (prior_sim_values == obs_value).mean()

            # Create handle for the plot
            prior_handle_1 = prior_label
            prior_handle_2 = 'P({} samples < actual) = {:.2f}'
            prior_handle_3 = 'P({} samples == actual) = {:.2f}'
            prior_handle_list =\
                [prior_handle_1,
                 prior_handle_2.format(prior_label,
                                       frac_prior_samples_below_obs),
                 prior_handle_3.format(prior_label,
                                       frac_prior_samples_equal_obs)]
            prior_handle = prior_handle_list.join('\n')

            # Plot the prior predicted counts
            _plot_predictive_counts(prior_sim_values,
                                    prior_color,
                                    prior_handle,
                                    ax,
                                    debug=debug)

        # Get and plot the posterior predictions.
        post_sim_values =\
            calc_num_simulated_obs_meeting_a_condition(
                post_sim_y, current_condition)

        frac_post_samples_below_obs =\
            (post_sim_values < obs_value).mean()

        frac_post_samples_equal_obs =\
            (post_sim_values == obs_value).mean()

        # Create the handle for the posterior samples
        post_handle_1 = post_label
        post_handle_2 = 'P({} samples < actual) = {:.0%}'
        post_handle_3 = 'P({} samples == actual) = {:.0%}'
        post_handle_list =\
            [post_handle_1,
             post_handle_2.format(post_label,
                                  frac_post_samples_below_obs),
             post_handle_3.format(post_label,
                                  frac_post_samples_equal_obs)]
        post_handle = '\n'.join(post_handle_list)

        if debug:
            bp()

        # Plot the posterior predicted counts
        _plot_predictive_counts(post_sim_values,
                                post_color,
                                post_handle,
                                ax,
                                debug=debug)

        # Plot the observed count
        min_y, max_y = ax.get_ylim()

        line_label = ('Observed = {:,.0f}')
        ax.vlines(obs_value,
                  min_y,
                  max_y,
                  linestyle='dashed',
                  label=line_label.format(obs_value))

        # Create the legend
        ax.legend(loc=legend_loc, fontsize=fontsize)
        # ax.legend(loc='best', fontsize=fontsize)

        # Label the axes
        if x_label == '':
            if isinstance(num, Number):
                value_label = ' == {:.2f}.'.format(num)
            else:
                value_label = ' == {}.'.format(num)
            current_x_label =\
                ('Number of ' + filter_name + ' with ' + column + value_label)
        else:
            current_x_label = deepcopy(x_label)

        ax.set_xlabel(current_x_label.format(num), fontsize=fontsize)
        ax.set_ylabel('Count',
                      fontsize=fontsize,
                      rotation=0,
                      labelpad=40)

        # Be sure to remove the right and upper spines of the plot.
        sbn.despine()

        # Create the title
        if title is not None and title != '':
            if not isinstance(title, basestring):
                msg = "`title` MUST be a string."
                raise TypeError(msg)
            ax.set_title(title.format(num))

        # Save the plot if desired
        if isinstance(output_file, basestring) and output_file != '':
            # Make the plot have a tight_layout
            fig.tight_layout()
            # Save the plot
            fig.savefig(output_file.format(num), dpi=dpi, bbox_inches='tight')

        # Show the plot if desired
        if show:
            fig.show()

    return None


def _plot_single_cdf_on_axis(x_vals,
                             axis,
                             color='#a6bddb',
                             linestyle='-',
                             label=None,
                             alpha=0.1):
    """
    Plots a CDF of `x_vals` on `axis` with the desired color, linestyle, label,
    and transparency (alpha) level.
    """
    # Create a function that will take in an array of values and
    # return an array of the same length which contains the CDF
    # value at each corresponding value in the passed array.
    cdf_func = ECDF(x_vals)
    # Create a sorted list of all of the unique values that were
    # sampled for this variable
    sorted_samples = np.sort(np.unique(x_vals))
    # Get the CDF values for each of the sorted values
    cdf_values = cdf_func(sorted_samples)
    # Plot the sorted, unique values versus their CDF values
    axis.plot(sorted_samples,
              cdf_values,
              c=color,
              ls=linestyle,
              alpha=alpha,
              label=label,
              drawstyle='steps-post')
    return None


def plot_simulated_cdf_traces(sim_y,
                              orig_df,
                              filter_idx,
                              col_to_plot,
                              choice_col,
                              sim_color='#a6bddb',
                              orig_color='#045a8d',
                              choice_condition=1,
                              thin_pct=None,
                              fig_and_ax=None,
                              label='Simulated',
                              title=None,
                              bar_alpha=0.5,
                              bar_color='#fee391',
                              n_traces=None,
                              rseed=None,
                              show=True,
                              figsize=(5, 3),
                              fontsize=12,
                              xlim=None,
                              ylim=None,
                              output_file=None,
                              dpi=500,
                              **kwargs):
    """
    Plots an observed cumulative density function (CDF) versus the simulated
    versions of that same CDF.

    Parameters
    ----------
    sim_y : 2D ndarray.
        The simulated outcomes. All elements should be zeros or ones. There
        should be one column for every set of simulated outcomes. There should
        be one row for every row of one's dataset.
    orig_df : pandas DataFrame.
        The dataframe containing the data used to estimate one's model. Should
        have the same number of rows as `sim_y`.
    filter_idx : 1D ndarray of booleans.
        Should have the same number of rows as `orig_df`. Will denote the rows
        that should be used to compute the CDF if their outcome is
        `choice_condition`.
    col_to_plot : str or 1D ndarray.
        If str, `col_to_plot` should be a column in `orig_df` whose data will
        be used to compute the CDFs. If ndarray, `col_to_plot` should contain
        the values that will be used to plot the cdf, and `col_to_plot.size`
        should equal `orig_df.shape[0]`.
    choice_col : str.
        The column in `orig_df` containing the data on the original outcomes.
    sim_color, orig_color : valid 'color' argument for matplotlib, optional.
        The colors that will be used to plot the simulated and observed CDFs,
        respectively. Default is `sim_color == '#a6bddb'` and
        `orig_color == '#045a8d'`.
    choice_condition : `{0, 1}`, optional.
        Denotes the outcome class that we wish to plot the CDFs for. If
        `choice_condition == 1`, then we will plot the CDFs for those where
        `sim_y == 1` and `filter_idx == True`. If `choice_condition == 0`, we
        will plot the CDFs for those rows where `sim_y == 0` and
        `filter_idx == True`. Default == 1.
    fig_and_ax : list of matplotlib figure and axis, or `None`, optional.
        Determines whether a new figure will be created for the plot or whether
        the plot will be drawn on the passed Axes object. If None, a new figure
        will be created. Default is `None`.
    label : str or None, optional.
        The label for the simulated CDFs. If None, no label will be displayed.
        Default = 'Simulated'.
    title : str or None, optional.
        The plot title. If None, no title will be displayed. Default is None.
    bar_alpha : float in (0.0, 1.0), optional.
        Denotes the opacity of the bar used to denote the proportion of
        simulations where no observations had `sim_y == choice_condition`.
        Higher values lower the bar's transparency. `0` leads to an invisible
        bar. Default == 0.5.
    bar_color : valid 'color' argument for matplotlib, optional.
        The color that will be used to plot the bar that shows the proportion
        of simulations where no observations had `sim_y == choice_condition`.
        Default is '#fee391'.
    thin_pct : float in (0.0, 1.0) or None, optional.
        Determines the percentage of the data (rows) to be used for plotting.
        If None, the full dataset will be used. Default is None.
    n_traces : int or None, optional.
        Should be less than `sim_y.shape[1]`. Denotes the number of simulated
        choices to randomly select for plotting. If None, all columns of
        `sim_y` will be used for plotting. Default is None.
    rseed : int or None, optional.
        Denotes the random seed to be used when selecting `n_traces` columns
        for plotting. This is useful for reproducing an exact plot when using
        `n_traces`. If None, no random seed will be set. Default is None.
    show : bool, optional.
        Determines whether `fig.show()` will be called after the plots have
        been drawn. Default is True.
    figsize : 2-tuple of ints, optional.
        If a new figure is created for this plot, this kwarg determines the
        width and height of the figure that is created. Default is `(5, 3)`.
    fontsize : int or None, optional.
        The fontsize to be used in the plot. Default is 12.
    xlim, ylim : 2-tuple of ints or None, optional.
        Denotes the extent that will be set on the x-axis and y-axis,
        respectively, of the matplotlib Axes instance that is drawn on. If
        None, then the extent will not be manually altered. Default is None.
    output_file : str, or None, optional.
        Denotes the relative or absolute filepath (including the file format)
        that is to be used to save the plot. If None, the plot will not be
        saved to file. Default is None.
    dpi : positive int, optional.
        Denotes the number of 'dots per inch' for the saved figure. Will only
        be used if `output_file is not None`. Default == 500.
    kwargs : passed to `ax.plot` call in matplotlib.

    Returns
    -------
    None.
    """
    if rseed is not None:
        np.random.seed(rseed)

    if n_traces is not None:
        selected_cols = np.random.choice(sim_y.shape[1], size=n_traces)
        sim_y = sim_y[:, selected_cols]

    if thin_pct is not None:
        # Determine the number of rows to select
        num_selected_rows = int(thin_pct * sim_y.shape[1])
        # Randomly choose rows to retain.
        selected_rows =\
            np.random.choice(sim_y.shape[0],
                             size=num_selected_rows,
                             replace=False)
        # Filter the simulated-y, df, and filtering values
        sim_y = sim_y[selected_rows, :]
        orig_df = orig_df.iloc[selected_rows, :]
        filter_idx = filter_idx[selected_rows]

    sample_iterator = PROGRESS(xrange(sim_y.shape[1]), desc='Calculating CDFs')

    # Create a function to evaluate the choice condition:
    def choice_evaluator(choice_array, choice_condition):
        if choice_condition in [0.0, 1.0]:
            return choice_array == choice_condition
        else:
            return choice_array

    # Determine if col_to_plot is a string or an array
    col_to_plot_is_string = isinstance(col_to_plot, basestring)
    col_to_plot_is_array = isinstance(col_to_plot, np.ndarray)

    # Get the original values
    orig_choices = orig_df[choice_col].values

    orig_plotting_idx =\
        filter_idx & choice_evaluator(orig_choices, choice_condition)

    if col_to_plot_is_string:
        orig_plotting_vals = orig_df.loc[orig_plotting_idx, col_to_plot].values
    elif col_to_plot_is_array:
        orig_plotting_vals = col_to_plot[orig_plotting_idx]

    if fig_and_ax is None:
        fig, axis = plt.subplots(1, figsize=figsize)
        fig_and_ax = [fig, axis]
    else:
        fig, axis = fig_and_ax

    # This will track how many simulated datasets had
    # no outcome meeting the choice condition and the
    # filter condition.
    num_null_choices = 0

    # store the minimum and maximum x-values
    min_x = orig_plotting_vals.min()
    max_x = orig_plotting_vals.max()

    for i in sample_iterator:
        current_choices = sim_y[:, i]

        current_num_pos =\
            (current_choices[np.where(filter_idx)] == choice_condition).sum()

        if current_num_pos == 0:
            num_null_choices += 1
            continue

        current_choice_validity =\
            choice_evaluator(current_choices, choice_condition)

        # Determine the final rows to use for plotting
        plotting_idx = filter_idx & current_choice_validity

        # Get the values for plotting
        if col_to_plot_is_string:
            current_plotting_vals =\
                orig_df.loc[plotting_idx, col_to_plot].values
        elif col_to_plot_is_array:
            current_plotting_vals = col_to_plot[plotting_idx]
        assert current_plotting_vals.size > 0

        # Update the plot extents
        min_x = min(current_plotting_vals.min(), min_x)
        max_x = max(current_plotting_vals.max(), max_x)

        _plot_single_cdf_on_axis(current_plotting_vals,
                                 axis,
                                 color=sim_color,
                                 alpha=0.5,
                                 **kwargs)

    # Plot the originally observed relationship
    _plot_single_cdf_on_axis(orig_plotting_vals,
                             axis,
                             color=orig_color,
                             label='Observed',
                             alpha=1.0,
                             **kwargs)

    if num_null_choices > 0:
        num_null_pct = num_null_choices / float(sim_y.shape[1])
        null_pct_density_equivalent = axis.get_ylim()[1] * num_null_pct
        null_label = "'No Obs' Simulations: {:.2%}".format(num_null_pct)
        axis.bar([0],
                 [null_pct_density_equivalent],
                 width=0.1 * np.ptp(orig_plotting_vals),
                 align='edge',
                 alpha=bar_alpha,
                 color=bar_color,
                 label=null_label)

    if label is not None:
        _patch = mpatches.Patch(color=sim_color, label=label)
        current_handles, current_labels = axis.get_legend_handles_labels()
        current_handles.append(_patch)
        current_labels.append(label)

        axis.legend(current_handles,
                    current_labels,
                    # loc=(1, 0.75),
                    loc='best',
                    fontsize=fontsize)

    # set the plot extents
    if xlim is None:
        axis.set_xlim((min_x, max_x))
    else:
        axis.set_xlim(xlim)

    if ylim is not None:
        axis.set_ylim(ylim)

    # Despine the plot
    sbn.despine()
    # Make plot labels
    axis.set_xlabel(col_to_plot, fontsize=fontsize)
    axis.set_ylabel('Cumulative\nDensity\nFunction',
                    fontsize=fontsize,
                    rotation=0,
                    labelpad=40)

    # Create the title
    if title is not None and title != '':
        if not isinstance(title, basestring):
            msg = "`title` MUST be a string."
            raise TypeError(msg)
        axis.set_title(title)

    # Save the plot if desired
    if output_file is not None:
        fig.tight_layout()
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')

    if show:
        fig.show()

    return None


def _determine_bin_obs(total, partitions):
    """
    Determines the number of observations that should be in a given partition.

    Parameters
    ----------
    total : positive int.
        Denotes the total number of observations that are to be partitioned.
    partitions : positive int.
        Denotes the number of partitions that are to be created. Should be
        less than or equal to `total`.

    Returns
    -------
    obs_per_partition : 1D ndarray of positive its.
        Denotes the number of observations to be placed in each partition.
        Will have one element per partition.
    """
    partitions_float = float(partitions)
    naive = int(total / partitions_float) * np.ones(partitions)
    correction = np.ones(partitions)
    correction[total % partitions:] = 0
    return (naive + correction).astype(int)


def _populate_bin_means_for_plots(x_vals,
                                  y_vals,
                                  obs_per_bin,
                                  mean_x,
                                  mean_y,
                                  auxillary_y=None,
                                  auxillary_mean=None):
    """
    Populate the mean per bin of the predicted probabilities, observed outcomes,
    and simulated outcomes.

    Parameters
    ----------
    x_vals : 1D ndarray of floats.
        Elements should be the sorted values to be placed on the x-axis.
    y_vals : 1D ndarray of floats.
        Elements should be the values to be averaged and placed on the y-axis.
        Should have been sorted in the same order as `x_vals`.
    obs_per_bin : 1D ndarray of positive ints.
        There should be one element per bin. Each element should denote the
        number of observations to be used in each partition.
    mean_x, mean_y : 1D ndarray.
        `mean_x.size` and `mean_y.size` should equal `obs_per_bin.size`.
    auxillary_y : 1D ndarray or None, optional.
        Same as `y_vals` except these elements denote additional values to be
        plotted on the y-axis.
    auxillary_mean : 1D ndarray or None, optional.
        Same as `mean_probs` and `mean_obs`.

    Returns
    -------
    mean_x : 1D ndarray.
        Will have 1 element per partition. Each value will denote the mean of
        the `x_vals` for all observations in the partition.
    mean_y : 1D ndarray.
        Will have 1 element per partition. Each value will denote the mean of
        the `y_vals` for all observations in the partition.
    auxillary_mean : 1D ndarray or None.
        Will have 1 element per partition. Each value will denote the mean of
        the `auxillary_y` for all observations in the partition. If
        `auxillary_mean` was passed as None, it will be returned as None.
    """
    # Initialize a row counter
    row_counter = 0

    # Iterate over each of the partitions
    for i in range(obs_per_bin.size):
        # Get the upper and lower ranges of the slice
        lower_row = row_counter
        upper_row = row_counter + obs_per_bin[i]

        # Get the particular observations we care about
        rel_x = x_vals[lower_row:upper_row]
        rel_y = y_vals[lower_row:upper_row]

        # Store the mean probs and mean y
        mean_x[i] = rel_x.mean()
        mean_y[i] = rel_y.mean()

        # Store the mean simulated y per group
        if auxillary_y is not None:
            rel_auxillary_y = auxillary_y[lower_row:upper_row]
            auxillary_mean[i] = rel_auxillary_y.mean()

        # Update the row counter
        row_counter += obs_per_bin[i]

    return mean_x, mean_y, auxillary_mean


def _check_reliability_args(probs, choices, partitions, sim_y):
    """
    Ensures `probs` is a 1D or 2D ndarray, that `choices` is a 1D ndarray, that
    `partitions` is an int, and that `sim_y` is a ndarray of the same shape as
    `probs` or None.
    """
    if not isinstance(probs, np.ndarray):
        msg = '`probs` MUST be an ndarray.'
        raise ValueError(msg)
    if probs.ndim not in [1, 2]:
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
    sim_to_prob_conditions = probs.ndim != 1 and sim_y.shape != probs.shape
    if sim_y is not None and sim_to_prob_conditions:
        msg = ('`sim_y` MUST have the same shape as `probs` if '
               '`probs.shape[1] != 1`.')
        raise ValueError(msg)
    return None


def plot_binned_reliability(probs,
                            choices,
                            partitions=10,
                            line_color=None,
                            line_label=None,
                            alpha=None,
                            sim_y=None,
                            sim_line_color=None,
                            sim_label=None,
                            sim_alpha=0.5,
                            x_label=None,
                            title=None,
                            fontsize=12,
                            ref_line=False,
                            figsize=(5, 3),
                            fig_and_ax=None,
                            legend=True,
                            progress=True,
                            show=True,
                            output_file=None,
                            dpi=500):
    """
    Creates a binned reliability plot based on the given probability
    predictions and the given observed outcomes.

    Parameters
    ----------
    probs : 1D or 2D ndarray.
        Each element should be in [0, 1]. There should be 1 column for each
        set of predicted probabilities.
    choices : 1D ndarray.
        Each element should be either a zero or a one. Elements should denote
        whether the alternative corresponding to the given row was chosen or
        not. A 'one' corresponds to a an outcome of 'success'.
    partitions : positive int.
        Denotes the number of partitions to split one's data into for binning.
    line_color : valid matplotlib color, or `None`, optional.
        Determines the color that is used to plot the predicted probabilities
        versus the observed choices. Default is `None`.
    line_label : str, or None, optional.
        Denotes the label to be used for the lines relating the predicted
        probabilities and the binned, empirical probabilities. Default is None.
    alpha : positive float in [0.0, 1.0], or `None`, optional.
        Determines the opacity of the elements drawn on the plot.
        0.0 == transparent and 1.0 == opaque. Default == 1.0.
    sim_y : 2D ndarray or None, optional.
        Denotes the choices that were simulated based on `probs`. If passed,
        `sim_y.shape` MUST equal `probs.shape` in order to ensure that lines
        are plotted for the predicted probabilities versus simulated choices.
        This kwarg is useful because it shows one the reference distribution of
        predicted probabilities versus choices that actually come from one's
        postulated model.
    sim_line_color : valid matplotlib color, or `None`, optional.
        Determines the color that is used to plot the predicted probabilities
        versus the simulated choices. Default is `None`.
    sim_line_label : str, or None, optional.
        Denotes the label to be used for the lines relating the predicted
        probabilities and the binned, empirical probabilities based on the
        simulated choices. Default is None.
    sim_alpha : positive float in [0.0, 1.0], or `None`, optional.
        Determines the opacity of the simulated reliability curves.
        0.0 == transparent and 1.0 == opaque. Default == 0.5.
    x_label : str, or None, optional.
        Denotes the label for the x-axis. If None, the x-axis will be labeled
        as 'Mean Predicted Probability'. Default is None.
    title : str, or None, optional.
        Denotes the title to be displayed for the plot. Default is None.
    fontsize : int or None, optional.
        The fontsize to be used in the plot. Default is 12.
    ref_line : bool, optional.
        Determines whether a diagonal line, y = x, will be plotted to show the
        expected relationship. Default is True.
    figsize : 2-tuple of positive ints.
        Determines the size of the created figure. Default == (5, 3).
    fig_and_ax : list of matplotlib figure and axis, or `None`, optional.
        Determines whether a new figure will be created for the plot or whether
        the plot will be drawn on existing axes. If None, a new figure will be
        created. Default is `None`.
    legend : bool, optional.
        Determines whether a legend is printed for the plot. Default == True.
    progress : bool, optional.
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
    _check_reliability_args(probs, choices, partitions, sim_y)

    # Make probs 2D if necessary
    probs = probs[:, None] if probs.ndim == 1 else probs

    # Create the figure and axes if need be
    if fig_and_ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
        fig_and_ax = [fig, ax]
    else:
        fig, ax = fig_and_ax

    # Choose colors for the plot if necesssary
    if line_color is None:
        line_color =\
            (0.6509803921568628, 0.807843137254902, 0.8901960784313725)
    if sim_line_color is None:
        sim_line_color =\
            (0.792156862745098, 0.6980392156862745, 0.8392156862745098)

    # Create the progressbar iterator if desired
    if progress and sim_y is not None:
        description = "Plotting" if sim_y is None else "Plotting Simulations"
        sim_iterator = PROGRESS(range(sim_y.shape[1]), desc=description)
    else:
        sim_iterator = range(probs.shape[1])

    # Determine the number of observations in each partition
    obs_per_partition = _determine_bin_obs(probs.shape[0], partitions)

    # Initialize an array for the mean probabilities and
    # observations in each group.
    mean_probs_per_group = np.zeros(partitions)
    mean_y_per_group = np.zeros(partitions)

    # Create a function to get the current probabilities when plotting the
    # simulated reliability plots
    def get_current_probs(col):
        current = probs[:, 0] if probs.shape[1] == 1 else probs[:, col]
        return current

    # Plot the simulated reliability curves, if desired
    if sim_y is not None:
        for i in sim_iterator:
            current_label = sim_label if i == 0 else None
            plot_binned_reliability(get_current_probs(i),
                                    sim_y[:, i],
                                    partitions=partitions,
                                    line_color=sim_line_color,
                                    line_label=current_label,
                                    alpha=sim_alpha,
                                    sim_y=None,
                                    sim_line_color=None,
                                    sim_label=None,
                                    title=None,
                                    fontsize=fontsize,
                                    ref_line=False,
                                    figsize=figsize,
                                    fig_and_ax=fig_and_ax,
                                    legend=False,
                                    progress=False,
                                    show=False,
                                    output_file=None,
                                    dpi=dpi)

    # Create the progressbar iterator if desired
    if progress:
        prob_iterator = PROGRESS(range(probs.shape[1]), desc="Plotting")
    else:
        prob_iterator = range(probs.shape[1])

    # Make the 'true' reliability plots
    for col in prob_iterator:
        # Get the current line label and probabilities
        current_line_label = line_label if col == 0 else None
        current_probs = probs[:, col]

        # Sort the array of probs and choices
        sort_order = np.argsort(current_probs)
        current_probs = current_probs[sort_order]
        current_choices = choices[sort_order]

        # Populate the bin means of predicted probabilities,
        # observed choices, and simulated choices
        population_results =\
            _populate_bin_means_for_plots(
                current_probs, current_choices,
                obs_per_partition, mean_probs_per_group, mean_y_per_group)
        mean_probs_per_group = population_results[0]
        mean_y_per_group = population_results[1]

        # Plot the mean predicted probs per group versus
        # the mean observations per group
        ax.plot(mean_probs_per_group,
                mean_y_per_group,
                c=line_color,
                alpha=alpha,
                label=current_line_label)

    # Create the reference line if desired
    if ref_line:
        # Determine the maximum value of the x-axis or y-axis
        max_ref_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        min_ref_val = max(ax.get_xlim()[0], ax.get_ylim()[0])
        # Determine the values to use to plot the reference line
        ref_vals = np.linspace(min_ref_val, max_ref_val, num=100)
        # Plot the reference line as a black dashed line
        ax.plot(ref_vals, ref_vals, 'k--', label='Perfect Calibration')

    # Label the plot axes
    if x_label is None:
        ax.set_xlabel('Mean Predicted Probability', fontsize=fontsize)
    else:
        ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel('Binned\nEmpirical\nProbability',
                  fontsize=fontsize,
                  rotation=0,
                  labelpad=40)

    # Make the title if desired
    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    # Make the legend, if desired
    if legend:
        ax.legend(loc='best', fontsize=fontsize)

    # Save the plot if desired
    if output_file is not None:
        fig.tight_layout()
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')

    # Show the plot if desired
    if show:
        fig.show()
    return None


def _check_mmplot_ref_vals(probs, ref_vals):
    """
    Checks argument validity for the marginal model plots. Ensures `ref_vals`
    is a 1D ndarray with the same number of rows as `probs`.
    """
    if not isinstance(ref_vals, np.ndarray) or ref_vals.ndim != 1:
        msg = "`ref_vals` MUST be a 1D ndarray."
        raise ValueError(msg)
    elif ref_vals.shape[0] != probs.shape[0]:
        msg = "`ref_vals` MUST have the same number of rows as `probs`."
        raise ValueError(msg)
    return None


def _plot_single_binned_x_vs_binned_y(x_vals,
                                      y_vals,
                                      obs_per_partition,
                                      mean_x_per_group,
                                      mean_y_per_group,
                                      color,
                                      alpha,
                                      label,
                                      ax):
    # Populate the bin means of predicted probabilities,
    # observed choices, and simulated choices
    population_results =\
        _populate_bin_means_for_plots(x_vals, y_vals, obs_per_partition,
                                      mean_x_per_group, mean_y_per_group)
    mean_x_per_group = population_results[0]
    mean_y_per_group = population_results[1]

    # Plot the mean predicted probs per group versus
    # the mean observations per group
    ax.plot(mean_x_per_group, mean_y_per_group,
            c=color, alpha=alpha, label=label)
    return None


def make_binned_marginal_model_plot(probs,
                                    choices,
                                    ref_vals,
                                    sim_y=None,
                                    partitions=10,
                                    y_color=None,
                                    prob_color=None,
                                    sim_color=None,
                                    y_label='Observed',
                                    prob_label='Predicted',
                                    sim_label='Simulated',
                                    x_label=None,
                                    alpha=None,
                                    title=None,
                                    fontsize=12,
                                    figsize=(5, 3),
                                    fig_and_ax=None,
                                    legend=True,
                                    progress=True,
                                    show=True,
                                    output_file=None,
                                    dpi=500):
    """
    Creates a binned marginal model plot based on the given probability
    predictions, observed outcomes, and refernce values.

    Parameters
    ----------
    probs : 1D or 2D ndarray.
        Each element should be in [0, 1]. There should be 1 column for each
        set of predicted probabilities.
    choices : 1D ndarray.
        Each element should be either a zero or a one. Elements should denote
        whether the alternative corresponding to the given row was chosen or
        not. A 'one' corresponds to a an outcome of 'success'.
    ref_vals : 1D ndarray of floats.
        These should be the elements to plot on the x-axis. `ref_vals` should
        represent a continuous variable. Should have the same number of rows as
        `probs` and `choices`.
    sim_y : 2D ndarray or None, optional.
        Denotes the choices that were simulated based on `probs`. If passed,
        `sim_y.shape` MUST equal `probs.shape` in order to ensure that lines
        are plotted for the simulated choices versus `ref_vals`. This kwarg is
        useful because it shows one the reference distribution of choices
        versus `ref_vals` for choices that actually come from one's model.
        Default = None.
    partitions : positive int.
        Denotes the number of partitions to split one's data into for binning.
    y_color, prob_color, sim_color : matplotlib color, or `None`, optional.
        Determines the color that is used to plot the observed choices,
        predicted probabilities, and simulated choices versus `ref_vals`.
        Default is `None`.
    y_label, prob_label, sim_label : str, or None, optional.
        Denotes the label to be used for the lines relating the observed
        choices, predicted probabilities, and simulated choices to the
        `ref_vals`. Default == ['Observed', 'Predicted', 'Simulated'].
    x_label : str, or None, optional.
        The label for the x-axis of the plot. If None, the x-axis will be
        labeled 'Binned, Mean Reference Values.' Default is `None`.
    alpha : positive float in [0.0, 1.0], or `None`, optional.
        Determines the opacity of the elements drawn on the plot.
        0.0 == transparent and 1.0 == opaque. Default is `None`.
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
    progress : bool, optional.
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
    _check_reliability_args(probs, choices, partitions, sim_y)
    _check_mmplot_ref_vals(probs, ref_vals)

    # Make probs 2D if necessary
    probs = probs[:, None] if probs.ndim == 1 else probs

    # Sort the arguments, if necesssary
    sort_order = np.argsort(ref_vals)
    ref_vals = ref_vals[sort_order]
    probs = probs[sort_order, :]
    choices = choices[sort_order]
    if sim_y is not None:
        sim_y = sim_y[sort_order, :]

    # Create the figure and axes if need be
    if fig_and_ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
        fig_and_ax = [fig, ax]
    else:
        fig, ax = fig_and_ax

    # Choose colors for the plot if necesssary
    if y_color is None:
        y_color =\
            (0.12156862745098039, 0.47058823529411764, 0.7058823529411765)
    if prob_color is None:
        prob_color =\
            (0.6509803921568628, 0.807843137254902, 0.8901960784313725)
    if sim_color is None:
        sim_color =\
            (0.792156862745098, 0.6980392156862745, 0.8392156862745098)

    # Create the progressbar iterator if desired
    if progress:
        description = "Plotting" if sim_y is None else "Plotting Simulations"
        prob_iterator = PROGRESS(range(probs.shape[1]), desc=description)
    else:
        prob_iterator = range(probs.shape[1])

    # Determine the number of observations in each partition
    obs_per_partition = _determine_bin_obs(probs.shape[0], partitions)

    # Initialize arrays to store the mean x- and y-values per group
    mean_x_per_group = np.zeros(partitions)
    mean_y_per_group = np.zeros(partitions)

    #####
    # Plot the simulated reliability curves, if desired
    #####
    if sim_y is not None:
        for i in prob_iterator:
            current_label = sim_label if i == 0 else None
            _plot_single_binned_x_vs_binned_y(ref_vals,
                                              sim_y[:, i],
                                              obs_per_partition,
                                              mean_x_per_group,
                                              mean_y_per_group,
                                              sim_color,
                                              alpha,
                                              current_label,
                                              ax)

    # Create the progressbar iterator if desired
    if progress:
        prob_iterator = PROGRESS(range(probs.shape[1]), desc="Plotting")
    else:
        prob_iterator = range(probs.shape[1])

    #####
    # Plot the probabilities versus the ref values.
    #####
    for col in prob_iterator:
        # Get the current line label and probabilities
        current_label = prob_label if col == 0 else None
        current_probs = probs[:, col]

        _plot_single_binned_x_vs_binned_y(ref_vals,
                                          current_probs,
                                          obs_per_partition,
                                          mean_x_per_group,
                                          mean_y_per_group,
                                          prob_color,
                                          alpha,
                                          current_label,
                                          ax)
    #####
    # Plot choices versus ref_vals
    #####
    # Make sure the 'true' relationship is not transparent
    observed_alpha = 1.0
    _plot_single_binned_x_vs_binned_y(ref_vals,
                                      choices,
                                      obs_per_partition,
                                      mean_x_per_group,
                                      mean_y_per_group,
                                      y_color,
                                      observed_alpha,
                                      y_label,
                                      ax)

    # Label the plot axes
    if x_label is None:
        ax.set_xlabel('Binned, Mean Reference Values', fontsize=fontsize)
    else:
        ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel('Binned,\nMean\nProbability',
                  fontsize=fontsize, rotation=0, labelpad=40)

    # Make the title if desired
    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    # Make the legend, if desired
    if legend:
        ax.legend(loc='best', fontsize=fontsize)

    # Despine the plot
    sbn.despine()

    # Save the plot if desired
    if output_file is not None:
        fig.tight_layout()
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')

    # Show the plot if desired
    if show:
        fig.show()
    return None


def plot_predicted_log_likelihoods(log_likes,
                                   obs_log_like,
                                   kde=True,
                                   fig_and_ax=None,
                                   figsize=(10, 6),
                                   sim_color='#a6bddb',
                                   sim_label='Simulated',
                                   obs_label='Observed',
                                   x_label='Log-Likelihood',
                                   y_label='Density',
                                   fontsize=12,
                                   output_file=None,
                                   dpi=500,
                                   show=True):
    """
    Plots the distribution of predicted log-likelihoods versus the observed log-
    likelihood.

    Parameters
    ----------
    log_likes : 1D ndarray of floats.
        The array of log-likelihood values, with 1 value per simulation.
    obs_log_like : int, float, or long.
        The scalar log-likelihood for one's model with the observed outcomes.
    kde : bool, optional.
        Determines whether a kernel density estimate is plotted. If `kde=False`,
        a cumulative density plot is made.
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
        observed log-likelihoods, the x-axis, and the y-axis. Defaults are
        `['Simulated', 'Observed', 'Log-Likelihood', 'Density']`.
    fontsize : int or None, optional.
        The fontsize to be used in the plot. Default is 12.
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

    # Plot the distribution of log-likelihoods estimate
    if kde:
        sbn.kdeplot(log_likes, ax=axis, label=sim_label)
    else:
        _plot_single_cdf_on_axis(log_likes,
                                 axis,
                                 color=sim_color,
                                 linestyle='-',
                                 label=sim_label,
                                 alpha=1.0)

    # Figure out the axis boundaries
    min_y, max_y = axis.get_ylim()

    # Calculate the percentile corresponding to the observed log-likelihood
    simulated_frac_below_observed =\
        (log_likes < obs_log_like).sum() / float(log_likes.size)

    # Create the vertical line to show the observed log-likelihood
    line_label =\
        obs_label + '\nP(samples < observed) = {:.0%}'
    axis.vlines(obs_log_like,
                min_y,
                max_y,
                linestyle='dashed',
                label=line_label.format(simulated_frac_below_observed))

    # Despine the plot
    sbn.despine()
    # Make plot labels
    axis.set_xlabel(x_label, fontsize=fontsize)
    axis.set_ylabel(y_label, fontsize=fontsize, rotation=0, labelpad=40)
    axis.legend(loc='best', fontsize=fontsize)

    # Save the plot if desired
    if output_file is not None:
        fig.tight_layout()
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')

    if show:
        fig.show()

    return None


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
    # Get the positions and counts of the ch values of x
    unique_pos = np.unique(x, return_index=True)[1]

    # Determine the unique values in the x-array, in their original order.
    unique_vals = x[np.sort(unique_pos)]

    # Get the counts of the chosen values of x
    _val_names, _val_counts = np.unique(x[obs_y == 1], return_counts=True)
    obs_df = pd.DataFrame({x_label: _val_names, y_label: _val_counts})

    # Initialize an array of the simulated number of observations per value
    num_per_value_per_sim =\
        np.empty((unique_vals.size, sim_y.shape[1]))

    # Create the object to iterate over while populating `num_per_value_per_sim`
    iterator = PROGRESS(unique_vals, desc='Unique x-values')

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
        Determines the size of the created figure. Default == (5, 3).
    box_color, obs_color : valid matplotlib color argument, optional.
        Denotes the color of the boxes on the boxplot and the color used to plot
        the observed distribution of `x`. Default is 'white' and '#045a8d'.
    obs_marker : valid matplotlib marker argument, optional.
        Determines the marker used to plot the observed distribution of `x`.
        Default is '*'.
    obs_size : int, optional.
        Determines the size of the marker used to plot the observed distribution
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
    # Ensure the display dict has all possible values that in x.
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

    # Add axis labels to the plot
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize, rotation=0, labelpad=40)

    # Ensure that the xticklabels are of the correct fontsize
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)

    # Draw the legend, ensuring that we only have one entry.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:1], labels[:1], loc='best', fontsize=fontsize)

    # Save the plot if desired
    if output_file is not None:
        fig.tight_layout()
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')

    if show:
        fig.show()

    return None
