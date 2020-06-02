"""
Helper functions for plotting.
"""
import sys
import gc
import numpy as np

import matplotlib.pyplot as plt

# Use statsmodels for empirical cdf function
import statsmodels.tools as sm_tools
import statsmodels.distributions as sm_dist

# Alias the empirical cdf function. The if-statement is used for compatibility
# with various statsmodels versions.
ECDF = sm_tools.tools.ECDF if hasattr(sm_tools.tools, 'ECDF') else sm_dist.ECDF

# Allow for python 2 and python 3 compatibility
try:
    basestring
except NameError:
    basestring = str


def _label_despine_save_and_show_plot(x_label,
                                      y_label,
                                      fig_and_ax,
                                      fontsize=12,
                                      y_rot=0,
                                      y_pad=40,
                                      title=None,
                                      output_file=None,
                                      show=True,
                                      dpi=500):
    """
    Adds the x-label, y-label, and title to the matplotlib Axes object. Also
    despines the figure, saves it (if desired), and shows it (if desired).

    Parameters
    ----------
    x_label, y_label : string.
        Determines the labels for the x-axis and y-axis respectively.
    fig_and_ax : list of matplotlib figure and axis.
        The matplotlib figure and axis that are being altered.
    fontsize : int or None, optional.
        The fontsize to be used in the plot. Default is 12.
    y_rot : int in [0, 360], optional.
        Denotes the angle by which to rotate the text of the y-axis label.
        Default == 0.
    y_pad : int, optional.
        Denotes the amount by which the text of the y-axis label will be offset
        from the y-axis itself. Default == 40.
    title : string or None, optional.
        Denotes the title to be displayed for the plot. Default is None.
    output_file : str, or None, optional.
        Denotes the relative or absolute filepath (including the file format)
        that is to be used to save the plot. If None, the plot will not be
        saved to file. Default is None.
    show : bool, optional.
        Determines whether the figure is shown after plotting is complete.
        Default == True.
    dpi : positive int, optional.
        Denotes the number of 'dots per inch' for the saved figure. Will only
        be used if `output_file is not None`. Default == 500.
    """
    # Ensure seaborn is imported
    if 'sbn' not in globals():
        import seaborn as sbn

    # Get the figure and axis as separate objects
    fig, axis = fig_and_ax

    # Despine the plot
    sbn.despine()
    # Make plot labels
    axis.set_xlabel(x_label, fontsize=fontsize)
    axis.set_ylabel(y_label, fontsize=fontsize, rotation=y_rot, labelpad=y_pad)
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

    # Explicitly close the figure
    notebook_env = bool(any([x in sys.modules for x in ['ipykernel', 'IPython']]))
    if not notebook_env:
        plt.cla()
        plt.close(fig)
        plt.close('all')
        gc.collect()
    return None


def _choice_evaluator(choice_array, choice_condition):
    """
    Determines which rows in `choice_array` meet the given `choice_condition`,
    where `choice_condition` is in the set `{0.0, 1.0}`.

    Parameters
    ----------
    choice_array : 1D ndarray of ints that are either 0 or 1.
    choice_condition : int in `{0, 1}`.

    Returns
    -------
    bool_mask : 1D ndarray of bools.
        Equal to `choice_array == choice_condition`
    """
    if choice_condition in [0.0, 1.0]:
        return choice_array == choice_condition
    else:
        msg = 'choice_condition MUST be either a 0 or a 1'
        raise ValueError(msg)


def _thin_rows(sim_y, thin_pct):
    """
    Randomly select `thin_pct` percentage of rows to be used in plotting.

    Parameters
    ----------
    sim_y : 2D ndarray of zeros and ones.
        Each row should represent an alternative for a given choice situation.
        Each column should represent a given simulated set of choices.
    thin_pct : float in (0.0, 1.0) or None, optional.
        Determines the percentage of the data (rows) to be used for plotting.
        If None, the full dataset will be used. Default is None.

    Returns
    -------
    selected_rows : 1D ndarray of bools.
        Denotes the randomly selected rows to be used in plotting.
    """
    # Determine the number of rows to select
    num_selected_rows = int(thin_pct * sim_y.shape[0])
    # Randomly choose rows to retain.
    selected_rows =\
        np.random.choice(sim_y.shape[0],
                         size=num_selected_rows,
                         replace=False)
    return selected_rows


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
