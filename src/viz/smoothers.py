# -*- coding: utf-8 -*-
"""
This file contains functions and classes for producing discrete and/or
continuous smooths of a binary or continuous variable against another
continuous variable.
"""
import numpy as np

# Use ExtRaTrees for continuously smoothed marginal model plots
from sklearn.ensemble import (ExtraTreesClassifier,
                              ExtraTreesRegressor)

try:
    # in Python 3 range returns an iterator instead of list
    # to maintain backwards compatibility use "old" version of range
    from past.builtins import range
except ImportError:
    pass


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
    Populate the mean per bin of predicted probabilities, observed outcomes,
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
        Same as `mean_x` and `mean_y`.

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


def _get_extra_smooth_xy(x, y,
                         n_estimators=50,
                         min_samples_leaf=10,
                         random_state=None):
    """
    Creates an ensemble of extremely randomized trees that predict y given x,
    and returns the smoothed (i.e. predicted) y and original x.

    Parameters
    ----------
    x, y : 1D ndarray of real values in [0, 1].
        X should be an array of continuous values. y should be an array of
        either continuous or binary (0 or 1) data.
    n_estimators : positive int, optional.
        Determines the number of trees in the ensemble. This parameter controls
        how smooth one's resulting estimate is. The more estimators the
        smoother one's estimated relationship and the lower the variance in
        that estimated relationship. Default == 50.
    min_samples_leaf : positive int, optional.
        Determines the minimum number of observations allowed in a leaf node in
        any tree in the ensemble. This parameter is conceptually equivalent to
        the bandwidth parameter in a kernel density estimator. Default == 10.
    random_state : positive int, or None, optional.
        Denotes the random seed to be used when constructing the ensemble of
        Extremely Randomized Trees. Default is None.

    Returns
    -------
    x, smoothed_y : 1D ndarray of real values.
        x is the same as above. `smoothed_y` is the predicted y values based on
        the ensemble of extremely randomized trees.
    """
    if not isinstance(x, np.ndarray) or len(x.shape) != 1:
        msg = 'x MUST be a 1D ndarray'
        raise ValueError(msg)
    if not isinstance(y, np.ndarray) or len(y.shape) != 1:
        msg = 'y MUST be a 1D ndarray'
        raise ValueError(msg)
    # The if condition checks if we are dealing with continuous y vs discrete y
    if ((y < 1.0) & (y > 0)).any():
        smoother = ExtraTreesRegressor(n_estimators=n_estimators,
                                       min_samples_leaf=min_samples_leaf,
                                       max_features=1,
                                       random_state=random_state)
        smoother.fit(x[:, None], y)
        smoothed_y = smoother.predict(x[:, None])
    else:
        smoother = ExtraTreesClassifier(n_estimators=n_estimators,
                                        min_samples_leaf=min_samples_leaf,
                                        max_features=1,
                                        random_state=random_state)
        smoother.fit(x[:, None], y)
        # Note we use [:, 1] to get the predicted probabilities of y = 1
        smoothed_y = smoother.predict_proba(x[:, None])[:, 1]
    return x, smoothed_y


class Smoother(object):
    """
    Base class for the discrete and continuous smoothers. Instances of
    subclasses of `Smoother` will take in raw X and Y values, and they will
    output new x and y values that can be plotted to show the smoothed
    conditional expectation function, E[y | x].
    """
    def __init__(self):
        return None

    def __call__(self, X, Y):
        """
        Takes in raw X and Y and produces smoothed_x and smoothed_y. The
        outputs can then be plotted to visualize the smoothed,
        conditional expectation function, E[y | x], according to the specified
        smoother.

        Parameters
        ----------
        X, Y : 1D ndarrays
            Should contain the raw data for which we want to visualize a smooth
            of the conditional expectation function, E[y | x].

        Returns
        -------
        smoothed_x, smoothed_y : 1D ndarrays
            Contains the smoothed values to be plotted, respectively, on the
            x-axis and y-axis to show the smoothed E[y | x].
        """
        return self.smooth(X, Y)

    def smooth(self, X, Y):
        """
        Takes in raw X and Y and produces smoothed_x and smoothed_y. The
        outputs can then be plotted to visualize the smoothed,
        conditional expectation function, E[y | x], according to the specified
        smoother.

        Parameters
        ----------
        X, Y : 1D ndarrays
            Should contain the raw data for which we want to visualize a smooth
            of the conditional expectation function, E[y | x].

        Returns
        -------
        smoothed_x, smoothed_y : 1D ndarrays
            Contains the smoothed values to be plotted, respectively, on the
            x-axis and y-axis to show the smoothed E[y | x].
        """
        raise NotImplementedError


class DiscreteSmoother(Smoother):
    """
    A Smoother object that takes in continuous X and binary or continuous Y and
    computes a discrete (i.e binned) smooth of the conditional expectation
    function, E[y | x].

    Parameters
    ----------
    num_obs : positive int.
        Determines the number of observations in the vectors of X and Y that
        will later be smoothed. This arg is needed to optimize computations
        when calculating many smooths.
    partitions : positive int, optional.
        Denotes the number of partitions to split one's data into for binning.
        Default == 10.
    """
    def __init__(self,
                 num_obs,
                 partitions=10):
        super(DiscreteSmoother, self).__init__()
        self.num_obs = num_obs
        self.partitions = partitions

        # Initialize attributes for discrete smoothing (i.e. binning)
        self.mean_x_per_group = np.zeros(self.partitions)
        self.mean_y_per_group = np.zeros(self.partitions)

        # Determine the number of observations in each partition
        self.obs_per_partition = _determine_bin_obs(num_obs, self.partitions)
        return None

    def smooth(self, X, Y):
        """
        Takes in raw X and Y and produces smoothed_x and smoothed_y. The
        outputs can then be plotted to visualize the smoothed,
        conditional expectation function, E[y | x], according to the specified
        smoother.

        Parameters
        ----------
        X, Y : 1D ndarrays
            Should contain the raw data for which we want to visualize a smooth
            of the conditional expectation function, E[y | x].

        Returns
        -------
        smoothed_x, smoothed_y : 1D ndarrays
            Contains the smoothed values to be plotted, respectively, on the
            x-axis and y-axis to show the smoothed E[y | x].
        """
        return _populate_bin_means_for_plots(X,
                                             Y,
                                             self.obs_per_partition,
                                             self.mean_x_per_group,
                                             self.mean_y_per_group)[:2]


class ContinuousSmoother(Smoother):
    """
    A Smoother object that takes in continuous X and binary or continuous Y and
    computes a continuous smooth of the conditional expectation function,
    E[y | x], using an ensemble of Extremely Randomized Trees.

    Parameters
    ----------
    n_estimators : positive int, optional.
        Determines the number of trees in the ensemble of Extremely Randomized
        Trees that is used to do continuous smoothing. This parameter controls
        how smooth one's resulting estimate is. The more estimators the
        smoother one's estimated relationship and the lower the variance in
        that estimated relationship. Default == 50.
    min_samples_leaf : positive int, optional.
        Determines the minimum number of observations allowed in a leaf node in
        any tree in the ensemble. This parameter is conceptually equivalent to
        the bandwidth parameter in a kernel density estimator. Default == 10.
    random_state : positive int, or None, optional.
        Denotes the random seed to be used when constructing the ensemble of
        Extremely Randomized Trees. Default is None.
    """
    def __init__(self,
                 n_estimators=50,
                 min_samples_leaf=10,
                 random_state=None):
        super(ContinuousSmoother, self).__init__()
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        return None

    def smooth(self, X, Y):
        """
        Takes in raw X and Y and produces smoothed_x and smoothed_y. The
        outputs can then be plotted to visualize the smoothed,
        conditional expectation function, E[y | x], according to the specified
        smoother.

        Parameters
        ----------
        X, Y : 1D ndarrays
            Should contain the raw data for which we want to visualize a smooth
            of the conditional expectation function, E[y | x].

        Returns
        -------
        smoothed_x, smoothed_y : 1D ndarrays
            Contains the smoothed values to be plotted, respectively, on the
            x-axis and y-axis to show the smoothed E[y | x].
        """
        return _get_extra_smooth_xy(X, Y,
                                    n_estimators=self.n_estimators,
                                    min_samples_leaf=self.min_samples_leaf,
                                    random_state=self.random_state)


class SmoothPlotter(object):
    """
    An object that plots single smooths of the conditional expectation
    function, E[y | x].

    Parameters
    ----------
    smoother : an instance or instance of a subclass of Smoother.
        This arg is used to produce the smooths that are then plotted.
    ax : an instance of matplotlib.Axes.
        The axi on which the smooth is plotted.
    """
    def __init__(self, smoother, ax):
        self.ax = ax
        self.smoother = smoother
        return None

    def plot(self, X, Y, label=None, color='#a6cee3', alpha=0.5, sort=False):
        """
        Plots a smooth estimate of the conditional expectation function E[y|x].

        Parameters
        ----------
        X, Y : 1D ndarrays
            Should contain the raw data for which we want to visualize a smooth
            of the conditional expectation function, E[y | x].
        label : str or None, optional.
            Denotes the label for the plotted curve. Default is None.
        color : valid matplotlib color, optional.
            The color that is used for the plotted curve. Default is '#a6cee3'.
        alpha : positive float in [0.0, 1.0], or `None`, optional.
            Determines the opacity of the elements drawn on the plot.
            0.0 == transparent and 1.0 == opaque. Default == 0.5.
        sort : bool, optional.
            Determines if `X` and `Y` will be sorted before they are smoothed
            and plotted. Default is False.

        Returns
        -------
        None.
        """
        if sort:
            sort_order = np.argsort(X)
            sorted_x, sorted_y = X[sort_order], Y[sort_order]
        else:
            sorted_x, sorted_y = X, Y
        # Get the smoothed x and y values to be plotted.
        plot_x, plot_y = self.smoother(sorted_x, sorted_y)
        # Make the desired plot
        self.ax.plot(plot_x, plot_y, c=color, alpha=alpha, label=label)
        return None
