import numpy as np
from scipy.stats import rankdata

__all__ = ['make_fitness']


class _Fitness(object):

    """
    Metric for measuring fitness of programs within the population
    
    Parameters:
        function (callable) -- a function with signature function that returns a floating point number
        greater_is_better (boolean) -- whether a higher value from 'function' indicates a better fit. Determine whether a larger or smaller value of the function is desirable 

    """

    def __init__(self, function, greater_is_better):
        self.function = function
        self.greater_is_better = greater_is_better
        self.sign = 1 if greater_is_better else -1 # If a larger value is better, the sign should be positive

    def __call__(self, *args):
        return self.function(*args)


def make_fitness(function, greater_is_better):
    """
    Make a fitness measure, a metric scoring the quality of a program's fit.
    This factory function creates a fitness measure object which measures the
    quality of a program's fit and thus its likelihood to undergo genetic
    operations into the next generation. The resulting object is able to be
    called with NumPy vectorized arguments and return a resulting floating
    point score quantifying the quality of the program's representation of the
    true relationship.
    
    Parameters:
        function (callable) -- Equivalent to the fitness attribute in the above _Fitness object.
        greater_is_better (bool) -- Equivalent to the greater_is_better attribute in the above _Fitness object.
    """
    return _Fitness(function=function,
                    greater_is_better=greater_is_better)


def _weighted_pearson(y, y_pred, w):
    """Pearson correlation coefficient."""
    with np.errstate(divide='ignore', invalid='ignore'):
        y_pred_demean = y_pred - np.average(y_pred, weights=w)
        y_demean = y - np.average(y, weights=w)
        corr = ((np.sum(w * y_pred_demean * y_demean) / np.sum(w)) /
                np.sqrt((np.sum(w * y_pred_demean ** 2) *
                         np.sum(w * y_demean ** 2)) /
                        (np.sum(w) ** 2)))
    if np.isfinite(corr):
        return np.abs(corr)
    return 0.


def _weighted_spearman(y, y_pred, w):
    """the weighted Spearman correlation coefficient."""
    y_pred_ranked = np.apply_along_axis(rankdata, 0, y_pred)
    y_ranked = np.apply_along_axis(rankdata, 0, y)
    return _weighted_pearson(y_pred_ranked, y_ranked, w)


def _mean_absolute_error(y, y_pred, w):
    """the mean absolute error."""
    return np.average(np.abs(y_pred - y), weights=w)


def _mean_square_error(y, y_pred, w):
    """the mean square error."""
    return np.average(((y_pred - y) ** 2), weights=w)


def _root_mean_square_error(y, y_pred, w):
    """the root mean square error."""
    return np.sqrt(np.average(((y_pred - y) ** 2), weights=w))


def _r2(y, y_pred, w):
    correlation_matrix = np.corrcoef(y_pred, y)
    correlation_xy = correlation_matrix[0, 1]
    r_2 = correlation_xy ** 2
    return np.abs(r_2)

weighted_pearson = _Fitness(function=_weighted_pearson,
                            greater_is_better=True)
weighted_spearman = _Fitness(function=_weighted_spearman,
                             greater_is_better=True)
mean_absolute_error = _Fitness(function=_mean_absolute_error,
                               greater_is_better=False)
mean_square_error = _Fitness(function=_mean_square_error,
                             greater_is_better=False)
root_mean_square_error = _Fitness(function=_root_mean_square_error,
                                  greater_is_better=False)
r_2 = _Fitness(function=_r2, 
               greater_is_better=True)

_fitness_map = {'pearson': weighted_pearson,
                'spearman': weighted_spearman,
                'mean absolute error': mean_absolute_error,
                'mse': mean_square_error,
                'rmse': root_mean_square_error,
                'r_2': r_2}
