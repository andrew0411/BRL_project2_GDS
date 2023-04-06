'''
유전 알고리즘을 진행하면서 사용할 fitness function 들 정의
'''

import numbers

import numpy as np
from joblib import wrap_non_picklable_objects
from scipy.stats import rankdata

__all__ = ['make_fitness']


class _Fitness(object):

    """
    Population 내의 프로그램에 대해 fitness 를 측정하는 metric
    
    This object is able to be called with NumPy vectorized arguments and return
    a resulting floating point score quantifying the quality of the program's
    representation of the true relationship.
    
    Parameters:
        function (callable) -- a function with signature function that returns a floating point number
                            -- function(y, y_pred, sample_weight)에서 y는 input target y vector, y_pred는 the predicted values from genetic program, sample_weight는 sample weight vector
        greater_is_better (boolean) -- whether a higher value from 'function' indicates a better fit. 해당 function의 값이 큰 게 좋은 것인지 작은게 좋은 것인지 설정 

    """

    def __init__(self, function, greater_is_better):
        self.function = function
        self.greater_is_better = greater_is_better
        self.sign = 1 if greater_is_better else -1 # function이 큰 값이 좋으면 부호 양수

    def __call__(self, *args):
        return self.function(*args)


def make_fitness(function, greater_is_better, wrap=True):
    """
    Make a fitness measure, a metric scoring the quality of a program's fit.
    This factory function creates a fitness measure object which measures the
    quality of a program's fit and thus its likelihood to undergo genetic
    operations into the next generation. The resulting object is able to be
    called with NumPy vectorized arguments and return a resulting floating
    point score quantifying the quality of the program's representation of the
    true relationship.
    
    Parameters:
        function (callable) -- 위의 _Fitness 객체의 fitness와 동일
        greater_is_better (bool) -- 위의 _Fitness 객체의 greater_is_better과 동일
        wrap (bool, optional) -- |default=True| Parallel하게 학습시킬 때 속도저하 

            When running in parallel, pickling of custom metrics is not supported
            by Python's default pickler. This option will wrap the function using
            cloudpickle allowing you to pickle your solution, but the evolution may
            run slightly more slowly. If you are running single-threaded in an
            interactive Python session or have no need to save the model, set to
            `False` for faster runs.
    """
    if not isinstance(greater_is_better, bool):
        raise ValueError('greater_is_better must be bool, got %s'
                         % type(greater_is_better))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))
    if function.__code__.co_argcount != 3:
        raise ValueError('function requires 3 arguments (y, y_pred, w),'
                         ' got %d.' % function.__code__.co_argcount)
    if not isinstance(function(np.array([1, 1]),
                      np.array([2, 2]),
                      np.array([1, 1])), numbers.Number):
        raise ValueError('function must return a numeric.')

    if wrap:
        return _Fitness(function=wrap_non_picklable_objects(function),
                        greater_is_better=greater_is_better)
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


def _log_loss(y, y_pred, w):
    """the log loss."""
    eps = 1e-15
    inv_y_pred = np.clip(1 - y_pred, eps, 1 - eps)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    score = y * np.log(y_pred) + (1 - y) * np.log(inv_y_pred)
    return np.average(-score, weights=w)


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
log_loss = _Fitness(function=_log_loss,
                    greater_is_better=False)

_fitness_map = {'pearson': weighted_pearson,
                'spearman': weighted_spearman,
                'mean absolute error': mean_absolute_error,
                'mse': mean_square_error,
                'rmse': root_mean_square_error,
                'log loss': log_loss}