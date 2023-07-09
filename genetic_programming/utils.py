import numbers
import numpy as np
from joblib import cpu_count
from sklearn.metrics import mean_squared_error, mean_absolute_error

def check_random_state(seed):
    '''
    Turn seed into a np.random.RandomState instance

    Parameters:
        seed -- Return randomstate instance, None | int | instance of Randomstate
    '''
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState instance' % seed)

def _get_n_jobs(n_jobs):
    """
    Number of CPUs during the computation
    
    Parameters:
        n_jobs (int) -- Number of jobs stated in joblib convention.
            
    Returns:
        n_jobs (int) -- The actual number of jobs as positive integer.
    """
    if n_jobs < 0:
        return max(cpu_count() + 1 + n_jobs, 1)
    elif n_jobs == 0:
        raise ValueError('Parameter n_jobs == 0 has no meaning.')
    else:
        return n_jobs


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(_get_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = (n_estimators // n_jobs) * np.ones(n_jobs,
                                                              dtype=np.int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


def r_2(pred, y):
    correlation_matrix = np.corrcoef(pred, y)
    correlation_xy = correlation_matrix[0, 1]
    r_2 = correlation_xy ** 2
    return r_2


def rmse(pred, y):
    return np.sqrt(mean_squared_error(y, pred))


def mae(pred, y):
    return mean_absolute_error(y, pred)
