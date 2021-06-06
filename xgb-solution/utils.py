import numpy as np
import datetime as dt
DATEFORMAT = '%m-%d-%Y'

def to_log(x):
    return np.log(x + 1)

def to_exp(x):
    return np.exp(x) - 1

def add_days(d, t):
    return dt.datetime.strptime(d, DATEFORMAT) + dt.timedelta(days=t)

def weighting(days, value=0.25):
    return 1. / days ** value
