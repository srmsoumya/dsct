import pandas as pd
from functools import wraps
from datetime import datetime as dt


def log(fn):
    @wraps(fn)
    def _inner(*args, **kwargs):
        tic = dt.now()
        r = fn(*args, **kwargs)
        tac = dt.now()
        print(f'Shape: {r.shape}, Execution time: {((tac - tic).microseconds)/1000:.2f} ms, Function: {fn.__name__}')
        return r
    return _inner

@log
def make_copy(df): 
    return df.copy()

# select features
@log
def select_features(df, features=None):
    return df[features]

# fix names
@log
def fix_names(df, d=None):
    return df.rename(columns=d)

# drop columns
@log
def drop_cols(df, cols=None):
    return df.drop(cols, axis=1)

# fix dtypes
@log
def fix_types(df):
    return df.assign(**{c:pd.Categorical(df[c]) for c in df.select_dtypes('object').columns})

# remove outlier
@log
def remove_outliers(df, cols=None):
    def iqr(c):
        return df[c].quantile(q=0.75) - df[c].quantile(q=0.25)

    def compute_bounds(c):
        box = iqr(c)
        L = df[c].quantile(q=0.25) - 1.5 * box
        U = df[c].quantile(q=0.75) + 1.5 * box

        return (L,U)
    
    for c in cols:
        L,U = compute_bounds(c)
        df = df.query(f'{c} <= @U and {c} >= @L')
    return df