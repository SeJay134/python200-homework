from prefect import flow, task
import numpy as np
import pandas as pd
from scipy import stats

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

@task(log_prints=True)
def create_series(arr):
    s = pd.Series(arr, name="values")
    return s

@task(log_prints=True)
def clean_data(s):
    clean = s.dropna()
    return clean

