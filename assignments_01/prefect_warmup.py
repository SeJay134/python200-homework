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

@task(log_prints=True)
def summarize_data(clean):
    mean_clean = np.mean(clean)
    median_clean = np.median(clean)
    std_clean = np.std(clean)
    mode_clean = stats.mode(clean)[0]
    return mean_clean, median_clean, std_clean, mode_clean

@flow
def data_pipeline(arr):
    s = create_series(arr)
    clean = clean_data(s)
    summ = summarize_data(clean)
    return summ

if __name__ == "__main__":
    mean, median, std, mode = data_pipeline(arr)
    print("mean:", mean, "median:", median, "std:", std, "mode:", mode)

# This pipeline is very simple and operates on a small dataset, so using Prefect introduces unnecessary overhead. 
# It requires additional setup, such as defining tasks and flows, 
# which makes the code more complex without providing significant benefits for such a small use case.

# Prefect can be useful in more realistic scenarios such as web scraping, data analysis, 
# and data visualization pipelines. For example, when data needs to be collected regularly, 
# cleaned, processed, and visualized automatically, Prefect helps manage dependencies between tasks, handle errors, 
# and schedule workflows.