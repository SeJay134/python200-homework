import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pathlib import Path

print('Part 2: Mini-Project -- Predicting Student Math Performance\n')

print('Task 1: Load and Explore')
# Load the dataset with the correct separator. Print the shape, the first five rows, 
# and the data types of all columns.

# Then plot a histogram of G3 with 21 bins (one per possible value, 0-20). 
# Add a title "Distribution of Final Math Grades", label both axes, and save to outputs/g3_distribution.png. 
# You should see a cluster of zeros sitting apart from the main distribution. 
# They represent the students who didn't take the final exam.

os.makedirs('outputs', exist_ok=True)

try:
    df = pd.read_csv('student_performance_math.csv', sep=';')
except Exception as e:
    print(f'error {e}')

print('df.info')
df.info()

print('df.shape\n', df.shape)
print('df.head(5)\n', df.head(5))

print('df.dtypes\n', df.dtypes)
print('df.isna().sum()\n', df.isna().sum())
print('df.duplicated().sum()\n', df.duplicated().sum())