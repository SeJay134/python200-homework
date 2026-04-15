import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)
from sklearn.inspection import DecisionBoundaryDisplay

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Part 2: Mini-Project -- Spam or Ham? A Classifier Shootout
print('Task 1: Load and Explore')

# How many emails are in the dataset? How balanced are the two classes? 
# What does that balance (or imbalance) mean for how you should interpret a raw accuracy score?

# What do you notice? 
# Are the differences between classes dramatic or subtle?

# What does this heavy skew toward zero tell you about the data? Why does the numeric scale 
# vary so dramatically across features (some are tiny fractions, others reach into the thousands)? 
# Why might that matter for some of the models you are about to build?

path = 'resources/spambase.data'
path_col_names = 'resources/spambase.names'

try:
    if os.path.exists(path):
        df = pd.read_csv(path, header=None)
        
        print(df.head()) # col 58
    else:
        print(f"File does not exist: {path}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

print(df.shape) # (4601, 58)

col_names = []
with open(path_col_names) as f:
    for i in f:
        i = i.strip()
        if ':' in i and not i.startswith('|'):
            name_of_col = i.split(':')[0]
            col_names.append(name_of_col)
print(len(col_names)) # 57
# print(col_names)

col_names.append('spam') # col N58
df.columns = col_names

print(len(col_names))   # 58
# print(df.columns)
print(df.head())

