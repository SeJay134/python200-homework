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

plt.boxplot([ham['word_freq_free'], spam['word_freq_free']], labels=['ham', 'spam'])
plt.title('word_freq_free: ham vs spam')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('outputs/word_freq_free.png')
plt.close()
plt.show()

plt.boxplot([ham['char_freq_!'], spam['char_freq_!']], labels=['ham', 'spam'])
plt.title('char_freq_!: ham vs spam')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('outputs/char_freq_!.png')
plt.close()
plt.show()

plt.boxplot([ham['capital_run_length_total'], spam['capital_run_length_total']], labels=['ham', 'spam'])
plt.ylabel('Frequency')
plt.grid(True)
plt.title('capital_run_length_total')
plt.savefig('outputs/capital_run_length_total.png')
plt.close()
plt.show()

# The dataset contains 4601 emails.

# There are 58 columns in total:
# 57 input features
# 1 target variable (spam)

# The dataset is moderately imbalanced:
# About 39% spam emails
# About 61% ham (non-spam) emails

# The dataset is imbalanced, accuracy alone can be misleading.
# A simple model that always predicts “ham” would still achieve around 60% accuracy, even though it has no real predictive power.
# Therefore, metrics like precision, recall, and F1-score are more meaningful than raw accuracy.

# word_freq_free: Spam emails generally have higher values, while ham emails are mostly near zero.
# char_freq_!: Spam emails contain more exclamation marks and have higher variability.
# capital_run_length_total: Spam emails tend to have much larger values, indicating more use of capital letters.
# Overall, spam and ham show clear but not perfectly separated distributions.

# The differences are noticeable and significant, but not perfectly separated.
# No single feature fully distinguishes spam from ham, but some features clearly show strong patterns.

# It indicates that most emails do not contain many of the tracked words or symbols.
# This makes the dataset sparse, which is typical for text-based features where most word frequencies are zero.

# Different features measure different things:
# Word frequency features are small fractions (0–1 range)
# Capital run length features can reach large values (hundreds or thousands)
# This creates a large difference in scale across features.

# Many models (such as Logistic Regression, KNN, and SVM) are sensitive to feature scaling.
# If features are not normalized, large-scale features (like capital run length) may dominate smaller ones.
# Therefore, feature scaling is important before training models.

print('Task 2: Prepare Your Data')

X = df.drop("spam", axis=1)
y = df["spam"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_shape = X_train.shape
print('X_train.shape', X_train_shape)
print('y_train.shape', y_train.shape)
X_test_shape = X_test.shape
print('X_test.shape', X_test_shape)
print('y_test.shape', y_test.shape)
print()