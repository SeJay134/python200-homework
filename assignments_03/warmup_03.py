import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.datasets import load_iris, load_digits # scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
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

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

print('Preprocessing Question 1')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train_shape = X_train.shape
print('X_train.shape', X_train_shape)
print('y_train.shape', y_train.shape)
X_test_shape = X_test.shape
print('X_test.shape', X_test_shape)
print('y_test.shape', y_test.shape)
print()

