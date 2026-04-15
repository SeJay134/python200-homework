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

print('Preprocessing Question 2')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('X_train_scaled.mean()', X_train_scaled.mean(axis=0))

# The means of all columns in X_train_scaled are very close to 0, 
# confirming that the StandardScaler has correctly centered the data. 
# Small deviations from zero are due to floating-point precision and are expected.

# KNN
print('KNN Question 1')

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

preds = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
print()

print('KNN Question 2')
# Add a comment: does scaling improve performance, hurt it, 
# or make no difference? Why might that be for this particular dataset?

knn_q2 = KNeighborsClassifier(n_neighbors=5)
knn_q2.fit(X_train_scaled, y_train)

preds_q2 = knn_q2.predict(X_test_scaled)

print("Accuracy_q2:", accuracy_score(y_test, preds_q2))
print(classification_report(y_test, preds_q2))

# Scaling slightly improves performance. 
# Scaling ensures that all features contribute equally to the distance calculation

print()

print('KNN Question 3')
# Add a comment: is this result more or less trustworthy than a single train/test split, and why?

cv_scores = cross_val_score(knn, X_train, y_train, cv=5)

print(cv_scores)
print(f"Mean: {cv_scores.mean():.3f}")
print(f"Std:  {cv_scores.std():.3f}")

# This result is more trustworthy than a single train/test split 
# because it evaluates the model across multiple different splits of the data.

print()

print('KNN Question 4')
# Add a comment identifying which k you would choose and why.

k_values = [1, 3, 5, 7, 9, 11, 13, 15]

for k in k_values:
    knn_q4 = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_q4, X_train, y_train, cv=5)
    print(f"k={k:2d}:  mean={scores.mean():.3f}  std={scores.std():.3f}")

# I chose k = 7 because it provides a high mean cross-validation accuracy while maintaining 
# a relatively low standard deviation. This indicates that the model performs well 
# on average and is stable across different folds. A good choice of k balances accuracy 
# and consistency, avoiding both overfitting (small k) and underfitting (large k).

print()

print('Classifier Evaluation Question 1')
# Add a comment: which pair of species does the model most often confuse (if any)?

cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()
plt.title("KNN Confusion Matrix (Iris)")
plt.savefig('outputs/knn_confusion_matrix.png')
plt.show()

# The model most often confuses versicolor and virginica. 
# These two classes have more similar feature values, especially in petal measurements, 
# making them harder to distinguish. In contrast, setosa is almost always classified 
# correctly because it is well separated from the other two classes.

print()

print('Decision Trees Question 1') 
# Add a comment comparing the Decision Tree accuracy to KNN. Then add a second comment: 
# given that Decision Trees don't rely on distance calculations, would scaled vs. 
# unscaled data affect the result?

model_dtc = DecisionTreeClassifier(max_depth=3, random_state=42)
model_dtc.fit(X_train, y_train)
preds = model_dtc.predict(X_test)

print("Accuracy_dtq1:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# The Decision Tree achieves similar (or slightly lower) accuracy compared to KNN. 
# While KNN relies on distance between points, the Decision Tree creates rule-based splits, 
# which may not capture the same fine-grained structure in the data.

# Scaling does not significantly affect Decision Trees because they do not rely 
# on distance calculations. Instead, they split data based on feature thresholds, 
# so the relative scale of features does not impact the model's decisions.