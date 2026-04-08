from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Part 1: Warmup Exercises
print('Part 1: Warmup Exercises\n')

# The scikit-learn API
print('The scikit-learn API\n')

# scikit-learn Question 1
print('scikit-learn Question 1')

# The core pattern in scikit-learn is create → fit → predict. Practice it here with a simple dataset: 
# years of work experience versus annual salary.

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000]).reshape(-1, 1)

# Create a LinearRegression model, fit it to this data, and then predict the salary for someone 
# with 4 years of experience and someone with 8 years. Print the slope (model.coef_[0]), the intercept 
# (model.intercept_), and the two predictions. Label each printed value.

print(years.shape, years.ndim)

model = LinearRegression()                      # 1. Create
model.fit(years, salary)                 # 2. Learn from data
y_predictions = model.predict(years)     # 3. Predict on new inputs

print('Intercept:', model.intercept_, '\n')

print('4 and 8 years')
new_value = np.array([4, 8]).reshape(-1, 1) # [[4], [8]]
prediction = model.predict(new_value)
print('Slope:', model.coef_[0][0]) # [0] - [8313], [0][0] - 8313
print('Intercept', model.intercept_[0]) # 4 - 34534
print('prediction', prediction[0][0]) # 4 - 67790
print('prediction', prediction[1][0], '\n') # 8 - 101046

# ---------------------------------------------------------------------------------

print('scikit-learn Question 2')
# scikit-learn requires the feature array X to be 2D even when you only have one feature. 
# Start with this 1D array:

x = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)
print(x)
# Print its shape. Use .reshape() to convert it to a 2D array and print the new shape. 
# Add a comment explaining, in your own words, why scikit-learn needs X to be 2D.

# it does not work with 1D array

# -----------------------------------------------------------------------------------

print('scikit-learn Question 3')
# K-Means is an unsupervised algorithm that follows the same create → fit → predict pattern 
# as everything else in scikit-learn. Use the code below to generate a synthetic dataset 
# with three natural clusters:

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from pathlib import Path

# X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)
# Create a KMeans model with n_clusters=3 and random_state=42, fit it to X_clusters, 
# and predict a cluster label for each point. Print the cluster centers (kmeans.cluster_centers_) 
# and how many points fell into each cluster using np.bincount(labels).
# Then create a scatter plot coloring each point by its cluster label, plot the cluster centers 
# as black X's, add a title and axis labels. Save the figure to outputs/kmeans_clusters.png.

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

fig, (ax1) = plt.subplots()

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)
print(kmeans.cluster_centers_)
print(np.bincount(labels))

ax1.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap='viridis', s=60, alpha=0.7)
ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', marker='x', s=200, linewidths=3)
ax1.set_title('KMeans model')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

path = Path('outputs/kmeans_clusters.png')
path.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(path, dpi=300)
plt.tight_layout()
plt.show()
print()

# ---------------------------------------------------------------------------------------------

print('Linear Regression\n')
# The questions below all use the same synthetic medical costs dataset: 
# 100 patients, each with an age (20 to 65), a smoker flag (0 = non-smoker, 1 = smoker), 
# and an annual medical cost as the target. Generate it once and reuse the variables throughout.

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)
print()

# ---------------------------------------------------------------------------------------------

print('Linear Regression Question 1')
# Before fitting anything, look at the data. Create a scatter plot of age on the x-axis and 
# cost on the y-axis. Color the points by smoker status by passing c=smoker and cmap="coolwarm" 
# to plt.scatter(). Add a title "Medical Cost vs Age", label both axes, and save 
# to outputs/cost_vs_age.png.
# Add a comment describing what you see. Are there two distinct groups visible? 
# What does that suggest about the smoker variable?

print(age, '\n')
print(smoker, '\n')

path = Path('outputs/cost_vs_age.png')
path.parent.mkdir(parents=True, exist_ok=True)
plt.figure()
plt.scatter(age, cost, c=smoker, cmap='coolwarm')
plt.title('Medical Cost vs Age')
plt.xlabel('age')
plt.ylabel('cost')
plt.savefig(path, dpi=300)
plt.show()

# There are two distinct groups visible. 
# One group has much higher costs, which likely represents smokers. 
# This suggests that smoking has a strong impact on medical costs.
print()

# ------------------------------------------------------------------------------

print('Linear Regression Question 2')
# Split the data into training and test sets using age as the only feature, an 80/20 split, 
# and random_state=42. Reshape age to a 2D array before using it as X. 
# Print the shapes of all four arrays.

age_resh = age.reshape(-1, 1)
print(age_resh, '\n')
cost_resh = cost.reshape(-1, 1)
print('cost_resh', cost_resh, '\n')

X = age_resh
y = cost_resh

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print('X_train', X_train, '\n')
print('X_test', X_test, '\n')
print('y_test', y_test, '\n')
print('y_train', y_train, '\n')

# ----------------------------------------------------------------------------------

print('Linear Regression Question 3')
# Fit a LinearRegression model to your training data from Question 2. 
# Print the slope and intercept. Then predict on the test set and print:
# RMSE: np.sqrt(np.mean((y_pred - y_test) ** 2))
# R² on the test set: model.score(X_test, y_test)
# Add a comment interpreting the slope in plain English -- what does it mean for medical costs?

model_q3 = LinearRegression()
model_q3.fit(X_train, y_train)
y_pred = model_q3.predict(X_test)

print('Slope:', model_q3.coef_[0])
print('Intercept:', model_q3.intercept_, '\n')

rmse_q3 = np.sqrt(np.mean((y_pred - y_test) ** 2))
print(f'rmse_q3: {rmse_q3:.4f}')
score_q3 = model_q3.score(X_test, y_test)
print(f'score_q3: {score_q3:.4f} \n')

# The slope represents the increase in medical cost per additional year of age.
# A positive slope means that older patients generally have higher medical costs

# ---------------------------------------------------------------------------------

print('Linear Regression Question 4')
# Now add smoker as a second feature and fit a new model.
X_full = np.column_stack([age, smoker])

X = age
y = cost

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_m, y_train_m)
# Split, fit, and print the test R². Compare it to the R² from Question 3 -- does adding 
# the smoker flag help? Print both coefficients:

print('age coefficient: ', model_full.coef_[0])
print('smoker coefficient: ', model_full.coef_[1])

# Add a comment interpreting the smoker coefficient: what does it represent in practical terms?

# The smoker coefficient represents how much extra medical cost is expected for a smoker
# compared to a non-smoker of the same age.
print()

# -----------------------------------------------------------------------------------



