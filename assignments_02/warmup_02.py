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




