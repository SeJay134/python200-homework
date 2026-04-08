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

counts, bins = np.histogram(df['G3'], bins=[x-0.5 for x in range(0,22)])

plt.figure(figsize=(7, 5))
plt.hist(df['G3'], bins=[x - 0.5 for x in range(0, 22)], edgecolor='black', linewidth=0.5)

for count, bin_start in zip(counts, bins[:-1]):
    plt.text(
        x=bin_start + 0.5,
        y=count + 0.2,
        s=int(count),
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.title('Distribution of Final Math Grades')
plt.xticks(range(0, 21, 1))
plt.xlabel('Final Grade (G3)')
plt.ylabel('Number of Students')
plt.savefig('outputs/g3_distribution.png')
plt.show()

print()

print('Task 2: Preprocess the Data')
# Handle the G3=0 rows first. Filter them out and save the result to a new DataFrame. 
# Print the shape before and after to confirm how many rows were removed. 
# Add a comment explaining your reasoning -- why would keeping these rows distort the model?

# Then convert the yes/no columns to 1/0 and the sex column to 0/1.

# Now check something interesting before moving on. 
# Compute the Pearson correlation between absences and G3 on both the original dataset and 
# the filtered one, and print both values. The difference is striking. 
# Add a comment explaining why filtering changes the result: what were students with G3=0 doing in 
# the original data that made absences look like a weak predictor? 
# You might want to explore scatter plots to help understand this.

print('df.shape', df.shape)
df_G3_filtered = df[df['G3'] != 0].copy()
print('df_G3_filtered', df_G3_filtered.shape)


