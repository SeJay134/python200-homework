# Pandas Question 1

import pandas as pd

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

print(df)
print()

df.info()
print()

for i in df:
    print(f"Num Rows: {len(df)}")
print()

print(df['grade'])
print()

# Pandas Question 2

for i in df['grade']:
    if  i > 80:
        print(i) # 85 90 95
print()

for i in df:
    print(i)

print()

for col in df.columns:
    for value in df[col]:
        print(value) # print all values
print()

for col in df.columns:
    for value in df[col]:
        if isinstance(value, int) and value > 80:
            print(value)
print() # 85 90 95
print()

print(df['grade'].values) # [85 72 90 68 95]
print()

print(df[df['grade'] > 80])
print()

print(df['grade'][df['grade'] > 80])
print()

# Pandas Question 3
df['grade_curved'] = [0, 0, 0, 0, 0]
df['grade_curved'] = df['grade'] + 5
print(df)
print()

# Pandas Question 4
df['name_upper'] = ['', '', '', '', '']
df['name_upper'] = df['name'].str.upper()
print(df[['name', 'name_upper']])
print()

# Pandas Question 5
avg_city = df.groupby('city')['grade'].mean()
print(avg_city)
print()

# Pandas Question 6
df.replace('Austin', 'Houston', inplace=True)
print(df[['name', 'city']])
print()

# Pandas Question 7
print(df.sort_values(by='grade', ascending=False)[:3])
print()

import numpy as np

# NumPy Review

# NumPy Question 1
print("NumPy Question 1")
arr1d = np.array([10, 20, 30, 40, 50])
print("arr1d.shape", arr1d.shape)
print("arr1d.ndim", arr1d.ndim)
print("arr1d.dtype", arr1d.dtype)
print()

# NumPy Question 2
print("NumPy Question 2")
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print("arr.size", arr.size)
print("arr.shape", arr.shape)
print()

# NumPy Question 3
print("NumPy Question 3")
two_d = arr[:2, 0:2] # 2D
print("two_d", two_d)
print("two_d.shape", two_d.shape)
print()

# NumPy Question 4 3x4 2x5
print("NumPy Question 4")
new_zero_arr = np.zeros((3, 4), dtype=int)
print("new_zero_arr 3x4", new_zero_arr)
print()
new_ones_arr = np.ones((2, 5), dtype=int)
print("new_zero_arr 2x5", new_ones_arr)
print()

# NumPy Question 5
print("NumPy Question 5")
arr_q5 = np.arange(0, 50, 5)
print("arr_q5", arr_q5)

print(arr_q5.shape)

mean_arr_q5 = np.mean(arr_q5)
print("mean_arr_q5", mean_arr_q5)

sum_arr_q5 = np.sum(arr_q5)
print("sum_arr_q5", sum_arr_q5)

std_arr_q5 = np.std(arr_q5)
print("std_arr_q5", std_arr_q5)
print()

# NumPy Question 6
print("NumPy Question 6")
arr_q6 = np.random.normal(size=200)
print("arr_q6", arr_q6)

mean_arr_q6 = np.mean(arr_q6)
print("mean_arr_q6", mean_arr_q6)
std_arr_q6 = np.std(arr_q6)
print("std_arr_q6", std_arr_q6)
print()


import matplotlib.pyplot as plt

# Matplotlib Question 1

print("Matplotlib Question 1")
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.figure(figsize=(6, 6))
plt.plot(x, y, label="line_plot_q1")
plt.title('Squares')
plt.xlabel("x")
plt.ylabel("y")

plt.show()

