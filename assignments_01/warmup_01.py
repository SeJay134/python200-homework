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

# Matplotlib Question 2
print("Matplotlib Question 2")
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]
plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("label")
plt.ylabel("label")

plt.show()

# Matplotlib Question 3
print("Matplotlib Question 3")
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1, y1, color="blue", label="datasets_1")
plt.scatter(x2, y2, color="red", label="datasets_2")
plt.title("scatter plot")
plt.xlabel("label")
plt.ylabel("label")

plt.show()

# Matplotlib Question 4
print("Matplotlib Question 4")

plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.title("plot_1_from_q1_line")

plt.subplot(1, 2, 2)
plt.bar(subjects, scores)
plt.title("plot_2_from_q2_bar")

plt.tight_layout()
plt.show()
print()

# Descriptive Stats Question 1
print("Descriptive Stats Question 1")
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]

mean_data = np.mean(data)
print("mean_data", mean_data)
median_data = np.median(data)
print("median_data", median_data)
variance_data = np.var(data)
print("variance_data", variance_data)
std_data = np.std(data)
print("std_data", std_data)
print()

# Descriptive Stats Question 2
print("Descriptive Stats Question 2")
arr_val_500 = np.random.normal(65, 10, 500)
print("arr_val_500", arr_val_500[:5])

plt.figure(figsize=(6, 6))
plt.hist(arr_val_500, bins=20)
plt.title("Distribution of Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")

plt.show()
print()

# Descriptive Stats Question 3
print("Descriptive Stats Question 3")
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.figure()
plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.xlabel("Group a and b")
plt.ylabel("Scores")

plt.show()
print()

# Descriptive Stats Question 4
print("Descriptive Stats Question 4")
normal_data = np.random.normal(50, 5, 200)

mean_normal_data = np.mean(normal_data)
print("mean_normal_data", mean_normal_data)
median_normal_data = np.median(normal_data)
print("median_normal_data", median_normal_data)
var_normal_data = np.var(normal_data)
print("var_normal_data", var_normal_data)
std_normal_data = np.std(normal_data)
print("std_normal_data", std_normal_data)

skewed_data = np.random.exponential(10, 200)
mean_skewed_data = np.mean(skewed_data)
print("mean_skewed_data", mean_skewed_data)
median_skewed_data = np.median(skewed_data)
print("median_skewed_data", median_skewed_data)
var_skewed_data = np.var(skewed_data)
print("var_skewed_data", var_skewed_data)
std_skewed_data = np.std(skewed_data)
print("std_skewed_data", std_skewed_data)

plt.figure()
plt.boxplot([normal_data, skewed_data], labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.xlabel("x")
plt.ylabel("y")

plt.show()

print()

# Descriptive Stats Question 5
from scipy import stats

print("Descriptive Stats Question 5")
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

mean_data1 = np.mean(data1)
print("mean_data1", mean_data1)
median_data1 = np.median(data1)
print("median_data1", median_data1)
mode_data1 = stats.mode(data1)
print("mode_data1", mode_data1)

mean_data2 = np.mean(data2)
print("mean_data2", mean_data2)
median_data2 = np.median(data2)
print("median_data2", median_data2)
mode_data2 = stats.mode(data2)
print("mode_data2", mode_data2)

# markdown
# The value 150 in data2 is an outlier because it is much larger than the other values
print()

# Hypothesis review
print("Hypothesis Question 1")
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stats, p_value = stats.ttest_ind(group_a, group_b)
print("t_stats", t_stats) # p high = H0, p low = not H0
print("p_value", p_value) # If p value <= alpha we reject the null hypothesis, p-value > 0.05, we fail to reject the null hypothesis
# Since the p-value is much less than 0.05, we reject the null hypothesis.
# There is a statistically significant difference between the two groups.
print()

