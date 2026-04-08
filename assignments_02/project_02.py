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

# Students with G3=0 didn't take the final exam.
# Keeping these rows would distort the model because it would treat missing exams as very low grades.

df_G3_filtered['sex'] = df_G3_filtered['sex'].replace({'F': 0, 'M': 1})
cols = ['schoolsup', 'internet', 'higher', 'activities']
for col in cols:
    df_G3_filtered[col] = df_G3_filtered[col].replace({'yes': 1, 'no': 0})

print('df_G3_filtered.info')
df_G3_filtered.info()

print('df_G3_filtered.shape\n', df_G3_filtered.shape)
print('df_G3_filtered.head(5)\n', df_G3_filtered.head(5))
print('df_G3_filtered.dtypes\n', df_G3_filtered.dtypes)

df_corr_not_filtered = df['absences'].corr(df['G3'])
print('df_corr_not_filtered', df_corr_not_filtered)
df_corr_filtered = df_G3_filtered['absences'].corr(df_G3_filtered['G3'])
print('df_corr_filtered', df_corr_filtered)

fig, axs = plt.subplots(1, 2, figsize=(12, 8))
axs[0].scatter(df['absences'], df['G3'], alpha=0.5)
axs[0].set_title('Original G3 and absences are not filtered')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')

axs[1].scatter(df_G3_filtered['absences'], df_G3_filtered['G3'], alpha=0.5)
axs[1].set_title('G3 and absences are filtered')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')

plt.show()

# Students with G3=0 likely didn't take the exam.
# Many of them have varying numbers of absences, which adds noise.
# This weakens the correlation between absences and G3 in the original dataset.
# After removing them, the relationship becomes clearer.

print()

print('Task 3: Exploratory Data Analysis')
# Compute the Pearson correlation between each numeric feature and G3 on the filtered dataset, 
# and print them sorted from most negative to most positive. 
# Which feature has the strongest relationship with G3? Are any results surprising?

# Then create at least two visualizations of your own choosing and save them to outputs/. 
# Use your judgment from previous weeks of data engineering to guide your use of plots. 
# Use the correlation results to guide you -- what relationships seem worth a closer look? 
# Add a comment for each plot describing what you see.

columns = ['sex', 'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'schoolsup', 'internet', 'higher', 'activities', 'absences', 'freetime', 'goout', 'Walc', 'G1', 'G2']
corr_columns = []

for col in columns:
    df_corr_col = df_G3_filtered[col].corr(df_G3_filtered['G3'])
    corr_columns.append(f'{col}: {df_corr_col}')
    #print(f"{col}: {df_corr_col}")

corr_columns_sorted = sorted(corr_columns, key=lambda x: float(x.split(': ')[1]))
for value in corr_columns_sorted:
    print(value)

fig, axs = plt.subplots(1, 2, figsize=(12, 8))
axs[0].scatter(df_G3_filtered['failures'], df_G3_filtered['G3'], alpha=0.5)
axs[0].set_title('G3 and failures')
axs[0].set_xlabel('Failures')
axs[0].set_ylabel('Final Grade (G3)')

axs[1].scatter(df_G3_filtered['schoolsup'], df_G3_filtered['G3'], alpha=0.5)
axs[1].set_title('G3 and schoolsup')
axs[1].set_xlabel('School Support (0/1)')
axs[1].set_ylabel('Final Grade (G3)')
plt.tight_layout()

plt.savefig('outputs/g3_Failures_School_Support.png')
plt.show()

# The strongest predictors of G3 are G2 and G1 (grades from previous assessments).
# Negative correlations for failures, schoolsup, and absences indicate that 
# more failures or absences → lower final grade.
# Small positive correlations (sex, internet, higher) suggest weak influence.

# A negative linear relationship is visible: more failures correspond to lower final grades.
# Most students with zero or few failures have high grades.

# Most important predictors of G3 are previous grades (G1, G2), failures, schoolsup, and absences.
# Filtering out G3=0 is important because students who didn’t take the final exam distort 
# correlations.
# Least influential features include freetime, activities, and sex.

print()

print("Task 4: Baseline Model")
# Build the simplest possible model: use failures alone to predict G3. 
# Split into training and test sets (80/20, random_state=42), fit a LinearRegression model, 
# and print the slope, RMSE, and R² on the test set.
# Add a comment: given that grades are on a 0-20 scale, 
# what do the slopes and RMSE tell you in plain English? Is R² better or 
# worse than you expected from exploratory data analysis?

X_q4 = df_G3_filtered[['failures']]
y_q4 = df_G3_filtered['G3']

X_train_q4, X_test_q4, y_train_q4, y_test_q4 = train_test_split(
    X_q4, y_q4, test_size=0.2, random_state=42
)

model_q4 = LinearRegression()
model_q4.fit(X_train_q4, y_train_q4)
y_pred_q4 = model_q4.predict(X_test_q4)

print('Slope_q4:', model_q4.coef_[0])
print('Intercept_q4:', model_q4.intercept_, '\n')

rmse_q4 = np.sqrt(np.mean((y_pred_q4 - y_test_q4) ** 2))
print(f'rmse_q4: {rmse_q4:.4f}')
score_q4 = model_q4.score(X_test_q4, y_test_q4)
print(f'score_q4: {score_q4:.4f} \n')

# The slope is negative, meaning that each additional failure reduces 
# the final grade by about 1.4 points.

# Since grades are on a 0–20 scale, this is a noticeable but not huge effect.
# The RMSE shows the average prediction error.

# For example, if RMSE ≈ 3, it means predictions are off by about ±3 grade points, 
# which is quite large relative to the scale.

# The R² score is relatively low, meaning that failures alone explains only 
# a small portion of the variation in final grades.

# This is expected based on exploratory analysis, since stronger predictors 
# like G1 and G2 showed much higher correlations with G3.

print()

print('Task 5: Build the Full Model')
# Now build a regression model using all of the numeric and binary features from 
# the Feature Guide:

df_clean = df_G3_filtered

feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]
X = df_clean[feature_cols].values
y = df_clean['G3'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Slope_q5:', model.coef_[0])
print('Intercept_q5:', model.intercept_, '\n')

rmse_q5 = np.sqrt(np.mean((y_pred - y_test) ** 2))
print(f'rmse_q5: {rmse_q5:.4f}')
score_q5 = model.score(X_test, y_test)
print(f'score_q5: {score_q5:.4f} \n')

# Split into training and test sets (80/20, random_state=42), fit a LinearRegression model, 
# and print both train R² and test R², as well as RMSE on the test set. 
# Compare the test R² to your baseline from Task 4 -- how much does adding more features help?
# Print each feature name alongside its coefficient:

for name, coef in zip(feature_cols, model.coef_):
    print(f'{name:12s}: {coef:+.3f}')

# Look carefully at the coefficients. Sort them mentally from largest to smallest. 
# Are any signs (positive or negative) surprising given what you know about the data? 
# For any surprising result, add a comment with your best explanation. 
# Then compare train R² to test R² -- are they close, or is there a gap? 
# What does that tell you about the model?

# Finally, add a comment answering: if you were deploying this model in production, 
# which features would you keep and which would you drop? Justify your choices based on what you see 
# in the numbers.
# ------------------------------------------------------------------------------------------------
# markdown
# Adding more features improved the model performance slightly.
# R² increased from 0.0895 to 0.1539, meaning the model explains more variance in G3.
# ------------------------------------------------------------------------------------------------
# schoolsup -2.062 Students with school support tend to have lower grades
# internet +0.834 Internet access slightly improves performance
# activities -0.009 No meaningful effect

# schoolsup (−2.062)
# This is the most surprising result.
# Intuitively, school support should improve performance, but the model shows a strong negative effect.
# A likely explanation is that students who receive support are already struggling, so the variable reflects underlying difficulty rather than causing lower grades.

# internet (+0.834)
# Slightly positive effect, which makes sense (access to resources), but the effect is not very strong.

# activities (~0)
# Almost no effect, which suggests extracurricular activities neither help nor harm grades significantly.
# ------------------------------------------------------------------------------------------------
# Train R² and Test R² are relatively close (no large gap).
# This suggests the model is not heavily overfitting.
# ------------------------------------------------------------------------------------------------
# If deploying this model in production:

# Keep:
# failures → strong negative effect
# studytime → meaningful positive effect
# higher → clear positive signal
# internet → useful contextual feature

# Possibly keep:
# Medu, Fedu → small but consistent positive effects

# Drop:
# freetime → near zero effect
# activities → no meaningful contribution
# traveltime → very weak effect

# Be careful with:
# schoolsup → strong negative coefficient but misleading (proxy for struggling students)
# ------------------------------------------------------------------------------------------------

print()

print('Task 6: Evaluate and Summarize')
# A useful way to evaluate a regression model visually is a predicted vs actual plot. 
# This is a scatter plot where each point in the test set becomes a dot, with the model's prediction 
# (y_hat) on the x-axis and the true value (y) on the y-axis. If the model were perfect, 
# every point would fall exactly on the diagonal (predicted = actual). Clusters or curves away from 
# the diagonal reveal systematic errors that RMSE alone won't show you. Random scattering around 
# the diagonal is expected, and acceptable, prediction error.
# Create this plot for your test set. Add a diagonal reference line (for y=y_predicted), 
# a title "Predicted vs Actual (Full Model)", labeled axes, and save to outputs/predicted_vs_actual.png. 
# Add a comment: does the model seem to struggle more at the high end, the low end, 
# or is error roughly uniform across grade levels? What does a value above or below the diagonal mean?
# Then write a plain-language summary in your comments statements covering:

# The size of the filtered dataset and the test set
# The RMSE and R² of your best model in plain language -- on a 0-20 scale, 
# what does a typical prediction error actually mean?
# Which two features have the largest positive and largest negative coefficients, 
# and what those mean
# One result that surprised you

# ---------------------------------------------------------------

predict_q6 = model.predict(X_test)
x = predict_q6
y = y_test

line_min_q6 = min(min(predict_q6), min(y_test))
line_max_q6 = max(max(predict_q6), max(y_test))

line_x_q6 = np.linspace(line_min_q6, line_max_q6, 100)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.5)
plt.plot(line_x_q6, line_x_q6, color='black', linestyle='--')
plt.title('Predicted vs Actual (Full Model)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('outputs/predicted_vs_actual.png', dpi=300)
plt.show()

# ----------------------------------------------------
# markdown
# The error appears to be roughly uniform across grade levels, 
# although there may be slightly more variation at higher grades. 
# The points are scattered around the diagonal without a strong pattern of increasing or 
# decreasing error.

# A point above the diagonal means the model underestimated the true value (actual > predicted).
# A point below the diagonal means the model overestimated the true value (predicted > actual).

# The filtered dataset contains N rows (after removing students with G3 = 0).
# The test set contains approximately 20% of the data.

# The RMSE is approximately 2.86, meaning the model’s predictions are typically off 
# by about ±3 grade points on a 0–20 scale.
# This is a relatively large error, indicating that predictions are not very precise.
# The R² is approximately 0.15, meaning the model explains only about 15% of the variation 
# in final grades.
# This indicates that the model has limited predictive power.

# Largest positive coefficient:
# internet (+0.834)
# Students with internet access or plans for higher education tend to have higher grades.
# Largest negative coefficient:
# schoolsup (−2.062)
# Students receiving school support tend to have lower grades.

# The most surprising result is the strong negative coefficient for schoolsup.
# This is unexpected because school support should help students.
# A likely explanation is that students who receive support are already struggling, 
# so this variable reflects underlying difficulty rather than causing lower grades.

print('Neglected Feature: The Power of G1')
# Add G1 (first period grade) as a feature to the full model from Task 5 and refit. 
# We kept it out because it is so powerful. Print the new test R². 
# The jump will be large -- from roughly 0.30 to somewhere around 0.80.
# Add a comment addressing these questions: does a high R² here mean G1 is causing G3? 
# Is this a useful model for identifying students who might struggle? 
# What might educators need to do if they wanted to intervene early, before G1 is even available?

df_clean_G1 = df_G3_filtered

feature_cols_G1 = ['failures', 'Medu', 'Fedu', 'studytime', 'higher', 'schoolsup',
                'internet', 'sex', 'freetime', 'activities', 'traveltime', 'G1']
X_G1 = df_clean_G1[feature_cols_G1].values
y_G1 = df_clean_G1['G3'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_G1, y_G1, test_size=0.2, random_state=42
)

model_G1 = LinearRegression()
model_G1.fit(X_train, y_train)
y_pred_G1 = model_G1.predict(X_test)

print('Slope_G1:', model_G1.coef_[0])
print('Intercept_G1:', model_G1.intercept_, '\n')

rmse_G1 = np.sqrt(np.mean((y_pred_G1 - y_test) ** 2))
print(f'rmse_G1: {rmse_G1:.4f}')
score_G1 = model_G1.score(X_test, y_test)
print(f'score_G1: {score_G1:.4f} \n')

for name, coef in zip(feature_cols_G1, model_G1.coef_):
    print(f'{name:12s}: {coef:+.3f}')

