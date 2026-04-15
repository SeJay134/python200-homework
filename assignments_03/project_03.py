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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit only on training data
X_test_scaled = scaler.transform(X_test)

pca = PCA()
pca.fit(X_train_scaled)
perc_exp_vals = np.cumsum(pca.explained_variance_ratio_)

plt.plot(range(1, len(perc_exp_vals) + 1), perc_exp_vals)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")
plt.axhline(y=0.90, color='r', linestyle='--')
plt.grid(True)
plt.savefig('outputs/pca_variance_explained_task2.png')
plt.show()

n_components_90 = np.argmax(perc_exp_vals >= 0.90) + 1
print("Components needed for 90% variance:", n_components_90)

X_train_pca = pca.transform(X_train_scaled)[:, :n_components_90]
X_test_pca  = pca.transform(X_test_scaled)[:, :n_components_90]

print('Task 3: A Classifier Comparison')
# After you have results for all your classifiers, write a comment summarizing what you see. 
# Which model performs best? For the classifiers where you compared PCA vs. non-PCA, 
# which worked better -- and does that match your hypothesis from Task 2? 
# For a spam filter specifically, is accuracy the right metric to optimize -- or would you 
# rather minimize false positives (legitimate email marked as spam) or 
# false negatives (spam that gets through)? Take a position and defend it.

# Given the costs described above, 
# which type of error does your best model make more often?

knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
preds_unscaled = knn_unscaled.predict(X_test)
print("Accuracy, unscaled:", accuracy_score(y_test, preds_unscaled))
print(classification_report(y_test, preds_unscaled))
print()

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
preds_scaled = knn_scaled.predict(X_test_scaled)
print("Accuracy, scaled:", accuracy_score(y_test, preds_scaled))
print(classification_report(y_test, preds_scaled))
print()

knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
preds_pca = knn_pca.predict(X_test_pca)
print("Accuracy, pca:", accuracy_score(y_test, preds_pca))
print(classification_report(y_test, preds_pca))
print()

# KNN performs poorly on unscaled data due to differences in feature scales. 
# Scaling significantly improves performance. 
# PCA slightly reduces performance compared to scaled data but reduces dimensionality 
# while maintaining reasonable accuracy.

# 3, 5, 10, and None
model_dtc3 = DecisionTreeClassifier(max_depth=3, random_state=42)
model_dtc3.fit(X_train, y_train)
preds3 = model_dtc3.predict(X_test)
print("Accuracy_dtc3:", accuracy_score(y_test, preds3))
print('max_depth 3', classification_report(y_test, preds3))

model_dtc5 = DecisionTreeClassifier(max_depth=5, random_state=42)
model_dtc5.fit(X_train, y_train)
preds5 = model_dtc5.predict(X_test)
print("Accuracy_dtc5:", accuracy_score(y_test, preds5))
print('max_depth 5', classification_report(y_test, preds5))

model_dtc10 = DecisionTreeClassifier(max_depth=10, random_state=42)
model_dtc10.fit(X_train, y_train)
preds10 = model_dtc10.predict(X_test)
print("Accuracy_dtc10:", accuracy_score(y_test, preds10))
print('max_depth 10', classification_report(y_test, preds10))

model_dtc_none = DecisionTreeClassifier(max_depth=None, random_state=42)
model_dtc_none.fit(X_train, y_train)
preds_none = model_dtc_none.predict(X_test)
print("Accuracy_dtc_none:", accuracy_score(y_test, preds_none))
print('max_depth none', classification_report(y_test, preds_none))

# As tree depth increases, training accuracy increases but test accuracy eventually decreases, 
# indicating overfitting.

# RandomForestClassifier
clf_task3 = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=0)
clf_task3.fit(X_train_scaled, y_train)
preds_clf_task3 = clf_task3.predict(X_test_scaled)
print("Accuracy_clf_task3:", accuracy_score(y_test, preds_clf_task3))
print('RandomForest (scaled)', classification_report(y_test, preds_clf_task3))

clf_task3_pca = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=0)
clf_task3_pca.fit(X_train_pca, y_train)
preds_clf_task3_pca = clf_task3_pca.predict(X_test_pca)
print("Accuracy_clf_task3_pca:", accuracy_score(y_test, preds_clf_task3_pca))
print('RandomForest (PCA)', classification_report(y_test, preds_clf_task3_pca))

# Random Forest performed strongly on both scaled and PCA-reduced data. 
# Scaling had little effect, while PCA slightly reduced performance due to loss of feature detail. 
# However, PCA still preserved most predictive power while reducing dimensionality.

log_reg_1 = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
log_reg_1.fit(X_train_scaled, y_train)
data_log_reg_1 = np.abs(log_reg_1.coef_).sum()
preds_clf_task3_log_reg_1 = log_reg_1.predict(X_test_scaled)
print("Accuracy_log_reg_1_scaled:", accuracy_score(y_test, preds_clf_task3_log_reg_1))
print('Log regretion (scaled)', classification_report(y_test, preds_clf_task3_log_reg_1))
print()

log_reg_2 = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
log_reg_2.fit(X_train_pca, y_train)
preds_clf_task3_log_reg_2 = log_reg_2.predict(X_test_pca)
print("Accuracy_log_reg_2_pca:", accuracy_score(y_test, preds_clf_task3_log_reg_2))
print('Log regretion (PCA)', classification_report(y_test, preds_clf_task3_log_reg_2))
print()

cm = confusion_matrix(y_test, preds_clf_task3_log_reg_1)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['ham', 'spam']
)

disp.plot()
plt.title("Best model Confusion Matrix")
plt.savefig('outputs/best_model_confusion_matrix.png')
plt.show()

# Logistic Regression trained on scaled data performs best, achieving the highest 
# accuracy of 0.93. It also maintains a strong balance between precision and recall 
# for both classes.

# KNN performed poorly on unscaled data (0.80 accuracy) but improved significantly 
# after scaling (~0.91), showing that it is sensitive to feature scale.
# Decision Trees improved as depth increased, but deeper trees showed signs of overfitting.
# Random Forest outperformed a single Decision Tree (0.92 accuracy), demonstrating 
# the benefit of ensemble methods.
# Logistic Regression achieved the best overall performance, especially on scaled data.

# Models trained on the full scaled data generally performed slightly better than those using 
# PCA-reduced data.
# PCA reduced dimensionality but caused a small drop in performance due to information loss.
# However, the difference was small, indicating that PCA still retained most of 
# the important structure in the data.

# Models like KNN and Logistic Regression, which depend on feature magnitudes, benefited from scaling and showed slight performance changes with PCA.
# Tree-based models (Decision Tree and Random Forest) were less affected by scaling and dimensionality reduction.

# As tree depth increases, training accuracy improves, but test performance eventually 
# plateaus or slightly decreases.
# This indicates overfitting, where the model memorizes training data but generalizes 
# less effectively.

# Accuracy is useful, but not sufficient for spam detection.
# It is more important to minimize false positives (legitimate emails marked as spam), 
# since this can negatively impact users.
# Therefore, precision for the spam class is especially important.

# The best model (Logistic Regression) produces slightly more false negatives than false positives.
# This means some spam emails are not detected, but fewer legitimate emails are incorrectly marked as spam.

# The confusion matrix shows that the model makes slightly more false negatives than false 
# positives. This means some spam emails are not detected, but fewer legitimate emails 
# are incorrectly classified as spam. Since false positives are more harmful in a spam filter, 
# this is a reasonable trade-off.
