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