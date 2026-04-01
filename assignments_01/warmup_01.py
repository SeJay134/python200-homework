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
