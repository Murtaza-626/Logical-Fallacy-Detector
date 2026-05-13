import pandas as pd
from datasets import load_dataset

# 1. Load the dataset just like your main project does
print("Loading data...")
dataset = load_dataset("tasksource/logical-fallacy")

# 2. Convert a specific split (like 'train') to a Pandas DataFrame
# The dataset is a dictionary, so we pick the 'train' key
df = dataset['train'].to_pandas()

# 3. Save it to a CSV file on your desktop
df.to_csv("logical_fallacy_data.csv", index=False)

print("Done! You can now open 'my_logical_fallacy_data.csv' in Excel.")