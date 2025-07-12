
from datasets import load_dataset
import pandas as pd

# Load the AG News dataset
dataset = load_dataset("ag_news")

# Convert the training set to pandas DataFrame
df = pd.DataFrame(dataset['train'])

# Show the first 5 rows
print(df.head())

# Show the label distribution
print(df['label'].value_counts())
