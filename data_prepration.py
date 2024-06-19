import pandas as pd

# Example of loading a CSV file
data = pd.read_json('data.json')

# Preprocess the data
# For instance, tokenizing text data
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.2)
