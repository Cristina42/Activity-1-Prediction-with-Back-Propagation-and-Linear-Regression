# Activity-1-Prediction-with-Back-Propagation-and-Linear-Regression

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

df_original = pd.read_csv('data.csv')

# Show shape of the variables
pd.set_option('display.max_columns', None) # so it shows all columns
print(df_original.head())

# Show mean, median, min and max values
print(df_original.describe().loc[['mean', '50%', 'min', 'max']])

# Filter only numeric columns
numeric_df = df_original.select_dtypes(include=['number'])

# Plot the boxplots
plt.subplots(figsize=(20, 8))
