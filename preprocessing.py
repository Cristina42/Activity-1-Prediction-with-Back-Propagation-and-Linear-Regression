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

count = 1
for i in numeric_df.columns:
    plt.subplot(2, 10, count)  # Adjust the layout (2 rows, 10 columns)
    sns.boxplot(data=numeric_df[i])
    plt.title(i, fontsize=12)
    count += 1

plt.suptitle('Boxplots of all numeric variables of the dataset', fontsize=16)
plt.subplots_adjust(top=0.85)  # Adjust the title space to avoid overlap
plt.tight_layout()  # Ensure that subplots do not overlap
plt.show()

# Count the number of rows where 'percentage expenditure' is greater than 100
count_above_1000 = df_original[df_original['percentage expenditure'] > 100].shape[0]

# Display the result
print(f"Number of rows with 'percentage expenditure' above 100: {count_above_1000}")

# Count the number of rows where 'percentage expenditure' is 0
count_of_0 = df_original[df_original['percentage expenditure'] == 0].shape[0]

# Display the result
print(f"Number of rows with 'percentage expenditure' of 0: {count_of_0}")

# Remove the 'percentage expenditure' column from the dataframe
df_new = df_original.drop(columns=['percentage expenditure'])

# Display the cleaned dataframe
print(df_new.head())

# Count the number of rows where 'measles' is greater than 1000
measles_above_1000 = df_original[df_original['Measles '] > 1000].shape[0]

# Display the result
print(f"Number of rows with measles values above 1000: {measles_above_1000}")

# Drop 'measles' from the new dataframe
df_new = df_new.drop(columns=['Measles '])


