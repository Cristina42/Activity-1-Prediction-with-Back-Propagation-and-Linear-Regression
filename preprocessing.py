# Activity-1-Prediction-with-Back-Propagation-and-Linear-Regression

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the dataset
df_original = pd.read_csv('data.csv')

# Show shape of the variables
pd.set_option('display.max_columns', None) # so it shows all columns
print(df_original.head())

# Show the columns
df_original.columns

# Show type of objects in the dataset
df.info()

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

# Verify the column has been removed
print(df_new.head())

# Drop the 'bmi' and 'thinness5-9' columns
df_new.drop([' BMI ', ' thinness  1-19 years'], axis=1, inplace=True)

# Check the updated dataframe
print(df_new.head())

# Count the number of values above 1000 in the 'under-five deaths' column
count_above_1000 = df_original[df_original['under-five deaths '] > 1000].shape[0]

# Print the result
print("Number of values above 1000 in 'under-five deaths':", count_above_1000)

# Remove rows where 'under-five deaths' is above 1000
df_new = df_new[df_new['under-five deaths '] <= 1000]

# Remove the 'infant deaths' column
df_new = df_new.drop(columns=['infant deaths'])

# Verify that the column is removed
print(df_new.columns)

# Select only numeric columns
numericcolumns_df = df_new.select_dtypes(include=['float64', 'int64'])

# Now compute correlation
correlation_matrix = numericcolumns_df.corr()

# Print the correlation with 'Life expectancy'
print(correlation_matrix['Life expectancy '])

# Remove the 'Population' column
df_new = df_new.drop(columns=['Population'])

# Verify that the column is removed
print(df_new.columns)

# Remove the 'Hepatitis B' column
df_new = df_new.drop(columns=['Hepatitis B'])

# Verify that the column is removed
print(df_new.columns)

# Show missing values
print(df_new.isna().sum())

# Remove missing values
df_cleaned = df_new.dropna()

# Check the new dataset
print(f"Original dataset shape: {df_new.shape}")
print(f"Cleaned dataset shape: {df_cleaned.shape}")

# Create a copy of df_cleaned to preserve the original dataset
df_with_encoded = df_cleaned.copy()

# Encode 'Country' with LabelEncoder
encoder = LabelEncoder()
df_with_encoded['Country_'] = encoder.fit_transform(df_with_encoded['Country'])

# Encode 'Status' using LabelEncoder or map
status_mapping = {"Developed": 0, "Developing": 1}  # Mapping manually
df_with_encoded['Status_'] = df_with_encoded['Status'].map(status_mapping)

# Drop the original 'Country' and 'Status' columns if you no longer need them
df_with_encoded.drop(['Country', 'Status'], axis=1, inplace=True)

# Check the final dataset shape and the first few rows
print(df_with_encoded.shape)
print(df_with_encoded.head())




# List of the 15 columns 
columns = df_with_encoded.columns

# Create the subplots
plt.subplots(figsize=(27, 15))

count = 1
for i in columns:
    plt.subplot(3, 5, count)  # 4 rows, 5 columns
    sns.histplot(data=df_with_encoded[i], kde=True) 
    plt.title(i, fontsize=12)
    count += 1

plt.suptitle('Histograms of all numeric variables of the dataset', fontsize=16)
plt.subplots_adjust(top=0.85)
plt.tight_layout()
plt.show()


# Save preprocessed dataset to a new CSV file
df_with_encoded.to_csv('preprocesseddataset.csv', index=False)

# Select randomly 80% of the patterns for training and validation, and the remaining 20% for test

# Step 1: Separate input features (X) and output feature (y)
X = df_normalized.drop(columns=["Life expectancy "])  # Input features
y = df_normalized["Life expectancy "]  # Output feature

# Step 2: Split the data into 80% training/validation and 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Step 3: Check the resulting shapes
print("Training+Validation set (X):", X_train_val.shape)
print("Test set (X):", X_test.shape)
print("Training+Validation set (y):", y_train_val.shape)
print("Test set (y):", y_test.shape)

# Combine input features and output labels for training/validation and test sets
train_data = pd.concat([X_train_val, y_train_val], axis=1)  # Concatenate X and y for training
test_data = pd.concat([X_test, y_test], axis=1)  # Concatenate X and y for testing

# Save the data to CSV files
train_data.to_csv('train_data.csv', index=False)  # Save the training data
test_data.to_csv('test_data.csv', index=False)  # Save the test data
