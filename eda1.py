import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset with a specific engine
file_path = r"C:\Users\DELL\Downloads\CARdeko\preprocessed_2_cars_dataset.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

# Columns relevant for correlation analysis with 'price'
relevant_columns = ['modelYear', 'KmsDriven', 'price', 'Mileage', 'ownerNo', 'EngineDisplacement', 'MaxPower', 'Torque']

# Display descriptive statistics
print(df[relevant_columns].describe())

# 1. Scatter Plots: Analyze relationship between price and relevant numerical columns
for col in relevant_columns:
    if col != 'price':
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[col], y=df['price'])
        plt.title(f'Scatter Plot: {col} vs Price')
        plt.xlabel(col)
        plt.ylabel('Price')
        plt.grid(True)
        plt.show()

# 2. Histograms: Check the distribution of relevant numerical columns
df[relevant_columns].hist(bins=30, figsize=(15, 10))
plt.suptitle('Histograms of Relevant Numerical Features')
plt.show()

# 3. Box Plots: Detect outliers in relevant numerical columns
for col in relevant_columns:
    if col != 'price':  # Exclude price for box plot
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[col])
        plt.title(f'Box Plot of {col}')
        plt.grid(True)
        plt.show()

# 4. Correlation Heatmap: Check correlations between relevant numerical features
plt.figure(figsize=(12, 8))
correlation_matrix = df[relevant_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Relevant Features')
plt.show()
