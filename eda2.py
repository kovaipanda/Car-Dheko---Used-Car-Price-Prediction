import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load the dataset
file_path = r"C:\Users\DELL\Downloads\CARdeko\preprocessed_2_cars_dataset.xlsx"

df = pd.read_excel(file_path)

# Descriptive statistics for numerical columns
numerical_columns = ['modelYear', 'KmsDriven', 'price', 'Mileage', 'ownerNo','oem','City','InsuranceValidity','FuelType','EngineDisplacement','MaxPower','Torque','bt']

# Correlation analysis
corr_matrix = df.corr()
print(corr_matrix['price'].sort_values(ascending=False))

# Feature Importance using RandomForest
X = df.drop('price', axis=1)  # Independent variables
y = df['price']  # Dependent variable (target)

# Train RandomForest model
model = RandomForestRegressor()
model.fit(X, y)

# Get feature importance
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importances = feature_importances.sort_values('Importance', ascending=False)

# Display feature importances
print(feature_importances)
