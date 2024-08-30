import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load the dataset
file_path = r"C:\Users\DELL\Downloads\CARdeko\preprocessed_2_cars_dataset.xlsx"
df = pd.read_excel(file_path)

# Define features and target variable
X = df.drop('price', axis=1)
y = df['price']

# Get feature names from the DataFrame
feature_names = X.columns.tolist()
print("Feature names and their order:", feature_names)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----------------------- Gradient Boosting Grid Search ------------------------

# Define Gradient Boosting model    
gb = GradientBoostingRegressor()

# Define parameter grid for Gradient Boosting
param_grid_gb = {
    'n_estimators': [1500],
    'max_depth': [7],
    'learning_rate': [0.05],
    'subsample': [0.9]
}

# Initialize GridSearchCV for Gradient Boosting
grid_search_gb = GridSearchCV(gb, param_grid_gb, cv=5, scoring='neg_mean_squared_error')
grid_search_gb.fit(X_train, y_train)

# Best parameters for Gradient Boosting
best_params_gb = grid_search_gb.best_params_
print(f"Best Gradient Boosting Params: {best_params_gb}")

# Train Gradient Boosting with best parameters
best_gb = GradientBoostingRegressor(
    n_estimators=best_params_gb['n_estimators'],
    max_depth=best_params_gb['max_depth'],
    learning_rate=best_params_gb['learning_rate'],
    subsample=best_params_gb['subsample']
)
best_gb.fit(X_train, y_train)

# ----------------------- Model Evaluation ------------------------

# Function to evaluate models
def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)  # Calculate RMSE
    r2 = r2_score(y_test, predictions)
    print(f"{model_name} - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}")
    return mae, mse, rmse, r2

# Evaluate Gradient Boosting
evaluate_model(best_gb, X_test, y_test, "Gradient Boosting")

# Save the trained model to a pickle file
model_path = r"C:\Users\DELL\Downloads\CARdeko\pic2.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(best_gb, f)

# Load the model from the pickle file
with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)

# Make predictions using the loaded model
predictions = loaded_model.predict(X_test)
print("Predictions using the loaded model:", predictions)


# Print the feature names and their order
print("Feature names and their order:", feature_names)
