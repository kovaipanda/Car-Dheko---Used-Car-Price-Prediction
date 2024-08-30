import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
file_path = r"C:\Users\DELL\Downloads\CARdeko\preprocessed_2_cars_dataset.xlsx"
df = pd.read_excel(file_path)

# Assuming df is your cleaned dataset
X = df.drop('price', axis=1)  # Features
y = df['price']  # Target variable (price)

# Split the dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate models
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mae, mse, r2

### 1. Linear Regression (no hyperparameter tuning needed)
lr = LinearRegression()
lr.fit(X_train, y_train)
mae_lr, mse_lr, r2_lr = evaluate_model(lr, X_test, y_test)
print(f"Linear Regression - MAE: {mae_lr}, MSE: {mse_lr}, R2: {r2_lr}")

### 2. Decision Tree with Grid Search
param_grid_dt = {
    'max_depth': [None, 10, 20,30,40,50],
    'min_samples_split': [2, 5, 10,30,50]
}
dt = DecisionTreeRegressor(random_state=42)
grid_search_dt = GridSearchCV(dt, param_grid_dt, cv=5, scoring='neg_mean_squared_error')
grid_search_dt.fit(X_train, y_train)
best_dt = grid_search_dt.best_estimator_
mae_dt, mse_dt, r2_dt = evaluate_model(best_dt, X_test, y_test)
print(f"Decision Tree - MAE: {mae_dt}, MSE: {mse_dt}, R2: {r2_dt}")
print(f"Best Decision Tree Params: {grid_search_dt.best_params_}")

### 3. Random Forest with Grid Search (already in your code)
param_grid_rf = {
    'n_estimators': [400,800,1000, 1500],
    'max_depth': [None, 10, 20,40,50],
    'min_samples_split': [2, 5, 10,30,50]
}
rf = RandomForestRegressor(random_state=42)
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_
mae_rf, mse_rf, r2_rf = evaluate_model(best_rf, X_test, y_test)
print(f"Random Forest - MAE: {mae_rf}, MSE: {mse_rf}, R2: {r2_rf}")
print(f"Best Random Forest Params: {grid_search_rf.best_params_}")

### 4. Gradient Boosting with Grid Search
param_grid_gb = {
    'n_estimators': [300,400,800,1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3,7,10]
}
gb = GradientBoostingRegressor(random_state=42)
grid_search_gb = GridSearchCV(gb, param_grid_gb, cv=5, scoring='neg_mean_squared_error')
grid_search_gb.fit(X_train, y_train)
best_gb = grid_search_gb.best_estimator_
mae_gb, mse_gb, r2_gb = evaluate_model(best_gb, X_test, y_test)
print(f"Gradient Boosting - MAE: {mae_gb}, MSE: {mse_gb}, R2: {r2_gb}")
print(f"Best Gradient Boosting Params: {grid_search_gb.best_params_}")

### 5. K-Nearest Neighbors (KNN) with Grid Search
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # 1 for Manhattan, 2 for Euclidean
}
knn = KNeighborsRegressor()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='neg_mean_squared_error')
grid_search_knn.fit(X_train, y_train)
best_knn = grid_search_knn.best_estimator_
mae_knn, mse_knn, r2_knn = evaluate_model(best_knn, X_test, y_test)
print(f"KNN - MAE: {mae_knn}, MSE: {mse_knn}, R2: {r2_knn}")
print(f"Best KNN Params: {grid_search_knn.best_params_}")

# Collect results for each model
results = {
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'KNN'],
    'MAE': [mae_lr, mae_dt, mae_rf, mae_gb, mae_knn],
    'MSE': [mse_lr, mse_dt, mse_rf, mse_gb, mse_knn],
    'R2': [r2_lr, r2_dt, r2_rf, r2_gb, r2_knn]
}

# Create DataFrame for comparison
results_df = pd.DataFrame(results)
print(results_df)
