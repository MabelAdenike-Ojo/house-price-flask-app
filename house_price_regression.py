# Import libraries
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
r2_score
)
from sklearn.ensemble import RandomForestRegressor
# Loading the dataset
data = pd.read_csv(r'C:\Users\HP\Downloads\house-prices-advanced-regression-techniques\train.csv')
# To print first 5 rows
print(data.head()) 
# Drop columns that are not useful for prediction
columns_to_drop = ['Id', 'Alley', 'PoolQC', 'Fence', 
'MiscFeature']
data = data.drop(columns=columns_to_drop, axis=1)
# Convert categorical variables into dummy variables 
data = pd.get_dummies(data, drop_first=True)
# Fill missing numerical values with the median of the respective columns
data = data.fillna(data.median())
# Split the data into features (X) and target variable (y)
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']
# Use an 80-20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
test_size=0.2, random_state=42)
# Initialize the Linear Regression model
model = LinearRegression()
# Fit the model to the training data
model.fit(X_train, y_train)
# Predict on the test set
y_pred = model.predict(X_test)
# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# Print the evaluation results
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R²): {r2}')
# RANDOM FOREST 
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Performance")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R²:", r2_score(y_test, y_pred_rf))

# Plot actual vs predicted house prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_pred), max(y_pred)], 
color='red', linewidth=2)
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()
# Initialize the Random Forest Regressor
model = RandomForestRegressor(random_state=42)
# Fit the model to the training data
model.fit(X_train, y_train)
#Predict on the test set
y_pred_rf = model.predict(X_test)
# Calculate evaluation metrics
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
# Print the evaluation results
print(f'Mean Absolute Error (MAE): {mae_rf}')
print(f'Mean Squared Error (MSE): {mse_rf}')
print(f'R-squared (R²): {r2_rf}')
# Plot actual vs predicted house prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_pred), max(y_pred)], 
color='red', linewidth=2)
plt.title('Actual vs Predicted House Prices (Random Forest)')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

# SAVE MODEL FOR DEPLOYMENT 
joblib.dump(rf_model, "house_price_model.joblib")
joblib.dump(X.columns, "model_columns.joblib")

print("\nModel and feature columns saved successfully!")
