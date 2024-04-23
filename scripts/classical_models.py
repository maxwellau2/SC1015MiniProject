import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from scripts.datatransformations import datatranformation
import joblib

trans = datatranformation()

# get columns as defined previously
top_predictors = ['month', 'hour', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'DEWP', 'projection']
# apply the transformation to selected columns
columns_to_transform = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'RAIN', 'WSPM']
target = 'PM2.5'
row = top_predictors + ['PM2.5']

# import data
res = pd.read_csv("cleaned.csv")
res['datetime'] = pd.to_datetime(res[['year', 'month', 'day', 'hour']])

# apply transformations
df = trans.log_transform_columns(res, columns_to_transform)
df['projection'] = df.apply(lambda row: trans.calculate_projection_vector(row, "SE"), axis=1)

# shuffle data
data = df.dropna().sample(frac=1)

# defining columns
X = data[top_predictors]
y = data[target]

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

best_model = None
best_name = ""
curr_model = ""
best_val_mse = float('inf')

# Initialize and fit the regression models
for i in range(4):
    if i == 0:
        print(f"{'=' * 8}Fitting LinearRegression Model{'=' * 8}")
        curr_model = "LinearRegression"
        model = LinearRegression()
    if i == 1:
        print(f"{'=' * 8}Fitting RandomForestRegressor Model{'=' * 8}")
        model = RandomForestRegressor(max_depth=6, random_state=0) #best
        curr_model = "RandomForestRegressor"
    if i == 2:
        print(f"{'=' * 8}Fitting AdaBoostRegressor Model{'=' * 8}")
        model = AdaBoostRegressor(random_state=0, n_estimators=10)
        curr_model = "AdaBoostRegressor"
    if i == 3:
        print(f"{'=' * 8}Fitting MLPRegressor Model{'=' * 8}")
        model = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=(12, 30, 12, 3), learning_rate="adaptive", solver="adam")
        curr_model = "MLPRegressor"
    model.fit(X_train, y_train),
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # metrics for train, validation, and test sets
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    joblib.dump(model, f'classicalModels/{curr_model}.pkl')

    if val_mse < best_val_mse:
        best_name = curr_model
        best_model = model
        best_val_mse = val_mse
        best_train_mse = train_mse
        best_val_r2 = val_r2
        best_test_mse = test_mse
        best_test_r2 = test_r2

    # metrics
    print("Train Mean Squared Error:", train_mse)
    print("Train R-squared:", train_r2)
    print("Validation Mean Squared Error:", val_mse)
    print("Validation R-squared:", val_r2)
    print("Test Mean Squared Error:", test_mse)
    print("Test R-squared:", test_r2)
    print("\n")

# get best model
print(f"Best Model : {best_name}")
print(f"Best Model Train Mean Squared Error: {best_train_mse}")
print(f"Best Model Validation Mean Squared Error: {best_val_mse}")
print(f"Best Model Validation R-squared: {best_val_r2}")
print(f"Best Model Test Mean Squared Error: {best_test_mse}")
print(f"Best Model Test R-squared: {best_test_r2}")

# Save the best model using joblib or pickle
import joblib
joblib.dump(best_model, f'classicalModels/{best_name}.pkl')