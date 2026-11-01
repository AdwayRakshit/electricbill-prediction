import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# STEP 1: Load the dataset
data = pd.read_csv("electricity_data.csv")

# STEP 2: Select input features (X)
X = data[
    [
        "temperature_c",
        "people_count",
        "ac_hours",
        "computer_count",
        "lighting_hours"
    ]
]

# STEP 3: Select the target (y)
y = data["energy_consumption_kwh"]

# STEP 4: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# STEP 5: Create the model
model = LinearRegression()

# STEP 6: Train the model
model.fit(X_train, y_train)
print("\nModel training completed!")

# STEP 7: Predict using test data
y_pred = model.predict(X_test)

# STEP 8: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nMODEL EVALUATION RESULTS")
print("------------------------")
print("MAE:", round(mae, 3), "kWh")
print("MSE:", round(mse, 3))
print("RMSE:", round(rmse, 3), "kWh")
print("R2 Score:", round(r2, 4))

# STEP 9: Display learned coefficients
print("\nLEARNED COEFFICIENTS")
for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", round(coefficient, 4))
print("Intercept:", round(model.intercept_, 4))

# STEP 10: Save the model and feature names
model_package = {
    "model": model,
    "features": list(X.columns)
}

with open("electricity_model.pkl", "wb") as file:
    pickle.dump(model_package, file)

print("\nModel saved successfully!")
print("File created: electricity_model.pkl")