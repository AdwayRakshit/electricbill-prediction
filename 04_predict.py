import pandas as pd
import pickle

# STEP 1: Load the saved model
with open("electricity_model.pkl", "rb") as file:
    model_package = pickle.load(file)

model = model_package["model"]
features = model_package["features"]

print("Saved model loaded successfully!")

# STEP 2: Enter new office conditions
temperature_c = float(input("Enter outdoor temperature in Celsius: "))
people_count = int(input("Enter number of people in the office: "))
ac_hours = float(input("Enter AC operating hours (0-1): "))
computer_count = int(input("Enter number of operating computers: "))
lighting_hours = float(input("Enter lighting operating hours (0-1): "))

# STEP 3: Prepare the input data
new_data = pd.DataFrame(
    [[
        temperature_c,
        people_count,
        ac_hours,
        computer_count,
        lighting_hours
    ]],
    columns=features
)

# STEP 4: Predict electricity consumption
prediction = model.predict(new_data)[0]

# STEP 5: Display the result
print("\nELECTRICITY CONSUMPTION PREDICTION")
print("----------------------------------")
print(
    "Estimated consumption:",
    round(max(0, prediction), 2),
    "kWh for one hour"
)