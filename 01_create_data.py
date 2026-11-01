import pandas as pd
import numpy as np

# Same sample data every run
np.random.seed(42)
number_of_records = 2000

# STEP 1: Generate input features
temperature_c = np.random.uniform(10, 40, number_of_records)
people_count = np.random.randint(0, 31, number_of_records)
ac_hours = np.random.uniform(0, 1, number_of_records)
computer_count = np.random.randint(0, 21, number_of_records)
lighting_hours = np.random.uniform(0, 1, number_of_records)

# STEP 2: Random variation (noise)
random_variation = np.random.normal(0, 0.8, number_of_records)

# STEP 3: Simulate electricity consumption in kWh
energy_consumption_kwh = (
    1.5
    + temperature_c * 0.08
    + people_count * 0.06
    + ac_hours * 3.0
    + computer_count * 0.12
    + lighting_hours * 0.8
    + random_variation
)

# Prevent negative consumption
energy_consumption_kwh = np.maximum(energy_consumption_kwh, 0.1)

# STEP 4: Create a DataFrame
data = pd.DataFrame({
    "temperature_c": temperature_c,
    "people_count": people_count,
    "ac_hours": ac_hours,
    "computer_count": computer_count,
    "lighting_hours": lighting_hours,
    "energy_consumption_kwh": energy_consumption_kwh
})

# STEP 5: Save the dataset
data.to_csv("electricity_data.csv", index=False)

print("Electricity dataset created successfully!")
print("Total records:", len(data))
print("\nFirst five records:")
print(data.head())
print("\nDataset saved as electricity_data.csv")