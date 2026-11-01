import pandas as pd
import matplotlib.pyplot as plt

# STEP 1: Load the dataset
data = pd.read_csv("electricity_data.csv")

# STEP 2: First five records
print("\nFirst five records:")
print(data.head())

# STEP 3: Dataset dimensions
print("\nDataset shape:")
print(data.shape)

# STEP 4: Data types and non-null counts
print("\nDataset information:")
data.info()

# STEP 5: Missing values
print("\nMissing values:")
print(data.isnull().sum())

# STEP 6: Statistical summary
print("\nStatistical summary:")
print(data.describe())

# STEP 7: Correlation with energy consumption
print("\nCorrelation with energy consumption:")
print(
    data.corr()["energy_consumption_kwh"]
    .sort_values(ascending=False)
)

# STEP 8: Temperature vs consumption
plt.scatter(data["temperature_c"], data["energy_consumption_kwh"])
plt.xlabel("Outdoor temperature (Celsius)")
plt.ylabel("Electricity consumption (kWh)")
plt.title("Temperature vs Electricity Consumption")
plt.grid(True)
plt.show()

# STEP 9: Computer count vs consumption
plt.scatter(data["computer_count"], data["energy_consumption_kwh"])
plt.xlabel("Number of computers")
plt.ylabel("Electricity consumption (kWh)")
plt.title("Computers vs Electricity Consumption")
plt.grid(True)
plt.show()