import pandas as pd
import numpy as np

# Set random seed to ensure reproducible results
np.random.seed(42)
n_samples = 500

print("Simulating German used car market data...")

# 1. Randomly generate model, first registration year, and current mileage for 500 cars
models = np.random.choice(["BMW M5", "3 Series Touring", "Audi R8"], n_samples)
years = np.random.randint(2016, 2026, n_samples)
mileages = np.random.randint(5000, 150000, n_samples)

# 2. Define the baseline new-car market price for each model
base_prices = {"BMW M5": 90000, "3 Series Touring": 35000, "Audi R8": 130000}
prices = []

# 3. Calculate a realistic used-car price for each vehicle using market depreciation rules
for m, y, mil in zip(models, years, mileages):
    p = base_prices[m]
    
    # Rule A: Newer cars cost more. Using 2016 as baseline, add €4,000 per more recent year
    p += (y - 2016) * 4000  
    
    # Rule B: Higher mileage lowers value. Subtract €150 per 1,000 km driven
    p -= (mil / 1000) * 150  
    
    # Rule C: Add random market noise (e.g. condition, optional extras fitted by previous owner)
    p += np.random.normal(0, 3000)  
    
    # Ensure price never goes negative; floor set at €10,000
    prices.append(max(int(p), 10000))

# 4. Pack the data into a standard DataFrame
df = pd.DataFrame({
    "model": models,
    "year": years,
    "mileage": mileages,
    "price": prices
})

# 5. Export and save as a local CSV file
df.to_csv("cars.csv", index=False)
print("✅ Done! Market dataset saved as `cars.csv` — 500 records of high-performance used car listings.")
