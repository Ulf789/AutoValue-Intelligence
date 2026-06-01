# model_engine.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_and_predict(user_model, user_year, user_mileage):
    # 1. Load the historical market dataset generated in the first step
    df = pd.read_csv('cars.csv')
    
    # 2. Split into features and target
    # X = "causes" (features): model, year, mileage
    X = df[['model', 'year', 'mileage']]
    # y = "effect" (label): the price we want to predict
    y = df['price']
    
    # 3. Handle categorical text data via One-Hot Encoding
    # Machine learning models only understand numbers, not strings like "BMW M5".
    # ColumnTransformer automatically converts text model names into a binary 0/1 matrix.
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['model'])],
        remainder='passthrough'
    )
    
    # 4. Assemble the Pipeline and choose the algorithm
    # We use RandomForestRegressor — it handles multi-feature valuation tasks like this very well.
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=30, random_state=42))
    ])
    
    # 5. Train the model (let the algorithm learn depreciation patterns from the dataset)
    pipeline.fit(X, y)
    
    # 6. Accept user input and run the valuation
    # Wrap the user-selected model, year, and mileage into the same format as the training data
    input_data = pd.DataFrame([[user_model, user_year, user_mileage]], columns=['model', 'year', 'mileage'])
    
    # Run the trained model to produce a price estimate
    predicted_price = pipeline.predict(input_data)[0]
    
    # Return price as an integer
    return int(predicted_price)

# 7. Quick local test: verify the model is working correctly
if __name__ == "__main__":
    print("Testing the machine learning valuation engine...")
    # Test: what should a 2022 BMW M5 with 40,000 km be worth?
    test_price = train_and_predict("BMW M5", 2022, 40000)
    print(f"Test result: A 2022 BMW M5 with 40,000 km — AI predicted base price: {test_price} EUR")
