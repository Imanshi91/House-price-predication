import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_csv('Housing.csv')

# Select features and target
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Print the coefficients and intercept
print("\nModel Coefficients:")
print(f"Area: {model.coef_[0]}")
print(f"Bedrooms: {model.coef_[1]}")
print(f"Bathrooms: {model.coef_[2]}")
print(f"Intercept: {model.intercept_}")

def predict_house_price(area, bedrooms, bathrooms):
    features = np.array([[area, bedrooms, bathrooms]])
    predicted_price = model.predict(features)[0]
    return predicted_price

# Example usage
area = 5000
bedrooms = 3
bathrooms = 2

predicted_price = predict_house_price(area, bedrooms, bathrooms)
print(f"\nPredicted price for a house with {area} sq ft, {bedrooms} bedrooms, and {bathrooms} bathrooms using linear regression: ${predicted_price:.2f}")
