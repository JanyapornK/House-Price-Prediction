import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data.csv")
house_prices = df['price']
house_sizes = df['sqft_living']
land_sizes = df['sqft_lot']

plt.scatter(house_sizes, house_prices, marker ='o', color='blue')
plt.title('House Prices vs House Size')
plt.xlabel('House Size (sqft)')
plt.ylabel('House Price ($)')
plt.show()

#train-test split
x_train, x_test, y_train, y_test = train_test_split(house_sizes, house_prices, test_size=0.2, random_state=42)

x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)

model = LinearRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

plt.scatter(x_test, y_test, marker='o', color='blue', label='Actual Prices')
plt.plot(x_test, predictions, color='red', label='Predicted Prices')
plt.title('House Price Prediction with Linear Regression')
plt.xlabel('House Size (sqft)')
plt.ylabel('House Price ($)')
plt.legend()
plt.show()

